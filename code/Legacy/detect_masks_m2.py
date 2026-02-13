#!/usr/bin/env python3
"""
Module 2 (STRUCTURAL) — extract ALL apical trunks from M1.5 best 3D frames (no time)

Goal:
- You have M1.5 "top frames" per event_group, saved as 3D TIFFs:
    bestframe_event_group_XXXX_..._rank01_3d.tif
    bestframe_event_group_XXXX_..._rank02_3d.tif
    bestframe_event_group_XXXX_..._rank03_3d.tif

This structural M2 does NOT use activity percentiles / event logic.
Instead it does geometry-first segmentation:

Per event_group_XXXX:
  1) Load rank01/02/03 volumes (Z,Y,X), build raw3d = max(vols)
  2) Build a tubularness image using Sato (vesselness) on raw3d
  3) Threshold tubularness to get candidate "trunk" mask
  4) Optional grow/refine mask using raw intensity (keeps trunk body)
  5) 3D connected components
  6) Filter components by:
       - minimum Y-span (must be a tall vertical object)
       - aspect ratio (tall vs wide)
       - verticality by PCA (principal axis ~ Y)
  7) Keep multiple trunks per event_group (all that pass filters)
  8) Optional global dedup across events by 3D Dice
  9) Save:
       - labelmaps/dend_###_labelmap.tif (uint16; label = id+1)
       - labelmap_backgrounds/dend_###_background_2dMIP.tif (float16 0..1)
       - event_crops_bg/event_group_XXXX_bg2d_zmip.tif (float16 0..1)
       - labelmap_previews/dend_###_preview.png
       - masks_manifest.csv

Notes:
- This is designed to recover the "green traced" trunks, including quieter ones.
- You can tune just 2 knobs first: VESS_P_THR and RAW_GROW_P_THR.

Dependencies:
  pip install scikit-image opencv-python tqdm tifffile

"""

from __future__ import annotations

import gc
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure, distance_transform_edt, median_filter
from skimage.filters import sato
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm


# ================== CONFIG ==================
DATE = "2025-12-02"
MOUSE = "rbp4cre_136_phpeb"
RUN = "run4"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

# ---- Input mode ----
# Set to True to skip M1.5 and read directly from M1 event_crops
# Set to False to use M1.5 best_frames (original behavior)
USE_EVENT_CROPS_DIRECTLY = True

# Input paths (auto-configured based on mode)
if USE_EVENT_CROPS_DIRECTLY:
    INPUT_FOLDER = BASE / "preprocessed" / "event_crops"
    IN_GLOB = "event_group_*.tif"
else:
    INPUT_FOLDER = BASE / "preprocessed" / "best_frames"
    IN_GLOB = "bestframe_*_rank??_3d.tif"

# ---- Guide support ----
USE_GUIDES = False                    # Set to True to use trunk guides
GUIDE_JSON = BASE / "preprocessed" / "guides" / "trunk_guides.json"
GUIDE_DISTANCE_THRESHOLD = 20.0        # pixels: max distance from guide polyline

# Outputs
suffix = "_guided" if USE_GUIDES else ""
OUT_LABELS   = BASE / f"labelmaps-test{suffix}"
OUT_PREV     = BASE / f"labelmap_previews-test{suffix}"
OUT_BGS_MASK = BASE / f"labelmap_backgrounds-test{suffix}"
OUT_BG_EVENT = BASE / f"event_crops_bg-test{suffix}"
MANIFEST     = BASE / f"masks_manifest-test{suffix}.csv"

for p in (OUT_LABELS, OUT_PREV, OUT_BGS_MASK, OUT_BG_EVENT):
    p.mkdir(parents=True, exist_ok=True)

# ---- Physical voxel size (μm) ----
VOXEL_SIZE = (4.8, 1.0, 1.2)  # (Z,Y,X)
VOXEL_VOL  = float(np.prod(VOXEL_SIZE))

# ---- Ignore top band only (surface haze); DO NOT hard-ban bottom for structural trunks ----
Y_IGNORE_TOP_FRAC = 0.12  # smaller than your event version; adjust as needed (0.0–0.2)

# ---- Denoising (for high-noise data) ----
USE_MEDIAN_FILTER = True   # Apply median filter before Gaussian (good for speckle noise)
MEDIAN_FILTER_SIZE = 3     # Size of median filter kernel

# ---- Pre-smoothing of raw ----
RAW_SMOOTH_SIGMA = (1.0, 1.5, 1.5)  # (Z,Y,X) - increased for noisy data

# ---- Temporal aggregation (when using event_crops directly) ----
# Instead of just top 3 frames, use temporal MIP across more frames for noise reduction
USE_TEMPORAL_MIP = True    # Use max projection across time for better SNR
TEMPORAL_MIP_FRAMES = 10   # Number of frames around peak to include in temporal MIP

# ---- Vesselness / tubularness (Sato) ----
# These are in voxels; tune if your trunks are thicker/thinner
SATO_SIGMAS = (1, 2, 3, 4, 5, 6)  # Extended range for thicker trunks

# Threshold on vesselness (percentile within deep region)
VESS_P_THR = 97.5          # lowered for faint trunks (was 98.8)
VESS_P_NORM = 99.9         # normalization percentile for vesselness map

# ---- Optional raw-intensity grow to fill trunk bodies ----
DO_RAW_GROW = True
RAW_GROW_P_THR = 96.5      # lowered for faint trunks (was 97.5)
GROW_DILATION_ITERS = 2    # dilation radius for grow mask
GROW_CONNECTIVITY = 2      # 2 => 18-connect struct for dilation

# ---- 3D cleanup ----
MIN_VOL_UM3 = 6000.0       # lowered for partial trunks (was 8000)
MAX_VOL_UM3 = None         # or a number, e.g. 300000

# ---- Geometry filters for trunks (relaxed for noisy data) ----
MIN_Y_SPAN_FRAC = 0.15     # lowered (was 0.18)
MIN_ASPECT_Y_OVER_X = 1.8  # relaxed (was 2.0)
MIN_ASPECT_Y_OVER_Z = 1.8  # relaxed (was 2.0)

# PCA verticality: principal axis should align with Y.
# Keep if |vy| >= MIN_VERT_COS (0.75 ~ within ~41 degrees of vertical)
MIN_VERT_COS = 0.75        # relaxed (was 0.85)

# ---- Per-event limits ----
KEEP_MAX_TRUNKS_PER_EVENT = 30  # prevents explosion if thresholds too low

# ---- Global dedup across events ----
DO_DEDUP = True
DUPLICATE_DICE = 0.60

# ---- Limits ----
MAX_MASKS_TOTAL = 255
RANKS_REQUIRED = (1, 2, 3)

# ================== REGEX PARSING ==================
RE_EVENT_GROUP = re.compile(r"(event_group_\d{4})")
RE_RANK = re.compile(r"_rank(\d{2})_3d$", re.IGNORECASE)

def parse_event_group(stem: str) -> str | None:
    m = RE_EVENT_GROUP.search(stem)
    return m.group(1) if m else None

def parse_rank(stem: str) -> int | None:
    m = RE_RANK.search(stem)
    return int(m.group(1)) if m else None


# ================== BEST FRAME SELECTION (for direct event_crops mode) ==================
def select_best_frames_from_crop(crop: np.ndarray, top_k: int = 3, top_z_planes: int = 15, 
                                  use_temporal_mip: bool = False, temporal_mip_frames: int = 10) -> List[np.ndarray]:
    """
    Select best frames from a 4D event crop (T,Z,Y,X) using M1.5-style scoring.
    
    If use_temporal_mip=True, returns a single volume that is the temporal MIP 
    across frames around the peak (better for noisy data).
    Otherwise returns list of 3D volumes (Z,Y,X) for the top_k best frames.
    """
    if crop.ndim != 4 or crop.shape[0] < 1:
        return []
    
    T, Z, Y, X = crop.shape
    P_HI, P_MID = 99.9, 60.0
    
    # Score each frame
    scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        vol = crop[t]
        # Z-MIP (optionally restricted to top planes)
        if top_z_planes and top_z_planes > 0:
            z0 = max(Z - top_z_planes, 0)
            mip = np.nanmax(vol[z0:], axis=0)
        else:
            mip = np.nanmax(vol, axis=0)
        scores[t] = float(np.percentile(mip, P_HI) - np.percentile(mip, P_MID))
    
    # Find peak
    peak = int(np.argmax(scores))
    
    if use_temporal_mip:
        # Temporal MIP: max projection across frames around peak for noise reduction
        half_win = temporal_mip_frames // 2
        t0 = max(peak - half_win, 0)
        t1 = min(peak + half_win + 1, T)
        temporal_mip = np.max(crop[t0:t1], axis=0)  # (Z,Y,X)
        return [temporal_mip]
    else:
        # Original behavior: select top_k individual frames
        half_window = 6
        w0 = max(peak - half_window, 0)
        w1 = min(peak + half_window, T - 1)
        window = np.arange(w0, w1 + 1, dtype=int)
        
        # Select with spacing
        order = window[np.argsort(scores[window])[::-1]]
        chosen = []
        min_sep = 1
        for t in order:
            t = int(t)
            if all(abs(t - c) >= min_sep for c in chosen):
                chosen.append(t)
            if len(chosen) >= top_k:
                break
        
        return [crop[t] for t in chosen]


# ================== HELPERS ==================
def contrast_stretch_01(img: np.ndarray, p_lo=2.0, p_hi=98.0) -> np.ndarray:
    m = img.astype(np.float32, copy=False)
    finite = np.isfinite(m)
    if not finite.any():
        return np.zeros_like(m, dtype=np.float32)
    lo, hi = np.percentile(m[finite], (p_lo, p_hi))
    if hi <= lo:
        return np.zeros_like(m, dtype=np.float32)
    m = (m - lo) / (hi - lo + 1e-6)
    return np.clip(m, 0, 1).astype(np.float32)

def z_mip_2d(vol_zyx: np.ndarray) -> np.ndarray:
    return np.nanmax(vol_zyx, axis=0).astype(np.float32)

def dice_coeff(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    tot   = a.sum(dtype=np.int64) + b.sum(dtype=np.int64)
    return (2.0 * inter / tot) if tot > 0 else 0.0

def save_event_bg(event_group: str, raw3d: np.ndarray) -> Path:
    bg2d = contrast_stretch_01(z_mip_2d(raw3d), 2, 98).astype(np.float16)
    out = OUT_BG_EVENT / f"{event_group}_bg2d_zmip.tif"
    tifffile.imwrite(out, bg2d.astype(np.float16), dtype=np.float16)
    return out

def edge_overlay_preview(bg2d_01: np.ndarray, mask_zyx: np.ndarray, title: str, out_png: Path):
    mip_mask = np.max(mask_zyx.astype(bool), axis=0)
    edges = cv2.Canny((mip_mask.astype(np.uint8) * 255), 0, 1) > 0

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.imshow(bg2d_01.astype(np.float32), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~edges, edges), cmap="autumn", alpha=0.85)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def pca_verticality(mask: np.ndarray) -> float:
    """
    Returns |vy| of principal axis unit vector (0..1).
    Higher => more vertical along Y axis.
    """
    coords = np.argwhere(mask)  # (N,3) in z,y,x
    if coords.shape[0] < 20:
        return 0.0

    # center
    c = coords.mean(axis=0, keepdims=True)
    X = coords - c

    # covariance and principal eigenvector
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]  # principal axis (z,y,x)
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(abs(v[1]))  # y component magnitude

def load_guides() -> Dict[str, any]:
    """Load trunk guides from JSON if available."""
    if not USE_GUIDES or not GUIDE_JSON.exists():
        return {}
    try:
        return json.loads(GUIDE_JSON.read_text())
    except Exception as e:
        print(f"[warn] Failed to load guides: {e}")
        return {}

def create_guide_mask(polylines: List[List[List[float]]], shape_yx: Tuple[int, int]) -> np.ndarray:
    """Create 2D distance mask from polyline guides."""
    Y, X = shape_yx
    mask = np.ones((Y, X), dtype=bool)
    
    if not polylines:
        return mask
    
    # Create distance transform from all polyline points
    guide_points = np.zeros((Y, X), dtype=bool)
    for poly in polylines:
        pts = np.array(poly, dtype=int)
        # Draw polyline on mask
        for i in range(len(pts) - 1):
            y0, x0 = pts[i]
            y1, x1 = pts[i + 1]
            rr, cc = np.linspace(y0, y1, num=100), np.linspace(x0, x1, num=100)
            rr, cc = rr.astype(int), cc.astype(int)
            valid = (rr >= 0) & (rr < Y) & (cc >= 0) & (cc < X)
            guide_points[rr[valid], cc[valid]] = True
    
    if not guide_points.any():
        return mask
    
    # Distance transform
    dist = distance_transform_edt(~guide_points)
    mask = dist <= GUIDE_DISTANCE_THRESHOLD
    return mask


# ================== MAIN ==================
def main():
    paths = sorted(INPUT_FOLDER.glob(IN_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files matched {IN_GLOB} in {INPUT_FOLDER}")

    # Load guides
    guides = load_guides()
    if USE_GUIDES and guides:
        print(f"[✓] Loaded {len(guides)} trunk guides")
    elif USE_GUIDES:
        print("[warn] USE_GUIDES=True but no guides found, proceeding without guides")

    # Different grouping logic based on input mode
    if USE_EVENT_CROPS_DIRECTLY:
        # Direct mode: each file is a 4D event crop (T,Z,Y,X)
        event_files: Dict[str, Path] = {}
        for p in paths:
            eg = parse_event_group(p.stem)
            if eg:
                event_files[eg] = p
        event_ids = sorted(event_files.keys())
        print(f"[Direct mode] Found {len(event_ids)} event crops (skipping M1.5)")
    else:
        # Best-frames mode: group by event_group and rank
        groups: Dict[str, Dict[int, Path]] = {}
        for p in paths:
            eg = parse_event_group(p.stem)
            rk = parse_rank(p.stem)
            if eg is None or rk is None:
                continue
            groups.setdefault(eg, {})[rk] = p
        event_ids = sorted(groups.keys())
        print(f"[Best-frames mode] Found {len(event_ids)} event groups with rank01/02/03 bestframes.")

    extracted = []
    total = 0

    st = generate_binary_structure(3, GROW_CONNECTIVITY)

    for eg in tqdm(event_ids, desc="Structural trunk extraction"):
        if total >= MAX_MASKS_TOTAL:
            break

        # Load volumes based on mode
        if USE_EVENT_CROPS_DIRECTLY:
            # Load 4D crop and select best frames inline
            crop_path = event_files[eg]
            crop = tifffile.imread(crop_path).astype(np.float32)
            if crop.ndim != 4:
                continue
            vols = select_best_frames_from_crop(
                crop, top_k=3, 
                use_temporal_mip=USE_TEMPORAL_MIP, 
                temporal_mip_frames=TEMPORAL_MIP_FRAMES
            )
            src_files = [crop_path.name]
            del crop
            gc.collect()
        else:
            # Original best-frames mode
            rank_map = groups[eg]
            if any(r not in rank_map for r in RANKS_REQUIRED):
                continue

            vols: List[np.ndarray] = []
            src_files: List[str] = []
            for r in RANKS_REQUIRED:
                p = rank_map[r]
                v = tifffile.imread(p).astype(np.float32)
                if v.ndim != 3:
                    vols = []
                    break
                vols.append(v)
                src_files.append(p.name)

        if not vols:
            continue

        Z, Y, X = vols[0].shape
        if any(v.shape != (Z, Y, X) for v in vols):
            continue

        y_top = int(Y_IGNORE_TOP_FRAC * Y)

        # raw structural volume for this event
        raw3d = np.maximum.reduce(vols)  # (Z,Y,X)
        bg_event_path = save_event_bg(eg, raw3d)

        # also keep bg2d for previews
        bg2d = contrast_stretch_01(z_mip_2d(raw3d), 2, 98).astype(np.float16)

        # 1) denoise (median filter for speckle + Gaussian smoothing)
        if USE_MEDIAN_FILTER:
            raw_sm = median_filter(raw3d, size=MEDIAN_FILTER_SIZE)
            raw_sm = gaussian_filter(raw_sm, sigma=RAW_SMOOTH_SIGMA)
        else:
            raw_sm = gaussian_filter(raw3d, sigma=RAW_SMOOTH_SIGMA)

        # 2) vesselness/tubularness
        # sato expects float image; output is >=0
        vess = sato(raw_sm, sigmas=SATO_SIGMAS, black_ridges=False).astype(np.float32)

        # ignore top band for threshold estimation
        vess_thr_img = vess.copy()
        if y_top > 0:
            vess_thr_img[:, :y_top, :] = np.nan
        vvals = vess_thr_img[np.isfinite(vess_thr_img)]
        if vvals.size == 0:
            del vols, raw3d, raw_sm, vess
            gc.collect()
            continue

        # normalize vesselness for stability
        vnorm = np.nanpercentile(vvals, VESS_P_NORM)
        if vnorm > 0:
            vess = vess / vnorm

        # 3) vesselness threshold
        thr_v = np.nanpercentile(vvals / (vnorm + 1e-9), VESS_P_THR)
        cand = vess > thr_v

        # do not hard-ban bottom; only ignore top (optional)
        if y_top > 0:
            cand[:, :y_top, :] = False

        # ---- Apply guide constraint if available ----
        if USE_GUIDES and eg in guides:
            guide_data = guides[eg]
            polylines = guide_data.get("polyline_xy", [])
            if polylines:
                # Handle both single and multiple polylines
                if not isinstance(polylines[0][0], list):
                    polylines = [polylines]
                guide_mask_2d = create_guide_mask(polylines, (Y, X))
                # Broadcast to 3D
                guide_mask_3d = np.broadcast_to(guide_mask_2d[None, :, :], (Z, Y, X))
                cand = cand & guide_mask_3d
                print(f"  [{eg}] Applied {len(polylines)} guide(s)")

        if not cand.any():
            del vols, raw3d, raw_sm, vess, cand
            gc.collect()
            continue

        # 4) optional grow/refine using raw intensity
        if DO_RAW_GROW:
            raw_thr_img = raw_sm.copy()
            if y_top > 0:
                raw_thr_img[:, :y_top, :] = np.nan
            rvals = raw_thr_img[np.isfinite(raw_thr_img)]
            if rvals.size > 0:
                thr_r = np.nanpercentile(rvals, RAW_GROW_P_THR)
                raw_hi = raw_sm > thr_r
                if y_top > 0:
                    raw_hi[:, :y_top, :] = False

                # grow: dilate cand and intersect with raw_hi
                g = cand.copy()
                for _ in range(int(GROW_DILATION_ITERS)):
                    g = binary_dilation(g, structure=st)
                cand = np.logical_and(g, raw_hi)

        # 5) 3D cleanup by volume (voxel count)
        min_vox = max(1, int(MIN_VOL_UM3 / VOXEL_VOL))
        cand = remove_small_objects(cand.astype(bool), min_size=min_vox, connectivity=3)
        if not cand.any():
            del vols, raw3d, raw_sm, vess, cand
            gc.collect()
            continue

        # 6) CC + geometry filters
        lbl = label(cand, connectivity=3)
        props = list(regionprops(lbl))

        trunks = []
        for r in props:
            vox = int(r.area)
            vol_um3 = vox * VOXEL_VOL
            if vol_um3 < MIN_VOL_UM3:
                continue
            if (MAX_VOL_UM3 is not None) and (vol_um3 > MAX_VOL_UM3):
                continue

            z0, y0, x0, z1, y1, x1 = r.bbox
            span_y = y1 - y0
            span_x = x1 - x0
            span_z = z1 - z0

            if span_y < int(MIN_Y_SPAN_FRAC * Y):
                continue
            if span_x <= 0 or span_z <= 0:
                continue

            if (span_y / max(1, span_x)) < MIN_ASPECT_Y_OVER_X:
                continue
            if (span_y / max(1, span_z)) < MIN_ASPECT_Y_OVER_Z:
                continue

            m = (lbl == r.label)
            vert = pca_verticality(m)
            if vert < MIN_VERT_COS:
                continue

            # score: raw mean inside mask (helps stable ordering)
            raw_mean = float(raw3d[m].mean())
            trunks.append((raw_mean, vol_um3, vert, m))

        if not trunks:
            del vols, raw3d, raw_sm, vess, cand, lbl
            gc.collect()
            continue

        # keep many; order by raw_mean (descending)
        trunks.sort(key=lambda t: t[0], reverse=True)
        trunks = trunks[:KEEP_MAX_TRUNKS_PER_EVENT]

        for raw_mean, vol_um3, vert, m in trunks:
            if total >= MAX_MASKS_TOTAL:
                break
            extracted.append({
                "mask": m.astype(np.uint8),
                "event_group": eg,
                "source_bestframes": ";".join(src_files),
                "bg2d": bg2d,
                "bg_event_path": str(bg_event_path),
                "volume_um3": float(vol_um3),
                "raw_mean": float(raw_mean),
                "verticality": float(vert),
            })
            total += 1

        del vols, raw3d, raw_sm, vess, cand, lbl
        gc.collect()

    if not extracted:
        print("No trunks extracted.")
        return

    # ===== Global dedup (same FOV, different events) =====
    if DO_DEDUP:
        print("Deduplicating across events...")
        unique = []
        for item in extracted:
            m = item["mask"].astype(bool)
            is_dup = False
            for u in unique:
                if m.shape != u["mask"].shape:
                    continue
                if dice_coeff(m, u["mask"].astype(bool)) > DUPLICATE_DICE:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(item)
    else:
        unique = extracted

    print(f"[✓] Retained {len(unique)} unique trunks (from {len(extracted)} candidates)")

    # ===== Save outputs =====
    print("Saving outputs...")
    with open(MANIFEST, "w", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "trunk_id",
                "labelmap_path",
                "background_path",
                "event_bg_path",
                "event_group",
                "source_bestframes",
                "voxel_size_z",
                "voxel_size_y",
                "voxel_size_x",
                "volume_um3",
                "raw_mean_score",
                "verticality",
                "sato_sigmas",
                "vess_p_thr",
                "raw_grow_p_thr",
                "min_y_span_frac",
                "min_aspect_y_over_x",
                "min_aspect_y_over_z",
            ],
        )
        writer.writeheader()

        for i, item in enumerate(unique):
            m = item["mask"].astype(np.uint8)
            eg = item["event_group"]
            bg2d = item["bg2d"]
            event_bg_path = item["bg_event_path"]

            mask_path = OUT_LABELS / f"dend_{i:03d}_labelmap.tif"
            bg_path   = OUT_BGS_MASK / f"dend_{i:03d}_background_2dMIP.tif"

            tifffile.imwrite(mask_path, (m * (i + 1)).astype(np.uint16))
            tifffile.imwrite(bg_path, bg2d.astype(np.float16), dtype=np.float16)

            prev_png = OUT_PREV / f"dend_{i:03d}_preview.png"
            edge_overlay_preview(bg2d, m, f"dend_{i:03d} ({eg})", prev_png)

            writer.writerow({
                "trunk_id": i,
                "labelmap_path": str(mask_path),
                "background_path": str(bg_path),
                "event_bg_path": str(event_bg_path),
                "event_group": eg,
                "source_bestframes": item["source_bestframes"],
                "voxel_size_z": VOXEL_SIZE[0],
                "voxel_size_y": VOXEL_SIZE[1],
                "voxel_size_x": VOXEL_SIZE[2],
                "volume_um3": float(item["volume_um3"]),
                "raw_mean_score": float(item["raw_mean"]),
                "verticality": float(item["verticality"]),
                "sato_sigmas": str(tuple(SATO_SIGMAS)),
                "vess_p_thr": float(VESS_P_THR),
                "raw_grow_p_thr": float(RAW_GROW_P_THR) if DO_RAW_GROW else "",
                "min_y_span_frac": float(MIN_Y_SPAN_FRAC),
                "min_aspect_y_over_x": float(MIN_ASPECT_Y_OVER_X),
                "min_aspect_y_over_z": float(MIN_ASPECT_Y_OVER_Z),
            })

    print(f"Manifest saved to: {MANIFEST}")
    print("Structural M2 complete ✅")


if __name__ == "__main__":
    main()
