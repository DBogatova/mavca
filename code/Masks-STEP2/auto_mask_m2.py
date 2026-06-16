#!/usr/bin/env python
"""
Module 2 — Dendrite-enhanced detector + 2D background for Napari

Pipeline:
  1) Load each event crop (T,Z,Y,X) of ΔF/F
  2) Build a dendrite-enhanced version of the crop:
       - light spatial smoothing
       - heavy spatial blur as background
       - high-pass (small-big Gaussians), clamp to positive
  3) On the enhanced stack:
       - ignore the top Y band (surface) for thresholding
       - compute a per-frame intensity percentile
       - threshold each frame separately → binary(T,Z,Y,X)
       - zero out top Y band again
  4) Find groups of active frames; time-OR → 3D mask per group
  5) 2D per-slice clean; 3D small-object removal
  6) 3D connected components; keep by volume range
  7) Deduplicate masks by Dice
  8) Save each 3D mask + 2D background + preview + manifest row
"""

import gc
import csv
import re
from pathlib import Path

import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops
from skimage.filters import sato
from tqdm import tqdm

# ================== CONFIG ==================
DATE = "2026-05-12"
MOUSE = "rbp4_132_phpeb"
RUN = "run9"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
EVENT_FOLDER = BASE / "preprocessed" / "event_crops"
OUT_LABELS   = BASE / "labelmaps"
OUT_PREV     = BASE / "labelmap_previews"
OUT_BGS      = BASE / "labelmap_backgrounds"   # 2D backgrounds for viz
MANIFEST     = BASE / "masks_manifest.csv"
for p in (OUT_LABELS, OUT_PREV, OUT_BGS):
    p.mkdir(parents=True, exist_ok=True)

# ---- Physical voxel size ----
VOXEL_SIZE = (3.9, 1.0, 1.2)       # (Z,Y,X) μm
VOXEL_VOL  = float(np.prod(VOXEL_SIZE))

# ---- Detection parameters ----
INTENSITY_PERCENTILE = 99.5        # per-frame percentile on enhanced deep stack
Y_IGNORE_TOP_FRAC = 0.18           # ignore top 18% of Y when detecting (surface)

# ---- Temporal aggregation ----
USE_TEMPORAL_MIP = True           # Set True to use temporal MIP instead of per-frame
TEMPORAL_MIP_FRAMES = 10           # Frames around peak for temporal MIP

# ---- Multi-window hysteresis (used when USE_TEMPORAL_MIP=True) ----
TOPK_PEAKS = 4                     # Number of peak windows to detect in
PEAK_MIN_SEP = 5                   # Minimum separation between peaks (frames)
SEED_PCT = 99.7                    # High-confidence seed percentile on sum_pos
CAND_PCT = 96.5                   # Lower candidate percentile on sum_pos (tune 97–98.5)

# ---- Vesselness enhancement (helps capture full trunks) ----
USE_VESSELNESS = True             # Set True to add Sato vesselness filter
SATO_SIGMAS = (0.5, 1, 2, 3, 4, 5)      # Scales for vesselness
VESSELNESS_PERCENTILE = 97.0       # Threshold for vesselness (used in per-frame mode)
VESS_SEED_PCT = 97.0              # Vesselness seed percentile (multi-window mode)
VESS_CAND_PCT = 95.0              # Vesselness candidate percentile (multi-window mode)

# ---- Intensity grow (fills trunk bodies) ----
DO_INTENSITY_GROW = True          # Set True to grow masks using raw intensity
GROW_PERCENTILE = 96.0             # Intensity threshold for growing
GROW_DILATION_ITERS = 3            # Dilation iterations

# ---- Best-frames mode (M1.5 output) ----
# "off"      = ignore M1.5, use event crops only (original auto_mask behavior)
# "guide"    = union bestframe seeds with auto_mask candidates
# "primary"  = detect directly from M1.5 best frames (skip auto_mask enhancement)
BESTFRAME_MODE = "guide"         # "off", "guide", or "primary"
BESTFRAMES_FOLDER = BASE / "preprocessed" / "best_frames"
BESTFRAME_GLOB = "bestframe_*_rank??_3d.tif"
BESTFRAME_INTENSITY_PCT = 94.0     # Percentile threshold on best frames

MAX_FRAME_GAP = 1
MIN_EVENT_LENGTH = 1               # allow even very brief events

# ---- MinIP detection (dark shadow masks from moving trunks) ----
USE_MINIP = True                   # detect dark outlines via temporal min projection
MINIP_PERCENTILE = 2.0             # threshold: voxels below this percentile of minIP
MINIP_MIN_VOL = 5000.0             # minimum volume (µm³) for minIP masks

# ---- Temporal baseline subtraction (removes static background) ----
SUBTRACT_TEMPORAL_BG = True        # subtract quietest frame from event crop before detection
TEMPORAL_BG_SCALE = 0.8            # scale factor (< 1.0 catches dimmer dendrites)

# ---- Volume gates (μm³) ----
MIN_VOL = 6000.0                    # keep small dendrites
MAX_VOL = None                  # None → no upper cap

# ---- Geometry filters (relaxed - to remove specks) ----
USE_GEOMETRY_FILTER = True
MIN_Y_SPAN_FRAC = 0.06             # Very relaxed - just remove tiny specks
MIN_ASPECT_Y_OVER_X = 0.8          # Very relaxed
MIN_ASPECT_Y_OVER_Z = 0.8          # Very relaxed

# ---- 2D/3D clean-up ----
SLICE_OPEN_K  = 1                  # per-slice open
SLICE_CLOSE_K = 5                  # per-slice close
SLICE_MIN_PIX = 7                 # min 2D pixels per slice before 3D CC

# ---- Deduplication ----
DUPLICATE_DICE = 0.70
MAX_DENDRITES_TOTAL = 255

# ================== HELPERS ==================

def group_frames(frames, gap):
    if len(frames) == 0:
        return []
    frames = np.array(frames, dtype=int); frames.sort()
    groups, cur = [], [int(frames[0])]
    for f in frames[1:]:
        f = int(f)
        if f - cur[-1] <= gap:
            cur.append(f)
        else:
            groups.append(cur); cur = [f]
    groups.append(cur)
    return groups

def dice_coeff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    tot   = a.sum(dtype=np.int64) + b.sum(dtype=np.int64)
    return (2.0 * inter / tot) if tot > 0 else 0.0

def z_mip_background_2d(stack_TZYX, t0, t1) -> np.ndarray:
    """
    2D background for Napari:
      - take time-max over [t0, t1)
      - then Z-MIP
      - then percentile-based contrast stretch
    """
    seg = stack_TZYX[t0:t1]  # (T,Z,Y,X)
    if seg.size == 0:
        return np.zeros(stack_TZYX.shape[2:], dtype=np.float16)

    # time-max: (Z,Y,X)
    vol_tmax = np.nanmax(seg, axis=0)

    # Z-MIP (max over Z): (Y,X)
    mip2d = np.nanmax(vol_tmax, axis=0).astype(np.float32)

    # contrast stretch between 2nd and 98th percentiles
    finite = np.isfinite(mip2d)
    if finite.any():
        p2, p98 = np.percentile(mip2d[finite], (2, 98))
        if p98 > p2:
            mip2d = (mip2d - p2) / (p98 - p2)
            mip2d = np.clip(mip2d, 0, 1)

    return mip2d.astype(np.float16)


def enhance_for_detection(stack_TZYX: np.ndarray) -> np.ndarray:
    """
    Enhance dendrite-like structures & suppress broad background.

    stack_TZYX: (T,Z,Y,X) ΔF/F event crop (float32)
    Returns a stack of same shape, used only for thresholding.
    """
    # 1) Light spatial smoothing to kill shot noise, keep structure
    sm = gaussian_filter(
        stack_TZYX,
        sigma=(0.0, 0.5, 1.0, 1.0)   # (T,Z,Y,X)
    )

    # 2) Very blurred background (captures sheet + gradients)
    bg = gaussian_filter(
        sm,
        sigma=(0.0, 4.0, 8.0, 8.0)
    )

    # 3) High-pass: local positive contrast above background
    enh = sm - bg
    enh[enh < 0] = 0.0

    # 4) Normalize per event for stable percentile behavior
    vmax = np.nanpercentile(enh, 99.9)
    if vmax > 0:
        enh = enh / vmax

    return enh.astype(np.float32)

# ================== MULTI-WINDOW HELPERS ==================

def find_topk_peaks(frame_max, k, min_sep):
    """Non-max suppression on 1-D signal. Returns up to k peak indices sorted descending by value."""
    signal = frame_max.copy()
    peaks = []
    for _ in range(k):
        idx = int(np.argmax(signal))
        if signal[idx] <= 0:
            break
        peaks.append(idx)
        lo = max(idx - min_sep, 0)
        hi = min(idx + min_sep + 1, len(signal))
        signal[lo:hi] = -np.inf
    return sorted(peaks)


def keep_components_touching_seed(cand, seed, connectivity=2):
    """Label cand in 3D; keep only components that overlap at least one seed voxel."""
    lbl = label(cand)
    touching = np.unique(lbl[seed & cand])
    touching = touching[touching > 0]
    if len(touching) == 0:
        return np.zeros_like(cand, dtype=bool)
    out = np.isin(lbl, touching)
    return out


# ================== BESTFRAME GUIDE ==================
RE_EVENT_GROUP = re.compile(r"(event_group_\d{4})")

def load_bestframe_guide():
    """Load M1.5 best frames grouped by event_group. Returns dict: eg -> list of 3D vols."""
    if BESTFRAME_MODE == "off" or not BESTFRAMES_FOLDER.exists():
        return {}
    
    paths = sorted(BESTFRAMES_FOLDER.glob(BESTFRAME_GLOB))
    if not paths:
        return {}
    
    groups = {}
    for p in paths:
        m = RE_EVENT_GROUP.search(p.stem)
        if not m:
            continue
        eg = m.group(1)
        vol = tifffile.imread(p).astype(np.float32)
        if vol.ndim == 3:
            groups.setdefault(eg, []).append(vol)
    
    print(f"[bestframe {BESTFRAME_MODE}] Loaded frames for {len(groups)} event groups")
    return groups


def bestframe_seed_mask(vols, y_cut, intensity_pct):
    """
    Build a seed mask from M1.5 best frames.
    Takes max across best frames, thresholds by intensity percentile.
    Returns 3D boolean mask (Z,Y,X).
    """
    if not vols:
        return None
    
    # Max across best frames
    combined = np.maximum.reduce(vols)  # (Z,Y,X)
    
    # Threshold
    thr_vol = combined.copy()
    if y_cut > 0:
        thr_vol[:, :y_cut, :] = np.nan
    thr = np.nanpercentile(thr_vol, intensity_pct)
    seed = combined > thr
    
    if y_cut > 0:
        seed[:, :y_cut, :] = False
    
    return seed


# ================== MAIN ==================
def main():
    event_paths = sorted(EVENT_FOLDER.glob("event_group_*.tif"))
    print(f"Found {len(event_paths)} grouped event files.")

    raw_masks = []  # tuples: (mask3d, vol_um3, event_path, t_start, t_end, bg2d)
    extracted = 0

    # Load bestframe guides if enabled
    bestframe_vols = load_bestframe_guide() if BESTFRAME_MODE != "off" else {}

    se_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLICE_OPEN_K, SLICE_OPEN_K))
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLICE_CLOSE_K, SLICE_CLOSE_K))

    for path in tqdm(event_paths, desc="Extracting dendrites"):
        if extracted >= MAX_DENDRITES_TOTAL:
            break

        # Extract event group ID for bestframe matching
        eg_match = RE_EVENT_GROUP.search(path.stem)
        eg_id = eg_match.group(1) if eg_match else path.stem

        stack = tifffile.imread(path).astype(np.float32)  # (T,Z,Y,X)
        T, Z, Y, X = stack.shape
        if T == 0:
            del stack
            continue

        y_cut = int(Y_IGNORE_TOP_FRAC * Y)
        st_grow = generate_binary_structure(3, 2) if DO_INTENSITY_GROW else None

        # ===== 0) Temporal baseline subtraction =====
        if SUBTRACT_TEMPORAL_BG and T > 2:
            # Find quietest frame (lowest mean intensity)
            frame_means = stack.mean(axis=(1, 2, 3))
            quiet_idx = int(np.argmin(frame_means))
            quiet_frame = stack[quiet_idx].copy()
            stack = stack - TEMPORAL_BG_SCALE * quiet_frame[np.newaxis]
            stack[stack < 0] = 0

        # ===== 1) Build dendrite-enhanced stack for detection =====
        det_stack = enhance_for_detection(stack)          # (T,Z,Y,X)

        # ===== 1) Temporal MIP mode OR per-frame mode =====
        if USE_TEMPORAL_MIP:
            # --- Multi-window detection ---
            frame_max = det_stack.max(axis=(1, 2, 3))
            peaks = find_topk_peaks(frame_max, TOPK_PEAKS, PEAK_MIN_SEP)
            half = TEMPORAL_MIP_FRAMES // 2
            print(f"  [{eg_id}] peaks={peaks}")

            union_cand = np.zeros((Z, Y, X), dtype=bool)

            for pi, peak in enumerate(peaks):
                t0 = max(peak - half, 0)
                t1 = min(peak + half + 1, T)

                # det_vol = temporal max (bright stuff)
                det_vol = np.max(det_stack[t0:t1], axis=0)  # (Z,Y,X)

                # sum_pos = temporal evidence (dim-but-consistent)
                pos = np.clip(det_stack[t0:t1], 0, None)
                sum_pos = pos.sum(axis=0).astype(np.float32)  # (Z,Y,X)
                sp_norm = np.nanpercentile(sum_pos, 99.9) + 1e-8
                sum_pos /= sp_norm

                # Deep region (exclude top band) for percentile computation only
                sum_pos_deep = sum_pos.copy()
                if y_cut > 0:
                    sum_pos_deep[:, :y_cut, :] = np.nan

                # --- Hysteresis: seeds + candidates from sum_pos ---
                seed_thr = np.nanpercentile(sum_pos_deep, SEED_PCT)
                cand_thr = np.nanpercentile(sum_pos_deep, CAND_PCT)
                seed = sum_pos > seed_thr
                cand = sum_pos > cand_thr

                # --- Vesselness on sum_pos ---
                if USE_VESSELNESS:
                    vess = sato(sum_pos, sigmas=SATO_SIGMAS, black_ridges=False)
                    vess_deep = vess.copy()
                    if y_cut > 0:
                        vess_deep[:, :y_cut, :] = np.nan
                    vess_seed_thr = np.nanpercentile(vess_deep, VESS_SEED_PCT)
                    vess_cand_thr = np.nanpercentile(vess_deep, VESS_CAND_PCT)
                    seed = seed | (vess > vess_seed_thr)
                    cand = cand | (vess > vess_cand_thr)

                # --- Keep only candidates connected to seeds ---
                cand = keep_components_touching_seed(cand, seed, connectivity=2)

                n_seed = seed.sum()
                n_cand_raw = (sum_pos > cand_thr).sum()
                n_final = cand.sum()
                print(f"    window {pi} [t={t0}:{t1}] seed={n_seed}  cand_raw={n_cand_raw}  final={n_final}")

                # Optional bestframe guide: union with seeds from M1.5 best frames
                if BESTFRAME_MODE == "guide" and eg_id in bestframe_vols:
                    bf_seed = bestframe_seed_mask(
                        bestframe_vols[eg_id], y_cut, BESTFRAME_INTENSITY_PCT
                    )
                    if bf_seed is not None and bf_seed.shape == cand.shape:
                        n_before = cand.sum()
                        cand = cand | bf_seed
                        n_added = cand.sum() - n_before
                        if n_added > 0:
                            print(f"    [{eg_id}] Bestframe guide added {n_added} voxels")

                # Optional intensity grow
                if DO_INTENSITY_GROW and cand.any():
                    raw_vol = np.max(stack[t0:t1], axis=0)
                    raw_for_thr = raw_vol.copy()
                    if y_cut > 0:
                        raw_for_thr[:, :y_cut, :] = np.nan
                    grow_thr = np.nanpercentile(raw_for_thr, GROW_PERCENTILE)
                    raw_hi = raw_vol > grow_thr
                    g = cand.copy()
                    for _ in range(GROW_DILATION_ITERS):
                        g = binary_dilation(g, structure=st_grow)
                    cand = g & raw_hi

                union_cand |= cand

            # --- Process the union mask through the rest of the pipeline ---
            if union_cand.any():
                mask_3d = union_cand
                t_start = max(min(peaks) - half, 0)
                t_end = min(max(peaks) + half + 1, T)

                cleaned = np.zeros_like(mask_3d, dtype=np.uint8)
                for z in range(Z):
                    sl = (mask_3d[z].astype(np.uint8) * 255)
                    sl = cv2.morphologyEx(sl, cv2.MORPH_OPEN,  se_open)
                    sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, se_close)
                    slb = sl > 0
                    if SLICE_MIN_PIX > 0:
                        lbl2 = label(slb)
                        keep2 = np.zeros_like(slb, bool)
                        for i2 in range(1, int(lbl2.max()) + 1):
                            rr = (lbl2 == i2)
                            if rr.sum() >= SLICE_MIN_PIX:
                                keep2 |= rr
                        slb = keep2
                    cleaned[z] = slb

                cleaned = remove_small_objects(cleaned.astype(bool), int(MIN_VOL / VOXEL_VOL), connectivity=1)
                if cleaned.any():
                    lbl3 = label(cleaned)
                    props = sorted(regionprops(lbl3), key=lambda r: r.area, reverse=True)
                    for r in props:
                        if extracted >= MAX_DENDRITES_TOTAL:
                            break
                        voxels = int(r.area)
                        vol_um3 = voxels * VOXEL_VOL
                        if vol_um3 < MIN_VOL:
                            continue
                        if (MAX_VOL is not None) and (vol_um3 > MAX_VOL):
                            continue
                        if USE_GEOMETRY_FILTER:
                            z0, y0, x0, z1, y1, x1 = r.bbox
                            span_y = y1 - y0
                            span_x = max(1, x1 - x0)
                            span_z = max(1, z1 - z0)
                            if span_y < int(MIN_Y_SPAN_FRAC * Y):
                                continue
                            if (span_y / span_x) < MIN_ASPECT_Y_OVER_X:
                                continue
                            if (span_y / span_z) < MIN_ASPECT_Y_OVER_Z:
                                continue
                        m = (lbl3 == r.label).astype(np.uint8)
                        bg2d = z_mip_background_2d(stack, t_start, t_end)
                        raw_masks.append((m, float(vol_um3), str(path), t_start, t_end, bg2d))
                        extracted += 1
            
            del stack, det_stack
            gc.collect()
            continue  # Skip per-frame processing

        # ===== Per-frame mode (original) =====
        det_for_thr = det_stack.copy()
        if y_cut > 0:
            det_for_thr[:, :, :y_cut, :] = np.nan

        per_frame_thr = np.nanpercentile(det_for_thr, INTENSITY_PERCENTILE, axis=(1, 2, 3))
        binary = det_stack > per_frame_thr[:, None, None, None]

        if y_cut > 0:
            binary[:, :, :y_cut, :] = False

        # ===== 2) Find active frames; group small gaps =====
        activity = binary.max(axis=(1, 2, 3))
        active_ts = np.where(activity > 0)[0]
        frame_groups = group_frames(active_ts, MAX_FRAME_GAP)

        # ===== 3) For each group → 3D mask from time-OR; clean; CC; volume =====
        for group in frame_groups:
            if extracted >= MAX_DENDRITES_TOTAL:
                break
            if len(group) < MIN_EVENT_LENGTH:
                continue

            t_start = int(group[0])
            t_end   = int(group[-1]) + 1

            # Time-OR to 3D (Z,Y,X)
            mask_3d = np.any(binary[t_start:t_end], axis=0)
            if not mask_3d.any():
                continue

            # --- Per-slice clean (open/close + 2D size filter) ---
            cleaned = np.zeros_like(mask_3d, dtype=np.uint8)
            for z in range(Z):
                sl = (mask_3d[z].astype(np.uint8) * 255)
                sl = cv2.morphologyEx(sl, cv2.MORPH_OPEN,  se_open)
                sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, se_close)
                slb = sl > 0
                if SLICE_MIN_PIX > 0:
                    lbl2 = label(slb)
                    keep2 = np.zeros_like(slb, bool)
                    for i2 in range(1, int(lbl2.max()) + 1):
                        rr = (lbl2 == i2)
                        if rr.sum() >= SLICE_MIN_PIX:
                            keep2 |= rr
                    slb = keep2
                cleaned[z] = slb

            # --- 3D small-object removal by physical volume ---
            cleaned = remove_small_objects(
                cleaned.astype(bool),
                int(MIN_VOL / VOXEL_VOL),
                connectivity=1
            )
            if not cleaned.any():
                continue

            lbl3 = label(cleaned)
            props = sorted(regionprops(lbl3), key=lambda r: r.area, reverse=True)

            # ===== 4) Save components that pass volume and geometry gates =====
            for r in props:
                voxels = int(r.area)
                vol_um3 = voxels * VOXEL_VOL
                if vol_um3 < MIN_VOL:
                    continue
                if (MAX_VOL is not None) and (vol_um3 > MAX_VOL):
                    continue

                # Geometry filter to remove specks
                if USE_GEOMETRY_FILTER:
                    z0, y0, x0, z1, y1, x1 = r.bbox
                    span_y = y1 - y0
                    span_x = max(1, x1 - x0)
                    span_z = max(1, z1 - z0)
                    
                    if span_y < int(MIN_Y_SPAN_FRAC * Y):
                        continue
                    if (span_y / span_x) < MIN_ASPECT_Y_OVER_X:
                        continue
                    if (span_y / span_z) < MIN_ASPECT_Y_OVER_Z:
                        continue

                m = (lbl3 == r.label).astype(np.uint8)

                # Per-mask 2D background from same time window on ORIGINAL stack
                bg2d = z_mip_background_2d(stack, t_start, t_end)

                raw_masks.append((m, float(vol_um3), str(path), t_start, t_end, bg2d))
                extracted += 1
                if extracted >= MAX_DENDRITES_TOTAL:
                    break

        del stack, det_stack, binary
        gc.collect()

    # ====== MINIP DETECTION (dark shadow masks) ======
    if USE_MINIP:
        print("\n--- MinIP detection (dark outlines) ---")
        minip_count = 0
        for path in tqdm(event_paths, desc="  minIP scan"):
            stack = tifffile.imread(str(path)).astype(np.float32)
            if stack.ndim != 4 or stack.shape[0] < 3:
                del stack; continue
            T, Z, Y, X = stack.shape

            # Temporal minimum projection → (Z, Y, X)
            min_proj = stack.min(axis=0)

            # Threshold: voxels below the Nth percentile are "dark shadows"
            thr = np.percentile(min_proj, MINIP_PERCENTILE)
            dark_mask = (min_proj < thr).astype(np.uint8)

            # Ignore top Y band (surface)
            y_cut = int(Y * Y_IGNORE_TOP_FRAC)
            if y_cut > 0:
                dark_mask[:, :y_cut, :] = 0

            # 2D per-slice cleanup
            se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLICE_OPEN_K, SLICE_OPEN_K))
            for z in range(Z):
                sl = dark_mask[z]
                sl = cv2.morphologyEx(sl, cv2.MORPH_OPEN, se_open)
                dark_mask[z] = sl

            # Remove small 3D objects
            dark_bool = dark_mask.astype(bool)
            min_vox = int(MINIP_MIN_VOL / VOXEL_VOL)
            dark_bool = remove_small_objects(dark_bool, min_size=min_vox)

            if not dark_bool.any():
                del stack, min_proj, dark_mask, dark_bool; continue

            # 3D connected components
            lbl3 = label(dark_bool)
            for r in regionprops(lbl3):
                vol_um3 = r.area * VOXEL_VOL
                if vol_um3 < MINIP_MIN_VOL:
                    continue
                if MAX_VOL is not None and vol_um3 > MAX_VOL:
                    continue
                m = (lbl3 == r.label).astype(np.uint8)
                # Background from temporal mean (better context for dark masks)
                bg2d = stack.mean(axis=0).max(axis=0).astype(np.float16)
                raw_masks.append((m, float(vol_um3), str(path), 0, T, bg2d))
                minip_count += 1

            del stack, min_proj, dark_mask, dark_bool, lbl3
            gc.collect()

        print(f"  MinIP: {minip_count} additional masks from dark outlines")

    # ====== DEDUP BY 3D DICE ======
    print("Deduplicating masks...")
    unique = []
    for m, vol, ep, ts, te, bg in raw_masks:
        is_dup = False
        for u in unique:
            if m.shape != u["mask"].shape:
                continue
            if dice_coeff(m.astype(bool), u["mask"].astype(bool)) > DUPLICATE_DICE:
                is_dup = True
                break
        if not is_dup:
            unique.append({
                "mask": m,
                "vol": vol,
                "event_path": ep,
                "t_start": ts,
                "t_end": te,
                "bg2d": bg
            })

    print(f"[✓] Retained {len(unique)} unique dendrites out of {len(raw_masks)}")

    # ====== SAVE MASKS, BACKGROUNDS, PREVIEWS, MANIFEST ======
    print("Saving outputs...")
    with open(MANIFEST, "w", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "dend_id",
                "labelmap_path",
                "background_path",
                "source_event_file",
                "event_t_start",
                "event_t_end",
                "voxel_size_z",
                "voxel_size_y",
                "voxel_size_x",
                "volume_um3",
            ],
        )
        writer.writeheader()

        for i, e in enumerate(unique):
            mask = e["mask"].astype(np.uint8)           # (Z,Y,X)
            bg2d = e["bg2d"]                            # (Y,X) float16
            src  = Path(e["event_path"]).name

            mask_path = OUT_LABELS / f"dend_{i:03d}_labelmap.tif"
            bg_path   = OUT_BGS   / f"dend_{i:03d}_background_2dMIP.tif"

            # Mask as single-object labelmap with value (i+1)
            tifffile.imwrite(mask_path, (mask * (i + 1)).astype(np.uint16))
            # 2D background image
            tifffile.imwrite(bg_path, bg2d, dtype=np.float16)

            # Quick preview: overlay Z-MIP of mask on background
            mip_mask = np.max(mask, axis=0).astype(bool)
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(bg2d.astype(np.float32), cmap="gray")
            edges = cv2.Canny((mip_mask.astype(np.uint8) * 255), 0, 1) > 0
            ax.imshow(np.ma.masked_where(~edges, edges), cmap="autumn", alpha=0.8)
            ax.set_title(f"dend_{i:03d} | {src}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(OUT_PREV / f"dend_{i:03d}_preview.png", dpi=150)
            plt.close(fig)

            writer.writerow({
                "dend_id": i,
                "labelmap_path": str(mask_path),
                "background_path": str(bg_path),
                "source_event_file": src,
                "event_t_start": int(e["t_start"]),
                "event_t_end": int(e["t_end"]),
                "voxel_size_z": VOXEL_SIZE[0],
                "voxel_size_y": VOXEL_SIZE[1],
                "voxel_size_x": VOXEL_SIZE[2],
                "volume_um3": float(e["vol"]),
            })

    print(f"Manifest saved to: {MANIFEST}")
    print("Module 2 (dendrite-enhanced + background) complete.")

if __name__ == "__main__":
    main()
