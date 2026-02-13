#!/usr/bin/env python3
"""
Module 2 HYBRID — Combines activity-based enhancement with structure-based detection

Best of both worlds:
- auto_mask's high-pass enhancement to boost local contrast
- detect_mask's vesselness filter for tubular structure detection
- Relaxed geometry filters to catch more trunks
- Connectivity-based merging for fragmented pieces

Pipeline:
  1) Load event crops (T,Z,Y,X)
  2) Build temporal MIP or select best frames
  3) Apply high-pass enhancement (small - big Gaussian)
  4) Apply Sato vesselness on enhanced image
  5) Threshold with relaxed parameters
  6) Optional intensity-based grow
  7) 3D connected components with relaxed geometry filters
  8) Merge nearby fragments
  9) Deduplicate and save

"""

from __future__ import annotations

import gc
import csv
import re
from pathlib import Path
from typing import Dict, List

import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure, label as ndimage_label
from skimage.filters import sato
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm


# ================== CONFIG ==================
DATE = "2025-12-02"
MOUSE = "rbp4cre_136_phpeb"
RUN = "run4"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

# Input
EVENT_FOLDER = BASE / "preprocessed" / "event_crops"
IN_GLOB = "event_group_*.tif"

# Outputs
OUT_LABELS   = BASE / "labelmaps_hybrid"
OUT_PREV     = BASE / "labelmap_previews_hybrid"
OUT_BGS      = BASE / "labelmap_backgrounds_hybrid"
MANIFEST     = BASE / "masks_manifest_hybrid.csv"

for p in (OUT_LABELS, OUT_PREV, OUT_BGS):
    p.mkdir(parents=True, exist_ok=True)

# ---- Physical voxel size (μm) ----
VOXEL_SIZE = (3.9, 1.0, 1.2)  # (Z,Y,X)
VOXEL_VOL  = float(np.prod(VOXEL_SIZE))

# ---- Surface exclusion ----
Y_IGNORE_TOP_FRAC = 0.15  # ignore top 15% (surface haze)

# ---- Temporal aggregation ----
USE_TEMPORAL_MIP = True    # Use max projection across time
TEMPORAL_MIP_FRAMES = 10   # Frames around peak

# ---- Enhancement (from auto_mask) ----
# High-pass: small Gaussian - big Gaussian
ENHANCE_SIGMA_SMALL = (0.5, 1.0, 1.0)  # (Z,Y,X) - preserves structure
ENHANCE_SIGMA_BIG   = (4.0, 8.0, 8.0)  # (Z,Y,X) - captures background

# ---- Vesselness (from detect_masks) ----
USE_VESSELNESS = True
SATO_SIGMAS = (1, 2, 3, 4, 5)  # Range for different trunk thicknesses

# ---- Thresholding ----
# Use both enhanced intensity AND vesselness
INTENSITY_PERCENTILE = 97.0  # Lowered further
VESSELNESS_PERCENTILE = 94.0  # Lowered further

# ---- Intensity grow (from detect_masks) ----
DO_INTENSITY_GROW = True
GROW_PERCENTILE = 92.0  # Lowered
GROW_DILATION_ITERS = 2

# ---- Volume gates (μm³) - relaxed ----
MIN_VOL_UM3 = 2000.0   # Lowered further
MAX_VOL_UM3 = 250000.0  # Higher

# ---- Geometry filters - VERY RELAXED ----
MIN_Y_SPAN_FRAC = 0.08      # Very low
MIN_ASPECT_Y_OVER_X = 1.2   # Very relaxed
MIN_ASPECT_Y_OVER_Z = 1.2   # Very relaxed
MIN_VERT_COS = 0.50         # ~60° from vertical

# ---- Fragment merging ----
DO_MERGE_FRAGMENTS = True
MERGE_DISTANCE_UM = 15.0    # Merge components within this distance

# ---- Cleanup ----
SLICE_MIN_PIX = 8           # Min pixels per 2D slice

# ---- Deduplication ----
DO_DEDUP = True
DUPLICATE_DICE = 0.55       # Slightly lower threshold
MAX_MASKS_TOTAL = 255


# ================== HELPERS ==================
RE_EVENT_GROUP = re.compile(r"(event_group_\d{4})")

def parse_event_group(stem: str) -> str | None:
    m = RE_EVENT_GROUP.search(stem)
    return m.group(1) if m else None


def select_temporal_mip(crop: np.ndarray, n_frames: int = 10) -> np.ndarray:
    """Select temporal MIP around peak activity frame."""
    if crop.ndim != 4 or crop.shape[0] < 1:
        return None
    
    T, Z, Y, X = crop.shape
    
    # Score each frame by max intensity
    scores = np.array([np.nanmax(crop[t]) for t in range(T)])
    peak = int(np.argmax(scores))
    
    # Temporal MIP around peak
    half = n_frames // 2
    t0 = max(peak - half, 0)
    t1 = min(peak + half + 1, T)
    
    return np.max(crop[t0:t1], axis=0)  # (Z,Y,X)


def enhance_highpass(vol: np.ndarray) -> np.ndarray:
    """High-pass enhancement: small blur - big blur, clamp positive."""
    sm = gaussian_filter(vol, sigma=ENHANCE_SIGMA_SMALL)
    bg = gaussian_filter(vol, sigma=ENHANCE_SIGMA_BIG)
    enh = sm - bg
    enh[enh < 0] = 0
    
    # Normalize
    vmax = np.nanpercentile(enh, 99.9)
    if vmax > 0:
        enh = enh / vmax
    
    return enh.astype(np.float32)


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


def z_mip_2d(vol: np.ndarray) -> np.ndarray:
    return np.nanmax(vol, axis=0).astype(np.float32)


def dice_coeff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    tot = a.sum(dtype=np.int64) + b.sum(dtype=np.int64)
    return (2.0 * inter / tot) if tot > 0 else 0.0


def pca_verticality(mask: np.ndarray) -> float:
    """Returns |vy| of principal axis (0..1). Higher = more vertical."""
    coords = np.argwhere(mask)
    if coords.shape[0] < 20:
        return 0.0
    
    c = coords.mean(axis=0, keepdims=True)
    X = coords - c
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(abs(v[1]))


def merge_nearby_components(lbl: np.ndarray, distance_um: float, voxel_size: tuple) -> np.ndarray:
    """Merge labeled components that are within distance_um of each other."""
    from scipy.ndimage import distance_transform_edt
    
    props = regionprops(lbl)
    if len(props) < 2:
        return lbl
    
    # Convert distance to voxels (use average of Y,X)
    avg_vox = (voxel_size[1] + voxel_size[2]) / 2
    dist_vox = distance_um / avg_vox
    
    # Build adjacency based on distance
    n = len(props)
    merge_groups = {i: {i} for i in range(n)}
    
    for i, p1 in enumerate(props):
        mask1 = (lbl == p1.label)
        # Dilate and check overlap with others
        dilated = binary_dilation(mask1, iterations=int(dist_vox))
        
        for j, p2 in enumerate(props):
            if j <= i:
                continue
            mask2 = (lbl == p2.label)
            if np.any(dilated & mask2):
                # Merge groups
                g1, g2 = None, None
                for k, g in merge_groups.items():
                    if i in g:
                        g1 = k
                    if j in g:
                        g2 = k
                if g1 is not None and g2 is not None and g1 != g2:
                    merge_groups[g1] = merge_groups[g1] | merge_groups[g2]
                    del merge_groups[g2]
    
    # Relabel based on merge groups
    new_lbl = np.zeros_like(lbl)
    new_id = 1
    for group in merge_groups.values():
        for idx in group:
            new_lbl[lbl == props[idx].label] = new_id
        new_id += 1
    
    return new_lbl


def edge_overlay_preview(bg2d: np.ndarray, mask: np.ndarray, title: str, out_path: Path):
    mip_mask = np.max(mask.astype(bool), axis=0)
    edges = cv2.Canny((mip_mask.astype(np.uint8) * 255), 0, 1) > 0
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.imshow(bg2d.astype(np.float32), cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~edges, edges), cmap="autumn", alpha=0.85)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ================== MAIN ==================
def main():
    event_paths = sorted(EVENT_FOLDER.glob(IN_GLOB))
    if not event_paths:
        raise FileNotFoundError(f"No files matched {IN_GLOB} in {EVENT_FOLDER}")
    
    print(f"Found {len(event_paths)} event crops")
    
    raw_masks = []
    st = generate_binary_structure(3, 2)
    
    for path in tqdm(event_paths, desc="Hybrid extraction"):
        if len(raw_masks) >= MAX_MASKS_TOTAL:
            break
        
        eg = parse_event_group(path.stem)
        if not eg:
            continue
        
        # Load and get temporal MIP
        crop = tifffile.imread(path).astype(np.float32)
        if crop.ndim != 4:
            continue
        
        T, Z, Y, X = crop.shape
        y_top = int(Y_IGNORE_TOP_FRAC * Y)
        
        if USE_TEMPORAL_MIP:
            vol = select_temporal_mip(crop, TEMPORAL_MIP_FRAMES)
        else:
            # Use max across all time
            vol = np.max(crop, axis=0)
        
        if vol is None:
            del crop
            continue
        
        # Background for visualization
        bg2d = contrast_stretch_01(z_mip_2d(vol), 2, 98)
        
        # === ENHANCEMENT ===
        enh = enhance_highpass(vol)
        
        # === VESSELNESS (optional) ===
        if USE_VESSELNESS:
            vess = sato(enh, sigmas=SATO_SIGMAS, black_ridges=False).astype(np.float32)
            # Normalize
            vmax = np.nanpercentile(vess, 99.9)
            if vmax > 0:
                vess = vess / vmax
        else:
            vess = None
        
        # === THRESHOLDING ===
        # Mask out surface for threshold computation
        enh_for_thr = enh.copy()
        if y_top > 0:
            enh_for_thr[:, :y_top, :] = np.nan
        
        # Intensity threshold
        thr_int = np.nanpercentile(enh_for_thr, INTENSITY_PERCENTILE)
        cand_int = enh > thr_int
        
        # Vesselness threshold (if used)
        if vess is not None:
            vess_for_thr = vess.copy()
            if y_top > 0:
                vess_for_thr[:, :y_top, :] = np.nan
            thr_vess = np.nanpercentile(vess_for_thr, VESSELNESS_PERCENTILE)
            cand_vess = vess > thr_vess
            # Combine: require EITHER high intensity OR high vesselness
            cand = cand_int | cand_vess
        else:
            cand = cand_int
        
        # Zero out surface
        if y_top > 0:
            cand[:, :y_top, :] = False
        
        print(f"  [{eg}] After threshold: {cand.sum()} voxels")
        
        if not cand.any():
            print(f"  [{eg}] No candidates after threshold")
            del crop, vol, enh
            gc.collect()
            continue
        
        # === INTENSITY GROW ===
        if DO_INTENSITY_GROW:
            vol_for_thr = vol.copy()
            if y_top > 0:
                vol_for_thr[:, :y_top, :] = np.nan
            thr_grow = np.nanpercentile(vol_for_thr, GROW_PERCENTILE)
            vol_hi = vol > thr_grow
            if y_top > 0:
                vol_hi[:, :y_top, :] = False
            
            g = cand.copy()
            for _ in range(GROW_DILATION_ITERS):
                g = binary_dilation(g, structure=st)
            cand = g & vol_hi
            print(f"  [{eg}] After grow: {cand.sum()} voxels")
        
        # === CLEANUP ===
        min_vox = max(1, int(MIN_VOL_UM3 / VOXEL_VOL))
        cand = remove_small_objects(cand.astype(bool), min_size=min_vox, connectivity=3)
        
        print(f"  [{eg}] After cleanup (min {min_vox} vox): {cand.sum()} voxels")
        
        if not cand.any():
            print(f"  [{eg}] No candidates after cleanup")
            del crop, vol, enh
            gc.collect()
            continue
        
        # === CONNECTED COMPONENTS ===
        lbl = label(cand, connectivity=3)
        n_components = lbl.max()
        print(f"  [{eg}] {n_components} components before geometry filter")
        
        # === MERGE FRAGMENTS ===
        if DO_MERGE_FRAGMENTS and n_components > 1:
            lbl = merge_nearby_components(lbl, MERGE_DISTANCE_UM, VOXEL_SIZE)
            print(f"  [{eg}] {lbl.max()} components after merging")
        
        # === GEOMETRY FILTERS (relaxed) ===
        props = regionprops(lbl)
        accepted = 0
        
        for r in props:
            if len(raw_masks) >= MAX_MASKS_TOTAL:
                break
            
            voxels = int(r.area)
            vol_um3 = voxels * VOXEL_VOL
            
            if vol_um3 < MIN_VOL_UM3:
                print(f"    comp {r.label}: REJECT vol={vol_um3:.0f} < {MIN_VOL_UM3}")
                continue
            if MAX_VOL_UM3 and vol_um3 > MAX_VOL_UM3:
                print(f"    comp {r.label}: REJECT vol={vol_um3:.0f} > {MAX_VOL_UM3}")
                continue
            
            z0, y0, x0, z1, y1, x1 = r.bbox
            span_y = y1 - y0
            span_x = max(1, x1 - x0)
            span_z = max(1, z1 - z0)
            
            min_y_span = int(MIN_Y_SPAN_FRAC * Y)
            if span_y < min_y_span:
                print(f"    comp {r.label}: REJECT y_span={span_y} < {min_y_span}")
                continue
            if (span_y / span_x) < MIN_ASPECT_Y_OVER_X:
                print(f"    comp {r.label}: REJECT aspect_yx={span_y/span_x:.2f} < {MIN_ASPECT_Y_OVER_X}")
                continue
            if (span_y / span_z) < MIN_ASPECT_Y_OVER_Z:
                print(f"    comp {r.label}: REJECT aspect_yz={span_y/span_z:.2f} < {MIN_ASPECT_Y_OVER_Z}")
                continue
            
            m = (lbl == r.label).astype(np.uint8)
            vert = pca_verticality(m)
            
            if vert < MIN_VERT_COS:
                print(f"    comp {r.label}: REJECT vert={vert:.2f} < {MIN_VERT_COS}")
                continue
            
            print(f"    comp {r.label}: ACCEPT vol={vol_um3:.0f} vert={vert:.2f}")
            accepted += 1
            raw_masks.append({
                "mask": m,
                "vol_um3": vol_um3,
                "event_group": eg,
                "event_path": str(path),
                "bg2d": bg2d.astype(np.float16),
                "verticality": vert,
            })
        
        print(f"  [{eg}] Accepted {accepted}/{len(props)} components")
        
        del crop, vol, enh, cand, lbl
        gc.collect()
    
    if not raw_masks:
        print("No masks extracted.")
        return
    
    # === DEDUPLICATION ===
    if DO_DEDUP:
        print("Deduplicating...")
        unique = []
        for item in raw_masks:
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
        unique = raw_masks
    
    print(f"[✓] Retained {len(unique)} unique masks (from {len(raw_masks)} candidates)")
    
    # === SAVE OUTPUTS ===
    print("Saving outputs...")
    with open(MANIFEST, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=[
            "mask_id", "labelmap_path", "background_path", "event_group",
            "source_file", "voxel_size_z", "voxel_size_y", "voxel_size_x",
            "volume_um3", "verticality"
        ])
        writer.writeheader()
        
        for i, item in enumerate(unique):
            m = item["mask"].astype(np.uint8)
            bg2d = item["bg2d"]
            
            mask_path = OUT_LABELS / f"dend_{i:03d}_labelmap.tif"
            bg_path = OUT_BGS / f"dend_{i:03d}_background_2dMIP.tif"
            prev_path = OUT_PREV / f"dend_{i:03d}_preview.png"
            
            tifffile.imwrite(mask_path, (m * (i + 1)).astype(np.uint16))
            tifffile.imwrite(bg_path, bg2d, dtype=np.float16)
            edge_overlay_preview(bg2d, m, f"dend_{i:03d} ({item['event_group']})", prev_path)
            
            writer.writerow({
                "mask_id": i,
                "labelmap_path": str(mask_path),
                "background_path": str(bg_path),
                "event_group": item["event_group"],
                "source_file": Path(item["event_path"]).name,
                "voxel_size_z": VOXEL_SIZE[0],
                "voxel_size_y": VOXEL_SIZE[1],
                "voxel_size_x": VOXEL_SIZE[2],
                "volume_um3": item["vol_um3"],
                "verticality": item["verticality"],
            })
    
    print(f"Manifest: {MANIFEST}")
    print("Hybrid M2 complete ✅")


if __name__ == "__main__":
    main()
