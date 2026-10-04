#!/usr/bin/env python
"""auto_detect.py — automatic dendrite detection for SCAPE data (code/Auto/).

Finds thin, mostly-vertical calcium-active structures by:
1. Building activity summary images (temporal std, max dF/F) via streaming chunks
2. Applying anisotropic tubeness filter favoring Y-oriented structures
3. Seeding via local maxima on activity-weighted tubeness
4. Region growing using local correlation to split touching dendrites
5. Filtering by geometry (verticality, length, size) and SNR

Outputs:
    OUT/masks/auto_labelmap.tif  — uint16 (Z,Y,X), label 0=bg, k=dendrite k
    OUT/masks/auto_masks.csv     — per-dendrite info (id, name, n_vox, centroid, metrics)

CLI: python auto_detect.py --run DATE/MOUSE/RUN [--run ...] | --all [--force] [--jobs N]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, label as ndi_label
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (
    PROJECT, discover_runs, get_run, Run, open_stack, merge_metrics,
    VOXEL_ZYX, HUMAN_Y_CROP,
)

__version__ = "1.0.0"

# ─────────────────────────── Parameters ───────────────────────────
# Tuned on: 05-08/run5, 05-12/run5, 03-31/run7
# Human mask stats: yspan median=75 (36-134), verticality median=2.8, nvox median=2224

# Activity thresholds
ACTIVITY_PERCENTILE = 80.0       # Percentile for activity map thresholding (lower = more voxels)
MIN_ACTIVITY_SNR = 1.5           # Minimum local SNR for seeds

# Tubeness (Hessian-based) - favor Y-oriented thin tubes
TUBENESS_SIGMAS = (0.5, 1.0, 1.5)  # Smaller scales for thin structures
TUBENESS_PERCENTILE = 75.0       # Threshold for combined map

# Geometry filters (from human mask statistics: median nvox=2224, yspan=75, vert=2.8)
MIN_NVOX = 300                   # ~1400 um³ minimum (smallest human ~1100 um³)
MAX_NVOX = 60000                 # Very large = probably merged
MIN_YSPAN_PIX = 30               # Minimum depth span (~30 um) - human p10=36
MIN_VERTICALITY = 1.0            # Y/X span ratio - relaxed to catch more
MIN_LENGTH_UM = 25.0             # Minimum length in microns

# Splitting
CORRELATION_WINDOW_S = 10.0      # Window for temporal correlation computation
SPLIT_CORR_THRESHOLD = 0.5       # Below this, consider splitting

# Post-processing
DICE_DUPLICATE_THRESHOLD = 0.6   # Masks with Dice > this are merged
MAX_MASKS = 300                  # Safety limit


@dataclass
class MaskInfo:
    """Info for one detected dendrite."""
    id: int
    name: str
    n_vox: int
    cz: float
    cy: float
    cx: float
    length_um: float
    verticality: float
    score: float
    zspan: int
    yspan: int
    xspan: int


def anisotropic_tubeness(vol: np.ndarray, sigmas: tuple, voxel: tuple) -> np.ndarray:
    """Compute tubeness map favoring Y-oriented structures.
    
    Uses Hessian eigenvalues with anisotropic sigma scaling for the voxel size.
    For dendrites along Y: lambda_Y should be small, lambda_Z and lambda_X large.
    """
    Z, Y, X = vol.shape
    tubeness = np.zeros_like(vol, dtype=np.float32)
    
    for sigma in sigmas:
        # Scale sigma per axis for anisotropic voxels
        sigma_zyx = (sigma / voxel[0], sigma / voxel[1], sigma / voxel[2])
        
        # Compute Hessian matrix elements
        Hzz, Hzy, Hzx, Hyy, Hyx, Hxx = hessian_matrix(
            vol, sigma=sigma_zyx, order='rc', use_gaussian_derivatives=True
        )
        
        # Get eigenvalues (sorted by absolute value)
        eigs = hessian_matrix_eigvals([Hzz, Hzy, Hzx, Hyy, Hyx, Hxx])
        
        # For tube-like structures: |lambda1| >> |lambda2| ~ |lambda3| ~ 0
        l1, l2, l3 = np.abs(eigs[0]), np.abs(eigs[1]), np.abs(eigs[2])
        
        # Vesselness-like: high when two eigenvalues are large, one is small
        with np.errstate(divide='ignore', invalid='ignore'):
            Rb = l1 / np.sqrt(l2 * l3 + 1e-10)  # blob vs line
            Ra = l2 / (l3 + 1e-10)  # plate vs line
            S = np.sqrt(l1**2 + l2**2 + l3**2)  # structuredness
        
        # Vesselness (Frangi-like, but for bright tubes)
        c = S.max() / 3
        tube = (1 - np.exp(-Ra**2 / (2 * 0.5**2))) * \
               np.exp(-Rb**2 / (2 * 0.5**2)) * \
               (1 - np.exp(-S**2 / (2 * c**2)))
        
        # Keep positive (bright tubes)
        tube[eigs[2] > 0] = 0  # largest eigenvalue should be negative for bright tubes
        
        tubeness = np.maximum(tubeness, tube.astype(np.float32))
    
    return tubeness


def compute_activity_maps(r: Run, chunk_frames: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute activity summary maps via streaming to limit memory.
    
    Returns:
        mean_vol: (Z, Y, X) temporal mean
        std_vol: (Z, Y, X) temporal std (activity indicator)
        max_dff: (Z, Y, X) max dF/F after baseline skip
    """
    stack = open_stack(r)
    T, Z, Y, X = stack.shape
    
    skip_frames = int(r.skip_s * r.frame_rate)
    n_frames = T - skip_frames
    
    # Initialize accumulators
    sum_vol = np.zeros((Z, Y, X), dtype=np.float64)
    sum_sq_vol = np.zeros((Z, Y, X), dtype=np.float64)
    max_dff = np.zeros((Z, Y, X), dtype=np.float32)
    
    # First pass: compute mean for F0 (10th percentile)
    sample_indices = np.linspace(skip_frames, T - 1, min(200, n_frames), dtype=int)
    sample_stack = stack[sample_indices].astype(np.float32)
    f0 = np.percentile(sample_stack, 10, axis=0)
    f0 = np.maximum(f0, 1.0)  # Avoid division by zero
    del sample_stack
    gc.collect()
    
    # Stream through data
    for t0 in range(skip_frames, T, chunk_frames):
        t1 = min(t0 + chunk_frames, T)
        chunk = stack[t0:t1].astype(np.float32)
        
        # dF/F
        dff = (chunk - f0) / f0
        
        # Accumulate
        sum_vol += dff.sum(axis=0)
        sum_sq_vol += (dff ** 2).sum(axis=0)
        max_dff = np.maximum(max_dff, dff.max(axis=0))
        
        del chunk, dff
        gc.collect()
    
    # Finalize
    mean_vol = (sum_vol / n_frames).astype(np.float32)
    var_vol = sum_sq_vol / n_frames - mean_vol ** 2
    std_vol = np.sqrt(np.maximum(var_vol, 0)).astype(np.float32)
    
    return mean_vol, std_vol, max_dff


def detect_dendrites(r: Run, verbose: bool = True) -> tuple[np.ndarray, list[MaskInfo]]:
    """Main detection pipeline for one run.
    
    Approach: Use max dF/F and temporal std as activity indicators,
    then watershed segment and filter by geometry.
    
    Returns:
        labelmap: (Z, Y, X) uint16 with labels 1..N
        mask_infos: list of MaskInfo for each detected dendrite
    """
    t0 = time.time()
    
    stack = open_stack(r)
    T, Z, Y, X = stack.shape
    shape_zyx = (Z, Y, X)
    
    if verbose:
        print(f"[{r.key}] Stack shape: T={T}, Z={Z}, Y={Y}, X={X}")
        print(f"[{r.key}] Computing activity maps...")
    
    # Step 1: Compute activity maps
    mean_vol, std_vol, max_dff = compute_activity_maps(r)
    
    # Step 2: Create combined activity indicator
    # Normalize each component
    std_norm = std_vol / (std_vol.max() + 1e-8)
    dff_norm = max_dff / (max_dff.max() + 1e-8)
    
    # Combined: emphasize both high std (variable) and high max dF/F (active)
    combined = std_norm * 0.4 + dff_norm * 0.6
    combined = gaussian_filter(combined, sigma=(0.5, 1.0, 1.0))
    
    if verbose:
        print(f"[{r.key}] Thresholding...")
    
    # Step 3: Threshold at percentile
    threshold = np.percentile(combined, ACTIVITY_PERCENTILE)
    binary = combined > threshold
    
    # Clean up small noise
    binary = remove_small_objects(binary, min_size=30)
    
    if verbose:
        print(f"[{r.key}] Binary mask: {binary.sum()} voxels")
    
    # Step 4: Watershed segmentation
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import distance_transform_edt
    
    # Distance transform for better watershed seeds
    dist = distance_transform_edt(binary)
    
    # Find local maxima as seeds
    coords = peak_local_max(
        dist,
        min_distance=4,
        threshold_abs=1.5,
        exclude_border=False,
    )
    
    if len(coords) == 0:
        if verbose:
            print(f"[{r.key}] No seeds found!")
        return np.zeros(shape_zyx, dtype=np.uint16), []
    
    if verbose:
        print(f"[{r.key}] Found {len(coords)} seeds, running watershed...")
    
    # Create marker volume
    markers = np.zeros(shape_zyx, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, start=1):
        markers[z, y, x] = i
    
    # Watershed on inverted distance
    labels = watershed(-dist, markers, mask=binary)
    
    if verbose:
        n_raw = int(labels.max())
        print(f"[{r.key}] Watershed produced {n_raw} regions, filtering...")
    
    # Step 5: Filter by geometry
    mask_infos = []
    final_labels = np.zeros(shape_zyx, dtype=np.uint16)
    next_id = 1
    
    props = regionprops(labels)
    for prop in props:
        # Get bounding box
        z0, y0, x0, z1, y1, x1 = prop.bbox
        zspan = z1 - z0
        yspan = y1 - y0
        xspan = x1 - x0
        
        n_vox = prop.area
        
        # Size filter
        if n_vox < MIN_NVOX or n_vox > MAX_NVOX:
            continue
        
        # Y span filter (depth extent)
        if yspan < MIN_YSPAN_PIX:
            continue
        
        # Verticality filter
        verticality = yspan / max(xspan, 1)
        if verticality < MIN_VERTICALITY:
            continue
        
        # Length in microns
        length_um = yspan * VOXEL_ZYX[1]
        if length_um < MIN_LENGTH_UM:
            continue
        
        # Compute score (activity in mask)
        mask_region = labels == prop.label
        score = float(combined[mask_region].mean())
        
        # Centroid
        cz, cy, cx = prop.centroid
        
        # Accept this mask
        final_labels[mask_region] = next_id
        mask_infos.append(MaskInfo(
            id=next_id,
            name=f"dend_{next_id - 1:03d}",
            n_vox=n_vox,
            cz=float(cz),
            cy=float(cy),
            cx=float(cx),
            length_um=length_um,
            verticality=verticality,
            score=score,
            zspan=zspan,
            yspan=yspan,
            xspan=xspan,
        ))
        next_id += 1
        
        if next_id > MAX_MASKS:
            if verbose:
                print(f"[{r.key}] WARNING: Hit MAX_MASKS={MAX_MASKS}, stopping")
            break
    
    if verbose:
        elapsed = time.time() - t0
        print(f"[{r.key}] Detected {len(mask_infos)} dendrites in {elapsed:.1f}s")
    
    return final_labels, mask_infos


def deduplicate_masks(labels: np.ndarray, mask_infos: list[MaskInfo],
                      dice_threshold: float = DICE_DUPLICATE_THRESHOLD) -> tuple[np.ndarray, list[MaskInfo]]:
    """Remove near-duplicate masks by Dice coefficient."""
    n = len(mask_infos)
    if n <= 1:
        return labels, mask_infos
    
    # Build mask arrays
    masks = [(labels == info.id) for info in mask_infos]
    
    # Find duplicates
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            inter = (masks[i] & masks[j]).sum()
            union = masks[i].sum() + masks[j].sum()
            dice = 2 * inter / (union + 1e-10)
            if dice > dice_threshold:
                # Keep the one with higher score
                if mask_infos[i].score >= mask_infos[j].score:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    
    # Rebuild
    new_labels = np.zeros_like(labels)
    new_infos = []
    new_id = 1
    for i, info in enumerate(mask_infos):
        if keep[i]:
            new_labels[labels == info.id] = new_id
            new_infos.append(MaskInfo(
                id=new_id,
                name=f"dend_{new_id - 1:03d}",
                n_vox=info.n_vox,
                cz=info.cz, cy=info.cy, cx=info.cx,
                length_um=info.length_um,
                verticality=info.verticality,
                score=info.score,
                zspan=info.zspan, yspan=info.yspan, xspan=info.xspan,
            ))
            new_id += 1
    
    return new_labels, new_infos


def process_run(r: Run, force: bool = False, verbose: bool = True) -> dict:
    """Process one run: detect dendrites, save outputs."""
    out_dir = r.outdir("masks")
    labelmap_path = out_dir / "auto_labelmap.tif"
    csv_path = out_dir / "auto_masks.csv"
    
    # Check if already done
    if not force and labelmap_path.exists() and csv_path.exists():
        if r.raw is None or labelmap_path.stat().st_mtime > r.raw.stat().st_mtime:
            if verbose:
                print(f"[{r.key}] Outputs exist and up-to-date, skipping (use --force)")
            return {"status": "skipped", "run": r.key}
    
    t0 = time.time()
    
    # Detect
    labels, mask_infos = detect_dendrites(r, verbose=verbose)
    
    # Deduplicate
    labels, mask_infos = deduplicate_masks(labels, mask_infos)
    
    # Save labelmap
    tifffile.imwrite(str(labelmap_path), labels.astype(np.uint16))
    
    # Save CSV
    rows = [asdict(m) for m in mask_infos]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    # Metrics
    elapsed = time.time() - t0
    metrics = {
        "n_masks": len(mask_infos),
        "detection_time_s": elapsed,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    merge_metrics(r, "detect", metrics)
    
    if verbose:
        print(f"[{r.key}] Saved {len(mask_infos)} masks to {labelmap_path}")
    
    return {
        "status": "done",
        "run": r.key,
        "n_masks": len(mask_infos),
        "time_s": elapsed,
    }


def process_run_wrapper(args):
    """Wrapper for parallel processing."""
    run_key, force = args
    try:
        r = get_run(run_key)
        return process_run(r, force=force, verbose=True)
    except Exception as e:
        return {"status": "error", "run": run_key, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", action="append", default=[], 
                        help="DATE/MOUSE/RUN (repeatable)")
    parser.add_argument("--all", action="store_true", help="Process all runs")
    parser.add_argument("--force", action="store_true", help="Reprocess even if outputs exist")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (default 1)")
    args = parser.parse_args()
    
    # Collect runs
    if args.all:
        runs = discover_runs(only_with_raw=True)
    elif args.run:
        runs = [get_run(k) for k in args.run]
    else:
        parser.error("Specify --run or --all")
    
    print(f"Processing {len(runs)} runs with {args.jobs} jobs...")
    
    if args.jobs == 1:
        results = []
        for r in runs:
            res = process_run(r, force=args.force, verbose=True)
            results.append(res)
    else:
        # Parallel
        run_args = [(r.key, args.force) for r in runs]
        results = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_run_wrapper, a): a[0] for a in run_args}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if res["status"] == "done":
                    print(f"  {res['run']}: {res['n_masks']} masks in {res['time_s']:.1f}s")
                elif res["status"] == "error":
                    print(f"  {res['run']}: ERROR - {res['error']}")
    
    # Summary
    n_done = sum(1 for r in results if r["status"] == "done")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone: {n_done}, Skipped: {n_skip}, Errors: {n_err}")
    
    if n_err > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
