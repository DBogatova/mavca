#!/usr/bin/env python
"""extract_traces.py — extract dF/F traces from auto and human masks (code/Auto/).

Uses the same extractor for both mask sets (apples-to-apples comparison):
- Core = mask voxels (or eroded mask if large enough)
- Shell = dilation(ball(3)) & ~dilation(ball(2)) of mask, excluding all other masks
- F0 = 10th percentile after skipping skip_s
- dF/F = (F_core - F_shell - F0) / F0 * 100 (percent)

Outputs:
    OUT/traces/dff_auto.csv               — Frame, time_s, dend_000..
    OUT/traces/dff_human_sameextractor.csv — same columns, human mask names (if available)

CLI: python extract_traces.py --run DATE/MOUSE/RUN [--run ...] | --all [--force] [--jobs N]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter1d
from skimage.morphology import ball

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (
    PROJECT, discover_runs, get_run, Run, open_stack, human_masks, merge_metrics,
    VOXEL_ZYX, HUMAN_Y_CROP,
)

__version__ = "1.0.0"

# Match M4 conventions
SKIP_PERCENTILE = 10  # F0 = 10th percentile
SMOOTH_SIGMA = 0.5    # Light temporal smoothing


def load_auto_masks(r: Run, shape_zyx: tuple) -> tuple[list[tuple[str, np.ndarray]], Path]:
    """Load auto-detected masks as (name, bool_mask) tuples.
    
    Prefers auto_labelmap_reviewed.tif when it exists (from napari review).
    Returns (masks, path_used).
    """
    # Prefer reviewed labelmap when it exists (from napari review tool)
    reviewed_path = r.out / "masks" / "auto_labelmap_reviewed.tif"
    original_path = r.out / "masks" / "auto_labelmap.tif"
    
    if reviewed_path.exists():
        labelmap_path = reviewed_path
    elif original_path.exists():
        labelmap_path = original_path
    else:
        return [], None
    
    labels = tifffile.imread(str(labelmap_path))
    if labels.shape != shape_zyx:
        raise ValueError(f"Auto labelmap shape {labels.shape} != stack shape {shape_zyx}")
    
    masks = []
    for label_id in range(1, int(labels.max()) + 1):
        mask = labels == label_id
        if mask.any():
            name = f"dend_{label_id - 1:03d}"
            masks.append((name, mask))
    return masks, labelmap_path


def build_shell(mask: np.ndarray, exclude_mask: np.ndarray) -> np.ndarray:
    """Build background shell: dilation(ball(3)) & ~dilation(ball(2)), excluding other masks."""
    inner = binary_dilation(mask, structure=ball(2))
    outer = binary_dilation(mask, structure=ball(3))
    shell = outer & ~inner
    shell = shell & ~exclude_mask
    return shell


def extract_traces(stack, masks: list[tuple[str, np.ndarray]], 
                   frame_rate: float, skip_s: float,
                   verbose: bool = True) -> pd.DataFrame:
    """Extract dF/F traces for all masks using core-shell background subtraction."""
    T, Z, Y, X = stack.shape
    skip_frames = int(round(skip_s * frame_rate))
    n_out_frames = T - skip_frames
    
    if n_out_frames <= 0:
        raise ValueError(f"No frames left after skipping {skip_s}s ({skip_frames} frames)")
    
    # Build combined mask for shell exclusion
    all_masks = np.zeros((Z, Y, X), dtype=bool)
    for _, m in masks:
        all_masks |= m
    
    # Build core and shell indices for each mask
    mask_data = []
    for name, mask in masks:
        # Core = eroded mask if large enough, else full mask
        core = binary_erosion(mask, structure=ball(1))
        if not core.any():
            core = mask.copy()
        
        # Shell = background ring, excluding all masks
        shell = build_shell(mask, all_masks)
        
        core_idx = np.argwhere(core)
        shell_idx = np.argwhere(shell) if shell.any() else np.zeros((0, 3), dtype=int)
        
        mask_data.append({
            "name": name,
            "core_idx": core_idx,
            "shell_idx": shell_idx,
        })
    
    # First pass: compute F0 (10th percentile) for each mask
    if verbose:
        print(f"  Computing F0 baseline (10th percentile, skipping {skip_s}s)...")
    
    n_sample = min(300, n_out_frames)
    sample_frames = np.linspace(skip_frames, T - 1, n_sample, dtype=int)
    
    f0_values = {d["name"]: [] for d in mask_data}
    for fi in sample_frames:
        vol = stack[fi].astype(np.float32)
        for d in mask_data:
            core_val = vol[d["core_idx"][:, 0], d["core_idx"][:, 1], d["core_idx"][:, 2]].mean()
            if len(d["shell_idx"]) > 0:
                shell_val = vol[d["shell_idx"][:, 0], d["shell_idx"][:, 1], d["shell_idx"][:, 2]].mean()
            else:
                shell_val = 0
            f0_values[d["name"]].append(core_val - shell_val)
        del vol
    
    f0 = {name: max(np.percentile(vals, SKIP_PERCENTILE), 1.0) for name, vals in f0_values.items()}
    
    # Second pass: extract full traces
    if verbose:
        print(f"  Extracting traces for {len(masks)} masks...")
    
    traces = {d["name"]: np.zeros(n_out_frames, dtype=np.float32) for d in mask_data}
    
    for t_out in range(n_out_frames):
        t_raw = t_out + skip_frames
        vol = stack[t_raw].astype(np.float32)
        
        for d in mask_data:
            core_val = vol[d["core_idx"][:, 0], d["core_idx"][:, 1], d["core_idx"][:, 2]].mean()
            if len(d["shell_idx"]) > 0:
                shell_val = vol[d["shell_idx"][:, 0], d["shell_idx"][:, 1], d["shell_idx"][:, 2]].mean()
            else:
                shell_val = 0
            
            f = core_val - shell_val
            dff = (f - f0[d["name"]]) / f0[d["name"]] * 100
            traces[d["name"]][t_out] = dff
        
        del vol
        if verbose and (t_out + 1) % 100 == 0:
            print(f"    Frame {t_out + 1}/{n_out_frames}")
    
    # Apply light smoothing
    for name in traces:
        traces[name] = gaussian_filter1d(traces[name], sigma=SMOOTH_SIGMA)
    
    # Build DataFrame
    # time_s = 0 exactly skip_s after frame 0 (behavior is shifted by the same skip_s), so the
    # trace clock and the behavior clock agree to the sub-frame level
    cols = {"Frame": np.arange(skip_frames, T), "time_s": (np.arange(skip_frames, T) / frame_rate - skip_s)}
    cols.update({name: traces[name] for name in sorted(traces.keys())})
    df = pd.DataFrame(cols)
    
    return df


def compare_with_existing(df_new: pd.DataFrame, existing_path: Path, 
                          name_prefix: str = "dend_") -> dict:
    """Compare new traces with existing ones, return correlation stats."""
    if not existing_path.exists():
        return {}
    
    df_old = pd.read_csv(existing_path)
    
    new_masks = [c for c in df_new.columns if c.startswith(name_prefix)]
    old_masks = [c for c in df_old.columns if c.startswith(name_prefix)]
    common = set(new_masks) & set(old_masks)
    
    if not common:
        return {"n_common": 0}
    
    df_new = df_new.set_index("Frame")
    df_old = df_old.set_index("Frame")
    common_frames = df_new.index.intersection(df_old.index)
    
    if len(common_frames) < 10:
        return {"n_common": len(common), "n_frames": len(common_frames)}
    
    correlations = []
    for name in common:
        new_ts = df_new.loc[common_frames, name].values
        old_ts = df_old.loc[common_frames, name].values
        
        new_centered = new_ts - new_ts.mean()
        old_centered = old_ts - old_ts.mean()
        denom = np.sqrt((new_centered ** 2).sum() * (old_centered ** 2).sum())
        if denom > 1e-6:
            r = (new_centered * old_centered).sum() / denom
            correlations.append(r)
    
    return {
        "n_common": len(common),
        "n_frames": len(common_frames),
        "median_r": float(np.median(correlations)) if correlations else None,
        "mean_r": float(np.mean(correlations)) if correlations else None,
        "min_r": float(np.min(correlations)) if correlations else None,
    }


def process_run(r: Run, force: bool = False, verbose: bool = True) -> dict:
    """Process one run: extract traces for auto and human masks."""
    out_dir = r.outdir("traces")
    auto_csv = out_dir / "dff_auto.csv"
    human_csv = out_dir / "dff_human_sameextractor.csv"
    
    # Check for auto labelmap (prefer reviewed version)
    reviewed_path = r.out / "masks" / "auto_labelmap_reviewed.tif"
    original_path = r.out / "masks" / "auto_labelmap.tif"
    if reviewed_path.exists():
        auto_labelmap = reviewed_path
    elif original_path.exists():
        auto_labelmap = original_path
    else:
        if verbose:
            print(f"[{r.key}] No auto labelmap found, run auto_detect.py first")
        return {"status": "error", "run": r.key, "error": "no auto labelmap"}
    
    if not force and auto_csv.exists():
        if auto_csv.stat().st_mtime > auto_labelmap.stat().st_mtime:
            if verbose:
                print(f"[{r.key}] Traces up-to-date, skipping (use --force)")
            return {"status": "skipped", "run": r.key}
    
    t0 = time.time()
    
    stack = open_stack(r)
    T, Z, Y, X = stack.shape
    shape_zyx = (Z, Y, X)
    
    if verbose:
        print(f"[{r.key}] Stack shape: T={T}, Z={Z}, Y={Y}, X={X}")
    
    # Load auto masks (with preference for reviewed version)
    auto_masks, labelmap_used = load_auto_masks(r, shape_zyx)
    if verbose:
        print(f"[{r.key}] Loaded {len(auto_masks)} auto masks from {labelmap_used.name if labelmap_used else 'none'}")
    
    # Extract auto traces
    if auto_masks:
        df_auto = extract_traces(stack, auto_masks, r.frame_rate, r.skip_s, verbose)
        df_auto.to_csv(auto_csv, index=False)
        if verbose:
            print(f"[{r.key}] Saved auto traces to {auto_csv}")
    else:
        df_auto = None
    
    # Load and process human masks
    h_masks = human_masks(r, shape_zyx)
    human_comparison = {}
    
    if h_masks:
        if verbose:
            print(f"[{r.key}] Extracting traces for {len(h_masks)} human masks...")
        
        df_human = extract_traces(stack, h_masks, r.frame_rate, r.skip_s, verbose)
        df_human.to_csv(human_csv, index=False)
        
        if r.human_traces is not None:
            human_comparison = compare_with_existing(df_human, r.human_traces)
            if verbose and human_comparison.get("median_r") is not None:
                print(f"[{r.key}] Human trace comparison: median_r={human_comparison['median_r']:.3f}")
    
    elapsed = time.time() - t0
    
    metrics = {
        "n_auto_masks": len(auto_masks) if auto_masks else 0,
        "n_human_masks": len(h_masks),
        "extraction_time_s": elapsed,
        "human_trace_comparison": human_comparison,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    merge_metrics(r, "traces", metrics)
    
    return {
        "status": "done",
        "run": r.key,
        "n_auto": len(auto_masks) if auto_masks else 0,
        "n_human": len(h_masks),
        "time_s": elapsed,
        "human_comparison": human_comparison,
    }


def process_run_wrapper(args):
    """Wrapper for parallel processing."""
    run_key, force = args
    try:
        r = get_run(run_key)
        return process_run(r, force=force, verbose=True)
    except Exception as e:
        import traceback
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
        run_args = [(r.key, args.force) for r in runs]
        results = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_run_wrapper, a): a[0] for a in run_args}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
    
    # Summary
    n_done = sum(1 for r in results if r["status"] == "done")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone: {n_done}, Skipped: {n_skip}, Errors: {n_err}")
    
    human_rs = [r["human_comparison"].get("median_r") 
                for r in results if r.get("human_comparison", {}).get("median_r") is not None]
    if human_rs:
        print(f"\nHuman trace sanity check (same extractor vs existing):")
        print(f"  Median r across runs: {np.median(human_rs):.3f}")
        print(f"  Range: [{min(human_rs):.3f}, {max(human_rs):.3f}]")
    
    return 1 if n_err > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
