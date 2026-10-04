#!/usr/bin/env python
"""global_ca.py - Compute global ΔF/F from raw SCAPE stacks.

Output: OUT/traces/global_ca.csv with columns time_s, global_dff (%)
        time_s = frame/frame_rate - skip_s; frames before skip dropped.

Global ΔF/F = mean of live voxels (>5th pct of temporal mean), F0 = 10th pct.
Streams the raw memmap in chunks (memory-efficient).

CLI:
    python code/Auto/global_ca.py --run 2026-03-31/rbp4_132_phpeb/run7
    python code/Auto/global_ca.py --all [--jobs 2] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scape_common import (
    PROJECT, discover_runs, get_run, open_stack, merge_metrics
)

CHUNK_T = 50


def compute_global_ca(r) -> dict:
    """Compute global ΔF/F for a single run."""
    t0 = time.time()
    
    stack = open_stack(r)
    T, Z, Y, X = stack.shape
    skip_frames = int(r.skip_s * r.frame_rate)
    
    # Live voxels (>5th pct of temporal mean)
    sample = np.asarray(stack[:min(100, T)]).astype(np.float32)
    tmean = sample.mean(axis=0)
    thr = np.percentile(tmean, 5)
    live_idx = np.flatnonzero((tmean > thr).ravel())
    n_live = live_idx.size
    del sample, tmean
    
    # F0 = 10th percentile after skip
    f0_vals = []
    for t in range(skip_frames, min(skip_frames + 500, T)):
        frame = np.asarray(stack[t]).astype(np.float32).ravel()
        f0_vals.append(frame[live_idx].mean())
    f0 = np.percentile(f0_vals, 10)
    del f0_vals
    
    # Global ΔF/F
    global_dff = np.empty(T, dtype=np.float32)
    for t0_chunk in range(0, T, CHUNK_T):
        t1_chunk = min(t0_chunk + CHUNK_T, T)
        chunk = np.asarray(stack[t0_chunk:t1_chunk]).astype(np.float32)
        for i in range(chunk.shape[0]):
            frame_mean = chunk[i].ravel()[live_idx].mean()
            global_dff[t0_chunk + i] = (frame_mean - f0) / (f0 + 1e-6) * 100.0
        del chunk
    
    del stack
    
    # Crop to after skip_s
    global_dff = global_dff[skip_frames:]
    T_out = len(global_dff)
    time_s = np.arange(T_out) / r.frame_rate
    
    # Save
    out_dir = r.outdir("traces")
    df = pd.DataFrame({"time_s": time_s, "global_dff": global_dff})
    out_path = out_dir / "global_ca.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")
    
    elapsed = time.time() - t0
    
    metrics = {
        "T_raw": int(T),
        "T_out": int(T_out),
        "n_live_voxels": int(n_live),
        "f0": float(f0),
        "global_dff_min": float(global_dff.min()),
        "global_dff_max": float(global_dff.max()),
        "global_dff_mean": float(global_dff.mean()),
        "elapsed_s": round(elapsed, 1),
    }
    merge_metrics(r, "global_ca", metrics)
    
    return metrics


def process_run(key: str, force: bool) -> tuple[str, bool, str]:
    """Process a single run."""
    try:
        r = get_run(key)
        out_csv = r.out / "traces" / "global_ca.csv"
        
        if out_csv.exists() and not force:
            if r.raw is not None and out_csv.stat().st_mtime > r.raw.stat().st_mtime:
                return key, True, "skipped (up-to-date)"
        
        metrics = compute_global_ca(r)
        return key, True, f"ok ({metrics['elapsed_s']:.1f}s, dff [{metrics['global_dff_min']:.1f}, {metrics['global_dff_max']:.1f}]%)"
    except Exception as e:
        return key, False, str(e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="DATE/MOUSE/RUN key (repeatable)")
    ap.add_argument("--all", action="store_true", help="Process all runs")
    ap.add_argument("--force", action="store_true", help="Rebuild even if up-to-date")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs (default 1)")
    args = ap.parse_args(argv)
    
    if not args.run and not args.all:
        ap.error("specify --run or --all")
    
    runs = discover_runs()
    keys = [r.key for r in runs] if args.all else args.run
    
    print(f"global_ca: processing {len(keys)} run(s), jobs={args.jobs}, force={args.force}")
    
    results = []
    if args.jobs == 1:
        for key in keys:
            result = process_run(key, args.force)
            print(f"  {result[0]}: {result[2]}")
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(process_run, k, args.force): k for k in keys}
            for fut in as_completed(futures):
                result = fut.result()
                print(f"  {result[0]}: {result[2]}")
                results.append(result)
    
    ok = sum(1 for _, s, _ in results if s)
    fail = len(results) - ok
    print(f"\nDone: {ok} ok, {fail} failed")
    
    # Notes
    notes_dir = Path(__file__).parent / "notes"
    notes_dir.mkdir(exist_ok=True)
    notes_file = notes_dir / "plots_movies.json"
    notes = json.loads(notes_file.read_text()) if notes_file.exists() else []
    notes.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": f"global_ca.py: {len(keys)} runs ({ok} ok, {fail} failed)",
        "files": [f"scape-auto/{k}/traces/global_ca.csv" for k in keys],
        "verified": ok == len(keys),
        "caveats": None if fail == 0 else f"{fail} runs failed"
    })
    notes_file.write_text(json.dumps(notes, indent=2))
    
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
