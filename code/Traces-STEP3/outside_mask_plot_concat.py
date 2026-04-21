#!/usr/bin/env python3
"""
Concatenated global Ca analysis across multiple runs.

Same analysis as outside_mask_plot.py but concatenates runs 
with per-run F0 baseline and run boundary markers.

Produces:
  - globalCa_comparison_concat.png (inside vs far vs global + M4-style avg)
  - globalCa_rings_concat.png (distance rings)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import (gaussian_filter1d, distance_transform_edt,
                           binary_erosion, binary_dilation)
from skimage.morphology import ball
import tifffile

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUNS = ["run6", "run7"]
MASK_RUN = "run7"  # masks from this run

FRAME_RATE = 5.0
CHUNK_T = 118
Y_CROP = 3
CROP_START_SECONDS = 11.0
SKIP_FIRST_SECONDS = 11.0

VOXEL_ZYX = (3.9, 1.0, 1.2)
SMOOTH_SIGMA = 1.5
ARTIFACT_Z = -0.5
EPS = 1e-8
F0_NFRAMES = 500
ROLLING_BASELINE = True
ROLLING_WINDOW_SEC = 60.0

USE_DENOM_FLOOR = True
DENOM_FLOOR_PCT = 5.0

RING_UM_EDGES = (5.0, 15.0, 30.0)  

SAVE_FIG = True
SHOW_FIG = True
FIG_DPI = 200

# ===== PATHS =====
BASE_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
                 "apical-dendrites-2025/scape-data") / DATE / MOUSE
MASK_FOLDER = BASE_ROOT / MASK_RUN / "labelmaps_curated_dynamic"
OUT_FOLDER = BASE_ROOT / MASK_RUN / "outside_mask_dynamics"
OUT_FOLDER.mkdir(exist_ok=True)


def get_raw_path(run):
    base = BASE_ROOT / run
    clean = base / "preprocessed" / "raw_clean.tif"
    if clean.exists():
        return clean
    return base / "raw" / f"runA_{run}_{MOUSE}-reslice-bin.tif"


def smooth(x):
    return gaussian_filter1d(x, sigma=SMOOTH_SIGMA) if SMOOTH_SIGMA > 0 else x


def rolling_baseline_correct(trace, window_frames):
    """Subtract rolling 10th percentile baseline from a 1D trace."""
    T = len(trace)
    if window_frames >= T:
        baseline = np.full(T, np.percentile(trace, 10))
    else:
        half = window_frames // 2
        baseline = np.empty(T, dtype=np.float32)
        for t in range(T):
            t0 = max(0, t - half)
            t1 = min(T, t + half)
            baseline[t] = np.percentile(trace[t0:t1], 10)
    return trace - baseline


def load_masks_and_build(Z, Y, X):
    """Load masks, build union, core/shell indices, ring masks."""
    mask_paths = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks in {MASK_FOLDER}")

    # Union mask
    union = np.zeros((Z, Y, X), dtype=bool)
    rois = []
    for p in mask_paths:
        m = tifffile.imread(p).astype(bool)
        mz, my, mx = m.shape
        if my > Y: m = m[:, :Y, :X]
        elif my < Y: m = np.pad(m, ((0,0),(0,Y-my),(0,0)), mode='constant')
        if not m.any():
            continue
        union |= m
        # Core/shell for M4-style
        core = binary_erosion(m, structure=ball(1))
        if not core.any():
            core = m.copy()
        inner = binary_dilation(m, structure=ball(2))
        outer = binary_dilation(m, structure=ball(3))
        shell = outer & ~inner
        rois.append({
            "core": np.flatnonzero(core.ravel()),
            "shell": np.flatnonzero(shell.ravel()) if shell.any() else np.array([], dtype=np.int64),
        })
        del m, core, shell

    # Flat masks
    inside_flat = union.ravel()
    outside_flat = (~union).ravel()

    # Far >15µm mask
    dist_um = distance_transform_edt(~union, sampling=VOXEL_ZYX).astype(np.float32)
    far_flat = ((dist_um > 15.0) & ~union).ravel()

    # Ring masks
    rings = {}
    edges = sorted(RING_UM_EDGES)
    lo = 0.0
    for hi in edges:
        m = (dist_um > lo) & (dist_um <= hi) & ~union
        rings[f"ring_{lo:g}_{hi:g}um"] = m.ravel()
        lo = hi
    rings[f"far_gt_{edges[-1]:g}um"] = (dist_um > edges[-1] ) & ~union
    rings[f"far_gt_{edges[-1]:g}um"] = rings[f"far_gt_{edges[-1]:g}um"].ravel()

    print(f"  {len(mask_paths)} masks, union={union.sum():,} vox, "
          f"far>15µm={far_flat.sum():,} vox, {len(rois)} ROIs")

    return union, inside_flat, outside_flat, far_flat, rings, rois


def process_one_run(run, inside_flat, outside_flat, far_flat, rings, rois, ZYX):
    """Process one run: compute F0, extract all traces. Returns dict of arrays."""
    raw_path = get_raw_path(run)
    if not raw_path.exists():
        print(f"  [skip] {raw_path}")
        return None

    print(f"\n--- {run}: {raw_path.name} ---")
    store = tifffile.memmap(str(raw_path), mode='r')
    T_raw = store.shape[0]
    Z, Y, X = ZYX

    # Y crop
    skip = int(max(SKIP_FIRST_SECONDS, CROP_START_SECONDS) * FRAME_RATE)
    crop = int(CROP_START_SECONDS * FRAME_RATE)

    # F0 baseline (10th percentile of early frames after skip)
    f0_chunks = []
    for t in range(skip, min(skip + F0_NFRAMES, T_raw)):
        frame = np.asarray(store[t]).astype(np.float32)
        if Y_CROP > 0:
            frame = frame[:, :-Y_CROP, :]
        f0_chunks.append(frame)
    f0 = np.percentile(np.stack(f0_chunks), 10, axis=0)
    del f0_chunks

    # Denom floor
    alpha = float(np.percentile(f0.ravel(), DENOM_FLOOR_PCT)) if USE_DENOM_FLOOR else 0.0

    T_out = T_raw - crop
    inside_tr = np.empty(T_out, np.float32)
    outside_tr = np.empty(T_out, np.float32)
    far_tr = np.empty(T_out, np.float32)
    global_tr = np.empty(T_out, np.float32)
    ring_trs = {name: np.empty(T_out, np.float32) for name in rings}
    m4_traces = [np.empty(T_out, np.float32) for _ in rois]

    out_idx = 0
    for t0 in range(crop, T_raw, CHUNK_T):
        t1 = min(t0 + CHUNK_T, T_raw)
        chunk = np.asarray(store[t0:t1]).astype(np.float32)
        if Y_CROP > 0:
            chunk = chunk[:, :, :-Y_CROP, :]

        for ti in range(chunk.shape[0]):
            dff = (chunk[ti] - f0) / (f0 + EPS + alpha)
            if ARTIFACT_Z is not None:
                dff[dff < ARTIFACT_Z] = 0.0
            flat = dff.ravel()

            global_tr[out_idx] = flat.mean()
            inside_tr[out_idx] = flat[inside_flat].mean()
            outside_tr[out_idx] = flat[outside_flat].mean()
            if far_flat.sum() > 0:
                far_tr[out_idx] = flat[far_flat].mean()

            for name, rmask in rings.items():
                if rmask.sum() > 0:
                    ring_trs[name][out_idx] = flat[rmask].mean()

            for mi, roi in enumerate(rois):
                cv = flat[roi["core"]].mean()
                sv = flat[roi["shell"]].mean() if roi["shell"].size > 0 else 0.0
                m4_traces[mi][out_idx] = cv - sv

            out_idx += 1

        print(f"    frames {t0}..{t1-1}")

    del store
    # Trim if T_raw wasn't exact multiple
    inside_tr = inside_tr[:out_idx]
    outside_tr = outside_tr[:out_idx]
    far_tr = far_tr[:out_idx]
    global_tr = global_tr[:out_idx]
    ring_trs = {k: v[:out_idx] for k, v in ring_trs.items()}
    m4_traces = [t[:out_idx] for t in m4_traces]

    return {
        "T": out_idx,
        "inside": inside_tr, "outside": outside_tr,
        "far": far_tr, "global": global_tr,
        "rings": ring_trs,
        "m4_avg": np.mean(m4_traces, axis=0),
    }


def main():
    print(f"=== Concatenated Outside-Mask Analysis ===")
    print(f"  {DATE}/{MOUSE}, runs: {RUNS}, masks from {MASK_RUN}\n")

    # Peek at dimensions from first run
    raw0 = get_raw_path(RUNS[0])
    tf = tifffile.TiffFile(str(raw0))
    shape = tf.series[0].shape
    tf.close()
    T0, Z, Y, X = shape
    if Y_CROP > 0:
        Y -= Y_CROP
    print(f"  Spatial: Z={Z}, Y={Y}, X={X}")

    # Load masks and build spatial structures
    print("Loading masks...")
    union, inside_flat, outside_flat, far_flat, rings, rois = \
        load_masks_and_build(Z, Y, X)

    # Process each run
    all_inside, all_outside, all_far, all_global = [], [], [], []
    all_m4 = []
    all_rings = {name: [] for name in rings}
    run_boundaries = [0.0]

    for run in RUNS:
        result = process_one_run(run, inside_flat, outside_flat, far_flat,
                                  rings, rois, (Z, Y, X))
        if result is None:
            continue
        T_run = result["T"]
        dur = T_run / FRAME_RATE

        all_inside.append(result["inside"])
        all_outside.append(result["outside"])
        all_far.append(result["far"])
        all_global.append(result["global"])
        all_m4.append(result["m4_avg"])
        for name in rings:
            all_rings[name].append(result["rings"][name])
        run_boundaries.append(run_boundaries[-1] + dur)

    # Concatenate
    cat = lambda arrs: np.concatenate(arrs)
    roll_win = int(ROLLING_WINDOW_SEC * FRAME_RATE) if ROLLING_BASELINE else 0

    def process_trace(arrs):
        tr = cat(arrs)
        if ROLLING_BASELINE:
            tr = rolling_baseline_correct(tr, roll_win)
        return smooth(tr) * 100

    inside_s = process_trace(all_inside)
    outside_s = process_trace(all_outside)
    far_s = process_trace(all_far)
    global_s = process_trace(all_global)
    m4_raw = cat(all_m4)
    if ROLLING_BASELINE:
        m4_raw = rolling_baseline_correct(m4_raw, roll_win)
    m4_s = smooth(m4_raw) * 100
    total_T = len(inside_s)
    time_s = np.arange(total_T) / FRAME_RATE

    # Motion-corrected M4 average (same interpolation as save_traces_m4)
    m4_raw = cat(all_m4)
    diff = np.diff(m4_raw, prepend=m4_raw[0])
    bad = (m4_raw < ARTIFACT_Z) | (diff < -0.3)
    bad_dilated = bad.copy()
    bad_dilated[1:] |= bad[:-1]
    bad_dilated[:-1] |= bad[1:]
    m4_corrected = m4_raw.copy()
    good_idx = np.where(~bad_dilated)[0]
    bad_idx = np.where(bad_dilated)[0]
    if len(good_idx) > 2 and len(bad_idx) > 0:
        m4_corrected[bad_idx] = np.interp(bad_idx, good_idx, m4_raw[good_idx])
    m4_corr_s = smooth(m4_corrected) * 100
    n_interp = bad_dilated.sum()
    print(f"  Motion correction: interpolated {n_interp} frames ({100*n_interp/total_T:.1f}%)")

    # === Comparison plot ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(time_s, inside_s + 8, lw=1.5, label="Inside masks (+8%)", color='tab:blue')
    axes[0].plot(time_s, far_s + 4, lw=1.5, alpha=0.85, label="Outside >15µm (+4%)", color='tab:orange')
    axes[0].plot(time_s, global_s, lw=1.5, alpha=0.85, label="Global average", color='tab:green')
    for b in run_boundaries[1:-1]:
        axes[0].axvline(b, color='red', ls='-', lw=1.5, alpha=0.4)
    for j, run in enumerate(RUNS):
        if j < len(run_boundaries) - 1:
            mid = (run_boundaries[j] + run_boundaries[j+1]) / 2
            axes[0].text(mid, -0.08, run, ha='center', fontsize=9, color='gray',
                         transform=axes[0].get_xaxis_transform())
    axes[0].set_ylabel("ΔF/F (%)")
    axes[0].set_title(f"{DATE} | {MOUSE} | {' + '.join(RUNS)}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(time_s, m4_s, lw=1.2, color='magenta', label="Avg core−shell")
    axes[1].axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    for b in run_boundaries[1:-1]:
        axes[1].axvline(b, color='red', ls='-', lw=1.5, alpha=0.4)
    axes[1].set_ylabel("ΔF/F (%)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out1 = OUT_FOLDER / "globalCa_comparison_concat.png"
    if SAVE_FIG:
        plt.savefig(out1, dpi=FIG_DPI)
        print(f"✅ {out1}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

    # === Rings plot (with offset, including inside masks) ===
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ring_offset = 4.0
    # Inside masks first, then rings by distance
    all_ring_plots = [("Inside masks", inside_s)]
    for name in rings:
        if all_rings[name]:
            all_ring_plots.append((name, process_trace(all_rings[name])))
    for i, (label, tr) in enumerate(all_ring_plots):
        ax2.plot(time_s, tr + i * ring_offset, lw=1.5, label=label)
    for b in run_boundaries[1:-1]:
        ax2.axvline(b, color='red', ls='-', lw=1.5, alpha=0.4)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("ΔF/F (%)")
    ax2.set_title(f"{DATE} | {MOUSE} | {' + '.join(RUNS)}  — distance rings")
    ax2.legend(ncols=2, fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()

    out2 = OUT_FOLDER / "globalCa_rings_concat.png"
    if SAVE_FIG:
        plt.savefig(out2, dpi=FIG_DPI)
        print(f"✅ {out2}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
