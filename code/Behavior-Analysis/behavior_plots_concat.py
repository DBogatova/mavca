#!/usr/bin/env python
"""
Concatenated behavior + Ca plots across multiple runs.

Plots: Ca global ΔF/F, Pupil, Whisker, Accelerometer — all concatenated
with vertical lines marking run boundaries.

Usage:
    python code/Behavior-Analysis/behavior_plots_concat.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import tifffile

# ===== CONFIG =====
DATE = "2026-04-16"
MOUSE = "rbp4_132_phpeb"
RUNS = ["run6", "run7"]
RUN_NUMS = ["006", "007"]  # for filenames

FRAME_RATE = 5  # Hz (imaging)
SKIP_FIRST_SECONDS = 13.0
CROP_START_SECONDS = 13.0  # cut this many seconds from the start of each run
HAS_ACH = False

BASE_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
                 "apical-dendrites-2025/scape-data") / DATE / MOUSE
OUTPUT_PATH = BASE_ROOT / RUNS[0] / "behavior_concat_plot.png"


def load_ca_one_run(run):
    """Compute global Ca ΔF/F from raw stack for one run."""
    base = BASE_ROOT / run
    raw_clean = base / "preprocessed" / "raw_clean.tif"
    raw_orig = base / "raw" / f"runB_{run}_{MOUSE}-reslice-bin.tif"
    raw_path = raw_clean if raw_clean.exists() else raw_orig

    if not raw_path.exists():
        print(f"  [skip] Raw stack not found: {raw_path}")
        return None, None

    store = tifffile.memmap(str(raw_path), mode='r')
    T = store.shape[0]
    skip = int(max(SKIP_FIRST_SECONDS, CROP_START_SECONDS) * FRAME_RATE)

    # Live voxels
    sample = np.asarray(store[:min(100, T)]).astype(np.float32)
    tmean = sample.mean(axis=0)
    live_idx = np.flatnonzero(tmean > np.percentile(tmean, 5))
    del sample, tmean

    # F0
    f0_frames = []
    for t in range(skip, min(skip + 500, T)):
        f0_frames.append(np.asarray(store[t]).astype(np.float32).ravel()[live_idx].mean())
    f0 = np.percentile(f0_frames, 10)
    del f0_frames

    # Global ΔF/F
    ca = np.empty(T, dtype=np.float32)
    for t0 in range(0, T, 50):
        t1 = min(t0 + 50, T)
        frames = np.asarray(store[t0:t1]).astype(np.float32)
        for i in range(frames.shape[0]):
            ca[t0 + i] = (frames[i].ravel()[live_idx].mean() - f0) / (f0 + 1e-6) * 100
    del store

    time = np.arange(T) / FRAME_RATE
    # Crop start
    mask = time >= CROP_START_SECONDS
    return time[mask] - CROP_START_SECONDS, ca[mask]


def load_behavior_one_run(run, run_num):
    """Load pupil + whisker from behavior .mat."""
    base = BASE_ROOT / run
    mat_path = base / "behavior" / f"{MOUSE}_{DATE.replace('20','',1).replace('-','-')}_Run{run_num}_behavior.mat"
    # Try exact name pattern
    if not mat_path.exists():
        # Search for it
        beh_dir = base / "behavior"
        if beh_dir.exists():
            mats = list(beh_dir.glob("*behavior.mat"))
            if mats:
                mat_path = mats[0]

    if not mat_path.exists():
        print(f"  [skip] Behavior mat not found for {run}")
        return None, None, None

    mat = loadmat(mat_path)
    pupil = mat['pupil']['pupil_raw'][0][0].flatten()
    pupil = gaussian_filter1d(pupil, sigma=2)  # light smooth
    whisker = mat['whisker']['whisker_smooth_long'][0][0].flatten()
    time = np.arange(len(pupil)) / 10.0
    mask = time >= CROP_START_SECONDS
    # Smooth whisker for cleaner plot
    whisker_sm = gaussian_filter1d(whisker, sigma=5)  # ~0.5s at 10 Hz
    return time[mask] - CROP_START_SECONDS, pupil[mask], whisker_sm[mask]


def load_accel_one_run(run, run_num):
    """Load accelerometer from CSV, time relative to trigger onset."""
    base = BASE_ROOT / run
    accel_csv = base / "trigger" / f"Run{run_num}_t1_accel.csv"
    if not accel_csv.exists():
        print(f"  [skip] Accel CSV not found for {run}")
        return None, None

    df = pd.read_csv(accel_csv)
    if 'accel_mag' in df.columns:
        accel = df['accel_mag'].values
    else:
        accel = df.iloc[:, 1].values

    # Use aligned_time_s (where 0 = trigger onset)
    if 'aligned_time_s' in df.columns:
        time = df['aligned_time_s'].values
    else:
        time = df['sample'].values / 1000.0

    return time, accel


def crop_accel(time, accel):
    """Crop accel: keep from CROP_START_SECONDS after trigger onset."""
    mask = (time >= CROP_START_SECONDS)
    accel_clean = np.abs(accel[mask])
    accel_clean = gaussian_filter1d(accel_clean, sigma=10)
    return time[mask] - CROP_START_SECONDS, accel_clean


def main():
    print(f"=== Concatenated Behavior: {DATE}/{MOUSE} ===")
    print(f"  Runs: {RUNS}\n")

    all_ca, all_pupil, all_whisker, all_accel = [], [], [], []
    all_ca_t, all_pupil_t, all_whisker_t, all_accel_t = [], [], [], []
    run_boundaries_ca = [0.0]
    run_boundaries_beh = [0.0]
    run_boundaries_acc = [0.0]

    for run, rnum in zip(RUNS, RUN_NUMS):
        print(f"--- {run} ---")

        # Ca
        t_ca, ca = load_ca_one_run(run)
        if t_ca is not None:
            offset = run_boundaries_ca[-1]
            all_ca_t.append(t_ca + offset)
            all_ca.append(ca)
            run_boundaries_ca.append(offset + t_ca[-1])
            # Crop behavior/accel to Ca duration
            ca_dur = t_ca[-1]
        else:
            ca_dur = 120.0  # fallback

        # Behavior
        t_beh, pupil, whisker = load_behavior_one_run(run, rnum)
        if t_beh is not None:
            # Crop to Ca duration
            mask = t_beh <= ca_dur
            offset = run_boundaries_beh[-1]
            all_pupil_t.append(t_beh[mask] + offset)
            # Subtract per-run median to align baselines
            p_seg = pupil[mask]
            if len(RUNS) > 1:
                p_seg = p_seg - np.median(p_seg)
            all_pupil.append(p_seg)
            all_whisker_t.append(t_beh[mask] + offset)
            all_whisker.append(whisker[mask])
            run_boundaries_beh.append(offset + t_beh[mask][-1])

        # Accel
        t_acc, accel = load_accel_one_run(run, rnum)
        if t_acc is not None:
            # Crop: keep only t >= 0 (trigger) and apply start crop
            t_acc, accel = crop_accel(t_acc, accel)
            mask = (t_acc >= 0) & (t_acc <= ca_dur)
            offset = run_boundaries_acc[-1]
            all_accel_t.append(t_acc[mask] + offset)
            all_accel.append(accel[mask])
            run_boundaries_acc.append(offset + t_acc[mask][-1])

    # Concatenate
    panels = []
    if all_ca:
        panels.append(("Ca ΔF/F (%)", np.concatenate(all_ca_t),
                        np.concatenate(all_ca), 'green', run_boundaries_ca))
    if all_pupil:
        _dp = np.concatenate(all_pupil); _tp = np.concatenate(all_pupil_t)
        print(f"DEBUG pupil: len={len(_dp)}, min={_dp.min():.4f}, max={_dp.max():.4f}, t_max={_tp[-1]:.1f}s")
        panels.append(("Pupil Dilation", _tp, _dp, 'blue', run_boundaries_beh))
    if all_whisker:
        panels.append(("Whisker Motion", np.concatenate(all_whisker_t),
                        np.concatenate(all_whisker), 'orange', run_boundaries_beh))
    if all_accel:
        panels.append(("Accelerometer", np.concatenate(all_accel_t),
                        np.concatenate(all_accel), 'purple', run_boundaries_acc))

    if not panels:
        print("No data to plot.")
        return

    # Plot
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 3 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (ylabel, t, data, color, boundaries) in zip(axes, panels):
        ax.plot(t, data, color=color, linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        # Run boundary lines
        for b in boundaries[1:-1]:
            ax.axvline(b, color='red', ls='-', lw=1.5, alpha=0.2)
        # Run labels
        for j, run in enumerate(RUNS):
            if j < len(boundaries) - 1:
                mid = (boundaries[j] + boundaries[j+1]) / 2
                ylim = ax.get_ylim()
                ax.text(mid, ylim[1], run, ha='center', va='top',
                        fontsize=8, color='gray')

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'{MOUSE} — {DATE} — {" + ".join(RUNS)}', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
