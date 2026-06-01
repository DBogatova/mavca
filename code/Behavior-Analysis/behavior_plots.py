#!/usr/bin/env python
"""
Combined behavior and calcium/ACh analysis plots (single run)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import tifffile

# ===== CONFIG =====
DATE = "2026-05-12"
MOUSE = "rbp4_132_phpeb"
RUN = "run5"

FRAME_RATE = 5  # Hz
SKIP_FIRST_SECONDS = 12.0
CROP_START_SECONDS = 12.0  # cut first N seconds from all signals
HAS_ACH = False

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
BEHAVIOR_MAT = BASE / "behavior" / "rbp4_132_phpeb_26-05-12_Run005_behavior.mat"
OUTPUT_PATH = BASE / "behavior_combined_plot.png"


def load_calcium_ach_data():
    """Compute global Ca ΔF/F from raw stack (mean of all non-dead voxels)."""
    raw_clean = BASE / "preprocessed" / "raw_clean.tif"
    raw_orig = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin.tif"
    raw_path = raw_clean if raw_clean.exists() else raw_orig

    if not raw_path.exists():
        print(f"Raw stack not found: {raw_path}")
        return None, None, None

    print(f"Loading raw stack: {raw_path.name}")
    store = tifffile.memmap(str(raw_path), mode='r')
    T = store.shape[0]
    skip = int(max(SKIP_FIRST_SECONDS, CROP_START_SECONDS) * FRAME_RATE)
    print(f"  Shape: {store.shape}, skipping first {skip} frames for F0")

    # Find dead voxels
    print("  Finding live voxels...")
    sample = np.asarray(store[:min(100, T)]).astype(np.float32)
    tmean = sample.mean(axis=0)
    live_idx = np.flatnonzero(tmean > np.percentile(tmean, 5))
    print(f"  Live voxels: {live_idx.size:,} / {live_mask.size:,}" if False else
          f"  Live voxels: {live_idx.size:,} / {tmean.size:,}")
    del sample, tmean

    # F0 baseline
    print("  Computing F0 baseline...")
    f0_frames = []
    for t in range(skip, min(skip + 500, T)):
        f0_frames.append(np.asarray(store[t]).astype(np.float32).ravel()[live_idx].mean())
    f0 = np.percentile(f0_frames, 10)
    print(f"  F0 = {f0:.1f}")
    del f0_frames

    # Global ΔF/F
    print("  Computing global ΔF/F...")
    ca_dff = np.empty(T, dtype=np.float32)
    for t0 in range(0, T, 50):
        t1 = min(t0 + 50, T)
        frames = np.asarray(store[t0:t1]).astype(np.float32)
        for i in range(frames.shape[0]):
            ca_dff[t0 + i] = (frames[i].ravel()[live_idx].mean() - f0) / (f0 + 1e-6) * 100
    del store

    time = np.arange(T) / FRAME_RATE
    # Crop
    mask = time >= CROP_START_SECONDS
    time, ca_dff = time[mask] - CROP_START_SECONDS, ca_dff[mask]
    print(f"  Global Ca ΔF/F: min={ca_dff.min():.2f}%, max={ca_dff.max():.2f}%")
    return time, None, ca_dff


def load_behavior_data():
    """Load pupil and whisker from behavior MAT file.
    Returns values normalized to [0, 1] using the max from the first 14s as reference."""
    if not BEHAVIOR_MAT.exists():
        print(f"Behavior MAT not found: {BEHAVIOR_MAT}")
        return None, None, None

    mat_data = loadmat(BEHAVIOR_MAT)
    pupil = mat_data['pupil']['pupil_raw'][0][0].flatten()
    pupil = gaussian_filter1d(pupil, sigma=2)
    whisker = mat_data['whisker']['whisker_smooth_long'][0][0].flatten()
    whisker = gaussian_filter1d(whisker, sigma=3)

    n = len(pupil)
    print(f"  Behavior: {n} samples at 10 Hz = {n/10:.1f}s")

    time = np.arange(n) / 10.0

    # Compute Basler-to-SCAPE offset from trigger CSV
    trigger_csvs = list((BASE / "trigger").glob("*_trigger.csv"))
    if trigger_csvs:
        trig = pd.read_csv(trigger_csvs[0])
        basler_start = trig.loc[trig['baslerExposureTrigger'].diff() == 1, 'time_s'].iloc[0]
        andor_start = trig.loc[trig['AndorXylaTrigger'].diff() == 1, 'time_s'].iloc[0]
        offset = andor_start - basler_start
        print(f"  Basler→SCAPE offset = {offset:.3f}s")
    else:
        offset = 0.0
        print("  No trigger CSV, assuming no Basler offset")

    total_crop = offset + CROP_START_SECONDS

    # Normalize to max from first 14s (full dynamic range reference)
    pre_mask = (time >= offset) & (time < total_crop)
    pupil_max = np.max(pupil[pre_mask]) if pre_mask.any() else np.max(pupil)
    whisker_max = np.max(np.abs(whisker[pre_mask])) if pre_mask.any() else np.max(np.abs(whisker))
    pupil_norm = pupil / (pupil_max + 1e-6)
    whisker_norm = whisker / (whisker_max + 1e-6)
    print(f"  Pupil max (pre-crop): {pupil_max:.1f}, post-crop range: {pupil_norm[time >= total_crop].min():.2f}-{pupil_norm[time >= total_crop].max():.2f}")
    print(f"  Whisker max (pre-crop): {whisker_max:.1f}, post-crop range: {whisker_norm[time >= total_crop].min():.2f}-{whisker_norm[time >= total_crop].max():.2f}")

    mask = time >= total_crop
    return time[mask] - total_crop, pupil_norm[mask], whisker_norm[mask]


def load_accelerometer_data():
    """Load accelerometer, normalized to max from first 14s."""
    run_num = RUN.replace("run", "").zfill(3)
    accel_csv = BASE / "trigger" / f"Run{run_num}_t1_accel.csv"

    if not accel_csv.exists():
        print(f"Accel CSV not found: {accel_csv}")
        return None, None, 0

    df = pd.read_csv(accel_csv)
    print(f"Accel columns: {list(df.columns)}")

    accel = df['accel_mag'].values if 'accel_mag' in df.columns else df.iloc[:, 1].values
    time = df['aligned_time_s'].values if 'aligned_time_s' in df.columns else df['sample'].values / 1000.0

    accel_clean = np.abs(accel)
    accel_clean = gaussian_filter1d(accel_clean, sigma=10)

    mask = time >= CROP_START_SECONDS
    return time[mask] - CROP_START_SECONDS, accel_clean[mask], 0


def plot_combined_signals():
    """Create combined plot of all signals."""
    time_ca, ach_dff, ca_dff = load_calcium_ach_data()
    time_behavior, pupil, whisker = load_behavior_data()
    time_accel, accel, _ = load_accelerometer_data()

    panels = []
    if time_ca is not None and HAS_ACH and ach_dff is not None:
        panels.append(("ACh ΔF/F (%)", time_ca, ach_dff, 'red'))
    if time_ca is not None:
        panels.append(("Ca ΔF/F (%)", time_ca, ca_dff, 'green'))
    if time_behavior is not None and pupil is not None:
        panels.append(("Pupil Dilation", time_behavior, pupil, 'blue'))
    if time_behavior is not None and whisker is not None:
        panels.append(("Whisker Motion", time_behavior, whisker, 'orange'))
    if time_accel is not None and accel is not None:
        panels.append(("Accelerometer", time_accel, accel, 'purple'))

    # Crop all to Ca duration
    if time_ca is not None:
        t_end = time_ca[-1]
        cropped = []
        for ylabel, t, data, color in panels:
            m = (t >= 0) & (t <= t_end)
            cropped.append((ylabel, t[m], data[m], color))
        panels = cropped

    if not panels:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (ylabel, t, data, color) in zip(axes, panels):
        ax.plot(t, data, color=color, linewidth=1.0)
        if ylabel == "Accelerometer":
            ax.set_ylim(0, 0.25)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'{MOUSE} — {DATE} — {RUN}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved: {OUTPUT_PATH}")


def main():
    print(f"Creating combined behavior plot for {MOUSE}-{DATE}-{RUN}")
    plot_combined_signals()

if __name__ == "__main__":
    main()
