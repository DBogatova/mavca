#!/usr/bin/env python3
"""
Render behavior traces as a movie with a moving vertical line.
Stacked panels: Accelerometer, Pupil, Global Ca.
Matches imaging frame rate for sync with mouse video.

Usage:
    python code/Visual-STEP4/accel_trace_movie.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import tifffile
import imageio

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUN = "run8"
RUN_NUM = "008"

CROP_START_SECONDS = 12.0
IMAGING_FRAME_RATE = 5
SKIP_FIRST_SECONDS = 12.0

FIG_W, FIG_H = 19.2, 5  # wide to match dual view (~1920px at 100 DPI)
DPI = 1000

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
ACCEL_CSV = BASE / "trigger" / f"Run{RUN_NUM}_t1_accel.csv"
OUTPUT = BASE / "overlays" / "behavior_trace_movie.mp4"


def load_accel():
    df = pd.read_csv(ACCEL_CSV)
    accel = df['accel_mag'].values if 'accel_mag' in df.columns else df.iloc[:, 1].values
    t = df['aligned_time_s'].values if 'aligned_time_s' in df.columns else df['sample'].values / 1000.0
    mask = t >= CROP_START_SECONDS
    a = np.abs(accel[mask])
    a = gaussian_filter1d(a, sigma=10)
    return t[mask] - CROP_START_SECONDS, a


def load_pupil():
    beh_dir = BASE / "behavior"
    mats = list(beh_dir.glob("*behavior.mat"))
    if not mats:
        return None, None
    mat = loadmat(str(mats[0]))
    p = mat['pupil']['pupil_raw'][0][0].flatten()
    p = gaussian_filter1d(p, sigma=2)
    t = np.arange(len(p)) / 10.0
    mask = t >= CROP_START_SECONDS
    return t[mask] - CROP_START_SECONDS, p[mask]


def load_global_ca():
    raw_clean = BASE / "preprocessed" / "raw_clean.tif"
    raw_orig = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin.tif"
    raw_path = raw_clean if raw_clean.exists() else raw_orig
    if not raw_path.exists():
        return None, None

    store = tifffile.memmap(str(raw_path), mode='r')
    T_raw = store.shape[0]
    skip = int(max(SKIP_FIRST_SECONDS, CROP_START_SECONDS) * IMAGING_FRAME_RATE)

    sample = np.asarray(store[:min(100, T_raw)]).astype(np.float32)
    tmean = sample.mean(axis=0)
    live_idx = np.flatnonzero(tmean > np.percentile(tmean, 5))
    del sample, tmean

    f0_vals = []
    for t in range(skip, min(skip + 500, T_raw)):
        f0_vals.append(np.asarray(store[t]).astype(np.float32).ravel()[live_idx].mean())
    f0 = np.percentile(f0_vals, 10)
    del f0_vals

    crop = int(CROP_START_SECONDS * IMAGING_FRAME_RATE)
    T_out = T_raw - crop
    ca = np.empty(T_out, dtype=np.float32)
    for t0 in range(crop, T_raw, 50):
        t1 = min(t0 + 50, T_raw)
        frames = np.asarray(store[t0:t1]).astype(np.float32)
        for i in range(frames.shape[0]):
            ca[t0 - crop + i] = (frames[i].ravel()[live_idx].mean() - f0) / (f0 + 1e-6) * 100
    del store

    time = np.arange(T_out) / IMAGING_FRAME_RATE
    return time, gaussian_filter1d(ca, sigma=1.0)


def main():
    print(f"=== Behavior Trace Movie: {DATE}/{MOUSE}/{RUN} ===\n")

    # Load all signals
    t_acc, accel = load_accel()
    t_pup, pupil = load_pupil()
    t_ca, ca = load_global_ca()

    # Get imaging duration
    raw_clean = BASE / "preprocessed" / "raw_clean.tif"
    raw_orig = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin.tif"
    raw_path = raw_clean if raw_clean.exists() else raw_orig
    tf = tifffile.TiffFile(str(raw_path))
    T_raw = tf.series[0].shape[0]
    tf.close()
    duration = (T_raw / IMAGING_FRAME_RATE) - CROP_START_SECONDS
    n_frames = int(duration * IMAGING_FRAME_RATE)
    print(f"  Duration: {duration:.1f}s, {n_frames} frames")

    # Build panels
    panels = []
    if t_acc is not None:
        mask = t_acc <= duration
        panels.append(("Accel", t_acc[mask], accel[mask], '#aa77ff'))
    if t_pup is not None:
        mask = t_pup <= duration
        panels.append(("Pupil", t_pup[mask], pupil[mask], '#4488ff'))
    if t_ca is not None:
        mask = t_ca <= duration
        panels.append(("Global Ca ΔF/F (%)", t_ca[mask], ca[mask], '#44cc44'))

    if not panels:
        print("No data to plot.")
        return

    # Precompute y-limits
    ylims = []
    for name, t, d, color in panels:
        ymin, ymax = d.min(), d.max()
        margin = (ymax - ymin) * 0.1 + 1e-6
        ylims.append((ymin - margin, ymax + margin))

    OUTPUT.parent.mkdir(exist_ok=True)
    writer = imageio.get_writer(str(OUTPUT), fps=IMAGING_FRAME_RATE,
                                codec="libx264", quality=8,
                                pixelformat="yuv420p", format="FFMPEG")

    print("Rendering...")
    for i in range(n_frames):
        current_t = i / IMAGING_FRAME_RATE

        fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_W, FIG_H),
                                  sharex=True)
        if len(panels) == 1:
            axes = [axes]
        fig.patch.set_facecolor('black')

        for ax, (name, t, d, color), (yl, yh) in zip(axes, panels, ylims):
            ax.set_facecolor('black')
            ax.plot(t, d, color=color, lw=1.0)
            ax.axvline(current_t, color='red', lw=2, alpha=0.8)
            ax.set_xlim(0, duration)
            ax.set_ylim(yl, yh)
            ax.set_ylabel(name, fontsize=9, color='white')
            ax.tick_params(labelsize=7, colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.15, color='white')

        axes[-1].set_xlabel("Time (s)", fontsize=9, color='white')
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(buf)
        plt.close(fig)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_frames}")

    writer.close()
    print(f"\n✅ Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
