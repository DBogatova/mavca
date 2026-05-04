#!/usr/bin/env python3
"""
Stacked ΔF/F traces (from M4 combo) + accelerometer + pupil below.

Usage:
    python code/Traces-STEP3/combo_with_behavior.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUN = "run7"
RUN_NUM = "007"

FRAME_RATE = 5.0
CROP_START_SECONDS = 12.0
TRACE_OFFSET = 5.0  # vertical offset between traces (%) — smaller = spikes look bigger

# Select specific cells (None = all, or list like ["dend_000", "dend_003", ...])
SELECTED_NAMES = None
#["dend_005", "dend_009", "dend_015","dend_020","dend_025", "dend_027", "dend_034", "dend_045", "dend_048", "dend_049","dend_057", "dend_058", "dend_059" , "dend_060",
               # "dend_061", "dend_062", "dend_063"] 
TOP_N_BY_SNR = 12  # if SELECTED_NAMES is None, pick top N by SNR
# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
TRACE_CSV = BASE / "traces" / "dff_traces_curated_bgsub.csv"
OUTPUT = BASE / "traces" / "combo_traces_with_behavior.png"


def main():
    # Load traces
    df = pd.read_csv(TRACE_CSV)
    dend_cols = [c for c in df.columns if c.startswith("dend_")]
    traces = df[dend_cols].values
    crop = int(CROP_START_SECONDS * FRAME_RATE)
    traces = traces[crop:]
    T = traces.shape[0]
    N = len(dend_cols)
    time_s = np.arange(T) / FRAME_RATE

    # Select traces
    if SELECTED_NAMES:
        sel_idx = [i for i, n in enumerate(dend_cols) if n in SELECTED_NAMES]
        traces = traces[:, sel_idx]
        dend_cols = [dend_cols[i] for i in sel_idx]
    elif TOP_N_BY_SNR:
        # Rank by SNR and pick top N
        snrs = []
        for i in range(traces.shape[1]):
            tr = traces[:, i]
            peak = np.percentile(tr, 95)
            noise = np.median(np.abs(tr - np.median(tr))) + 1e-6
            snrs.append(peak / noise)
        top_idx = np.argsort(snrs)[::-1][:TOP_N_BY_SNR]
        traces = traces[:, top_idx]
        dend_cols = [dend_cols[i] for i in top_idx]

    N = traces.shape[1]
    print(f"  Plotting {N} traces")

    # Load pupil
    t_pup, pupil = None, None
    beh_dir = BASE / "behavior"
    if beh_dir.exists():
        mats = list(beh_dir.glob("*behavior.mat"))
        if mats:
            mat = loadmat(str(mats[0]))
            p = mat['pupil']['pupil_raw'][0][0].flatten()
            pupil = gaussian_filter1d(p, sigma=2)
            t_pup = np.arange(len(pupil)) / 10.0
            mask = (t_pup >= CROP_START_SECONDS) & (t_pup <= CROP_START_SECONDS + time_s[-1])
            t_pup, pupil = t_pup[mask] - CROP_START_SECONDS, pupil[mask]

    # Load accel
    t_acc, accel = None, None
    accel_csv = BASE / "trigger" / f"Run{RUN_NUM}_t1_accel.csv"
    if accel_csv.exists():
        adf = pd.read_csv(accel_csv)
        a = adf['accel_mag'].values if 'accel_mag' in adf.columns else adf.iloc[:, 1].values
        t = adf['aligned_time_s'].values if 'aligned_time_s' in adf.columns else adf['sample'].values / 1000.0
        mask = (t >= CROP_START_SECONDS) & (t <= CROP_START_SECONDS + time_s[-1])
        accel = np.abs(a[mask])
        accel = gaussian_filter1d(accel, sigma=10)
        t_acc = t[mask] - CROP_START_SECONDS

    # Plot
    n_panels = 1
    if t_acc is not None: n_panels += 1
    if t_pup is not None: n_panels += 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 8 + 3 * (n_panels - 1)),
                              sharex=True,
                              gridspec_kw={"height_ratios": [4] + [1] * (n_panels - 1)})
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # Stacked traces
    ax = axes[ax_idx]
    for i, name in enumerate(dend_cols):
        tr = traces[:, i]
        ax.plot(time_s, tr + i * TRACE_OFFSET,
                color=cm.turbo(i / max(1, N)), lw=0.8)
        cell_label = f"Cell {name.split('_')[1]}"
        ax.text(time_s[-1] + 1, i * TRACE_OFFSET, cell_label, va='center', fontsize=7)

    # Scale bar
    sx = time_s[-1] - 7
    sy = TRACE_OFFSET * (N - 0.5)
    ax.plot([sx, sx], [sy, sy + 5], color='k', lw=2)
    ax.text(sx + 1, sy + 0.5, "5%", va='center', fontsize=9)

    ax.set_xlim(0, time_s[-1])
    ax.set_ylim(-TRACE_OFFSET, N * TRACE_OFFSET + 5)
    ax.set_ylabel("ΔF/F (%) + offset")
    ax.set_yticks([])
    ax.set_title(f"{DATE} | {MOUSE} | {RUN} — Stacked ΔF/F Traces")
    ax.grid(alpha=0.2)
    ax_idx += 1

    # Accel
    if t_acc is not None:
        axes[ax_idx].plot(t_acc, accel, color='purple', lw=0.8)
        axes[ax_idx].set_ylabel("Accelerometer")
        axes[ax_idx].grid(alpha=0.3)
        ax_idx += 1

    # Pupil
    if t_pup is not None:
        axes[ax_idx].plot(t_pup, pupil, color='blue', lw=1.0)
        axes[ax_idx].set_ylabel("Pupil")
        axes[ax_idx].grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT}")
    plt.show()


if __name__ == "__main__":
    main()
