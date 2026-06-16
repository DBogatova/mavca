#!/usr/bin/env python3
"""
Compact 2-panel figure: individual dendrite spikes + accelerometer only.

A vertically-compact variant of all_spikes_plot.py's top section — just the
spike raster (each dendrite's supra-threshold ΔF/F overlaid) and the
accelerometer. Saved as a vector PDF (Arial).

Usage:
    python code/Traces-STEP3/spikes_accel_compact.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d

# Arial, embedded as editable TrueType in the PDF
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUN = "run8"
RUN_NUM = "008"
FRAME_RATE = 5.0          # this mouse acquired at 5 Hz
CROP_START_SECONDS = 12.0
DFF_THRESHOLD = 0.3       # spike-raster threshold (matches all_spikes_plot.py)

# Compact figure geometry
FIG_W = 14.0
FIG_H = 3.2               # vertically compact (2 panels)
HEIGHT_RATIOS = [2.5, 1]  # spikes : accel

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
OUT_PDF = BASE / "traces" / "spikes_accel_compact.pdf"


def load_spikes():
    df = pd.read_csv(BASE / "traces" / "dff_traces_curated_bgsub.csv")
    df = df.iloc[int(CROP_START_SECONDS * FRAME_RATE):]   # drop warmup
    dend = [c for c in df.columns if c.startswith("dend_")]
    traces = df[dend].values
    t = np.arange(traces.shape[0]) / FRAME_RATE
    print(f"  spikes: {len(dend)} dendrites, {traces.shape[0]} frames ({t[-1]:.1f}s)")
    return t, traces


def load_accel():
    df = pd.read_csv(BASE / "trigger" / f"Run{RUN_NUM}_t1_accel.csv")
    accel = df["accel_mag"].values if "accel_mag" in df.columns else df.iloc[:, 1].values
    t = df["aligned_time_s"].values if "aligned_time_s" in df.columns else df["sample"].values / 1000.0
    m = t >= CROP_START_SECONDS
    a = gaussian_filter1d(np.abs(accel[m]), sigma=10)
    return t[m] - CROP_START_SECONDS, a


def main():
    print(f"=== Compact spikes + accel: {DATE}/{MOUSE}/{RUN} ===")
    t_sp, traces = load_spikes()
    t_acc, accel = load_accel()
    N = traces.shape[1]

    # Clip both to the shortest common duration
    dur = min(t_sp[-1], t_acc[-1])
    ms = t_sp <= dur
    ma = t_acc <= dur
    t_sp, traces = t_sp[ms], traces[ms]
    t_acc, accel = t_acc[ma], accel[ma]

    fig, (ax_sp, ax_ac) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H), sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})

    # Spike raster: each dendrite's supra-threshold ΔF/F overlaid
    for i in range(N):
        tr = traces[:, i].copy()
        tr[tr < DFF_THRESHOLD] = np.nan
        ax_sp.plot(t_sp, tr, lw=0.7, alpha=0.8, color=cm.turbo(i / max(1, N)))
    ax_sp.set_ylabel("ΔF/F (%)")
    ax_sp.set_title(f"{DATE} | {MOUSE} | {RUN} — Dendrite spikes (>{DFF_THRESHOLD})",
                    fontsize=10)
    ax_sp.grid(alpha=0.3)

    # Accelerometer
    ax_ac.plot(t_acc, accel, color="purple", lw=0.8)
    ax_ac.set_ylim(0, 0.25)
    ax_ac.set_ylabel("Accel")
    ax_ac.grid(alpha=0.3)
    ax_ac.set_xlabel("Time (s)")
    ax_ac.set_xlim(0, dur)

    plt.tight_layout()
    OUT_PDF.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved vector PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()
