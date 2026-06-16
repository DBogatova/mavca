#!/usr/bin/env python3
"""
All dendrite spikes + behavior (with ACh), single run, vector PDF.

Panels (top → bottom):
    spikes raster → accelerometer → global Ca ΔF/F → global ACh ΔF/F → pupil
(Whisker omitted by request; ACh inserted directly below global Ca.)

Global Ca and ACh (bleach-corrected + baseline re-zeroed), pupil, and
accelerometer are loaded via behavior_plots.py, so they match
behavior_combined_plot.pdf exactly. Spikes come from the curated per-dendrite
ΔF/F traces.

Usage:
    python code/Traces-STEP3/all_spikes_vs_behavior_ach.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Reuse behavior_plots loaders + config (Ca, ACh, pupil, accel) for consistency
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Behavior-Analysis"))
import behavior_plots as bp  # noqa: E402

# Arial, embedded as editable TrueType in the PDF
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42

# ===== CONFIG =====
DFF_THRESHOLD = 0.3            # spike-raster threshold (matches all_spikes_plot.py)
FR = bp.FRAME_RATE            # Hz (from behavior_plots config)
CROP = bp.CROP_START_SECONDS  # seconds cropped off the front
BASE = bp.BASE
OUT_PDF = BASE / "traces" / "all_spikes_vs_behavior.pdf"


def load_spikes():
    csv = BASE / "traces" / "dff_traces_curated_bgsub.csv"
    df = pd.read_csv(csv)
    df = df.iloc[int(CROP * FR):]          # drop the warmup, same crop as the rest
    dend = [c for c in df.columns if c.startswith("dend_")]
    traces = df[dend].values
    t = np.arange(traces.shape[0]) / FR
    print(f"  {len(dend)} dendrites, {traces.shape[0]} frames ({t[-1]:.1f}s)")
    return t, traces


def main():
    print(f"=== All spikes + behavior + ACh: {bp.DATE}/{bp.MOUSE}/{bp.RUN} ===\n")

    t_sp, traces = load_spikes()
    print("Loading global Ca + ACh...")
    t_ca, ach, ca = bp.load_calcium_ach_data()         # cropped, bleach-corrected, re-zeroed
    print("Loading pupil...")
    t_beh, pupil, _whisker = bp.load_behavior_data()   # whisker intentionally ignored
    print("Loading accel...")
    t_acc, accel, _ = bp.load_accelerometer_data()

    # Clip everything to the shortest common duration
    dur = t_sp[-1]
    for tt in (t_ca, t_beh, t_acc):
        if tt is not None and len(tt):
            dur = min(dur, tt[-1])
    m = t_sp <= dur
    t_sp, traces = t_sp[m], traces[m]
    N = traces.shape[1]

    # Build panels in the requested order
    panels = [dict(kind="spikes", ylabel="ΔF/F (%)", t=t_sp, d=traces)]
    if t_acc is not None:
        mm = t_acc <= dur
        panels.append(dict(kind="line", ylabel="Accel", color="purple",
                           t=t_acc[mm], d=accel[mm], ylim=(0, 0.25)))
    if t_ca is not None:
        mm = t_ca <= dur
        panels.append(dict(kind="line", ylabel="Global Ca\nΔF/F (%)", color="green",
                           t=t_ca[mm], d=ca[mm]))
        if ach is not None:
            panels.append(dict(kind="line", ylabel="Global ACh\nΔF/F (%)", color="red",
                               t=t_ca[mm], d=ach[mm]))
        else:
            print("  (no ACh trace available — skipping ACh panel)")
    if t_beh is not None:
        mm = t_beh <= dur
        panels.append(dict(kind="line", ylabel="Pupil Dilation", color="blue",
                           t=t_beh[mm], d=pupil[mm]))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), sharex=True,
                             gridspec_kw={"height_ratios": [2.5] + [1.5] * (n - 1)})
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, panels):
        if p["kind"] == "spikes":
            for i in range(N):
                tr = p["d"][:, i].copy()
                tr[tr < DFF_THRESHOLD] = np.nan
                ax.plot(p["t"], tr, lw=0.8, alpha=0.7, color=cm.turbo(i / max(1, N)))
            ax.set_title(f"{bp.DATE} | {bp.MOUSE} | {bp.RUN} — "
                         f"Dendrite spikes (>{DFF_THRESHOLD})")
        else:
            ax.plot(p["t"], p["d"], color=p["color"], lw=1.2)
            if p.get("ylim"):
                ax.set_ylim(*p["ylim"])
        ax.set_ylabel(p["ylabel"])
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    OUT_PDF.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Saved vector PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()
