#!/usr/bin/env python3
"""
Export a single frame as publication snapshots:
  - behavior trace  -> TRUE vector PDF (matplotlib line plot, cursor at the frame)
  - 3D movie frame  -> full-resolution PNG (raster) + raster-embedded PDF

The behavior panels reuse the exact loaders from accel_trace_movie.py, so the
static figure matches the movie frame-for-frame. Both run7 movies share the same
5 fps / 535-frame timebase, so FRAME indexes both identically
(movie time = FRAME / IMAGING_FRAME_RATE).

Usage:
    python code/Visual-STEP4/export_frame_snapshot.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use Arial; keep PDF text as embedded TrueType (editable, not outlined)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42

# Reuse the movie's own config + signal loaders (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import accel_trace_movie as atm

# ===== CONFIG =====
FRAME = 258                      # frame index in the movie timebase
IFR = atm.IMAGING_FRAME_RATE     # 5 Hz
T_MARK = FRAME / IFR             # 51.6 s in movie time

BASE = atm.BASE
OVERLAYS = BASE / "overlays"
THREE_D_MOVIE = OVERLAYS / "run7_3d.mp4"

BEHAVIOR_PDF = OVERLAYS / f"behavior_trace_frame{FRAME}.pdf"
THREE_D_PNG = OVERLAYS / f"run7_3d_frame{FRAME}.png"
THREE_D_PDF = OVERLAYS / f"run7_3d_frame{FRAME}.pdf"


def export_behavior_trace_pdf():
    """Static vector PDF of the stacked behavior panels with cursor at FRAME."""
    print(f"Loading behavior signals (cursor at frame {FRAME} = {T_MARK:.2f}s)...")
    t_acc, accel = atm.load_accel()
    t_pup, pupil = atm.load_pupil()
    t_ca, ca = atm.load_global_ca()

    # Imaging duration (same computation as the movie)
    raw_clean = BASE / "preprocessed" / "raw_clean.tif"
    raw_orig = BASE / "raw" / f"runB_{atm.RUN}_{atm.MOUSE}-reslice-bin.tif"
    raw_path = raw_clean if raw_clean.exists() else raw_orig
    import tifffile
    with tifffile.TiffFile(str(raw_path)) as tf:
        T_raw = tf.series[0].shape[0]
    duration = (T_raw / IFR) - atm.CROP_START_SECONDS

    panels = []
    if t_acc is not None:
        m = t_acc <= duration
        panels.append(("Accel", t_acc[m], accel[m], "#aa77ff"))
    if t_pup is not None:
        m = t_pup <= duration
        panels.append(("Pupil", t_pup[m], pupil[m], "#4488ff"))
    if t_ca is not None:
        m = t_ca <= duration
        panels.append(("Global Ca ΔF/F (%)", t_ca[m], ca[m], "#44cc44"))
    if not panels:
        print("No behavior data to plot.")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(atm.FIG_W, atm.FIG_H),
                             sharex=True)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor("black")

    for ax, (name, t, d, color) in zip(axes, panels):
        ax.set_facecolor("black")
        ax.plot(t, d, color=color, lw=1.0)
        ax.axvline(T_MARK, color="red", lw=2, alpha=0.8)
        ax.set_xlim(0, duration)
        if name == "Accel":
            ax.set_ylim(0, 0.10)
        else:
            ymin, ymax = d.min(), d.max()
            margin = (ymax - ymin) * 0.1 + 1e-6
            ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_ylabel(name, fontsize=9, color="white")
        ax.tick_params(labelsize=7, colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, color="white")

    axes[-1].set_xlabel("Time (s)", fontsize=9, color="white")
    fig.tight_layout(pad=0.5)
    # Vector PDF; keep the black background on save
    fig.savefig(str(BEHAVIOR_PDF), format="pdf", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ vector PDF: {BEHAVIOR_PDF}")


def export_3d_frame():
    """Extract FRAME from the 3D movie (raster) and wrap into a PDF."""
    if not THREE_D_MOVIE.exists():
        print(f"3D movie not found: {THREE_D_MOVIE}")
        return
    print(f"Extracting frame {FRAME} from {THREE_D_MOVIE.name} (full resolution)...")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(THREE_D_MOVIE),
        "-vf", f"select=eq(n\\,{FRAME})",
        "-vframes", "1",
        str(THREE_D_PNG),
    ]
    subprocess.run(cmd, check=True)
    print(f"  ✅ raster PNG: {THREE_D_PNG}")

    # Wrap the PNG into a PDF at native resolution (raster-embedded, NOT vector)
    img = plt.imread(str(THREE_D_PNG))
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(str(THREE_D_PDF), format="pdf", dpi=100)
    plt.close(fig)
    print(f"  ✅ raster-embedded PDF: {THREE_D_PDF}")


def main():
    OVERLAYS.mkdir(exist_ok=True)
    print(f"=== Frame {FRAME} snapshot: {atm.DATE}/{atm.MOUSE}/{atm.RUN} ===\n")
    export_behavior_trace_pdf()
    print()
    export_3d_frame()


if __name__ == "__main__":
    main()
