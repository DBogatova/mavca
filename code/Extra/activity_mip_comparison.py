#!/usr/bin/env python3
"""
Compare masked ΔF/F MIP between active and quiet periods.

Both panels use temporal max, but each is independently contrast-stretched
to show the brightest dendrites in each period clearly.

Usage:
    python code/Extra/activity_mip_comparison.py
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUN = "run7"

FRAME_RATE = 5.0
CROP_START_SECONDS = 12.0
SPLIT_TIME = 25.0  # seconds

Y_CROP = 3

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
RAW_PATH = BASE / "preprocessed" / "raw_clean.tif"
if not RAW_PATH.exists():
    RAW_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin.tif"
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
OUTPUT = BASE / "traces" / "activity_mip_active_vs_quiet.png"


def main():
    print(f"=== Activity MIP Comparison ===\n")

    store = tifffile.memmap(str(RAW_PATH), mode='r')
    T_raw, Z, Y_raw, X = store.shape
    Y = Y_raw - Y_CROP if Y_CROP > 0 else Y_raw
    print(f"  Stack: T={T_raw}, Z={Z}, Y={Y}, X={X}")

    # Union mask
    mask_paths = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    union = np.zeros((Z, Y, X), dtype=bool)
    for p in mask_paths:
        m = tifffile.imread(p).astype(bool)
        my = m.shape[1]
        if my < Y: m = np.pad(m, ((0,0),(0,Y-my),(0,0)))
        elif my > Y: m = m[:, :Y, :]
        union |= m
    print(f"  {len(mask_paths)} masks")

    # F0 — 10th percentile across entire recording (true minimum baseline)
    print("  Computing F0 (10th percentile, full recording)...")
    f0_frames = []
    # Sample every 5th frame for speed
    for t in range(0, T_raw, 5):
        frame = np.asarray(store[t]).astype(np.float32)
        if Y_CROP > 0: frame = frame[:, :Y, :]
        f0_frames.append(frame)
    f0 = np.percentile(np.stack(f0_frames), 10, axis=0)
    del f0_frames

    # Time ranges
    # Time ranges — start from beginning to capture early activity
    t0_active = int(CROP_START_SECONDS * FRAME_RATE)  # 0 if no crop
    t1_active = t0_active + int(SPLIT_TIME * FRAME_RATE)
    t0_quiet = t1_active
    t1_quiet = T_raw

    # Temporal max per period (masked) — preserves individual dendrite peaks
    def compute_max(t0, t1):
        vol = np.full((Z, Y, X), -np.inf, dtype=np.float32)
        for t in range(t0, t1, 2):
            frame = np.asarray(store[t]).astype(np.float32)
            if Y_CROP > 0: frame = frame[:, :Y, :]
            dff = (frame - f0) / (f0 + 1e-6)
            dff[~union] = -np.inf
            np.maximum(vol, dff, out=vol)
        vol[vol == -np.inf] = 0
        return vol

    print("  Active max...")
    active = compute_max(t0_active, t1_active)
    print("  Quiet max...")
    quiet = compute_max(t0_quiet, t1_quiet)
    del store

    # Z-MIP
    active_mip = active.max(axis=0) * 100
    quiet_mip = quiet.max(axis=0) * 100

    # Independent normalization per panel
    def normalize(mip):
        vals = mip[mip > 0]
        if vals.size == 0: return mip
        lo = np.percentile(vals, 5)
        hi = np.percentile(vals, 99)
        return np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)

    active_norm = normalize(active_mip)
    quiet_norm = normalize(quiet_mip)

    # Gamma correction — boosts mid-tones (active has more mid-range → looks brighter)
    GAMMA = 0.5  # <1 = brighter midtones
    active_norm = np.power(active_norm, GAMMA)
    quiet_norm = np.power(quiet_norm, GAMMA)

    # Green colormap
    green_cmap = LinearSegmentedColormap.from_list("green",
        [(0, 0, 0), (0.05, 0.2, 0.05), (0.2, 0.7, 0.1), (0.4, 1.0, 0.3)])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black',
                              gridspec_kw={"width_ratios": [1, 1, 0.04]})

    axes[0].set_facecolor('black')
    im = axes[0].imshow(active_norm, cmap=green_cmap, vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f"Active (0–{SPLIT_TIME:.0f}s)", color='white', fontsize=12)
    axes[0].axis('off')

    axes[1].set_facecolor('black')
    axes[1].imshow(quiet_norm, cmap=green_cmap, vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f"Quiet ({SPLIT_TIME:.0f}s–end)", color='white', fontsize=12)
    axes[1].axis('off')

    # Colorbar
    cbar = fig.colorbar(im, cax=axes[2])
    cbar.set_label("Normalized max ΔF/F", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.suptitle(f"{DATE} | {MOUSE} | {RUN}", fontsize=13, color='white')
    plt.tight_layout()

    OUTPUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches='tight', facecolor='black')
    print(f"\n✅ Saved: {OUTPUT}")
    plt.show()


if __name__ == "__main__":
    main()
