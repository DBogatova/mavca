#!/usr/bin/env python3
"""
ACh–Ca Voxelwise Correlation Map

For every voxel in the Ca (green) channel, compute the Pearson correlation
between its ΔF/F time-series and the global ACh (red) ΔF/F signal.

Output:
  - 3D correlation volume (Z,Y,X) saved as TIF  → fly-through in Napari/Fiji
  - Z-MIP of correlation map (Y,X) saved as PNG
  - Optional: lag-map (which lag gives peak correlation per voxel)

This answers: "where in the Ca volume do slow ACh-like waves originate?"

Usage:
    python ach_ca_correlation_map.py
    python ach_ca_correlation_map.py --view   # open result in Napari
"""

import sys
import gc
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# ================== CONFIG ==================
DATE = "2025-12-02"
MOUSE = "rbp4cre_136_phpeb"
RUN = "run4"

FS_HZ = 5.0                       # frame rate
SKIP_FIRST_SECONDS = 15.0          # skip initial transient

# ---- Paths ----
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
RAW_CA_PATH  = BASE / "raw" / f"runA_{RUN}_{MOUSE}_binimagej_reslice_green.tif"
RAW_ACH_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_binimagej_reslice_red.tif"

OUT_DIR = BASE / "correlation_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Processing ----
SPATIAL_SMOOTH_SIGMA = (0, 0.5, 1.0, 1.0)  # (T,Z,Y,X) light denoise on Ca
TEMPORAL_SMOOTH_SIGMA = 2.0        # temporal smooth on both signals (frames)
DFF_BASELINE_WINDOW = 301          # rolling percentile window for ΔF/F
DFF_BASELINE_PCT = 20.0            # percentile for baseline

# ---- Correlation ----
MAX_LAG_FRAMES = 30                # max lag for lag-map (0 = zero-lag only)
ACH_MODE = "global"                # "global" = single global ACh trace
                                   # "voxel"  = voxelwise ACh (slower, spatially resolved)

# ---- Output ----
CORRMAP_PATH   = OUT_DIR / "ach_ca_corrmap_3d.tif"
CORRMAP_MIP    = OUT_DIR / "ach_ca_corrmap_zmip.png"
LAGMAP_PATH    = OUT_DIR / "ach_ca_lagmap_3d.tif"
LAGMAP_MIP     = OUT_DIR / "ach_ca_lagmap_zmip.png"


# ================== HELPERS ==================

def load_4d(path):
    """Load a 4D TIF stack as float32."""
    print(f"  Loading: {path.name}")
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D (T,Z,Y,X), got shape {arr.shape}")
    return arr


def rolling_percentile_baseline_3d(stack, win, pct, chunk_t=50):
    """
    Compute a rolling-percentile F0 for each voxel.
    Done in temporal chunks to limit memory.
    Returns F0 array same shape as stack.
    """
    T, Z, Y, X = stack.shape
    half = win // 2
    f0 = np.empty_like(stack)

    for t in range(T):
        t0 = max(t - half, 0)
        t1 = min(t + half + 1, T)
        f0[t] = np.percentile(stack[t0:t1], pct, axis=0)

    return f0


def compute_dff_stack(stack, win, pct):
    """Compute voxelwise ΔF/F using rolling percentile baseline."""
    print("  Computing ΔF/F baseline...")
    f0 = rolling_percentile_baseline_3d(stack, win, pct)
    dff = (stack - f0) / (f0 + 1e-6)
    del f0
    gc.collect()
    return dff


def compute_global_trace(stack, win, pct):
    """Compute global mean trace → ΔF/F."""
    raw_trace = stack.mean(axis=(1, 2, 3))
    # Rolling percentile baseline on 1D trace
    half = win // 2
    T = len(raw_trace)
    f0 = np.empty(T, dtype=np.float32)
    for t in range(T):
        t0 = max(t - half, 0)
        t1 = min(t + half + 1, T)
        f0[t] = np.percentile(raw_trace[t0:t1], pct)
    dff = (raw_trace - f0) / (f0 + 1e-6)
    return dff.astype(np.float32)


def pearson_voxelwise(ca_dff, ach_signal):
    """
    Compute Pearson r between each Ca voxel's time-series and ach_signal.

    ca_dff: (T, Z, Y, X)
    ach_signal: (T,) — global ACh trace

    Returns: corrmap (Z, Y, X) float32
    """
    T, Z, Y, X = ca_dff.shape
    # Reshape to (T, N)
    flat = ca_dff.reshape(T, -1)

    # Z-score both
    ach_z = (ach_signal - ach_signal.mean()) / (ach_signal.std() + 1e-12)
    flat_mean = flat.mean(axis=0, keepdims=True)
    flat_std = flat.std(axis=0, keepdims=True) + 1e-12
    flat_z = (flat - flat_mean) / flat_std

    # Pearson r = mean of element-wise product of z-scores
    r = (ach_z[:, None] * flat_z).mean(axis=0)

    return r.reshape(Z, Y, X).astype(np.float32)


def lagged_correlation_voxelwise(ca_dff, ach_signal, max_lag):
    """
    For each voxel, compute Pearson r at lags -max_lag..+max_lag.
    Returns:
      best_r:   (Z,Y,X) — peak |r| value
      best_lag: (Z,Y,X) — lag (in frames) at peak |r|
    """
    T, Z, Y, X = ca_dff.shape
    N = Z * Y * X
    flat = ca_dff.reshape(T, N)

    best_r = np.zeros(N, dtype=np.float32)
    best_lag = np.zeros(N, dtype=np.int16)

    ach_z = (ach_signal - ach_signal.mean()) / (ach_signal.std() + 1e-12)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = ach_z[:T + lag]
            c = flat[-lag:, :]
        elif lag > 0:
            a = ach_z[lag:]
            c = flat[:T - lag, :]
        else:
            a = ach_z
            c = flat

        c_mean = c.mean(axis=0, keepdims=True)
        c_std = c.std(axis=0, keepdims=True) + 1e-12
        c_z = (c - c_mean) / c_std

        r = (a[:, None] * c_z).mean(axis=0)

        better = np.abs(r) > np.abs(best_r)
        best_r[better] = r[better]
        best_lag[better] = lag

    return best_r.reshape(Z, Y, X), best_lag.reshape(Z, Y, X)


# ================== MAIN ==================

def main():
    # --- Load ---
    print("Loading stacks...")
    ca_4d = load_4d(RAW_CA_PATH)
    ach_4d = load_4d(RAW_ACH_PATH)

    if ca_4d.shape != ach_4d.shape:
        raise ValueError(f"Shape mismatch: Ca {ca_4d.shape} vs ACh {ach_4d.shape}")

    # Skip initial frames
    if SKIP_FIRST_SECONDS > 0:
        skip = int(SKIP_FIRST_SECONDS * FS_HZ)
        ca_4d = ca_4d[skip:]
        ach_4d = ach_4d[skip:]
        print(f"  Skipped first {SKIP_FIRST_SECONDS}s ({skip} frames)")

    T, Z, Y, X = ca_4d.shape
    print(f"  Shape: T={T}, Z={Z}, Y={Y}, X={X}")

    # --- Preprocess Ca ---
    print("Preprocessing Ca...")
    if any(s > 0 for s in SPATIAL_SMOOTH_SIGMA):
        ca_4d = gaussian_filter(ca_4d, sigma=SPATIAL_SMOOTH_SIGMA)
    if TEMPORAL_SMOOTH_SIGMA > 0:
        ca_4d = gaussian_filter1d(ca_4d, sigma=TEMPORAL_SMOOTH_SIGMA, axis=0)

    print("Computing Ca ΔF/F (voxelwise)...")
    ca_dff = compute_dff_stack(ca_4d, DFF_BASELINE_WINDOW, DFF_BASELINE_PCT)
    del ca_4d
    gc.collect()

    # --- Preprocess ACh ---
    print("Preprocessing ACh...")
    if TEMPORAL_SMOOTH_SIGMA > 0:
        ach_4d = gaussian_filter1d(ach_4d, sigma=TEMPORAL_SMOOTH_SIGMA, axis=0)

    print("Computing ACh global ΔF/F...")
    ach_trace = compute_global_trace(ach_4d, DFF_BASELINE_WINDOW, DFF_BASELINE_PCT)
    del ach_4d
    gc.collect()

    # --- Correlation ---
    print("Computing voxelwise ACh–Ca correlation...")
    if MAX_LAG_FRAMES > 0:
        best_r, best_lag = lagged_correlation_voxelwise(ca_dff, ach_trace, MAX_LAG_FRAMES)
        print(f"  Lag range: {-MAX_LAG_FRAMES} to +{MAX_LAG_FRAMES} frames "
              f"({-MAX_LAG_FRAMES/FS_HZ:.1f}s to +{MAX_LAG_FRAMES/FS_HZ:.1f}s)")
    else:
        best_r = pearson_voxelwise(ca_dff, ach_trace)
        best_lag = np.zeros((Z, Y, X), dtype=np.int16)

    del ca_dff
    gc.collect()

    # --- Stats ---
    print(f"\n  Correlation stats:")
    print(f"    mean |r| = {np.abs(best_r).mean():.4f}")
    print(f"    max  r   = {best_r.max():.4f}")
    print(f"    min  r   = {best_r.min():.4f}")
    print(f"    voxels with |r| > 0.3: {(np.abs(best_r) > 0.3).sum()}")
    print(f"    voxels with |r| > 0.5: {(np.abs(best_r) > 0.5).sum()}")
    if MAX_LAG_FRAMES > 0:
        print(f"    median best lag = {np.median(best_lag):.1f} frames "
              f"({np.median(best_lag)/FS_HZ:.2f}s)")

    # --- Save 3D volumes ---
    print("\nSaving outputs...")
    tifffile.imwrite(CORRMAP_PATH, best_r)
    print(f"  Correlation map: {CORRMAP_PATH}")

    if MAX_LAG_FRAMES > 0:
        tifffile.imwrite(LAGMAP_PATH, best_lag)
        print(f"  Lag map: {LAGMAP_PATH}")

    # --- Z-MIP plots ---
    # Correlation MIP
    r_mip = np.nanmax(best_r, axis=0)  # (Y, X)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(r_mip, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r (ACh–Ca)")
    ax.set_title(f"ACh–Ca Correlation Z-MIP\n{DATE} | {MOUSE} | {RUN}")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(CORRMAP_MIP, dpi=200)
    plt.close(fig)
    print(f"  Correlation MIP: {CORRMAP_MIP}")

    # Lag MIP (at max |r| voxel per column)
    if MAX_LAG_FRAMES > 0:
        # For each (y,x), pick the z with highest |r|
        best_z = np.argmax(np.abs(best_r), axis=0)
        lag_mip = np.take_along_axis(best_lag, best_z[None, :, :], axis=0)[0]
        lag_mip_sec = lag_mip.astype(np.float32) / FS_HZ

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(lag_mip_sec, cmap="coolwarm",
                        vmin=-MAX_LAG_FRAMES / FS_HZ, vmax=MAX_LAG_FRAMES / FS_HZ,
                        aspect="auto")
        plt.colorbar(im, ax=ax, label="Best lag (s), +ve = Ca lags ACh")
        ax.set_title(f"ACh–Ca Lag Map Z-MIP\n{DATE} | {MOUSE} | {RUN}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        fig.tight_layout()
        fig.savefig(LAGMAP_MIP, dpi=200)
        plt.close(fig)
        print(f"  Lag MIP: {LAGMAP_MIP}")

    print("\nDone.")


def view_results():
    """Open the correlation map in Napari for fly-through."""
    import napari

    print("Loading correlation map for viewing...")
    corrmap = tifffile.imread(CORRMAP_PATH)

    viewer = napari.Viewer(title="ACh–Ca Correlation Map")
    viewer.add_image(
        corrmap, name="ACh-Ca correlation",
        colormap="RdBu_r",
        contrast_limits=(-0.5, 0.5),
        scale=(3.9, 1.0, 1.2),  # Z, Y, X in μm
    )

    if LAGMAP_PATH.exists():
        lagmap = tifffile.imread(LAGMAP_PATH).astype(np.float32) / FS_HZ
        viewer.add_image(
            lagmap, name="Best lag (s)",
            colormap="coolwarm",
            contrast_limits=(-MAX_LAG_FRAMES / FS_HZ, MAX_LAG_FRAMES / FS_HZ),
            scale=(3.9, 1.0, 1.2),
            visible=False,
        )

    napari.run()


if __name__ == "__main__":
    if "--view" in sys.argv:
        view_results()
    else:
        main()
