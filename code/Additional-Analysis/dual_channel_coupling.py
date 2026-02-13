#!/usr/bin/env python3
"""
Dual-Channel Ca--ACh Coupling Analysis

Comprehensive voxelwise analysis of calcium (green) vs acetylcholine (red)
coupling in SCAPE microscopy data.

Analyses:
  (1) Same-voxel correlation maps (zero-lag, max-lag, lag-at-max)
      — on low-pass (<0.3 Hz) and high-pass (residual) separately
  (2) Global ACh regression onto each Ca voxel → beta + R² maps
      — with optional global-Ca control regression
  (3) PCA on Ca and ACh separately → spatial maps, timecourses,
      cross-correlation matrix of PC timecourses between channels
  (4) Voxelwise frequency maps: low/high power ratio per channel

Controls:
  - corr(Ca_voxel, global_Ca) and corr(ACh_voxel, global_ACh)
  - Inside-dendrite vs outside-dendrite coupling comparison
  - Spectral bleedthrough check

Usage:
    python dual_channel_coupling.py              # run all analyses
    python dual_channel_coupling.py --view       # open results in Napari
    python dual_channel_coupling.py --only corr  # run only correlation block
    python dual_channel_coupling.py --only regr  # run only regression block
    python dual_channel_coupling.py --only pca   # run only PCA block
    python dual_channel_coupling.py --only freq  # run only frequency block
"""

import sys
import gc
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA

# ================== CONFIG ==================
DATE = "2025-12-02"
MOUSE = "rbp4cre_136_phpeb"
RUN = "run4"

FS_HZ = 5.0
SKIP_FIRST_SECONDS = 7.0
VOXEL_SIZE = (3.9, 1.0, 1.2)      # (Z, Y, X) µm

# ---- Paths ----
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
RAW_CA_PATH  = BASE / "raw" / f"runA_{RUN}_{MOUSE}_binimagej_reslice_green.tif"
RAW_ACH_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_binimagej_reslice_red.tif"
MASK_FOLDER  = BASE / "labelmaps_curated_dynamic"

OUT = BASE / "coupling_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Preprocessing ----
SPATIAL_SMOOTH = (0, 0.5, 1.0, 1.0)   # (T,Z,Y,X) light spatial denoise
DETREND_WINDOW = 301                    # rolling percentile window for ΔF/F
DETREND_PCT = 20.0                      # percentile for baseline
LOWPASS_CUTOFF = 0.3                    # Hz — separates slow waves from spikes
HIGHPASS_CUTOFF = 0.3                   # Hz — same boundary

# ---- Correlation (block 1) ----
MAX_LAG_SEC = 10.0                      # ±10 s lag search
CORR_CHUNK = 5000                       # voxels per chunk (memory management)

# ---- Regression (block 2) ----
REGRESS_GLOBAL_CA_CONTROL = True        # also regress out global Ca as control

# ---- PCA (block 3) ----
N_PCS = 6                              # number of PCs to extract per channel
PCA_DOWNSAMPLE_SPATIAL = 2             # spatial downsample for PCA (speed)

# ---- Frequency (block 4) ----
FREQ_LOW_BAND = (0.01, 0.3)            # Hz
FREQ_HIGH_BAND = (0.3, 2.0)            # Hz


# ================== PREPROCESSING ==================

def load_4d(path: Path) -> np.ndarray:
    print(f"  Loading {path.name} ...")
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D, got {arr.shape}")
    return arr


def skip_and_align(ca, ach):
    """Skip initial frames, verify shapes match."""
    if ca.shape != ach.shape:
        T_min = min(ca.shape[0], ach.shape[0])
        print(f"  [warn] T mismatch: Ca={ca.shape[0]}, ACh={ach.shape[0]} → trimming to {T_min}")
        ca = ca[:T_min]
        ach = ach[:T_min]
    if SKIP_FIRST_SECONDS > 0:
        skip = int(SKIP_FIRST_SECONDS * FS_HZ)
        ca = ca[skip:]
        ach = ach[skip:]
        print(f"  Skipped first {SKIP_FIRST_SECONDS}s ({skip} frames)")
    return ca, ach


def detrend_dff(stack):
    """Voxelwise ΔF/F with rolling percentile baseline."""
    T, Z, Y, X = stack.shape
    half = DETREND_WINDOW // 2
    f0 = np.empty_like(stack)
    for t in range(T):
        t0 = max(t - half, 0)
        t1 = min(t + half + 1, T)
        f0[t] = np.percentile(stack[t0:t1], DETREND_PCT, axis=0)
    dff = (stack - f0) / (f0 + 1e-6)
    return dff


def bandpass_stack(stack, lo, hi, axis=0):
    """Butterworth bandpass along time axis."""
    nyq = FS_HZ / 2.0
    lo_n = max(lo / nyq, 0.001)
    hi_n = min(hi / nyq, 0.999)
    b, a = butter(3, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, stack, axis=axis).astype(np.float32)


def lowpass_stack(stack, cutoff, axis=0):
    nyq = FS_HZ / 2.0
    wn = min(cutoff / nyq, 0.999)
    b, a = butter(3, wn, btype="low")
    return filtfilt(b, a, stack, axis=axis).astype(np.float32)


def highpass_stack(stack, cutoff, axis=0):
    nyq = FS_HZ / 2.0
    wn = max(cutoff / nyq, 0.001)
    b, a = butter(3, wn, btype="high")
    return filtfilt(b, a, stack, axis=axis).astype(np.float32)


def preprocess(ca_raw, ach_raw):
    """Full preprocessing pipeline. Returns dff stacks + low/high splits."""
    print("Preprocessing...")

    # Spatial smooth
    if any(s > 0 for s in SPATIAL_SMOOTH):
        ca_raw = gaussian_filter(ca_raw, sigma=SPATIAL_SMOOTH)
        ach_raw = gaussian_filter(ach_raw, sigma=SPATIAL_SMOOTH)

    # ΔF/F
    print("  Ca ΔF/F...")
    ca_dff = detrend_dff(ca_raw)
    print("  ACh ΔF/F...")
    ach_dff = detrend_dff(ach_raw)

    del ca_raw, ach_raw
    gc.collect()

    # Low/high split
    print("  Frequency split...")
    ca_low = lowpass_stack(ca_dff, LOWPASS_CUTOFF)
    ca_high = highpass_stack(ca_dff, HIGHPASS_CUTOFF)
    ach_low = lowpass_stack(ach_dff, LOWPASS_CUTOFF)
    ach_high = highpass_stack(ach_dff, HIGHPASS_CUTOFF)

    return ca_dff, ach_dff, ca_low, ca_high, ach_low, ach_high


def load_dendrite_mask():
    """Load union of curated dendrite masks. Returns bool (Z,Y,X) or None."""
    if not MASK_FOLDER.exists():
        return None
    paths = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    if not paths:
        return None
    union = None
    for p in paths:
        m = tifffile.imread(p) > 0
        if union is None:
            union = m
        else:
            if m.shape == union.shape:
                union |= m
    print(f"  Loaded dendrite mask from {len(paths)} files, {union.sum()} voxels")
    return union


# ================== BLOCK 1: CORRELATION MAPS ==================

def _zscore_flat(flat):
    """Z-score each column (voxel) of (T, N) array."""
    m = flat.mean(axis=0, keepdims=True)
    s = flat.std(axis=0, keepdims=True) + 1e-12
    return (flat - m) / s


def corr_at_lag(ca_flat_z, ach_z, lag):
    """Pearson r at a single lag. ca_flat_z: (T,N), ach_z: (T,)."""
    T = len(ach_z)
    if lag < 0:
        a = ach_z[:T + lag]
        c = ca_flat_z[-lag:]
    elif lag > 0:
        a = ach_z[lag:]
        c = ca_flat_z[:T - lag]
    else:
        a, c = ach_z, ca_flat_z
    return (a[:, None] * c).mean(axis=0)


def compute_correlation_maps(ca_stack, ach_stack, label=""):
    """
    Compute r0, rmax, lag_at_rmax for ca vs global-ach.
    Returns dict of (Z,Y,X) arrays.
    """
    T, Z, Y, X = ca_stack.shape
    N = Z * Y * X
    max_lag = int(MAX_LAG_SEC * FS_HZ)

    # Global ACh trace
    ach_trace = ach_stack.mean(axis=(1, 2, 3))
    ach_z = (ach_trace - ach_trace.mean()) / (ach_trace.std() + 1e-12)

    ca_flat = ca_stack.reshape(T, N)
    ca_flat_z = _zscore_flat(ca_flat)

    # Zero-lag
    r0 = corr_at_lag(ca_flat_z, ach_z, 0)

    # Lag sweep
    best_r = r0.copy()
    best_lag = np.zeros(N, dtype=np.int16)

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        r = corr_at_lag(ca_flat_z, ach_z, lag)
        better = np.abs(r) > np.abs(best_r)
        best_r[better] = r[better]
        best_lag[better] = lag

    tag = f"_{label}" if label else ""
    results = {
        f"r0{tag}": r0.reshape(Z, Y, X).astype(np.float32),
        f"rmax{tag}": best_r.reshape(Z, Y, X).astype(np.float32),
        f"lag_at_rmax{tag}": best_lag.reshape(Z, Y, X),
    }

    print(f"  [{label or 'full'}] r0 mean={np.abs(r0).mean():.4f}, "
          f"rmax mean={np.abs(best_r).mean():.4f}, "
          f"|r|>0.3: {(np.abs(best_r) > 0.3).sum()}")

    return results


def run_correlation(ca_dff, ach_dff, ca_low, ca_high, ach_low, ach_high):
    """Block 1: correlation maps on full, low-pass, and high-pass."""
    print("\n=== BLOCK 1: Correlation Maps ===")
    results = {}

    print("  Full-band correlation...")
    results.update(compute_correlation_maps(ca_dff, ach_dff, "full"))

    print("  Low-pass correlation...")
    results.update(compute_correlation_maps(ca_low, ach_low, "low"))

    print("  High-pass correlation...")
    results.update(compute_correlation_maps(ca_high, ach_high, "high"))

    # Save
    for name, vol in results.items():
        p = OUT / f"corr_{name}.tif"
        tifffile.imwrite(p, vol)
    print(f"  Saved {len(results)} correlation volumes to {OUT}")

    # MIP plots
    for band in ["full", "low", "high"]:
        rmax = results[f"rmax_{band}"]
        lag = results[f"lag_at_rmax_{band}"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].imshow(results[f"r0_{band}"].max(axis=0),
                              cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        axes[0].set_title(f"r₀ Z-MIP ({band})")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(rmax.max(axis=0),
                              cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        axes[1].set_title(f"r_max Z-MIP ({band})")
        plt.colorbar(im1, ax=axes[1])

        best_z = np.argmax(np.abs(rmax), axis=0)
        lag_mip = np.take_along_axis(lag, best_z[None], axis=0)[0].astype(np.float32) / FS_HZ
        im2 = axes[2].imshow(lag_mip, cmap="coolwarm",
                              vmin=-MAX_LAG_SEC, vmax=MAX_LAG_SEC, aspect="auto")
        axes[2].set_title(f"Lag at r_max ({band})")
        plt.colorbar(im2, ax=axes[2], label="s")

        fig.suptitle(f"ACh–Ca Correlation ({band}-band) | {DATE} {MOUSE} {RUN}")
        fig.tight_layout()
        fig.savefig(OUT / f"corr_mip_{band}.png", dpi=200)
        plt.close(fig)

    # Histograms
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, band in enumerate(["full", "low", "high"]):
        rmax = results[f"rmax_{band}"].ravel()
        axes[i].hist(rmax, bins=100, range=(-0.8, 0.8), color="steelblue", alpha=0.7)
        axes[i].axvline(0, color="k", ls="--")
        axes[i].set_title(f"r_max distribution ({band})")
        axes[i].set_xlabel("r")
    fig.tight_layout()
    fig.savefig(OUT / "corr_histograms.png", dpi=200)
    plt.close(fig)

    return results


# ================== BLOCK 2: REGRESSION MAPS ==================

def run_regression(ca_dff, ach_dff):
    """
    Regress global ACh(t) onto each Ca voxel → beta map + R² map.
    Optionally also regress out global Ca as control.
    """
    print("\n=== BLOCK 2: Regression Maps ===")
    T, Z, Y, X = ca_dff.shape
    N = Z * Y * X

    # Global ACh trace
    ach_g = ach_dff.mean(axis=(1, 2, 3))
    ach_g_z = (ach_g - ach_g.mean()) / (ach_g.std() + 1e-12)

    ca_flat = ca_dff.reshape(T, N)

    # Simple regression: Ca_voxel = beta * ACh_global + intercept
    print("  Regressing global ACh onto each Ca voxel...")
    ca_mean = ca_flat.mean(axis=0)
    ca_centered = ca_flat - ca_mean[None, :]

    # beta = cov(ca, ach) / var(ach)
    cov = (ach_g_z[:, None] * ca_centered).mean(axis=0)
    beta_ach = cov  # since ach_g_z has unit variance

    # Predicted and residual
    predicted = ach_g_z[:, None] * beta_ach[None, :]
    residual = ca_centered - predicted
    ss_res = (residual ** 2).sum(axis=0)
    ss_tot = (ca_centered ** 2).sum(axis=0) + 1e-12
    r2_ach = 1.0 - ss_res / ss_tot
    r2_ach = np.clip(r2_ach, 0, 1)

    results = {
        "beta_ach": beta_ach.reshape(Z, Y, X).astype(np.float32),
        "r2_ach": r2_ach.reshape(Z, Y, X).astype(np.float32),
    }

    print(f"  ACh regression: mean R²={r2_ach.mean():.4f}, "
          f"max R²={r2_ach.max():.4f}, "
          f"voxels R²>0.1: {(r2_ach > 0.1).sum()}")

    # Control: also regress global Ca
    if REGRESS_GLOBAL_CA_CONTROL:
        print("  Control: regressing global Ca onto each Ca voxel...")
        ca_g = ca_dff.mean(axis=(1, 2, 3))
        ca_g_z = (ca_g - ca_g.mean()) / (ca_g.std() + 1e-12)

        cov_ca = (ca_g_z[:, None] * ca_centered).mean(axis=0)
        beta_ca = cov_ca

        pred_ca = ca_g_z[:, None] * beta_ca[None, :]
        res_ca = ca_centered - pred_ca
        ss_res_ca = (res_ca ** 2).sum(axis=0)
        r2_ca = 1.0 - ss_res_ca / ss_tot
        r2_ca = np.clip(r2_ca, 0, 1)

        results["beta_globalca"] = beta_ca.reshape(Z, Y, X).astype(np.float32)
        results["r2_globalca"] = r2_ca.reshape(Z, Y, X).astype(np.float32)

        # Unique ACh variance = R²_ach - R²_shared (approximate)
        # More precisely: partial R² from multiple regression
        # Build design matrix [ach_g_z, ca_g_z]
        X_design = np.column_stack([ach_g_z, ca_g_z])  # (T, 2)
        XtX_inv = np.linalg.pinv(X_design.T @ X_design)
        betas_multi = XtX_inv @ X_design.T @ ca_centered  # (2, N)
        pred_multi = X_design @ betas_multi
        ss_res_multi = ((ca_centered - pred_multi) ** 2).sum(axis=0)
        r2_multi = 1.0 - ss_res_multi / ss_tot
        r2_multi = np.clip(r2_multi, 0, 1)

        # Unique ACh = R²_multi - R²_globalca_only
        r2_ach_unique = np.clip(r2_multi - r2_ca, 0, 1)
        results["r2_ach_unique"] = r2_ach_unique.reshape(Z, Y, X).astype(np.float32)

        print(f"  Unique ACh R²: mean={r2_ach_unique.mean():.4f}, "
              f"max={r2_ach_unique.max():.4f}")

    # Save
    for name, vol in results.items():
        tifffile.imwrite(OUT / f"regr_{name}.tif", vol)

    # MIP plots
    fig, axes = plt.subplots(1, 3 if REGRESS_GLOBAL_CA_CONTROL else 2, figsize=(15, 4))
    im0 = axes[0].imshow(results["beta_ach"].max(axis=0),
                          cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    axes[0].set_title("β(ACh→Ca) Z-MIP")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(results["r2_ach"].max(axis=0),
                          cmap="hot", vmin=0, vmax=0.3, aspect="auto")
    axes[1].set_title("R²(ACh→Ca) Z-MIP")
    plt.colorbar(im1, ax=axes[1])

    if REGRESS_GLOBAL_CA_CONTROL:
        im2 = axes[2].imshow(results["r2_ach_unique"].max(axis=0),
                              cmap="hot", vmin=0, vmax=0.15, aspect="auto")
        axes[2].set_title("R²(ACh unique) Z-MIP")
        plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"ACh→Ca Regression | {DATE} {MOUSE} {RUN}")
    fig.tight_layout()
    fig.savefig(OUT / "regr_mip.png", dpi=200)
    plt.close(fig)

    return results


# ================== BLOCK 3: PCA ==================

def run_pca(ca_dff, ach_dff, dendrite_mask=None):
    """
    PCA on Ca and ACh separately → spatial maps, timecourses,
    cross-correlation matrix of PC timecourses between channels.
    """
    print("\n=== BLOCK 3: PCA ===")
    T, Z, Y, X = ca_dff.shape
    ds = PCA_DOWNSAMPLE_SPATIAL

    # Optionally downsample spatially for speed
    if ds > 1:
        ca_ds = ca_dff[:, ::ds, ::ds, ::ds]
        ach_ds = ach_dff[:, ::ds, ::ds, ::ds]
    else:
        ca_ds = ca_dff
        ach_ds = ach_dff

    T_ds, Z_ds, Y_ds, X_ds = ca_ds.shape
    N_ds = Z_ds * Y_ds * X_ds

    ca_flat = ca_ds.reshape(T_ds, N_ds)
    ach_flat = ach_ds.reshape(T_ds, N_ds)

    # PCA
    print(f"  Ca PCA ({N_ds} voxels, {N_PCS} PCs)...")
    pca_ca = PCA(n_components=N_PCS)
    ca_scores = pca_ca.fit_transform(ca_flat)  # (T, n_pcs)
    ca_components = pca_ca.components_          # (n_pcs, N)
    ca_var = pca_ca.explained_variance_ratio_

    print(f"  ACh PCA ({N_ds} voxels, {N_PCS} PCs)...")
    pca_ach = PCA(n_components=N_PCS)
    ach_scores = pca_ach.fit_transform(ach_flat)
    ach_components = pca_ach.components_
    ach_var = pca_ach.explained_variance_ratio_

    print(f"  Ca  variance explained: {ca_var}")
    print(f"  ACh variance explained: {ach_var}")

    # Cross-correlation matrix between Ca PCs and ACh PCs
    print("  Cross-correlation matrix (Ca PCs × ACh PCs)...")
    cc_matrix = np.zeros((N_PCS, N_PCS), dtype=np.float32)
    for i in range(N_PCS):
        for j in range(N_PCS):
            cc_matrix[i, j] = np.corrcoef(ca_scores[:, i], ach_scores[:, j])[0, 1]

    # Save spatial maps
    for ch, comps, var_exp in [("ca", ca_components, ca_var),
                                ("ach", ach_components, ach_var)]:
        for k in range(N_PCS):
            spatial = comps[k].reshape(Z_ds, Y_ds, X_ds).astype(np.float32)
            tifffile.imwrite(OUT / f"pca_{ch}_pc{k:02d}_spatial.tif", spatial)

    # Save timecourses
    np.savez(OUT / "pca_timecourses.npz",
             ca_scores=ca_scores, ach_scores=ach_scores,
             ca_var=ca_var, ach_var=ach_var,
             cc_matrix=cc_matrix)

    # --- Plots ---
    # Spatial maps (Z-MIP)
    for ch, comps, var_exp in [("Ca", ca_components, ca_var),
                                ("ACh", ach_components, ach_var)]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for k in range(min(N_PCS, 6)):
            ax = axes[k // 3, k % 3]
            spatial = comps[k].reshape(Z_ds, Y_ds, X_ds)
            mip = spatial.max(axis=0)
            vmax = np.percentile(np.abs(mip), 99)
            ax.imshow(mip, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"PC{k} ({var_exp[k]:.1%})")
            ax.axis("off")
        fig.suptitle(f"{ch} PCA Spatial Maps (Z-MIP)")
        fig.tight_layout()
        fig.savefig(OUT / f"pca_{ch.lower()}_spatial_mip.png", dpi=200)
        plt.close(fig)

    # Timecourses
    t_sec = np.arange(T_ds) / FS_HZ
    fig, axes = plt.subplots(N_PCS, 2, figsize=(14, 2 * N_PCS), sharex=True)
    for k in range(N_PCS):
        axes[k, 0].plot(t_sec, ca_scores[:, k], color="green", lw=0.8)
        axes[k, 0].set_ylabel(f"Ca PC{k}")
        axes[k, 1].plot(t_sec, ach_scores[:, k], color="red", lw=0.8)
        axes[k, 1].set_ylabel(f"ACh PC{k}")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("PCA Timecourses")
    fig.tight_layout()
    fig.savefig(OUT / "pca_timecourses.png", dpi=200)
    plt.close(fig)

    # Cross-correlation matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cc_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(N_PCS))
    ax.set_xticklabels([f"ACh PC{i}" for i in range(N_PCS)], rotation=45, ha="right")
    ax.set_yticks(range(N_PCS))
    ax.set_yticklabels([f"Ca PC{i}" for i in range(N_PCS)])
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Ca PC × ACh PC Cross-Correlation")
    fig.tight_layout()
    fig.savefig(OUT / "pca_cross_correlation.png", dpi=200)
    plt.close(fig)

    print(f"  Saved PCA results to {OUT}")
    return cc_matrix


# ================== BLOCK 4: FREQUENCY MAPS ==================

def run_frequency(ca_dff, ach_dff):
    """
    Voxelwise frequency maps: power in low band, high band, ratio.
    """
    print("\n=== BLOCK 4: Frequency Maps ===")
    T, Z, Y, X = ca_dff.shape

    results = {}
    for ch_name, stack in [("ca", ca_dff), ("ach", ach_dff)]:
        print(f"  {ch_name} frequency decomposition...")

        low = bandpass_stack(stack, FREQ_LOW_BAND[0], FREQ_LOW_BAND[1])
        high = bandpass_stack(stack, FREQ_HIGH_BAND[0], FREQ_HIGH_BAND[1])

        # Power = variance over time
        pow_low = np.var(low, axis=0)
        pow_high = np.var(high, axis=0)
        ratio = pow_low / (pow_high + 1e-12)

        results[f"pow_low_{ch_name}"] = pow_low.astype(np.float32)
        results[f"pow_high_{ch_name}"] = pow_high.astype(np.float32)
        results[f"pow_ratio_{ch_name}"] = ratio.astype(np.float32)

        print(f"    low power mean={pow_low.mean():.6f}, "
              f"high power mean={pow_high.mean():.6f}, "
              f"ratio mean={ratio.mean():.2f}")

    # Save
    for name, vol in results.items():
        tifffile.imwrite(OUT / f"freq_{name}.tif", vol)

    # MIP comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, ch in enumerate(["ca", "ach"]):
        for col, metric in enumerate(["pow_low", "pow_high", "pow_ratio"]):
            ax = axes[row, col]
            vol = results[f"{metric}_{ch}"]
            mip = vol.max(axis=0)
            if "ratio" in metric:
                vmax = np.percentile(mip, 98)
                im = ax.imshow(mip, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            else:
                vmax = np.percentile(mip, 99)
                im = ax.imshow(mip, cmap="hot", vmin=0, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{ch.upper()} {metric}")
            ax.axis("off")

    fig.suptitle(f"Frequency Power Maps | {DATE} {MOUSE} {RUN}")
    fig.tight_layout()
    fig.savefig(OUT / "freq_power_mip.png", dpi=200)
    plt.close(fig)

    return results


# ================== CONTROLS ==================

def run_controls(ca_dff, ach_dff, corr_results, dendrite_mask=None):
    """
    Control analyses:
    - corr(Ca_voxel, global_Ca) — shared-motion artifact
    - corr(ACh_voxel, global_ACh) — ACh spatial structure
    - Inside vs outside dendrite mask comparison
    - Bleedthrough check: corr(Ca_mean, ACh_mean) at zero lag
    """
    print("\n=== CONTROLS ===")
    T, Z, Y, X = ca_dff.shape

    # Global self-correlation maps
    for ch_name, stack in [("ca", ca_dff), ("ach", ach_dff)]:
        g_trace = stack.mean(axis=(1, 2, 3))
        g_z = (g_trace - g_trace.mean()) / (g_trace.std() + 1e-12)
        flat = stack.reshape(T, -1)
        flat_z = _zscore_flat(flat)
        r_self = (g_z[:, None] * flat_z).mean(axis=0).reshape(Z, Y, X).astype(np.float32)
        tifffile.imwrite(OUT / f"ctrl_self_corr_{ch_name}.tif", r_self)
        print(f"  {ch_name} self-corr: mean={r_self.mean():.4f}, max={r_self.max():.4f}")

    # Bleedthrough check
    ca_g = ca_dff.mean(axis=(1, 2, 3))
    ach_g = ach_dff.mean(axis=(1, 2, 3))
    r_bleed = np.corrcoef(ca_g, ach_g)[0, 1]
    print(f"  Bleedthrough check: corr(global_Ca, global_ACh) = {r_bleed:.4f}")

    # Inside vs outside dendrite mask
    if dendrite_mask is not None and "rmax_full" in corr_results:
        rmax = corr_results["rmax_full"]
        inside = rmax[dendrite_mask].ravel()
        outside = rmax[~dendrite_mask].ravel()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(inside, bins=80, range=(-0.6, 0.6), alpha=0.6, label=f"Inside dendrites (n={len(inside)})", color="green")
        ax.hist(outside, bins=80, range=(-0.6, 0.6), alpha=0.6, label=f"Outside dendrites (n={len(outside)})", color="gray")
        ax.axvline(0, color="k", ls="--")
        ax.set_xlabel("r_max (ACh–Ca)")
        ax.set_ylabel("Voxel count")
        ax.set_title("ACh–Ca Coupling: Inside vs Outside Dendrites")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "ctrl_inside_vs_outside.png", dpi=200)
        plt.close(fig)

        print(f"  Inside dendrites:  mean r_max = {inside.mean():.4f}")
        print(f"  Outside dendrites: mean r_max = {outside.mean():.4f}")

    # Summary text
    with open(OUT / "controls_summary.txt", "w") as f:
        f.write(f"Bleedthrough: corr(global_Ca, global_ACh) = {r_bleed:.4f}\n")
        if dendrite_mask is not None and "rmax_full" in corr_results:
            f.write(f"Inside dendrites:  mean rmax = {inside.mean():.4f}\n")
            f.write(f"Outside dendrites: mean rmax = {outside.mean():.4f}\n")


# ================== Z FLY-THROUGH VIDEO ==================

def export_flythrough(vol, name, cmap="RdBu_r", vmin=-0.5, vmax=0.5):
    """Export a Z fly-through as mp4 for a 3D volume."""
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print(f"  [skip] ffmpeg not available for {name} video")
        return

    Z = vol.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(vol[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax)
    title = ax.set_title(f"{name} Z=0")
    ax.axis("off")

    def update(z):
        im.set_data(vol[z])
        title.set_text(f"{name} Z={z}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=Z, interval=200, blit=True)
    out_path = OUT / f"flythrough_{name}.mp4"
    anim.save(str(out_path), writer=FFMpegWriter(fps=5))
    plt.close(fig)
    print(f"  Saved fly-through: {out_path}")


# ================== MAIN ==================

def main(only=None):
    print(f"=== Dual-Channel Coupling Analysis ===")
    print(f"    {DATE} | {MOUSE} | {RUN}")
    print(f"    Output: {OUT}\n")

    # Load
    ca_raw = load_4d(RAW_CA_PATH)
    ach_raw = load_4d(RAW_ACH_PATH)
    ca_raw, ach_raw = skip_and_align(ca_raw, ach_raw)

    T, Z, Y, X = ca_raw.shape
    print(f"  Working shape: T={T}, Z={Z}, Y={Y}, X={X}\n")

    # Preprocess
    ca_dff, ach_dff, ca_low, ca_high, ach_low, ach_high = preprocess(ca_raw, ach_raw)
    del ca_raw, ach_raw
    gc.collect()

    # Load dendrite mask
    dendrite_mask = load_dendrite_mask()

    corr_results = {}

    # Block 1: Correlation
    if only is None or only == "corr":
        corr_results = run_correlation(ca_dff, ach_dff, ca_low, ca_high, ach_low, ach_high)

    # Block 2: Regression
    if only is None or only == "regr":
        run_regression(ca_dff, ach_dff)

    # Block 3: PCA
    if only is None or only == "pca":
        run_pca(ca_dff, ach_dff, dendrite_mask)

    # Block 4: Frequency
    if only is None or only == "freq":
        run_frequency(ca_dff, ach_dff)

    # Controls
    if only is None:
        run_controls(ca_dff, ach_dff, corr_results, dendrite_mask)

    # Fly-through videos for key volumes
    if only is None or only == "corr":
        for band in ["full", "low", "high"]:
            p = OUT / f"corr_rmax_{band}.tif"
            if p.exists():
                vol = tifffile.imread(p)
                export_flythrough(vol, f"rmax_{band}")

    print(f"\n=== All done. Results in {OUT} ===")


def view_results():
    """Open key results in Napari."""
    import napari

    viewer = napari.Viewer(title="ACh–Ca Coupling")
    scale = VOXEL_SIZE

    for name, cmap, clim in [
        ("corr_rmax_full", "RdBu_r", (-0.5, 0.5)),
        ("corr_rmax_low", "RdBu_r", (-0.5, 0.5)),
        ("corr_rmax_high", "RdBu_r", (-0.5, 0.5)),
        ("regr_r2_ach", "hot", (0, 0.3)),
        ("regr_beta_ach", "RdBu_r", (-0.3, 0.3)),
    ]:
        p = OUT / f"{name}.tif"
        if p.exists():
            vol = tifffile.imread(p)
            viewer.add_image(vol, name=name, colormap=cmap,
                             contrast_limits=clim, scale=scale, visible=False)

    # Make the first one visible
    if len(viewer.layers) > 0:
        viewer.layers[0].visible = True

    napari.run()


if __name__ == "__main__":
    if "--view" in sys.argv:
        view_results()
    elif "--only" in sys.argv:
        idx = sys.argv.index("--only")
        block = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        main(only=block)
    else:
        main()
