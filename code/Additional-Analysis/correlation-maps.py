#!/usr/bin/env python3
"""
correlation-maps.py  (Module 4B — crop-aware + correlation + LINE candidate mode)

Key addition:
- --candidate_mode lines / hybrid
    lines  : detect thin trunks using event-window max|ΔF/F| + Frangi ridge filter + skeleton
    hybrid : same as lines, but also enforces corrmap >= --hybrid_r_min within candidate voxels

Notes:
- No full ΔF/F file stored. Everything computed on the fly from RAW + F0.
- Works with 3D masks (Z,Y,X) and RAW (T,Z,Y,X) or 2D masks (Y,X) and RAW (T,Y,X).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks

# scikit-image (you have 0.21.0)
from skimage.filters import frangi
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    disk,
    skeletonize,
)
from skimage.measure import label as sk_label
from skimage.measure import regionprops


# -----------------------------
# I/O helpers
# -----------------------------

def load_raw(path: Path, mmap: bool = True) -> np.ndarray:
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None)

    if suf in (".tif", ".tiff"):
        try:
            return tifffile.memmap(str(path))
        except Exception:
            return tifffile.imread(str(path))

    raise ValueError(f"Unsupported RAW input: {path.suffix}. Use .npy or .tif/.tiff")


def load_mask_tiff(path: Path) -> np.ndarray:
    m = tifffile.imread(str(path))
    return (m > 0)


def save_tiff(path: Path, arr: np.ndarray, compress: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if compress:
        kwargs["compression"] = "zlib"
    tifffile.imwrite(str(path), arr, **kwargs)


def safe_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if (not np.isfinite(sd)) or sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / (sd + eps)


def regress_out_1d(y: np.ndarray, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    vx = np.nanvar(x)
    if (not np.isfinite(vx)) or vx < eps:
        return y
    y0 = y - np.nanmean(y)
    x0 = x - np.nanmean(x)
    beta = np.nanmean(y0 * x0) / (vx + eps)
    return y - beta * x


# -----------------------------
# Masks
# -----------------------------

def build_union_mask(mask_folder: Path) -> Tuple[np.ndarray, List[Path]]:
    mask_folder = Path(mask_folder)
    files = sorted(list(mask_folder.glob("*.tif*")))
    if not files:
        raise FileNotFoundError(f"No mask TIFFs found in: {mask_folder}")

    union = None
    for p in files:
        m = load_mask_tiff(p)
        if union is None:
            union = m.copy()
        else:
            if union.shape != m.shape:
                raise ValueError(f"Mask shape mismatch: {p.name} has {m.shape}, expected {union.shape}")
            union |= m

    return union.astype(bool), files


# -----------------------------
# F0 / ΔF/F on the fly
# -----------------------------

def compute_f0(
    raw: np.ndarray,
    mode: str,
    f0_frames: int,
    percentile: float,
    percentile_samples: int,
    percentile_stride: int,
    eps: float = 1e-6,
) -> np.ndarray:
    T = int(raw.shape[0])
    f0_frames = int(min(max(f0_frames, 1), T))

    if mode == "last_mean":
        block = raw[T - f0_frames:T].astype(np.float32, copy=False)
        f0 = np.nanmean(block, axis=0)
        return f0 + eps

    if mode == "last_median":
        block = raw[T - f0_frames:T].astype(np.float32, copy=False)
        f0 = np.nanmedian(block, axis=0)
        return f0 + eps

    if mode == "percentile":
        idx = np.arange(0, T, max(1, int(percentile_stride)))
        if idx.size > percentile_samples:
            step = max(1, idx.size // percentile_samples)
            idx = idx[::step][:percentile_samples]
        samp = raw[idx].astype(np.float32, copy=False)
        f0 = np.nanpercentile(samp, percentile, axis=0)
        return f0 + eps

    raise ValueError(f"Unknown f0_mode: {mode}")


def dff_from_raw_window(raw_window: np.ndarray, f0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rw = raw_window.astype(np.float32, copy=False)
    return (rw - f0) / (f0 + eps)


# -----------------------------
# Mean image + tissue mask
# -----------------------------

def mean_image_over_time(raw: np.ndarray, chunk_t: int = 50) -> np.ndarray:
    T = int(raw.shape[0])
    acc = None
    n = 0
    for t0 in range(0, T, chunk_t):
        t1 = min(T, t0 + chunk_t)
        blk = raw[t0:t1].astype(np.float32, copy=False)
        s = np.nansum(blk, axis=0)
        if acc is None:
            acc = s
        else:
            acc += s
        n += (t1 - t0)
    return acc / max(n, 1)


def build_tissue_mask_from_mean(mean_img: np.ndarray, tissue_q: float = 15.0, union_mask: Optional[np.ndarray] = None) -> np.ndarray:
    thr = np.nanpercentile(mean_img, tissue_q)
    tissue = mean_img > thr
    if union_mask is not None and union_mask.shape == tissue.shape:
        tissue |= union_mask
    tissue = ndimage.binary_opening(tissue, iterations=1)
    tissue = ndimage.binary_closing(tissue, iterations=1)
    return tissue.astype(bool)


# -----------------------------
# Traces without storing ΔF/F
# -----------------------------

def mask_trace_from_raw(raw: np.ndarray, mask: np.ndarray, f0: np.ndarray, chunk_t: int = 50) -> np.ndarray:
    T = int(raw.shape[0])
    idx = np.where(mask)
    if idx[0].size == 0:
        return np.zeros((T,), dtype=np.float32)

    out = np.empty((T,), dtype=np.float32)
    for t0 in range(0, T, chunk_t):
        t1 = min(T, t0 + chunk_t)
        dff = dff_from_raw_window(raw[t0:t1], f0)
        for k, t in enumerate(range(t0, t1)):
            out[t] = np.nanmean(dff[k][idx])
    return out


def global_tissue_trace_from_raw(raw: np.ndarray, tissue_mask: np.ndarray, f0: np.ndarray, chunk_t: int = 50) -> np.ndarray:
    T = int(raw.shape[0])
    idx = np.where(tissue_mask)
    out = np.empty((T,), dtype=np.float32)
    for t0 in range(0, T, chunk_t):
        t1 = min(T, t0 + chunk_t)
        dff = dff_from_raw_window(raw[t0:t1], f0)
        for k, t in enumerate(range(t0, t1)):
            out[t] = np.nanmean(dff[k][idx])
    return out


# -----------------------------
# Event detection
# -----------------------------

def detect_events_find_peaks(
    trace: np.ndarray,
    z_thresh: float,
    min_sep_frames: int,
    prominence_z: float,
    require_positive: bool = True,
) -> List[int]:
    z = safe_zscore(trace)
    z_use = z if require_positive else np.abs(z)

    peaks, _ = find_peaks(
        z_use,
        height=z_thresh,
        distance=max(1, int(min_sep_frames)),
        prominence=max(0.0, float(prominence_z)),
    )
    return list(map(int, peaks))


def event_window(peak: int, T: int, pre: int, post: int) -> Tuple[int, int]:
    t0 = max(0, peak - pre)
    t1 = min(T, peak + post + 1)
    return t0, t1


# -----------------------------
# Correlation maps (tile-based)
# -----------------------------

@dataclass
class CorrConfig:
    tile_y: int = 64
    tile_x: int = 64
    eps: float = 1e-8


def corrmap_for_window_raw(
    raw: np.ndarray,
    f0: np.ndarray,
    tissue_mask: np.ndarray,
    seed_trace_full_for_corr: np.ndarray,
    t0: int,
    t1: int,
    regress_global: bool,
    global_trace_full: Optional[np.ndarray],
    cfg: CorrConfig,
) -> np.ndarray:
    Tw = int(t1 - t0)
    if Tw <= 1:
        raise ValueError(f"Window too short: {t0}:{t1}")

    seed = seed_trace_full_for_corr[t0:t1].astype(np.float32, copy=False)
    seed_z = safe_zscore(seed, eps=cfg.eps)
    denom = float(max(Tw - 1, 1))

    if tissue_mask.ndim == 3:
        Z, Y, X = tissue_mask.shape
        corr = np.zeros((Z, Y, X), dtype=np.float32)

        for y0 in range(0, Y, cfg.tile_y):
            y1 = min(Y, y0 + cfg.tile_y)
            for x0 in range(0, X, cfg.tile_x):
                x1 = min(X, x0 + cfg.tile_x)

                tile_tissue = tissue_mask[:, y0:y1, x0:x1]
                if not tile_tissue.any():
                    continue

                raw_blk = raw[t0:t1, :, y0:y1, x0:x1].astype(np.float32, copy=False)
                f0_blk = f0[:, y0:y1, x0:x1].astype(np.float32, copy=False)

                dff_blk = (raw_blk - f0_blk[None, ...]) / (f0_blk[None, ...] + 1e-6)

                if regress_global and (global_trace_full is not None):
                    gv = global_trace_full[t0:t1].astype(np.float32, copy=False)
                    var_g = float(np.nanvar(gv)) + cfg.eps
                    if var_g > cfg.eps:
                        g0 = gv - np.nanmean(gv)
                        v0 = dff_blk - np.nanmean(dff_blk, axis=0, keepdims=True)
                        beta = np.nanmean(v0 * g0[:, None, None, None], axis=0) / var_g
                        dff_blk = dff_blk - beta[None, ...] * gv[:, None, None, None]

                mu = np.nanmean(dff_blk, axis=0, keepdims=True)
                sd = np.nanstd(dff_blk, axis=0, keepdims=True) + cfg.eps
                zvox = (dff_blk - mu) / sd

                r = np.nanmean(zvox * seed_z[:, None, None, None], axis=0) * (Tw / denom)
                r = np.where(tile_tissue, r, 0.0)
                corr[:, y0:y1, x0:x1] = r.astype(np.float32, copy=False)

        return corr

    if tissue_mask.ndim == 2:
        Y, X = tissue_mask.shape
        corr = np.zeros((Y, X), dtype=np.float32)

        for y0 in range(0, Y, cfg.tile_y):
            y1 = min(Y, y0 + cfg.tile_y)
            for x0 in range(0, X, cfg.tile_x):
                x1 = min(X, x0 + cfg.tile_x)

                tile_tissue = tissue_mask[y0:y1, x0:x1]
                if not tile_tissue.any():
                    continue

                raw_blk = raw[t0:t1, y0:y1, x0:x1].astype(np.float32, copy=False)
                f0_blk = f0[y0:y1, x0:x1].astype(np.float32, copy=False)

                dff_blk = (raw_blk - f0_blk[None, ...]) / (f0_blk[None, ...] + 1e-6)

                if regress_global and (global_trace_full is not None):
                    gv = global_trace_full[t0:t1].astype(np.float32, copy=False)
                    var_g = float(np.nanvar(gv)) + cfg.eps
                    if var_g > cfg.eps:
                        g0 = gv - np.nanmean(gv)
                        v0 = dff_blk - np.nanmean(dff_blk, axis=0, keepdims=True)
                        beta = np.nanmean(v0 * g0[:, None, None], axis=0) / var_g
                        dff_blk = dff_blk - beta[None, ...] * gv[:, None, None]

                mu = np.nanmean(dff_blk, axis=0, keepdims=True)
                sd = np.nanstd(dff_blk, axis=0, keepdims=True) + cfg.eps
                zvox = (dff_blk - mu) / sd

                r = np.nanmean(zvox * seed_z[:, None, None], axis=0) * (Tw / denom)
                r = np.where(tile_tissue, r, 0.0)
                corr[y0:y1, x0:x1] = r.astype(np.float32, copy=False)

        return corr

    raise ValueError(f"Unexpected tissue_mask ndim: {tissue_mask.ndim}")


# -----------------------------
# LINE candidates: event max|ΔF/F| → frangi → skeleton → long thin components
# -----------------------------

@dataclass
class LinesConfig:
    sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0)
    frangi_black_ridges: bool = False
    frangi_alpha: float = 0.5
    frangi_beta: float = 0.5
    frangi_gamma: float = 15.0

    ridge_q: float = 99.0          # threshold on frangi response (percentile)
    min_length_px: int = 60        # after skeletonize: minimum skeleton pixels
    dilate_radius_px: int = 2      # thicken skeleton into a mask

    # Build 3D from 2D lines using 3D maxabs map:
    voxel_q: float = 85.0          # within-window maxabs ΔF/F voxel percentile used as cutoff in 3D
    vertical_only: bool = True     # keep near-vertical components (optional)


def maxabs_dff_3d_window(raw: np.ndarray, f0: np.ndarray, t0: int, t1: int) -> np.ndarray:
    """Compute max_t |ΔF/F| per voxel for the window, without storing all frames."""
    Tw = int(t1 - t0)
    if Tw <= 0:
        raise ValueError("Empty window")
    # allocate
    maxabs = np.zeros(raw.shape[1:], dtype=np.float32)

    for t in range(t0, t1):
        dff = dff_from_raw_window(raw[t:t+1], f0)[0]  # (Z,Y,X) or (Y,X)
        maxabs = np.maximum(maxabs, np.abs(dff).astype(np.float32, copy=False))

    return maxabs


def lines_candidates_from_window(
    raw: np.ndarray,
    f0: np.ndarray,
    tissue_mask: np.ndarray,
    union_mask: Optional[np.ndarray],
    t0: int,
    t1: int,
    cfg: LinesConfig,
    corrmap: Optional[np.ndarray] = None,
    hybrid_r_min: float = 0.35,
) -> np.ndarray:
    """
    Returns labeled 3D candidates (same shape as union_mask/tissue_mask).
    """
    maxabs3d = maxabs_dff_3d_window(raw, f0, t0, t1)

    # 2D map for ridge detection: max over Z (if 3D)
    if maxabs3d.ndim == 3:
        activity2d = np.nanmax(maxabs3d, axis=0)
    else:
        activity2d = maxabs3d

    # Normalize robustly
    a = activity2d.astype(np.float32, copy=False)
    lo, hi = np.nanpercentile(a, 1.0), np.nanpercentile(a, 99.5)
    if not np.isfinite(hi - lo) or (hi - lo) <= 1e-8:
        return np.zeros_like(tissue_mask, dtype=np.int32)
    a = np.clip((a - lo) / (hi - lo), 0, 1)

    # Frangi ridge enhancement
    ridge = frangi(
        a,
        sigmas=cfg.sigmas,
        black_ridges=cfg.frangi_black_ridges,
        alpha=cfg.frangi_alpha,
        beta=cfg.frangi_beta,
        gamma=cfg.frangi_gamma,
    ).astype(np.float32, copy=False)

    thr = np.nanpercentile(ridge, cfg.ridge_q)
    bw2 = ridge >= thr

    # Clean + skeletonize
    bw2 = binary_opening(bw2, disk(1))
    bw2 = binary_closing(bw2, disk(1))
    bw2 = remove_small_objects(bw2, min_size=max(10, cfg.min_length_px // 2))

    sk = skeletonize(bw2)

    # Label skeleton components and filter by length + optional verticality
    lab2 = sk_label(sk)
    if lab2.max() == 0:
        return np.zeros_like(tissue_mask, dtype=np.int32)

    keep2 = np.zeros_like(lab2, dtype=bool)
    for rp in regionprops(lab2):
        length = int(rp.area)  # skeleton pixels
        if length < cfg.min_length_px:
            continue

        if cfg.vertical_only:
            # orientation in radians: 0 ~ horizontal, +/- pi/2 ~ vertical
            # keep if close to vertical
            ori = float(rp.orientation)
            if abs(abs(ori) - (np.pi / 2)) > (np.pi / 6):  # within 30° of vertical
                continue

        keep2[lab2 == rp.label] = True

    if not keep2.any():
        return np.zeros_like(tissue_mask, dtype=np.int32)

    # Thicken skeleton back into 2D mask
    if cfg.dilate_radius_px > 0:
        keep2 = ndimage.binary_dilation(keep2, iterations=cfg.dilate_radius_px)

    # Convert to 3D candidate masks using maxabs3d
    # 3D cutoff based on voxel_q within tissue (window-specific)
    if tissue_mask.ndim == 3:
        vox_pool = maxabs3d[tissue_mask]
    else:
        vox_pool = maxabs3d[tissue_mask]
    if vox_pool.size == 0:
        return np.zeros_like(tissue_mask, dtype=np.int32)

    vthr = np.nanpercentile(vox_pool, cfg.voxel_q)

    if tissue_mask.ndim == 3:
        # Expand keep2 across z where maxabs3d is strong
        keep3 = np.zeros_like(tissue_mask, dtype=bool)
        yy, xx = np.where(keep2)
        for y, x in zip(yy, xx):
            zmask = maxabs3d[:, y, x] >= vthr
            if zmask.any():
                keep3[:, y, x] = zmask
    else:
        keep3 = keep2 & (maxabs3d >= vthr)

    # enforce tissue / union exclusion
    keep3 &= tissue_mask
    if union_mask is not None:
        keep3 &= ~union_mask

    # HYBRID correlation check (optional)
    if corrmap is not None:
        keep3 &= (corrmap >= hybrid_r_min)

    # Label 3D candidates
    lab3, _ = ndimage.label(keep3)
    return lab3.astype(np.int32)


# -----------------------------
# Corr-threshold candidates (old)
# -----------------------------

@dataclass
class CandidateConfig:
    r_thresh: float = 0.55
    min_voxels: int = 300
    min_zspan: int = 2
    dilate_iters: int = 0


def extract_candidates_corr(corrmap: np.ndarray, union_mask: Optional[np.ndarray], tissue_mask: np.ndarray, cfg: CandidateConfig) -> np.ndarray:
    if corrmap.shape != tissue_mask.shape:
        raise ValueError(f"corrmap shape {corrmap.shape} != tissue_mask shape {tissue_mask.shape}")
    if union_mask is not None and union_mask.shape != corrmap.shape:
        raise ValueError(f"union_mask shape {union_mask.shape} != corrmap shape {corrmap.shape}")

    work = corrmap.copy()
    work[~tissue_mask] = 0.0
    if union_mask is not None:
        work[union_mask] = 0.0

    bw = work >= cfg.r_thresh
    if cfg.dilate_iters > 0:
        bw = ndimage.binary_dilation(bw, iterations=cfg.dilate_iters)

    labeled, n = ndimage.label(bw)
    if n == 0:
        return labeled.astype(np.int32)

    slices = ndimage.find_objects(labeled)
    for i, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        comp = (labeled[slc] == i)
        vox = int(comp.sum())
        if vox < cfg.min_voxels:
            labeled[slc][comp] = 0
            continue
        if labeled.ndim == 3:
            zspan = slc[0].stop - slc[0].start
            if zspan < cfg.min_zspan:
                labeled[slc][comp] = 0
                continue

    labeled2, _ = ndimage.label(labeled > 0)
    return labeled2.astype(np.int32)


# -----------------------------
# Previews
# -----------------------------

def to_2d_mip_over_z(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return np.nanmax(x, axis=0)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 2D/3D, got ndim={x.ndim}")


def window_background_2d(raw_w: np.ndarray, f0: np.ndarray, mode: str) -> np.ndarray:
    if mode == "win_raw_mip":
        if raw_w.ndim == 4:
            raw2 = np.nanmax(raw_w, axis=1)  # (Tw,Y,X)
        else:
            raw2 = raw_w
        return np.nanmean(raw2.astype(np.float32), axis=0)

    dff_w = dff_from_raw_window(raw_w, f0)
    if dff_w.ndim == 4:
        dff2 = np.nanmax(dff_w, axis=1)  # (Tw,Y,X)
    else:
        dff2 = dff_w

    if mode == "win_dff_maxabs":
        return np.nanmax(np.abs(dff2.astype(np.float32)), axis=0)

    if mode == "win_dff_max":
        return np.nanmax(dff2.astype(np.float32), axis=0)

    raise ValueError(f"Unknown preview_bg: {mode}")


def plot_preview(
    out_png: Path,
    bg2d: np.ndarray,
    seed_mask: np.ndarray,
    cand_mask: Optional[np.ndarray],
    seed_trace_evt: np.ndarray,
    t0: int,
    t1: int,
    cand_trace_window: Optional[np.ndarray],
    title: str,
    global_trace: Optional[np.ndarray] = None,
    peak_frame: Optional[int] = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    seed_m = to_2d_mip_over_z(seed_mask.astype(np.uint8))
    cand_m = to_2d_mip_over_z(cand_mask.astype(np.uint8)) if cand_mask is not None else None

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(bg2d, cmap="gray")
    ax1.set_title("Background (preview)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(bg2d, cmap="gray")
    ax2.contour(seed_m > 0, levels=[0.5])
    if cand_m is not None:
        ax2.contour(cand_m > 0, levels=[0.5])
    ax2.set_title("Seed (and candidate) overlay")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(seed_trace_evt, label="seed (full)")
    if global_trace is not None:
        ax3.plot(global_trace, label="global (full)", linewidth=1)

    ax3.axvspan(t0, t1 - 1, alpha=0.2)
    if peak_frame is not None:
        ax3.axvline(peak_frame, linestyle="--", linewidth=1)

    if cand_trace_window is not None:
        tmp = np.full_like(seed_trace_evt, np.nan, dtype=np.float32)
        tmp[t0:t1] = cand_trace_window.astype(np.float32, copy=False)
        ax3.plot(tmp, label="candidate (window)")

    ax3.set_title("Traces (window shaded)")
    ax3.legend(loc="best")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("ΔF/F")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def parse_sigmas(s: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return (1.0, 2.0, 3.0)
    return tuple(float(p) for p in parts)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--raw", type=str, required=True)
    ap.add_argument("--masks", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--y_crop", type=int, default=0)
    ap.add_argument("--x_crop", type=int, default=0)

    ap.add_argument("--f0_mode", type=str, default="last_mean", choices=["last_mean", "last_median", "percentile"])
    ap.add_argument("--f0_frames", type=int, default=30)
    ap.add_argument("--f0_percentile", type=float, default=20.0)
    ap.add_argument("--f0_percentile_samples", type=int, default=120)
    ap.add_argument("--f0_percentile_stride", type=int, default=10)

    ap.add_argument("--tissue_mask", type=str, default=None)
    ap.add_argument("--tissue_q", type=float, default=15.0)

    ap.add_argument("--mode", type=str, default="event", choices=["event", "mask", "both"])
    ap.add_argument("--regress_global", action="store_true")

    # Event windows
    ap.add_argument("--window_source", type=str, default="seed", choices=["seed", "global"])
    ap.add_argument("--event_z", type=float, default=2.5)
    ap.add_argument("--event_prom", type=float, default=0.75)
    ap.add_argument("--event_min_sep", type=int, default=8)
    ap.add_argument("--event_pre", type=int, default=15)
    ap.add_argument("--event_post", type=int, default=30)
    ap.add_argument("--max_events", type=int, default=15)
    ap.add_argument("--event_smooth_sigma", type=float, default=1.0)

    # Corr tiling
    ap.add_argument("--tile_y", type=int, default=64)
    ap.add_argument("--tile_x", type=int, default=64)

    # Candidate mode
    ap.add_argument("--candidate_mode", type=str, default="corr", choices=["corr", "lines", "hybrid"])
    ap.add_argument("--hybrid_r_min", type=float, default=0.35)

    # Corr candidate params
    ap.add_argument("--r_thresh", type=float, default=0.55)
    ap.add_argument("--min_voxels", type=int, default=300)
    ap.add_argument("--min_zspan", type=int, default=2)
    ap.add_argument("--dilate", type=int, default=0)

    # Lines candidate params
    ap.add_argument("--lines_sigmas", type=str, default="1,2,3")
    ap.add_argument("--lines_ridge_q", type=float, default=99.0)
    ap.add_argument("--lines_min_length", type=int, default=60)
    ap.add_argument("--lines_dilate_radius", type=int, default=2)
    ap.add_argument("--lines_voxel_q", type=float, default=85.0)
    ap.add_argument("--lines_keep_vertical", action="store_true",
                    help="Keep near-vertical line components (recommended for trunks).")

    ap.add_argument("--chunk_t", type=int, default=50)

    ap.add_argument("--preview_bg", type=str, default="win_dff_maxabs",
                    choices=["mean", "win_raw_mip", "win_dff_maxabs", "win_dff_max"])

    args = ap.parse_args()

    raw_path = Path(args.raw)
    masks_folder = Path(args.masks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Load] RAW: {raw_path}")
    raw = load_raw(raw_path, mmap=True)
    print(f"  RAW shape (before crop): {tuple(raw.shape)}")

    if args.y_crop or args.x_crop:
        if raw.ndim == 4:
            raw = raw[:, :, args.y_crop:, args.x_crop:]
        elif raw.ndim == 3:
            raw = raw[:, args.y_crop:, args.x_crop:]
        else:
            raise ValueError(f"Unexpected raw ndim: {raw.ndim}")
    print(f"  RAW shape (after crop):  {tuple(raw.shape)}")

    T = int(raw.shape[0])

    print(f"[Load] curated masks from: {masks_folder}")
    union_mask, seed_files = build_union_mask(masks_folder)

    raw_spatial = raw.shape[1:]
    if union_mask.shape != raw_spatial:
        raise ValueError(
            "Mask/RAW shape mismatch.\n"
            f"  union_mask shape: {union_mask.shape}\n"
            f"  raw spatial shape: {raw_spatial}\n"
            "Fix by setting --y_crop/--x_crop so RAW matches masks."
        )

    print(f"[Compute] F0 mode={args.f0_mode}")
    f0 = compute_f0(
        raw=raw,
        mode=args.f0_mode,
        f0_frames=args.f0_frames,
        percentile=args.f0_percentile,
        percentile_samples=args.f0_percentile_samples,
        percentile_stride=args.f0_percentile_stride,
    )
    save_tiff(out / "masks" / "f0.tif", f0.astype(np.float32))

    print("[Compute] mean image over time (QC)")
    mean_img = mean_image_over_time(raw, chunk_t=args.chunk_t)
    save_tiff(out / "qc" / "mean_image.tif", mean_img.astype(np.float32))

    if args.tissue_mask:
        tissue = load_mask_tiff(Path(args.tissue_mask))
        if tissue.shape != raw_spatial:
            raise ValueError(f"Provided tissue_mask shape {tissue.shape} != raw spatial {raw_spatial}")
        print("[Load] tissue mask from file")
    else:
        tissue = build_tissue_mask_from_mean(mean_img, tissue_q=args.tissue_q, union_mask=union_mask)

    save_tiff(out / "masks" / "union_mask.tif", union_mask.astype(np.uint8))
    save_tiff(out / "masks" / "tissue_mask.tif", tissue.astype(np.uint8))

    # Global trace when needed
    global_trace_evt = None
    global_trace_corr = None
    if args.window_source == "global" or args.regress_global:
        print("[Compute] global tissue ΔF/F trace")
        global_trace_evt = global_tissue_trace_from_raw(raw, tissue, f0, chunk_t=args.chunk_t)
        if args.event_smooth_sigma and args.event_smooth_sigma > 0:
            global_trace_evt = ndimage.gaussian_filter1d(global_trace_evt.astype(np.float32), args.event_smooth_sigma)
        global_trace_corr = global_trace_evt.copy()

    # Global windows
    global_windows: List[Tuple[str, int, int, Optional[int]]] = []
    if args.mode in ("event", "both") and args.window_source == "global":
        peaks = detect_events_find_peaks(
            global_trace_evt,
            z_thresh=args.event_z,
            min_sep_frames=args.event_min_sep,
            prominence_z=args.event_prom,
            require_positive=True,
        )
        z = safe_zscore(global_trace_evt)
        peaks = sorted(peaks, key=lambda i: float(z[i]), reverse=True)[: args.max_events]
        peaks.sort()
        print(f"[Global windows] peaks={len(peaks)}")
        for ei, pk in enumerate(peaks, start=1):
            t0, t1 = event_window(pk, T, args.event_pre, args.event_post)
            global_windows.append((f"event_{ei:03d}", t0, t1, pk))

    cand_cfg = CandidateConfig(
        r_thresh=args.r_thresh,
        min_voxels=args.min_voxels,
        min_zspan=args.min_zspan,
        dilate_iters=args.dilate,
    )
    corr_cfg = CorrConfig(tile_y=args.tile_y, tile_x=args.tile_x)

    lines_cfg = LinesConfig(
        sigmas=parse_sigmas(args.lines_sigmas),
        ridge_q=args.lines_ridge_q,
        min_length_px=args.lines_min_length,
        dilate_radius_px=args.lines_dilate_radius,
        voxel_q=args.lines_voxel_q,
        vertical_only=bool(args.lines_keep_vertical),
    )

    rows: List[Dict] = []
    row_id = 0

    print(f"[Seeds] {len(seed_files)} curated masks")
    for si, seed_path in enumerate(seed_files, start=1):
        seed_name = seed_path.stem
        print(f"\n[Seed {si}/{len(seed_files)}] {seed_name}")

        seed_mask = load_mask_tiff(seed_path)

        seed_trace_evt = mask_trace_from_raw(raw, seed_mask, f0, chunk_t=args.chunk_t)
        if args.event_smooth_sigma and args.event_smooth_sigma > 0:
            seed_trace_evt = ndimage.gaussian_filter1d(seed_trace_evt.astype(np.float32), args.event_smooth_sigma)

        seed_trace_corr = seed_trace_evt.astype(np.float32, copy=True)
        if args.regress_global and global_trace_corr is not None:
            seed_trace_corr = regress_out_1d(seed_trace_corr, global_trace_corr)

        windows: List[Tuple[str, int, int, Optional[int]]] = []
        if args.mode in ("mask", "both"):
            windows.append(("mask_full", 0, T, None))

        if args.mode in ("event", "both"):
            if args.window_source == "global":
                windows.extend(global_windows)
                if si == 1:
                    print(f"  Using GLOBAL windows: {len(global_windows)}")
            else:
                peaks = detect_events_find_peaks(
                    seed_trace_evt,
                    z_thresh=args.event_z,
                    min_sep_frames=args.event_min_sep,
                    prominence_z=args.event_prom,
                    require_positive=True,
                )
                z = safe_zscore(seed_trace_evt)
                peaks = sorted(peaks, key=lambda i: float(z[i]), reverse=True)[: args.max_events]
                peaks.sort()
                print(f"  Detected events (seed): {len(peaks)} (cap={args.max_events})")
                for ei, pk in enumerate(peaks, start=1):
                    t0, t1 = event_window(pk, T, args.event_pre, args.event_post)
                    windows.append((f"event_{ei:03d}", t0, t1, pk))

        for win_name, t0, t1, pk in windows:
            print(f"  [Corr] {win_name}: frames {t0}:{t1}")

            corrmap = corrmap_for_window_raw(
                raw=raw,
                f0=f0,
                tissue_mask=tissue,
                seed_trace_full_for_corr=seed_trace_corr,
                t0=t0,
                t1=t1,
                regress_global=args.regress_global,
                global_trace_full=global_trace_corr,
                cfg=corr_cfg,
            )

            corr_dir = out / "corrmaps" / seed_name
            save_tiff(corr_dir / f"{win_name}_corrmap.tif", corrmap.astype(np.float16))

            # candidate extraction
            if args.candidate_mode == "corr":
                labeled = extract_candidates_corr(corrmap, union_mask=union_mask, tissue_mask=tissue, cfg=cand_cfg)
            elif args.candidate_mode == "lines":
                labeled = lines_candidates_from_window(
                    raw=raw, f0=f0, tissue_mask=tissue, union_mask=union_mask,
                    t0=t0, t1=t1, cfg=lines_cfg,
                    corrmap=None, hybrid_r_min=args.hybrid_r_min
                )
            else:  # hybrid
                labeled = lines_candidates_from_window(
                    raw=raw, f0=f0, tissue_mask=tissue, union_mask=union_mask,
                    t0=t0, t1=t1, cfg=lines_cfg,
                    corrmap=corrmap, hybrid_r_min=args.hybrid_r_min
                )

            cand_dir = out / "candidates" / seed_name / win_name
            save_tiff(cand_dir / "candidates_labeled.tif", labeled.astype(np.int32))

            n_cand = int(labeled.max())
            print(f"  [Cands] {n_cand} (mode={args.candidate_mode})")

            # Background image for previews
            if args.preview_bg == "mean":
                bg2d = to_2d_mip_over_z(mean_img.astype(np.float32))
            else:
                bg2d = window_background_2d(raw[t0:t1].astype(np.float32, copy=False), f0, args.preview_bg)

            if n_cand == 0:
                plot_preview(
                    out_png=out / "previews" / seed_name / win_name / "seed_only.png",
                    bg2d=bg2d,
                    seed_mask=seed_mask,
                    cand_mask=None,
                    seed_trace_evt=seed_trace_evt,
                    t0=t0,
                    t1=t1,
                    cand_trace_window=None,
                    title=f"{seed_name} | {win_name} | seed only",
                    global_trace=global_trace_evt,
                    peak_frame=pk,
                )
                continue

            for ci in range(1, n_cand + 1):
                cand_mask = (labeled == ci)
                if not cand_mask.any():
                    continue

                # Candidate trace only in window
                idx = np.where(cand_mask)
                dff_w = dff_from_raw_window(raw[t0:t1].astype(np.float32, copy=False), f0)
                cand_trace_w = np.array([np.nanmean(dff_w[k][idx]) for k in range(dff_w.shape[0])], dtype=np.float32)
                if args.regress_global and global_trace_corr is not None:
                    cand_trace_w = regress_out_1d(cand_trace_w, global_trace_corr[t0:t1])

                save_tiff(cand_dir / f"cand_{ci:03d}.tif", cand_mask.astype(np.uint8))

                prev_path = out / "previews" / seed_name / win_name / f"cand_{ci:03d}.png"
                plot_preview(
                    out_png=prev_path,
                    bg2d=bg2d,
                    seed_mask=seed_mask,
                    cand_mask=cand_mask,
                    seed_trace_evt=seed_trace_evt,
                    t0=t0,
                    t1=t1,
                    cand_trace_window=cand_trace_w,
                    title=f"{seed_name} | {win_name} | cand {ci:03d}",
                    global_trace=global_trace_evt,
                    peak_frame=pk,
                )

                r_vals = corrmap[cand_mask]
                r_mean = float(np.nanmean(r_vals)) if r_vals.size else float("nan")
                r_max = float(np.nanmax(r_vals)) if r_vals.size else float("nan")
                vox = int(cand_mask.sum())

                seed_coords = np.argwhere(seed_mask)
                cand_coords = np.argwhere(cand_mask)
                dist = None
                if seed_coords.size and cand_coords.size:
                    dist = float(np.linalg.norm(seed_coords.mean(axis=0) - cand_coords.mean(axis=0)))

                row_id += 1
                rows.append({
                    "row_id": row_id,
                    "seed": seed_name,
                    "window": win_name,
                    "peak_frame": pk,
                    "t0": t0,
                    "t1": t1,
                    "cand_id": f"{ci:03d}",
                    "voxels": vox,
                    "r_mean": r_mean,
                    "r_max": r_max,
                    "dist_to_seed_vox": dist,
                    "cand_mask_path": str((cand_dir / f"cand_{ci:03d}.tif").resolve()),
                    "preview_path": str(prev_path.resolve()),
                })

    csv_path = out / "candidates.csv"
    print(f"\n[Save] {csv_path}")
    fieldnames = [
        "row_id", "seed", "window", "peak_frame", "t0", "t1", "cand_id",
        "voxels", "r_mean", "r_max", "dist_to_seed_vox", "cand_mask_path", "preview_path"
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    print("[Done]")


if __name__ == "__main__":
    main()
    
