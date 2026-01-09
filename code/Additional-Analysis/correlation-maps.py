#!/usr/bin/env python3
"""
correlation-maps.py  (Module 4B — RAW-only, Y/X crop-aware)

Seeded voxel correlation maps + candidate mask discovery (post-Module 4).

You do NOT need a precomputed ΔF/F file.
This script:
  - loads RAW (T,Z,Y,X) from TIFF (or NPY)
  - applies optional cropping (e.g., Y_CROP=3 => raw[:, :, 3:, :])
  - computes a single F0 volume (default: mean of last N frames)
  - computes ΔF/F mask traces without storing full ΔF/F
  - detects events on each seed mask trace (optional)
  - computes voxel-wise correlation maps per window (tile-based)
  - extracts candidate correlated clusters outside curated union masks
  - saves corrmaps, candidates, previews, and candidates.csv

Assumptions
-----------
- Curated masks are TIFF files in a folder. Each is either binary or a labelmap; >0 is treated as mask.
- Masks are in the same coordinate system as your *cropped* raw.
  If your pipeline uses Y_CROP=3, run with: --y_crop 3

Outputs
-------
<out>/
  masks/ union_mask.tif, tissue_mask.tif, f0.tif
  qc/ mean_image.tif
  corrmaps/<seed>/<window>_corrmap.tif
  candidates/<seed>/<window>/candidates_labeled.tif + cand_###.tif
  previews/<seed>/<window>/cand_###.png (and seed_only.png)
  candidates.csv
  run_config.json
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


# -----------------------------
# I/O helpers
# -----------------------------

def load_raw(path: Path, mmap: bool = True) -> np.ndarray:
    """
    Load RAW stack.
    Supports:
      - .npy (recommended for mmap)
      - .tif/.tiff (tries memmap; falls back to imread)

    Expected shapes:
      - (T, Z, Y, X) preferred
      - (T, Y, X) allowed (if already MIP)
    """
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


def regress_out(y: np.ndarray, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Regress x out of y (1D):
      y_res = y - beta*x, beta = cov(y,x)/var(x)
    """
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
# Masks / labelmaps
# -----------------------------

def build_union_mask(mask_folder: Path, allow_list: Optional[List[str]] = None) -> Tuple[np.ndarray, List[Path]]:
    """
    Union of all masks in folder.
    Each TIFF must be (Z,Y,X) or (Y,X). >0 treated as mask.
    Returns (union_mask, mask_paths_used).
    """
    mask_folder = Path(mask_folder)
    files = sorted(list(mask_folder.glob("*.tif*")))

    if allow_list:
        allow = set(allow_list)
        files = [p for p in files if p.name in allow or p.stem in allow]

    if not files:
        raise FileNotFoundError(f"No mask TIFFs found in: {mask_folder}")

    union = None
    for p in files:
        m = load_mask_tiff(p)
        if union is None:
            union = m.copy()
        else:
            if union.shape != m.shape:
                raise ValueError(f"Shape mismatch: {p.name} has {m.shape}, expected {union.shape}")
            union |= m

    return union.astype(bool), files


# -----------------------------
# F0 and ΔF/F-on-the-fly
# -----------------------------

def compute_f0(
    raw: np.ndarray,
    mode: str,
    f0_frames: int,
    percentile: float,
    percentile_samples: int,
    percentile_stride: int,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute a single F0 volume.

    Modes:
      - last_mean: mean of last f0_frames frames
      - last_median: median of last f0_frames frames
      - percentile: approximate percentile over time using sampled frames

    Returns F0 with shape spatial dims (Z,Y,X) or (Y,X).
    """
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
    """
    ΔF/F for a time window:
      dff = (raw - f0)/(f0+eps)
    raw_window: (Tw, ...) same spatial dims as f0
    """
    rw = raw_window.astype(np.float32, copy=False)
    return (rw - f0) / (f0 + eps)


# -----------------------------
# Tissue mask & mean image
# -----------------------------

def mean_image_over_time(raw: np.ndarray, chunk_t: int = 50) -> np.ndarray:
    """
    Compute mean image over time without loading everything at once.
    Returns mean over T with shape spatial dims.
    """
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
# Traces (mask + global) without storing full ΔF/F
# -----------------------------

def mask_trace_from_raw(
    raw: np.ndarray,
    mask: np.ndarray,
    f0: np.ndarray,
    chunk_t: int = 50
) -> np.ndarray:
    """
    Compute ΔF/F trace for a mask over all T without storing full ΔF/F.
    """
    T = int(raw.shape[0])
    idx = np.where(mask)
    if idx[0].size == 0:
        return np.zeros((T,), dtype=np.float32)

    out = np.empty((T,), dtype=np.float32)

    for t0 in range(0, T, chunk_t):
        t1 = min(T, t0 + chunk_t)
        rw = raw[t0:t1].astype(np.float32, copy=False)
        dff = dff_from_raw_window(rw, f0)
        for k, t in enumerate(range(t0, t1)):
            out[t] = np.nanmean(dff[k][idx])

    return out


def global_tissue_trace_from_raw(
    raw: np.ndarray,
    tissue_mask: np.ndarray,
    f0: np.ndarray,
    chunk_t: int = 50
) -> np.ndarray:
    """
    global(t) = mean ΔF/F across tissue voxels per frame
    """
    T = int(raw.shape[0])
    idx = np.where(tissue_mask)
    out = np.empty((T,), dtype=np.float32)

    for t0 in range(0, T, chunk_t):
        t1 = min(T, t0 + chunk_t)
        rw = raw[t0:t1].astype(np.float32, copy=False)
        dff = dff_from_raw_window(rw, f0)
        for k, t in enumerate(range(t0, t1)):
            out[t] = np.nanmean(dff[k][idx])

    return out


# -----------------------------
# Event detection
# -----------------------------

def detect_events_simple(trace: np.ndarray, z_thresh: float = 2.5, min_sep_frames: int = 8) -> List[int]:
    z = safe_zscore(trace)
    peaks = np.where((z[1:-1] > z[:-2]) & (z[1:-1] > z[2:]) & (z[1:-1] >= z_thresh))[0] + 1
    if peaks.size == 0:
        return []

    peaks = list(peaks)
    peaks.sort(key=lambda i: z[i], reverse=True)

    kept: List[int] = []
    for p in peaks:
        if all(abs(p - k) >= min_sep_frames for k in kept):
            kept.append(p)

    kept.sort()
    return kept


def event_window(peak: int, T: int, pre: int, post: int) -> Tuple[int, int]:
    t0 = max(0, peak - pre)
    t1 = min(T, peak + post + 1)
    return t0, t1


# -----------------------------
# Correlation maps (tile-based, RAW->ΔF/F window)
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
    seed_trace_full: np.ndarray,   # ΔF/F seed trace over full run
    t0: int,
    t1: int,
    regress_global: bool,
    global_trace_full: Optional[np.ndarray],
    cfg: CorrConfig,
) -> np.ndarray:
    """
    Compute voxel-wise corrmap for frames [t0:t1) without storing full-run ΔF/F.
    Output shape matches tissue_mask (Z,Y,X) or (Y,X).
    """
    Tw = int(t1 - t0)
    if Tw <= 1:
        raise ValueError(f"Window too short for correlation: {t0}:{t1}")

    seed = seed_trace_full[t0:t1].astype(np.float32, copy=False)

    if regress_global and global_trace_full is not None:
        g = global_trace_full[t0:t1].astype(np.float32, copy=False)
        seed = regress_out(seed, g)

    seed_z = safe_zscore(seed, eps=cfg.eps)  # (Tw,)
    denom = float(max(Tw - 1, 1))

    if tissue_mask.ndim == 3:
        Z, Y, X = tissue_mask.shape
        corr = np.zeros((Z, Y, X), dtype=np.float32)

        for y0 in range(0, Y, cfg.tile_y):
            y1_ = min(Y, y0 + cfg.tile_y)
            for x0 in range(0, X, cfg.tile_x):
                x1_ = min(X, x0 + cfg.tile_x)

                tile_tissue = tissue_mask[:, y0:y1_, x0:x1_]
                if not tile_tissue.any():
                    continue

                raw_blk = raw[t0:t1, :, y0:y1_, x0:x1_].astype(np.float32, copy=False)
                f0_blk = f0[:, y0:y1_, x0:x1_].astype(np.float32, copy=False)

                dff_blk = (raw_blk - f0_blk[None, ...]) / (f0_blk[None, ...] + 1e-6)

                if regress_global and global_trace_full is not None:
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
                corr[:, y0:y1_, x0:x1_] = r.astype(np.float32, copy=False)

        return corr

    if tissue_mask.ndim == 2:
        Y, X = tissue_mask.shape
        corr = np.zeros((Y, X), dtype=np.float32)

        for y0 in range(0, Y, cfg.tile_y):
            y1_ = min(Y, y0 + cfg.tile_y)
            for x0 in range(0, X, cfg.tile_x):
                x1_ = min(X, x0 + cfg.tile_x)

                tile_tissue = tissue_mask[y0:y1_, x0:x1_]
                if not tile_tissue.any():
                    continue

                raw_blk = raw[t0:t1, y0:y1_, x0:x1_].astype(np.float32, copy=False)
                f0_blk = f0[y0:y1_, x0:x1_].astype(np.float32, copy=False)

                dff_blk = (raw_blk - f0_blk[None, ...]) / (f0_blk[None, ...] + 1e-6)

                if regress_global and global_trace_full is not None:
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
                corr[y0:y1_, x0:x1_] = r.astype(np.float32, copy=False)

        return corr

    raise ValueError(f"Unexpected tissue_mask ndim: {tissue_mask.ndim}")


# -----------------------------
# Candidate extraction
# -----------------------------

@dataclass
class CandidateConfig:
    r_thresh: float = 0.55
    min_voxels: int = 300
    min_zspan: int = 2
    dilate_iters: int = 0


def extract_candidates(
    corrmap: np.ndarray,
    union_mask: Optional[np.ndarray],
    tissue_mask: np.ndarray,
    cfg: CandidateConfig
) -> np.ndarray:
    """
    corrmap -> labeled candidates (int32) with filtering.
    Excludes union_mask and outside tissue_mask.
    """
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

def mip(arr: np.ndarray) -> np.ndarray:
    return np.nanmax(arr, axis=0) if arr.ndim == 3 else arr


def plot_preview(
    out_png: Path,
    mean_img: np.ndarray,
    seed_mask: np.ndarray,
    cand_mask: Optional[np.ndarray],
    seed_trace: np.ndarray,
    cand_trace_window: Optional[np.ndarray],
    t0: int,
    t1: int,
    title: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    mean_m = mip(mean_img)
    seed_m = mip(seed_mask.astype(np.uint8))
    cand_m = mip(cand_mask.astype(np.uint8)) if cand_mask is not None else None

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(mean_m, cmap="gray")
    ax1.set_title("Mean image (MIP)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(mean_m, cmap="gray")
    ax2.contour(seed_m > 0, levels=[0.5])
    if cand_m is not None:
        ax2.contour(cand_m > 0, levels=[0.5])
    ax2.set_title("Seed (and candidate) overlay")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(seed_trace, label="seed")
    ax3.axvspan(t0, t1 - 1, alpha=0.2)

    if cand_trace_window is not None:
        tmp = np.full_like(seed_trace, np.nan, dtype=np.float32)
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

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--raw", type=str, required=True, help="RAW stack (tif/tiff or npy), shape (T,Z,Y,X)")
    ap.add_argument("--masks", type=str, required=True, help="Folder with curated mask TIFFs (same coords as cropped raw)")
    ap.add_argument("--out", type=str, required=True, help="Output folder")

    # Cropping to match your pipeline coordinates (e.g., Y_CROP=3)
    ap.add_argument("--y_crop", type=int, default=0, help="Crop from top of Y: raw[:, :, y_crop:, :]")
    ap.add_argument("--x_crop", type=int, default=0, help="Crop from left of X: raw[:, :, :, x_crop:]")

    # F0 config
    ap.add_argument("--f0_mode", type=str, default="last_mean",
                    choices=["last_mean", "last_median", "percentile"])
    ap.add_argument("--f0_frames", type=int, default=30, help="Used for last_mean/last_median")
    ap.add_argument("--f0_percentile", type=float, default=20.0, help="Used for percentile mode")
    ap.add_argument("--f0_percentile_samples", type=int, default=120, help="Frames to sample for percentile mode")
    ap.add_argument("--f0_percentile_stride", type=int, default=10, help="Stride for sampling in percentile mode")

    # Tissue mask
    ap.add_argument("--tissue_mask", type=str, default=None, help="Optional tissue mask TIFF (same coords as cropped raw)")
    ap.add_argument("--tissue_q", type=float, default=15.0, help="Percentile threshold for tissue mask from mean image")

    # Mode
    ap.add_argument("--mode", type=str, default="event", choices=["event", "mask", "both"])
    ap.add_argument("--regress_global", action="store_true", help="Regress out global tissue ΔF/F before correlation")

    # Event detection
    ap.add_argument("--event_z", type=float, default=2.5)
    ap.add_argument("--event_min_sep", type=int, default=8)
    ap.add_argument("--event_pre", type=int, default=10)
    ap.add_argument("--event_post", type=int, default=20)
    ap.add_argument("--max_events_per_seed", type=int, default=15)

    # Corr tiling
    ap.add_argument("--tile_y", type=int, default=64)
    ap.add_argument("--tile_x", type=int, default=64)

    # Candidate extraction
    ap.add_argument("--r_thresh", type=float, default=0.55)
    ap.add_argument("--min_voxels", type=int, default=300)
    ap.add_argument("--min_zspan", type=int, default=2)
    ap.add_argument("--dilate", type=int, default=0)

    # Runtime / chunking
    ap.add_argument("--chunk_t", type=int, default=50, help="Time chunk for trace computations / mean image")

    args = ap.parse_args()

    raw_path = Path(args.raw)
    masks_folder = Path(args.masks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Load] RAW: {raw_path}")
    raw = load_raw(raw_path, mmap=True)
    print(f"  RAW shape (before crop): {tuple(raw.shape)}")

    # Apply cropping immediately so everything else matches mask coordinates
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

    # Safety check: masks must match cropped raw spatial shape
    raw_spatial = raw.shape[1:]  # for 4D: (Z,Y,X), for 3D: (Y,X)
    if union_mask.shape != raw_spatial:
        raise ValueError(
            "Mask/RAW shape mismatch.\n"
            f"  union_mask shape: {union_mask.shape}\n"
            f"  raw spatial shape: {raw_spatial}\n"
            "Fix by setting --y_crop/--x_crop so RAW matches masks."
        )

    # F0
    print(f"[Compute] F0 mode={args.f0_mode}")
    f0 = compute_f0(
        raw=raw,
        mode=args.f0_mode,
        f0_frames=args.f0_frames,
        percentile=args.f0_percentile,
        percentile_samples=args.f0_percentile_samples,
        percentile_stride=args.f0_percentile_stride,
    )
    save_tiff(out / "masks" / "f0.tif", f0.astype(np.float32), compress=True)

    # Mean image (for previews and tissue mask)
    print("[Compute] mean image over time (for tissue mask + previews)")
    mean_img = mean_image_over_time(raw, chunk_t=args.chunk_t)
    save_tiff(out / "qc" / "mean_image.tif", mean_img.astype(np.float32), compress=True)

    # Tissue mask
    if args.tissue_mask:
        tissue = load_mask_tiff(Path(args.tissue_mask))
        if tissue.shape != raw_spatial:
            raise ValueError(f"Provided tissue_mask shape {tissue.shape} != raw spatial {raw_spatial}")
        print("[Load] tissue mask from file")
    else:
        tissue = build_tissue_mask_from_mean(mean_img, tissue_q=args.tissue_q, union_mask=union_mask)

    save_tiff(out / "masks" / "union_mask.tif", union_mask.astype(np.uint8), compress=True)
    save_tiff(out / "masks" / "tissue_mask.tif", tissue.astype(np.uint8), compress=True)

    # Global trace (optional)
    global_trace = None
    if args.regress_global:
        print("[Compute] global tissue ΔF/F trace")
        global_trace = global_tissue_trace_from_raw(raw, tissue, f0, chunk_t=args.chunk_t)

    cand_cfg = CandidateConfig(
        r_thresh=args.r_thresh,
        min_voxels=args.min_voxels,
        min_zspan=args.min_zspan,
        dilate_iters=args.dilate,
    )
    corr_cfg = CorrConfig(tile_y=args.tile_y, tile_x=args.tile_x)

    rows: List[Dict] = []
    row_id = 0

    print(f"[Seeds] {len(seed_files)} curated masks")

    for si, seed_path in enumerate(seed_files, start=1):
        seed_name = seed_path.stem
        print(f"\n[Seed {si}/{len(seed_files)}] {seed_name}")

        seed_mask = load_mask_tiff(seed_path)

        # Seed trace (ΔF/F) over full run
        seed_trace = mask_trace_from_raw(raw, seed_mask, f0, chunk_t=args.chunk_t)
        if args.regress_global and global_trace is not None:
            seed_trace = regress_out(seed_trace, global_trace)

        # Determine windows
        windows: List[Tuple[str, int, int, Optional[int]]] = []

        if args.mode in ("mask", "both"):
            windows.append(("mask_full", 0, T, None))

        if args.mode in ("event", "both"):
            peaks = detect_events_simple(seed_trace, z_thresh=args.event_z, min_sep_frames=args.event_min_sep)
            if len(peaks) > args.max_events_per_seed:
                peaks = peaks[:args.max_events_per_seed]
            print(f"  Detected events: {len(peaks)} (cap={args.max_events_per_seed})")
            for ei, pk in enumerate(peaks, start=1):
                t0, t1 = event_window(pk, T, args.event_pre, args.event_post)
                windows.append((f"event_{ei:03d}", t0, t1, pk))

        for win_name, t0, t1, pk in windows:
            print(f"  [Corr] {win_name}: frames {t0}:{t1}")

            corrmap = corrmap_for_window_raw(
                raw=raw,
                f0=f0,
                tissue_mask=tissue,
                seed_trace_full=seed_trace,
                t0=t0,
                t1=t1,
                regress_global=args.regress_global,
                global_trace_full=global_trace,
                cfg=corr_cfg,
            )

            corr_dir = out / "corrmaps" / seed_name
            save_tiff(corr_dir / f"{win_name}_corrmap.tif", corrmap.astype(np.float16), compress=True)

            labeled = extract_candidates(corrmap, union_mask=union_mask, tissue_mask=tissue, cfg=cand_cfg)
            cand_dir = out / "candidates" / seed_name / win_name
            save_tiff(cand_dir / "candidates_labeled.tif", labeled.astype(np.int32), compress=True)

            n_cand = int(labeled.max())
            print(f"  [Cands] {n_cand} components")

            if n_cand == 0:
                plot_preview(
                    out_png=out / "previews" / seed_name / win_name / "seed_only.png",
                    mean_img=mean_img,
                    seed_mask=seed_mask,
                    cand_mask=None,
                    seed_trace=seed_trace,
                    cand_trace_window=None,
                    t0=t0,
                    t1=t1,
                    title=f"{seed_name} | {win_name} | seed only",
                )
                continue

            # Save each candidate
            for ci in range(1, n_cand + 1):
                cand_mask = (labeled == ci)
                if not cand_mask.any():
                    continue

                # Candidate trace only for the window (fast)
                idx = np.where(cand_mask)
                raw_w = raw[t0:t1].astype(np.float32, copy=False)
                dff_w = dff_from_raw_window(raw_w, f0)
                cand_trace_w = np.array([np.nanmean(dff_w[k][idx]) for k in range(dff_w.shape[0])], dtype=np.float32)
                if args.regress_global and global_trace is not None:
                    cand_trace_w = regress_out(cand_trace_w, global_trace[t0:t1])

                save_tiff(cand_dir / f"cand_{ci:03d}.tif", cand_mask.astype(np.uint8), compress=True)

                prev_path = out / "previews" / seed_name / win_name / f"cand_{ci:03d}.png"
                plot_preview(
                    out_png=prev_path,
                    mean_img=mean_img,
                    seed_mask=seed_mask,
                    cand_mask=cand_mask,
                    seed_trace=seed_trace,
                    cand_trace_window=cand_trace_w,
                    t0=t0,
                    t1=t1,
                    title=f"{seed_name} | {win_name} | cand {ci:03d}",
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
