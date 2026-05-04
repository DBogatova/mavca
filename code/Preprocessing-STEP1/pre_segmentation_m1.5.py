#!/usr/bin/env python3
"""
Module 1.5: Best-frame selection per event crop

This is the closest-to-what-worked selector that produced filenames like:
  bestframe_event_group_0009_peak319_t00010_rank01_mip.png

Key behaviors:
- Works on event crops saved by Module 1 (T,Z,Y,X)
- For each frame t in the crop:
    * compute Z-MIP (optionally only TOP_Z_PLANES)
    * score(t) = p_hi(MIP) - p_mid(MIP)  (dendrite-favoring "sparse bright" score)
- Find the peak frame (argmax of score trace)
- Select TOP_K frames from a window around the peak, enforcing MIN_SEP
- Save:
    bestframe_<event_group_####>_peak<peak_tag>_t<local>_rank##_3d.tif
    bestframe_<event_group_####>_peak<peak_tag>_t<local>_rank##_mip.png

CRITICAL: PNGs use percentile contrast-stretch to [0,1] before saving.
This is what makes trunks look "sharp" instead of gray/foggy.

Peak tag behavior (legacy-matching):
- If input filename contains "..._peak#####..." use that number (e.g., peak56, peak00319)
- Otherwise use the local peak index (peak_local)

No extra scoring terms. No Laplacian. No tubeness. No surprises.
"""

from __future__ import annotations

import re
import gc
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# =========================
# CONFIG
# =========================
DATE = "2026-04-16"
MOUSE = "rbp4_132_phpeb"
RUN = "run7"


BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

# Point this to the folder containing your Module 1 event crops
EVENT_FOLDER = BASE / "preprocessed" / "event_crops"   # <<< change if needed

# Output folder
OUT_FOLDER = BASE / "preprocessed" / "best_frames"
OUT_FOLDER.mkdir(exist_ok=True, parents=True)

# ---- scoring ----
P_HI = 99.9
P_MID = 60.0

# Dendrite-favoring: subtract broad spatial background before scoring
# This removes the surface strip (which is spatially smooth/broad)
SUBTRACT_SPATIAL_BG = True
SPATIAL_BG_SIGMA = 15.0  # gaussian blur for background estimation (pixels)

# Sparseness bonus: reward frames where bright pixels are spatially sparse
# (thin dendrites = few bright pixels; surface = many bright pixels)
USE_SPARSENESS = True
SPARSENESS_THRESHOLD_PCT = 95.0  # what counts as "bright"

# Optional: restrict Z for MIP (use top Z planes, highest index)
TOP_Z_PLANES = 15        # None to use all Z; 15 often works well for apicals

# Optional: very light MIP smoothing for stability
# Set to 0.0 if you want maximum crispness
MIP_SMOOTH_SIGMA = 0.0

# ---- selection ----
PEAK_HALF_WINDOW = 6     # choose candidates from [peak-6, peak+6]
TOP_K = 3                # how many frames to save per crop
MIN_SEP = 1              # enforce spacing between selected frames

# ---- PNG visualization (contrast stretch) ----
PNG_P_LO = 1.0           # lower percentile for display scaling
PNG_P_HI = 99.7          # upper percentile for display scaling
PNG_DPI = 200


# =========================
# Regex helpers (legacy)
# =========================
RE_EVENT_GROUP_ID = re.compile(r"(event_group_\d{4})")
RE_PEAK_IN_NAME = re.compile(r"_peak(\d{1,6})")  # accepts peak56 or peak00319


def extract_event_group_id(stem: str) -> str:
    m = RE_EVENT_GROUP_ID.search(stem)
    return m.group(1) if m else stem


def extract_peak_tag_from_name(stem: str) -> int | None:
    m = RE_PEAK_IN_NAME.search(stem)
    return int(m.group(1)) if m else None


# =========================
# Helpers
# =========================
def mip_z(vol_zyx: np.ndarray) -> np.ndarray:
    """
    Z-MIP on (Z,Y,X), optionally restricted to TOP_Z_PLANES (highest index planes).
    """
    v = vol_zyx
    if TOP_Z_PLANES is not None and TOP_Z_PLANES > 0:
        z0 = max(v.shape[0] - TOP_Z_PLANES, 0)
        v = v[z0:, :, :]
    return np.nanmax(v, axis=0)


def score_mip(mip: np.ndarray) -> float:
    """
    Score a MIP for dendrite-like activity.
    
    1) Optionally subtract broad spatial background (removes surface strip)
    2) Base score: p_hi - p_mid (bright sparse structures)
    3) Optionally add sparseness bonus (fewer bright pixels = more dendrite-like)
    """
    m = mip.astype(np.float32, copy=False)
    if MIP_SMOOTH_SIGMA and MIP_SMOOTH_SIGMA > 0:
        m = gaussian_filter(m, sigma=float(MIP_SMOOTH_SIGMA))
    
    # Subtract broad spatial background (kills surface strip)
    if SUBTRACT_SPATIAL_BG:
        bg = gaussian_filter(m, sigma=SPATIAL_BG_SIGMA)
        m = m - bg
        m[m < 0] = 0
    
    base_score = float(np.percentile(m, P_HI) - np.percentile(m, P_MID))
    
    if USE_SPARSENESS:
        # Fraction of pixels above threshold — lower = sparser = more dendrite-like
        thr = np.percentile(m, SPARSENESS_THRESHOLD_PCT)
        bright_frac = (m > thr).mean()
        # Invert: sparse frames get bonus (multiply by 1/fraction, capped)
        sparseness_bonus = 1.0 / (bright_frac + 0.01)
        return base_score * min(sparseness_bonus, 10.0)
    
    return base_score


def contrast_stretch_01(img: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    """
    Percentile stretch to [0,1] for consistent PNG appearance.
    """
    m = img.astype(np.float32, copy=False)
    finite = np.isfinite(m)
    if not finite.any():
        return np.zeros_like(m, dtype=np.float32)

    lo, hi = np.percentile(m[finite], (p_lo, p_hi))
    if hi <= lo:
        return np.zeros_like(m, dtype=np.float32)

    m = (m - lo) / (hi - lo)
    return np.clip(m, 0, 1).astype(np.float32)


def save_mip_png(path: Path, mip: np.ndarray):
    """
    Save MIP PNG with fixed [0,1] scaling after contrast stretch.
    """
    m01 = contrast_stretch_01(mip, PNG_P_LO, PNG_P_HI)
    plt.figure(figsize=(5, 5))
    plt.imshow(m01, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=PNG_DPI)
    plt.close()


def select_topk_with_spacing(cands: np.ndarray, scores: np.ndarray, k: int, min_sep: int) -> list[int]:
    """
    Choose up to k indices from cands with highest scores, enforcing min separation.
    Returns chosen indices (subset of cands).
    """
    if cands.size == 0:
        return []
    order = cands[np.argsort(scores[cands])[::-1]]
    chosen: list[int] = []
    for t in order:
        t = int(t)
        if all(abs(t - c) >= min_sep for c in chosen):
            chosen.append(t)
        if len(chosen) >= k:
            break
    return chosen


# =========================
# MAIN
# =========================
def main():
    paths = sorted(EVENT_FOLDER.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No .tif files found in {EVENT_FOLDER}")

    print(f"Input:  {EVENT_FOLDER}")
    print(f"Output: {OUT_FOLDER}")
    print(f"Found {len(paths)} crops.")
    print(f"TOP_Z_PLANES={TOP_Z_PLANES}, MIP_SMOOTH_SIGMA={MIP_SMOOTH_SIGMA}, TOP_K={TOP_K}")

    for p in paths:
        stem = p.stem
        eg_id = extract_event_group_id(stem)

        crop = tifffile.imread(p).astype(np.float32)  # (T,Z,Y,X)
        if crop.ndim != 4 or crop.shape[0] < 1:
            print(f"Skip {p.name} (shape {crop.shape})")
            del crop
            continue

        T = crop.shape[0]

        # score trace
        scores = np.zeros(T, dtype=np.float32)
        for t in range(T):
            mip = mip_z(crop[t])
            scores[t] = score_mip(mip)

        peak_local = int(np.argmax(scores))

        # legacy peak tag behavior
        peak_tag = extract_peak_tag_from_name(stem)
        if peak_tag is None:
            peak_tag = peak_local

        # candidate window around peak
        w0 = max(peak_local - PEAK_HALF_WINDOW, 0)
        w1 = min(peak_local + PEAK_HALF_WINDOW, T - 1)
        window = np.arange(w0, w1 + 1, dtype=int)

        chosen = select_topk_with_spacing(window, scores, TOP_K, MIN_SEP)
        # save in rank order (best first)
        chosen = sorted(chosen, key=lambda t: float(scores[t]), reverse=True)

        for rank, t_local in enumerate(chosen, start=1):
            vol = crop[t_local]  # (Z,Y,X)
            mip = mip_z(vol)

            out3d = OUT_FOLDER / f"bestframe_{eg_id}_peak{peak_tag}_t{t_local:05d}_rank{rank:02d}_3d.tif"
            outpng = OUT_FOLDER / f"bestframe_{eg_id}_peak{peak_tag}_t{t_local:05d}_rank{rank:02d}_mip.png"

            tifffile.imwrite(out3d, vol.astype(np.float32), photometric="minisblack")
            save_mip_png(outpng, mip)

        del crop
        gc.collect()

    print("Done ✅")


if __name__ == "__main__":
    main()
