#!/usr/bin/env python3
"""
Global Ca sanity check with RANDOM CONTROL (ONE RUN)

Adds a critical control:
- Randomly sample same number of voxels as dendrites from OUTSIDE region
- Show that (fake_inside - outside) has no structured events

This confirms dendrite-specific signal is real and not global noise.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

try:
    import tifffile
except ImportError:
    tifffile = None


# =================
# ===== CONFIG =====
# =================
DATE = "2025-12-02"
MOUSE = "rbp4cre_136_phpeb"
RUN = "run5"

FRAME_RATE = 5.0
CHUNK_T = 118
Y_CROP = 3

EXCLUDE_TOP_ONLY_FOR_TRACES = True
EXCLUDE_TOP_FRACTION = 0.01

F0_NFRAMES = 30
EPS = 1e-8

ARTIFACT_Z = -0.5
SMOOTH_SIGMA = 0.5

# Random control
DO_RANDOM_CONTROL = True
N_SHUFFLES = 20
RANDOM_SEED = 0

SAVE_FIG = True
SHOW_FIG = True
FIG_DPI = 200


# ================
# ===== PATHS =====
# ================
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "scape-data" / DATE / MOUSE / RUN

RAW_CLEAN_PATH = BASE / "preprocessed" / "raw_clean.tif"
RAW_ORIG_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_binimagej_reslice_green.tif"
RAW_STACK_PATH = RAW_CLEAN_PATH if RAW_CLEAN_PATH.exists() else RAW_ORIG_PATH

MASK_FOLDER = BASE / "labelmaps_curated_dynamic"

TRACE_FOLDER = BASE / "traces"
TRACE_FOLDER.mkdir(exist_ok=True)

OUT_PNG = TRACE_FOLDER / "globalCa_random_control.png"


# ==========================
# ===== Helpers ============
# ==========================
def apply_y_crop(arr, y_crop):
    if y_crop <= 0:
        return arr
    return arr[:, :, :-y_crop, :] if arr.ndim == 4 else arr[:, :-y_crop, :]


def load_stack_memmap(path):
    if tifffile is None:
        raise ImportError("tifffile required")
    return tifffile.memmap(str(path))


def discover_mask_files(folder):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in (".npy", ".tif", ".tiff")])


def load_mask(path):
    if path.suffix == ".npy":
        m = np.load(path)
    else:
        m = tifffile.imread(str(path))
    if m.ndim == 4:
        m = np.any(m != 0, axis=0)
    return m.astype(bool)


def build_union_mask(files, shape):
    union = np.zeros(shape, bool)
    for f in files:
        union |= load_mask(f)
    return union


def compute_f0(stack):
    return np.nanmean(stack[-F0_NFRAMES:], axis=0).astype(np.float32)


def make_trace_masks(union):
    Z, Y, X = union.shape
    allowed = np.ones_like(union, bool)
    if EXCLUDE_TOP_ONLY_FOR_TRACES:
        allowed[:, :int(Y * EXCLUDE_TOP_FRACTION), :] = False
    inside = union & allowed
    outside = (~union) & allowed
    return inside.reshape(-1), outside.reshape(-1), allowed.reshape(-1)


def mean_trace_from_flatmask(stack, f0, flatmask):
    T = stack.shape[0]
    out = np.zeros(T)
    for t0 in range(0, T, CHUNK_T):
        t1 = min(T, t0 + CHUNK_T)
        dff = (stack[t0:t1] - f0) / (f0 + EPS)
        dff[dff < ARTIFACT_Z] = 0
        out[t0:t1] = np.nanmean(dff.reshape(dff.shape[0], -1)[:, flatmask], axis=1)
    return gaussian_filter1d(out, SMOOTH_SIGMA)


# =================
# ===== Main ======
# =================
def main():
    stack = load_stack_memmap(RAW_STACK_PATH)
    stack = apply_y_crop(stack, Y_CROP)
    T, Z, Y, X = stack.shape

    union = build_union_mask(discover_mask_files(MASK_FOLDER), (Z, Y, X))
    coverage = union.mean() * 100
    print(f"Union mask coverage: {coverage:.2f}%")

    f0 = compute_f0(stack)
    inside_flat, outside_flat, all_flat = make_trace_masks(union)

    inside = mean_trace_from_flatmask(stack, f0, inside_flat)
    outside = mean_trace_from_flatmask(stack, f0, outside_flat)
    dendrite_specific = inside - outside

    # ---------- RANDOM CONTROL ----------
    rng = np.random.default_rng(RANDOM_SEED)
    pool = np.flatnonzero(outside_flat)
    n_inside = inside_flat.sum()

    fake_specific = []
    for k in range(N_SHUFFLES):
        chosen = rng.choice(pool, size=n_inside, replace=False)
        fake_mask = np.zeros(Z * Y * X, bool)
        fake_mask[chosen] = True
        fake = mean_trace_from_flatmask(stack, f0, fake_mask)
        fake_specific.append(fake - outside)

    fake_specific = np.array(fake_specific)

    # ---------- PLOT ----------
    time_s = np.arange(T) / FRAME_RATE
    plt.figure(figsize=(12, 4))
    for k in range(N_SHUFFLES):
        plt.plot(time_s, fake_specific[k], color="gray", alpha=0.25)
    plt.plot(time_s, dendrite_specific, color="crimson", lw=2, label="Real (inside − outside)")
    plt.axhline(0, color="k", ls="--", alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F")
    plt.title(f"Random control sanity check ({coverage:.2f}% coverage)")
    plt.legend()
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(OUT_PNG, dpi=FIG_DPI)
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
