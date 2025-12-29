#!/usr/bin/env python3
"""
Global Ca outside masks (union) vs global Ca over full volume

For ONE run at a time:
- Build a union mask from your curated labelmaps (MASK_FOLDER)
- Compute three Ca timecourses (ΔF/F):
    1) global_all(t): mean ΔF/F across entire volume
    2) global_outside(t): mean ΔF/F across voxels OUTSIDE union mask
    3) global_inside(t): mean ΔF/F across voxels INSIDE union mask
- Overlay-plot them and save to TRACE_FOLDER

Memory-safe:
- Reads the stack in time chunks (CHUNK_T)
- Uses an F0 defined from the LAST F0_NFRAMES frames (per-voxel), so we don't
  need to load the whole movie to compute per-voxel baseline.

Assumptions:
- RAW_STACK is shaped (T, Z, Y, X)
- Masks are 3D (Z, Y, X), nonzero = inside mask
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

FRAME_RATE = 5.0      # Hz
CHUNK_T = 118         # time frames per chunk
Y_CROP = 3  # number of pixels to crop from bottom in Y

# Global trace computation
EXCLUDE_TOP_FRACTION = 0.01         # exclude top 30% of Y when computing global traces
EXCLUDE_TOP_ONLY_FOR_TRACES = True  # only apply exclusion to traces, not masks

# ΔF/F baseline (per-voxel) computed from last N frames
F0_NFRAMES = 30
EPS = 1e-8

# Trace cleanup / smoothing (same style as your pipeline)
ARTIFACT_Z = -0.5     # replace ΔF/F < ARTIFACT_Z with 0 (before smoothing)
SMOOTH_SIGMA = 0.5    # gaussian_filter1d sigma (in frames)

# Plot / output
SAVE_FIG = True
SHOW_FIG = True
FIG_DPI = 200


# ================
# ===== PATHS =====
# ================
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "scape-data" / DATE / MOUSE / RUN

RAW_CLEAN_PATH = BASE / "preprocessed" / "raw_clean.tif"
RAW_ORIG_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_binimagej_reslice_green.tif"   # adjust if your naming differs
RAW_STACK_PATH = RAW_CLEAN_PATH if RAW_CLEAN_PATH.exists() else RAW_ORIG_PATH

MASK_FOLDER = BASE / "labelmaps_curated_dynamic"  # union across these masks

TRACE_FOLDER = BASE / "traces"
TRACE_FOLDER.mkdir(exist_ok=True)

OUT_PNG = TRACE_FOLDER / "global_ca_all_vs_outside_masks.png"
OUT_NPY = TRACE_FOLDER / "global_ca_all_vs_outside_masks.npy"


# ==========================
# ===== I/O helpers =========
# ==========================
def apply_y_crop(arr, y_crop):
    """
    Crop Y dimension from bottom only.
    Supports:
      - (T,Z,Y,X)
      - (Z,Y,X)
    """
    if y_crop is None or y_crop <= 0:
        return arr

    if arr.ndim == 4:      # (T,Z,Y,X)
        return arr[:, :, :-y_crop, :]
    elif arr.ndim == 3:    # (Z,Y,X)
        return arr[:, :-y_crop, :]
    else:
        raise ValueError(f"Unsupported array shape for Y-crop: {arr.shape}")

def _require_tifffile():
    if tifffile is None:
        raise ImportError("tifffile is required. Install with: pip install tifffile")


def load_stack_memmap(path: Path) -> np.ndarray:
    """Memory-map a TIFF stack."""
    _require_tifffile()
    if not path.exists():
        raise FileNotFoundError(f"Stack not found: {path}")
    return tifffile.memmap(str(path))


def ensure_tzyx(stack: np.ndarray) -> np.ndarray:
    """
    Ensure stack is (T,Z,Y,X). If your data is already (T,Z,Y,X), this is no-op.
    If you sometimes have (Z,Y,X,T) etc., adjust here.
    """
    if stack.ndim != 4:
        raise ValueError(f"Expected 4D stack (T,Z,Y,X). Got shape={stack.shape}")

    # Most of your pipeline uses (T,Z,Y,X), so we assume it's correct.
    return stack


def discover_mask_files(mask_folder: Path) -> list[Path]:
    if not mask_folder.exists():
        raise FileNotFoundError(f"MASK_FOLDER not found: {mask_folder}")

    exts = (".npy", ".npz", ".tif", ".tiff")
    files = sorted([p for p in mask_folder.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No mask files found in: {mask_folder}")
    return files


def load_mask(path: Path) -> np.ndarray:
    """
    Load a mask volume.
    Accepts:
      - .npy
      - .npz (single array)
      - .tif/.tiff (3D)
    Returns: boolean mask (Z,Y,X)
    """
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path, mmap_mode="r")
    elif suf == ".npz":
        npz = np.load(path)
        if len(npz.files) == 1:
            arr = npz[npz.files[0]]
        elif "arr_0" in npz.files:
            arr = npz["arr_0"]
        else:
            raise ValueError(f"NPZ has multiple arrays; ambiguous. Keys: {npz.files}")
    elif suf in (".tif", ".tiff"):
        _require_tifffile()
        arr = tifffile.imread(str(path))
    else:
        raise ValueError(f"Unsupported mask type: {path}")

    if arr.ndim == 4:
        # If some masks are time-resolved, union over time
        arr = np.any(arr != 0, axis=0)

    if arr.ndim != 3:
        raise ValueError(f"Mask must be 3D (Z,Y,X) (or 4D collapsible). Got {arr.shape} from {path}")

    return (arr != 0)


def build_union_mask(mask_files: list[Path], target_zyx: tuple[int, int, int]) -> np.ndarray:
    Z, Y, X = target_zyx
    union = np.zeros((Z, Y, X), dtype=bool)

    for f in mask_files:
        m = load_mask(f)
        if m.shape != (Z, Y, X):
            # Adjust mask to match target shape
            Zm, Ym, Xm = m.shape
            if (Zm, Xm) != (Z, X):
                raise ValueError(f"Z/X mismatch for {f}: mask ({Zm},{Xm}) vs target ({Z},{X})")
            
            if Ym < Y:
                # Pad Y dimension
                pad_y = Y - Ym
                m = np.pad(m, ((0,0), (0,pad_y), (0,0)), mode='constant')
            elif Ym > Y:
                # Crop Y dimension
                m = m[:, :Y, :]
                
        union |= m

    return union


# ==========================================
# ===== Core computation (chunked) ==========
# ==========================================
def compute_f0_from_last_frames(stack_tzyx: np.ndarray, nframes: int) -> np.ndarray:
    """
    Compute per-voxel F0 as mean of the last nframes.
    Returns F0 with shape (Z,Y,X) float32.
    """
    T = stack_tzyx.shape[0]
    n = int(min(max(1, nframes), T))
    last = np.asarray(stack_tzyx[T - n:T], dtype=np.float32)  # load only last n frames
    f0 = np.nanmean(last, axis=0).astype(np.float32)
    return f0


def compute_traces_chunked(stack_tzyx: np.ndarray, union_mask_zyx: np.ndarray, f0_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute global_all(t), global_outside(t), and global_inside(t) in chunks without storing full ΔF/F stack.
    """
    T, Z, Y, X = stack_tzyx.shape
    outside = ~union_mask_zyx
    inside = union_mask_zyx
    
    # Exclude top fraction for trace computation
    if EXCLUDE_TOP_ONLY_FOR_TRACES:
        y_exclude = int(EXCLUDE_TOP_FRACTION * Y)
        trace_mask = np.ones((Z, Y, X), dtype=bool)
        trace_mask[:, :y_exclude, :] = False
        
        # Apply exclusion to all masks
        all_mask = trace_mask
        outside_mask = outside & trace_mask
        inside_mask = inside & trace_mask
    else:
        all_mask = np.ones((Z, Y, X), dtype=bool)
        outside_mask = outside
        inside_mask = inside
    
    all_flat = all_mask.reshape(-1)
    outside_flat = outside_mask.reshape(-1)
    inside_flat = inside_mask.reshape(-1)

    global_all = np.zeros(T, dtype=np.float32)
    global_out = np.zeros(T, dtype=np.float32)
    global_in = np.zeros(T, dtype=np.float32)

    for t0 in range(0, T, CHUNK_T):
        t1 = min(T, t0 + CHUNK_T)
        chunk = np.asarray(stack_tzyx[t0:t1], dtype=np.float32)

        # ΔF/F per voxel (using per-voxel F0 from last frames)
        dff = (chunk - f0_zyx[None, ...]) / (f0_zyx[None, ...] + EPS)

        # Artifact clamp (same style as your pipeline)
        if ARTIFACT_Z is not None:
            dff[dff < ARTIFACT_Z] = 0.0

        # Compute means efficiently by flattening spatial dims
        dff2 = dff.reshape(dff.shape[0], -1)  # (tchunk, N)

        global_all[t0:t1] = np.nanmean(dff2[:, all_flat], axis=1)
        global_out[t0:t1] = np.nanmean(dff2[:, outside_flat], axis=1)
        global_in[t0:t1] = np.nanmean(dff2[:, inside_flat], axis=1)

        print(f"Processed frames {t0}..{t1-1}")

    return global_all, global_out, global_in


# ======================
# ===== Plotting =======
# ======================
def plot_overlay(time_s: np.ndarray, global_all: np.ndarray, global_out: np.ndarray, global_in: np.ndarray, out_png: Path | None):
    # Correlations (finite-only)
    finite = np.isfinite(global_all) & np.isfinite(global_out) & np.isfinite(global_in)
    corr_out = np.corrcoef(global_all[finite], global_out[finite])[0, 1] if finite.sum() > 3 else np.nan
    corr_in = np.corrcoef(global_all[finite], global_in[finite])[0, 1] if finite.sum() > 3 else np.nan

    plt.figure(figsize=(11, 4.2))
    plt.plot(time_s, global_all, lw=2, label="Global Ca (all voxels)")
    plt.plot(time_s, global_out, lw=2, alpha=0.85, label="Global Ca (outside masks)")
    plt.plot(time_s, global_in, lw=2, alpha=0.85, label="Global Ca (inside masks)")
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F")
    plt.title(f"{DATE} | {MOUSE} | {RUN}  (out corr={corr_out:.2f}, in corr={corr_in:.2f})")
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_png, dpi=FIG_DPI)
        print(f"Saved: {out_png}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close()


# =================
# ===== Main ======
# =================
def main():
    print("\n=== Loading stack ===")
    stack = load_stack_memmap(RAW_STACK_PATH)
    stack = ensure_tzyx(stack)
    T, Z, Y, X = stack.shape
    print(f"Stack: {RAW_STACK_PATH}")
    print(f"Shape: T={T}, Z={Z}, Y={Y}, X={X}")
    
    # Crop bottom 3 pixels to match mask Y dimension
    if Y_CROP and Y_CROP > 0:
        stack = apply_y_crop(stack, Y_CROP)
        T, Z, Y, X = stack.shape
        print(f"After Y crop: T={T}, Z={Z}, Y={Y}, X={X}")

    print("\n=== Building union mask ===")
    mask_files = discover_mask_files(MASK_FOLDER)
    print(f"Found {len(mask_files)} mask files in {MASK_FOLDER}")
    union = build_union_mask(mask_files, target_zyx=(Z, Y, X))
    print(f"Union mask coverage: {union.mean()*100:.2f}% of voxels")

    print("\n=== Computing F0 (last frames) ===")
    f0 = compute_f0_from_last_frames(stack, nframes=F0_NFRAMES)
    print(f"F0 shape: {f0.shape}")

    print("\n=== Computing traces (chunked) ===")
    global_all, global_out, global_in = compute_traces_chunked(stack, union, f0)

    print("\n=== Smoothing traces ===")
    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        global_all_s = gaussian_filter1d(global_all, sigma=SMOOTH_SIGMA)
        global_out_s = gaussian_filter1d(global_out, sigma=SMOOTH_SIGMA)
        global_in_s = gaussian_filter1d(global_in, sigma=SMOOTH_SIGMA)
    else:
        global_all_s = global_all
        global_out_s = global_out
        global_in_s = global_in

    time_s = np.arange(T, dtype=np.float32) / float(FRAME_RATE)

    print("\n=== Saving outputs ===")
    out = {
        "time_s": time_s,
        "global_all": global_all,
        "global_outside": global_out,
        "global_inside": global_in,
        "global_all_smooth": global_all_s,
        "global_outside_smooth": global_out_s,
        "global_inside_smooth": global_in_s,
        "meta": {
            "DATE": DATE,
            "MOUSE": MOUSE,
            "RUN": RUN,
            "RAW_STACK_PATH": str(RAW_STACK_PATH),
            "MASK_FOLDER": str(MASK_FOLDER),
            "FRAME_RATE": FRAME_RATE,
            "CHUNK_T": CHUNK_T,
            "F0_NFRAMES": F0_NFRAMES,
            "ARTIFACT_Z": ARTIFACT_Z,
            "SMOOTH_SIGMA": SMOOTH_SIGMA,
        }
    }
    np.save(OUT_NPY, out, allow_pickle=True)
    print(f"Saved: {OUT_NPY}")

    print("\n=== Plotting overlay (smoothed) ===")
    plot_overlay(time_s, global_all_s, global_out_s, global_in_s, OUT_PNG if SAVE_FIG else None)

    print("\nDone.")


if __name__ == "__main__":
    main()