#!/usr/bin/env python3
"""
Global Ca analysis with micron-based rings.

What this script does:
1) Builds a UNION dendrite mask from curated labelmaps.
2) Computes global traces (ΔF/F) in a memory-safe chunked way:
   - All voxels (in allowed region)
   - Outside masks
   - Inside masks
   - Optional tissue-restricted versions (based on F0 brightness)
   - Optional absolute ΔF versions
3) Neuropil/surround test using **distance-to-dendrite in MICRONS**:
   - Rings: 0--5 µm, 5--10 µm, 10--15 µm, and far >15 µm (all OUTSIDE union)
   - Uses distance_transform_edt with sampling=(Z_um, Y_um, X_um)
4) Saves plots + a .npy bundle of outputs + metadata.

Notes:
- Your masks were generated after Y_CROP in Module 1. This script applies the same Y_CROP to the raw stack
  to match mask dimensions.
- "Allowed region" optionally excludes a bright top band ONLY for trace computation (not masks).
- ΔF/F uses F0 from the last F0_NFRAMES frames.
- ΔF/F denominator floor protects against huge ΔF/F in dim voxels.

Assumptions:
- Stack is (T,Z,Y,X)
- Masks are 3D (Z,Y,X) where nonzero = inside dendrite mask
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, distance_transform_edt

try:
    import tifffile
except ImportError:
    tifffile = None


# =================
# ===== CONFIG =====
# =================
DATE = "2025-12-25"
MOUSE = "rAi162_phpeb"
RUN = "run1"

FRAME_RATE = 5.0
CHUNK_T = 118
Y_CROP = 3  # crop bottom Y pixels to match masks

# ===== Voxel sizes (MICRONS) =====
# !!! Set these to your dataset values !!!
# If unsure: XY is often ~0.5–0.8 µm; Z is your effective step in µm after reslice/deskew.
VOXEL_Z_UM = 3.9
VOXEL_Y_UM = 0.5
VOXEL_X_UM = 0.5

# Optional: exclude top fraction ONLY for trace computation (not masks)
EXCLUDE_TOP_ONLY_FOR_TRACES = True
EXCLUDE_TOP_FRACTION = 0.01  # e.g., 0.30 = exclude top 30% of Y

# Baseline
F0_NFRAMES = 30
EPS = 1e-8

# ΔF/F safety (prevents huge ΔF/F in dim voxels)
USE_DENOM_FLOOR = True
DENOM_FLOOR_MODE = "percentile"  # "percentile" or "absolute"
DENOM_FLOOR_PCT = 5.0
DENOM_FLOOR_ABS = 20.0

# Also compute absolute ΔF in parallel?
COMPUTE_DELTAF = True

# Artifact clamp / smoothing
ARTIFACT_Z = -0.5
SMOOTH_SIGMA = 0.5

# Tissue mask (answers "coverage in functional volume")
USE_TISSUE_MASK = True
TISSUE_MODE = "percentile"     # "percentile" or "mean_plus_kstd"
TISSUE_F0_PCT = 10.0           # keep voxels with F0 >= this percentile (within allowed region)
TISSUE_KSTD = 1.0

# Neuropil surround test: micron-based rings (outside only)
DO_UM_RINGS = True
RING_UM_EDGES = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)  # rings: 0–5, 5–10, 10–15, far >30

# ===== FUNCTIONAL VOXEL DETECTION =====
# Detect voxels with significant activity (not just bright F0)
USE_ACTIVITY_MASK = True
ACTIVITY_THRESHOLD_MODE = "mad"  # "mad", "percentile", or "zscore"
ACTIVITY_MAD_FACTOR = 3.0        # voxels with temporal MAD > 3x median MAD
ACTIVITY_PERCENTILE = 90.0       # or top 10% most variable voxels
ACTIVITY_ZSCORE = 2.0            # or voxels with peak z-score > 2

# ===== NEUROPIL CONTAMINATION CORRECTION =====
CONTAMINATION_RINGS = (2.0, 5.0, 10.0)  # analyze contamination vs distance

# ===== VOLUME-WEIGHTED COMPARISONS =====
BOOTSTRAP_COMPARISONS = True     # bootstrap to get confidence intervals
N_BOOTSTRAP = 1000

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
RAW_ORIG_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_green.tif"
RAW_STACK_PATH = RAW_CLEAN_PATH if RAW_CLEAN_PATH.exists() else RAW_ORIG_PATH

MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
TRACE_FOLDER = BASE / "outside_mask_dynamics"
TRACE_FOLDER.mkdir(exist_ok=True)

OUT_NPY = TRACE_FOLDER / "globalCa_umrings.npy"
OUT_PNG_MAIN = TRACE_FOLDER / "globalCa_umrings_main.png"
OUT_PNG_TISSUE = TRACE_FOLDER / "globalCa_umrings_tissue.png"
OUT_PNG_RINGS = TRACE_FOLDER / "globalCa_umrings_rings.png"
OUT_PNG_DELTAF = TRACE_FOLDER / "globalCa_umrings_deltaF.png"


# ==========================
# ===== I/O helpers =========
# ==========================
def apply_y_crop(arr: np.ndarray, y_crop: int) -> np.ndarray:
    if y_crop is None or y_crop <= 0:
        return arr
    if arr.ndim == 4:
        return arr[:, :, :-y_crop, :]
    if arr.ndim == 3:
        return arr[:, :-y_crop, :]
    raise ValueError(f"Unsupported array shape for Y-crop: {arr.shape}")


def _require_tifffile():
    if tifffile is None:
        raise ImportError("tifffile is required. Install with: pip install tifffile")


def load_stack_memmap(path: Path) -> np.ndarray:
    _require_tifffile()
    if not path.exists():
        raise FileNotFoundError(f"Stack not found: {path}")
    return tifffile.memmap(str(path))


def ensure_tzyx(stack: np.ndarray) -> np.ndarray:
    if stack.ndim != 4:
        raise ValueError(f"Expected 4D stack (T,Z,Y,X). Got shape={stack.shape}")
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
            Zm, Ym, Xm = m.shape
            if (Zm, Xm) != (Z, X):
                raise ValueError(f"Z/X mismatch for {f}: mask ({Zm},{Xm}) vs target ({Z},{X})")
            if Ym < Y:
                m = np.pad(m, ((0, 0), (0, Y - Ym), (0, 0)), mode="constant")
            elif Ym > Y:
                m = m[:, :Y, :]
        union |= m

    return union


# ==========================================
# ===== Core computation helpers ============
# ==========================================
def compute_f0_from_last_frames(stack_tzyx: np.ndarray, nframes: int) -> np.ndarray:
    T = stack_tzyx.shape[0]
    n = int(min(max(1, nframes), T))
    last = np.asarray(stack_tzyx[T - n:T], dtype=np.float32)
    return np.nanmean(last, axis=0).astype(np.float32)


def make_allowed_region_mask(Z: int, Y: int, X: int) -> np.ndarray:
    allowed = np.ones((Z, Y, X), dtype=bool)
    if EXCLUDE_TOP_ONLY_FOR_TRACES and EXCLUDE_TOP_FRACTION and EXCLUDE_TOP_FRACTION > 0:
        y_exclude = int(round(EXCLUDE_TOP_FRACTION * Y))
        y_exclude = max(0, min(Y, y_exclude))
        if y_exclude > 0:
            allowed[:, :y_exclude, :] = False
    return allowed


def compute_tissue_mask_from_f0(f0_zyx: np.ndarray, allowed_zyx: np.ndarray) -> tuple[np.ndarray, float]:
    f0_allowed = f0_zyx[allowed_zyx]
    if f0_allowed.size < 10:
        raise ValueError("Allowed region too small to compute tissue mask")

    if TISSUE_MODE == "percentile":
        thr = float(np.nanpercentile(f0_allowed, TISSUE_F0_PCT))
    elif TISSUE_MODE == "mean_plus_kstd":
        thr = float(np.nanmean(f0_allowed) + TISSUE_KSTD * np.nanstd(f0_allowed))
    else:
        raise ValueError(f"Unknown TISSUE_MODE: {TISSUE_MODE}")

    tissue = (f0_zyx >= thr) & allowed_zyx
    return tissue, thr


def compute_denom_floor(f0_zyx: np.ndarray, region_zyx: np.ndarray) -> float:
    if not USE_DENOM_FLOOR:
        return 0.0
    if DENOM_FLOOR_MODE == "absolute":
        return float(DENOM_FLOOR_ABS)

    vals = f0_zyx[region_zyx]
    if vals.size < 10:
        return 0.0
    return float(np.nanpercentile(vals, DENOM_FLOOR_PCT))


def _apply_metric(chunk_tzyx: np.ndarray, f0_zyx: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray | None]:
    chunk = np.asarray(chunk_tzyx, dtype=np.float32)
    dF = (chunk - f0_zyx[None, ...]) if COMPUTE_DELTAF else None
    denom = (f0_zyx[None, ...] + EPS + alpha)
    dff = (chunk - f0_zyx[None, ...]) / denom

    if ARTIFACT_Z is not None:
        dff[dff < ARTIFACT_Z] = 0.0

    return dff, dF


def compute_mean_trace_chunked(stack_tzyx: np.ndarray,
                               f0_zyx: np.ndarray,
                               flatmask: np.ndarray,
                               alpha: float) -> tuple[np.ndarray, np.ndarray | None]:
    T = stack_tzyx.shape[0]
    out_dff = np.zeros(T, dtype=np.float32)
    out_dF = np.zeros(T, dtype=np.float32) if COMPUTE_DELTAF else None

    for t0 in range(0, T, CHUNK_T):
        t1 = min(T, t0 + CHUNK_T)
        dff, dF = _apply_metric(stack_tzyx[t0:t1], f0_zyx, alpha)

        dff2 = dff.reshape(dff.shape[0], -1)
        out_dff[t0:t1] = np.nanmean(dff2[:, flatmask], axis=1)

        if COMPUTE_DELTAF and dF is not None:
            dF2 = dF.reshape(dF.shape[0], -1)
            out_dF[t0:t1] = np.nanmean(dF2[:, flatmask], axis=1)

        print(f"Processed frames {t0}..{t1-1}")

    return out_dff, out_dF


def smooth(x: np.ndarray) -> np.ndarray:
    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        return gaussian_filter1d(x, sigma=SMOOTH_SIGMA)
    return x


def compute_activity_mask(stack_tzyx: np.ndarray, allowed_zyx: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Detect functionally active voxels based on temporal variability.
    This addresses the issue of including too many 'dead' voxels in averages.
    """
    print("Computing activity-based functional mask...")
    
    # Compute temporal statistics for each voxel
    temporal_mad = np.nanmedian(np.abs(stack_tzyx - np.nanmedian(stack_tzyx, axis=0)[None, ...]), axis=0)
    temporal_std = np.nanstd(stack_tzyx, axis=0)
    
    # Only consider allowed region
    allowed_mad = temporal_mad[allowed_zyx]
    allowed_std = temporal_std[allowed_zyx]
    
    if ACTIVITY_THRESHOLD_MODE == "mad":
        # Voxels with high temporal variability (MAD-based)
        mad_threshold = np.nanmedian(allowed_mad) * ACTIVITY_MAD_FACTOR
        active = (temporal_mad > mad_threshold) & allowed_zyx
        threshold_used = mad_threshold
        
    elif ACTIVITY_THRESHOLD_MODE == "percentile":
        # Top X% most variable voxels
        std_threshold = np.nanpercentile(allowed_std, ACTIVITY_PERCENTILE)
        active = (temporal_std > std_threshold) & allowed_zyx
        threshold_used = std_threshold
        
    elif ACTIVITY_THRESHOLD_MODE == "zscore":
        # Voxels with significant peak responses
        z_scores = (stack_tzyx - np.nanmean(stack_tzyx, axis=0)[None, ...]) / (np.nanstd(stack_tzyx, axis=0)[None, ...] + EPS)
        peak_z = np.nanmax(z_scores, axis=0)
        active = (peak_z > ACTIVITY_ZSCORE) & allowed_zyx
        threshold_used = ACTIVITY_ZSCORE
    
    stats = {
        "total_allowed": int(allowed_zyx.sum()),
        "active_voxels": int(active.sum()),
        "activity_fraction": float(active.sum() / allowed_zyx.sum()),
        "threshold_used": threshold_used
    }
    
    print(f"Activity mask: {stats['active_voxels']}/{stats['total_allowed']} voxels ({stats['activity_fraction']*100:.1f}%)")
    return active, stats


def compute_contamination_analysis(union_zyx: np.ndarray, allowed_zyx: np.ndarray) -> dict[str, np.ndarray]:
    """
    Analyze neuropil at different distances to assess dendrite contamination.
    """
    print("Computing contamination rings...")
    
    outside_allowed = (~union_zyx) & allowed_zyx
    dist_um = distance_transform_edt(~union_zyx, sampling=(VOXEL_Z_UM, VOXEL_Y_UM, VOXEL_X_UM))
    
    rings = {}
    for i, (r_min, r_max) in enumerate(zip([0] + list(CONTAMINATION_RINGS), CONTAMINATION_RINGS + [np.inf])):
        if r_max == np.inf:
            mask = (dist_um > r_min) & outside_allowed
            name = f"neuropil_gt{r_min}um"
        else:
            mask = (dist_um > r_min) & (dist_um <= r_max) & outside_allowed
            name = f"neuropil_{r_min}-{r_max}um"
        
        rings[name] = mask
        print(f"{name}: {mask.sum()} voxels")
    
    return rings


# ==========================================
# ===== Micron-distance rings ===============
# ==========================================
def build_um_rings(union_zyx: np.ndarray, allowed_zyx: np.ndarray, edges_um: tuple[float, ...]) -> dict[str, np.ndarray]:
    """
    Distance-to-dendrite rings in microns, OUTSIDE union, within allowed region.

    Uses distance_transform_edt on (~union) so that:
      dist_um[v] = distance from v to nearest union voxel (in microns),
    provided sampling=(Z_um, Y_um, X_um).

    Returns dict:
      ring_0_5um, ring_5_10um, ring_10_15um, far_gt_15um
    """
    edges = [float(e) for e in edges_um]
    edges = sorted([e for e in edges if e > 0])

    outside_allowed = (~union_zyx) & allowed_zyx
    if outside_allowed.sum() < 10:
        raise ValueError("Outside-allowed region too small")

    # distance_transform_edt expects "features" where input==0.
    # If we pass (~union) as bool: union voxels are False (0) => features => distances computed to union.
    dist_um = distance_transform_edt(~union_zyx, sampling=(VOXEL_Z_UM, VOXEL_Y_UM, VOXEL_X_UM)).astype(np.float32)

    rings: dict[str, np.ndarray] = {}
    lo = 0.0
    for hi in edges:
        m = (dist_um > lo) & (dist_um <= hi) & outside_allowed
        rings[f"ring_{lo:g}_{hi:g}um"] = m
        lo = hi

    rings[f"far_gt_{edges[-1]:g}um"] = (dist_um > edges[-1]) & outside_allowed
    return rings


# ======================
# ===== Plotting =======
# ======================
def plot_overlay(time_s: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 title: str, labels: tuple[str, str, str], out_png: Path | None, ylab: str):
    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    # Require minimum sample size for reliable correlation
    min_samples = max(30, len(a) // 10)  # At least 30 points or 10% of data
    corr_ab = np.corrcoef(a[finite], b[finite])[0, 1] if finite.sum() > min_samples else np.nan
    corr_ac = np.corrcoef(a[finite], c[finite])[0, 1] if finite.sum() > min_samples else np.nan

    plt.figure(figsize=(11, 4.2))
    plt.plot(time_s, a, lw=2, label=labels[0])
    plt.plot(time_s, b, lw=2, alpha=0.85, label=labels[1])
    plt.plot(time_s, c, lw=2, alpha=0.85, label=labels[2])
    plt.xlabel("Time (s)")
    plt.ylabel(ylab)
    plt.title(f"{title} (corr {labels[1]}={corr_ab:.2f}, corr {labels[2]}={corr_ac:.2f})")
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=FIG_DPI)
        print(f"Saved: {out_png}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close()


def plot_many(time_s: np.ndarray, traces: dict[str, np.ndarray], out_png: Path | None, title: str, ylab: str):
    plt.figure(figsize=(11, 4.5))
    for name, tr in traces.items():
        plt.plot(time_s, tr, lw=1.8, label=name)
    plt.xlabel("Time (s)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(ncols=2, fontsize=9)
    plt.tight_layout()

    if out_png is not None:
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

    if Y_CROP and Y_CROP > 0:
        stack = apply_y_crop(stack, Y_CROP)
        T, Z, Y, X = stack.shape
        print(f"After Y_CROP={Y_CROP} (bottom): T={T}, Z={Z}, Y={Y}, X={X}")

    allowed = make_allowed_region_mask(Z, Y, X)

    print("\n=== Building union mask ===")
    mask_files = discover_mask_files(MASK_FOLDER)
    print(f"Found {len(mask_files)} mask files in {MASK_FOLDER}")
    union = build_union_mask(mask_files, target_zyx=(Z, Y, X))

    cov_total = float(union.mean() * 100.0)
    print(f"Union mask coverage (total voxels): {cov_total:.2f}%")

    print("\n=== Computing F0 (last frames) ===")
    f0 = compute_f0_from_last_frames(stack, nframes=F0_NFRAMES)
    print(f"F0 shape: {f0.shape}")

    tissue = None
    tissue_thr = None
    denom_region = allowed

    if USE_TISSUE_MASK:
        tissue, tissue_thr = compute_tissue_mask_from_f0(f0, allowed)
        tissue_frac = float(tissue.mean() * 100.0)
        cov_tissue = float((union & tissue).sum() / max(1, tissue.sum()) * 100.0)
        print(f"Tissue mask fraction (of total): {tissue_frac:.2f}%")
        print(f"Union coverage within tissue voxels: {cov_tissue:.2f}%")
        print(f"Tissue F0 threshold ({TISSUE_MODE}): {tissue_thr:.3f}")
        denom_region = tissue  # use tissue region to set denom floor

    alpha = compute_denom_floor(f0, denom_region) if USE_DENOM_FLOOR else 0.0
    print(f"ΔF/F denom floor alpha: {alpha:.3f}  (mode={DENOM_FLOOR_MODE})")

    # Core masks (allowed region)
    all_mask = allowed
    inside = union & allowed
    outside = (~union) & allowed

    # Tissue-restricted masks
    if USE_TISSUE_MASK and tissue is not None:
        all_t = allowed & tissue
        inside_t = inside & tissue
        outside_t = outside & tissue
    else:
        all_t = inside_t = outside_t = None

    time_s = np.arange(T, dtype=np.float32) / float(FRAME_RATE)

    print("\n=== Computing main traces (ΔF/F) ===")
    all_dff, all_dF = compute_mean_trace_chunked(stack, f0, all_mask.reshape(-1), alpha)
    out_dff, out_dF = compute_mean_trace_chunked(stack, f0, outside.reshape(-1), alpha)
    in_dff, in_dF = compute_mean_trace_chunked(stack, f0, inside.reshape(-1), alpha)

    all_dff_s = smooth(all_dff)
    out_dff_s = smooth(out_dff)
    in_dff_s = smooth(in_dff)

    plot_overlay(
        time_s,
        all_dff_s, out_dff_s, in_dff_s,
        title=f"{DATE} | {MOUSE} | {RUN}  (allowed region)",
        labels=("All (allowed)", "Outside (allowed)", "Inside (allowed)"),
        out_png=OUT_PNG_MAIN if SAVE_FIG else None,
        ylab="ΔF/F"
    )
    
    # Additional plot: inside_masks, outside_masks_15microns, global average
    if DO_UM_RINGS:
        print("\n=== Computing 15µm outside ring for comparison plot ===")
        # Create far >15µm ring mask
        outside_allowed = (~union) & allowed
        dist_um = distance_transform_edt(~union, sampling=(VOXEL_Z_UM, VOXEL_Y_UM, VOXEL_X_UM)).astype(np.float32)
        far_gt15um = (dist_um > 15.0) & outside_allowed
        
        if far_gt15um.sum() > 200:
            far_dff, _ = compute_mean_trace_chunked(stack, f0, far_gt15um.reshape(-1), alpha)
            far_dff_s = smooth(far_dff)
            
            plot_overlay(
                time_s,
                in_dff_s, far_dff_s, all_dff_s,
                title=f"{DATE} | {MOUSE} | {RUN}  (inside vs far >15µm vs global)",
                labels=("Inside masks", "Outside masks >15µm", "Global average"),
                out_png=TRACE_FOLDER / "globalCa_comparison.png" if SAVE_FIG else None,
                ylab="ΔF/F"
            )

    # Tissue plot
    allT_dff_s = outT_dff_s = inT_dff_s = None
    if USE_TISSUE_MASK and tissue is not None:
        print("\n=== Computing tissue-only traces (ΔF/F) ===")
        allT_dff, _ = compute_mean_trace_chunked(stack, f0, all_t.reshape(-1), alpha)
        outT_dff, _ = compute_mean_trace_chunked(stack, f0, outside_t.reshape(-1), alpha)
        inT_dff, _ = compute_mean_trace_chunked(stack, f0, inside_t.reshape(-1), alpha)

        allT_dff_s = smooth(allT_dff)
        outT_dff_s = smooth(outT_dff)
        inT_dff_s = smooth(inT_dff)

        plot_overlay(
            time_s,
            allT_dff_s, outT_dff_s, inT_dff_s,
            title=f"{DATE} | {MOUSE} | {RUN}  (tissue-only)",
            labels=("All (tissue)", "Outside (tissue)", "Inside (tissue)"),
            out_png=OUT_PNG_TISSUE if SAVE_FIG else None,
            ylab="ΔF/F"
        )

    # Micron rings
    ring_traces = {}
    if DO_UM_RINGS:
        print("\n=== Micron-ring analysis (outside only) ===")
        rings = build_um_rings(union, allowed, RING_UM_EDGES)

        for name, m in rings.items():
            nvox = int(m.sum())
            print(f"{name}: {nvox} voxels")
            if nvox < 200:
                print(f"  skipping {name} (too few voxels)")
                continue
            tr, _ = compute_mean_trace_chunked(stack, f0, m.reshape(-1), alpha)
            ring_traces[name] = smooth(tr)

        if ring_traces:
            plot_many(
                time_s,
                ring_traces,
                out_png=OUT_PNG_RINGS if SAVE_FIG else None,
                title=f"{DATE} | {MOUSE} | {RUN}  outside rings vs distance-to-dendrite (µm)",
                ylab="ΔF/F"
            )

    # Optional ΔF plot
    if COMPUTE_DELTAF and all_dF is not None and out_dF is not None and in_dF is not None:
        print("\n=== Plotting absolute ΔF (sanity) ===")
        all_dF_s = smooth(all_dF)
        out_dF_s = smooth(out_dF)
        in_dF_s = smooth(in_dF)

        plot_overlay(
            time_s,
            all_dF_s, out_dF_s, in_dF_s,
            title=f"{DATE} | {MOUSE} | {RUN}  absolute ΔF (allowed region)",
            labels=("ΔF all", "ΔF outside", "ΔF inside"),
            out_png=OUT_PNG_DELTAF if SAVE_FIG else None,
            ylab="ΔF (a.u.)"
        )

    # Save bundle
    print("\n=== Saving outputs ===")
    out = {
        "time_s": time_s,
        "meta": {
            "DATE": DATE,
            "MOUSE": MOUSE,
            "RUN": RUN,
            "RAW_STACK_PATH": str(RAW_STACK_PATH),
            "MASK_FOLDER": str(MASK_FOLDER),
            "FRAME_RATE": FRAME_RATE,
            "CHUNK_T": CHUNK_T,
            "Y_CROP_bottom": Y_CROP,
            "EXCLUDE_TOP_ONLY_FOR_TRACES": EXCLUDE_TOP_ONLY_FOR_TRACES,
            "EXCLUDE_TOP_FRACTION": EXCLUDE_TOP_FRACTION,
            "F0_NFRAMES": F0_NFRAMES,
            "ARTIFACT_Z": ARTIFACT_Z,
            "SMOOTH_SIGMA": SMOOTH_SIGMA,
            "VOXEL_Z_UM": VOXEL_Z_UM,
            "VOXEL_Y_UM": VOXEL_Y_UM,
            "VOXEL_X_UM": VOXEL_X_UM,
            "USE_DENOM_FLOOR": USE_DENOM_FLOOR,
            "DENOM_FLOOR_MODE": DENOM_FLOOR_MODE,
            "DENOM_FLOOR_PCT": DENOM_FLOOR_PCT,
            "DENOM_FLOOR_ABS": DENOM_FLOOR_ABS,
            "DENOM_FLOOR_ALPHA": alpha,
            "USE_TISSUE_MASK": USE_TISSUE_MASK,
            "TISSUE_MODE": TISSUE_MODE,
            "TISSUE_F0_PCT": TISSUE_F0_PCT,
            "TISSUE_KSTD": TISSUE_KSTD,
            "TISSUE_F0_THRESHOLD": tissue_thr,
            "DO_UM_RINGS": DO_UM_RINGS,
            "RING_UM_EDGES": RING_UM_EDGES,
            "COMPUTE_DELTAF": COMPUTE_DELTAF,
        },
        "coverage_total_percent": float(union.mean() * 100.0),
        "coverage_within_tissue_percent": None,
        "f0_stats_allowed": {
            "min": float(np.nanmin(f0[allowed])),
            "p1": float(np.nanpercentile(f0[allowed], 1)),
            "p5": float(np.nanpercentile(f0[allowed], 5)),
            "p50": float(np.nanpercentile(f0[allowed], 50)),
            "p95": float(np.nanpercentile(f0[allowed], 95)),
            "p99": float(np.nanpercentile(f0[allowed], 99)),
            "max": float(np.nanmax(f0[allowed])),
        },
        "traces_dff_allowed": {
            "all": all_dff_s,
            "outside": out_dff_s,
            "inside": in_dff_s,
            "inside_minus_outside": in_dff_s - out_dff_s,
            "all_minus_outside": all_dff_s - out_dff_s,
        },
        "traces_dff_tissue": None,
        "ring_traces_dff_um": ring_traces if ring_traces else None,
        "traces_dF_allowed": None,
    }

    if USE_TISSUE_MASK and tissue is not None:
        out["coverage_within_tissue_percent"] = float((union & tissue).sum() / max(1, tissue.sum()) * 100.0)
        if allT_dff_s is not None:
            out["traces_dff_tissue"] = {
                "all": allT_dff_s,
                "outside": outT_dff_s,
                "inside": inT_dff_s,
                "inside_minus_outside": inT_dff_s - outT_dff_s,
                "all_minus_outside": allT_dff_s - outT_dff_s,
            }

    if COMPUTE_DELTAF and all_dF is not None:
        out["traces_dF_allowed"] = {
            "all": smooth(all_dF),
            "outside": smooth(out_dF),
            "inside": smooth(in_dF),
            "inside_minus_outside": smooth(in_dF) - smooth(out_dF),
            "all_minus_outside": smooth(all_dF) - smooth(out_dF),
        }

    np.save(OUT_NPY, out, allow_pickle=True)
    print(f"Saved: {OUT_NPY}")
    print("\nDone.")


if __name__ == "__main__":
    main()
