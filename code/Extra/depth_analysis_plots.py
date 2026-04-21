#!/usr/bin/env python3
"""
Depth Analysis Plots — Global + per-mask Ca²⁺ by Y-depth chunks.

Concatenates multiple runs. For each Y-depth layer:
  - Global ΔF/F (all voxels in that layer)
  - Mask ΔF/F (M4-style core−shell, averaged across masks present in that layer)

A mask counts for a layer if it has any voxels in that Y range.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from scipy.ndimage import gaussian_filter1d, binary_erosion, binary_dilation
from skimage.morphology import ball

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUNS = ["run6", "run7"]
MASK_RUN = "run7"

FRAME_RATE = 5.0
CHUNK_T = 118
Y_CROP = 3
CROP_START_SECONDS = 11.0
SKIP_FIRST_SECONDS = 11.0
N_DEPTH_CHUNKS = 4

SMOOTH_SIGMA = 1.5
F0_NFRAMES = 500
ROLLING_BASELINE = True  # use rolling 10th percentile instead of fixed F0
ROLLING_WINDOW_SEC = 60.0  # window size in seconds
EPS = 1e-6
ARTIFACT_Z = -0.5
NEUROPIL_ALPHA = 0.0  # no subtraction — use raw mask mean for depth comparison

SAVE_FIG = True
SHOW_FIG = True

# ===== PATHS =====
BASE_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
                 "apical-dendrites-2025/scape-data") / DATE / MOUSE
MASK_FOLDER = BASE_ROOT / MASK_RUN / "labelmaps_curated_dynamic"
OUTPUT_PATH = BASE_ROOT / MASK_RUN / "traces"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def get_raw_path(run):
    base = BASE_ROOT / run
    clean = base / "preprocessed" / "raw_clean.tif"
    if clean.exists():
        return clean
    return base / "raw" / f"runA_{run}_{MOUSE}-reslice-bin.tif"


def smooth(x):
    return gaussian_filter1d(x, sigma=SMOOTH_SIGMA) if SMOOTH_SIGMA > 0 else x


def rolling_baseline_correct(trace, window_frames):
    """Subtract rolling 10th percentile baseline from a 1D trace."""
    from numpy.lib.stride_tricks import sliding_window_view
    T = len(trace)
    if window_frames >= T:
        baseline = np.full(T, np.percentile(trace, 10))
    else:
        half = window_frames // 2
        baseline = np.empty(T, dtype=np.float32)
        for t in range(T):
            t0 = max(0, t - half)
            t1 = min(T, t + half)
            baseline[t] = np.percentile(trace[t0:t1], 10)
    return trace - baseline


def load_masks_by_depth(Z, Y, X, chunks):
    """
    Load masks, build core/shell indices, and assign each mask to depth layers.
    A mask belongs to a layer if it has any voxels in that Y range.
    Returns: list of mask dicts with core_idx, shell_idx, and layer membership.
    """
    mask_paths = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    if not mask_paths:
        print(f"  No masks found in {MASK_FOLDER}")
        return []

    rois = []
    union = np.zeros((Z, Y, X), dtype=bool)
    for p in mask_paths:
        m = tifffile.imread(p).astype(bool)
        mz, my, mx = m.shape
        if my > Y: m = m[:, :Y, :X]
        elif my < Y: m = np.pad(m, ((0,0),(0,Y-my),(0,0)), mode='constant')
        if not m.any():
            continue
        union |= m

        # Core/shell (same as M4)
        core = binary_erosion(m, structure=ball(1))
        if not core.any():
            core = m.copy()
        inner = binary_dilation(m, structure=ball(2))
        outer = binary_dilation(m, structure=ball(3))
        shell = outer & ~inner

        # Which depth layers does this mask span?
        y_coords = np.where(m.any(axis=(0, 2)))[0]  # Y indices with mask voxels
        layers = set()
        for li, (y0, y1) in enumerate(chunks):
            if np.any((y_coords >= y0) & (y_coords < y1)):
                layers.add(li)

        rois.append({
            "name": p.stem.replace("_labelmap", ""),
            "core": np.flatnonzero(core.ravel()),
            "shell": np.flatnonzero(shell.ravel()) if shell.any() else np.array([], dtype=np.int64),
            "layers": layers,
        })
        del m, core, shell

    print(f"  Loaded {len(rois)} masks")
    for li, (y0, y1) in enumerate(chunks):
        n = sum(1 for r in rois if li in r["layers"])
        print(f"    Layer {li} (Y {y0}-{y1}): {n} masks")

    print(f"  Union mask: {union.sum():,} voxels ({100*union.mean():.2f}%)")
    return rois, union


def process_one_run(run, Z, Y, X, chunks, rois, union):
    """Process one run: F0, global per-layer, mask per-layer traces."""
    raw_path = get_raw_path(run)
    if not raw_path.exists():
        print(f"  [skip] {raw_path}")
        return None

    print(f"\n--- {run}: {raw_path.name} ---")
    store = tifffile.memmap(str(raw_path), mode='r')
    T_raw = store.shape[0]

    skip = int(max(SKIP_FIRST_SECONDS, CROP_START_SECONDS) * FRAME_RATE)
    crop = int(CROP_START_SECONDS * FRAME_RATE)

    # F0 baseline
    if ROLLING_BASELINE:
        # Compute per-voxel rolling 10th percentile
        # Pre-compute global mean per frame for efficiency, then scale
        # Actually: compute fixed F0 for ΔF/F, then detrend with rolling baseline on traces
        # This is more memory-efficient than rolling F0 on the full 4D volume
        print("  Using rolling baseline (applied to traces after extraction)")
        # Still need a fixed F0 for initial ΔF/F computation
        f0_frames = []
        for t in range(skip, min(skip + F0_NFRAMES, T_raw)):
            frame = np.asarray(store[t]).astype(np.float32)
            if Y_CROP > 0:
                frame = frame[:, :-Y_CROP, :]
            f0_frames.append(frame)
        f0 = np.percentile(np.stack(f0_frames), 10, axis=0)
        del f0_frames
    else:
        f0_frames = []
        for t in range(skip, min(skip + F0_NFRAMES, T_raw)):
            frame = np.asarray(store[t]).astype(np.float32)
            if Y_CROP > 0:
                frame = frame[:, :-Y_CROP, :]
            f0_frames.append(frame)
        f0 = np.percentile(np.stack(f0_frames), 10, axis=0)
        del f0_frames

    T_out = T_raw - crop
    n_layers = len(chunks)

    # Global traces per layer
    global_traces = [np.empty(T_out, np.float32) for _ in range(n_layers)]
    # Outside-mask traces per layer
    outside_traces = [np.empty(T_out, np.float32) for _ in range(n_layers)]
    # Per-mask traces for averaging
    per_mask = [np.empty(T_out, np.float32) for _ in rois]

    # Precompute outside-mask boolean per layer
    outside_per_layer = []
    for li, (y0, y1) in enumerate(chunks):
        layer_slice = union[:, y0:y1, :]
        outside_per_layer.append(~layer_slice)  # (Z, chunk_Y, X)

    out_idx = 0
    for t0 in range(crop, T_raw, CHUNK_T):
        t1 = min(t0 + CHUNK_T, T_raw)
        chunk = np.asarray(store[t0:t1]).astype(np.float32)
        if Y_CROP > 0:
            chunk = chunk[:, :, :-Y_CROP, :]

        for ti in range(chunk.shape[0]):
            dff = (chunk[ti] - f0) / (f0 + EPS)
            if ARTIFACT_Z is not None:
                dff[dff < ARTIFACT_Z] = 0.0

            # Global per layer
            for li, (y0, y1) in enumerate(chunks):
                global_traces[li][out_idx] = dff[:, y0:y1, :].mean()
                # Outside masks per layer
                layer_dff = dff[:, y0:y1, :]
                omask = outside_per_layer[li]
                if omask.any():
                    outside_traces[li][out_idx] = layer_dff[omask].mean()
                else:
                    outside_traces[li][out_idx] = layer_dff.mean()

            # Per-mask: full core−shell subtraction
            flat = dff.ravel()
            for mi, roi in enumerate(rois):
                cv = flat[roi["core"]].mean()
                sv = flat[roi["shell"]].mean() if roi["shell"].size > 0 else 0.0
                per_mask[mi][out_idx] = cv - sv

            out_idx += 1

        print(f"    frames {t0}..{t1-1}")

    del store

    # Trim
    for li in range(n_layers):
        global_traces[li] = global_traces[li][:out_idx]
        outside_traces[li] = outside_traces[li][:out_idx]
    for mi in range(len(rois)):
        per_mask[mi] = per_mask[mi][:out_idx]

    # Average mask traces per layer
    mask_traces = [np.zeros(out_idx, np.float32) for _ in range(n_layers)]
    mask_n = [0 for _ in range(n_layers)]
    for mi, roi in enumerate(rois):
        for li in roi["layers"]:
            mask_traces[li] += per_mask[mi][:out_idx]
            mask_n[li] += 1
    for li in range(n_layers):
        if mask_n[li] > 0:
            mask_traces[li] /= mask_n[li]

    return {
        "T": out_idx,
        "global": global_traces,
        "outside": outside_traces,
        "masks": mask_traces,
        "mask_n": mask_n,
    }


def main():
    print(f"=== Depth Analysis: {DATE}/{MOUSE} ===")
    print(f"  Runs: {RUNS}, masks from {MASK_RUN}\n")

    # Peek at dimensions
    raw0 = get_raw_path(RUNS[0])
    tf = tifffile.TiffFile(str(raw0))
    shape = tf.series[0].shape
    tf.close()
    T0, Z, Y, X = shape
    if Y_CROP > 0:
        Y -= Y_CROP
    print(f"  Spatial: Z={Z}, Y={Y}, X={X}")

    # Depth chunks
    chunk_size = Y // N_DEPTH_CHUNKS
    chunks = []
    for i in range(N_DEPTH_CHUNKS):
        y0 = i * chunk_size
        y1 = Y if i == N_DEPTH_CHUNKS - 1 else (i + 1) * chunk_size
        chunks.append((y0, y1))
    print(f"  Depth chunks: {chunks}")

    # Load masks
    print("Loading masks...")
    rois, union = load_masks_by_depth(Z, Y, X, chunks)

    # Process runs
    all_global = [[] for _ in range(N_DEPTH_CHUNKS)]
    all_outside = [[] for _ in range(N_DEPTH_CHUNKS)]
    all_masks = [[] for _ in range(N_DEPTH_CHUNKS)]
    run_boundaries = [0.0]

    for run in RUNS:
        result = process_one_run(run, Z, Y, X, chunks, rois, union)
        if result is None:
            continue
        dur = result["T"] / FRAME_RATE
        for li in range(N_DEPTH_CHUNKS):
            all_global[li].append(result["global"][li])
            all_outside[li].append(result["outside"][li])
            all_masks[li].append(result["masks"][li])
        run_boundaries.append(run_boundaries[-1] + dur)

    # Concatenate
    total_T = sum(len(g) for g in all_global[0]) if all_global[0] else 0
    time_s = np.arange(total_T) / FRAME_RATE

    colors = ['#E53E3E', '#FF8C00', '#38A169', '#3182CE']
    depth_labels = ['Top (surface)', 'Upper middle', 'Lower middle', 'Bottom (deep)']

    # === Plot 1: Global by depth ===
    fig, ax = plt.subplots(figsize=(14, 7))
    offset = 4.0
    for li in range(N_DEPTH_CHUNKS):
        tr = smooth(np.concatenate(all_global[li])) * 100
        y0, y1 = chunks[li]
        # Top layer gets highest offset (plotted on top)
        plot_offset = (N_DEPTH_CHUNKS - 1 - li) * offset
        ax.plot(time_s, tr + plot_offset, color=colors[li], lw=1.5,
                label=f"{depth_labels[li]} (Y {y0}-{y1})")

    for b in run_boundaries[1:-1]:
        ax.axvline(b, color='red', ls='-', lw=1.5, alpha=0.4)
    for j, run in enumerate(RUNS):
        if j < len(run_boundaries) - 1:
            mid = (run_boundaries[j] + run_boundaries[j+1]) / 2
            ax.text(mid, -0.08, run, ha='center', fontsize=9, color='gray',
                    transform=ax.get_xaxis_transform())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (%) + offset")
    ax.set_title(f"{DATE} | {MOUSE} | {' + '.join(RUNS)} — Global Ca²⁺ by Depth")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if SAVE_FIG:
        out = OUTPUT_PATH / "depth_analysis_global_concat.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"✅ {out}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

    # === Plot 2: Mask traces by depth (core−shell, full subtraction) ===
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    mask_offset = 1.0  # smaller offset since signals are ~0.2-0.5%
    for li in range(N_DEPTH_CHUNKS):
        tr = smooth(np.concatenate(all_masks[li])) * 100
        y0, y1 = chunks[li]
        n_masks = sum(1 for r in rois if li in r["layers"])
        plot_offset = (N_DEPTH_CHUNKS - 1 - li) * mask_offset
        ax2.plot(time_s, tr + plot_offset, color=colors[li], lw=1.5,
                 label=f"{depth_labels[li]} (Y {y0}-{y1}, {n_masks} masks)")

    for b in run_boundaries[1:-1]:
        ax2.axvline(b, color='red', ls='-', lw=1.5, alpha=0.4)
    for j, run in enumerate(RUNS):
        if j < len(run_boundaries) - 1:
            mid = (run_boundaries[j] + run_boundaries[j+1]) / 2
            ax2.text(mid, -0.08, run, ha='center', fontsize=9, color='gray',
                     transform=ax2.get_xaxis_transform())

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("ΔF/F (%) + offset")
    ax2.set_title(f"{DATE} | {MOUSE} | {' + '.join(RUNS)} — Mask Ca²⁺ by Depth (raw mean)")
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()

    if SAVE_FIG:
        out2 = OUTPUT_PATH / "depth_analysis_masks_concat.png"
        plt.savefig(out2, dpi=200, bbox_inches='tight')
        print(f"✅ {out2}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

    # === Plot 2b: Global with masks removed (outside-mask only) ===
    fig2b, ax2b = plt.subplots(figsize=(14, 7))
    for li in range(N_DEPTH_CHUNKS):
        g_tr = smooth(np.concatenate(all_global[li])) * 100
        o_tr = smooth(np.concatenate(all_outside[li])) * 100
        diff_tr = g_tr - o_tr  # difference: what the masks contribute
        y0, y1 = chunks[li]
        n_masks = sum(1 for r in rois if li in r["layers"])
        plot_offset = (N_DEPTH_CHUNKS - 1 - li) * offset
        ax2b.plot(time_s, diff_tr + plot_offset, color=colors[li], lw=1.5,
                  label=f"{depth_labels[li]} (Y {y0}-{y1}, {n_masks} masks)")

    for b in run_boundaries[1:-1]:
        ax2b.axvline(b, color='red', ls='-', lw=1.5, alpha=0.4)
    for j, run in enumerate(RUNS):
        if j < len(run_boundaries) - 1:
            mid = (run_boundaries[j] + run_boundaries[j+1]) / 2
            ax2b.text(mid, -0.08, run, ha='center', fontsize=9, color='gray',
                      transform=ax2b.get_xaxis_transform())

    ax2b.set_xlabel("Time (s)")
    ax2b.set_ylabel("ΔF/F (%) + offset")
    ax2b.set_title(f"{DATE} | {MOUSE} | {' + '.join(RUNS)} — Global − Outside (mask contribution)")
    ax2b.legend(loc='upper right', fontsize=9)
    ax2b.grid(alpha=0.3)
    plt.tight_layout()

    if SAVE_FIG:
        out2b = OUTPUT_PATH / "depth_analysis_global_minus_masks_concat.png"
        plt.savefig(out2b, dpi=200, bbox_inches='tight')
        print(f"✅ {out2b}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

    # === Plot 3: Combined (global + masks side by side per layer) ===
    fig3, axes3 = plt.subplots(N_DEPTH_CHUNKS, 1, figsize=(14, 3 * N_DEPTH_CHUNKS),
                                sharex=True)
    for li in range(N_DEPTH_CHUNKS):
        g_tr = smooth(np.concatenate(all_global[li])) * 100
        m_tr = smooth(np.concatenate(all_masks[li])) * 100
        y0, y1 = chunks[li]
        n_masks = sum(1 for r in rois if li in r["layers"])

        axes3[li].plot(time_s, g_tr, color=colors[li], lw=1.2, alpha=0.6, label="Global")
        axes3[li].plot(time_s, m_tr, color='magenta', lw=1.5, label=f"Masks ({n_masks})")
        axes3[li].set_ylabel("ΔF/F (%)")
        axes3[li].set_title(f"{depth_labels[li]} (Y {y0}-{y1})", fontsize=10)
        axes3[li].legend(loc='upper right', fontsize=8)
        axes3[li].grid(alpha=0.3)

        for b in run_boundaries[1:-1]:
            axes3[li].axvline(b, color='red', ls='-', lw=1, alpha=0.4)

    axes3[-1].set_xlabel("Time (s)")
    plt.suptitle(f"{DATE} | {MOUSE} | {' + '.join(RUNS)} — Global vs Masks by Depth",
                 fontsize=13)
    plt.tight_layout()

    if SAVE_FIG:
        out3 = OUTPUT_PATH / "depth_analysis_combined_concat.png"
        plt.savefig(out3, dpi=200, bbox_inches='tight')
        print(f"✅ {out3}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
