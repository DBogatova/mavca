#!/usr/bin/env python
"""
Concatenated ΔF/F traces across multiple runs sharing the same masks.

For each run:
  - Load raw stack, skip first N seconds
  - Compute F0 baseline (20th percentile) per run independently
  - Extract core ΔF/F traces using shared masks
  - Background-subtract using shell

Concatenate all runs and produce:
  - Combined CSV and PKL
  - Stacked trace plot with run boundaries marked
  - Per-ROI preview PDFs

Usage:
    python code/Traces-STEP3/concat_traces_multi_run.py
"""

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d, binary_erosion, binary_dilation
from skimage.morphology import ball
import pickle
import csv
import gc

mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['axes.unicode_minus'] = False

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUNS = ["run6", "run7"]

FRAME_RATE = 5  # Hz
SKIP_FIRST_SECONDS = 0.0
ARTIFACT_Z = -0.5
SMOOTH_SIGMA = 0.5
CHUNK_T = 100

# ===== PATHS =====
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "scape-data" / DATE / MOUSE

# Masks from run8 (shared across all runs)
MASK_RUN = "run7"
MASK_FOLDER = BASE / MASK_RUN / "labelmaps_curated_dynamic"

# Output goes into run8's traces folder
OUT_FOLDER = BASE / MASK_RUN / "traces_concat"
OUT_FOLDER.mkdir(exist_ok=True)
PREVIEW_FOLDER = BASE / MASK_RUN / "trace_previews_concat"
PREVIEW_FOLDER.mkdir(exist_ok=True)


def get_raw_path(run):
    base = BASE / run
    clean = base / "preprocessed" / "raw_clean.tif"
    if clean.exists():
        return clean
    return base / "raw" / f"runA_{run}_{MOUSE}-reslice-bin.tif"


def load_masks(Z, Y, X):
    """Load curated masks, build core/shell indices."""
    mask_paths = sorted(MASK_FOLDER.glob("dend_*.tif"))
    rois = []
    for path in mask_paths:
        name = path.stem.replace("_labelmap", "")
        m = tifffile.imread(path).astype(bool)
        mz, my, mx = m.shape
        if my < Y:
            m = np.pad(m, ((0,0),(0,Y-my),(0,0)), mode='constant')
        elif my > Y:
            m = m[:, :Y, :]
        if not m.any():
            continue
        core = binary_erosion(m, structure=ball(1))
        if not core.any():
            core = m.copy()
        shell = binary_dilation(m, structure=ball(3)) & ~m
        rois.append({
            "name": name,
            "mask": m,
            "core_idx": np.flatnonzero(core.ravel()),
            "shell_idx": np.flatnonzero(shell.ravel()) if shell.any() else np.array([], dtype=np.int64),
        })
        del m, core, shell
    return rois


def extract_traces_one_run(run, rois):
    """Extract ΔF/F traces for one run. Returns dict of name → trace array."""
    raw_path = get_raw_path(run)
    print(f"\n  Loading {run}: {raw_path.name}")
    tf = tifffile.TiffFile(str(raw_path))
    store = tf.series[0]
    shape = store.shape
    if len(shape) == 3:
        T_raw, Y, X = shape
        Z = 1
    else:
        T_raw, Z, Y, X = shape
    print(f"    Shape: T={T_raw}, Z={Z}, Y={Y}, X={X}")

    skip_frames = int(SKIP_FIRST_SECONDS * FRAME_RATE)
    T = T_raw - skip_frames
    print(f"    Skipping first {SKIP_FIRST_SECONDS}s ({skip_frames} frames) → T={T}")

    # Compute F0 baseline
    print(f"    Computing F0 baseline...")
    f0_data = []
    for t0 in range(skip_frames, T_raw, CHUNK_T):
        t1 = min(t0 + CHUNK_T, T_raw)
        chunk = np.asarray(store.asarray()[t0:t1]).astype(np.float32)
        if len(shape) == 3:
            chunk = chunk[:, np.newaxis, :, :]
        f0_data.append(chunk)
        if sum(c.shape[0] for c in f0_data) > 500:
            break
    f0_stack = np.concatenate(f0_data, axis=0)
    f0_vol = np.percentile(f0_stack, 20, axis=0)
    del f0_data, f0_stack
    gc.collect()

    # Extract traces
    traces = {roi["name"]: np.empty(T, dtype=np.float32) for roi in rois}

    frame_idx = 0
    for t0 in range(skip_frames, T_raw, CHUNK_T):
        t1 = min(t0 + CHUNK_T, T_raw)
        chunk = np.asarray(store.asarray()[t0:t1]).astype(np.float32)
        if len(shape) == 3:
            chunk = chunk[:, np.newaxis, :, :]

        for t_rel in range(chunk.shape[0]):
            vol = (chunk[t_rel] - f0_vol) / (f0_vol + 1e-6)
            vol_flat = vol.ravel()
            for roi in rois:
                core_val = vol_flat[roi["core_idx"]].mean()
                if roi["shell_idx"].size > 0:
                    shell_val = vol_flat[roi["shell_idx"]].mean()
                else:
                    shell_val = 0.0
                traces[roi["name"]][frame_idx] = core_val - shell_val
            frame_idx += 1

        print(f"    Processed frames {t0}-{t1-1}")
        del chunk
        gc.collect()

    tf.close()
    return traces, T


def main():
    print(f"=== Concatenated Traces: {DATE}/{MOUSE} ===")
    print(f"  Runs: {RUNS}")
    print(f"  Masks from: {MASK_RUN}")

    # Peek at first run for dimensions
    raw0 = get_raw_path(RUNS[0])
    tf = tifffile.TiffFile(str(raw0))
    shape = tf.series[0].shape
    tf.close()
    if len(shape) == 3:
        _, Y, X = shape; Z = 1
    else:
        _, Z, Y, X = shape

    # Load masks
    print(f"\nLoading masks from {MASK_FOLDER}...")
    rois = load_masks(Z, Y, X)
    print(f"  {len(rois)} ROIs loaded")

    # Extract traces per run
    all_traces = {roi["name"]: [] for roi in rois}
    run_lengths = []
    run_boundaries = [0]

    for run in RUNS:
        traces, T = extract_traces_one_run(run, rois)
        run_lengths.append(T)
        run_boundaries.append(run_boundaries[-1] + T)
        for name in all_traces:
            all_traces[name].append(traces[name])

    # Concatenate
    total_T = sum(run_lengths)
    print(f"\nTotal frames: {total_T} ({total_T/FRAME_RATE:.1f}s)")

    # Bleach correction: per-run exponential baseline
    # Each run gets its own fit so inter-run differences don't bias the correction
    print("Bleach correction (per-run exponential baseline)...")
    from scipy.optimize import curve_fit

    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

    labels = []
    traces_pct = []
    for roi in rois:
        name = roi["name"]
        cat = np.concatenate(all_traces[name])
        cat[cat < ARTIFACT_Z] = 0.0

        # Correct each run segment independently
        offset = 0
        for j, rl in enumerate(run_lengths):
            seg = cat[offset:offset+rl]
            t_fit = np.arange(rl, dtype=np.float64)
            smooth_seg = gaussian_filter1d(seg, sigma=FRAME_RATE * 10)
            try:
                p0 = [smooth_seg[0] - smooth_seg[-1], 1e-4, smooth_seg[-1]]
                popt, _ = curve_fit(exp_decay, t_fit, smooth_seg, p0=p0, maxfev=5000)
                baseline = exp_decay(t_fit, *popt)
                cat[offset:offset+rl] = seg - baseline + np.median(baseline)
            except (RuntimeError, ValueError):
                cat[offset:offset+rl] = seg - np.median(seg)
            offset += rl

        smoothed = gaussian_filter1d(cat, sigma=SMOOTH_SIGMA) * 100.0
        labels.append(name)
        traces_pct.append(smoothed.astype(np.float32))
    print(f"  Corrected {len(labels)} traces across {len(RUNS)} runs")

    # Save CSV
    csv_path = OUT_FOLDER / "dff_traces_concat.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frame"] + labels)
        for t in range(total_T):
            w.writerow([t] + [f"{traces_pct[i][t]:.4f}" for i in range(len(labels))])
    print(f"✅ CSV: {csv_path}")

    # Save PKL
    pkl_path = OUT_FOLDER / "dff_traces_concat.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(list(zip(labels, traces_pct)), f)
    print(f"✅ PKL: {pkl_path}")

    # ===== Stacked plot =====
    t_axis = np.arange(total_T) / float(FRAME_RATE)
    offset = 6.0

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (name, trace) in enumerate(zip(labels, traces_pct)):
        ax.plot(t_axis, trace + i * offset,
                color=cm.turbo(i / max(1, len(traces_pct))), lw=0.8)
        try:
            cell_label = f"Cell {name.split('_')[1]}"
        except Exception:
            cell_label = name
        ax.text(t_axis[-1] + 1, i * offset, cell_label, va='center', fontsize=7)

    # Run boundaries — semi-transparent vertical lines
    for b in run_boundaries[1:-1]:
        t_sec = b / FRAME_RATE
        ax.axvline(t_sec, color='red', ls='-', lw=1.5, alpha=0.4, zorder=5)

    # Run labels at top
    for j, run in enumerate(RUNS):
        mid = (run_boundaries[j] + run_boundaries[j+1]) / 2 / FRAME_RATE
        ax.text(mid, len(labels) * offset + 2, run, ha='center', fontsize=9, color='gray')

    # Scale bar
    if len(labels) > 0:
        sx = t_axis[-1] - 7
        sy = offset * (len(labels) - 0.5)
        ax.plot([sx, sx], [sy, sy + 10], color='k', lw=2)
        ax.text(sx + 1, sy + 1, "10%", va='center', fontsize=9)

    ax.set_xlim(0, t_axis[-1])
    ax.set_ylim(-offset, len(labels) * offset + 5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (%) + offset")
    ax.set_yticks([])
    ax.set_title(f"Concatenated ΔF/F — {' + '.join(RUNS)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = PREVIEW_FOLDER / "combo_traces_concat.pdf"
    fig.savefig(plot_path, format="pdf")
    plt.close(fig)
    print(f"✅ Plot: {plot_path}")

    # ===== Per-ROI previews (MIP + trace) =====
    print("Saving per-ROI previews...")
    for i, roi in enumerate(rois):
        name = roi["name"]
        mip = roi["mask"].max(axis=0).astype(bool)

        fig, (ax_img, ax_trace) = plt.subplots(
            1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 3]}
        )

        # Left: colored MIP
        rgb = np.zeros((*mip.shape, 3), dtype=np.float32)
        color = (0.2, 0.7, 0.2)
        for c in range(3):
            rgb[..., c][mip] = color[c]
        ax_img.imshow(rgb)
        try:
            cell_number = name.split('_')[1]
            title_left = f"Cell {cell_number} — MIP"
        except Exception:
            title_left = f"{name} — MIP"
        ax_img.set_title(title_left)
        ax_img.axis("off")

        # Right: ΔF/F trace with run boundaries
        ax_trace.plot(t_axis, traces_pct[i], color='teal', lw=1.2)
        for b in run_boundaries[1:-1]:
            t_sec = b / FRAME_RATE
            ax_trace.axvline(t_sec, color='red', ls='-', lw=1.5, alpha=0.4)
        # Run labels
        for j, run in enumerate(RUNS):
            mid = (run_boundaries[j] + run_boundaries[j+1]) / 2 / FRAME_RATE
            ax_trace.text(mid, ax_trace.get_ylim()[1] if i > 0 else traces_pct[i].max() * 0.9,
                          run, ha='center', fontsize=8, color='gray', va='top')
        ax_trace.set_title("ΔF/F Trace (%)")
        ax_trace.set_xlabel("Time (s)")
        ax_trace.set_ylabel("ΔF/F (%)")
        ax_trace.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(PREVIEW_FOLDER / f"{name}_trace_mip.pdf", format="pdf")
        plt.close(fig)

    print(f"✅ {len(rois)} per-ROI previews saved to {PREVIEW_FOLDER}")

    # ===== Curated combo plot: top N by SNR =====
    TOP_N = 15
    print(f"\nRanking traces by SNR for top-{TOP_N} plot...")

    # SNR = 95th percentile / MAD of baseline
    snr_list = []
    for i, (name, trace) in enumerate(zip(labels, traces_pct)):
        peak = np.percentile(trace, 95)
        # Baseline noise: MAD of the lower 50% of values
        low_half = trace[trace < np.median(trace)]
        if len(low_half) > 10:
            noise = np.median(np.abs(low_half - np.median(low_half))) + 1e-6
        else:
            noise = np.std(trace) + 1e-6
        snr = peak / noise
        snr_list.append((i, name, snr, peak))

    snr_list.sort(key=lambda x: x[2], reverse=True)
    top = snr_list[:TOP_N]

    print("  Selected:")
    for rank, (idx, name, snr, peak) in enumerate(top):
        print(f"    {rank+1:2d}. {name}  SNR={snr:.1f}  peak={peak:.1f}%")

    fig, ax = plt.subplots(figsize=(14, 6))
    offset_v = 8.0
    for rank, (idx, name, snr, peak) in enumerate(top):
        trace = traces_pct[idx]
        ax.plot(t_axis, trace + rank * offset_v,
                color=cm.turbo(rank / max(1, TOP_N)), lw=1.0)
        try:
            cell_label = f"Cell {name.split('_')[1]}"
        except Exception:
            cell_label = name
        ax.text(t_axis[-1] + 1, rank * offset_v, cell_label, va='center', fontsize=8)

    for b in run_boundaries[1:-1]:
        t_sec = b / FRAME_RATE
        ax.axvline(t_sec, color='red', ls='-', lw=1.5, alpha=0.4, zorder=5)

    for j, run in enumerate(RUNS):
        mid = (run_boundaries[j] + run_boundaries[j+1]) / 2 / FRAME_RATE
        ax.text(mid, TOP_N * offset_v + 2, run, ha='center', fontsize=9, color='gray')

    # Scale bar
    sx = t_axis[-1] - 7
    sy = offset_v * (TOP_N - 0.5)
    ax.plot([sx, sx], [sy, sy + 10], color='k', lw=2)
    ax.text(sx + 1, sy + 1, "10%", va='center', fontsize=9)

    ax.set_xlim(0, t_axis[-1])
    ax.set_ylim(-offset_v, TOP_N * offset_v + 5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (%) + offset")
    ax.set_yticks([])
    ax.set_title(f"Top {TOP_N} ΔF/F Traces (by SNR) — {' + '.join(RUNS)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    curated_path = PREVIEW_FOLDER / "combo_traces_top_snr.pdf"
    fig.savefig(curated_path, format="pdf")
    plt.close(fig)
    print(f"✅ Curated plot: {curated_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
