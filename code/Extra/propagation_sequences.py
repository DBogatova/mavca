#!/usr/bin/env python
"""
Detect and visualize sequential activation patterns across dendrites.

Finds repeating propagation sequences (A→B→C with consistent 200ms delays)
and plots them spatially using mask centroids, plus temporal raster plots.

Usage:
    python code/Extra/propagation_sequences.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from collections import defaultdict
from itertools import combinations

# ===== CONFIG =====
DATE = "2026-04-16"
MOUSE = "rbp4_132_phpeb"
RUN = "run1"

FRAME_RATE = 5.0  # Hz
VOXEL_SIZE = (3.9, 1.0, 1.2)  # Z, Y, X in µm
DFF_THRESHOLD = 1.5  # event detection threshold
MIN_SEQ_REPEATS = 3  # minimum repetitions to count as a sequence
MAX_LAG = 2  # max frames between sequential activations (400ms)

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
TRACE_CSV = BASE / "traces" / "dff_traces_curated_bgsub.csv"
OUTPUT = BASE / "traces" / "propagation_sequences.png"


def load_centroids():
    """Compute centroid (in µm) for each mask."""
    centroids = {}
    mask_paths = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    for path in mask_paths:
        name = path.stem.replace("_labelmap", "")
        m = tifffile.imread(path).astype(bool)
        if not m.any():
            continue
        coords = np.argwhere(m)  # (N, 3) in Z, Y, X voxel coords
        centroid_vox = coords.mean(axis=0)
        centroid_um = centroid_vox * np.array(VOXEL_SIZE)
        centroids[name] = centroid_um
    return centroids


def detect_events(traces_df):
    """Detect event onsets for each dendrite. Returns dict of name → onset frames."""
    dend_cols = [c for c in traces_df.columns if 'dend' in c]
    onsets = {}
    for col in dend_cols:
        tr = traces_df[col].values
        above = tr > DFF_THRESHOLD
        # Onset = first frame crossing threshold (rising edge)
        edges = np.diff(above.astype(int))
        onset_frames = np.where(edges == 1)[0] + 1
        onsets[col] = onset_frames
    return onsets


def find_pair_sequences(onsets, max_lag=MAX_LAG):
    """Find directed pairs where A fires before B within max_lag frames."""
    dend_names = list(onsets.keys())
    pair_counts = defaultdict(int)
    pair_instances = defaultdict(list)

    for i, a in enumerate(dend_names):
        for b in dend_names[i+1:]:
            for oa in onsets[a]:
                for ob in onsets[b]:
                    lag = ob - oa
                    if 1 <= lag <= max_lag:
                        pair_counts[(a, b)] += 1
                        pair_instances[(a, b)].append(oa)
                    elif 1 <= -lag <= max_lag:
                        pair_counts[(b, a)] += 1
                        pair_instances[(b, a)].append(ob)
    return pair_counts, pair_instances


def find_triplet_sequences(onsets, max_lag=MAX_LAG):
    """Find A→B→C sequences where each step is within max_lag frames."""
    dend_names = list(onsets.keys())
    triplet_counts = defaultdict(int)
    triplet_instances = defaultdict(list)

    # Build onset lookup for fast search
    onset_set = {name: set(frames) for name, frames in onsets.items()}

    for a in dend_names:
        for oa in onsets[a]:
            # Find B that fires 1-max_lag frames after A
            for b in dend_names:
                if b == a:
                    continue
                for lag1 in range(1, max_lag + 1):
                    if (oa + lag1) in onset_set[b]:
                        # Find C that fires 1-max_lag frames after B
                        for c in dend_names:
                            if c == a or c == b:
                                continue
                            for lag2 in range(1, max_lag + 1):
                                if (oa + lag1 + lag2) in onset_set[c]:
                                    triplet_counts[(a, b, c)] += 1
                                    triplet_instances[(a, b, c)].append(oa)
    return triplet_counts, triplet_instances


def plot_sequences(centroids, pair_counts, triplet_counts, traces_df, onsets):
    """Create visualization of propagation sequences."""
    dend_cols = [c for c in traces_df.columns if 'dend' in c]

    # Filter significant sequences
    sig_pairs = {k: v for k, v in pair_counts.items() if v >= MIN_SEQ_REPEATS}
    sig_triplets = {k: v for k, v in triplet_counts.items() if v >= MIN_SEQ_REPEATS}

    print(f"\n=== RESULTS ===")
    print(f"Significant pairs (≥{MIN_SEQ_REPEATS} repeats): {len(sig_pairs)}")
    print(f"Significant triplets (≥{MIN_SEQ_REPEATS} repeats): {len(sig_triplets)}")

    if sig_pairs:
        print("\nTop 15 directed pairs:")
        for (a, b), count in sorted(sig_pairs.items(), key=lambda x: -x[1])[:15]:
            dist = np.linalg.norm(centroids.get(a, [0,0,0]) - centroids.get(b, [0,0,0]))
            print(f"  {a} → {b}: {count}x (dist={dist:.0f} µm)")

    if sig_triplets:
        print(f"\nTop 10 triplet sequences:")
        for (a, b, c), count in sorted(sig_triplets.items(), key=lambda x: -x[1])[:10]:
            print(f"  {a} → {b} → {c}: {count}x")

    # --- FIGURE ---
    fig = plt.figure(figsize=(18, 12))

    # Panel 1: Spatial map (Y vs X) with arrows for top sequences
    ax1 = fig.add_subplot(221)
    # Plot all centroids
    for name, pos in centroids.items():
        z, y, x = pos
        ax1.scatter(x, y, s=30, c='gray', alpha=0.4, zorder=1)
        ax1.text(x + 2, y + 2, name.split('_')[1], fontsize=5, alpha=0.5)

    # Draw arrows for top pairs
    colors = plt.cm.hot(np.linspace(0.2, 0.8, min(10, len(sig_pairs))))
    for idx, ((a, b), count) in enumerate(sorted(sig_pairs.items(), key=lambda x: -x[1])[:10]):
        if a in centroids and b in centroids:
            za, ya, xa = centroids[a]
            zb, yb, xb = centroids[b]
            ax1.annotate("", xy=(xb, yb), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="->", color=colors[idx],
                                       lw=1.5 + count * 0.3, alpha=0.8))
            mid_x, mid_y = (xa + xb) / 2, (ya + yb) / 2
            ax1.text(mid_x, mid_y, f"{count}x", fontsize=7, color=colors[idx],
                    ha='center', fontweight='bold')

    ax1.set_xlabel("X (µm)")
    ax1.set_ylabel("Y depth (µm)")
    ax1.set_title("Spatial Map — Propagation Pairs")
    ax1.invert_yaxis()

    # Panel 2: Spatial map (Y vs Z)
    ax2 = fig.add_subplot(222)
    for name, pos in centroids.items():
        z, y, x = pos
        ax2.scatter(z, y, s=30, c='gray', alpha=0.4, zorder=1)

    for idx, ((a, b), count) in enumerate(sorted(sig_pairs.items(), key=lambda x: -x[1])[:10]):
        if a in centroids and b in centroids:
            za, ya, xa = centroids[a]
            zb, yb, xb = centroids[b]
            ax2.annotate("", xy=(zb, yb), xytext=(za, ya),
                        arrowprops=dict(arrowstyle="->", color=colors[idx],
                                       lw=1.5 + count * 0.3, alpha=0.8))

    ax2.set_xlabel("Z (µm)")
    ax2.set_ylabel("Y depth (µm)")
    ax2.set_title("Spatial Map — Z vs Depth")
    ax2.invert_yaxis()

    # Panel 3: Raster of top sequence participants
    ax3 = fig.add_subplot(212)
    # Get all dendrites involved in top sequences
    seq_dends = set()
    top_pairs = sorted(sig_pairs.items(), key=lambda x: -x[1])[:10]
    for (a, b), _ in top_pairs:
        seq_dends.add(a)
        seq_dends.add(b)
    seq_dends = sorted(seq_dends)

    if seq_dends:
        time = np.arange(len(traces_df)) / FRAME_RATE
        for i, name in enumerate(seq_dends):
            if name in onsets and len(onsets[name]) > 0:
                onset_times = onsets[name] / FRAME_RATE
                ax3.scatter(onset_times, np.full_like(onset_times, i),
                           s=15, c='black', zorder=2)
            # Light trace background
            if name in traces_df.columns:
                tr = traces_df[name].values
                tr_norm = np.clip(tr / 5.0, 0, 1)  # normalize for color
                for t_idx in range(len(tr)):
                    if tr[t_idx] > DFF_THRESHOLD:
                        ax3.barh(i, 1/FRAME_RATE, left=t_idx/FRAME_RATE,
                                height=0.8, color='green', alpha=tr_norm[t_idx] * 0.7)

        ax3.set_yticks(range(len(seq_dends)))
        ax3.set_yticklabels([d.split('_')[1] for d in seq_dends], fontsize=7)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Dendrite")
        ax3.set_title(f"Event Raster — Sequence Participants ({DATE}/{RUN})")
        ax3.set_xlim(0, time[-1])

    plt.suptitle(f"Propagation Sequences — {MOUSE} {DATE}/{RUN}\n"
                 f"Threshold={DFF_THRESHOLD} dFF, min repeats={MIN_SEQ_REPEATS}, "
                 f"max lag={MAX_LAG} frames ({MAX_LAG/FRAME_RATE*1000:.0f}ms)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT}")
    plt.show()


def main():
    print(f"=== Propagation Sequence Analysis ===")
    print(f"  {DATE}/{MOUSE}/{RUN}\n")

    # Load data
    print("Loading centroids...")
    centroids = load_centroids()
    print(f"  {len(centroids)} masks")

    print("Loading traces...")
    df = pd.read_csv(TRACE_CSV)
    if 'Frame' in df.columns:
        df = df.drop(columns=['Frame'])
    print(f"  {df.shape[1]} dendrites, {df.shape[0]} frames ({df.shape[0]/FRAME_RATE:.0f}s)")

    print("Detecting events...")
    onsets = detect_events(df)
    total_events = sum(len(v) for v in onsets.values())
    print(f"  {total_events} total events across {sum(1 for v in onsets.values() if len(v) > 0)} active dendrites")

    print("Finding pair sequences...")
    pair_counts, pair_instances = find_pair_sequences(onsets)

    print("Finding triplet sequences...")
    triplet_counts, triplet_instances = find_triplet_sequences(onsets)

    # Plot
    plot_sequences(centroids, pair_counts, triplet_counts, df, onsets)


if __name__ == "__main__":
    main()
