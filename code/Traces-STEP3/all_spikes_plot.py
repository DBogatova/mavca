#!/usr/bin/env python3
"""
All dendrite spikes + behavior + cross-correlation + coherence.

Uses same behavior loading as behavior_plots_concat.py.
Panels: spikes → accel → global Ca → pupil → whisker

Usage:
    python code/Traces-STEP3/all_spikes_plot.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import tifffile

# ===== CONFIG =====
DATE = "2026-04-16"
MOUSE = "rbp4_132_phpeb"
RUN = "run7"

# Concatenation: set runs + run nums, or leave empty for single run
CONCAT_RUNS = []
RUN_NUMS = []
MASK_RUN = "run1"

FRAME_RATE = 5.0
CROP_START_SECONDS = 12.0
SKIP_FIRST_SECONDS = 12.0
DFF_THRESHOLD = 0.3
SHOW_CLUSTER_BANDS = False

# ===== PATHS =====
BASE_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
                 "apical-dendrites-2025/scape-data") / DATE / MOUSE
BASE = BASE_ROOT / RUN


# ===== TRACE LOADING =====
def load_traces():
    if CONCAT_RUNS:
        csv_path = BASE_ROOT / MASK_RUN / "traces_concat" / "dff_traces_concat.csv"
        if not csv_path.exists():
            print(f"  Concat CSV not found: {csv_path}")
            return None, None, None
        df = pd.read_csv(csv_path)
    else:
        csv_path = BASE / "traces" / "dff_traces_curated_bgsub.csv"
        df = pd.read_csv(csv_path)
        # Crop for single run
        crop = int(CROP_START_SECONDS * FRAME_RATE)
        df = df.iloc[crop:]

    dend_cols = [c for c in df.columns if c.startswith("dend_")]
    traces = df[dend_cols].values
    time_s = np.arange(traces.shape[0]) / FRAME_RATE
    print(f"  {len(dend_cols)} traces, {traces.shape[0]} frames ({time_s[-1]:.1f}s)")
    return time_s, traces, dend_cols


# ===== BEHAVIOR LOADING (same as behavior_plots_concat.py) =====
def load_behavior_one_run(run, run_num):
    base = BASE_ROOT / run
    beh_dir = base / "behavior"
    if not beh_dir.exists():
        return None, None, None
    mats = list(beh_dir.glob("*behavior.mat"))
    if not mats:
        return None, None, None
    mat = loadmat(str(mats[0]))
    pupil = mat['pupil']['pupil_raw'][0][0].flatten()
    pupil = gaussian_filter1d(pupil, sigma=2)
    whisker = mat['whisker']['whisker_smooth_long'][0][0].flatten()
    whisker = gaussian_filter1d(whisker, sigma=5)
    t = np.arange(len(pupil)) / 10.0
    mask = t >= CROP_START_SECONDS
    return t[mask] - CROP_START_SECONDS, pupil[mask], whisker[mask]


def load_accel_one_run(run, run_num):
    base = BASE_ROOT / run
    csv_path = base / "trigger" / f"Run{run_num}_t1_accel.csv"
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    accel = df['accel_mag'].values if 'accel_mag' in df.columns else df.iloc[:, 1].values
    t = df['aligned_time_s'].values if 'aligned_time_s' in df.columns else df['sample'].values / 1000.0
    mask = t >= CROP_START_SECONDS
    a = np.abs(accel[mask])
    a = gaussian_filter1d(a, sigma=10)
    return t[mask] - CROP_START_SECONDS, a


def load_ca_one_run(run):
    base = BASE_ROOT / run
    raw_clean = base / "preprocessed" / "raw_clean.tif"
    raw_orig = base / "raw" / f"runA_{run}_{MOUSE}-reslice-bin.tif"
    raw_path = raw_clean if raw_clean.exists() else raw_orig
    if not raw_path.exists():
        return None, None
    store = tifffile.memmap(str(raw_path), mode='r')
    T = store.shape[0]
    skip = int(max(SKIP_FIRST_SECONDS, CROP_START_SECONDS) * FRAME_RATE)
    sample = np.asarray(store[:min(100, T)]).astype(np.float32)
    tmean = sample.mean(axis=0)
    live_idx = np.flatnonzero(tmean > np.percentile(tmean, 5))
    del sample, tmean
    f0_vals = []
    for t in range(skip, min(skip + 500, T)):
        f0_vals.append(np.asarray(store[t]).astype(np.float32).ravel()[live_idx].mean())
    f0 = np.percentile(f0_vals, 10)
    del f0_vals
    ca = np.empty(T, dtype=np.float32)
    for t0 in range(0, T, 50):
        t1 = min(t0 + 50, T)
        frames = np.asarray(store[t0:t1]).astype(np.float32)
        for i in range(frames.shape[0]):
            ca[t0 + i] = (frames[i].ravel()[live_idx].mean() - f0) / (f0 + 1e-6) * 100
    del store
    time = np.arange(T) / FRAME_RATE
    mask = time >= CROP_START_SECONDS
    return time[mask] - CROP_START_SECONDS, ca[mask]


def load_all_behavior():
    """Load and concatenate all behavior, same logic as behavior_plots_concat."""
    runs = CONCAT_RUNS if CONCAT_RUNS else [RUN]
    rnums = RUN_NUMS if CONCAT_RUNS else [RUN.replace("run", "").zfill(3)]

    all_pupil_t, all_pupil = [], []
    all_whisker_t, all_whisker = [], []
    all_accel_t, all_accel = [], []
    all_ca_t, all_ca = [], []
    run_boundaries = [0.0]

    for run, rnum in zip(runs, rnums):
        print(f"  --- {run} ---")

        # Get run duration from raw stack
        base = BASE_ROOT / run
        raw_clean = base / "preprocessed" / "raw_clean.tif"
        raw_orig = base / "raw" / f"runA_{run}_{MOUSE}-reslice-bin.tif"
        raw_path = raw_clean if raw_clean.exists() else raw_orig
        if raw_path.exists():
            tf = tifffile.TiffFile(str(raw_path))
            T_raw = tf.series[0].shape[0]
            tf.close()
            run_dur = (T_raw - int(CROP_START_SECONDS * FRAME_RATE)) / FRAME_RATE
        else:
            run_dur = 120.0

        offset = run_boundaries[-1]

        # Ca
        t_ca, ca = load_ca_one_run(run)
        if t_ca is not None:
            all_ca_t.append(t_ca + offset)
            all_ca.append(ca)

        # Behavior
        t_beh, pupil, whisker = load_behavior_one_run(run, rnum)
        if t_beh is not None:
            mask = t_beh <= run_dur
            # Subtract per-run median to align pupil baselines
            p_seg = pupil[mask]
            if len(runs) > 1:
                p_seg = p_seg - np.median(p_seg)
            all_pupil_t.append(t_beh[mask] + offset)
            all_pupil.append(p_seg)
            all_whisker_t.append(t_beh[mask] + offset)
            all_whisker.append(whisker[mask])

        # Accel
        t_acc, accel = load_accel_one_run(run, rnum)
        if t_acc is not None:
            mask = (t_acc >= 0) & (t_acc <= run_dur)
            all_accel_t.append(t_acc[mask] + offset)
            all_accel.append(accel[mask])

        run_boundaries.append(offset + run_dur)

    result = {"run_boundaries": run_boundaries}
    if all_ca: result["ca"] = (np.concatenate(all_ca_t), np.concatenate(all_ca))
    if all_pupil: result["pupil"] = (np.concatenate(all_pupil_t), np.concatenate(all_pupil))
    if all_whisker: result["whisker"] = (np.concatenate(all_whisker_t), np.concatenate(all_whisker))
    if all_accel: result["accel"] = (np.concatenate(all_accel_t), np.concatenate(all_accel))
    return result


# ===== MAIN =====
def main():
    print(f"=== All Spikes: {DATE}/{MOUSE} ===\n")

    time_s, traces, names = load_traces()
    if traces is None:
        return
    T, N = traces.shape

    print("Loading behavior...")
    beh = load_all_behavior()
    rb = beh["run_boundaries"]
    if "pupil" in beh:
        t, d = beh["pupil"]
        print(f"  DEBUG pupil: len={len(d)}, min={d.min():.4f}, max={d.max():.4f}, t_max={t[-1]:.1f}s")

    # Clip traces to match behavior duration
    max_dur = rb[-1] if rb else time_s[-1]
    clip = time_s <= max_dur
    time_s = time_s[clip]
    traces = traces[clip]
    T, N = traces.shape

    # Also clip behavior to trace duration
    trace_dur = time_s[-1]
    for key in ["pupil", "whisker", "accel", "ca"]:
        if key in beh:
            t, d = beh[key]
            mask = t <= trace_dur
            beh[key] = (t[mask], d[mask])

    # Build panels: spikes, accel, global Ca, pupil, whisker
    n_panels = 1
    if "accel" in beh: n_panels += 1
    if "ca" in beh: n_panels += 1
    if "pupil" in beh: n_panels += 1
    if "whisker" in beh: n_panels += 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.5 * n_panels), sharex=True,
                              gridspec_kw={"height_ratios": [2.5] + [1.5] * (n_panels - 1)})
    if n_panels == 1: axes = [axes]
    ax_idx = 0

    # === Spikes ===
    ax = axes[ax_idx]
    for i in range(N):
        tr = traces[:, i].copy()
        tr[tr < DFF_THRESHOLD] = np.nan
        ax.plot(time_s, tr, lw=0.8, alpha=0.7, color=cm.turbo(i / max(1, N)))
    ax.set_ylabel("ΔF/F (%)")
    title_runs = " + ".join(CONCAT_RUNS) if CONCAT_RUNS else RUN
    ax.set_title(f"{DATE} | {MOUSE} | {title_runs} — Dendrite spikes (>{DFF_THRESHOLD}%)")
    ax.grid(alpha=0.3)
    ax_idx += 1

    # === Accel ===
    if "accel" in beh:
        t, d = beh["accel"]
        axes[ax_idx].plot(t, d, color='purple', lw=0.8)
        axes[ax_idx].set_ylabel("Accel")
        axes[ax_idx].grid(alpha=0.3)
        ax_idx += 1

    # === Global Ca ===
    if "ca" in beh:
        t, d = beh["ca"]
        axes[ax_idx].plot(t, d, color='green', lw=1.2)
        axes[ax_idx].set_ylabel("Global Ca\nΔF/F (%)")
        axes[ax_idx].grid(alpha=0.3)
        ax_idx += 1

    # === Pupil ===
    if "pupil" in beh:
        t, d = beh["pupil"]
        axes[ax_idx].plot(t, d, color='blue', lw=1.0)
        axes[ax_idx].set_ylabel("Pupil Dilation")
        axes[ax_idx].set_ylim(d.min() - 0.02, d.max() + 0.02)
        axes[ax_idx].grid(alpha=0.3)
        ax_idx += 1

    # === Whisker ===
    if "whisker" in beh:
        t, d = beh["whisker"]
        axes[ax_idx].plot(t, d, color='orange', lw=1.0)
        axes[ax_idx].set_ylabel("Whisker Motion")
        axes[ax_idx].set_ylim(d.min() - 0.02, d.max() + 0.02)
        axes[ax_idx].grid(alpha=0.3)

    # Run boundaries
    for b in rb[1:-1]:
        for ax in axes:
            ax.axvline(b, color='red', ls='-', lw=1.5, alpha=0.2)
    runs_list = CONCAT_RUNS if CONCAT_RUNS else [RUN]
    for j, run in enumerate(runs_list):
        if j < len(rb) - 1:
            mid = (rb[j] + rb[j+1]) / 2
            axes[0].text(mid, -0.08, run, ha='center', fontsize=9, color='gray',
                         transform=axes[0].get_xaxis_transform())

    # Cluster bands
    if SHOW_CLUSTER_BANDS:
        event_rate_det = gaussian_filter1d(
            (traces > DFF_THRESHOLD).astype(np.float32).mean(axis=1), sigma=FRAME_RATE * 0.5)
        cluster_thr = event_rate_det.mean() + 1.0 * event_rate_det.std()
        in_cluster = event_rate_det > cluster_thr
        changes = np.diff(in_cluster.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if in_cluster[0]: starts = np.concatenate([[0], starts])
        if in_cluster[-1]: ends = np.concatenate([ends, [len(in_cluster)]])
        for s, e in zip(starts, ends):
            t0 = time_s[s]; t1 = time_s[min(e, len(time_s) - 1)]
            for ax in axes:
                ax.axvspan(t0, t1, alpha=0.15, color='cornflowerblue', zorder=0)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    out_dir = BASE_ROOT / (MASK_RUN if CONCAT_RUNS else RUN) / "traces"
    out_dir.mkdir(exist_ok=True)
    suffix = "_concat" if CONCAT_RUNS else ""
    out_path = out_dir / f"all_spikes_vs_behavior{suffix}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved: {out_path}")
    plt.show()

    # ============================================================
    # === Heatmap plot: dendrites × time + behavior ===
    # ============================================================
    n_hm_panels = 1  # heatmap
    if "accel" in beh: n_hm_panels += 1
    if "ca" in beh: n_hm_panels += 1
    if "pupil" in beh: n_hm_panels += 1
    if "whisker" in beh: n_hm_panels += 1

    fig_hm, axes_hm = plt.subplots(n_hm_panels, 1, figsize=(16, 2.5 * n_hm_panels),
                                     sharex=True,
                                     gridspec_kw={"height_ratios": [3] + [1] * (n_hm_panels - 1)})
    if n_hm_panels == 1: axes_hm = [axes_hm]
    hm_idx = 0

    # Heatmap: clip below threshold, show as image
    hm_data = traces.copy()
    hm_data[hm_data < DFF_THRESHOLD] = 0
    vmax = np.percentile(hm_data[hm_data > 0], 95) if (hm_data > 0).any() else 1

    axes_hm[hm_idx].imshow(hm_data.T, aspect='auto', cmap='coolwarm',
                            extent=[time_s[0], time_s[-1], N - 0.5, -0.5],
                            interpolation='none', vmin=0, vmax=vmax)
    axes_hm[hm_idx].set_ylabel("Dendrite")
    axes_hm[hm_idx].set_title(f"{DATE} | {MOUSE} | {title_runs} — Activity Heatmap (>{DFF_THRESHOLD}%)")
    tick_step = max(1, N // 10)
    axes_hm[hm_idx].set_yticks(range(0, N, tick_step))
    axes_hm[hm_idx].set_yticklabels([n.replace("dend_", "") for n in names[::tick_step]], fontsize=7)
    hm_idx += 1

    # Accel
    if "accel" in beh:
        t_a, d_a = beh["accel"]
        axes_hm[hm_idx].plot(t_a, d_a, color='purple', lw=0.8)
        axes_hm[hm_idx].set_ylabel("Accel")
        axes_hm[hm_idx].grid(alpha=0.3)
        hm_idx += 1

    # Global Ca
    if "ca" in beh:
        t_c, d_c = beh["ca"]
        axes_hm[hm_idx].plot(t_c, d_c, color='green', lw=1.2)
        axes_hm[hm_idx].set_ylabel("Global Ca\nΔF/F (%)")
        axes_hm[hm_idx].grid(alpha=0.3)
        hm_idx += 1

    # Pupil
    if "pupil" in beh:
        t_p, d_p = beh["pupil"]
        axes_hm[hm_idx].plot(t_p, d_p, color='blue', lw=1.0)
        axes_hm[hm_idx].set_ylabel("Pupil")
        axes_hm[hm_idx].grid(alpha=0.3)
        hm_idx += 1

    # Whisker
    if "whisker" in beh:
        t_w, d_w = beh["whisker"]
        axes_hm[hm_idx].plot(t_w, d_w, color='orange', lw=1.0)
        axes_hm[hm_idx].set_ylabel("Whisker")
        axes_hm[hm_idx].grid(alpha=0.3)

    # Run boundaries on all heatmap panels
    for b in rb[1:-1]:
        for ax in axes_hm:
            ax.axvline(b, color='red', ls='-', lw=1.5, alpha=0.3)

    axes_hm[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    hm_path = out_dir / f"heatmap_vs_behavior{suffix}.png"
    plt.savefig(hm_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {hm_path}")
    plt.show()

    # ============================================================
    # === Cross-correlation + Coherence + STA ===
    # ============================================================
    from scipy.signal import coherence as sig_coherence

    event_rate = gaussian_filter1d(
        (traces > DFF_THRESHOLD).astype(np.float32).mean(axis=1),
        sigma=FRAME_RATE * 1.0)

    max_lag = int(5.0 * FRAME_RATE)
    lags = np.arange(-max_lag, max_lag + 1) / FRAME_RATE

    def xcorr_norm(a, b, ml):
        az = (a - a.mean()) / (a.std() + 1e-12)
        bz = (b - b.mean()) / (b.std() + 1e-12)
        cc = np.zeros(2 * ml + 1)
        for i, lag in enumerate(range(-ml, ml + 1)):
            if lag < 0:   cc[i] = np.mean(az[-lag:] * bz[:lag])
            elif lag > 0: cc[i] = np.mean(az[:-lag] * bz[lag:])
            else:         cc[i] = np.mean(az * bz)
        return cc

    # Resample behavior to Ca frame rate
    beh_rs = {}
    for key in ["pupil", "whisker", "accel"]:
        if key in beh:
            t_b, d_b = beh[key]
            beh_rs[key.capitalize()] = np.interp(time_s, t_b, d_b)
    # Add global Ca
    if "ca" in beh:
        t_ca, d_ca = beh["ca"]
        global_ca_rs = np.interp(time_s, t_ca, d_ca)
    else:
        global_ca_rs = None

    if beh_rs:
        n_beh = len(beh_rs)

        # Cross-correlation
        fig_cc, axes_cc = plt.subplots(1, n_beh, figsize=(6 * n_beh, 5))
        if n_beh == 1: axes_cc = [axes_cc]
        for ax, (name, brs) in zip(axes_cc, beh_rs.items()):
            cc = xcorr_norm(event_rate, brs, max_lag)
            ax.plot(lags, cc, lw=2, color='coral', label="Event rate (local)")
            if global_ca_rs is not None:
                ccg = xcorr_norm(global_ca_rs, brs, max_lag)
                ax.plot(lags, ccg, lw=2, color='green', ls='--', label="Global Ca")
            ax.axvline(0, color='gray', ls='--', lw=0.8)
            ax.axhline(0, color='gray', ls='--', lw=0.8)
            ax.set_xlabel("Lag (s)  [Ca leads →]")
            ax.set_ylabel("Correlation")
            ax.set_title(f"Xcorr vs {name}")
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"xcorr_eventrate_vs_behavior{suffix}.png", dpi=200, bbox_inches='tight')
        print(f"✅ xcorr saved"); plt.show()

        # Coherence
        nperseg = int(10.0 * FRAME_RATE)
        fig_coh, axes_coh = plt.subplots(1, n_beh, figsize=(6 * n_beh, 5))
        if n_beh == 1: axes_coh = [axes_coh]
        for ax, (name, brs) in zip(axes_coh, beh_rs.items()):
            f, c = sig_coherence(event_rate, brs, fs=FRAME_RATE, nperseg=nperseg, noverlap=nperseg//2)
            ax.plot(f, c, lw=2, color='coral', label="Event rate")
            if global_ca_rs is not None:
                fg, cg = sig_coherence(global_ca_rs, brs, fs=FRAME_RATE, nperseg=nperseg, noverlap=nperseg//2)
                ax.plot(fg, cg, lw=2, color='green', ls='--', label="Global Ca")
            ax.axvspan(0.2, 0.5, alpha=0.1, color='red', label="0.2–0.5 Hz")
            ax.set_xlim(0.05, 1.0)
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Coherence")
            ax.set_title(f"Coherence vs {name}")
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"coherence_vs_behavior{suffix}.png", dpi=200, bbox_inches='tight')
        print(f"✅ coherence saved"); plt.show()

        # STA
        ec_thr = event_rate.mean() + 2 * event_rate.std()
        spike_frames = np.where(event_rate > ec_thr)[0]
        window = int(3.0 * FRAME_RATE)
        spike_frames = spike_frames[(spike_frames >= window) & (spike_frames < T - window)]
        if len(spike_frames) > 1:
            keep = np.concatenate([[True], np.diff(spike_frames) > 2])
            spike_frames = spike_frames[keep]

        if len(spike_frames) >= 3:
            sta_t = np.arange(-window, window + 1) / FRAME_RATE
            beh_colors = {"Pupil": "blue", "Whisker": "orange", "Accel": "purple"}
            fig_sta, axes_sta = plt.subplots(1, n_beh, figsize=(6 * n_beh, 4))
            if n_beh == 1: axes_sta = [axes_sta]
            for ax, (name, brs) in zip(axes_sta, beh_rs.items()):
                snips = np.array([brs[sf-window:sf+window+1] for sf in spike_frames])
                m = snips.mean(0); se = snips.std(0) / np.sqrt(len(snips))
                c = beh_colors.get(name, 'teal')
                ax.fill_between(sta_t, m-se, m+se, alpha=0.3, color=c)
                ax.plot(sta_t, m, lw=2, color=c)
                ax.axvline(0, color='red', ls='--', lw=1.5, label="Ca event")
                ax.set_xlabel("Time rel. to event (s)"); ax.set_ylabel(name)
                ax.set_title(f"STA {name} (n={len(spike_frames)})"); ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"spike_triggered_avg{suffix}.png", dpi=200, bbox_inches='tight')
            print(f"✅ STA saved"); plt.show()


if __name__ == "__main__":
    main()
