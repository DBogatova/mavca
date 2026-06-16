#!/usr/bin/env python
"""
Combined behavior and calcium/ACh analysis plots (single run)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.io import loadmat
import tifffile

# Vector-friendly fonts: Arial, embedded as editable TrueType in the PDF
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42

# ===== CONFIG =====
DATE = "2026-02-17"
MOUSE = "rbp4cre_138_phpeb"
RUN = "run7"

FRAME_RATE = 6  # Hz
SKIP_FIRST_SECONDS = 12.0
CROP_START_SECONDS = 12.0  # cut first N seconds from all signals
HAS_ACH = True
APPLY_PUPIL_TRIGGER_OFFSET = True  # pupil/whisker Basler→SCAPE offset (off if aligned in preprocessing)

# ACh bleach correction: fit a monotonic exponential to a rolling low-percentile
# baseline (so transients/tonic bumps don't bias the fit), then remove that slow drift.
BLEACH_CORRECT_ACH = True
BLEACH_WINDOW_S = 30.0  # rolling-percentile window for the baseline envelope (s)
BLEACH_PCT = 10         # percentile used for the baseline envelope

# ACh↔Ca lead/lag analysis: search window for the cross-correlation peak (s)
MAX_LAG_S = 5.0

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/data") / DATE / MOUSE / RUN
BEHAVIOR_MAT = BASE / "behavior" / "rbp4cre_138_phpeb_26-02-17_Run007_behavior.mat"
OUTPUT_PATH = BASE / "behavior_combined_plot.pdf"


def _compute_global_dff(raw_path, label):
    """Compute global ΔF/F from a raw stack (mean of all non-dead voxels)."""
    print(f"Loading {label} stack: {raw_path.name}")
    store = tifffile.memmap(str(raw_path), mode='r')
    T = store.shape[0]
    skip = int(max(SKIP_FIRST_SECONDS, CROP_START_SECONDS) * FRAME_RATE)
    print(f"  Shape: {store.shape}, skipping first {skip} frames for F0")

    # Find dead voxels
    print("  Finding live voxels...")
    sample = np.asarray(store[:min(100, T)]).astype(np.float32)
    tmean = sample.mean(axis=0)
    live_idx = np.flatnonzero(tmean > np.percentile(tmean, 5))
    print(f"  Live voxels: {live_idx.size:,} / {tmean.size:,}")
    del sample, tmean

    # F0 baseline
    print("  Computing F0 baseline...")
    f0_frames = []
    for t in range(skip, min(skip + 500, T)):
        f0_frames.append(np.asarray(store[t]).astype(np.float32).ravel()[live_idx].mean())
    f0 = np.percentile(f0_frames, 10)
    print(f"  F0 = {f0:.1f}")
    del f0_frames

    # Global ΔF/F
    print("  Computing global ΔF/F...")
    dff = np.empty(T, dtype=np.float32)
    for t0 in range(0, T, 50):
        t1 = min(t0 + 50, T)
        frames = np.asarray(store[t0:t1]).astype(np.float32)
        for i in range(frames.shape[0]):
            dff[t0 + i] = (frames[i].ravel()[live_idx].mean() - f0) / (f0 + 1e-6) * 100
    del store
    return T, dff


def _bleach_correct_ach(dff, fps):
    """Remove slow photobleach drift from an ACh ΔF/F trace.

    Fits a monotonic exponential to a rolling low-percentile baseline envelope so
    that fast transients and genuine slow tonic elevations do not bias the fit,
    then subtracts the fitted drift. The monotonic exponential form means a
    transient tonic bump in the envelope cannot be fully removed (only a true
    slow monotonic decay is). Falls back to a linear detrend, then to no-op.
    """
    from scipy.optimize import curve_fit

    y = np.asarray(dff, dtype=np.float64)
    T = y.size
    if T < 10:
        return dff

    # Rolling low-percentile baseline envelope (excludes transients).
    win = max(3, int(round(BLEACH_WINDOW_S * fps)))
    win = min(win, T if T % 2 else T - 1)
    if win % 2 == 0:
        win += 1
    env = percentile_filter(y, percentile=BLEACH_PCT, size=win, mode="nearest")

    x = np.arange(T, dtype=np.float64)

    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

    # Fit monotonic exponential to the envelope.
    span = float(env.max() - env.min()) or 1.0
    p0 = (span, 1.0 / max(1.0, T / 4.0), float(np.median(env)))
    try:
        (a, b, c), _ = curve_fit(
            exp_decay, x, env, p0=p0, maxfev=20000,
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        )
        drift = exp_decay(x, a, b, c)
        method = f"exp (tau={1.0/b/fps:.1f}s)" if b > 0 else "exp"
    except Exception:
        # Fallback: linear detrend of the envelope.
        try:
            m, q = np.polyfit(x, env, 1)
            drift = m * x + q
            method = "linear"
        except Exception:
            print("  ACh bleach correction: fit failed, leaving trace uncorrected")
            return dff

    # Remove the slow drift, then re-zero so the resting level (low percentile)
    # sits at 0 — keeps ACh at ~0 at rest with positive transients, instead of
    # letting the trace float below zero (F0 was from an early window only).
    corrected = y - drift
    corrected = corrected - np.percentile(corrected, BLEACH_PCT)
    print(f"  ACh bleach correction applied ({method}, window={win/fps:.0f}s, "
          f"pct={BLEACH_PCT}); baseline re-zeroed to {BLEACH_PCT}th pct")
    return corrected.astype(np.float32)


def load_calcium_ach_data():
    """Compute global Ca (and optionally ACh) ΔF/F from raw stacks."""
    raw_clean = BASE / "preprocessed" / "raw_clean.tif"
    raw_orig = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-green.tif"
    ca_path = raw_clean if raw_clean.exists() else raw_orig

    if not ca_path.exists():
        print(f"Raw stack not found: {ca_path}")
        return None, None, None

    T, ca_dff = _compute_global_dff(ca_path, "Ca (green)")

    # ACh from the raw red channel
    ach_dff = None
    if HAS_ACH:
        ach_path = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-red.tif"
        if ach_path.exists():
            _, ach_dff = _compute_global_dff(ach_path, "ACh (red)")
        else:
            print(f"ACh red stack not found: {ach_path}")

    # ACh bleach correction on the full trace (before cropping)
    if ach_dff is not None and BLEACH_CORRECT_ACH:
        ach_dff = _bleach_correct_ach(ach_dff, FRAME_RATE)

    # Align Ca/ACh lengths if they differ
    if ach_dff is not None and len(ach_dff) != T:
        n = min(len(ach_dff), T)
        print(f"  Note: Ca/ACh frame counts differ ({T} vs {len(ach_dff)}), truncating to {n}")
        T, ca_dff, ach_dff = n, ca_dff[:n], ach_dff[:n]

    time = np.arange(T) / FRAME_RATE
    # Crop
    mask = time >= CROP_START_SECONDS
    time = time[mask] - CROP_START_SECONDS
    ca_dff = ca_dff[mask]
    if ach_dff is not None:
        ach_dff = ach_dff[mask]
        print(f"  Global ACh ΔF/F: min={ach_dff.min():.2f}%, max={ach_dff.max():.2f}%")
    print(f"  Global Ca ΔF/F: min={ca_dff.min():.2f}%, max={ca_dff.max():.2f}%")
    return time, ach_dff, ca_dff


def load_behavior_data():
    """Load pupil and whisker from behavior MAT file.
    Returns values normalized to [0, 1] using the max from the first 14s as reference."""
    if not BEHAVIOR_MAT.exists():
        print(f"Behavior MAT not found: {BEHAVIOR_MAT}")
        return None, None, None

    mat_data = loadmat(BEHAVIOR_MAT)
    pupil = mat_data['pupil']['pupil_raw'][0][0].flatten()
    pupil = gaussian_filter1d(pupil, sigma=2)
    whisker = mat_data['whisker']['whisker_smooth_long'][0][0].flatten()
    whisker = gaussian_filter1d(whisker, sigma=3)

    n = len(pupil)
    print(f"  Behavior: {n} samples at 10 Hz = {n/10:.1f}s")

    time = np.arange(n) / 10.0

    # Compute Basler-to-SCAPE offset from trigger CSV
    trigger_csvs = list((BASE / "trigger").glob("*_trigger.csv"))
    if trigger_csvs:
        trig = pd.read_csv(trigger_csvs[0])
        basler_start = trig.loc[trig['baslerExposureTrigger'].diff() == 1, 'time_s'].iloc[0]
        andor_start = trig.loc[trig['AndorXylaTrigger'].diff() == 1, 'time_s'].iloc[0]
        offset = andor_start - basler_start
        print(f"  Basler→SCAPE offset = {offset:.3f}s")
    else:
        offset = 0.0
        print("  No trigger CSV, assuming no Basler offset")

    if not APPLY_PUPIL_TRIGGER_OFFSET:
        offset = 0.0
        print("  Pupil/whisker trigger offset DISABLED (aligned in preprocessing)")

    total_crop = offset + CROP_START_SECONDS

    # Normalize to max from first 14s (full dynamic range reference)
    pre_mask = (time >= offset) & (time < total_crop)
    pupil_max = np.max(pupil[pre_mask]) if pre_mask.any() else np.max(pupil)
    whisker_max = np.max(np.abs(whisker[pre_mask])) if pre_mask.any() else np.max(np.abs(whisker))
    pupil_norm = pupil / (pupil_max + 1e-6)
    whisker_norm = whisker / (whisker_max + 1e-6)
    print(f"  Pupil max (pre-crop): {pupil_max:.1f}, post-crop range: {pupil_norm[time >= total_crop].min():.2f}-{pupil_norm[time >= total_crop].max():.2f}")
    print(f"  Whisker max (pre-crop): {whisker_max:.1f}, post-crop range: {whisker_norm[time >= total_crop].min():.2f}-{whisker_norm[time >= total_crop].max():.2f}")

    mask = time >= total_crop
    return time[mask] - total_crop, pupil_norm[mask], whisker_norm[mask]


def load_accelerometer_data():
    """Load accelerometer, normalized to max from first 14s."""
    run_num = RUN.replace("run", "").zfill(3)
    accel_csv = BASE / "trigger" / f"Run{run_num}_t1_accel.csv"

    if not accel_csv.exists():
        print(f"Accel CSV not found: {accel_csv}")
        return None, None, 0

    df = pd.read_csv(accel_csv)
    print(f"Accel columns: {list(df.columns)}")

    accel = df['accel_mag'].values if 'accel_mag' in df.columns else df.iloc[:, 1].values
    time = df['aligned_time_s'].values if 'aligned_time_s' in df.columns else df['sample'].values / 1000.0

    accel_clean = np.abs(accel)
    accel_clean = gaussian_filter1d(accel_clean, sigma=10)

    mask = time >= CROP_START_SECONDS
    return time[mask] - CROP_START_SECONDS, accel_clean[mask], 0


def analyze_ach_ca_lag(time, ach, ca, fps):
    """Estimate the lead/lag between ACh and Ca via cross-correlation.

    Returns (best_ach_lead_s, peak_r, ach_lead_axis, r_axis) where a POSITIVE
    'ACh lead' means ACh rises before Ca. Computes the lag two ways:
      - level:  on the z-scored traces (overall co-fluctuation)
      - rise :  on the temporal derivatives (timing of rising edges)
    """
    from scipy.signal import correlate, correlation_lags

    a = np.asarray(ach, dtype=np.float64)
    c = np.asarray(ca, dtype=np.float64)
    n = min(a.size, c.size)
    a, c = a[:n], c[:n]
    if n < 8:
        print("  ACh↔Ca lag: trace too short, skipping")
        return None, None, None, None

    def _z(v):
        v = v - v.mean()
        s = v.std()
        return v / s if s > 0 else v

    def _peak(sig_a, sig_c):
        xa, xc = _z(sig_a), _z(sig_c)
        r = correlate(xa, xc, mode="full") / n
        lags = correlation_lags(xa.size, xc.size, mode="full")
        ach_lead_s = -lags / fps  # positive => ACh leads Ca
        order = np.argsort(ach_lead_s)
        ach_lead_s, r = ach_lead_s[order], r[order]
        win = np.abs(ach_lead_s) <= MAX_LAG_S
        al, rr = ach_lead_s[win], r[win]
        k = int(np.argmax(rr))
        return al[k], float(rr[k]), al, rr

    lag_level, r_level, axis, r_axis = _peak(a, c)
    lag_rise, r_rise, _, _ = _peak(np.gradient(a), np.gradient(c))
    r0 = float(np.corrcoef(_z(a), _z(c))[0, 1])

    def _phrase(lag):
        if abs(lag) < (1.0 / fps):
            return "≈ simultaneous (within one frame)"
        who = "ACh leads Ca" if lag > 0 else "Ca leads ACh"
        return f"{who} by {abs(lag):.2f} s"

    print("\n  ── ACh ↔ Ca timing ──")
    print(f"  zero-lag Pearson r         : {r0:+.3f}")
    print(f"  level cross-corr peak      : {_phrase(lag_level)}  (r={r_level:+.3f})")
    print(f"  rise/rate cross-corr peak  : {_phrase(lag_rise)}  (r={r_rise:+.3f})")
    if lag_level <= 0 and lag_rise <= 0:
        print("  → no ACh lead detected at this resolution")
    elif lag_rise > 0:
        print(f"  → ACh rise precedes Ca rise by ~{lag_rise:.2f} s")
    return lag_level, r_level, axis, r_axis


def plot_xcorr(ach_lead_axis, r_axis, best_lag, out_path):
    """Plot the ACh→Ca cross-correlation vs lead time and mark the peak."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ach_lead_axis, r_axis, color='black', lw=1.2)
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.axvline(best_lag, color='red', ls='-', lw=1.0,
               label=f'peak: ACh lead = {best_lag:+.2f} s')
    ax.set_xlabel('ACh lead time relative to Ca (s)   [positive → ACh first]')
    ax.set_ylabel('Cross-correlation (r)')
    ax.set_title(f'{MOUSE} — {DATE} — {RUN}: ACh→Ca cross-correlation')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved cross-correlation plot: {out_path}")


def plot_combined_signals():
    """Create combined plot of all signals."""
    time_ca, ach_dff, ca_dff = load_calcium_ach_data()
    time_behavior, pupil, whisker = load_behavior_data()
    time_accel, accel, _ = load_accelerometer_data()

    # ACh↔Ca lead/lag (does ACh rise before Ca?)
    if time_ca is not None and ach_dff is not None and ca_dff is not None:
        best_lag, _, axis, r_axis = analyze_ach_ca_lag(time_ca, ach_dff, ca_dff, FRAME_RATE)
        if axis is not None:
            plot_xcorr(axis, r_axis, best_lag, BASE / "ach_ca_xcorr.png")

    panels = []
    if time_ca is not None and HAS_ACH and ach_dff is not None:
        panels.append(("ACh ΔF/F (%)", time_ca, ach_dff, 'red'))
    if time_ca is not None:
        panels.append(("Ca ΔF/F (%)", time_ca, ca_dff, 'green'))
    if time_behavior is not None and pupil is not None:
        panels.append(("Pupil Dilation", time_behavior, pupil, 'blue'))
    if time_behavior is not None and whisker is not None:
        panels.append(("Whisker Motion", time_behavior, whisker, 'orange'))
    if time_accel is not None and accel is not None:
        panels.append(("Accelerometer", time_accel, accel, 'purple'))

    # Crop all to Ca duration
    if time_ca is not None:
        t_end = time_ca[-1]
        cropped = []
        for ylabel, t, data, color in panels:
            m = (t >= 0) & (t <= t_end)
            cropped.append((ylabel, t[m], data[m], color))
        panels = cropped

    if not panels:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 1.6 * len(panels)),
                             sharex=True)
    if len(panels) == 1:
        axes = [axes]

    # Signal name written on each panel, above the trace, in the trace color
    NAME_MAP = {
        "ACh ΔF/F (%)": "ACh",
        "Ca ΔF/F (%)": "Global Ca",
        "Pupil Dilation": "Pupil",
        "Whisker Motion": "Whisking",
        "Accelerometer": "Accelerometer",
    }
    for ax, (ylabel, t, data, color) in zip(axes, panels):
        ax.plot(t, data, color=color, linewidth=1.8)
        if ylabel == "Accelerometer":
            ax.set_ylim(0, 0.25)
        ax.set_ylabel("")
        ax.text(0.008, 0.94, NAME_MAP.get(ylabel, ylabel), transform=ax.transAxes,
                ha='left', va='top', color=color, fontweight='bold', fontsize=20)
        ax.grid(False)
        ax.tick_params(labelsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    axes[-1].set_xlabel('Time (s)', fontsize=18)
    plt.tight_layout(h_pad=0.4)
    plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {OUTPUT_PATH}")


def main():
    print(f"Creating combined behavior plot for {MOUSE}-{DATE}-{RUN}")
    plot_combined_signals()

if __name__ == "__main__":
    main()
