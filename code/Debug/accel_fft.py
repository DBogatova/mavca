#!/usr/bin/env python3
"""Quick FFT of accelerometer to identify periodic oscillations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path

DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUN = "run10"
RUN_NUM = "010"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
csv_path = BASE / "trigger" / f"Run{RUN_NUM}_t1_accel.csv"

df = pd.read_csv(csv_path)
accel = df['accel_mag'].values if 'accel_mag' in df.columns else df.iloc[:, 1].values
t = df['aligned_time_s'].values if 'aligned_time_s' in df.columns else df['sample'].values / 1000.0

# Use only imaging period (after trigger)
mask = (t >= 11) & (t <= 120)
accel_seg = accel[mask]  # raw, no abs/smoothing
fs = 1000  # 1 kHz

# Quiet periods (raw < 0.05)
quiet_mask = np.abs(accel_seg) < 0.05
accel_quiet = accel_seg.copy()
accel_quiet[~quiet_mask] = 0

f, psd = welch(accel_seg, fs=fs, nperseg=8192)
f_q, psd_q = welch(accel_quiet, fs=fs, nperseg=8192)

from scipy.signal import find_peaks

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 0-15 Hz
axes[0].semilogy(f[f <= 15], psd[f <= 15], label="All data")
axes[0].semilogy(f_q[f_q <= 15], psd_q[f_q <= 15], label="Quiet only", alpha=0.8)
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("PSD")
axes[0].set_title(f"Accelerometer PSD (raw) — {DATE}/{MOUSE}/{RUN} (0-15 Hz)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 0-7 Hz
axes[1].semilogy(f[f <= 7], psd[f <= 7], label="All data")
axes[1].semilogy(f_q[f_q <= 7], psd_q[f_q <= 7], label="Quiet only", alpha=0.8)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("PSD")
axes[1].set_title("Zoom: 0-7 Hz")
axes[1].legend()
axes[1].grid(alpha=0.3)

# 0-1 Hz linear
axes[2].plot(f[f <= 1], psd[f <= 1], label="All data", lw=2)
axes[2].plot(f_q[f_q <= 1], psd_q[f_q <= 1], label="Quiet only", lw=2, alpha=0.8)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("PSD (linear)")
axes[2].set_title("Zoom: 0-1 Hz (linear)")
axes[2].legend()
axes[2].grid(alpha=0.3)

# Mark peaks — red full line = present in quiet too, orange half line = all-data only
for ax, fmax in zip(axes, [15, 7, 1]):
    fm = f[f <= fmax]
    pm = psd[f <= fmax]
    pm_q = psd_q[f_q <= fmax]
    peaks, _ = find_peaks(pm, prominence=pm.max() * 0.05)
    peaks_q, _ = find_peaks(pm_q, prominence=pm_q.max() * 0.05)
    quiet_freqs = [fm[p] for p in peaks_q] if len(peaks_q) > 0 else []
    for p in peaks:
        freq = fm[p]
        in_quiet = any(abs(freq - qf) < 0.2 for qf in quiet_freqs)
        if in_quiet:
            ax.axvline(freq, color='red', ls='--', alpha=0.5, lw=1.5)
            ax.text(freq, pm[p] * 1.3, f"{freq:.2f}", fontsize=8, color='red', ha='center')
        else:
            ax.axvline(freq, color='purple', ls='--', alpha=0.5, lw=1.5,
                       ymin=0.5, ymax=1.0)
            ax.text(freq, pm[p] * 1.3, f"{freq:.2f}", fontsize=8, color='purple', ha='center')

plt.tight_layout()
out = BASE / "trigger" / "accel_fft.png"
plt.savefig(out, dpi=200)
print(f"Saved: {out}")
plt.show()
