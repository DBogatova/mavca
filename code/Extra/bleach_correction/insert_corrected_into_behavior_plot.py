#!/usr/bin/env python
"""Rebuild run4 behavior_combined_plot with the bleach-CORRECTED ACh/Ca traces.

ACh panel <- trace_bleach_corrected.csv  (first trace corrected)
Ca  panel <- calcium_bleach_corrected.csv (second, "Calcium")
Behavior panels (pupil, whisker, accelerometer) reloaded from run4 and aligned to
imaging via the Andor first-trigger offset. Imaging assumed 5 Hz (0.2 s/frame).
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

BC = Path("/Users/daria/Desktop/femtonics-data/bleach_correction")
RUN4 = Path("/Volumes/IMAC/data/2025-12-02/rbp4cre_136_phpeb/run4")
FPS = 5.0
OUT = RUN4 / "behavior_combined_plot_corrected.png"

# ---- corrected imaging traces ----
ach = pd.read_csv(BC / "trace_bleach_corrected.csv")["Mean_corrected"].to_numpy(float)
ca  = pd.read_csv(BC / "calcium_bleach_corrected.csv")["Mean_corrected"].to_numpy(float)
N = min(ach.size, ca.size)
ach, ca = ach[:N], ca[:N]
t_img = np.arange(N) / FPS
T_end = t_img[-1]

# ---- imaging start offset (Andor first trigger, in behavior/DAQ clock) ----
info = (RUN4 / "trigger" / "Run004_t1_trigger_info.txt").read_text()
m = re.search(r"Andor trigger time \(s\):\s*([\d.]+)", info)
ANDOR = float(m.group(1)) if m else 0.0

# ---- behavior (pupil, whisker) 10 Hz ----
mat = loadmat(RUN4 / "behavior" / "rbp4cre_136_phpeb_25-12-02_Run004_behavior.mat")
pupil = gaussian_filter1d(mat["pupil"]["pupil_raw"][0][0].flatten().astype(float), 2)
whisk = gaussian_filter1d(mat["whisker"]["whisker_smooth_long"][0][0].flatten().astype(float), 3)
t_beh = np.arange(pupil.size) / 10.0 - ANDOR          # -> imaging clock

# ---- accelerometer ----
adf = pd.read_csv(RUN4 / "trigger" / "Run004_t1_accel.csv")
accmag = adf["accMag"].to_numpy(float)
t_acc = adf["time_s"].to_numpy(float) - ANDOR

def crop(t, *arrs):
    msk = (t >= 0) & (t <= T_end)
    return (t[msk],) + tuple(a[msk] for a in arrs)

t_beh_c, pupil_c, whisk_c = crop(t_beh, pupil, whisk)
t_acc_c, acc_c = crop(t_acc, accmag)

def norm14(t, y):                                     # normalize to max over first 14 s
    pre = (t >= 0) & (t < 14)
    mx = np.max(np.abs(y[pre])) if pre.any() else np.max(np.abs(y))
    return y / (mx + 1e-9)

pupil_n = norm14(t_beh_c, pupil_c)
whisk_n = norm14(t_beh_c, whisk_c)

# ---- alignment sanity check: does the Andor shift improve ACh<->pupil corr? ----
def corr_at(shift):
    tb = np.arange(pupil.size) / 10.0 - shift
    p = np.interp(t_img, tb, pupil)                   # pupil on imaging grid
    a = ach - ach.mean(); p = p - p.mean()
    return float(np.corrcoef(a, p)[0, 1])
print(f"Andor offset = {ANDOR:.3f}s | ACh<->pupil corr: shift0={corr_at(0):+.3f}  "
      f"shift{ANDOR:.1f}={corr_at(ANDOR):+.3f}")
print(f"imaging: {N} frames @ {FPS}Hz = {T_end:.1f}s | pupil {pupil.size/10:.1f}s -> aligned {t_beh_c[-1]:.1f}s")

# ---- plot ----
panels = [
    ("ACh Signal",     "ACh (corrected)",  t_img,   ach,     "red"),
    ("Ca Signal",      "Ca (corrected)",   t_img,   ca,      "green"),
    ("Pupil Signal",   "Pupil Dilation",   t_beh_c, pupil_n, "blue"),
    ("Whisker Signal", "Whisker Motion",   t_beh_c, whisk_n, "orange"),
    ("Accelerometer",  "Acceleration",     t_acc_c, acc_c,   "purple"),
]
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
for ax, (title, ylab, t, y, c) in zip(axes, panels):
    ax.plot(t, y, color=c, lw=1.0)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xlim(0, T_end)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
axes[-1].set_xlabel("Time (s)", fontsize=12)
fig.suptitle("rbp4cre_136_phpeb - 2025-12-02 - run4", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig(OUT, dpi=150)
print("saved:", OUT)
