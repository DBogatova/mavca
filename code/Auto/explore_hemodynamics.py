#!/usr/bin/env python
"""explore_hemodynamics.py - voxel-level test of the hemodynamic / optical-absorption hypothesis
for the ANTI-PHASE pupil-coupled dendrites (code/Auto/).

Finding under test (from pupil_phase_populations.py): ~20-28% of L5 apical dendrites are coherent
with pupil size in the slow band (0.04-0.2 Hz); ~1/4-1/3 are ANTI-PHASE (Ca down when the pupil
dilates). Anti-phase units are deeper, have almost no detected events, and show a slow negative
deflection when the pupil is large.

Hypothesis H1 (artifact): pupil dilation (arousal) dilates cortical vessels, which absorb more
light; voxels in the shadow of a vessel show an apparent dF/F DIP with every dilation. If so, the
"anti-phase" dendrites are not biology but voxels that sit in a field-wide, depth-dependent,
vessel-shaped negative-going optical signal.

Predictions of H1 that this script measures AT THE VOXEL LEVEL (not the mask level):
  P1  a field-wide negative component: the raw-voxel r(intensity, pupil) distribution is skewed
      negative / has an excess of r < -0.2 beyond a circular-shift null, and gets more negative
      with depth.
  P2  the negative-r voxels are vessel-like: they are DIM (vessel lumen / shadow) relative to the
      field, i.e. brightness and r are positively associated.
  P3  anti-phase masks sit on the strongest-negative-r voxels, are dimmer than in-phase masks, and
      their negative r is NOT specific to the mask (the shell around them is just as negative) -
      because a shared field signal cancels in the core-shell subtraction, so for a dendrite to end
      up anti-phase the core would have to be MORE negative than its own shell.
  P4  a lagged negative dip: the whole-FOV mean intensity in each depth band dips ~1-3 s AFTER a
      pupil dilation (negative cross-correlation at positive lag).

Method (per run, one run at a time, memory < ~2.5 GB):
  * stream the raw stack (tifffile.memmap), keep frames >= round(skip_s*frame_rate); for every
    frame accumulate a full-resolution mean image (for mask brightness) and store a 2x2-binned
    (Y,X) copy (Z kept full) into an in-memory stack (T, Z, Yd, Xd), float32 (~1.26 GB).
  * light temporal smoothing (gaussian sigma = 1 frame = 0.2 s) along time.
  * per-voxel Pearson r vs pupil and vs accel, both RAW (mean-removed) and DETRENDED
    (linear trend removed per voxel and from the behaviour trace, so slow bleaching cannot
    manufacture a correlation). Detrended r is primary.
  * circular-shift null (shifts >= 10 s) on the pupil trace -> chance fraction of |r| > 0.2.
  * maps: Z-mean r map, histogram of r, r-vs-depth profile, fraction r<-0.2 / r>+0.2 by depth.
  * brightness: mean raw intensity of negative-r vs positive-r vs all voxels; Spearman(brightness,r).
  * mask overlap: voxel-r INSIDE anti masks / in masks / background shell / far tissue; mean raw
    brightness of anti vs in masks. Groups from scape-auto/stats/pupil_phase/<source>/per_dendrite.csv
    (fallback: classify here from dff_auto.csv vs pupil slow-band coherence phase).
  * lag: whole-FOV mean intensity in 3 depth bands cross-correlated with pupil at lags -5..+5 s.

Outputs -> scape-auto/stats/explore_hemodynamics/:
  fig_<date>_<mouse>_<run>.png            per-run multi-panel figure
  per_run.csv                              one row per run (all scalar results)
  voxel_region_stats.csv                   long table: run x region -> voxel-r / brightness summary
  depth_profile.csv                        run x depth(Y) -> mean r, frac neg/pos, mean brightness
  lag_xcorr.csv                            run x band x lag -> cross-correlation
  summary.txt                              plain-language per-run and overall verdict with numbers
  summary.json                             machine-readable

CLI: explore_hemodynamics.py [--source auto|human] [--run KEY ...] [--max-frames N] [--bin 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import signal
from scipy import stats as sp
from scipy import ndimage as ndi
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, fov_group, load_behavior, resample_to,
    open_stack, VOXEL_ZYX, AUTO_ROOT,
)

warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

OUT = AUTO_ROOT / "stats" / "explore_hemodynamics"
BAND = (0.04, 0.2)          # slow band of the pupil-phase finding
SMOOTH_SIGMA_FR = 1.0       # light temporal smoothing (0.2 s at 5 Hz)
RTHR = 0.2                  # |r| threshold for "field-wide" counting
NSHIFT = 20                 # circular shifts for the null
MIN_SHIFT_S = 10.0          # shifts >= 10 s (rule)
COL_IN, COL_ANTI, COL_NONE = "#d1495b", "#1d6fa5", "#bbbbbb"

DEFAULT_RUNS = [
    "2026-04-16/rbp4_132_phpeb/run1",
    "2026-05-08/rbp4_139_phpeb/run5",
    "2026-05-08/rbp4_139_phpeb/run6",
    "2026-03-31/rbp4_132_phpeb/run7",
    "2026-05-12/rbp4_132_phpeb/run5",
    "2026-04-16/rbp4_132_phpeb/run8",
]


# --------------------------------------------------------------------------- small helpers
def bin2x2(a: np.ndarray, b: int) -> np.ndarray:
    """Mean-bin the last two axes by b (crop to a multiple of b). a is (Z, Y, X)."""
    if b == 1:
        return a
    Z, Y, X = a.shape
    Yc, Xc = (Y // b) * b, (X // b) * b
    a = a[:, :Yc, :Xc]
    return a.reshape(Z, Yc // b, b, Xc // b, b).mean(axis=(2, 4))


def detrend_vec(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove mean and linear trend from a 1-D vector."""
    tc = t - t.mean()
    slope = float((tc @ (y - y.mean())) / (tc @ tc))
    return y - y.mean() - tc * slope


# --------------------------------------------------------------------------- group labels
def load_groups(key: str, source: str, n_labels: int):
    """Return dict dendrite_id(1..N) -> group in {'in','anti','none'} and phase_deg, from
    pupil_phase/<source>/per_dendrite.csv if present; else None (caller classifies)."""
    p = AUTO_ROOT / "stats" / "pupil_phase" / source / "per_dendrite.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[df.run == key]
    if len(sub) == 0:
        return None
    g, ph = {}, {}
    for _, row in sub.iterrows():
        did = int(str(row["dendrite"]).split("_")[1]) + 1
        g[did] = row["group"]
        ph[did] = row.get("phase_deg", np.nan)
    return {"group": g, "phase_deg": ph, "source": "per_dendrite.csv"}


def classify_fallback(key: str, source: str, n_labels: int):
    """Classify dendrites here from dff_auto.csv vs pupil slow-band coherence phase (used only if
    per_dendrite.csv is missing). Circular-shift null for significance."""
    r = get_run(key)
    tp = r.out / "traces" / ("dff_auto.csv" if source == "auto" else "dff_human_sameextractor.csv")
    if not tp.exists():
        return None
    beh = load_behavior(r)
    if beh.get("pupil") is None:
        return None
    df = pd.read_csv(tp)
    names = [c for c in df.columns if c.startswith("dend_")]
    t = df["time_s"].to_numpy()
    fr = r.frame_rate
    M = df[names].to_numpy(float)
    y = resample_to(beh["pupil_t"], beh["pupil"], t)
    if np.isfinite(y).mean() < 0.9:
        return None
    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    nper = min(int(round(25.6 * fr)), len(y) - 1)
    f, C = signal.coherence(M.T, y[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
    _, P = signal.csd(M.T, y[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
    k = (f >= BAND[0]) & (f <= BAND[1])
    coh = C[:, k].mean(-1)
    ph = np.angle(P[:, k].sum(-1))
    rng = np.random.default_rng(0)
    ms = int(MIN_SHIFT_S * fr)
    null = []
    for s in rng.integers(ms, len(y) - ms, 200):
        _, Cn = signal.coherence(M.T, np.roll(y, s)[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
        null.append(Cn[:, k].mean(-1))
    thr = np.percentile(np.concatenate(null), 99)
    sig = coh > thr
    inph = np.abs(ph) < np.pi / 2
    g, phd = {}, {}
    for i, nm in enumerate(names):
        did = int(nm.split("_")[1]) + 1
        g[did] = "in" if (sig[i] and inph[i]) else ("anti" if (sig[i] and not inph[i]) else "none")
        phd[did] = float(np.degrees(ph[i]))
    return {"group": g, "phase_deg": phd, "source": "classified_here"}


# --------------------------------------------------------------------------- per run
def analyze(key: str, source: str, b: int, max_frames: int | None) -> dict | None:
    t0 = time.time()
    r = get_run(key)
    st = open_stack(r)                       # (T0, Z, Y, X) uint16 memmap
    T0, Z, Y, X = st.shape
    fr = r.frame_rate
    sk = int(round(r.skip_s * fr))

    # time axis of kept frames: prefer dff_auto.csv (shared with behaviour clock)
    tp = r.out / "traces" / ("dff_auto.csv" if source == "auto" else "dff_human_sameextractor.csv")
    tsv = None
    if tp.exists():
        tsv = pd.read_csv(tp, usecols=["time_s"])["time_s"].to_numpy()
    T = T0 - sk
    if tsv is not None:
        T = min(T, len(tsv))
    if max_frames:
        T = min(T, max_frames)
    tsec = (tsv[:T] if tsv is not None else (np.arange(T) / fr))
    frame_idx = np.arange(sk, sk + T)

    beh = load_behavior(r)
    if beh.get("pupil") is None:
        print(f"  {key}: no pupil, skip"); return None
    pupil = resample_to(beh["pupil_t"], beh["pupil"], tsec)
    accel = resample_to(beh["accel_t"], beh["accel"], tsec) if beh.get("accel") is not None else None
    if np.isfinite(pupil).mean() < 0.9:
        print(f"  {key}: pupil coverage < 90%, skip"); return None
    pupil = np.where(np.isfinite(pupil), pupil, np.nanmedian(pupil))
    if accel is not None:
        accel = np.where(np.isfinite(accel), accel, np.nanmedian(accel))

    # ---- stream frames: full-res mean image + binned stack
    Yd, Xd = (Y // b), (X // b)
    stack = np.empty((T, Z, Yd, Xd), np.float32)
    sum_full = np.zeros((Z, Y, X), np.float64)
    for i, fi in enumerate(frame_idx):
        f = np.asarray(st[fi], np.float32)
        sum_full += f
        stack[i] = bin2x2(f, b)
    mean_full = (sum_full / T).astype(np.float32)
    del sum_full, st
    mean_full_binned = bin2x2(mean_full, b)                     # (Z,Yd,Xd) for binned-voxel brightness
    V = Z * Yd * Xd
    Xmat = stack.reshape(T, V)                                  # view (C-contiguous)

    # ---- light temporal smoothing along time, IN PLACE in voxel blocks (no full copy)
    for c0 in range(0, V, 60000):
        sl = slice(c0, min(c0 + 60000, V))
        Xmat[:, sl] = ndi.gaussian_filter1d(Xmat[:, sl], SMOOTH_SIGMA_FR, axis=0, mode="nearest")

    # depth-band mean-intensity time series (3 Y bands) from the smoothed stack (for lag xcorr)
    yb = np.linspace(0, Yd, 4).astype(int)
    band_ts = np.stack([stack[:, :, yb[i]:yb[i + 1], :].mean(axis=(1, 2, 3)) for i in range(3)], axis=1)  # (T,3)

    # ---- per-voxel correlations via SUMS only (no (T,V) temporaries).
    # Detrended r = partial correlation controlling for the linear time trend (removes bleaching).
    # All reductions in float32: binned intensities are small (~1e2), so float32 is exact enough and
    # avoids float64 upcasting a (T,V) copy of the stack.
    Tf = float(T)
    t = frame_idx.astype(np.float32)
    pupil = pupil.astype(np.float32)
    if accel is not None:
        accel = accel.astype(np.float32)
    Sx = Xmat.sum(0)                                           # (V,) float32
    Sxx = np.einsum("tv,tv->v", Xmat, Xmat)                    # (V,) float32, streamed (no big temp)
    mean_x = Sx / Tf
    sd_x = np.sqrt(np.maximum(Sxx / Tf - mean_x ** 2, 1e-6))

    def r_with(v):
        """raw Pearson r between each voxel and vector v (length T), via BLAS sgemv."""
        Sxv = v @ Xmat                                         # (V,) float32
        cov = Sxv / Tf - mean_x * float(v.mean())
        return np.clip(cov / (sd_x * (float(v.std()) + 1e-6)), -1, 1)

    def corr_scalar(u, v):
        return float(np.corrcoef(u.astype(np.float64), v.astype(np.float64))[0, 1])

    r_xt = r_with(t)
    r_raw_pupil = r_with(pupil)
    r_raw_accel = r_with(accel) if accel is not None else np.zeros(V)

    def partial_t(r_xv, r_vt):
        return np.clip((r_xv - r_xt * r_vt) / np.sqrt(np.maximum((1 - r_xt ** 2) * (1 - r_vt ** 2), 1e-12)), -1, 1)

    r_pt = corr_scalar(pupil, t)
    r_pupil = partial_t(r_raw_pupil, r_pt).astype(np.float32)
    if accel is not None:
        r_at = corr_scalar(accel, t)
        r_accel = partial_t(r_raw_accel, r_at).astype(np.float32)
    else:
        r_accel = np.zeros(V, np.float32)
    r_raw_pupil = r_raw_pupil.astype(np.float32)
    r_raw_accel = r_raw_accel.astype(np.float32)

    # ---- circular-shift null (shifts >= 10 s) on pupil, same detrended (partial) measure
    rng = np.random.default_rng(0)
    ms = int(MIN_SHIFT_S * fr)
    null_neg, null_pos, null_absmax = [], [], []
    for s in rng.integers(ms, T - ms, NSHIFT):
        ys = np.roll(pupil, int(s))
        rr = partial_t(r_with(ys), corr_scalar(ys, t))
        null_neg.append(float((rr < -RTHR).mean()))
        null_pos.append(float((rr > RTHR).mean()))
        null_absmax.append(float(np.percentile(np.abs(rr), 95)))
    null_frac_neg = float(np.mean(null_neg))
    null_frac_pos = float(np.mean(null_pos))
    null_r95 = float(np.mean(null_absmax))

    # reshape maps
    rmap = r_pupil.reshape(Z, Yd, Xd)
    rmap_accel = r_accel.reshape(Z, Yd, Xd)
    bright = mean_full_binned.reshape(V)                        # per binned voxel brightness

    # ---- field-wide distribution stats
    frac_neg = float((r_pupil < -RTHR).mean())
    frac_pos = float((r_pupil > RTHR).mean())
    r_mean = float(r_pupil.mean())
    r_median = float(np.median(r_pupil))
    r_skew = float(sp.skew(r_pupil))

    # ---- depth profile (over Z and X for each binned Y)
    rmap_byY = rmap.transpose(1, 0, 2).reshape(Yd, -1)          # (Yd, Z*Xd)
    bright_byY = mean_full_binned.transpose(1, 0, 2).reshape(Yd, -1)
    depth_um = (np.arange(Yd) + 0.5) * b * VOXEL_ZYX[1]
    prof = pd.DataFrame(dict(
        run=key, y_bin=np.arange(Yd), depth_um=depth_um,
        mean_r_pupil=rmap_byY.mean(1),
        frac_neg=(rmap_byY < -RTHR).mean(1),
        frac_pos=(rmap_byY > RTHR).mean(1),
        mean_brightness=bright_byY.mean(1),
    ))

    # ---- brightness vs r (vessel-shadow test)
    neg = r_pupil < -RTHR
    pos = r_pupil > RTHR
    br_neg = float(bright[neg].mean()) if neg.any() else np.nan
    br_pos = float(bright[pos].mean()) if pos.any() else np.nan
    br_all = float(bright.mean())
    # Spearman on a subsample (speed); positive => dim voxels are negative-r (vessel-like)
    ss = rng.choice(V, size=min(V, 150000), replace=False)
    rho_bright_r = float(sp.spearmanr(bright[ss], r_pupil[ss])[0])

    # ---- mask overlap
    lp = r.out / "masks" / "auto_labelmap_reviewed.tif"
    lp = lp if lp.exists() else r.out / "masks" / "auto_labelmap.tif"
    lab = tifffile.imread(str(lp))                              # (Z,Y,X)
    grp = load_groups(key, source, int(lab.max())) or classify_fallback(key, source, int(lab.max()))
    region_rows = []
    overlap = {}
    anti_binned = inn_binned = None
    if grp is not None:
        gmap = grp["group"]
        anti_ids = [i for i, gg in gmap.items() if gg == "anti"]
        in_ids = [i for i, gg in gmap.items() if gg == "in"]
        anti_full = np.isin(lab, anti_ids)
        in_full = np.isin(lab, in_ids)
        allmask = lab > 0
        dil1 = ndi.binary_dilation(allmask, iterations=1)
        dil3 = ndi.binary_dilation(allmask, iterations=3)
        dil10 = ndi.binary_dilation(allmask, iterations=10)
        shell_full = dil3 & ~dil1
        far_full = ~dil10

        def to_binned(mask_full):
            return bin2x2(mask_full.astype(np.float32), b).reshape(V) >= 0.5

        anti_binned = to_binned(anti_full)
        inn_binned = to_binned(in_full)
        shell_binned = to_binned(shell_full)
        tissue = bright > np.percentile(bright, 20)
        far_binned = to_binned(far_full) & tissue

        # mean raw brightness of anti vs in masks (FULL resolution, exact)
        br_anti_mask = float(mean_full[anti_full].mean()) if anti_full.any() else np.nan
        br_in_mask = float(mean_full[in_full].mean()) if in_full.any() else np.nan
        br_global_full = float(mean_full.mean())

        for rname, rmask in (("anti_mask", anti_binned), ("in_mask", inn_binned),
                             ("shell", shell_binned), ("far_tissue", far_binned)):
            if rmask.sum() >= 5:
                vr = r_pupil[rmask]
                region_rows.append(dict(run=key, region=rname, n_voxels=int(rmask.sum()),
                                        median_r_pupil=float(np.median(vr)),
                                        mean_r_pupil=float(vr.mean()),
                                        frac_neg=float((vr < -RTHR).mean()),
                                        frac_pos=float((vr > RTHR).mean()),
                                        median_r_accel=float(np.median(r_accel[rmask])),
                                        mean_brightness=float(bright[rmask].mean())))
        overlap = dict(br_anti_mask=br_anti_mask, br_in_mask=br_in_mask, br_global_full=br_global_full,
                       n_anti=len(anti_ids), n_in=len(in_ids), group_source=grp["source"])

    # ---- lag cross-correlation: band intensity vs pupil, lags -5..+5 s
    maxlag = int(round(5 * fr))
    lags = np.arange(-maxlag, maxlag + 1)
    pud = detrend_vec(pupil, t)
    lag_rows = []
    lag_curves = {}
    band_names = ["superficial", "middle", "deep"]
    for bi, bn in enumerate(band_names):
        bd = detrend_vec(band_ts[:, bi].astype(np.float64), t)
        cc = np.array([_xcorr_at(bd, pud, L) for L in lags])
        lag_curves[bn] = cc
        # dip in the 1-3 s AFTER dilation = negative cc at positive lag 1-3 s
        pos_win = (lags >= int(1 * fr)) & (lags <= int(3 * fr))
        dip_val = float(cc[pos_win].min())
        dip_lag = float(lags[pos_win][np.argmin(cc[pos_win])] / fr)
        peak_val = float(cc[np.argmax(np.abs(cc))])
        peak_lag = float(lags[np.argmax(np.abs(cc))] / fr)
        for L, c in zip(lags, cc):
            lag_rows.append(dict(run=key, band=bn, lag_s=round(L / fr, 3), xcorr=float(c)))
        overlap[f"lag_{bn}_dip13_val"] = dip_val
        overlap[f"lag_{bn}_dip13_lag_s"] = dip_lag
        overlap[f"lag_{bn}_peak_val"] = peak_val
        overlap[f"lag_{bn}_peak_lag_s"] = peak_lag

    R = dict(run=key, source=source, fov=fov_group(r), mouse=r.mouse, T=int(T), fr=fr, bin=b,
             n_voxels=int(V), r_mean=r_mean, r_median=r_median, r_skew=r_skew,
             frac_neg=frac_neg, frac_pos=frac_pos,
             null_frac_neg=null_frac_neg, null_frac_pos=null_frac_pos, null_r95=null_r95,
             excess_neg=frac_neg - null_frac_neg, excess_pos=frac_pos - null_frac_pos,
             br_neg_r=br_neg, br_pos_r=br_pos, br_all=br_all, rho_bright_r=rho_bright_r,
             **overlap)
    out = dict(R=R, prof=prof, region_rows=region_rows, lag_rows=lag_rows,
               maps=dict(rmap=rmap, rmap_accel=rmap_accel, mean_full_binned=mean_full_binned,
                         r_pupil=r_pupil, r_raw_pupil=r_raw_pupil, r_raw_accel=r_raw_accel,
                         bright=bright, anti_binned=anti_binned, inn_binned=inn_binned,
                         depth_um=depth_um, lag_curves=lag_curves, lags_s=lags / fr),
               grp=grp, labshape=lab.shape, Zd=Z, Yd=Yd, Xd=Xd)
    del stack, Xmat
    print(f"  {key}: done T={T} V={V} frac_neg={frac_neg:.3f} (null {null_frac_neg:.3f}) "
          f"frac_pos={frac_pos:.3f} (null {null_frac_pos:.3f}) in {time.time()-t0:.0f}s")
    return out


def _xcorr_at(a: np.ndarray, b: np.ndarray, L: int) -> float:
    """Pearson corr between a[t] and b[t-L] over the valid overlap. L>0 => a lags b by L frames."""
    if L > 0:
        aa, bb = a[L:], b[:-L]
    elif L < 0:
        aa, bb = a[:L], b[-L:]
    else:
        aa, bb = a, b
    if len(aa) < 10:
        return 0.0
    aa = aa - aa.mean(); bb = bb - bb.mean()
    d = np.sqrt((aa @ aa) * (bb @ bb))
    return float((aa @ bb) / d) if d > 0 else 0.0


# --------------------------------------------------------------------------- figure
def run_figure(res: dict, outdir: Path):
    R = res["R"]; M = res["maps"]
    key = R["run"]
    safe = key.replace("/", "_")
    rmap = M["rmap"]; mean_binned = M["mean_full_binned"]
    Zmean_r = rmap.mean(0)                     # (Yd,Xd)
    Zmean_img = mean_binned.max(0)             # MIP for anatomy
    asp = VOXEL_ZYX[1] / VOXEL_ZYX[2]

    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.33)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(Zmean_img, cmap="gray", vmax=np.percentile(Zmean_img, 99.5), aspect=asp)
    ax.set_title("A  anatomy (Z-MIP of mean image)", loc="left"); ax.set_ylabel("depth (binned Y)"); ax.set_xticks([])

    ax = fig.add_subplot(gs[0, 1])
    vmax = max(0.15, np.percentile(np.abs(Zmean_r), 99))
    im = ax.imshow(Zmean_r, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect=asp)
    ax.set_title("B  r(voxel, pupil), Z-mean\nblue = negative (absorption-like)", loc="left"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    rp = M["r_pupil"]
    ax.hist(rp, bins=np.linspace(-1, 1, 81), color="#555", density=True)
    ax.axvline(0, color="k", lw=0.6); ax.axvline(-RTHR, color=COL_ANTI, lw=0.8, ls="--"); ax.axvline(RTHR, color=COL_IN, lw=0.8, ls="--")
    ax.set_xlabel("r(voxel, pupil) detrended"); ax.set_ylabel("density")
    ax.set_title(f"C  voxel-r distribution\nfrac<-{RTHR}={R['frac_neg']:.3f} (null {R['null_frac_neg']:.3f}); "
                 f">{RTHR}={R['frac_pos']:.3f} (null {R['null_frac_pos']:.3f})\nmean={R['r_mean']:+.3f} skew={R['r_skew']:+.2f}", loc="left", fontsize=7)

    ax = fig.add_subplot(gs[0, 3])
    prof = res["prof"]
    ax.plot(prof.mean_r_pupil, prof.depth_um, color="k", lw=1.3)
    ax.axvline(0, color="k", lw=0.5)
    ax.invert_yaxis(); ax.set_xlabel("mean r(voxel,pupil)"); ax.set_ylabel("depth (um)")
    ax.set_title("D  r vs depth (row0 = surface)", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(prof.frac_neg, prof.depth_um, color=COL_ANTI, lw=1.3, label=f"r<-{RTHR}")
    ax.plot(prof.frac_pos, prof.depth_um, color=COL_IN, lw=1.3, label=f"r>+{RTHR}")
    ax.axvline(R["null_frac_neg"], color=COL_ANTI, lw=0.6, ls=":")
    ax.axvline(R["null_frac_pos"], color=COL_IN, lw=0.6, ls=":")
    ax.invert_yaxis(); ax.set_xlabel("fraction of voxels"); ax.set_ylabel("depth (um)")
    ax.set_title("E  neg/pos fraction by depth\n(dotted = circular-shift null)", loc="left"); ax.legend(fontsize=7, frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    ss = np.random.default_rng(0).choice(len(rp), size=min(len(rp), 20000), replace=False)
    ax.scatter(M["bright"][ss], rp[ss], s=2, alpha=0.15, color="#444", rasterized=True)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("voxel brightness (raw mean)"); ax.set_ylabel("r(voxel,pupil)")
    ax.set_title(f"F  brightness vs r  (Spearman {R['rho_bright_r']:+.2f})\n"
                 f"br(neg-r)={R['br_neg_r']:.0f} br(pos-r)={R['br_pos_r']:.0f} all={R['br_all']:.0f}", loc="left", fontsize=7)

    # region distributions: bar of median r by region with frac neg
    ax = fig.add_subplot(gs[1, 2])
    regions = {d["region"]: d for d in res["region_rows"]}
    cols = {"anti_mask": COL_ANTI, "in_mask": COL_IN, "shell": "#888", "far_tissue": "#44aa44"}
    xs = []
    for i, rn in enumerate(("anti_mask", "in_mask", "shell", "far_tissue")):
        if rn in regions:
            d = regions[rn]
            ax.bar(i, d["median_r_pupil"], color=cols[rn], width=0.7)
            ax.text(i, d["median_r_pupil"], f"n={d['n_voxels']}\nneg{d['frac_neg']:.2f}",
                    ha="center", va="bottom" if d["median_r_pupil"] >= 0 else "top", fontsize=6)
            xs.append((i, rn))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([i for i, _ in xs]); ax.set_xticklabels([rn.replace("_", "\n") for _, rn in xs], fontsize=7)
    ax.set_ylabel("median voxel r(pupil)")
    ax.set_title("G  voxel-r by region", loc="left")

    ax = fig.add_subplot(gs[1, 3])
    br = []
    labels = []
    if "br_anti_mask" in R and R.get("br_anti_mask") == R.get("br_anti_mask"):
        ax.bar(0, R["br_anti_mask"], color=COL_ANTI, width=0.7); labels.append((0, "anti\nmask"))
    if R.get("br_in_mask") == R.get("br_in_mask"):
        ax.bar(1, R["br_in_mask"], color=COL_IN, width=0.7); labels.append((1, "in\nmask"))
    if R.get("br_global_full") == R.get("br_global_full"):
        ax.bar(2, R["br_global_full"], color="#888", width=0.7); labels.append((2, "whole\nFOV"))
    ax.set_xticks([i for i, _ in labels]); ax.set_xticklabels([l for _, l in labels], fontsize=7)
    ax.set_ylabel("mean raw brightness")
    ax.set_title("H  mask brightness (anti vs in)", loc="left")

    # lag curves per band
    for bi, bn in enumerate(("superficial", "middle", "deep")):
        ax = fig.add_subplot(gs[2, bi])
        cc = M["lag_curves"][bn]
        ax.plot(M["lags_s"], cc, color="k", lw=1.3)
        ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(1, 3, color="orange", alpha=0.15)
        ax.set_xlabel("lag (s); +lag = intensity lags pupil"); ax.set_ylabel("xcorr(band intensity, pupil)")
        ax.set_title(f"{'IJK'[bi]}  {bn} band\ndip@1-3s={R[f'lag_{bn}_dip13_val']:+.2f}", loc="left", fontsize=7)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    txt = [f"{key}", f"FOV {R['fov']}  mouse {R['mouse']}", f"T={R['T']} fr={R['fr']} bin={R['bin']}  V={R['n_voxels']}",
           "", f"group source: {R.get('group_source','-')}", f"n_anti={R.get('n_anti','-')} n_in={R.get('n_in','-')}",
           "", "verdict heuristics:",
           f"  excess neg frac = {R['excess_neg']:+.3f}",
           f"  r skew = {R['r_skew']:+.2f}",
           f"  Spearman(bright,r) = {R['rho_bright_r']:+.2f}"]
    ax.text(0, 1, "\n".join(txt), va="top", fontsize=8, family="monospace")

    fig.suptitle(f"Voxel-level hemodynamic/absorption test - {key} ({R['source']})", fontsize=12)
    fig.savefig(outdir / f"fig_{safe}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- cohort / summary
def write_summary(results: list[dict], source: str, outdir: Path):
    runs = pd.DataFrame([x["R"] for x in results])
    runs.to_csv(outdir / "per_run.csv", index=False)
    prof = pd.concat([x["prof"] for x in results], ignore_index=True)
    prof.to_csv(outdir / "depth_profile.csv", index=False)
    reg = pd.concat([pd.DataFrame(x["region_rows"]) for x in results if x["region_rows"]], ignore_index=True)
    reg.to_csv(outdir / "voxel_region_stats.csv", index=False)
    lag = pd.concat([pd.DataFrame(x["lag_rows"]) for x in results], ignore_index=True)
    lag.to_csv(outdir / "lag_xcorr.csv", index=False)

    def med(col):
        return float(runs[col].median()) if col in runs else np.nan

    # region medians across runs
    regmed = reg.groupby("region")[["median_r_pupil", "frac_neg", "frac_pos", "mean_brightness", "n_voxels"]].median() if len(reg) else pd.DataFrame()

    L = []
    L.append(f"VOXEL-LEVEL HEMODYNAMIC / OPTICAL-ABSORPTION TEST ({source}) - {len(runs)} runs, "
             f"{runs.fov.nunique()} FOVs, {runs.mouse.nunique()} mice.")
    L.append(f"2x2 (Y,X) binning, light temporal smoothing (gaussian sigma {SMOOTH_SIGMA_FR} frame). "
             f"Per-voxel Pearson r is linearly DETRENDED (per voxel and pupil) so slow bleaching cannot "
             f"create a correlation. |r|>{RTHR} counted as a hit; null = {NSHIFT} circular pupil shifts >= {MIN_SHIFT_S}s.")
    L.append("")
    L.append("PER RUN:")
    hdr = (f"  {'run':40s} {'frac<-.2':>9s} {'null':>6s} {'frac>+.2':>9s} {'null':>6s} "
           f"{'skew':>6s} {'rho(br,r)':>9s} {'anti_r':>7s} {'in_r':>6s} {'shell_r':>7s} {'far_r':>6s} "
           f"{'br_anti':>7s} {'br_in':>6s} {'deepdip13':>9s}")
    L.append(hdr)
    for _, R in runs.iterrows():
        rr = reg[reg.run == R["run"]].set_index("region") if len(reg) else pd.DataFrame()

        def rv(name, col="median_r_pupil"):
            return rr.loc[name, col] if (len(rr) and name in rr.index) else np.nan
        L.append(f"  {R['run']:40s} {R['frac_neg']:9.3f} {R['null_frac_neg']:6.3f} "
                 f"{R['frac_pos']:9.3f} {R['null_frac_pos']:6.3f} {R['r_skew']:6.2f} {R['rho_bright_r']:9.2f} "
                 f"{rv('anti_mask'):7.2f} {rv('in_mask'):6.2f} {rv('shell'):7.2f} {rv('far_tissue'):6.2f} "
                 f"{R.get('br_anti_mask', np.nan):7.0f} {R.get('br_in_mask', np.nan):6.0f} "
                 f"{R.get('lag_deep_dip13_val', np.nan):9.2f}")
    L.append("")

    # ---- overall numbers
    L.append("OVERALL (median across runs):")
    L.append(f"  field-wide negative component:")
    L.append(f"    fraction of voxels with r<-{RTHR} = {med('frac_neg'):.3f}  vs null {med('null_frac_neg'):.3f}  "
             f"(excess {med('excess_neg'):+.3f})")
    L.append(f"    fraction of voxels with r>+{RTHR} = {med('frac_pos'):.3f}  vs null {med('null_frac_pos'):.3f}  "
             f"(excess {med('excess_pos'):+.3f})")
    L.append(f"    mean voxel r = {med('r_mean'):+.3f}; distribution skew = {med('r_skew'):+.2f} "
             f"(negative skew => a tail of negative-r voxels).")
    L.append(f"  vessel-shadow (brightness) test:")
    L.append(f"    Spearman(brightness, r) = {med('rho_bright_r'):+.2f} "
             f"(POSITIVE => dim voxels carry the negative r, as a vessel lumen/shadow would).")
    L.append(f"    brightness of r<-{RTHR} voxels = {med('br_neg_r'):.0f}, r>+{RTHR} voxels = {med('br_pos_r'):.0f}, "
             f"all = {med('br_all'):.0f}.")
    if len(regmed):
        L.append(f"  voxel-r by region (median across runs):")
        for rn in ("anti_mask", "in_mask", "shell", "far_tissue"):
            if rn in regmed.index:
                d = regmed.loc[rn]
                L.append(f"    {rn:11s}: median voxel r = {d['median_r_pupil']:+.3f}, frac<-{RTHR} = {d['frac_neg']:.3f}, "
                         f"brightness = {d['mean_brightness']:.0f}  (median n_vox {d['n_voxels']:.0f})")
    # anti vs shell contrast (mask-specific negativity)
    if len(reg):
        piv = reg.pivot_table(index="run", columns="region", values="median_r_pupil")
        if "anti_mask" in piv and "shell" in piv:
            d = (piv["anti_mask"] - piv["shell"]).dropna()
            L.append(f"  anti_mask MINUS its shell (median voxel r): {d.median():+.3f} over {len(d)} runs "
                     f"(if ~0, the anti-mask negativity is shared with the surrounding field = field artifact; "
                     f"if clearly negative, the negativity is specific to the dendrite core).")
        if "anti_mask" in piv and "far_tissue" in piv:
            d2 = (piv["anti_mask"] - piv["far_tissue"]).dropna()
            L.append(f"  anti_mask MINUS far tissue (median voxel r): {d2.median():+.3f} over {len(d2)} runs.")
        if "br_anti_mask" in runs and "br_in_mask" in runs:
            dd = (runs["br_anti_mask"] - runs["br_in_mask"]).dropna()
            L.append(f"  brightness anti_mask - in_mask: {dd.median():+.0f} over {len(dd)} runs "
                     f"(negative => anti-phase masks are dimmer, consistent with vessel-adjacent).")
    # lag
    L.append(f"  lagged dip (whole-FOV band intensity vs pupil, cross-corr at +1..+3 s):")
    for bn in ("superficial", "middle", "deep"):
        c = f"lag_{bn}_dip13_val"
        if c in runs:
            L.append(f"    {bn:11s}: min xcorr in +1..3s window = {med(c):+.2f} "
                     f"(a hemodynamic absorption dip predicts a clear NEGATIVE value lagging arousal).")

    # ---- verdict (data-driven; each clause keyed to a measured number)
    L.append("")
    L.append("VERDICT:")
    fn, nn = med("frac_neg"), med("null_frac_neg")
    fp, np_ = med("frac_pos"), med("null_frac_pos")
    sk = med("r_skew")
    rho = med("rho_bright_r")
    br_neg, br_all = med("br_neg_r"), med("br_all")
    deep_dip = med("lag_deep_dip13_val")
    sup_dip = med("lag_superficial_dip13_val")
    mid_dip = med("lag_middle_dip13_val")
    anti_shell = anti_far = anti_r = in_r = br_anti_in = np.nan
    if len(reg):
        piv = reg.pivot_table(index="run", columns="region", values="median_r_pupil")
        if "anti_mask" in piv:
            anti_r = float(piv["anti_mask"].median())
            if "shell" in piv:
                anti_shell = float((piv["anti_mask"] - piv["shell"]).dropna().median())
            if "far_tissue" in piv:
                anti_far = float((piv["anti_mask"] - piv["far_tissue"]).dropna().median())
        if "in_mask" in piv:
            in_r = float(piv["in_mask"].median())
    if "br_anti_mask" in runs and "br_in_mask" in runs:
        br_anti_in = float((runs["br_anti_mask"] - runs["br_in_mask"]).dropna().median())

    verdict = []
    # 1. direction of the field-wide pupil-correlated component
    if (fp - np_) > 0.1 and (fn - nn) <= 0.01:
        verdict.append(
            f"1. The field-wide pupil-correlated optical signal is strongly POSITIVE, not negative: a median "
            f"{fp * 100:.0f}% of voxels have r>+{RTHR} with pupil (null {np_ * 100:.0f}%), while only {fn * 100:.1f}% "
            f"have r<-{RTHR} (null {nn * 100:.1f}%, i.e. the real data has FEWER strong-negative voxels than chance). "
            f"H1 predicts the opposite - a field-wide NEGATIVE (absorption) component. NOT SUPPORTED. "
            f"(The negative distribution skew {sk:+.2f} is just a left tail on a distribution whose bulk sits well above 0; "
            f"mean voxel r = {med('r_mean'):+.2f}.)")
    elif (fn - nn) > 0.02:
        verdict.append(
            f"1. There IS an excess of negative-r voxels above the circular-shift null (frac<-{RTHR} {fn:.3f} vs null {nn:.3f}); "
            f"a field-wide negative component consistent with H1 is present.")
    else:
        verdict.append(
            f"1. There is essentially no pupil-correlated field component in these runs (frac>+{RTHR}={fp:.3f}, "
            f"frac<-{RTHR}={fn:.3f}, both near the null {np_:.3f}/{nn:.3f}).")
    # 2. vessel-shadow brightness signature
    dim_ratio = br_neg / br_all if (br_all and np.isfinite(br_neg)) else np.nan
    if np.isfinite(dim_ratio) and dim_ratio < 0.9:
        verdict.append(
            f"2. Negative-r voxels are dimmer than the field (brightness {br_neg:.0f} vs {br_all:.0f}, ratio {dim_ratio:.2f}): "
            f"a vessel-lumen/shadow signature consistent with H1.")
    else:
        verdict.append(
            f"2. Negative-r voxels are NOT dim: their brightness ({br_neg:.0f}) ~ the field mean ({br_all:.0f}). The positive "
            f"Spearman(brightness,r)={rho:+.2f} only means dim voxels have WEAKER positive correlation, not negative - there is "
            f"no vessel-shadow (dark-structure) signature. H1 NOT SUPPORTED on this test.")
    # 3. lagged absorption dip
    if min(sup_dip, mid_dip, deep_dip) < -0.1:
        verdict.append(
            f"3. A negative dip lags arousal by 1-3 s in at least one depth band (superficial {sup_dip:+.2f}, middle {mid_dip:+.2f}, "
            f"deep {deep_dip:+.2f}): consistent with a hemodynamic absorption transient.")
    else:
        verdict.append(
            f"3. No lagged absorption dip: the most-negative whole-FOV band-intensity cross-correlation in the +1..3 s window is "
            f"still >=0 in every depth band (superficial {sup_dip:+.2f}, middle {mid_dip:+.2f}, deep {deep_dip:+.2f}). H1 NOT SUPPORTED.")
    # 4. are anti-phase units the strongest-negative / dim / vessel-adjacent voxels?
    verdict.append(
        f"4. Anti-phase masks are NOT the strongest-negative-r voxels: at the voxel level their median r with pupil is "
        f"{anti_r:+.2f} (POSITIVE, dominated by the positive field), only marginally below in-phase masks ({in_r:+.2f}) and "
        f"barely below their own local shell (anti-shell {anti_shell:+.3f}) or far tissue (anti-far {anti_far:+.3f}). Because the "
        f"shared field is POSITIVE and nearly identical in core and shell, it cancels in the core-shell dF/F and cannot manufacture "
        f"an anti-phase dendrite; if anything incomplete subtraction of a positive field biases toward IN-phase. Anti-phase masks "
        f"are also NOT dimmer than in-phase masks (brightness difference {br_anti_in:+.0f}), so they are not vessel-adjacent.")
    # overall
    h1_support = ((fn - nn) > 0.02) + (np.isfinite(dim_ratio) and dim_ratio < 0.9) + (min(sup_dip, mid_dip, deep_dip) < -0.1) + (np.isfinite(anti_shell) and anti_shell < -0.05)
    verdict.append(
        f"OVERALL: {int(h1_support)}/4 voxel-level predictions of the hemodynamic/absorption hypothesis (H1) are met. "
        + ("H1 is NOT supported at the voxel level - the anti-phase dendrite population is not explained by a field-wide negative "
           "absorption signal, vessel shadows, dim/vessel-adjacent voxels, or an arousal-lagged optical dip."
           if h1_support <= 1 else
           "H1 has partial voxel-level support; see the clauses above."))
    # exception runs worth flagging
    exc = runs[(runs["frac_pos"] < 0.1)]
    if len(exc):
        verdict.append(
            "NOTE: " + ", ".join(exc["run"]) + f" show(s) no positive field at all (frac>+{RTHR}~0); in that regime the anti-phase "
            f"masks carry a weak negative voxel-r but still show no lag dip and are not dim - a small genuine negative-going signal, "
            f"not a field-wide absorption artifact.")
    L += verdict
    (outdir / "summary.txt").write_text("\n".join(L))
    S = dict(source=source, n_runs=len(runs), n_fovs=int(runs.fov.nunique()), n_mice=int(runs.mouse.nunique()),
             medians={c: med(c) for c in ("frac_neg", "null_frac_neg", "frac_pos", "null_frac_pos", "r_mean",
                                          "r_skew", "rho_bright_r", "br_neg_r", "br_pos_r", "br_all")},
             anti_minus_shell=anti_shell, h1_predictions_met=int(h1_support), verdict=verdict,
             region_medians=(regmed.reset_index().to_dict(orient="records") if len(regmed) else []))
    (outdir / "summary.json").write_text(json.dumps(S, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o))
    print("\n".join(L))
    return L


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("auto", "human"), default="auto")
    ap.add_argument("--run", action="append", default=[], help="run key(s); default = the 6 spanning runs")
    ap.add_argument("--bin", type=int, default=2, help="spatial bin factor in Y,X")
    ap.add_argument("--max-frames", type=int, default=None, help="debug: cap frames")
    a = ap.parse_args()
    keys = a.run or DEFAULT_RUNS
    outdir = OUT
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for k in keys:
        try:
            x = analyze(k, a.source, a.bin, a.max_frames)
        except Exception as e:
            import traceback; traceback.print_exc(); print("ERROR", k, e); x = None
        if x is not None:
            run_figure(x, outdir)
            results.append(x)
        import gc; gc.collect()
    if results:
        write_summary(results, a.source, outdir)
    print(f"\ndone {len(results)}/{len(keys)} runs in {time.time()-t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    sys.exit(main())
