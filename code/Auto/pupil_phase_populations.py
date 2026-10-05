#!/usr/bin/env python
"""pupil_phase_populations.py - the two pupil-coupled dendrite populations: figure, statistics and
artifact controls (code/Auto/).

Finding under test: ~20% of apical dendrites are coherent with pupil size in the slow band
(0.04-0.2 Hz). Two thirds are IN PHASE (Ca up when the pupil dilates), one third ANTI-PHASE
(Ca down when the pupil dilates). The anti-phase group is deeper, fires less, does not follow
locomotion, and a dendrite keeps its phase across runs of the same field of view.

Alternative explanations that must be excluded before calling this biology:
  H1  hemodynamic / absorption artifact: arousal dilates vessels, more light is absorbed, deep
      voxels show an apparent dF/F DIP with every dilation. Prediction: anti-phase signal is a
      slow negative deflection shared with the out-of-mask background, absent from the event
      train, stronger with depth, and present in the raw (un-subtracted) trace of the shell.
  H2  blink / tracking dips in the pupil trace: sharp drops create broadband noise. Prediction:
      classification changes when frames around dips are removed.
  H3  locomotion confound: pupil co-varies with running; anti-phase could be 'anti-locomotion'.
      Prediction: phase disappears when pupil is partialled for accel + whisker, or in still frames.
  H4  statistical artifact of few Welch segments: phase is unstable. Prediction: phase differs
      between the two halves of the run and is not reproducible across runs.
  H5  mask artifact: anti-phase units are fragments / background. Prediction: the two halves of
      the dendrite (top / bottom along Y) disagree in phase; anti-phase units have no events.

Controls computed per run (source auto; human as replication):
  * phase from (a) the dF/F trace, (b) the high-passed trace (> 0.1 Hz, removes slow dips),
    (c) the binary event train (positive transients only), (d) pupil partialled for accel+whisker,
    (e) frames around pupil dips removed, (f) each half of the run, (g) top vs bottom half of the
    mask (raw stack, own shells);
  * background (all voxels outside every mask, in the same depth band) vs pupil: phase and coherence;
  * event-triggered pupil (pupil around Ca onsets) and pupil-dilation-triggered Ca per group;
  * group properties: depth, size, event rate, amplitude, locomotion / whisker coupling, global
    participation, within/between group r; LMM group ~ depth + rate + locomotion (FOV random).
  * cross-run identity: phase agreement matrix for dendrites coherent in both runs.

Outputs: scape-auto/stats/pupil_phase/<source>/ {per_dendrite.csv, per_run.csv, summary.txt,
         summary.json, fig_pupil_phase.png/pdf (talk figure), fig_controls.png/pdf}
CLI: pupil_phase_populations.py [--source auto|human|both] [--jobs N] [--example KEY]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    discover_runs, get_run, Run, fov_group, load_behavior, resample_to, human_masks, human_masks_valid,
    open_stack, VOXEL_ZYX, AUTO_ROOT,
)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore")
OUT = AUTO_ROOT / "stats" / "pupil_phase"
BAND = (0.04, 0.2)
SEG_S = 25.6
NSH = 300
COL_IN, COL_ANTI, COL_NONE = "#d1495b", "#1d6fa5", "#bbbbbb"


# ----------------------------------------------------------------------------- helpers
def zc(M):
    M = M - M.mean(0)
    return M / (np.linalg.norm(M, axis=0) + 1e-12)


def rz(M):
    med = np.median(M, 0)
    return (M - med) / (np.median(np.abs(M - med), 0) * 1.4826 + 1e-9)


def coh_phase(X, y, fr, band=BAND):
    """X (T,N), y (T,) -> band coherence (N,), band phase (N,) in radians (|phase|<pi/2 = in phase)."""
    nper = int(round(SEG_S * fr))
    nper = min(nper, len(y) - 1)
    f, C = signal.coherence(X.T, y[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
    _, P = signal.csd(X.T, y[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
    k = (f >= band[0]) & (f <= band[1])
    return C[:, k].mean(-1), np.angle(P[:, k].sum(-1))


def classify(X, y, fr, rng, nsh=NSH, alpha=0.01):
    """Return coh, phase, significant (coh above shift null 1-alpha), in-phase bool."""
    c, ph = coh_phase(X, y, fr)
    T = len(y)
    ms = int(10 * fr)
    null = np.concatenate([coh_phase(X, np.roll(y, s), fr)[0] for s in rng.integers(ms, T - ms, nsh)])
    thr = np.percentile(null, 100 * (1 - alpha))
    return c, ph, c > thr, np.abs(ph) < np.pi / 2


def highpass(M, fr, fc=0.1):
    b, a = signal.butter(2, fc / (fr / 2), "high")
    return signal.filtfilt(b, a, M, axis=0)


def event_train(M, fr):
    Z = rz(M)
    A = Z > 3
    E = np.zeros_like(M)
    for j in range(M.shape[1]):
        a = np.r_[False, A[:, j], False]
        d = np.diff(a.astype(int))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for s, e in zip(st, en):
            if e - s >= 2:
                E[s:e, j] = 1
    return E


def partial_out(y, regs):
    X = np.c_[np.ones(len(y)), *regs]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def dip_frames(y, fr, pad_s=1.0):
    """Frames around fast pupil drops (blinks / lost tracking): dy below -4 MAD."""
    dy = np.diff(y, prepend=y[0])
    med = np.median(dy)
    mad = np.median(np.abs(dy - med)) * 1.4826 + 1e-12
    bad = dy < med - 4 * mad
    return ndi.binary_dilation(bad, iterations=int(pad_s * fr))


def half_mask_traces(r: Run, units: list[np.ndarray], shape, excl):
    """Top/bottom-half raw traces with own shells (Z,Y,X flat indices). Returns (T, n, 2)."""
    Z, Y, X = shape
    st = ndi.generate_binary_structure(3, 1)

    def shell(vid):
        m = np.zeros(Z * Y * X, bool)
        m[vid] = True
        m = m.reshape(shape)
        sl = tuple(slice(max(0, int(c.min()) - 4), int(c.max()) + 5) for c in np.unravel_index(vid, shape))
        o = ndi.binary_dilation(m[sl], st, iterations=3) & ~ndi.binary_dilation(m[sl], st, iterations=1)
        full = np.zeros(shape, bool)
        full[sl] = o
        return np.flatnonzero(full.ravel() & ~excl)
    halves = []
    for vid in units:
        yy = np.unravel_index(vid, shape)[1]
        med = np.median(yy)
        a, b = vid[yy < med], vid[yy >= med]
        halves.append((a, shell(a), b, shell(b)) if a.size >= 15 and b.size >= 15 else None)
    s = open_stack(r)
    sk = int(round(r.skip_s * r.frame_rate))
    T = s.shape[0] - sk
    out = np.full((T, len(units), 2), np.nan)
    core = np.zeros((T, len(units)))      # whole mask, NO shell subtraction
    shl = np.zeros((T, len(units)))       # shell alone
    for t in range(T):
        f = np.asarray(s[t + sk], np.float32).ravel()
        for k, h in enumerate(halves):
            core[t, k] = f[units[k]].mean()
            if h is not None:
                sh = np.r_[h[1], h[3]]
                shl[t, k] = f[sh].mean() if sh.size else np.nan
                out[t, k, 0] = f[h[0]].mean() - (f[h[1]].mean() if h[1].size else 0)
                out[t, k, 1] = f[h[2]].mean() - (f[h[3]].mean() if h[3].size else 0)
    return out, core, shl


def load_units(r: Run, source: str, names):
    if source == "auto":
        lp = r.out / "masks" / "auto_labelmap_reviewed.tif"
        lp = lp if lp.exists() else r.out / "masks" / "auto_labelmap.tif"
        lab = tifffile.imread(str(lp))
        units = [np.flatnonzero(lab.ravel() == int(n.split("_")[1]) + 1) for n in names]
        shape = lab.shape
        excl = lab.ravel() > 0
    else:
        shape = open_stack(r).shape[1:]
        hm = dict(human_masks(r, shape))
        units = [np.flatnonzero(hm[n].ravel()) for n in names]
        excl = np.zeros(int(np.prod(shape)), bool)
        for u in units:
            excl[u] = True
    return units, shape, excl


def depth_band_background(r: Run, shape, excl, units, fr):
    """Out-of-mask background trace in three depth thirds -> (T, 3)."""
    Z, Y, X = shape
    yy = np.unravel_index(np.flatnonzero(~excl), shape)[1]
    s = open_stack(r)
    sk = int(round(r.skip_s * fr))
    T = s.shape[0] - sk
    idx = np.flatnonzero(~excl)
    bands = [idx[(yy >= Y * i / 3) & (yy < Y * (i + 1) / 3)] for i in range(3)]
    out = np.zeros((T, 3))
    for t in range(T):
        f = np.asarray(s[t + sk], np.float32).ravel()
        for i, b in enumerate(bands):
            out[t, i] = f[b].mean()
    return out


# ----------------------------------------------------------------------------- per run
def analyze(key: str, source: str, heavy=True) -> dict | None:
    r = get_run(key)
    p = r.out / "traces" / ("dff_auto.csv" if source == "auto" else "dff_human_sameextractor.csv")
    if not p.exists() or (source == "human" and not human_masks_valid(r)):
        return None
    beh = load_behavior(r)
    if beh.get("pupil") is None:
        return None
    df = pd.read_csv(p)
    names = [c for c in df.columns if c.startswith("dend_")]
    t = df["time_s"].to_numpy()
    fr = r.frame_rate
    M = df[names].to_numpy(float)
    T, N = M.shape
    y = resample_to(beh["pupil_t"], beh["pupil"], t)
    if np.isfinite(y).mean() < 0.9:
        return None
    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    regs = []
    B = {}
    for k in ("accel", "whisker"):
        if beh.get(k) is not None:
            v = resample_to(beh[f"{k}_t"], beh[k], t)
            v = np.where(np.isfinite(v), v, np.nanmedian(v))
            B[k] = v
            regs.append(v)
    g = pd.read_csv(r.out / "traces" / "global_ca.csv")
    gy = resample_to(g["time_s"].to_numpy(), g["global_dff"].to_numpy(), t)
    gy = np.where(np.isfinite(gy), gy, np.nanmedian(gy))
    rng = np.random.default_rng(0)
    t0 = time.time()

    # ---- main classification
    coh, ph, sig, inph = classify(M, y, fr, rng)
    grp = np.where(sig & inph, "in", np.where(sig & ~inph, "anti", "none"))
    D = pd.DataFrame(dict(run=key, source=source, fov=fov_group(r), mouse=r.mouse, dendrite=names,
                          coh=coh, phase_deg=np.degrees(ph), sig=sig, group=grp))
    # ---- properties
    units, shape, excl = load_units(r, source, names)
    cent = np.array([np.mean(np.unravel_index(u, shape), axis=1) for u in units]) * np.array(VOXEL_ZYX)
    D["depth_um"], D["x_um"], D["n_vox"] = cent[:, 1], cent[:, 2], [u.size for u in units]
    E = event_train(M, fr)
    on = (np.diff(E, axis=0, prepend=0) == 1)
    D["event_rate_per_min"] = on.sum(0) / (T / fr / 60)
    Z = rz(M)
    D["event_amp_z"] = [Z[E[:, j] > 0, j].mean() if (E[:, j] > 0).any() else np.nan for j in range(N)]
    for k, v in B.items():
        D[f"r_{k}"] = (zc(M) * zc(v[:, None])).sum(0)
    D["r_global"] = (zc(M) * zc(gy[:, None])).sum(0)
    D["r_pupil"] = (zc(M) * zc(y[:, None])).sum(0)
    nact = (Z > 3).sum(1)
    glob = nact >= max(2, 0.2 * N)
    D["global_participation"] = [(E[glob, j] > 0).mean() if glob.any() else np.nan for j in range(N)]
    # slow component (< 0.1 Hz) of each dendrite vs pupil: sign of r  (hemodynamic dip would be negative)
    lp = M - highpass(M, fr, 0.1)
    D["r_pupil_slow"] = (zc(lp) * zc(y[:, None])).sum(0)
    D["r_pupil_fast"] = (zc(highpass(M, fr, 0.1)) * zc(y[:, None])).sum(0)
    # fraction of time spent below baseline when pupil is large (dip test)
    big = y > np.percentile(y, 75)
    D["z_median_when_pupil_large"] = np.median(Z[big], 0)
    D["z_median_when_pupil_small"] = np.median(Z[y < np.percentile(y, 25)], 0)

    # ---- controls: alternative classifications
    alt = {}
    alt["highpass"] = classify(highpass(M, fr, 0.1), y, fr, rng, nsh=100)
    alt["events"] = classify(E + 1e-6 * rng.standard_normal(E.shape), y, fr, rng, nsh=100)
    if regs:
        alt["pupil_partial"] = classify(M, partial_out(y, regs), fr, rng, nsh=100)
    bad = dip_frames(y, fr)
    if bad.mean() < 0.5:
        keep = ~bad
        alt["no_dips"] = classify(M[keep], y[keep], fr, rng, nsh=100)
    h = T // 2
    alt["half1"] = classify(M[:h], y[:h], fr, rng, nsh=100)
    alt["half2"] = classify(M[h:], y[h:], fr, rng, nsh=100)
    for nm, (c_, p_, s_, i_) in alt.items():
        D[f"phase_deg_{nm}"] = np.degrees(p_)
        D[f"sig_{nm}"] = s_
        D[f"inphase_{nm}"] = i_
    if "accel" in B:   # still frames only
        still = B["accel"] < np.percentile(B["accel"], 50)
        alt["still"] = classify(M[still], y[still], fr, rng, nsh=100)
        D["phase_deg_still"], D["sig_still"], D["inphase_still"] = np.degrees(alt["still"][1]), alt["still"][2], alt["still"][3]
    R = dict(run=key, source=source, fov=fov_group(r), mouse=r.mouse, n=N, n_in=int((grp == "in").sum()),
             n_anti=int((grp == "anti").sum()), frac_sig=float(sig.mean()),
             frac_anti_of_sig=float((grp == "anti").sum() / max(1, sig.sum())),
             pupil_dip_frames_frac=float(bad.mean()))
    for nm in alt:
        both = sig & D[f"sig_{nm}"].to_numpy()
        R[f"agree_{nm}"] = float((inph[both] == D[f"inphase_{nm}"].to_numpy()[both]).mean()) if both.sum() >= 3 else np.nan
        R[f"n_both_{nm}"] = int(both.sum())
        # does the anti group stay anti?
        a_ = grp == "anti"
        R[f"anti_still_anti_{nm}"] = float((~D.loc[a_, f"inphase_{nm}"]).mean()) if a_.sum() >= 2 else np.nan
        R[f"anti_sig_{nm}"] = float(D.loc[a_, f"sig_{nm}"].mean()) if a_.sum() >= 2 else np.nan
    # group properties
    for gname in ("in", "anti", "none"):
        m = grp == gname
        if m.sum():
            R[f"{gname}_depth"] = float(D.depth_um[m].median())
            R[f"{gname}_rate"] = float(D.event_rate_per_min[m].median())
            R[f"{gname}_amp"] = float(D.event_amp_z[m].median())
            R[f"{gname}_nvox"] = float(D.n_vox[m].median())
            R[f"{gname}_r_global"] = float(D.r_global[m].median())
            R[f"{gname}_participation"] = float(D.global_participation[m].median())
            R[f"{gname}_r_pupil_slow"] = float(D.r_pupil_slow[m].median())
            R[f"{gname}_r_pupil_fast"] = float(D.r_pupil_fast[m].median())
            R[f"{gname}_zmed_pupil_large"] = float(D.z_median_when_pupil_large[m].median())
            for k in B:
                R[f"{gname}_r_{k}"] = float(D[f"r_{k}"][m].median())
    C = zc(M).T @ zc(M)
    iu = np.triu_indices(N, 1)
    gi, gj = grp[iu[0]], grp[iu[1]]
    for a_, b_ in (("in", "in"), ("anti", "anti"), ("in", "anti"), ("none", "none")):
        m = ((gi == a_) & (gj == b_)) | ((gi == b_) & (gj == a_))
        R[f"r_{a_}_{b_}"] = float(C[iu][m].mean()) if m.sum() >= 3 else np.nan
    # pupil-dilation-triggered Ca per group, and Ca-onset-triggered pupil per group
    dy = ndi.gaussian_filter1d(y, 2)
    dyd = np.diff(dy, prepend=dy[0])
    thr = np.percentile(dyd, 90)
    dil = np.flatnonzero((dyd[1:] > thr) & (dyd[:-1] <= thr)) + 1
    w = int(4 * fr)
    dil = dil[(dil >= w) & (dil < T - w)]
    trig = {}
    if dil.size >= 3:
        for gname in ("in", "anti"):
            m = grp == gname
            if m.sum():
                seg = np.stack([Z[i - w:i + w + 1][:, m].mean(1) - Z[i - w:i][:, m].mean() for i in dil])
                trig[f"dilation_{gname}"] = seg.mean(0)
        R["n_dilation_onsets"] = int(dil.size)
    yn = (y - np.median(y)) / (np.percentile(y, 95) - np.percentile(y, 5) + 1e-12)
    for gname in ("in", "anti"):
        m = np.flatnonzero(grp == gname)
        ons = [i for j in m for i in np.flatnonzero(on[:, j]) if w <= i < T - w]
        if len(ons) >= 5:
            seg = np.stack([yn[i - w:i + w + 1] - yn[i - w:i].mean() for i in ons])
            trig[f"ca_onset_{gname}"] = seg.mean(0)
            R[f"n_onsets_{gname}"] = len(ons)
            R[f"pupil_after_{gname}_onsets"] = float(seg[:, w + int(1 * fr):w + int(4 * fr)].mean())
    # ---- heavy: raw-stack controls (background by depth band, half-mask phase)
    if heavy:
        bgb = depth_band_background(r, shape, excl, units, fr)
        bg_dff = (bgb - np.percentile(bgb, 10, 0)) / np.percentile(bgb, 10, 0)
        cb, pb = coh_phase(bg_dff, y, fr)
        for i, nm in enumerate(("superficial", "middle", "deep")):
            R[f"bg_{nm}_coh"] = float(cb[i])
            R[f"bg_{nm}_phase_deg"] = float(np.degrees(pb[i]))
            R[f"bg_{nm}_r_pupil"] = float(np.corrcoef(bg_dff[:, i], y)[0, 1])
        sel = np.flatnonzero(sig)
        if sel.size:
            H, core, shl = half_mask_traces(r, [units[j] for j in sel], shape, excl)
            # over-subtraction test: phase of the un-subtracted core and of the shell alone
            cz_, cph = coh_phase(core, y, fr)
            sz_, sph = coh_phase(np.nan_to_num(shl), y, fr)
            D.loc[sel, "phase_core_only_deg"] = np.degrees(cph)
            D.loc[sel, "coh_core_only"] = cz_
            D.loc[sel, "phase_shell_deg"] = np.degrees(sph)
            D.loc[sel, "r_core_pupil"] = (zc(core) * zc(y[:, None])).sum(0)
            D.loc[sel, "r_shell_pupil"] = (zc(np.nan_to_num(shl)) * zc(y[:, None])).sum(0)
            for gname in ("in", "anti"):
                m = grp[sel] == gname
                if m.sum() >= 2:
                    R[f"core_inphase_{gname}"] = float((np.abs(cph[m]) < np.pi / 2).mean())
                    R[f"core_coh_{gname}"] = float(np.median(cz_[m]))
                    R[f"shell_inphase_{gname}"] = float((np.abs(sph[m]) < np.pi / 2).mean())
                    R[f"r_core_pupil_{gname}"] = float(np.median(D.loc[sel[m], "r_core_pupil"]))
                    R[f"r_shell_pupil_{gname}"] = float(np.median(D.loc[sel[m], "r_shell_pupil"]))
            okh = np.isfinite(H[0, :, 0])
            if okh.any():
                _, ph1 = coh_phase(np.nan_to_num(H[:, :, 0]), y, fr)
                _, ph2 = coh_phase(np.nan_to_num(H[:, :, 1]), y, fr)
                agree = (np.abs(ph1) < np.pi / 2) == (np.abs(ph2) < np.pi / 2)
                D.loc[sel, "halves_agree"] = agree
                D.loc[sel, "phase_top_deg"] = np.degrees(ph1)
                D.loc[sel, "phase_bottom_deg"] = np.degrees(ph2)
                for gname in ("in", "anti"):
                    m = grp[sel] == gname
                    if m.sum() >= 2:
                        R[f"halves_agree_{gname}"] = float(agree[m & okh].mean())
    R["time_s"] = round(time.time() - t0, 1)
    return dict(run=R, dend=D, trig={k: v.tolist() for k, v in trig.items()}, fr=fr)


# ----------------------------------------------------------------------------- cohort
def cohort(source, res, example):
    out = OUT / source
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([x["run"] for x in res])
    dend = pd.concat([x["dend"] for x in res], ignore_index=True)
    runs.to_csv(out / "per_run.csv", index=False)
    dend.to_csv(out / "per_dendrite.csv", index=False)
    fov = runs.groupby(["fov", "mouse"]).mean(numeric_only=True).reset_index()
    fov.to_csv(out / "per_fov.csv", index=False)
    S = dict(source=source, n_runs=len(runs), n_fovs=int(runs.fov.nunique()), n_mice=int(runs.mouse.nunique()), tests=[])
    L = [f"PUPIL-PHASE POPULATIONS ({source} dendrites): {len(runs)} runs, {runs.fov.nunique()} FOVs, {runs.mouse.nunique()} mice, "
         f"{len(dend)} dendrite-run entries. Coherence band {BAND[0]}-{BAND[1]} Hz, significance = above the 99th percentile of "
         f"{NSH} circular shifts of the pupil trace. In phase = |phase| < 90 deg.", ""]

    def wil(a, b):
        d_ = fov[[a, b]].dropna()
        if len(d_) < 4:
            return np.nan, len(d_)
        return float(sp.wilcoxon(d_[a], d_[b]).pvalue), len(d_)

    nin, nanti = dend.groupby("run").group.apply(lambda g: (g == "in").sum()), dend.groupby("run").group.apply(lambda g: (g == "anti").sum())
    L.append(f"1. Prevalence: {fov.frac_sig.median() * 100:.0f}% of dendrites per FOV are coherent with pupil; of these "
             f"{fov.frac_anti_of_sig.median() * 100:.0f}% are anti-phase (median over FOVs). Totals: {int(nin.sum())} in-phase, "
             f"{int(nanti.sum())} anti-phase dendrite-run entries. Anti-phase dendrites were found in {(runs.n_anti > 0).sum()}/{len(runs)} runs.")
    for prop, lab in (("depth", "depth (um)"), ("rate", "event rate (/min)"), ("amp", "event amplitude (robust z)"),
                      ("nvox", "size (voxels)"), ("r_accel", "r with locomotion"), ("r_whisker", "r with whisking"),
                      ("r_global", "r with global Ca"), ("participation", "global-event participation")):
        a, b = f"in_{prop}", f"anti_{prop}"
        if a in fov and b in fov:
            p, n = wil(a, b)
            S["tests"].append(dict(property=prop, in_median=float(fov[a].median()), anti_median=float(fov[b].median()), p=p, n_fovs=n))
            L.append(f"   {lab:32s}: in-phase {fov[a].median():8.3f}   anti-phase {fov[b].median():8.3f}   (Wilcoxon over {n} FOVs p={p:.2g})")
    L.append(f"2. Mutual coupling: r within in-phase {fov.r_in_in.median():.3f}, within anti-phase {fov.r_anti_anti.median():.3f}, "
             f"between the groups {fov.r_in_anti.median():+.3f}, among non-coherent {fov.r_none_none.median():.3f}.")
    # LMM
    try:
        import statsmodels.formula.api as smf
        d2 = dend[dend.sig].copy()
        d2["anti"] = (d2.group == "anti").astype(float)
        for c in ("depth_um", "event_rate_per_min", "r_accel", "n_vox"):
            if c in d2:
                d2[c + "_z"] = d2.groupby("fov")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        form = "anti ~ depth_um_z + event_rate_per_min_z" + (" + r_accel_z" if "r_accel_z" in d2 else "") + " + n_vox_z"
        m = smf.mixedlm(form, d2.dropna(subset=[c for c in d2.columns if c.endswith("_z")]), groups="fov").fit(reml=True)
        L.append("3. LMM P(anti-phase | coherent) ~ standardised depth + event rate + locomotion r + size (random intercept FOV): "
                 + ", ".join(f"{k} {m.params[k]:+.3f} (p={m.pvalues[k]:.2g})" for k in m.params.index if k.endswith("_z")) + ".")
        S["lmm"] = {k: dict(coef=float(m.params[k]), p=float(m.pvalues[k])) for k in m.params.index if k.endswith("_z")}
    except Exception as e:
        L.append(f"3. LMM failed: {e}")
    # controls
    L.append("4. Controls (fraction of anti-phase dendrites that remain anti-phase / remain significant when the analysis is changed; "
             "agreement = phase agreement of all dendrites significant in both versions):")
    for nm, desc in (("highpass", "trace high-passed > 0.1 Hz (slow dips removed)"), ("events", "binary event train only (positive transients)"),
                     ("pupil_partial", "pupil partialled for locomotion + whisking"), ("no_dips", "frames around pupil dips removed"),
                     ("still", "still frames only (accel below median)"), ("half1", "first half of the run"), ("half2", "second half of the run")):
        if f"anti_still_anti_{nm}" in fov:
            L.append(f"   {desc:52s}: anti stays anti {fov[f'anti_still_anti_{nm}'].median() * 100:5.0f}%   still significant "
                     f"{fov[f'anti_sig_{nm}'].median() * 100:4.0f}%   agreement {fov[f'agree_{nm}'].median() * 100:4.0f}% (50% = chance)")
            S["tests"].append(dict(control=nm, anti_stays_anti=float(fov[f"anti_still_anti_{nm}"].median()), anti_sig=float(fov[f"anti_sig_{nm}"].median()),
                                   agreement=float(fov[f"agree_{nm}"].median())))
    if "halves_agree_anti" in fov:
        L.append(f"   top vs bottom half of each mask (raw, own shells)     : phase agrees in {fov.halves_agree_anti.median() * 100:.0f}% of anti-phase "
                 f"and {fov.halves_agree_in.median() * 100:.0f}% of in-phase dendrites (50% = chance).")
    if "core_inphase_anti" in fov:
        L.append(f"4b. OVER-SUBTRACTION test (raw stack, whole mask without shell subtraction): of the anti-phase dendrites, "
                 f"{fov.core_inphase_anti.median() * 100:.0f}% have an IN-PHASE un-subtracted core (vs {fov.core_inphase_in.median() * 100:.0f}% of in-phase "
                 f"dendrites); core-only coherence {fov.core_coh_anti.median():.2f} (anti) / {fov.core_coh_in.median():.2f} (in). "
                 f"r(core, pupil) {fov.r_core_pupil_anti.median():+.2f} (anti) vs {fov.r_core_pupil_in.median():+.2f} (in); "
                 f"r(shell, pupil) {fov.r_shell_pupil_anti.median():+.2f} (anti) vs {fov.r_shell_pupil_in.median():+.2f} (in). "
                 f"If anti-phase were real, the core itself would be anti-phase; if it is over-subtraction, the core is in phase or flat and the shell is in phase.")
        S["oversubtraction"] = dict(core_inphase_anti=float(fov.core_inphase_anti.median()), core_inphase_in=float(fov.core_inphase_in.median()),
                                    r_core_pupil_anti=float(fov.r_core_pupil_anti.median()), r_shell_pupil_anti=float(fov.r_shell_pupil_anti.median()))
    if "bg_deep_phase_deg" in fov:
        L.append("5. Out-of-mask background vs pupil (hemodynamic test): " + "; ".join(
            f"{nm}: coherence {fov[f'bg_{nm}_coh'].median():.2f}, phase {fov[f'bg_{nm}_phase_deg'].median():+.0f} deg, r {fov[f'bg_{nm}_r_pupil'].median():+.2f}"
            for nm in ("superficial", "middle", "deep")) + ". A hemodynamic dip predicts phase near +-180 deg and negative r, strongest deep.")
        L.append(f"   Slow (<0.1 Hz) component of the dendrite trace vs pupil: r in-phase {fov.in_r_pupil_slow.median():+.2f}, anti-phase "
                 f"{fov.anti_r_pupil_slow.median():+.2f}; fast (>0.1 Hz) component: in-phase {fov.in_r_pupil_fast.median():+.2f}, anti-phase "
                 f"{fov.anti_r_pupil_fast.median():+.2f}. Median robust z when the pupil is large (top quartile): in-phase "
                 f"{fov.in_zmed_pupil_large.median():+.2f}, anti-phase {fov.anti_zmed_pupil_large.median():+.2f} (a dip below baseline would be < 0).")
    if "pupil_after_anti_onsets" in fov:
        p, n = wil("pupil_after_in_onsets", "pupil_after_anti_onsets")
        L.append(f"6. Pupil 1-4 s after Ca onsets (range-normalised): in-phase dendrites {fov.pupil_after_in_onsets.median():+.3f}, anti-phase "
                 f"{fov.pupil_after_anti_onsets.median():+.3f} (p={p:.2g}, {n} FOVs).")
    # identity
    J = []
    for f_, g in dend.groupby("fov"):
        ks = sorted(g.run.unique())
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a = g[g.run == ks[i]].set_index("dendrite")
                b = g[g.run == ks[j]].set_index("dendrite")
                c = a.index.intersection(b.index)
                both = c[(a.loc[c, "sig"] & b.loc[c, "sig"]).to_numpy(bool)]
                if len(both) < 3:
                    continue
                ga, gb = a.loc[both, "group"], b.loc[both, "group"]
                J.append(dict(fov=f_, run_a=ks[i], run_b=ks[j], n_both=len(both), agree=float((ga == gb).mean()),
                              in_in=int(((ga == "in") & (gb == "in")).sum()), anti_anti=int(((ga == "anti") & (gb == "anti")).sum()),
                              in_anti=int(((ga == "in") & (gb == "anti")).sum()), anti_in=int(((ga == "anti") & (gb == "in")).sum()),
                              rho_coh=float(sp.spearmanr(a.loc[c, "coh"], b.loc[c, "coh"])[0])))
    Jd = pd.DataFrame(J)
    if len(Jd):
        Jd.to_csv(out / "identity_across_runs.csv", index=False)
        tot = Jd[["in_in", "anti_anti", "in_anti", "anti_in"]].sum()
        # exact binomial on pooled agreement vs 0.5
        k_, n_ = int(tot.in_in + tot.anti_anti), int(tot.sum())
        pb = sp.binomtest(k_, n_, 0.5).pvalue if n_ else np.nan
        L.append(f"7. Identity across runs of the same FOV ({len(Jd)} run pairs): dendrites coherent in both runs keep their phase in "
                 f"{k_}/{n_} cases ({100 * k_ / max(n_, 1):.0f}%; binomial vs 50%: p={pb:.2g}); in->in {int(tot.in_in)}, anti->anti {int(tot.anti_anti)}, "
                 f"in->anti {int(tot.in_anti)}, anti->in {int(tot.anti_in)}. Coherence magnitude rank-correlation across runs: {Jd.rho_coh.median():.2f}.")
        S["identity"] = dict(n_pairs=len(Jd), agree=k_ / max(n_, 1), n=n_, p=pb, table=tot.to_dict())
    L += ["", "Interpretation guide: H1 hemodynamic dip -> background and slow component anti-phase, event train not; H2 blinks -> changes "
          "with dips removed; H3 locomotion -> changes with partialled pupil / still frames; H4 chance -> halves and runs disagree; "
          "H5 mask artifact -> top/bottom halves disagree, anti-phase units have no events."]
    (out / "summary.txt").write_text("\n".join(L))
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))
    talk_figure(source, res, runs, fov, dend, Jd, example, out)
    controls_figure(source, res, runs, fov, dend, out)
    return L


def _example(res, example):
    for x in res:
        if x["run"]["run"] == example:
            return x
    # most anti-phase dendrites with at least 8 in-phase
    return max(res, key=lambda x: min(x["run"]["n_anti"], 8) + min(x["run"]["n_in"], 8) + x["run"]["n_anti"] * 0.1)


def talk_figure(source, res, runs, fov, dend, Jd, example, out):
    ex = _example(res, example)
    key = ex["run"]["run"]
    r = get_run(key)
    fr = ex["fr"]
    D = ex["dend"]
    df = pd.read_csv(r.out / "traces" / ("dff_auto.csv" if source == "auto" else "dff_human_sameextractor.csv"))
    t = df["time_s"].to_numpy()
    beh = load_behavior(r)
    y = resample_to(beh["pupil_t"], beh["pupil"], t)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)
    # A traces
    ax = fig.add_subplot(gs[0, :3])
    yn = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-12)
    ax.plot(t, yn * 3 + 9.5, color="k", lw=1.2, label="pupil (normalised)")
    inn = D[D.group == "in"].sort_values("coh", ascending=False).dendrite.tolist()[:3]
    ann = D[D.group == "anti"].sort_values("coh", ascending=False).dendrite.tolist()[:3]
    off = 0
    for n_ in ann[::-1]:
        v = df[n_].to_numpy()
        v = (v - np.median(v)) / (np.percentile(v, 99) - np.median(v) + 1e-9)
        ax.plot(t, v * 1.6 + off, color=COL_ANTI, lw=0.8)
        ax.text(t[-1] + 1, off, n_, color=COL_ANTI, fontsize=7, va="center")
        off += 1.6
    for n_ in inn[::-1]:
        v = df[n_].to_numpy()
        v = (v - np.median(v)) / (np.percentile(v, 99) - np.median(v) + 1e-9)
        ax.plot(t, v * 1.6 + off, color=COL_IN, lw=0.8)
        ax.text(t[-1] + 1, off, n_, color=COL_IN, fontsize=7, va="center")
        off += 1.6
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"A  {key}: pupil (black) with the three strongest in-phase (red) and anti-phase (blue) dendrites", loc="left")
    ax.set_xlim(t[0], t[-1])
    # B phase histogram pooled
    ax = fig.add_subplot(gs[0, 3], projection="polar")
    ph = np.radians(dend.loc[dend.sig, "phase_deg"])
    ax.hist(ph, bins=24, color="#777")
    ax.set_theta_zero_location("E")
    ax.set_title("B  phase of all pupil-coherent dendrites\n(0 = in phase, 180 = anti-phase)", fontsize=8)
    # C map
    ax = fig.add_subplot(gs[1, :2])
    refp = r.out / "reference" / "ref_activity.tif"
    if refp.exists():
        bg = np.clip(tifffile.imread(str(refp)), 0, None).max(0)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=np.percentile(bg, 99.5), aspect=VOXEL_ZYX[1] / VOXEL_ZYX[2], extent=[0, bg.shape[1] * VOXEL_ZYX[2], bg.shape[0] * VOXEL_ZYX[1], 0])
    for gname, c, s_ in (("none", COL_NONE, 8), ("in", COL_IN, 28), ("anti", COL_ANTI, 28)):
        m = D.group == gname
        ax.scatter(D.x_um[m], D.depth_um[m], s=s_, color=c, edgecolors="k" if gname != "none" else "none", lw=0.4, alpha=0.9 if gname != "none" else 0.5)
    ax.set_xlabel("lateral position (um)")
    ax.set_ylabel("depth (um)")
    ax.set_title(f"C  where they are: in-phase (red, n={int((D.group == 'in').sum())}) vs anti-phase (blue, n={int((D.group == 'anti').sum())}); grey = not coherent", loc="left")
    # D depth distributions over all FOVs
    ax = fig.add_subplot(gs[1, 2])
    dd = dend[dend.sig].copy()
    dd["depth_rel"] = dd.groupby("run").depth_um.transform(lambda x: x - x.median())
    for gname, c in (("in", COL_IN), ("anti", COL_ANTI)):
        v = dd.loc[dd.group == gname, "depth_rel"]
        ax.hist(v, bins=np.arange(-100, 101, 10), histtype="step", color=c, lw=1.5, density=True, label=f"{gname}-phase (n={len(v)})")
    ax.set_xlabel("depth relative to the FOV median (um)")
    ax.set_ylabel("density")
    p_depth = next((x["p"] for x in json.loads((out / "summary.json").read_text())["tests"] if x.get("property") == "depth"), np.nan)
    ax.set_title(f"D  anti-phase dendrites lie deeper (p={p_depth:.2g}, FOV level)", loc="left")
    ax.legend(fontsize=7, frameon=False)
    # E properties per FOV
    ax = fig.add_subplot(gs[1, 3])
    props = [("rate", "events/min"), ("r_accel", "r locomotion"), ("r_global", "r global Ca"), ("participation", "global-event\nparticipation")]
    xs = np.arange(len(props))
    for i, (p_, lab) in enumerate(props):
        a, b = fov.get(f"in_{p_}"), fov.get(f"anti_{p_}")
        if a is None or b is None:
            continue
        sc = 1.0 if p_ != "rate" else 1 / max(a.max(), b.max(), 1e-9) * 1.0
        for f_ in range(len(fov)):
            ax.plot([i - 0.15, i + 0.15], [a.iloc[f_] * sc, b.iloc[f_] * sc], color="#999", lw=0.5)
        ax.scatter(np.full(len(a), i - 0.15), a * sc, color=COL_IN, s=14, zorder=3)
        ax.scatter(np.full(len(b), i + 0.15), b * sc, color=COL_ANTI, s=14, zorder=3)
    ax.set_xticks(xs, [p[1] for p in props], fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("E  per FOV (lines join the same FOV); rate scaled to max", loc="left")
    # F triggered averages
    ax = fig.add_subplot(gs[2, 0])
    for gname, c in (("in", COL_IN), ("anti", COL_ANTI)):
        segs = [np.asarray(x["trig"][f"dilation_{gname}"]) for x in res if f"dilation_{gname}" in x["trig"]]
        if segs:
            Lmin = min(len(s_) for s_ in segs)
            Mx = np.stack([s_[:Lmin] for s_ in segs])
            tt = np.arange(Lmin) / fr - 4
            ax.plot(tt, Mx.mean(0), color=c, lw=1.5, label=f"{gname}-phase ({len(segs)} runs)")
            ax.fill_between(tt, Mx.mean(0) - Mx.std(0) / np.sqrt(len(segs)), Mx.mean(0) + Mx.std(0) / np.sqrt(len(segs)), color=c, alpha=0.2, lw=0)
    ax.axvline(0, color="k", ls="--", lw=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time from pupil dilation onset (s)")
    ax.set_ylabel("dendrite Ca (robust z, baseline-subtracted)")
    ax.set_title("F  Ca around dilation onsets", loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[2, 1])
    for gname, c in (("in", COL_IN), ("anti", COL_ANTI)):
        segs = [np.asarray(x["trig"][f"ca_onset_{gname}"]) for x in res if f"ca_onset_{gname}" in x["trig"]]
        if segs:
            Lmin = min(len(s_) for s_ in segs)
            Mx = np.stack([s_[:Lmin] for s_ in segs])
            tt = np.arange(Lmin) / fr - 4
            ax.plot(tt, Mx.mean(0), color=c, lw=1.5, label=f"{gname}-phase ({len(segs)} runs)")
            ax.fill_between(tt, Mx.mean(0) - Mx.std(0) / np.sqrt(len(segs)), Mx.mean(0) + Mx.std(0) / np.sqrt(len(segs)), color=c, alpha=0.2, lw=0)
    ax.axvline(0, color="k", ls="--", lw=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time from dendrite Ca onset (s)")
    ax.set_ylabel("pupil (range-normalised, baseline-subtracted)")
    ax.set_title("G  pupil around Ca events of each group", loc="left")
    ax.legend(fontsize=7, frameon=False)
    # H identity
    ax = fig.add_subplot(gs[2, 2])
    if len(Jd):
        tot = Jd[["in_in", "in_anti", "anti_in", "anti_anti"]].sum().to_numpy().reshape(2, 2)
        ax.imshow(tot, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(tot[i, j]), ha="center", va="center", fontsize=12, color="k" if tot[i, j] < tot.max() * 0.6 else "w")
        ax.set_xticks([0, 1], ["in", "anti"])
        ax.set_yticks([0, 1], ["in", "anti"])
        ax.set_xlabel("phase in run B")
        ax.set_ylabel("phase in run A")
        agree = (tot[0, 0] + tot[1, 1]) / max(tot.sum(), 1)
        ax.set_title(f"H  same dendrite, another run of the same FOV:\nphase kept in {agree * 100:.0f}% (chance 50%)", loc="left")
    # I coupling matrix
    ax = fig.add_subplot(gs[2, 3])
    mat = np.array([[fov.r_in_in.median(), fov.r_in_anti.median()], [fov.r_in_anti.median(), fov.r_anti_anti.median()]])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1], ["in", "anti"])
    ax.set_yticks([0, 1], ["in", "anti"])
    ax.set_title(f"I  mean pairwise r within / between groups\n(non-coherent pairs: {fov.r_none_none.median():+.3f})", loc="left")
    fig.suptitle(f"Two pupil-coupled populations of L5 apical dendrites ({source} dendrites, {len(runs)} runs, {runs.fov.nunique()} FOVs, {runs.mouse.nunique()} mice)", fontsize=11)
    fig.savefig(out / "fig_pupil_phase.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "fig_pupil_phase.pdf", bbox_inches="tight")
    plt.close(fig)


def controls_figure(source, res, runs, fov, dend, out):
    fig, ax = plt.subplots(1, 5, figsize=(21, 4))
    a5 = ax[4]
    for gname, c in (("in", COL_IN), ("anti", COL_ANTI)):
        d_ = dend[dend.group == gname].dropna(subset=["phase_core_only_deg"]) if "phase_core_only_deg" in dend else pd.DataFrame()
        if len(d_):
            a5.scatter(d_.phase_deg, d_.phase_core_only_deg, s=6, color=c, alpha=0.6, label=f"{gname} (core in-phase {(np.abs(d_.phase_core_only_deg) < 90).mean() * 100:.0f}%)")
    a5.axhline(0, color="k", lw=0.3)
    a5.set_xlabel("phase of core - shell trace (deg)")
    a5.set_ylabel("phase of un-subtracted core (deg)")
    a5.set_title("over-subtraction test: does the raw core\nshow the same phase as the subtracted trace?", fontsize=8)
    a5.legend(fontsize=7, frameon=False)
    a = ax[0]
    names, vals_a, vals_s = [], [], []
    for nm, lab in (("highpass", "> 0.1 Hz only"), ("events", "event train"), ("pupil_partial", "pupil | accel,whisk"), ("no_dips", "dips removed"),
                    ("still", "still frames"), ("half1", "1st half"), ("half2", "2nd half")):
        if f"anti_still_anti_{nm}" in fov:
            names.append(lab)
            vals_a.append(fov[f"anti_still_anti_{nm}"].dropna().to_numpy() * 100)
            vals_s.append(fov[f"agree_{nm}"].dropna().to_numpy() * 100)
    if names:
        a.boxplot(vals_a, positions=np.arange(len(names)) - 0.18, widths=0.3, patch_artist=True, boxprops=dict(facecolor=COL_ANTI, alpha=0.5), showfliers=False)
        a.boxplot(vals_s, positions=np.arange(len(names)) + 0.18, widths=0.3, patch_artist=True, boxprops=dict(facecolor="#999", alpha=0.5), showfliers=False)
        a.set_xticks(range(len(names)), names, rotation=35, fontsize=7)
        a.axhline(50, color="k", ls="--", lw=0.6)
        a.set_ylabel("%")
        a.set_title("anti-phase stays anti-phase (blue) / all phases agree (grey)\nunder alternative analyses; 50% = chance", fontsize=8)
    a = ax[1]
    for i, nm in enumerate(("superficial", "middle", "deep")):
        if f"bg_{nm}_phase_deg" in fov:
            a.scatter(np.full(len(fov), i) + np.random.default_rng(i).normal(0, 0.05, len(fov)), fov[f"bg_{nm}_phase_deg"], s=16, color="#444")
    a.axhline(0, color=COL_IN, lw=0.8)
    a.axhline(180, color=COL_ANTI, lw=0.8)
    a.axhline(-180, color=COL_ANTI, lw=0.8)
    a.set_xticks([0, 1, 2], ["superficial", "middle", "deep"])
    a.set_ylabel("phase of out-of-mask background vs pupil (deg)")
    a.set_title("hemodynamic test: background by depth\n(anti-phase = +-180)", fontsize=8)
    a = ax[2]
    for gname, c in (("in", COL_IN), ("anti", COL_ANTI)):
        d_ = dend[(dend.group == gname)].dropna(subset=["phase_top_deg"]) if "phase_top_deg" in dend else pd.DataFrame()
        if len(d_):
            a.scatter(d_.phase_top_deg, d_.phase_bottom_deg, s=6, color=c, alpha=0.6, label=f"{gname} (agree {d_.halves_agree.mean() * 100:.0f}%)")
    a.axhline(0, color="k", lw=0.3)
    a.axvline(0, color="k", lw=0.3)
    a.set_xlabel("phase, top half of mask (deg)")
    a.set_ylabel("phase, bottom half (deg)")
    a.set_title("mask test: two halves of each dendrite, raw stack", fontsize=8)
    a.legend(fontsize=7, frameon=False)
    a = ax[3]
    for gname, c in (("in", COL_IN), ("anti", COL_ANTI)):
        d_ = dend[dend.group == gname]
        a.scatter(d_.r_pupil_slow, d_.r_pupil_fast, s=6, color=c, alpha=0.5, label=gname)
    a.axhline(0, color="k", lw=0.3)
    a.axvline(0, color="k", lw=0.3)
    a.set_xlabel("r(pupil, slow < 0.1 Hz component)")
    a.set_ylabel("r(pupil, fast > 0.1 Hz component)")
    a.set_title("is the anti-phase carried by a slow dip or by fast events?", fontsize=8)
    a.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Artifact controls for the pupil-phase populations ({source})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_controls.png", dpi=140)
    fig.savefig(out / "fig_controls.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("auto", "human", "both"), default="both")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--example", default="2026-04-16/rbp4_132_phpeb/run1")
    ap.add_argument("--light", action="store_true", help="skip raw-stack controls")
    a = ap.parse_args()
    keys = a.run or [r.key for r in discover_runs()]
    t0 = time.time()
    for s in (("auto", "human") if a.source == "both" else (a.source,)):
        res = []
        with ProcessPoolExecutor(max(1, min(a.jobs, 3))) as ex:
            futs = {ex.submit(analyze, k, s, not a.light): k for k in keys}
            for f in as_completed(futs):
                try:
                    x = f.result()
                    if x is not None:
                        res.append(x)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("ERROR", futs[f], e)
        res.sort(key=lambda x: x["run"]["run"])
        print("\n".join(cohort(s, res, a.example)), "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
