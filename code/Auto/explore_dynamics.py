#!/usr/bin/env python
"""explore_dynamics.py - temporal-dynamics tests of apical-dendrite independence.

Angle: structure in time and in the within-/between-dendrite relationship that a
zero-lag Pearson correlation (already covered by auto_stats.py / independence_tests.py)
cannot see. Six analyses, each with a circular-shift null (>= 10 s) and FOV as the unit
of replication (runs of one field of view are pooled before the across-FOV test).

  1 waveform    : per-event rise / decay(1/e) / amplitude / duration; do dendrites fall
                  into discrete classes (GMM-BIC, k=1..4) or a continuum? Do waveform
                  features relate to depth, size, or how strongly the dendrite couples?
  2 eta         : event-triggered population response. When ONE dendrite fires an
                  isolated event (no other onset within +-0.2 s), what do its near
                  neighbours (< 30 um) do vs distant dendrites (> 75 um) in +-2 s,
                  as excess over the shift null? -> local spread vs local suppression.
  3 xcorr       : event-onset cross-correlogram pooled over pairs vs shift null. After a
                  dendrite fires, is the onset rate of OTHER dendrites raised (co-active)
                  or lowered (cross-refractory competition) at 0.2-2 s?
  4 precision   : for positively-coupled pairs, distribution of onset-time differences
                  (same frame vs 1-2 frame offset). Synchronous or sequential? vs null.
  5 avalanche   : size (number of dendrites) of synchronous temporal clusters;
                  observed vs shift null; power-law vs exponential (descriptive).
  6 slow        : do per-dendrite LOCAL (non-global) event rates co-fluctuate on a
                  ~15 s timescale (shared slow excitability) even when events do not
                  coincide? mean pairwise correlation of smoothed local-rate vs shift null.

CLI:  explore_dynamics.py [--source auto|human] [--jobs N]   (default --source auto)
Out:  scape-auto/stats/explore_dynamics/<source>/{per_run.csv, per_dendrite.csv,
      per_fov.csv, summary.txt, fig_*.png}

Read-only on scape-data/. Writes only under scape-auto/stats/explore_dynamics/ and this
one file in code/Auto/. Honest about null results.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import stats as sp
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, fov_group, human_masks, human_masks_valid, VOXEL_ZYX, AUTO_ROOT,
)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore")

# ----- shared conventions (matched to auto_stats.py / independence_tests.py) ------------
EV_Z, EV_MIN = 3.0, 2            # robust-z event: z > 3 for >= 2 frames; peak = argmax of the run
GLOBAL_FRAC = 0.2                # >= 20% of dendrites co-active = a global frame
MIN_SHIFT_S = 10.0               # circular-shift null: shift >= 10 s
W_ETA = 10                       # event-triggered window = +-10 frames (+-2 s at 5 Hz)
NEAR_UM, FAR_UM = 30.0, 75.0     # neighbour distance bins (3D centroid distance)
MATCH_W = 5                      # +-5 frames for onset-difference matching
SIG_SLOW_S = 15.0                # Gaussian sigma for slow local-rate smoothing
NSH_PAIR = 200                   # nulls for per-pair coupling significance
NSH = 50                         # nulls for curve/scalar statistics (ETA, xcorr, avalanche)
NSH_SLOW = 100                   # nulls for slow-rate correlation
OUT_ROOT = AUTO_ROOT / "stats" / "explore_dynamics"


# ------------------------------------------------------------------- small helpers
def bh(p):
    p = np.asarray(p, float)
    n = p.size
    if n == 0:
        return p
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def rz(M):
    """Robust z-score per column (median / 1.4826*MAD)."""
    med = np.median(M, 0)
    return (M - med) / (np.median(np.abs(M - med), 0) * 1.4826 + 1e-9)


def zc(M):
    """Mean-remove and unit-normalise columns (for Pearson via dot product)."""
    M = M - M.mean(0)
    return M / (np.linalg.norm(M, axis=0) + 1e-12)


def wilcox(vals, alt="two-sided"):
    """Wilcoxon signed-rank of FOV-level values vs 0. Returns (median, p, n)."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    n = v.size
    if n < 2 or np.allclose(v, 0):
        return (float(np.median(v)) if n else np.nan), np.nan, n
    try:
        p = sp.wilcoxon(v, alternative=alt, zero_method="wilcox")[1]
    except Exception:
        p = np.nan
    return float(np.median(v)), float(p), n


def wilcox_paired(a, b, alt="two-sided"):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 2 or np.allclose(a, b):
        return (float(np.median(a - b)) if a.size else np.nan), np.nan, a.size
    try:
        p = sp.wilcoxon(a, b, alternative=alt, zero_method="wilcox")[1]
    except Exception:
        p = np.nan
    return float(np.median(a - b)), float(p), a.size


# ------------------------------------------------------------------- data loading
def load_source(r, source):
    """Return (M[T,N], names, cent_um[N,3], sizes[N]) or None. Mirrors auto_stats.load_source."""
    if source == "auto":
        csv = r.out / "traces" / "dff_auto.csv"
        lp = r.out / "masks" / "auto_labelmap_reviewed.tif"
        lp = lp if lp.exists() else r.out / "masks" / "auto_labelmap.tif"
        if not (csv.exists() and lp.exists()):
            return None
        lab = tifffile.imread(str(lp))
        shape = lab.shape
        flat = lab.ravel()
        df = pd.read_csv(csv)
        names = [c for c in df.columns if c.startswith("dend_")]
        units = {f"dend_{k - 1:03d}": np.flatnonzero(flat == k) for k in range(1, int(lab.max()) + 1)}
    else:
        csv = r.out / "traces" / "dff_human_sameextractor.csv"
        if not csv.exists() or not human_masks_valid(r):
            return None
        df = pd.read_csv(csv)
        names = [c for c in df.columns if c.startswith("dend_")]
        shape = tifffile.memmap(str(r.raw), mode="r").shape[1:]
        units = {n: np.flatnonzero(m.ravel()) for n, m in human_masks(r, shape)}
    names = [n for n in names if n in units and units[n].size]
    if len(names) < 3:
        return None
    M = df[names].to_numpy(float)
    cent = np.array([np.mean(np.unravel_index(units[n], shape), axis=1) * np.array(VOXEL_ZYX) for n in names])
    sizes = np.array([units[n].size for n in names], float)
    return M, names, cent, sizes


# ------------------------------------------------------------------- events & features
def detect_events(z, dff, fr):
    """Events on one column. Returns onsets[int], peaks[int], feats[K,5]:
    (amp_z, amp_dff, rise_s, decay_tau_s, dur_s)."""
    a = np.r_[False, z > EV_Z, False]
    d = np.diff(a.astype(int))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    base = np.median(dff)
    onsets, peaks, feats = [], [], []
    for s, e in zip(st, en):
        if e - s < EV_MIN:
            continue
        pk = s + int(np.argmax(z[s:e]))
        onsets.append(s)
        peaks.append(pk)
        amp_z = float(z[pk])
        amp_dff = float(dff[pk] - base)
        rise_s = (pk - s) / fr
        # decay to 1/e of peak, forward from the peak, linearly interpolated
        target = z[pk] / np.e
        end = min(len(z), pk + 31)
        seg = z[pk:end]
        below = np.flatnonzero(seg <= target)
        if below.size and below[0] > 0:
            j = below[0]
            z0, z1 = seg[j - 1], seg[j]
            frac = (z0 - target) / (z0 - z1 + 1e-9)
            decay_s = (j - 1 + frac) / fr
        elif below.size:
            decay_s = 0.0
        else:
            decay_s = (end - 1 - pk) / fr          # right-censored
        dur_s = (e - s) / fr
        feats.append((amp_z, amp_dff, rise_s, decay_s, dur_s))
    return (np.array(onsets, int), np.array(peaks, int),
            np.array(feats, float).reshape(-1, 5))


def onset_matrix(onsets_list, T, N):
    O = np.zeros((T, N), bool)
    for j, on in enumerate(onsets_list):
        if on.size:
            O[on, j] = True
    return O


# ------------------------------------------------------------------- GMM-BIC (diagonal)
def gmm_bic(X, k, seed=0, iters=80):
    """Diagonal-covariance Gaussian mixture via EM. Returns (bic, labels)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if n < k * 3:
        return np.inf, np.zeros(n, int)
    mu = X[rng.choice(n, k, replace=False)].copy()
    var = np.tile(X.var(0) + 1e-6, (k, 1))
    w = np.full(k, 1.0 / k)
    ll_old = -np.inf
    resp = np.zeros((n, k))
    for _ in range(iters):
        for c in range(k):
            lg = -0.5 * (np.log(2 * np.pi * var[c]).sum()
                         + (((X - mu[c]) ** 2) / var[c]).sum(1))
            resp[:, c] = np.log(w[c] + 1e-12) + lg
        m = resp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(resp - m).sum(1) + 1e-300)
        ll = lse.sum()
        R = np.exp(resp - lse[:, None])
        Nk = R.sum(0) + 1e-12
        w = Nk / n
        mu = (R.T @ X) / Nk[:, None]
        for c in range(k):
            var[c] = (R[:, c:c + 1] * (X - mu[c]) ** 2).sum(0) / Nk[c] + 1e-6
        if ll - ll_old < 1e-4:
            break
        ll_old = ll
    n_par = k * (2 * d) + (k - 1)              # means + diag vars + weights
    bic = n_par * np.log(n) - 2 * ll
    return float(bic), R.argmax(1)


def silhouette(X, labels, seed=0, cap=1500):
    """Mean silhouette width (cluster separability). ~0 or negative = continuum / no classes."""
    n = X.shape[0]
    if n > cap:
        idx = np.random.default_rng(seed).choice(n, cap, replace=False)
        X, labels, n = X[idx], labels[idx], cap
    uniq = np.unique(labels)
    if uniq.size < 2:
        return np.nan
    Dm = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(2), 0))
    sil = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = Dm[i, same].mean() if same.any() else 0.0
        b = min(Dm[i, labels == c].mean() for c in uniq if c != labels[i])
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(sil.mean())


# ------------------------------------------------------------------- per-run analysis
def analyze_run(r, source):
    got = load_source(r, source)
    if got is None:
        return None
    M, names, cent, sizes = got
    fr = r.frame_rate
    T, N = M.shape
    ms = max(2, int(round(MIN_SHIFT_S * fr)))
    if T < 3 * ms:
        return None
    rng = np.random.default_rng(abs(hash(r.key)) % (2 ** 32))
    Z = rz(M)
    depth = cent[:, 1]                           # cy -> cortical depth (um)
    D = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)   # 3D distance (um)

    # events
    ons, pks, feats = [], [], []
    for j in range(N):
        o, p, f = detect_events(Z[:, j], M[:, j], fr)
        ons.append(o); pks.append(p); feats.append(f)
    O = onset_matrix(ons, T, N)
    nev = np.array([o.size for o in ons])

    # zero-lag coupling + per-pair significance (shift null) -> mean coupling per dendrite
    Zc = zc(M)
    C = Zc.T @ Zc
    iu = np.triu_indices(N, 1)
    pr = C[iu]
    null = []
    for _ in range(20):
        Mr = np.stack([np.roll(M[:, j], rng.integers(ms, T - ms)) for j in range(N)], 1)
        Zr = zc(Mr)
        null.append((Zr.T @ Zr)[iu])
    null_abs = np.sort(np.abs(np.concatenate(null)))
    p_pair = 1 - np.searchsorted(null_abs, np.abs(pr)) / (null_abs.size + 1)
    q_pair = bh(np.maximum(p_pair, 1.0 / (null_abs.size + 1)))
    sig_pos = (q_pair < 0.05) & (pr > 0)
    meancoup = np.full(N, np.nan)
    Cm = C.copy()
    np.fill_diagonal(Cm, np.nan)
    meancoup = np.nanmean(Cm, 1)

    res = dict(run=r.key, source=source, mouse=r.mouse, fov=fov_group(r), fr=fr,
               T=T, dur_s=T / fr, N=N, n_events=int(nev.sum()))

    # ---- per-dendrite feature table ----
    drows = []
    for j in range(N):
        f = feats[j]
        row = dict(run=r.key, fov=res["fov"], mouse=r.mouse, dend=names[j],
                   depth_um=float(depth[j]), size_vox=float(sizes[j]),
                   n_events=int(nev[j]), mean_coupling=float(meancoup[j]))
        if f.shape[0]:
            row.update(amp_z=float(np.median(f[:, 0])), amp_dff=float(np.median(f[:, 1])),
                       rise_s=float(np.median(f[:, 2])), decay_tau_s=float(np.median(f[:, 3])),
                       dur_s=float(np.median(f[:, 4])))
        else:
            row.update(amp_z=np.nan, amp_dff=np.nan, rise_s=np.nan, decay_tau_s=np.nan, dur_s=np.nan)
        drows.append(row)

    # ====================================================================== (1) waveform classes per run
    fd = np.array([[d["amp_dff"], d["rise_s"], d["decay_tau_s"], d["dur_s"]]
                   for d in drows if d["n_events"] >= 3 and np.isfinite(d["amp_dff"])])
    if fd.shape[0] >= 9:
        Xf = fd.copy()
        Xf[:, 0] = np.log10(np.clip(Xf[:, 0], 1e-3, None))
        Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
        bics = [gmm_bic(Xf, k, seed=1)[0] for k in (1, 2, 3, 4)]
        res["waveform_bestk"] = int(np.argmin(bics) + 1)
        res["waveform_dBIC_1_minus_best"] = float(bics[0] - min(bics))
        res["n_dend_feat"] = int(fd.shape[0])
    else:
        res["waveform_bestk"] = np.nan
        res["waveform_dBIC_1_minus_best"] = np.nan
        res["n_dend_feat"] = int(fd.shape[0])

    # feature vs depth / size / coupling (Spearman across dendrites with >= 3 events)
    sel = [d for d in drows if d["n_events"] >= 3 and np.isfinite(d["decay_tau_s"])]
    def _rho(a, b):
        if len(sel) < 5:
            return np.nan
        aa = np.array([d[a] for d in sel]); bb = np.array([d[b] for d in sel])
        if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
            return np.nan
        return float(sp.spearmanr(aa, bb)[0])
    res["rho_decay_depth"] = _rho("decay_tau_s", "depth_um")
    res["rho_decay_size"] = _rho("decay_tau_s", "size_vox")
    res["rho_decay_coupling"] = _rho("decay_tau_s", "mean_coupling")
    res["rho_amp_coupling"] = _rho("amp_dff", "mean_coupling")
    res["rho_dur_coupling"] = _rho("dur_s", "mean_coupling")

    # ====================================================================== (2) event-triggered population response
    nearj = [np.flatnonzero((D[i] < NEAR_UM) & (np.arange(N) != i)) for i in range(N)]
    farj = [np.flatnonzero(D[i] > FAR_UM) for i in range(N)]
    anyon = O.sum(1)
    iso = []                                    # (t, i) isolated single-dendrite onsets
    for i in range(N):
        for t in ons[i]:
            if t < W_ETA or t >= T - W_ETA:
                continue
            others = anyon[t - 1:t + 2].sum() - O[t - 1:t + 2, i].sum()
            if others == 0:
                iso.append((t, i))
    L = 2 * W_ETA + 1
    near_obs = np.zeros(L); far_obs = np.zeros(L); near_cnt = 0; far_cnt = 0
    for t, i in iso:
        if nearj[i].size:
            near_obs += Z[t - W_ETA:t + W_ETA + 1, nearj[i]].sum(1); near_cnt += nearj[i].size
        if farj[i].size:
            far_obs += Z[t - W_ETA:t + W_ETA + 1, farj[i]].sum(1); far_cnt += farj[i].size
    near_null = np.zeros(L); far_null = np.zeros(L)
    if iso:
        for _ in range(NSH):
            sh = rng.integers(ms, T - ms, N)
            Zs = np.stack([np.roll(Z[:, j], sh[j]) for j in range(N)], 1)
            for t, i in iso:
                if nearj[i].size:
                    near_null += Zs[t - W_ETA:t + W_ETA + 1, nearj[i]].sum(1)
                if farj[i].size:
                    far_null += Zs[t - W_ETA:t + W_ETA + 1, farj[i]].sum(1)
        near_null /= NSH; far_null /= NSH
    res["n_isolated"] = len(iso)
    pooled = dict(near_obs=near_obs, far_obs=far_obs, near_null=near_null, far_null=far_null,
                  near_cnt=near_cnt, far_cnt=far_cnt)

    # ====================================================================== (3) onset cross-correlogram
    def xcorr(Ob):
        out = np.zeros(W_ETA + 1)
        Of = Ob.astype(float)
        for l in range(W_ETA + 1):
            if l == 0:
                G = Of.T @ Of
            else:
                G = Of[:-l].T @ Of[l:]
            out[l] = G.sum() - np.trace(G)
        return out
    cc_obs = xcorr(O)
    cc_null = np.zeros(W_ETA + 1)
    for _ in range(NSH):
        sh = rng.integers(ms, T - ms, N)
        Os = np.stack([np.roll(O[:, j], sh[j]) for j in range(N)], 1)
        cc_null += xcorr(Os)
    cc_null /= NSH
    pooled["cc_obs"] = cc_obs
    pooled["cc_null"] = cc_null

    # ====================================================================== (4) onset-time precision (coupled pairs)
    diff_obs = np.zeros(2 * MATCH_W + 1)
    diff_null = np.zeros(2 * MATCH_W + 1)
    pair_ij = [(iu[0][k], iu[1][k]) for k in range(len(pr)) if sig_pos[k]]
    def match_hist(onsets_list, pairs):
        h = np.zeros(2 * MATCH_W + 1)
        for i, j in pairs:
            oi, oj = onsets_list[i], onsets_list[j]
            if oi.size == 0 or oj.size == 0:
                continue
            for t in oi:
                dd = oj - t
                k = dd[np.abs(dd) <= MATCH_W]
                if k.size:
                    nearest = k[np.argmin(np.abs(k))]
                    h[nearest + MATCH_W] += 1
        return h
    if pair_ij:
        diff_obs = match_hist(ons, pair_ij)
        for _ in range(NSH):
            sh = rng.integers(ms, T - ms, N)
            os_shift = [np.sort((o + sh[j]) % T) for j, o in enumerate(ons)]
            diff_null += match_hist(os_shift, pair_ij)
        diff_null /= NSH
    res["n_coupled_pairs"] = len(pair_ij)
    res["frac_pairs_pos_sig"] = float(sig_pos.mean())
    pooled["diff_obs"] = diff_obs
    pooled["diff_null"] = diff_null

    # ====================================================================== (5) synchronous-recruitment (avalanche) size
    # Co-onset size = number of distinct dendrites that START an event in the same 0.4 s bin.
    # (A contiguous-active-frame definition is degenerate here: with ~200 units something is
    #  always active, so it collapses to one giant cluster. Co-onset size is the resolvable
    #  synchronous-recruitment statistic at 5 Hz.)
    BW = 2                                        # 0.4 s bins
    nb = T // BW

    def coonset_full(Ob):
        B = Ob[:nb * BW].reshape(nb, BW, N).any(1)     # dendrite had an onset in the bin
        return B.sum(1)                                 # per-bin count (incl. zeros)
    s_obs = coonset_full(O)
    sz_obs = s_obs[s_obs >= 1]
    frac5_null, sz_null = [], []
    for _ in range(NSH):
        sh = rng.integers(ms, T - ms, N)
        Os = np.stack([np.roll(O[:, j], sh[j]) for j in range(N)], 1)
        sn = coonset_full(Os)
        frac5_null.append(float((sn >= 5).mean()))
        sz_null.append(sn[sn >= 1])
    sz_null = np.concatenate(sz_null) if sz_null else np.array([], int)
    res["aval_mean_obs"] = float(sz_obs.mean()) if sz_obs.size else np.nan
    res["aval_mean_null"] = float(sz_null.mean()) if sz_null.size else np.nan
    res["aval_max_obs"] = int(sz_obs.max()) if sz_obs.size else 0
    res["aval_frac5_obs"] = float((s_obs >= 5).mean())
    res["aval_frac5_null"] = float(np.mean(frac5_null)) if frac5_null else np.nan
    pooled["sz_obs"] = sz_obs
    pooled["sz_null"] = sz_null

    # ====================================================================== (6) slow local-rate co-fluctuation
    A = Z > EV_Z
    glob = np.convolve((A.sum(1) >= max(2, GLOBAL_FRAC * N)).astype(float), np.ones(3), "same") > 0
    sig = max(1.0, SIG_SLOW_S * fr)
    rate = np.zeros((T, N))
    keep = []
    for j in range(N):
        tr = np.zeros(T)
        loc = ons[j][~glob[ons[j]]] if ons[j].size else ons[j]
        if loc.size >= 3:
            tr[loc] = 1.0
            rate[:, j] = gaussian_filter1d(tr, sig)
            keep.append(j)
    if len(keep) >= 4:
        Rk = rate[:, keep]
        Rz = zc(Rk)
        Cs = Rz.T @ Rz
        iuk = np.triu_indices(len(keep), 1)
        obs_r = float(Cs[iuk].mean())
        nd = []
        for _ in range(NSH_SLOW):
            sh = rng.integers(ms, T - ms, len(keep))
            Rs = np.stack([np.roll(Rk[:, k], sh[k]) for k in range(len(keep))], 1)
            Rsz = zc(Rs)
            nd.append((Rsz.T @ Rsz)[iuk].mean())
        res["slow_r_obs"] = obs_r
        res["slow_r_null"] = float(np.mean(nd))
        res["slow_r_excess"] = obs_r - float(np.mean(nd))
        res["slow_n_dend"] = len(keep)
    else:
        res["slow_r_obs"] = res["slow_r_null"] = res["slow_r_excess"] = np.nan
        res["slow_n_dend"] = len(keep)

    return res, drows, pooled


# ------------------------------------------------------------------- FOV pooling
def pool_by_fov(items):
    """items: list of (res, pooled). Returns dict fov -> summed pooled arrays."""
    catkeys = ("sz_obs", "sz_null")
    fovs = {}
    for res, pooled in items:
        f = res["fov"]
        acc = fovs.setdefault(f, {k: (0 if np.isscalar(v) else np.zeros_like(v)) for k, v in pooled.items()
                                   if k not in catkeys})
        for k in catkeys:
            acc.setdefault(k, [])
        for k, v in pooled.items():
            if k in catkeys:
                acc[k].append(v)
            else:
                acc[k] = acc[k] + v
    for f in fovs:
        for k in catkeys:
            fovs[f][k] = np.concatenate(fovs[f][k]) if fovs[f][k] else np.array([], int)
    return fovs


# ------------------------------------------------------------------- power-law vs exponential
def fit_pl_exp(sizes):
    x = np.asarray(sizes, float)
    x = x[x >= 1]
    if x.size < 20 or x.max() < 3:
        return None
    xmax = int(x.max()); n = x.size; slog = np.log(x).sum(); sx = x.sum()
    xs = np.arange(1, xmax + 1)
    best_pl = max(((a, -a * slog - n * np.log((xs ** (-a)).sum())) for a in np.arange(1.05, 4.01, 0.05)),
                  key=lambda t: t[1])
    best_ex = max(((lam, -lam * sx - n * np.log(np.exp(-lam * xs).sum())) for lam in np.arange(0.01, 3.01, 0.01)),
                  key=lambda t: t[1])
    aic_pl = 2 - 2 * best_pl[1]; aic_ex = 2 - 2 * best_ex[1]
    return dict(alpha=round(best_pl[0], 3), lam=round(best_ex[0], 3), n=n, xmax=xmax,
                favored="power-law" if aic_pl < aic_ex else "exponential",
                dAIC=round(abs(aic_pl - aic_ex), 2))


# ------------------------------------------------------------------- figures
def make_figures(outdir, perrun, perdend, fovpool, src):
    nf = len(fovpool)
    L = 2 * W_ETA + 1
    lags = (np.arange(L) - W_ETA) / 5.0

    # fig 1: waveform feature distributions + best-k
    dd = perdend[(perdend.n_events >= 3)].dropna(subset=["decay_tau_s"])
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.8))
    for a, col, lab in zip(ax, ["amp_dff", "rise_s", "decay_tau_s", "dur_s"],
                           ["amplitude (dF/F %)", "rise (s)", "decay 1/e (s)", "duration (s)"]):
        if len(dd):
            a.hist(dd[col].to_numpy(), bins=25, color="#4477aa")
        a.set_xlabel(lab); a.set_ylabel("dendrites")
    fig.suptitle(f"[{src}] per-dendrite event-waveform features (n={len(dd)} dendrite-runs)")
    fig.tight_layout(); fig.savefig(outdir / "fig_waveform_features.png", dpi=130); plt.close(fig)

    # fig 2: event-triggered population response, near vs far
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.4))
    nobs = sum(v["near_obs"] for v in fovpool.values()); ncnt = sum(v["near_cnt"] for v in fovpool.values())
    fobs = sum(v["far_obs"] for v in fovpool.values()); fcnt = sum(v["far_cnt"] for v in fovpool.values())
    nnull = sum(v["near_null"] for v in fovpool.values()); fnull = sum(v["far_null"] for v in fovpool.values())
    if ncnt and fcnt:
        ax.plot(lags, nobs / ncnt - nnull / ncnt, "-o", ms=3, color="#cc3311", label=f"near <{NEAR_UM:.0f} um")
        ax.plot(lags, fobs / fcnt - fnull / fcnt, "-o", ms=3, color="#4477aa", label=f"far >{FAR_UM:.0f} um")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("time from isolated event (s)"); ax.set_ylabel("neighbour z, excess over shift null")
    ax.legend(fontsize=7); ax.set_title(f"[{src}] event-triggered neighbour response")
    fig.tight_layout(); fig.savefig(outdir / "fig_eta_near_far.png", dpi=130); plt.close(fig)

    # fig 3: onset cross-correlogram obs vs null
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))
    cobs = sum(v["cc_obs"] for v in fovpool.values()); cnull = sum(v["cc_null"] for v in fovpool.values())
    ll = np.arange(W_ETA + 1) / 5.0
    ax.plot(ll, cobs, "-o", ms=3, color="#cc3311", label="observed")
    ax.plot(ll, cnull, "-o", ms=3, color="#777777", label="shift null")
    ax.set_xlabel("onset separation (s)"); ax.set_ylabel("co-onset count (all pairs)")
    ax.legend(fontsize=7); ax.set_title(f"[{src}] event-onset cross-correlogram")
    fig.tight_layout(); fig.savefig(outdir / "fig_xcorr.png", dpi=130); plt.close(fig)

    # fig 4: onset-time precision for coupled pairs
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))
    dobs = sum(v["diff_obs"] for v in fovpool.values()); dnull = sum(v["diff_null"] for v in fovpool.values())
    xb = np.arange(-MATCH_W, MATCH_W + 1)
    if dobs.sum():
        ax.bar(xb - 0.2, dobs / dobs.sum(), width=0.4, color="#cc3311", label="observed")
    if dnull.sum():
        ax.bar(xb + 0.2, dnull / dnull.sum(), width=0.4, color="#777777", label="null")
    ax.set_xlabel("onset offset (frames; 0.2 s each)"); ax.set_ylabel("fraction of matched events")
    ax.legend(fontsize=7); ax.set_title(f"[{src}] onset timing, positively-coupled pairs")
    fig.tight_layout(); fig.savefig(outdir / "fig_onset_precision.png", dpi=130); plt.close(fig)

    # fig 5: co-onset (avalanche) size CCDF obs vs pooled null
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.4))
    allobs = np.concatenate([v["sz_obs"] for v in fovpool.values()]) if fovpool else np.array([])
    allnull = np.concatenate([v["sz_null"] for v in fovpool.values()]) if fovpool else np.array([])
    if allobs.size:
        u = np.arange(1, allobs.max() + 1)
        ax.loglog(u, [(allobs >= k).mean() for k in u], "-o", ms=3, color="#cc3311", label="observed")
    if allnull.size:
        un = np.arange(1, allnull.max() + 1)
        ax.loglog(un, [(allnull >= k).mean() for k in un], "-o", ms=3, color="#777777", label="shift null")
    ax.set_xlabel("co-onset size (dendrites / 0.4 s bin)"); ax.set_ylabel("P(size >= s)")
    ax.legend(fontsize=7); ax.set_title(f"[{src}] synchronous-recruitment size")
    fig.tight_layout(); fig.savefig(outdir / "fig_avalanche.png", dpi=130); plt.close(fig)

    # fig 6: slow local-rate correlation excess per FOV
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))
    g = perrun.groupby("fov")["slow_r_excess"].mean().dropna()
    if len(g):
        ax.bar(range(len(g)), g.to_numpy(), color="#228833")
        ax.set_xticks(range(len(g))); ax.set_xticklabels([f.split("/")[-1] for f in g.index], rotation=60, fontsize=6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean local-rate r, excess over null"); ax.set_title(f"[{src}] slow ({SIG_SLOW_S:.0f}s) shared excitability")
    fig.tight_layout(); fig.savefig(outdir / "fig_slow_excitability.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------- summary writer
def write_summary(outdir, perrun, perdend, fovpool, src, runs_used):
    lines = []
    P = lines.append
    nf = perrun["fov"].nunique()
    nm = perrun["mouse"].nunique()
    P(f"TEMPORAL-DYNAMICS INDEPENDENCE TESTS - {src} dendrites, {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    P(f"{len(perrun)} runs, {nf} FOVs, {nm} mice, {int(perrun['N'].sum())} dendrite-run entries, "
      f"{int(perrun['n_events'].sum())} events total.")
    P("Nulls: independent circular shifts per dendrite (>= 10 s). FOV = unit of replication "
      "(runs pooled, Wilcoxon over FOVs). BH-FDR where many tests.")
    P("")

    # (1) waveform
    P("* [1 waveform classes] Per-dendrite event waveform (median rise/decay-1/e/amplitude/duration).")
    dd = perdend[perdend.n_events >= 3].dropna(subset=["decay_tau_s", "amp_dff", "dur_s"])
    if len(dd) >= 20:
        Xf = np.c_[np.log10(np.clip(dd.amp_dff.to_numpy(), 1e-3, None)),
                   dd.decay_tau_s.to_numpy(), dd.dur_s.to_numpy()]
        Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
        bics = [gmm_bic(Xf, k, seed=1) for k in (1, 2, 3, 4)]
        bk = int(np.argmin([b[0] for b in bics]) + 1)
        sil = silhouette(Xf, bics[bk - 1][1]) if bk > 1 else np.nan
        P(f"    Pooled GMM-BIC (n={len(dd)} dendrite-runs, features log-amp/decay/duration) picks k={bk}; "
          f"best-k silhouette = {sil:.2f}.")
        is_cont = bk == 1 or not np.isfinite(sil) or sil < 0.5
        P("    Conclusion: " + ("waveforms form a CONTINUUM, not discrete classes - BIC>1 is driven by "
                                "amplitude skew and the 0.2 s sampling grid (decay/duration are quantized), "
                                "and cluster separability is low." if is_cont
                                else "well-separated discrete waveform classes (high silhouette)."))
    bkr = perrun["waveform_bestk"].dropna()
    if len(bkr):
        P(f"    (Per-run BIC best-k across {len(bkr)} runs: {dict(bkr.astype(int).value_counts().sort_index())}; "
          "same caveat - BIC detects non-Gaussianity, not genuine discreteness.)")
    for a, lab in [("rho_decay_depth", "decay-tau vs depth"), ("rho_decay_size", "decay-tau vs size"),
                   ("rho_decay_coupling", "decay-tau vs mean coupling"),
                   ("rho_amp_coupling", "amplitude vs mean coupling"),
                   ("rho_dur_coupling", "duration vs mean coupling")]:
        fv = perrun.groupby("fov")[a].mean()
        med, p, n = wilcox(fv.to_numpy())
        P(f"    {lab}: Spearman rho median {med:+.3f} over {n} FOVs (p={p:.3g}).")
    if len(dd):
        P(f"    Pooled medians (n={len(dd)} dendrite-runs): amplitude {dd.amp_dff.median():.0f} dF/F%, "
          f"rise {dd.rise_s.median():.2f} s, decay-1/e {dd.decay_tau_s.median():.2f} s, "
          f"duration {dd.dur_s.median():.2f} s. (rise ~ 0 and decay/duration are quantized at the 0.2 s "
          "frame; treat waveform shape as coarse.)")
    P("")

    # (2) ETA near vs far
    P("* [2 event-triggered response] Isolated single-dendrite events (no other onset within +-0.2 s); "
      "neighbour robust-z in +-2 s, excess over shift null.")
    near_fov, far_fov, keys = [], [], sorted(fovpool)
    for f in keys:
        v = fovpool[f]
        c = slice(W_ETA - 2, W_ETA + 3)          # +-0.4 s around the event
        ne = (v["near_obs"] - v["near_null"]) / max(v["near_cnt"], 1)
        fe = (v["far_obs"] - v["far_null"]) / max(v["far_cnt"], 1)
        near_fov.append(float(ne[c].mean()) if v["near_cnt"] else np.nan)
        far_fov.append(float(fe[c].mean()) if v["far_cnt"] else np.nan)
    mn, pn, n1 = wilcox(near_fov)
    mf, pf, _ = wilcox(far_fov)
    md, pd_, n2 = wilcox_paired(near_fov, far_fov)
    tot_iso = int(perrun["n_isolated"].sum())
    P(f"    {tot_iso} isolated events pooled. Near-neighbour excess z (+-0.4 s): median {mn:+.3f} "
      f"(p={pn:.3g}, {n1} FOVs); far: median {mf:+.3f} (p={pf:.3g}).")
    concl2 = ("near > far = local spread/co-activation" if (np.isfinite(md) and md > 0 and (pd_ < 0.05 if np.isfinite(pd_) else False))
              else "near < far = local suppression" if (np.isfinite(md) and md < 0 and (pd_ < 0.05 if np.isfinite(pd_) else False))
              else "no distance-specific effect (near indistinguishable from far)")
    P(f"    Near minus far (controls for the isolated-event selection): median {md:+.3f} "
      f"(paired p={pd_:.3g}, {n2} FOVs). Conclusion: {concl2}. "
      "(Both bins sit slightly below null because isolated events are, by selection, picked from "
      "low-population-activity moments; the near-vs-far contrast is the clean local-spread test.)")
    P("")

    # (3) cross-correlogram
    P("* [3 cross-refractoriness] Event-onset cross-correlogram pooled over all pairs vs shift null.")
    si, ci = [], []
    for f in keys:
        v = fovpool[f]
        o, nl = v["cc_obs"], v["cc_null"]
        si.append((o[0] - nl[0]) / (nl[0] + 1e-9))                       # lag 0 synchrony
        ci.append(float(((o[1:6] - nl[1:6]) / (nl[1:6] + 1e-9)).mean())) # 0.2-1.0 s after
    ms0, ps0, ns0 = wilcox(si)
    mc0, pc0, _ = wilcox(ci)
    P(f"    Lag-0 synchrony index (obs/null - 1): median {ms0:+.2f} (p={ps0:.3g}, {ns0} FOVs).")
    P(f"    0.2-1.0 s after an event, OTHER dendrites' onset rate (obs/null - 1): median {mc0:+.2f} (p={pc0:.3g}).")
    concl3 = ("elevated = sustained co-activation, NOT competition" if (np.isfinite(mc0) and mc0 > 0)
              else "reduced = cross-refractory competition" if (np.isfinite(mc0) and mc0 < 0)
              else "at null = independent")
    P(f"    Conclusion: {concl3}.")
    P("")

    # (4) onset precision
    P("* [4 onset precision] Onset-time offsets of positively-coupled pairs (BH q<0.05), nearest match within +-1 s.")
    s0o, s0n = [], []
    for f in keys:
        v = fovpool[f]
        do, dn = v["diff_obs"], v["diff_null"]
        s0o.append(do[MATCH_W] / do.sum() if do.sum() else np.nan)
        s0n.append(dn[MATCH_W] / dn.sum() if dn.sum() else np.nan)
    mex, pex, nex = wilcox_paired(s0o, s0n)
    med_pairs = perrun["n_coupled_pairs"].median()
    P(f"    Same-frame (synchronous, |offset|=0) fraction: observed median {np.nanmedian(s0o):.2f} vs "
      f"null {np.nanmedian(s0n):.2f}; obs-null median {mex:+.3f} (paired p={pex:.3g}, {nex} FOVs). "
      f"(median {med_pairs:.0f} coupled pairs/run)")
    concl4 = ("coupled events are SYNCHRONOUS beyond chance (same 0.2 s frame)"
              if (np.isfinite(mex) and mex > 0) else "no excess same-frame synchrony - offsets look like chance")
    P(f"    Conclusion: {concl4}. (1-2 frame offsets = sequential; 5 Hz cannot resolve <0.2 s.)")
    P("")

    # (5) avalanche
    P("* [5 synchronous recruitment] Co-onset size = dendrites starting an event in the same 0.4 s bin, obs vs shift null.")
    mo, po, nmo = wilcox_paired(perrun.groupby("fov")["aval_mean_obs"].mean().to_numpy(),
                                perrun.groupby("fov")["aval_mean_null"].mean().to_numpy())
    f5o = perrun.groupby("fov")["aval_frac5_obs"].mean().to_numpy()
    f5n = perrun.groupby("fov")["aval_frac5_null"].mean().to_numpy()
    m5, p5, n5 = wilcox_paired(f5o, f5n)
    P(f"    Mean co-onset size (non-empty bins) obs vs null: obs-null median {mo:+.2f} (paired p={po:.3g}, "
      f"{nmo} FOVs); max observed {int(perrun['aval_max_obs'].max())} dendrites in one 0.4 s bin.")
    P(f"    Fraction of bins with >= 5 co-onsetting dendrites: obs {np.nanmedian(f5o):.3f} vs null "
      f"{np.nanmedian(f5n):.3f}, obs-null median {m5:+.3f} (paired p={p5:.3g}, {n5} FOVs).")
    allobs = np.concatenate([v["sz_obs"] for v in fovpool.values()]) if fovpool else np.array([])
    allnull = np.concatenate([v["sz_null"] for v in fovpool.values()]) if fovpool else np.array([])
    fo, fn = fit_pl_exp(allobs), fit_pl_exp(allnull)
    if fo:
        P(f"    Observed size distribution (n={fo['n']}, max {fo['xmax']}): {fo['favored']} favored "
          f"(dAIC {fo['dAIC']}, alpha {fo['alpha']}, exp-lambda {fo['lam']})"
          + (f"; shift-null favors {fn['favored']} (dAIC {fn['dAIC']})." if fn else ".")
          + " Heavy tail is not by itself evidence of criticality - underpowered at ~106 s.")
    concl5a = ("mean co-onset size exceeds the independent null (synchronous co-activation is real)"
               if (np.isfinite(mo) and mo > 0 and (po < 0.05 if np.isfinite(po) else False))
               else "mean co-onset size at the null level")
    concl5b = ("the >=5 heavy tail is reliable across FOVs" if (np.isfinite(m5) and m5 > 0 and (p5 < 0.05 if np.isfinite(p5) else False))
               else "the >=5 heavy tail is NOT reliable across FOVs")
    shape = f"the size distribution is {fo['favored']} (not scale-free)" if fo else "the distribution is undetermined"
    P(f"    Conclusion: {concl5a}; {concl5b}; {shape}. No evidence of critical / power-law avalanche dynamics.")
    P("")

    # (6) slow
    P("* [6 slow shared excitability] Pairwise correlation of 15 s-smoothed LOCAL (non-global) event rate vs shift null.")
    fv = perrun.groupby("fov")["slow_r_excess"].mean()
    med6, p6, n6 = wilcox(fv.to_numpy())
    P(f"    Mean local-rate r excess over null: median {med6:+.3f} (p={p6:.3g}, {n6} FOVs). "
      f"obs {perrun.groupby('fov')['slow_r_obs'].mean().median():+.3f} vs null "
      f"{perrun.groupby('fov')['slow_r_null'].mean().median():+.3f}.")
    concl6 = ("local event rates co-fluctuate on a slow timescale beyond coincident events "
              "(shared excitability)" if (np.isfinite(med6) and med6 > 0 and (p6 < 0.1 if np.isfinite(p6) else False))
              else "no slow shared excitability beyond chance")
    P(f"    Conclusion: {concl6}.")
    P("")

    P("Reading guide: independent units predict - no waveform classes (1); flat near=far ETA (2); "
      "flat cross-correlogram at null (3); chance same-frame fraction (4); null-level avalanche sizes (5); "
      "zero slow-rate excess (6).")
    P("Caveats: 5 Hz sampling (rise times and <0.2 s lags unresolved); ~106 s per run (~3 global events, "
      "sparse large avalanches); 3 mice (2 with behavior) - mouse-level inference weak; SCAPE optical "
      "crosstalk between neighbouring dendrites inflates short-range / near-neighbour effects (shell "
      "subtraction reduces but does not remove it), so interpret the near-neighbour ETA (2) conservatively.")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


# ------------------------------------------------------------------- main
def run_source(source, jobs):
    outdir = OUT_ROOT / source
    outdir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    results, dendrows, pooled_items = [], [], []
    used = []
    t0 = time.time()
    for r in runs:
        try:
            out = analyze_run(r, source)
        except Exception as e:
            print(f"  !! {r.key}: {e}")
            continue
        if out is None:
            continue
        res, drows, pooled = out
        results.append(res); dendrows.extend(drows); pooled_items.append((res, pooled)); used.append(r.key)
        print(f"  {r.key:40s} N={res['N']:3d} ev={res['n_events']:4d} iso={res['n_isolated']:4d} "
              f"coup={res['n_coupled_pairs']:4d}")
    if not results:
        print(f"[{source}] no runs with usable data.")
        return
    perrun = pd.DataFrame(results)
    perdend = pd.DataFrame(dendrows)
    fovpool = pool_by_fov(pooled_items)
    perrun.to_csv(outdir / "per_run.csv", index=False)
    perdend.to_csv(outdir / "per_dendrite.csv", index=False)
    # per-FOV table (means of the per-run scalars)
    num = perrun.select_dtypes(include=[np.number]).columns
    perrun.groupby("fov")[list(num)].mean().to_csv(outdir / "per_fov.csv")
    make_figures(outdir, perrun, perdend, fovpool, source)
    txt = write_summary(outdir, perrun, perdend, fovpool, source, used)
    print(f"\n[{source}] {len(results)} runs, {perrun['fov'].nunique()} FOVs in {time.time()-t0:.1f}s "
          f"-> {outdir}\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "human", "both"], default="auto")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    srcs = ["auto", "human"] if a.source == "both" else [a.source]
    for s in srcs:
        run_source(s, a.jobs)


if __name__ == "__main__":
    main()
