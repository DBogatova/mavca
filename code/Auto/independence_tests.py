#!/usr/bin/env python
"""independence_tests.py - battery of tests for 'are L5 apical dendrites independent units?'
(code/Auto/). Looks for dependencies (and independencies) beyond zero-lag Pearson r.

Every test is run against a null that keeps each dendrite's own dynamics and only breaks the
alignment between dendrites (independent circular shifts >= 10 s per dendrite; 20-200 shifts).
Runs sharing a field of view are averaged before cohort tests (FOV = unit of replication);
per-dendrite models are mixed models with a random intercept per FOV.

Tests (per run, source auto or human):
 A  event synchrony      : population co-activation (robust z > 3) - fraction of frames with >= k
                           active vs null; excess of large synchronous events (higher-order).
 B  pairwise MI          : mutual information of binary activity, vs null, BH-FDR; compares the
                           set of MI-dependent pairs with the Pearson-significant pairs
                           (dependence that Pearson misses = nonlinear / event-level).
 C  lagged structure     : cross-correlation at lags up to +-3 s; pairs whose peak is at a nonzero
                           lag (sequences); lag vs distance (propagation speed); direction along
                           depth (deep leads superficial = bottom-up, or top-down).
 D  spatial              : r vs 3D distance decay length; lateral vs depth separation (same
                           column = same trunk candidate); negative pairs beyond null.
 E  latent factors       : PCA variance of PC1 vs null; # of significant PCs; dependence left
                           after regressing out global Ca + behavior (residual pairwise r, PCs);
                           functional clusters (modularity vs null) and their spatial compactness.
 F  state gating         : pairwise r and synchrony in high vs low pupil, moving vs still.
 G  self dynamics        : inter-event-interval CV and Fano factor vs Poisson; amplitude vs
                           preceding interval (facilitation / depression).
 H  common input         : similarity of behavior tuning of a pair vs its r (signal vs noise
                           correlation logic).
 I  global events        : per-dendrite participation probability, its relation to depth/size;
                           leader consistency (Kendall W of latency ranks) vs shuffle.
 J  identity across runs : same-FOV runs: are event rate, accel coupling and pupil phase of a
                           dendrite stable (Spearman across dendrites)?
 K  dendrite coherence   : mean dendrite-dendrite coherence by frequency band vs null.
 L  pupil phase groups   : in-phase vs anti-phase dendrites: depth, mutual coupling, spatial mixing.

Outputs: scape-auto/stats/independence/<source>/{per_run.csv, per_dendrite.csv, per_fov.csv,
         summary.txt, summary.json, fig_*.png/pdf}
CLI: independence_tests.py [--source auto|human|both] [--run KEY ...] [--jobs N]
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
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, fov_group, load_behavior, resample_to, human_masks, human_masks_valid,
    VOXEL_ZYX, AUTO_ROOT,
)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore")
OUT = AUTO_ROOT / "stats" / "independence"
ZTHR = 3.0
NSH = 20          # null shifts for matrix-valued statistics
NSH_P = 200       # null shifts for per-pair p values
MAXLAG_S = 3.0


# ----------------------------------------------------------------------------- helpers
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
    med = np.median(M, 0)
    return (M - med) / (np.median(np.abs(M - med), 0) * 1.4826 + 1e-9)


def zc(M):
    M = M - M.mean(0)
    return M / (np.linalg.norm(M, axis=0) + 1e-12)


def shift_all(M, rng, ms):
    T = M.shape[0]
    return np.stack([np.roll(M[:, j], rng.integers(ms, T - ms)) for j in range(M.shape[1])], 1)


def onsets_from_active(A):
    """A (T,N) bool -> list of onset index arrays, runs >= 2 frames."""
    out = []
    for j in range(A.shape[1]):
        a = np.r_[False, A[:, j], False]
        d = np.diff(a.astype(int))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out.append(st[(en - st) >= 2])
    return out


def binary_mi(A):
    """Pairwise MI (bits) between binary columns of A (T,N)."""
    A = A.astype(float)
    T, N = A.shape
    p1 = A.mean(0)
    p11 = (A.T @ A) / T
    p10 = p1[:, None] - p11
    p01 = p1[None, :] - p11
    p00 = 1 - p1[:, None] - p1[None, :] + p11
    mi = np.zeros((N, N))
    for pij, pi, pj in ((p11, p1[:, None], p1[None, :]), (p10, p1[:, None], 1 - p1[None, :]),
                        (p01, 1 - p1[:, None], p1[None, :]), (p00, 1 - p1[:, None], 1 - p1[None, :])):
        with np.errstate(divide="ignore", invalid="ignore"):
            term = pij * np.log2(pij / (pi * pj))
        mi += np.where(pij > 0, term, 0)
    return mi


def lagged_corr(Z, maxlag):
    """Z (T,N) z-scored -> C (L, N, N) with C[l] = corr(x_i(t), x_j(t+lag_l)); lags -maxlag..maxlag."""
    T, N = Z.shape
    lags = np.arange(-maxlag, maxlag + 1)
    C = np.zeros((lags.size, N, N))
    for k, L in enumerate(lags):
        if L >= 0:
            a, b = Z[:T - L], Z[L:]
        else:
            a, b = Z[-L:], Z[:T + L]
        C[k] = zc(a).T @ zc(b)
    return lags, C


def coherence_matrix(M, fr, band):
    nper = int(round(25.6 * fr))
    N = M.shape[1]
    f, _ = signal.coherence(M[:, 0], M[:, 0], fs=fr, nperseg=nper, noverlap=nper // 2)
    k = (f >= band[0]) & (f <= band[1])
    out = np.zeros((N, N))
    for i in range(N):
        _, C = signal.coherence(M[:, i][None, :], M[:, i + 1:].T, fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1) if i + 1 < N else (None, np.zeros((0, f.size)))
        out[i, i + 1:] = C[:, k].mean(-1)
    return out + out.T


def modularity(C, labels):
    """Newman modularity of a weighted (positive part) correlation graph."""
    W = np.clip(C, 0, None).copy()
    np.fill_diagonal(W, 0)
    m = W.sum() / 2
    if m <= 0:
        return 0.0
    k = W.sum(1)
    same = labels[:, None] == labels[None, :]
    return float(((W - np.outer(k, k) / (2 * m)) * same).sum() / (2 * m))


def best_clusters(C, kmax=12):
    N = C.shape[0]
    D = 1 - C
    np.fill_diagonal(D, 0)
    L = linkage(D[np.triu_indices(N, 1)], "average")
    best = (0.0, np.ones(N, int))
    for k in range(2, min(kmax, N - 1) + 1):
        lab = fcluster(L, k, "maxclust")
        q = modularity(C, lab)
        if q > best[0]:
            best = (q, lab)
    return best


# ----------------------------------------------------------------------------- loading
def load_source(r: Run, source: str):
    if source == "auto":
        csv = r.out / "traces" / "dff_auto.csv"
        mcsv = r.out / "masks" / "auto_masks.csv"
        if not csv.exists() or not mcsv.exists():
            return None
        df = pd.read_csv(csv)
        names = [c for c in df.columns if c.startswith("dend_")]
        m = pd.read_csv(mcsv).set_index("name")
        cent = np.array([[m.loc[n, "cz"], m.loc[n, "cy"], m.loc[n, "cx"]] for n in names]) * np.array(VOXEL_ZYX)
        size = np.array([m.loc[n, "n_vox"] for n in names], float)
    else:
        csv = r.out / "traces" / "dff_human_sameextractor.csv"
        if not csv.exists() or not human_masks_valid(r):
            return None
        df = pd.read_csv(csv)
        names = [c for c in df.columns if c.startswith("dend_")]
        shape = tifffile.memmap(str(r.raw), mode="r").shape[1:]
        hm = dict(human_masks(r, shape))
        cent = np.array([np.argwhere(hm[n]).mean(0) for n in names]) * np.array(VOXEL_ZYX)
        size = np.array([hm[n].sum() for n in names], float)
    return df, names, cent, size


# ----------------------------------------------------------------------------- per run
def analyze(key: str, source: str) -> dict | None:
    r = get_run(key)
    got = load_source(r, source)
    if got is None:
        return None
    df, names, cent, size = got
    fr = r.frame_rate
    t = df["time_s"].to_numpy()
    M = df[names].to_numpy(float)
    T, N = M.shape
    if N < 8:
        return None
    rng = np.random.default_rng(0)
    ms = int(10 * fr)
    Z = rz(M)
    A = Z > ZTHR
    ons = onsets_from_active(A)
    iu = np.triu_indices(N, 1)
    dist = np.linalg.norm(cent[iu[0]] - cent[iu[1]], axis=1)
    ddepth = cent[iu[1], 1] - cent[iu[0], 1]          # j deeper than i -> positive
    dlat = np.hypot(cent[iu[0], 0] - cent[iu[1], 0], cent[iu[0], 2] - cent[iu[1], 2])
    R = dict(run=key, source=source, mouse=r.mouse, fov=fov_group(r), n=N, T=T, fr=fr, dur_min=T / fr / 60)
    D = pd.DataFrame(dict(run=key, source=source, fov=fov_group(r), dendrite=names, depth_um=cent[:, 1],
                          x_um=cent[:, 2], z_um=cent[:, 0], n_vox=size, n_events=[len(o) for o in ons]))
    D["event_rate_per_min"] = D.n_events / R["dur_min"]

    # ---- behavior & global
    beh = load_behavior(r)
    B = {}
    for k in ("pupil", "whisker", "accel"):
        if beh.get(k) is not None:
            y = resample_to(beh[f"{k}_t"], beh[k], t)
            if np.isfinite(y).mean() > 0.9:
                B[k] = np.where(np.isfinite(y), y, np.nanmedian(y))
    gpath = r.out / "traces" / "global_ca.csv"
    g = pd.read_csv(gpath)
    gy = resample_to(g["time_s"].to_numpy(), g["global_dff"].to_numpy(), t)
    gy = np.where(np.isfinite(gy), gy, np.nanmedian(gy))

    # ---- Pearson baseline
    C0 = zc(M).T @ zc(M)
    pr = C0[iu]
    nullC = np.stack([(lambda S: (zc(S).T @ zc(S))[iu])(shift_all(M, rng, ms)) for _ in range(NSH)])
    hi = np.percentile(nullC, 97.5)
    lo = np.percentile(nullC, 2.5)
    pear_sig = pr > np.percentile(nullC, 99.5)
    R.update(mean_r=float(pr.mean()), frac_pos_sig=float((pr > hi).mean()), frac_neg_sig=float((pr < lo).mean()),
             frac_neg_sig_null_expect=0.025)

    # ---- A synchrony
    nact = A.sum(1)
    syn_null = np.stack([shift_all(A, rng, ms).sum(1) for _ in range(NSH)])
    ks = [2, 3, 5, int(max(5, 0.1 * N)), int(max(8, 0.25 * N))]
    for k in ks:
        obs = (nact >= k).mean()
        nul = (syn_null >= k).mean(1)
        R[f"frac_frames_ge{k}"] = float(obs)
        R[f"null_frames_ge{k}"] = float(nul.mean())
        R[f"excess_frames_ge{k}"] = float(obs - nul.mean())
        R[f"ratio_frames_ge{k}"] = float(obs / max(nul.mean(), 1.0 / (T * NSH)))   # floor: 1 frame in all shifts
    R["sync_10pct_obs"] = R[f"frac_frames_ge{ks[3]}"]
    R["sync_10pct_null"] = R[f"null_frames_ge{ks[3]}"]
    R["sync_10pct_ratio"] = R[f"ratio_frames_ge{ks[3]}"]
    R["sync_25pct_obs"] = R[f"frac_frames_ge{ks[4]}"]
    R["sync_25pct_null"] = R[f"null_frames_ge{ks[4]}"]
    R["sync_5_ratio"] = R["ratio_frames_ge5"]
    R["frac_frames_silent"] = float((nact == 0).mean())
    R["frac_frames_silent_null"] = float((syn_null == 0).mean())
    # bimodality of the population state: variance of #active vs independent null (Fano of co-activity)
    R["coactivity_fano"] = float(nact.var() / max(nact.mean(), 1e-9))
    R["coactivity_fano_null"] = float((syn_null.var(1) / np.maximum(syn_null.mean(1), 1e-9)).mean())
    R["max_coactive_frac"] = float(nact.max() / N)
    R["max_coactive_frac_null"] = float(syn_null.max(1).mean() / N)

    # ---- B mutual information
    mi = binary_mi(A)[iu]
    mi_null = np.stack([binary_mi(shift_all(A, rng, ms))[iu] for _ in range(NSH)])
    mi_thr = np.percentile(mi_null, 99.5)
    mi_sig = mi > mi_thr
    R.update(frac_mi_sig=float(mi_sig.mean()), frac_pearson_sig=float(pear_sig.mean()),
             frac_mi_not_pearson=float((mi_sig & ~pear_sig).mean()),
             frac_pearson_not_mi=float((pear_sig & ~mi_sig).mean()),
             mi_pearson_jaccard=float((mi_sig & pear_sig).sum() / max(1, (mi_sig | pear_sig).sum())))

    # ---- C lagged structure
    maxlag = int(round(MAXLAG_S * fr))
    lags, CL = lagged_corr(Z, maxlag)
    CLp = CL[:, iu[0], iu[1]]                      # (L, P): corr(x_i(t), x_j(t+lag))
    k0 = maxlag
    peak = np.abs(CLp).argmax(0)
    rpeak = CLp[peak, np.arange(len(pr))]
    lag_s = lags[peak] / fr                         # >0: j follows i
    nullL = []
    for _ in range(NSH):
        S = rz(shift_all(M, rng, ms))
        _, Cn = lagged_corr(S, maxlag)
        nullL.append(np.abs(Cn[:, iu[0], iu[1]]).max(0))
    nullL = np.concatenate(nullL)
    lag_sig = np.abs(rpeak) > np.percentile(nullL, 99.5)
    seq = lag_sig & (np.abs(lags[peak]) >= 1) & (np.abs(rpeak) > np.abs(CLp[k0]) + 0.05)
    R.update(frac_lag_sig=float(lag_sig.mean()), frac_sequential=float(seq.mean()),
             frac_sequential_of_sig=float(seq.sum() / max(1, lag_sig.sum())))
    # direction along depth among sequential pairs: lag>0 means j (deeper if ddepth>0) follows i
    if seq.sum() >= 5:
        deeper_follows = np.sign(lag_s[seq]) * np.sign(ddepth[seq])      # +1: deeper dendrite follows (top-down)
        deeper_follows = deeper_follows[deeper_follows != 0]
        R["topdown_frac"] = float((deeper_follows > 0).mean()) if deeper_follows.size else np.nan
        R["n_sequential"] = int(seq.sum())
        R["seq_median_lag_s"] = float(np.median(np.abs(lag_s[seq])))
        R["seq_median_dist_um"] = float(np.median(dist[seq]))
        ok = dist[seq] > 0
        R["seq_speed_um_per_s_median"] = float(np.median(dist[seq][ok] / np.maximum(np.abs(lag_s[seq][ok]), 1 / fr)))
        R["seq_rho_lag_dist"] = float(sp.spearmanr(np.abs(lag_s[seq]), dist[seq])[0])
    # ---- D spatial
    def expfit(d, y):
        try:
            from scipy.optimize import curve_fit
            p, _ = curve_fit(lambda x, a, L, c: a * np.exp(-x / L) + c, d, y, p0=(0.2, 50, 0), maxfev=5000,
                             bounds=([-1, 2, -1], [1, 2000, 1]))
            return float(p[1]), float(p[0])
        except Exception:
            return np.nan, np.nan
    bins = np.arange(0, 400, 20)
    idx = np.digitize(dist, bins)
    prof = np.array([pr[idx == k].mean() if (idx == k).sum() > 10 else np.nan for k in range(1, len(bins))])
    ok = np.isfinite(prof)
    L_um, amp = expfit(bins[:-1][ok] + 10, prof[ok]) if ok.sum() > 4 else (np.nan, np.nan)
    R.update(corr_decay_length_um=L_um, corr_decay_amp=amp,
             rho_r_dist=float(sp.spearmanr(dist, pr)[0]),
             rho_r_lateral=float(sp.spearmanr(dlat, pr)[0]), rho_r_depthdiff=float(sp.spearmanr(np.abs(ddepth), pr)[0]))
    col = (dlat < 15) & (np.abs(ddepth) > 30)          # same column, different depth: same trunk?
    row = (dlat > 40) & (np.abs(ddepth) < 15)          # same depth, lateral neighbours
    R["r_same_column"] = float(pr[col].mean()) if col.sum() >= 5 else np.nan
    R["r_same_depth_lateral"] = float(pr[row].mean()) if row.sum() >= 5 else np.nan
    R["n_same_column_pairs"] = int(col.sum())
    neg = pr < lo
    R["neg_pairs_median_dist_um"] = float(np.median(dist[neg])) if neg.any() else np.nan
    R["pos_pairs_median_dist_um"] = float(np.median(dist[pr > hi])) if (pr > hi).any() else np.nan

    # ---- E latent factors
    ev = np.linalg.eigvalsh(np.cov(M.T))[::-1]
    ev_null = np.stack([np.linalg.eigvalsh(np.cov(shift_all(M, rng, ms).T))[::-1] for _ in range(NSH)])
    R["pc1_var_frac"] = float(ev[0] / ev.sum())
    R["pc1_var_frac_null"] = float((ev_null[:, 0] / ev_null.sum(1)).mean())
    R["n_sig_pcs"] = int((ev > np.percentile(ev_null, 97.5, 0)).sum())
    R["n_sig_pcs_frac"] = R["n_sig_pcs"] / N
    X = np.c_[np.ones(T), gy, *[B[k] for k in B]]
    beta = np.linalg.lstsq(X, M, rcond=None)[0]
    Res = M - X @ beta
    R["var_explained_global_beh_median"] = float(np.median(1 - Res.var(0) / M.var(0)))
    Cr = zc(Res).T @ zc(Res)
    prr = Cr[iu]
    nullR = np.stack([(lambda S: (zc(S).T @ zc(S))[iu])(shift_all(Res, rng, ms)) for _ in range(NSH)])
    R.update(resid_mean_r=float(prr.mean()), resid_frac_pos_sig=float((prr > np.percentile(nullR, 97.5)).mean()),
             resid_frac_neg_sig=float((prr < np.percentile(nullR, 2.5)).mean()))
    evr = np.linalg.eigvalsh(np.cov(Res.T))[::-1]
    evr_null = np.stack([np.linalg.eigvalsh(np.cov(shift_all(Res, rng, ms).T))[::-1] for _ in range(NSH)])
    R["resid_n_sig_pcs"] = int((evr > np.percentile(evr_null, 97.5, 0)).sum())
    q, lab = best_clusters(Cr)
    qn = [best_clusters((lambda S: zc(S).T @ zc(S))(shift_all(Res, rng, ms)))[0] for _ in range(5)]
    R.update(resid_modularity=float(q), resid_modularity_null=float(np.mean(qn)), n_clusters=int(lab.max()))
    # cluster compactness: mean within-cluster centroid distance vs random relabelling
    same = lab[iu[0]] == lab[iu[1]]
    if same.sum() >= 5 and (~same).sum() >= 5:
        within = dist[same].mean()
        rnd = [dist[(lambda l: l[iu[0]] == l[iu[1]])(rng.permutation(lab))].mean() for _ in range(200)]
        R["cluster_within_dist_um"] = float(within)
        R["cluster_within_dist_null_um"] = float(np.mean(rnd))
        R["cluster_compact_p"] = float((1 + (np.array(rnd) <= within).sum()) / 201)
    D["cluster"] = lab

    # ---- F state gating
    def state_split(y, name):
        hi_ = y > np.percentile(y, 60)
        lo_ = y < np.percentile(y, 40)
        if hi_.sum() < 30 or lo_.sum() < 30:
            return
        for nm, sel in ((f"{name}_high", hi_), (f"{name}_low", lo_)):
            Cs = zc(M[sel]).T @ zc(M[sel])
            R[f"r_{nm}"] = float(Cs[iu].mean())
            As = A[sel]
            nul = np.stack([shift_all(A, rng, ms)[sel].sum(1) for _ in range(10)])
            R[f"sync_{nm}_excess"] = float(((As.sum(1) >= 5).mean() - (nul >= 5).mean()) * 100)   # % of frames, >= 5 co-active
            R[f"rate_{nm}"] = float(As.mean())
        R[f"r_gain_{name}"] = R[f"r_{name}_high"] - R[f"r_{name}_low"]
        R[f"sync_gain_{name}"] = R[f"sync_{name}_high_excess"] - R[f"sync_{name}_low_excess"]
    for k in ("pupil", "accel", "whisker"):
        if k in B:
            state_split(B[k], k)

    # ---- G self dynamics
    cv, fano, fac = [], [], []
    for j, o in enumerate(ons):
        if len(o) >= 4:
            iei = np.diff(o) / fr
            cv.append(iei.std() / iei.mean())
            amp = np.array([M[s:s + int(2 * fr), j].max() for s in o[1:]])
            fac.append(sp.spearmanr(iei, amp)[0] if len(iei) >= 5 else np.nan)
        else:
            cv.append(np.nan)
            fac.append(np.nan)
        # Fano factor of counts in 10 s bins
        nb = int(T // (10 * fr))
        if nb >= 5 and len(o):
            cnt = np.histogram(o, bins=nb, range=(0, nb * 10 * fr))[0]
            fano.append(cnt.var() / max(cnt.mean(), 1e-9))
        else:
            fano.append(np.nan)
    D["iei_cv"] = cv
    D["fano_10s"] = fano
    D["amp_vs_prev_iei_rho"] = fac
    R.update(iei_cv_median=float(np.nanmedian(cv)), fano_median=float(np.nanmedian(fano)),
             facilitation_rho_median=float(np.nanmedian(fac)),
             frac_dend_burst_cv_gt_1_2=float(np.nanmean(np.array(cv) > 1.2)))

    # ---- H common input: tuning similarity vs pairwise r
    if len(B) >= 2:
        tun = np.stack([(zc(M) * zc(B[k][:, None])).sum(0) for k in B], 1)      # N x nbeh
        for k_, kname in enumerate(B):
            D[f"r_{kname}"] = tun[:, k_]
        tz = zc(tun.T).T if tun.shape[1] > 1 else tun
        sim = (tun[iu[0]] * tun[iu[1]]).sum(1) / (np.linalg.norm(tun[iu[0]], axis=1) * np.linalg.norm(tun[iu[1]], axis=1) + 1e-12)
        R["rho_tuningsim_pairr"] = float(sp.spearmanr(sim, pr)[0])
        R["rho_tuningsim_residr"] = float(sp.spearmanr(sim, prr)[0])
        # pairs with similar tuning but residual-independent, etc.
    # ---- I global events
    glob_thr = max(2, int(0.2 * N))
    gframes = nact >= glob_thr
    a = np.r_[False, gframes, False]
    d_ = np.diff(a.astype(int))
    gst, gen = np.flatnonzero(d_ == 1), np.flatnonzero(d_ == -1)
    R["n_global_events"] = int(len(gst))
    if len(gst) >= 3:
        part = np.zeros(N)
        ranks = []
        for s_, e_ in zip(gst, gen):
            w = slice(max(0, s_ - 2), min(T, e_ + 2))
            act = A[w].any(0)
            part += act
            # latency = first active frame in window, for participating dendrites
            lat = np.where(act, A[w].argmax(0), np.nan)
            ranks.append(lat)
        part /= len(gst)
        D["global_participation"] = part
        R["participation_median"] = float(np.median(part))
        R["participation_bimodality"] = float(((part < 0.2) | (part > 0.8)).mean())
        R["rho_participation_depth"] = float(sp.spearmanr(part, cent[:, 1])[0])
        R["rho_participation_size"] = float(sp.spearmanr(part, size)[0])
        R["rho_participation_rate"] = float(sp.spearmanr(part, D.event_rate_per_min)[0])
        Lt = np.array(ranks)                                       # events x N latency (nan if absent)
        keep = np.isfinite(Lt).sum(0) >= max(3, 0.5 * len(gst))
        if keep.sum() >= 5:
            Lk = Lt[:, keep]
            # Kendall W over dendrites with mean-imputed missing; vs shuffled latencies within event
            def kw(X):
                Xr = np.apply_along_axis(sp.rankdata, 1, np.where(np.isfinite(X), X, np.nanmean(X, 1, keepdims=True)))
                Rs = Xr.sum(0)
                m, n = X.shape
                return 12 * ((Rs - Rs.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n))
            W = kw(Lk)
            Wn = [kw(np.stack([rng.permutation(row) for row in Lk])) for _ in range(200)]
            R["leader_kendall_W"] = float(W)
            R["leader_kendall_W_p"] = float((1 + (np.array(Wn) >= W).sum()) / 201)
            meanlat = np.nanmean(Lk, 0)
            R["rho_leadlatency_depth"] = float(sp.spearmanr(meanlat, cent[keep, 1])[0])
            D.loc[keep, "global_mean_latency_fr"] = meanlat

    # ---- K dendrite-dendrite coherence by band
    for bn, band in (("slow", (0.04, 0.2)), ("mid", (0.2, 0.5)), ("fast", (0.5, 1.0))):
        Cm = coherence_matrix(M, fr, band)[iu]
        Cn = np.concatenate([coherence_matrix(shift_all(M, rng, ms), fr, band)[iu] for _ in range(3)])
        R[f"coh_{bn}_mean"] = float(Cm.mean())
        R[f"coh_{bn}_excess"] = float(Cm.mean() - Cn.mean())
        R[f"coh_{bn}_frac_sig"] = float((Cm > np.percentile(Cn, 99)).mean())
        if bn == "slow":
            R["rho_cohslow_dist"] = float(sp.spearmanr(dist, Cm)[0])
        if bn == "fast":
            R["rho_cohfast_dist"] = float(sp.spearmanr(dist, Cm)[0])

    # ---- L pupil phase groups
    if "pupil" in B:
        y = B["pupil"]
        nper = int(round(25.6 * fr))
        f, Pxy = signal.csd(M.T, y[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
        _, Cxy = signal.coherence(M.T, y[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
        kb = (f >= 0.04) & (f <= 0.2)
        ph = np.angle(Pxy[:, kb].sum(-1))
        coh_p = Cxy[:, kb].mean(-1)
        cn = np.concatenate([signal.coherence(M.T, np.roll(y, rng.integers(ms, T - ms))[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)[1][:, kb].mean(-1) for _ in range(20)])
        sigp = coh_p > np.percentile(cn, 99)
        inph = np.abs(ph) < np.pi / 2
        D["pupil_coh_slow"] = coh_p
        D["pupil_inphase"] = inph
        D["pupil_coh_sig"] = sigp
        ip, ap = sigp & inph, sigp & ~inph
        R.update(n_pupil_inphase=int(ip.sum()), n_pupil_antiphase=int(ap.sum()))
        if ip.sum() >= 3 and ap.sum() >= 3:
            R["pupil_inphase_depth_um"] = float(cent[ip, 1].mean())
            R["pupil_antiphase_depth_um"] = float(cent[ap, 1].mean())
            R["pupil_phase_depth_p"] = float(sp.mannwhitneyu(cent[ip, 1], cent[ap, 1]).pvalue)
            grp = np.where(ip, 1, np.where(ap, 2, 0))
            gi, gj = grp[iu[0]], grp[iu[1]]
            R["r_within_inphase"] = float(pr[(gi == 1) & (gj == 1)].mean())
            R["r_within_antiphase"] = float(pr[(gi == 2) & (gj == 2)].mean())
            R["r_between_phase_groups"] = float(pr[((gi == 1) & (gj == 2)) | ((gi == 2) & (gj == 1))].mean())
            if "accel" in B:
                ra = (zc(M) * zc(B["accel"][:, None])).sum(0)
                R["pupil_inphase_r_accel"] = float(ra[ip].mean())
                R["pupil_antiphase_r_accel"] = float(ra[ap].mean())
            R["pupil_inphase_rate"] = float(D.event_rate_per_min[ip].mean())
            R["pupil_antiphase_rate"] = float(D.event_rate_per_min[ap].mean())
    pairs = pd.DataFrame(dict(run=key, fov=fov_group(r), a=np.array(names)[iu[0]], b=np.array(names)[iu[1]], r=pr,
                              resid_r=prr, mi=mi, mi_sig=mi_sig, pearson_sig=pear_sig, rpeak=rpeak, lag_s=lag_s,
                              sequential=seq, dist_um=dist, ddepth_um=ddepth, dlat_um=dlat))
    return dict(run=R, dend=D, pairs=pairs)


# ----------------------------------------------------------------------------- cohort
def cohort(source, res):
    out = OUT / source
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([x["run"] for x in res])
    dend = pd.concat([x["dend"] for x in res], ignore_index=True)
    pairs = pd.concat([x["pairs"] for x in res], ignore_index=True)
    runs.to_csv(out / "per_run.csv", index=False)
    dend.to_csv(out / "per_dendrite.csv", index=False)
    pairs.sample(min(len(pairs), 200000), random_state=0).to_csv(out / "pairs_sample.csv", index=False)
    fov = runs.groupby(["fov", "mouse"]).mean(numeric_only=True).reset_index()
    fov.to_csv(out / "per_fov.csv", index=False)
    S = dict(source=source, n_runs=len(runs), n_fovs=int(runs.fov.nunique()), n_mice=int(runs.mouse.nunique()),
             n_dendrite_entries=int(len(dend)), findings=[])
    L = [f"INDEPENDENCE TEST BATTERY - {source} dendrites: {len(runs)} runs, {runs.fov.nunique()} FOVs, "
         f"{runs.mouse.nunique()} mice, {len(dend)} dendrite-run entries, {len(pairs)} pairs.",
         "Nulls: independent circular shifts per dendrite. FOV = unit of replication (Wilcoxon over FOV means).", ""]

    def W(col, mu=0.0):
        v = fov[col].dropna()
        if len(v) < 4:
            return np.nan, len(v), np.nan
        return float(sp.wilcoxon(v - mu).pvalue) if (v != mu).any() else 1.0, len(v), float(v.median())

    def add(name, text, **kw):
        S["findings"].append(dict(name=name, text=text, **kw))
        L.append(f"* [{name}] {text}")

    p, n, med = W("sync_10pct_obs")
    add("A_synchrony", f">= 10% of dendrites co-active in {fov.sync_10pct_obs.median() * 100:.1f}% of frames vs "
        f"{fov.sync_10pct_null.median() * 100:.3f}% expected under independence (p={p:.2g}, {n} FOVs); >= 25% co-active: "
        f"{fov.sync_25pct_obs.median() * 100:.2f}% vs {fov.sync_25pct_null.median() * 100:.4f}%. >= 5 co-active: "
        f"{fov.frac_frames_ge5.median() * 100:.1f}% vs {fov.null_frames_ge5.median() * 100:.1f}% ({fov.sync_5_ratio.median():.1f}x). "
        f"Max simultaneous fraction {fov.max_coactive_frac.median():.2f} vs {fov.max_coactive_frac_null.median():.2f} expected. "
        f"Silent frames (no dendrite active): {fov.frac_frames_silent.median() * 100:.0f}% vs {fov.frac_frames_silent_null.median() * 100:.0f}% "
        f"expected; Fano factor of the number of co-active dendrites {fov.coactivity_fano.median():.1f} vs {fov.coactivity_fano_null.median():.1f}. "
        f"The population alternates between silence and bursts of co-activation far beyond what independent units would produce.", p=p)
    p, n, med = W("frac_mi_sig")
    add("B_mutual_information", f"Median {med * 100:.1f}% of pairs show significant binary-event mutual information "
        f"(vs {fov.frac_pearson_sig.median() * 100:.1f}% by Pearson at the same 99.5% null level). Pairs found only by MI: "
        f"{fov.frac_mi_not_pearson.median() * 100:.1f}%; only by Pearson: {fov.frac_pearson_not_mi.median() * 100:.1f}%; "
        f"Jaccard {fov.mi_pearson_jaccard.median():.2f}. Nonlinear / event-level dependence not captured by r is "
        f"{'present' if fov.frac_mi_not_pearson.median() > fov.frac_pearson_not_mi.median() else 'not larger than the reverse'}.", median=med)
    p, n, med = W("frac_sequential")
    td = fov.topdown_frac.dropna()
    ptd = sp.wilcoxon(td - 0.5).pvalue if len(td) >= 4 else np.nan
    add("C_sequences", f"{med * 100:.1f}% of pairs are sequentially coupled (peak |r| at a non-zero lag, above null and "
        f"> zero-lag r + 0.05); among significantly coupled pairs {fov.frac_sequential_of_sig.median() * 100:.0f}% are sequential. "
        f"Median lag {fov.seq_median_lag_s.median():.2f} s over {fov.seq_median_dist_um.median():.0f} um "
        f"(median apparent speed {fov.seq_speed_um_per_s_median.median():.0f} um/s). Direction: the deeper dendrite follows in "
        f"{td.median() * 100:.0f}% of depth-separated sequential pairs (50% = no preferred direction; p={ptd:.2g}).", p=ptd)
    p, n, med = W("rho_r_dist")
    add("D_spatial", f"r falls with distance (Spearman {med:+.3f}, p={p:.2g}); decay length {fov.corr_decay_length_um.median():.0f} um "
        f"(exp fit, amplitude {fov.corr_decay_amp.median():.2f}). Lateral separation rho {fov.rho_r_lateral.median():+.3f} vs depth "
        f"separation rho {fov.rho_r_depthdiff.median():+.3f}. Same-column pairs (lateral < 15 um, depth gap > 30 um; same-trunk "
        f"candidates) r = {fov.r_same_column.median():.3f} vs same-depth lateral neighbours {fov.r_same_depth_lateral.median():.3f}. "
        f"Negative pairs beyond null: {fov.frac_neg_sig.median() * 100:.1f}% (2.5% expected), median distance "
        f"{fov.neg_pairs_median_dist_um.median():.0f} um vs {fov.pos_pairs_median_dist_um.median():.0f} um for positive pairs.")
    p, n, med = W("pc1_var_frac")
    add("E_latent", f"PC1 explains {med * 100:.0f}% of variance vs {fov.pc1_var_frac_null.median() * 100:.0f}% under independence; "
        f"{fov.n_sig_pcs.median():.0f} significant PCs ({fov.n_sig_pcs_frac.median() * 100:.0f}% of N). Global Ca + behavior explain a "
        f"median {fov.var_explained_global_beh_median.median() * 100:.0f}% of each dendrite's variance. After removing them, residual "
        f"mean r = {fov.resid_mean_r.median():.3f}, {fov.resid_frac_pos_sig.median() * 100:.0f}% positive / {fov.resid_frac_neg_sig.median() * 100:.0f}% "
        f"negative pairs beyond null, {fov.resid_n_sig_pcs.median():.0f} residual PCs: structure that is NOT the shared field signal. "
        f"Residual clusters: modularity {fov.resid_modularity.median():.2f} vs {fov.resid_modularity_null.median():.2f} null, "
        f"{fov.n_clusters.median():.0f} clusters; within-cluster distance {fov.cluster_within_dist_um.median():.0f} um vs "
        f"{fov.cluster_within_dist_null_um.median():.0f} um random (p<0.05 in {(runs.cluster_compact_p < 0.05).sum()}/{runs.cluster_compact_p.notna().sum()} runs).")
    for k in ("pupil", "accel", "whisker"):
        if f"r_gain_{k}" in fov:
            p, n, med = W(f"r_gain_{k}")
            p2, _, med2 = W(f"sync_gain_{k}")
            add(f"F_state_{k}", f"High vs low {k}: pairwise r {fov[f'r_{k}_high'].median():.3f} vs {fov[f'r_{k}_low'].median():.3f} "
                f"(gain {med:+.3f}, p={p:.2g}); excess synchronous frames (>= 5 co-active, above null) {fov[f'sync_{k}_high_excess'].median():.1f}% vs "
                f"{fov[f'sync_{k}_low_excess'].median():.1f}% (p={p2:.2g}); activity {fov[f'rate_{k}_high'].median() * 100:.1f}% vs "
                f"{fov[f'rate_{k}_low'].median() * 100:.1f}% of frames.", p=p)
    p, n, med = W("iei_cv_median", 1)
    pf, _, mf = W("fano_median", 1)
    pfa, _, mfa = W("facilitation_rho_median")
    add("G_self", f"Inter-event-interval CV {med:.2f} (1 = Poisson; p={p:.2g}), Fano factor of 10 s counts {mf:.2f} (p={pf:.2g}); "
        f"{fov.frac_dend_burst_cv_gt_1_2.median() * 100:.0f}% of dendrites bursty (CV > 1.2). Event amplitude vs preceding interval "
        f"Spearman {mfa:+.2f} (p={pfa:.2g}; + = recovery/depression, - = facilitation).", p=p)
    if "rho_tuningsim_pairr" in fov:
        p, n, med = W("rho_tuningsim_pairr")
        p2, _, med2 = W("rho_tuningsim_residr")
        add("H_common_input", f"Pairs with similar behavior tuning are more correlated (rho {med:+.3f}, p={p:.2g}); after removing "
            f"global Ca + behavior the relation is {med2:+.3f} (p={p2:.2g}). Shared behavioral drive explains part of the pairwise structure.", p=p)
    if "leader_kendall_W" in fov:
        p, n, med = W("rho_participation_depth")
        kwp = (runs.leader_kendall_W_p < 0.05).sum()
        add("I_global_events", f"{fov.n_global_events.median():.0f} global events per run. Participation probability median "
            f"{fov.participation_median.median():.2f}; {fov.participation_bimodality.median() * 100:.0f}% of dendrites are all-or-none "
            f"(<0.2 or >0.8). Participation vs depth rho {med:+.2f} (p={p:.2g}), vs size {fov.rho_participation_size.median():+.2f}, vs own "
            f"event rate {fov.rho_participation_rate.median():+.2f}. Leader consistency: Kendall W {fov.leader_kendall_W.median():.2f}, "
            f"significant in {kwp}/{runs.leader_kendall_W_p.notna().sum()} runs; lead latency vs depth rho {fov.rho_leadlatency_depth.median():+.2f} "
            f"(- = deeper dendrites lead).")
    for bn in ("slow", "mid", "fast"):
        p, n, med = W(f"coh_{bn}_excess")
        L.append(f"  K dendrite-dendrite coherence {bn}: excess {med:+.3f} over null (p={p:.2g}), {fov[f'coh_{bn}_frac_sig'].median() * 100:.0f}% pairs significant")
    S["findings"].append(dict(name="K_coherence", slow=float(fov.coh_slow_excess.median()), mid=float(fov.coh_mid_excess.median()),
                              fast=float(fov.coh_fast_excess.median()), rho_slow_dist=float(fov.rho_cohslow_dist.median()),
                              rho_fast_dist=float(fov.rho_cohfast_dist.median())))
    L.append(f"  K coherence vs distance: slow rho {fov.rho_cohslow_dist.median():+.3f}, fast rho {fov.rho_cohfast_dist.median():+.3f}")
    if "pupil_phase_depth_p" in fov:
        d_ = fov.dropna(subset=["pupil_inphase_depth_um"])
        add("L_pupil_groups", f"Dendrites in phase with pupil (n median {fov.n_pupil_inphase.median():.0f}) vs anti-phase "
            f"({fov.n_pupil_antiphase.median():.0f}): depth {d_.pupil_inphase_depth_um.median():.0f} vs {d_.pupil_antiphase_depth_um.median():.0f} um "
            f"(Mann-Whitney p<0.05 in {(runs.pupil_phase_depth_p < 0.05).sum()}/{runs.pupil_phase_depth_p.notna().sum()} runs); event rate "
            f"{d_.pupil_inphase_rate.median():.2f} vs {d_.pupil_antiphase_rate.median():.2f}/min; r within in-phase {d_.r_within_inphase.median():.3f}, "
            f"within anti-phase {d_.r_within_antiphase.median():.3f}, between groups {d_.r_between_phase_groups.median():.3f}; "
            f"r with locomotion {d_.pupil_inphase_r_accel.median():+.3f} (in-phase) vs {d_.pupil_antiphase_r_accel.median():+.3f} (anti-phase).")
    # J identity across runs
    J = []
    for f_, g in dend.groupby("fov"):
        ks = sorted(g.run.unique())
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a = g[g.run == ks[i]].set_index("dendrite")
                b = g[g.run == ks[j]].set_index("dendrite")
                c = a.index.intersection(b.index)
                if len(c) < 10:
                    continue
                row = dict(fov=f_, run_a=ks[i], run_b=ks[j], n=len(c))
                for col in ("event_rate_per_min", "r_accel", "r_pupil", "r_whisker", "pupil_coh_slow", "global_participation", "iei_cv"):
                    if col in a and col in b:
                        x, y = a.loc[c, col], b.loc[c, col]
                        ok = np.isfinite(x) & np.isfinite(y)
                        row[f"rho_{col}"] = float(sp.spearmanr(x[ok], y[ok])[0]) if ok.sum() >= 10 else np.nan
                if "pupil_inphase" in a and "pupil_coh_sig" in a:
                    both = c[(a.loc[c, "pupil_coh_sig"] & b.loc[c, "pupil_coh_sig"]).to_numpy(bool)]
                    row["pupil_phase_agreement"] = float((a.loc[both, "pupil_inphase"] == b.loc[both, "pupil_inphase"]).mean()) if len(both) >= 5 else np.nan
                    row["n_both_pupil_sig"] = int(len(both))
                J.append(row)
    Jd = pd.DataFrame(J)
    if len(Jd):
        Jd.to_csv(out / "identity_across_runs.csv", index=False)
        txt = "; ".join(f"{c[4:]} rho {Jd[c].median():+.2f}" for c in Jd.columns if c.startswith("rho_") and Jd[c].notna().any())
        pa = Jd.pupil_phase_agreement.dropna() if "pupil_phase_agreement" in Jd else pd.Series(dtype=float)
        add("J_identity", f"Same dendrite across runs of one FOV ({len(Jd)} run pairs): {txt}. Pupil phase (in/anti) agrees in "
            f"{pa.median() * 100:.0f}% of dendrites significant in both runs (50% = chance)." if len(pa) else
            f"Same dendrite across runs of one FOV ({len(Jd)} run pairs): {txt}.")
    # per-dendrite mixed models
    try:
        import statsmodels.formula.api as smf
        d2 = dend.dropna(subset=["global_participation", "depth_um"]).copy()
        if len(d2) > 50:
            d2["dz"] = d2.groupby("fov").depth_um.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            d2["sz"] = d2.groupby("fov").n_vox.transform(lambda x: (np.log(x) - np.log(x).mean()) / (np.log(x).std() + 1e-9))
            d2["rz_"] = d2.groupby("fov").event_rate_per_min.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            m = smf.mixedlm("global_participation ~ dz + sz + rz_", d2, groups=d2["fov"]).fit(reml=True)
            add("I_participation_LMM", "Global-event participation ~ depth + log size + own event rate (LMM, random intercept FOV): "
                + ", ".join(f"{k} {m.params[k]:+.3f} (p={m.pvalues[k]:.2g})" for k in ("dz", "sz", "rz_")) + ".")
    except Exception as e:
        L.append(f"  (LMM failed: {e})")
    L += ["", "Reading guide: independent units predict ratio ~1 in A, few MI/sequential pairs in B/C, no residual structure in E,",
          "no state gain in F, CV ~1 in G, no leader consistency in I, chance-level identity stability in J.",
          "Caveats: 5 Hz sampling (lags < 0.2 s unresolved), ~106 s per run, 2 mice with behavior, 3 in total; SCAPE optical",
          "crosstalk between neighbouring dendrites inflates short-range correlations (the shell subtraction reduces but does not remove it)."]
    (out / "summary.txt").write_text("\n".join(L))
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))
    figures(source, runs, fov, dend, pairs, out)
    return L


def figures(source, runs, fov, dend, pairs, out):
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    a = ax[0, 0]
    for k, lab in ((2, ">=2"), (5, ">=5")):
        pass
    cols = [c for c in fov.columns if c.startswith("frac_frames_ge")]
    kk = [int(c.split("ge")[1]) for c in cols]
    o = np.argsort(kk)
    for _, row in fov.iterrows():
        a.plot([kk[i] for i in o], [row[cols[i]] * 100 for i in o], "-o", ms=3, lw=0.7, alpha=0.7)
        a.plot([kk[i] for i in o], [row["null" + cols[i][4:]] * 100 for i in o], ":", lw=0.7, color="k", alpha=0.5)
    a.set_yscale("log")
    a.set_xlabel("k dendrites co-active")
    a.set_ylabel("% frames (solid obs, dotted independent)")
    a.set_title("A  synchrony excess (one line per FOV)")
    a = ax[0, 1]
    a.scatter(fov.frac_pearson_sig * 100, fov.frac_mi_sig * 100, s=18)
    m_ = max(fov.frac_pearson_sig.max(), fov.frac_mi_sig.max()) * 100
    a.plot([0, m_], [0, m_], "k--", lw=0.6)
    a.set_xlabel("% pairs significant (Pearson)")
    a.set_ylabel("% pairs significant (binary MI)")
    a.set_title("B  dependence: linear vs event-level")
    a = ax[0, 2]
    sq = pairs[pairs.sequential]
    if len(sq):
        a.scatter(sq.dist_um, np.abs(sq.lag_s), s=3, alpha=0.3)
        a.set_xlabel("distance (um)")
        a.set_ylabel("|lag| (s)")
    a.set_title(f"C  sequential pairs (n={len(sq)})")
    a = ax[0, 3]
    bins = np.arange(0, 400, 20)
    for f_, g in pairs.groupby("fov"):
        idx = np.digitize(g.dist_um, bins)
        a.plot(bins[:-1] + 10, [g.r[idx == k].mean() if (idx == k).sum() > 10 else np.nan for k in range(1, len(bins))], lw=0.7, alpha=0.6)
        a.plot(bins[:-1] + 10, [g.resid_r[idx == k].mean() if (idx == k).sum() > 10 else np.nan for k in range(1, len(bins))], lw=0.7, ls="--", alpha=0.6, color="k")
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("distance (um)")
    a.set_ylabel("mean r (solid) / residual r (dashed)")
    a.set_title("D/E  r vs distance, raw and after global+behavior")
    a = ax[1, 0]
    a.scatter(fov.pc1_var_frac_null * 100, fov.pc1_var_frac * 100, s=18, label="PC1 % var")
    a.plot([0, 60], [0, 60], "k--", lw=0.6)
    a.set_xlabel("independent null")
    a.set_ylabel("observed")
    a.set_title("E  shared variance (PC1)")
    a = ax[1, 1]
    for k, c in (("pupil", "#2f6fdf"), ("accel", "#8a3fd1"), ("whisker", "#e69500")):
        if f"r_gain_{k}" in fov:
            a.scatter(np.full(len(fov), ["pupil", "accel", "whisker"].index(k)) + np.random.default_rng(1).normal(0, 0.05, len(fov)), fov[f"r_gain_{k}"], color=c, s=18)
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks([0, 1, 2], ["pupil", "accel", "whisker"])
    a.set_ylabel("pair r (high state) - (low state)")
    a.set_title("F  state gating of coupling")
    a = ax[1, 2]
    a.hist(dend.iei_cv.dropna(), 40, color="#777")
    a.axvline(1, color="k", ls="--")
    a.set_xlabel("inter-event-interval CV (1 = Poisson)")
    a.set_title("G  self dynamics")
    a = ax[1, 3]
    if "global_participation" in dend:
        d_ = dend.dropna(subset=["global_participation"])
        a.scatter(d_.depth_um, d_.global_participation, s=3, alpha=0.3, c=pd.factorize(d_.fov)[0], cmap="tab20")
        a.set_xlabel("depth (um)")
        a.set_ylabel("global-event participation")
    a.set_title("I  who joins global events")
    fig.suptitle(f"Independence test battery - {source} dendrites", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_independence.png", dpi=140)
    fig.savefig(out / "fig_independence.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("auto", "human", "both"), default="both")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    keys = a.run or [r.key for r in discover_runs()]
    t0 = time.time()
    allL = {}
    for s in (("auto", "human") if a.source == "both" else (a.source,)):
        res = []
        with ProcessPoolExecutor(max(1, min(a.jobs, 5))) as ex:
            futs = {ex.submit(analyze, k, s): k for k in keys}
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
        allL[s] = cohort(s, res)
        print("\n".join(allL[s]), "\n")
    if len(allL) == 2:
        (OUT / "compare_auto_vs_human.txt").write_text("AUTO\n" + "\n".join(allL["auto"]) + "\n\nHUMAN\n" + "\n".join(allL["human"]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
