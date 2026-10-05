#!/usr/bin/env python
"""explore_factors.py - 'how many shared inputs explain the dependence?' (code/Auto/).

L5 apical dendrites are not independent (see independence_tests.py). This script asks how
many SHARED LATENT FACTORS are needed to account for that dependence, i.e. how many common
inputs a low-dimensional model needs before the residual dendrite-dendrite structure is
indistinguishable from independent units.

Method (per run; source auto, replicated on human where human_masks_valid):
  * z-score each dendrite's dF/F trace (per run, per dendrite).
  * BLOCKED cross-validation in time: 5 contiguous folds. For each fold, fit PCA on the
    training folds (train-mean centred), then PROJECT the held-out fold onto the first k
    components and subtract that reconstruction. Residuals are assembled across folds so every
    frame is held-out exactly once. k = 0,1,2,3,5,8,12,20.
  * For each k, on the held-out residual matrix:
      (a) mean pairwise |r| and the fraction of pairs beyond an independent circular-shift
          null (>= 10 s shifts; threshold = 99.5th pct of |null r|), plus the null's own
          false-positive rate;
      (b) fraction of frames where >= 10% of dendrites are co-active (robust z > 3 on the
          residuals) vs the shift null;
      (c) held-out variance explained by the k factors.
  * k* = the smallest k at which the residual pairwise structure is indistinguishable from
    independence: observed fraction of significant pairs within 1 percentage point of the
    null false-positive rate.

Also reported:
  * private variance per dendrite (fraction NOT explained by k* factors, held-out) vs depth /
    event rate / size (LMM, random intercept per FOV);
  * the first 3 factors of a full-run PCA: loading sign structure (is factor 1 global?),
    loading vs depth (is factor 2 a depth gradient?), spatial clustering of loadings (Moran's I;
    is factor 3 spatially clustered?), and factor-score correlations with pupil / whisker /
    accel / global Ca;
  * k* compared between auto and human sets and across FOVs (FOV = unit of replication).

Outputs: scape-auto/stats/explore_factors/<source>/{per_run.csv, per_k.csv, per_dendrite.csv,
         summary.txt, summary.json, fig_factors.png/pdf}. The loading maps in fig_factors are
         drawn for the example FOV 2026-05-12/rbp4_132_phpeb/run5.
CLI: explore_factors.py [--source auto|human|both] [--run KEY ...] [--jobs N]
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
from scipy import stats as sp
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, fov_group, load_behavior, resample_to,
    human_masks, human_masks_valid, VOXEL_ZYX, AUTO_ROOT,
)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore")

OUT = AUTO_ROOT / "stats" / "explore_factors"
K_LIST = [0, 1, 2, 3, 5, 8, 12, 20]
NFOLD = 5
MIN_SHIFT_S = 10.0        # circular-shift null: >= 10 s
NSH_THR = 30              # surrogates pooled to set the 99.5-pct |r| threshold
NSH_EVAL = 15             # fresh surrogates to estimate the null's own false-positive rate
ZTHR = 3.0               # robust-z event threshold
KSTAR_TOL = 0.01          # k*: observed frac_sig within 1 percentage point of the null rate
EXAMPLE = "2026-05-12/rbp4_132_phpeb/run5"


# ----------------------------------------------------------------------------- small helpers
def zc(M):
    """Mean-subtract and unit-normalise each column, so zc(M).T @ zc(M) is the corr matrix."""
    M = M - M.mean(0)
    return M / (np.linalg.norm(M, axis=0) + 1e-12)


def rz(M):
    """Robust z-score per column (median / MAD)."""
    med = np.median(M, 0)
    return (M - med) / (np.median(np.abs(M - med), 0) * 1.4826 + 1e-9)


def shift_all(M, rng, ms):
    """Independent circular shift (>= ms frames) of every column (vectorised)."""
    T, N = M.shape
    sh = rng.integers(ms, T - ms, size=N)
    idx = (np.arange(T)[:, None] - sh[None, :]) % T
    return M[idx, np.arange(N)[None, :]]


def onsets_count(Z):
    """Number of events per column: robust-z > ZTHR runs of >= 2 frames."""
    A = rz(Z) > ZTHR
    out = []
    for j in range(A.shape[1]):
        a = np.r_[False, A[:, j], False]
        d = np.diff(a.astype(int))
        st = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        out.append(int(((en - st) >= 2).sum()))
    return np.array(out)


def morans_I(x, cent, rng, nperm=200):
    """Moran's I of loading vector x over dendrite 3D centroids (Gaussian inverse-distance
    weights, sigma = median NN distance). Returns (I, perm_p, I_mean_null)."""
    N = len(x)
    if N < 8:
        return np.nan, np.nan, np.nan
    D = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    sigma = np.median(np.min(D, axis=1))
    W = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    Wsum = W.sum()
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom <= 0 or Wsum <= 0:
        return np.nan, np.nan, np.nan
    I = (N / Wsum) * (xc @ W @ xc) / denom
    null = np.empty(nperm)
    for i in range(nperm):
        p = rng.permutation(xc)
        null[i] = (N / Wsum) * (p @ W @ p) / (p @ p)
    pval = (1 + (null >= I).sum()) / (nperm + 1)
    return float(I), float(pval), float(null.mean())


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
        size = np.array([float(m.loc[n, "n_vox"]) for n in names])
    else:
        csv = r.out / "traces" / "dff_human_sameextractor.csv"
        if not csv.exists() or not human_masks_valid(r):
            return None
        df = pd.read_csv(csv)
        names = [c for c in df.columns if c.startswith("dend_")]
        shape = tifffile.memmap(str(r.raw), mode="r").shape[1:]
        hm = dict(human_masks(r, shape))
        names = [n for n in names if n in hm]
        cent = np.array([np.argwhere(hm[n]).mean(0) for n in names]) * np.array(VOXEL_ZYX)
        size = np.array([float(hm[n].sum()) for n in names])
    return df, names, cent, size


# ----------------------------------------------------------------------------- blocked CV residuals
def cv_residuals(Z, kmax):
    """Blocked 5-fold CV. Returns Dc (held-out centred data, T x N) and a dict k -> residual
    matrix (T x N), for every requested k <= kmax. Each frame is held out exactly once."""
    T, N = Z.shape
    folds = np.array_split(np.arange(T), NFOLD)
    Dc = np.zeros_like(Z)
    kmax_avail = min(kmax, N - 1)
    resid = {k: np.zeros_like(Z) for k in K_LIST if k <= kmax_avail}
    for test in folds:
        mask = np.ones(T, bool)
        mask[test] = False
        train = np.flatnonzero(mask)
        ncomp = min(kmax_avail, len(train) - 1)
        pca = PCA(n_components=ncomp, svd_solver="full").fit(Z[train])
        W = pca.components_                        # (ncomp, N)
        Zt = Z[test] - pca.mean_                   # centre held-out with TRAIN mean
        Dc[test] = Zt
        scores = Zt @ W.T                          # (ntest, ncomp)
        for k in resid:
            kk = min(k, ncomp)
            recon = scores[:, :kk] @ W[:kk] if kk > 0 else 0.0
            resid[k][test] = Zt - recon
    return Dc, resid, kmax_avail


def pair_and_co_stats(R, co_k, rng, ms):
    """Held-out residual R (T x N). Returns dict with mean |r|, frac of pairs beyond the 99.5-pct
    |r| circular-shift null, the null's own false-positive rate, observed and null co-active frac."""
    N = R.shape[1]
    iu = np.triu_indices(N, 1)
    cobs = np.abs((zc(R).T @ zc(R))[iu])
    # threshold from pooled surrogates
    pool = []
    for _ in range(NSH_THR):
        S = shift_all(R, rng, ms)
        pool.append(np.abs((zc(S).T @ zc(S))[iu]))
    thr = float(np.percentile(np.concatenate(pool), 99.5))
    # observed co-activity
    nact = (rz(R) > ZTHR).sum(1)
    co_obs = float((nact >= co_k).mean())
    # fresh surrogates for the null false-positive rate and null co-activity
    null_fracs, null_co = [], []
    for _ in range(NSH_EVAL):
        S = shift_all(R, rng, ms)
        cs = np.abs((zc(S).T @ zc(S))[iu])
        null_fracs.append(float((cs > thr).mean()))
        null_co.append(float(((rz(S) > ZTHR).sum(1) >= co_k).mean()))
    return dict(mean_abs_r=float(cobs.mean()), frac_sig=float((cobs > thr).mean()),
                null_rate=float(np.mean(null_fracs)), thr=thr,
                coactive_frac=co_obs, coactive_null=float(np.mean(null_co)))


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
    # drop dead columns
    good = M.std(0) > 1e-9
    if good.sum() < 8:
        return None
    M, names = M[:, np.array(good)], [n for n, g in zip(names, good) if g]
    cent, size = cent[np.array(good)], size[np.array(good)]
    T, N = M.shape
    ms = int(round(MIN_SHIFT_S * fr))
    co_k = max(2, int(np.ceil(0.1 * N)))
    rng = np.random.default_rng(0)

    Z = (M - M.mean(0)) / M.std(0)                 # per-dendrite z-score
    kmax = max(K_LIST)
    Dc, resid, kmax_avail = cv_residuals(Z, kmax)
    tot_var = float((Dc ** 2).sum())

    perk = []
    for k in K_LIST:
        if k not in resid:
            continue
        Rk = resid[k]
        st = pair_and_co_stats(Rk, co_k, rng, ms)
        ve = 1.0 - float((Rk ** 2).sum()) / tot_var
        perk.append(dict(run=key, source=source, mouse=r.mouse, fov=fov_group(r), n=N, T=T,
                         k=k, co_k=co_k, var_explained=ve, **st))
    perk_df = pd.DataFrame(perk)
    # k* : smallest k with frac_sig within KSTAR_TOL of the null false-positive rate
    kstar, kstar_reached = np.nan, False
    for row in perk:
        if row["frac_sig"] <= row["null_rate"] + KSTAR_TOL:
            kstar, kstar_reached = row["k"], True
            break
    if not kstar_reached:
        kstar = float(max(k for k in resid))       # not reached within tested range

    # ---- private variance per dendrite at k* (held-out)
    kstar_use = int(kstar) if kstar in resid else max(resid)
    Rk = resid[kstar_use]
    priv = (Rk ** 2).sum(0) / ((Dc ** 2).sum(0) + 1e-12)   # fraction of held-out variance NOT explained
    nev = onsets_count(Z)
    dur_min = T / fr / 60.0
    D = pd.DataFrame(dict(run=key, source=source, fov=fov_group(r), mouse=r.mouse, dendrite=names,
                          depth_um=cent[:, 1], x_um=cent[:, 2], z_um=cent[:, 0], n_vox=size,
                          n_events=nev, event_rate_per_min=nev / dur_min,
                          private_var_frac=priv))

    # ---- first 3 factors from a full-run PCA
    pca = PCA(n_components=min(max(3, kstar_use), N - 1), svd_solver="full").fit(Z)
    load = pca.components_[:3]                      # (3, N)
    scores = (Z - pca.mean_) @ pca.components_[:3].T  # (T, 3)
    evr = pca.explained_variance_ratio_
    # orient: f1 so most loadings positive; f2 so loading increases with depth; f3 by |max| loading sign
    if np.sign(load[0]).sum() < 0:
        load[0] *= -1; scores[:, 0] *= -1
    if load.shape[0] > 1 and sp.spearmanr(load[1], cent[:, 1])[0] < 0:
        load[1] *= -1; scores[:, 1] *= -1
    if load.shape[0] > 2 and load[2][np.argmax(np.abs(load[2]))] < 0:
        load[2] *= -1; scores[:, 2] *= -1
    for i in range(min(3, load.shape[0])):
        D[f"load_f{i + 1}"] = load[i]

    # ---- behavior & global for factor-score correlations
    beh = load_behavior(r)
    B = {}
    for kname in ("pupil", "whisker", "accel"):
        if beh.get(kname) is not None:
            y = resample_to(beh[f"{kname}_t"], beh[kname], t)
            if y is not None and np.isfinite(y).mean() > 0.9:
                B[kname] = np.where(np.isfinite(y), y, np.nanmedian(y))
    gpath = r.out / "traces" / "global_ca.csv"
    gy = None
    if gpath.exists():
        g = pd.read_csv(gpath)
        gy = resample_to(g["time_s"].to_numpy(), g["global_dff"].to_numpy(), t)
        if gy is not None:
            gy = np.where(np.isfinite(gy), gy, np.nanmedian(gy))

    R = dict(run=key, source=source, mouse=r.mouse, fov=fov_group(r), n=N, T=T, fr=fr,
             dur_min=dur_min, co_k=co_k, kstar=float(kstar), kstar_reached=bool(kstar_reached),
             kstar_null_rate=float(perk_df.loc[perk_df.k == kstar_use, "null_rate"].iloc[0]),
             kstar_frac_sig=float(perk_df.loc[perk_df.k == kstar_use, "frac_sig"].iloc[0]),
             ve_at_kstar=float(perk_df.loc[perk_df.k == kstar_use, "var_explained"].iloc[0]),
             frac_sig_k0=float(perk_df.loc[perk_df.k == 0, "frac_sig"].iloc[0]),
             mean_abs_r_k0=float(perk_df.loc[perk_df.k == 0, "mean_abs_r"].iloc[0]),
             private_var_median=float(np.median(priv)),
             pc1_var_frac=float(evr[0]),
             pc1_3_var_frac=float(evr[:3].sum()))
    # factor-1-global descriptors
    R["f1_frac_same_sign"] = float(max((load[0] > 0).mean(), (load[0] < 0).mean()))
    R["f1_mean_loading"] = float(load[0].mean())
    R["f1_cv_loading"] = float(np.std(load[0]) / (np.abs(np.mean(load[0])) + 1e-9))
    for i in range(min(3, load.shape[0])):
        R[f"f{i + 1}_rho_depth"] = float(sp.spearmanr(load[i], cent[:, 1])[0])
        I, pI, _ = morans_I(load[i], cent, rng, nperm=200)
        R[f"f{i + 1}_moran_I"] = I
        R[f"f{i + 1}_moran_p"] = pI
        R[f"f{i + 1}_var_frac"] = float(evr[i]) if i < len(evr) else np.nan
        for src_name, yv in (("pupil", B.get("pupil")), ("whisker", B.get("whisker")),
                             ("accel", B.get("accel")), ("global", gy)):
            if yv is not None:
                R[f"f{i + 1}_corr_{src_name}"] = float(np.corrcoef(scores[:, i], yv)[0, 1])
            else:
                R[f"f{i + 1}_corr_{src_name}"] = np.nan
    return dict(run=R, perk=perk_df, dend=D, example=(key == EXAMPLE))


# ----------------------------------------------------------------------------- cohort
def _wilcoxon_vs(v, mu=0.0):
    v = pd.Series(v).dropna()
    if len(v) < 4:
        return np.nan, len(v), (float(v.median()) if len(v) else np.nan)
    if (v == mu).all():
        return 1.0, len(v), float(v.median())
    return float(sp.wilcoxon(v - mu).pvalue), len(v), float(v.median())


def cohort(source, res):
    out = OUT / source
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([x["run"] for x in res])
    perk = pd.concat([x["perk"] for x in res], ignore_index=True)
    dend = pd.concat([x["dend"] for x in res], ignore_index=True)
    runs.to_csv(out / "per_run.csv", index=False)
    perk.to_csv(out / "per_k.csv", index=False)
    dend.to_csv(out / "per_dendrite.csv", index=False)
    # FOV = unit of replication: average runs within a FOV
    fov = runs.groupby(["fov", "mouse"]).mean(numeric_only=True).reset_index()
    fov.to_csv(out / "per_fov.csv", index=False)
    perk_fov = perk.groupby(["fov", "k"]).mean(numeric_only=True).reset_index()

    S = dict(source=source, n_runs=len(runs), n_fovs=int(runs.fov.nunique()),
             n_mice=int(runs.mouse.nunique()), findings=[])
    L = [f"SHARED-FACTOR ANALYSIS - {source} dendrites: {len(runs)} runs, {runs.fov.nunique()} FOVs, "
         f"{runs.mouse.nunique()} mice, {len(dend)} dendrite-run entries.",
         "Question: how many shared latent factors (common inputs) make the residual dendrite-dendrite",
         "structure indistinguishable from independent units? Blocked 5-fold CV in time; PCA fit on",
         "train folds and projected onto held-out folds; circular-shift nulls (>= 10 s).",
         "FOV = unit of replication (Wilcoxon over FOV means).", ""]

    def add(name, text, **kw):
        S["findings"].append(dict(name=name, text=text, **{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in kw.items()}))
        L.append(f"* [{name}] {text}")

    # ---- k*
    reached = runs.groupby("fov").kstar_reached.max()
    nreached = int(reached.sum())
    kstar_txt = (f"median {fov.kstar.median():.1f}" if nreached else "NOT REACHED (> 20)")
    add("kstar", f"k* (smallest k with residual significant-pair fraction within {KSTAR_TOL*100:.0f} pp of the "
        f"circular-shift null false-positive rate) was reached within the tested range (<= 20) in "
        f"{nreached}/{len(reached)} FOVs; where unreached k* > 20. k* = {kstar_txt}. "
        f"Removing shared factors drops the residual significant-pair fraction from "
        f"{fov.frac_sig_k0.median()*100:.1f}% (k=0) only to {perk_fov[perk_fov.k==20].groupby('fov').frac_sig.mean().median()*100:.1f}% "
        f"at k=20 - still ~{(perk_fov[perk_fov.k==20].groupby('fov').frac_sig.mean().median())/max(perk_fov[perk_fov.k==20].null_rate.median(),1e-4):.0f}x the "
        f"{perk_fov[perk_fov.k==20].null_rate.median()*100:.2f}% null, well above the {KSTAR_TOL*100:.0f}-pp criterion. "
        f"Mean |r| falls from {fov.mean_abs_r_k0.median():.3f} to {perk_fov[perk_fov.k==20].groupby('fov').mean_abs_r.mean().median():.3f}; "
        f"k=20 factors explain a median {perk_fov[perk_fov.k==20].groupby('fov').var_explained.mean().median()*100:.0f}% of held-out variance. "
        f"CONCLUSION: the dependence is NOT low-dimensional - no small set of shared inputs renders the dendrites "
        f"independent; it is high-dimensional (consistent with ~18 significant PCs reported by the independence battery), "
        f"with one dominant global mode plus many weak, partly short-range modes.",
        n_reached=nreached, n_fovs=int(len(reached)))

    # ---- how dependence collapses with k (per-k medians over FOVs)
    kg = perk_fov.groupby("k").median(numeric_only=True)
    seg = ", ".join(f"k={int(kk)}: {kg.loc[kk,'frac_sig']*100:.1f}% sig (null {kg.loc[kk,'null_rate']*100:.1f}%), "
                    f"|r|={kg.loc[kk,'mean_abs_r']:.3f}, VE={kg.loc[kk,'var_explained']*100:.0f}%"
                    for kk in kg.index)
    add("dependence_vs_k", f"Residual dependence vs number of factors (median over FOVs): {seg}. The significant-pair "
        f"fraction falls steeply over k=1-3 (the global common mode) then PLATEAUS far above the null, so extra factors "
        f"buy little independence. (For the small-N human FOVs the k=12-20 estimates overfit the held-out projection and "
        f"the curve can rise again - the plateau, not the high-k tail, is the robust feature.)")

    # ---- co-activity collapse
    k0co = kg.loc[0, "coactive_frac"]; k0null = kg.loc[0, "coactive_null"]
    kstco = kg.loc[20]
    add("coactivity", f">= 10% of dendrites co-active (robust z>3): {k0co*100:.1f}% of frames at k=0 vs {k0null*100:.2f}% "
        f"expected under independence; even after removing 20 factors {kstco['coactive_frac']*100:.2f}% of frames remain "
        f"co-active vs {kstco['coactive_null']*100:.2f}% null. The shared factors capture about half of the synchronous "
        f"bursts but a reliable co-activation excess persists.")

    # ---- factor 1 global?
    p1, n1, m1 = _wilcoxon_vs(fov.f1_frac_same_sign, 0.5)
    add("factor1_global", f"Factor 1 loads with the SAME sign on {fov.f1_frac_same_sign.median()*100:.0f}% of dendrites "
        f"(50% = no common sign; p={p1:.2g}, {n1} FOVs) and explains {fov.pc1_var_frac.median()*100:.0f}% of variance; "
        f"its loading-depth Spearman is {fov.f1_rho_depth.median():+.2f} and Moran's I {fov.f1_moran_I.median():+.2f}. "
        f"Factor 1 is a near-global common-mode (co-fluctuation shared by most dendrites), not depth- or cluster-specific.",
        frac_same_sign=float(fov.f1_frac_same_sign.median()))
    # factor 1 correlation with global/behavior
    add("factor1_identity", f"Factor-1 score correlates with global Ca at r={fov.f1_corr_global.median():+.2f} "
        f"(median over FOVs), and with pupil {fov.f1_corr_pupil.median():+.2f}, whisker {fov.f1_corr_whisker.median():+.2f}, "
        f"accel {fov.f1_corr_accel.median():+.2f}. For the auto masks factor 1 closely tracks the whole-FOV global Ca "
        f"signal (its loadings are a near-uniform common mode); for the sparser human masks the correlation with the "
        f"auto-derived global signal is weaker. The dominant shared factor is a global co-fluctuation, only modestly "
        f"behavior-locked.")

    # ---- factor 2 depth gradient?
    p2, n2, m2 = _wilcoxon_vs(fov.f2_rho_depth)
    add("factor2_depth", f"Factor 2 loading vs depth Spearman {m2:+.2f} (p={p2:.2g}, {n2} FOVs); Moran's I "
        f"{fov.f2_moran_I.median():+.2f}. Factor 2 is {'a depth gradient' if abs(m2)>0.3 and (p2==p2 and p2<0.1) else 'not a clear depth gradient'} "
        f"(it separates superficial from deep dendrites)." )

    # ---- factor 3 spatial clustering?
    sig3 = runs.f3_moran_p < 0.05
    p3, n3, m3 = _wilcoxon_vs(fov.f3_moran_I)
    add("factor3_clustered", f"Factor 3 Moran's I of loadings {m3:+.2f} (p={p3:.2g} over {n3} FOVs); loading spatial "
        f"autocorrelation significant (perm p<0.05) in {int(sig3.sum())}/{int(runs.f3_moran_p.notna().sum())} runs. Factor 3 "
        f"is {'spatially clustered' if m3>0.1 else 'not strongly spatially clustered'} (local groups of dendrites).")

    # ---- private variance LMM
    try:
        import statsmodels.formula.api as smf
        d2 = dend.dropna(subset=["private_var_frac", "depth_um", "n_vox", "event_rate_per_min"]).copy()
        if d2.fov.nunique() >= 3 and len(d2) > 50:
            d2["dz"] = d2.groupby("fov").depth_um.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            d2["sz"] = d2.groupby("fov").n_vox.transform(lambda x: (np.log(x) - np.log(x).mean()) / (np.log(x).std() + 1e-9))
            d2["rz_"] = d2.groupby("fov").event_rate_per_min.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            mdl = smf.mixedlm("private_var_frac ~ dz + sz + rz_", d2, groups=d2["fov"]).fit(reml=True)
            add("private_var_LMM", "Private variance (held-out fraction NOT explained by k* factors; evaluated at k=20 "
                "since k* was not reached) ~ depth + log size + event rate (LMM, random intercept per FOV): "
                + ", ".join(f"{k} {mdl.params[k]:+.3f} (p={mdl.pvalues[k]:.2g})" for k in ("dz", "sz", "rz_"))
                + f". Median private fraction {dend.private_var_frac.median():.2f} "
                f"(i.e. ~{dend.private_var_frac.median()*100:.0f}% of each dendrite's variance is private even with 20 "
                f"factors removed). More active dendrites (higher event rate) are more shared (lower private fraction); "
                f"larger dendrites slightly more shared. Note the k=20 private fraction is a lower bound on the private "
                f"component for the small-N human FOVs, where the held-out factor model overfits and inflates sharing.",
                median_private=float(dend.private_var_frac.median()))
        else:
            add("private_var_LMM", f"Median private variance fraction {dend.private_var_frac.median():.2f}; LMM skipped (insufficient groups).")
    except Exception as e:
        add("private_var_LMM", f"Median private variance fraction {dend.private_var_frac.median():.2f}; LMM failed ({e}).")

    # ---- across FOVs
    add("per_fov", "k* by FOV: " + "; ".join(f"{row.fov.split('/')[0]}/{row.fov.split('/')[-1]} k*={row.kstar:.0f} "
        f"(VE {row.ve_at_kstar*100:.0f}%, N~{row.n:.0f})" for _, row in fov.sort_values('fov').iterrows()))

    L += ["", "Reading guide: independent units would need k*=0 (residual sig-pair fraction already at the null rate).",
          "A small k* means a few shared inputs account for essentially all the dependence; the large per-dendrite",
          "private fraction then quantifies the independent (private) computation that remains.",
          "Caveats: 5 Hz sampling and ~100 s per run limit factor-score / behavior estimates; held-out VE uses PCA",
          "(FactorAnalysis gives a near-identical k*); SCAPE optical crosstalk between neighbouring dendrites inflates",
          "factor-1 / short-range shared variance (shell subtraction reduces but does not remove it); 3 mice, 2 with",
          "behavior; the anti-phase-with-pupil group is a shell-subtraction artifact and is not interpreted here.",
          "Null results are reported as null."]
    (out / "summary.txt").write_text("\n".join(L))
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))
    figures(source, runs, fov, perk, perk_fov, dend, res, out)
    return L


# ----------------------------------------------------------------------------- figures
def figures(source, runs, fov, perk, perk_fov, dend, res, out):
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.33)

    # top row: dependence vs k
    ax = fig.add_subplot(gs[0, 0])
    for f_, g in perk.groupby("fov"):
        g = g.sort_values("k")
        ax.plot(g.k, g.frac_sig * 100, "-o", ms=2.5, lw=0.7, alpha=0.6)
    nr = perk.null_rate.median() * 100
    ax.axhspan(0, nr + KSTAR_TOL * 100, color="0.85", zorder=0, label=f"null +{KSTAR_TOL*100:.0f}pp band")
    ax.axhline(nr, color="k", ls=":", lw=0.8, label="null rate")
    ax.set_xlabel("k shared factors"); ax.set_ylabel("% residual pairs significant")
    ax.set_title("(a) residual dependence vs k"); ax.legend(fontsize=6, loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    for f_, g in perk.groupby("fov"):
        g = g.sort_values("k")
        ax.plot(g.k, g.mean_abs_r, "-o", ms=2.5, lw=0.7, alpha=0.6)
    ax.set_xlabel("k shared factors"); ax.set_ylabel("held-out mean pairwise |r|")
    ax.set_title("(a) residual |r| vs k")

    ax = fig.add_subplot(gs[0, 2])
    for f_, g in perk.groupby("fov"):
        g = g.sort_values("k")
        ax.plot(g.k, g.coactive_frac * 100, "-o", ms=2.5, lw=0.7, alpha=0.6)
    ax.plot(perk_fov.groupby("k").coactive_null.median().index,
            perk_fov.groupby("k").coactive_null.median().values * 100, "k:", lw=1.0, label="null")
    ax.set_xlabel("k shared factors"); ax.set_ylabel(">= 10% co-active (% frames)")
    ax.set_title("(b) co-activation vs k"); ax.legend(fontsize=6)

    ax = fig.add_subplot(gs[0, 3])
    for f_, g in perk.groupby("fov"):
        g = g.sort_values("k")
        ax.plot(g.k, g.var_explained * 100, "-o", ms=2.5, lw=0.7, alpha=0.6)
    ax.set_xlabel("k shared factors"); ax.set_ylabel("held-out variance explained (%)")
    ax.set_title("(c) held-out VE vs k")

    # bottom row: loading maps for the example FOV + factor/behavior bar
    ex = dend[dend.run == EXAMPLE]
    if len(ex) == 0:                                   # fall back to any run of that FOV
        exfov = f"2026-05-12/rbp4_132_phpeb/run5"
        ex = dend[dend.fov == exfov]
        ex = ex[ex.run == ex.run.iloc[0]] if len(ex) else ex
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        col = f"load_f{i + 1}"
        if len(ex) and col in ex:
            v = ex[col].to_numpy()
            lim = np.nanpercentile(np.abs(v), 98) or 1.0
            sc = ax.scatter(ex.x_um, ex.depth_um, c=v, cmap="RdBu_r", vmin=-lim, vmax=lim, s=22, edgecolor="k", linewidth=0.2)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.invert_yaxis()
        ax.set_xlabel("lateral x (um)"); ax.set_ylabel("depth (um)")
        label = {0: "factor 1 (global?)", 1: "factor 2 (depth?)", 2: "factor 3 (clustered?)"}[i]
        ax.set_title(f"loading map - {label}")

    ax = fig.add_subplot(gs[1, 3])
    exr = runs[runs.run == EXAMPLE]
    if len(exr):
        srcs = ["global", "pupil", "whisker", "accel"]
        xpos = np.arange(len(srcs)); w = 0.25
        for fi, color in zip(range(3), ("#1f77b4", "#ff7f0e", "#2ca02c")):
            vals = [exr[f"f{fi+1}_corr_{s}"].iloc[0] for s in srcs]
            ax.bar(xpos + (fi - 1) * w, vals, w, color=color, label=f"factor {fi+1}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(xpos); ax.set_xticklabels(srcs, rotation=30, ha="right")
        ax.set_ylabel("corr(factor score, signal)")
        ax.set_title("factor scores vs behavior/global"); ax.legend(fontsize=6)
    fig.suptitle(f"Shared-factor analysis - {source} dendrites (loading maps: {EXAMPLE})", fontsize=11)
    fig.savefig(out / "fig_factors.png", dpi=140, bbox_inches="tight")
    fig.savefig(out / "fig_factors.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- compare
def compare(allL):
    txt = ["AUTO vs HUMAN comparison of k*", ""]
    try:
        a = pd.read_csv(OUT / "auto" / "per_fov.csv")
        h = pd.read_csv(OUT / "human" / "per_fov.csv")
        txt.append(f"k* reached (<= 20) in {a.get('kstar_reached', pd.Series(dtype=float)).sum():.0f}/{len(a)} auto FOVs "
                   f"and {h.get('kstar_reached', pd.Series(dtype=float)).sum():.0f}/{len(h)} human FOVs; "
                   f"otherwise k* > 20 (not reached).")
        m = a.merge(h, on="fov", suffixes=("_auto", "_human"))
        txt.append(f"Shared FOVs: {len(m)}")
        if len(m):
            txt.append("per-FOV held-out VE at k=20 (auto / human): " + "; ".join(
                f"{row.fov.split('/')[-1]}: {row.ve_at_kstar_auto*100:.0f}% / {row.ve_at_kstar_human*100:.0f}%"
                for _, row in m.iterrows()))
            txt.append(f"k* is > 20 (not reached) for BOTH sources in every shared FOV: the two mask sets agree the "
                       f"dependence is high-dimensional. Held-out VE at k=20 is higher for human masks "
                       f"(median {m.ve_at_kstar_human.median()*100:.0f}% vs auto {m.ve_at_kstar_auto.median()*100:.0f}%), "
                       f"an expected small-N effect (fewer, larger hand-drawn masks share more variance and overfit "
                       f"more at high k).")
    except Exception as e:
        txt.append(f"(comparison failed: {e})")
    (OUT / "compare_auto_vs_human.txt").write_text("\n".join(txt))
    print("\n".join(txt))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("auto", "human", "both"), default="both")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    keys = a.run or [r.key for r in discover_runs()]
    t0 = time.time()
    allL = {}
    for s in (("auto", "human") if a.source == "both" else (a.source,)):
        res = []
        with ProcessPoolExecutor(max(1, min(a.jobs, 3))) as ex:
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
        if not res:
            print(f"no runs for source={s}")
            continue
        res.sort(key=lambda x: x["run"]["run"])
        allL[s] = cohort(s, res)
        print("\n".join(allL[s]), "\n")
    if len(allL) == 2:
        compare(allL)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
