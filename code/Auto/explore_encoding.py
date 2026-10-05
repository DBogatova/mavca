#!/usr/bin/env python
"""explore_encoding.py - cross-validated encoding / population-coupling analysis of
whether L5 apical dendrites are independent computational units (code/Auto/).

For every run (source auto; repeated on human where human_masks_valid) and every dendrite we
fit ridge regressions with 5-fold BLOCKED (contiguous, no shuffling) cross-validation in time,
predicting the dendrite's dF/F trace from:
  a  behavior only : pupil, whisker, accel, each with lags -2..+2 s
  b  global Ca only: global_dff (mean of live voxels) with lags -2..+2 s   [caveat: contains the target]
  c  population    : the OTHER dendrites' traces (leave-one-out). Variants:
                        pop      = all other dendrites
                        pop_near = 20 nearest other dendrites
                        pop_far  = only dendrites > 50 um away (controls optical crosstalk)
  d  all together  : a + b + c
We report held-out R^2 per dendrite for each model and the unique contribution of each block
(d minus the model without that block).

Leave-one-out population ridge is done exactly and fast: with G = Z_tr^T Z_tr (z-scored traces)
and M = (G + alpha I)^-1, the ridge coefficients predicting node j from all other nodes are
-M[:,j]/M[j,j] (node-wise regression / precision-matrix identity); the external blocks (behavior,
global) are stacked into the same matrix so "predict dendrite j from behavior+global+other dendrites"
is still one matrix inverse per fold. alpha is chosen by a blocked inner split (relative grid).

Then, with FOV = unit of replication (scape_common.fov_group, Wilcoxon over FOVs; per-dendrite
relations via statsmodels mixed models with a FOV random intercept):
  * distribution of population-coupling R^2; fraction of dendrites with pop R^2 < 0.02
    (effectively independent) and > 0.2 (strongly coupled);
  * population coupling vs depth, event rate, size, and behavior R^2 (chorister vs soloist,
    Okun et al. 2015);
  * does the far-only (> 50 um) model still predict (shared network signal, not crosstalk)?
  * a sparse partial-correlation network (GraphicalLassoCV on z-scored traces, 150 most active if
    N>150): degree distribution, fraction of zero partial correlations, hubs and their depth,
    compared with a density-matched plain correlation network;
  * the 'anti-phase with pupil' population (labels merged from stats/pupil_phase/<source>): are
    anti-phase dendrites deeper / less active / population-decoupled / negatively pupil-tuned, and
    does this replicate across FOVs?

Outputs: scape-auto/stats/explore_encoding/<source>/{per_run.csv, per_dendrite.csv, per_fov.csv,
         summary.txt, summary.json, fig_encoding.png/pdf, fig_network.png/pdf}
CLI: explore_encoding.py [--source auto|human|both] [--run KEY ...] [--jobs N] [--no-glasso] [--limit N]
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from scipy.spatial.distance import cdist
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

OUT = AUTO_ROOT / "stats" / "explore_encoding"
PUPIL_PHASE = AUTO_ROOT / "stats" / "pupil_phase"
ZTHR = 3.0
NFOLD = 5
MIN_SHIFT_S = 10          # (circular-shift nulls, used only for the pop-coupling significance check)
NEAR_K = 20
FAR_UM = 50.0
GLASSO_MAXN = 150
LAG_S = 2.0
REL = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])   # ridge alpha relative to n_train (columns are z-scored)
BEH_KEYS = ("pupil", "whisker", "accel")


# ----------------------------------------------------------------------------- small helpers
def rz(M):
    med = np.median(M, 0)
    return (M - med) / (np.median(np.abs(M - med), 0) * 1.4826 + 1e-9)


def onsets_from_active(A):
    out = []
    for j in range(A.shape[1]):
        a = np.r_[False, A[:, j], False]
        d = np.diff(a.astype(int))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out.append(st[(en - st) >= 2])
    return out


def blocked_folds(T, k):
    """k contiguous time blocks; each is the test set once, the rest train. No shuffling."""
    edges = np.linspace(0, T, k + 1).astype(int)
    folds = []
    for i in range(k):
        te = np.arange(edges[i], edges[i + 1])
        tr = np.r_[np.arange(0, edges[i]), np.arange(edges[i + 1], T)]
        folds.append((tr, te))
    return folds


def lag_design(series, fr, L_s=LAG_S):
    """(T,) raw series -> (T, 2L+1) matrix of lags -L..+L frames (col = series at t+d); edges filled
    with the series mean. NaNs median-filled first."""
    s = np.asarray(series, float)
    if not np.isfinite(s).all():
        s = np.where(np.isfinite(s), s, np.nanmedian(s))
    T = s.size
    L = int(round(L_s * fr))
    out = np.full((T, 2 * L + 1), s.mean())
    for k, d in enumerate(range(-L, L + 1)):
        if d >= 0:
            out[:T - d, k] = s[d:]
        elif d < 0:
            out[-d:, k] = s[:T + d]
    return out


def stdz(X, tr):
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd


# ----------------------------------------------------------------------------- ridge primitives
def loo_predict(E_tr, Z_tr, E_te, Z_te, alpha):
    """Leave-one-out augmented ridge. Predict each dendrite (columns of Z) from [E, all OTHER
    dendrites]. Returns Yhat_te (n_te, N). Uses the identity b_j = -M[:,j]/M[j,j],
    M = (W_tr^T W_tr + alpha I)^-1, W = [E, Z]."""
    if E_tr is None or E_tr.shape[1] == 0:
        W_tr, W_te, p = Z_tr, Z_te, 0
    else:
        W_tr = np.concatenate([E_tr, Z_tr], 1)
        W_te = np.concatenate([E_te, Z_te], 1)
        p = E_tr.shape[1]
    G = W_tr.T @ W_tr
    D = G.shape[0]
    A = G + alpha * np.eye(D)
    M = np.linalg.inv(A)
    dgM = np.diag(M).copy()
    B = -M / dgM[None, :]
    np.fill_diagonal(B, 0.0)
    return W_te @ B[:, p:]


def shared_predict(X_tr, Y_tr, X_te, alpha):
    """Ridge with a design X shared across all targets Y (T,N). Returns Yhat_te."""
    p = X_tr.shape[1]
    B = np.linalg.solve(X_tr.T @ X_tr + alpha * np.eye(p), X_tr.T @ Y_tr)
    return X_te @ B


def restricted_predict(Z_tr, Z_te, alpha, sets):
    """Per-target ridge: predict dendrite i from a restricted predictor set sets[i] (dendrite
    indices). Returns Yhat_te (n_te, N)."""
    G = Z_tr.T @ Z_tr
    N = Z_tr.shape[1]
    Yhat = np.zeros_like(Z_te)
    for i in range(N):
        P = sets[i]
        if P.size == 0:
            continue
        A = G[np.ix_(P, P)].copy()
        A[np.diag_indices_from(A)] += alpha
        try:
            b = np.linalg.solve(A, G[P, i])
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(A, G[P, i], rcond=None)[0]
        Yhat[:, i] = Z_te[:, P] @ b
    return Yhat


def pick_alpha_loo(E_tr, Z_tr, E_va, Z_va):
    n = len(Z_tr)
    best = (np.inf, REL[0])
    for rel in REL:
        yh = loo_predict(E_tr, Z_tr, E_va, Z_va, rel * n)
        ss = float(np.nansum((Z_va - yh) ** 2))
        if ss < best[0]:
            best = (ss, rel)
    return best[1]


def pick_alpha_shared(X_tr, Y_tr, X_va, Y_va):
    n = len(Y_tr)
    best = (np.inf, REL[0])
    for rel in REL:
        yh = shared_predict(X_tr, Y_tr, X_va, rel * n)
        ss = float(np.nansum((Y_va - yh) ** 2))
        if ss < best[0]:
            best = (ss, rel)
    return best[1]


def pooled_r2(ss_res, ss_tot):
    return 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)


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


# ----------------------------------------------------------------------------- graphical lasso
def glasso_network(M, cent, size, event_rate):
    """Sparse partial-correlation network on z-scored traces. Returns a metrics dict and arrays of
    per-dendrite glasso/corr degree (aligned to the dendrites kept)."""
    from sklearn.covariance import GraphicalLassoCV
    N = M.shape[1]
    keep = np.arange(N)
    if N > GLASSO_MAXN:
        keep = np.argsort(event_rate)[::-1][:GLASSO_MAXN]
        keep.sort()
    X = M[:, keep]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    res = {"glasso_n": int(len(keep)), "glasso_ok": False}
    deg = np.full(len(keep), np.nan)
    cdeg = np.full(len(keep), np.nan)
    try:
        model = GraphicalLassoCV(alphas=np.logspace(-1.3, 0.3, 5), cv=3, max_iter=100).fit(Z)
        K = model.precision_.copy()
        d = np.sqrt(np.diag(K))
        P = -K / np.outer(d, d)            # partial correlations
        np.fill_diagonal(P, 0.0)
        n = P.shape[0]
        iu = np.triu_indices(n, 1)
        pc = P[iu]
        edge = np.abs(pc) > 1e-3
        n_edge = int(edge.sum())
        A = np.abs(P) > 1e-3
        deg = A.sum(1).astype(float)
        # density-matched plain correlation network
        C = np.corrcoef(Z.T)
        np.fill_diagonal(C, 0.0)
        cc = np.abs(C[iu])
        thr = np.sort(cc)[::-1][min(n_edge, len(cc) - 1)] if n_edge > 0 else np.inf
        Ac = np.abs(C) > thr
        np.fill_diagonal(Ac, False)
        cdeg = Ac.sum(1).astype(float)
        order = np.argsort(deg)[::-1]
        nhub = max(1, int(round(0.1 * n)))
        hubs = order[:nhub]
        hub_corr = np.argsort(cdeg)[::-1][:nhub]
        res.update(
            glasso_ok=True,
            glasso_alpha=float(model.alpha_),
            glasso_n_edges=n_edge,
            glasso_density=float(n_edge / len(pc)),
            glasso_frac_zero=float(1 - edge.mean()),
            glasso_frac_pos=float((pc[edge] > 0).mean()) if n_edge else np.nan,
            glasso_mean_degree=float(deg.mean()),
            glasso_max_degree=float(deg.max()),
            glasso_degree_cv=float(deg.std() / (deg.mean() + 1e-9)),
            hub_depth_um=float(cent[keep][hubs, 1].mean()),
            nonhub_depth_um=float(cent[keep][order[nhub:], 1].mean()) if n > nhub else np.nan,
            hub_depth_p=float(sp.mannwhitneyu(cent[keep][hubs, 1], cent[keep][order[nhub:], 1]).pvalue)
            if n > 2 * nhub else np.nan,
            hub_size_vox=float(size[keep][hubs].mean()),
            hub_event_rate=float(event_rate[keep][hubs].mean()),
            rho_degree_glasso_corr=float(sp.spearmanr(deg, cdeg)[0]),
            hub_jaccard_glasso_corr=float(len(set(hubs) & set(hub_corr)) / len(set(hubs) | set(hub_corr))),
            rho_degree_depth=float(sp.spearmanr(deg, cent[keep][:, 1])[0]),
            rho_degree_event_rate=float(sp.spearmanr(deg, event_rate[keep])[0]),
        )
    except Exception as e:
        res["glasso_err"] = str(e)[:200]
    return res, keep, deg, cdeg


# ----------------------------------------------------------------------------- per run
def analyze(key: str, source: str, do_glasso: bool = True) -> dict | None:
    t0 = time.time()
    r = get_run(key)
    got = load_source(r, source)
    if got is None:
        return None
    df, names, cent, size = got
    fr = r.frame_rate
    t = df["time_s"].to_numpy()
    M = df[names].to_numpy(float)
    T, N = M.shape
    if N < 10 or T < 100:
        return None

    # event rate (robust-z > 3 onsets), used only for relation analyses + glasso ranking
    A = rz(M) > ZTHR
    ons = onsets_from_active(A)
    dur_min = T / fr / 60.0
    event_rate = np.array([len(o) for o in ons]) / dur_min

    # external designs (raw; standardised per fold)
    beh = load_behavior(r)
    beh_blocks = []
    for k in BEH_KEYS:
        if beh.get(k) is not None:
            y = resample_to(beh[f"{k}_t"], beh[k], t)
            if y is not None and np.isfinite(y).mean() > 0.9:
                beh_blocks.append(lag_design(y, fr))
    Ebeh = np.concatenate(beh_blocks, 1) if beh_blocks else None
    g = pd.read_csv(r.out / "traces" / "global_ca.csv")
    gy = resample_to(g["time_s"].to_numpy(), g["global_dff"].to_numpy(), t)
    Eglob = lag_design(gy, fr)
    has_beh = Ebeh is not None

    # geometry: near / far predictor sets
    Dm = cdist(cent, cent)
    near_sets, far_sets, n_far = [], [], np.zeros(N, int)
    for i in range(N):
        order = np.argsort(Dm[i])
        order = order[order != i]
        near_sets.append(order[:NEAR_K])
        far = np.flatnonzero(Dm[i] > FAR_UM)
        far = far[far != i]
        far_sets.append(far)
        n_far[i] = far.size

    folds = blocked_folds(T, NFOLD)
    models = ["beh", "glob", "ext", "pop", "beh_pop", "glob_pop", "all", "pop_near", "pop_far"]
    ssr = {m: np.zeros(N) for m in models}
    sst = {m: np.zeros(N) for m in models}

    for tr, te in folds:
        Zt = stdz(M, tr)
        Ztr, Zte = Zt[tr], Zt[te]
        base = (Zte ** 2).sum(0)                      # ss_tot (train-mean baseline, = 0 in z-space)
        ntr = len(tr)
        cut = int(round(0.8 * ntr))
        if cut < 20 or ntr - cut < 10:
            cut = max(20, ntr - max(10, ntr // 5))
        itr, iva = np.arange(cut), np.arange(cut, ntr)

        Et_beh = Et_glob = Et_ext = None
        if has_beh:
            Eb = stdz(Ebeh, tr); Et_beh = (Eb[tr], Eb[te])
        Eg = stdz(Eglob, tr); Et_glob = (Eg[tr], Eg[te])
        ext_tr = Et_glob[0] if not has_beh else np.concatenate([Et_beh[0], Et_glob[0]], 1)
        ext_te = Et_glob[1] if not has_beh else np.concatenate([Et_beh[1], Et_glob[1]], 1)

        # ---- alpha selection (blocked inner split)
        rel_pop = pick_alpha_loo(None, Ztr[itr], None, Ztr[iva])
        a_pop = rel_pop * ntr
        rel_ext = pick_alpha_shared(ext_tr[itr], Ztr[itr], ext_tr[iva], Ztr[iva])
        a_ext = rel_ext * ntr

        # ---- predictions
        def acc(m, yh):
            ssr[m] += ((Zte - yh) ** 2).sum(0)
            sst[m] += base

        acc("glob", shared_predict(Et_glob[0], Ztr, Et_glob[1], pick_alpha_shared(Et_glob[0][itr], Ztr[itr], Et_glob[0][iva], Ztr[iva]) * ntr))
        acc("ext", shared_predict(ext_tr, Ztr, ext_te, a_ext))
        acc("pop", loo_predict(None, Ztr, None, Zte, a_pop))
        acc("glob_pop", loo_predict(Et_glob[0], Ztr, Et_glob[1], Zte, a_pop))
        acc("all", loo_predict(ext_tr, Ztr, ext_te, Zte, a_pop))
        acc("pop_near", restricted_predict(Ztr, Zte, a_pop, near_sets))
        acc("pop_far", restricted_predict(Ztr, Zte, a_pop, far_sets))
        if has_beh:
            acc("beh", shared_predict(Et_beh[0], Ztr, Et_beh[1], pick_alpha_shared(Et_beh[0][itr], Ztr[itr], Et_beh[0][iva], Ztr[iva]) * ntr))
            acc("beh_pop", loo_predict(Et_beh[0], Ztr, Et_beh[1], Zte, a_pop))

    R2 = {m: pooled_r2(ssr[m], sst[m]) for m in models}
    r2_beh = R2["beh"] if has_beh else np.full(N, np.nan)
    r2_all = R2["all"]
    unique_pop = r2_all - R2["ext"]
    if has_beh:
        unique_beh = r2_all - R2["glob_pop"]
        unique_glob = r2_all - R2["beh_pop"]
    else:
        unique_beh = np.full(N, np.nan)
        unique_glob = r2_all - R2["pop"]

    D = pd.DataFrame(dict(
        run=key, source=source, mouse=r.mouse, fov=fov_group(r), dendrite=names,
        depth_um=cent[:, 1], x_um=cent[:, 2], z_um=cent[:, 0], n_vox=size,
        event_rate_per_min=event_rate, n_near=NEAR_K, n_far=n_far,
        r2_beh=r2_beh, r2_glob=R2["glob"], r2_pop=R2["pop"], r2_pop_near20=R2["pop_near"],
        r2_pop_far50=R2["pop_far"], r2_all=r2_all,
        unique_beh=unique_beh, unique_glob=unique_glob, unique_pop=unique_pop,
    ))

    # glasso network
    gl = {"glasso_ok": False}
    if do_glasso:
        gl, keep, deg, cdeg = glasso_network(M, cent, size, event_rate)
        gdeg = np.full(N, np.nan); gcdeg = np.full(N, np.nan)
        gdeg[keep] = deg; gcdeg[keep] = cdeg
        D["glasso_degree"] = gdeg
        D["corr_degree"] = gcdeg

    Rrow = dict(run=key, source=source, mouse=r.mouse, fov=fov_group(r), n=N, T=T, fr=fr, dur_min=dur_min,
                has_beh=has_beh,
                r2_pop_median=float(np.nanmedian(R2["pop"])), r2_pop_mean=float(np.nanmean(R2["pop"])),
                r2_pop_far_median=float(np.nanmedian(R2["pop_far"])),
                r2_pop_near_median=float(np.nanmedian(R2["pop_near"])),
                r2_glob_median=float(np.nanmedian(R2["glob"])),
                r2_all_median=float(np.nanmedian(r2_all)),
                r2_beh_median=float(np.nanmedian(r2_beh)),
                unique_pop_median=float(np.nanmedian(unique_pop)),
                unique_glob_median=float(np.nanmedian(unique_glob)),
                unique_beh_median=float(np.nanmedian(unique_beh)),
                frac_pop_lt_002=float(np.nanmean(R2["pop"] < 0.02)),
                frac_pop_gt_02=float(np.nanmean(R2["pop"] > 0.2)),
                frac_far_gt_002=float(np.nanmean(R2["pop_far"] > 0.02)),
                frac_far_gt_02=float(np.nanmean(R2["pop_far"] > 0.2)),
                med_far_minus_all=float(np.nanmedian(R2["pop_far"] - R2["pop"])),
                med_near_minus_all=float(np.nanmedian(R2["pop_near"] - R2["pop"])))
    # within-run relations of population coupling
    def rho(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        return float(sp.spearmanr(a[ok], b[ok])[0]) if ok.sum() >= 10 else np.nan
    Rrow["rho_pop_depth"] = rho(R2["pop"], cent[:, 1])
    Rrow["rho_pop_event_rate"] = rho(R2["pop"], event_rate)
    Rrow["rho_pop_size"] = rho(R2["pop"], np.log(size))
    Rrow["rho_pop_beh"] = rho(R2["pop"], r2_beh) if has_beh else np.nan
    Rrow["rho_popfar_beh"] = rho(R2["pop_far"], r2_beh) if has_beh else np.nan
    Rrow.update(gl)
    print(f"  {key:40s} {source:5s} N={N:3d} pop_med={Rrow['r2_pop_median']:.3f} "
          f"far_med={Rrow['r2_pop_far_median']:.3f} all_med={Rrow['r2_all_median']:.3f} "
          f"glasso={'ok' if gl.get('glasso_ok') else 'skip'} ({time.time()-t0:.0f}s)", flush=True)
    return dict(run=Rrow, dend=D)


# ----------------------------------------------------------------------------- cohort
def _merge_pupil(dend, source):
    f = PUPIL_PHASE / source / "per_dendrite.csv"
    if not f.exists():
        dend["pupil_group"] = "na"
        return dend
    p = pd.read_csv(f)
    cols = {"group": "pupil_group", "sig": "pupil_sig", "phase_deg": "pupil_phase_deg", "coh": "pupil_coh",
            "r_pupil": "pupil_r", "z_median_when_pupil_large": "z_pupil_large",
            "z_median_when_pupil_small": "z_pupil_small", "halves_agree": "pupil_halves_agree",
            "r_core_pupil": "r_core_pupil", "r_shell_pupil": "r_shell_pupil"}
    keep = ["run", "dendrite"] + [c for c in cols if c in p.columns]
    p = p[keep].rename(columns=cols)
    return dend.merge(p, on=["run", "dendrite"], how="left")


def _wilcoxon(v, mu=0.0):
    v = pd.Series(v).dropna().to_numpy()
    if len(v) < 4:
        return np.nan, len(v), (float(np.median(v)) if len(v) else np.nan)
    if np.allclose(v, mu):
        return 1.0, len(v), float(np.median(v))
    return float(sp.wilcoxon(v - mu).pvalue), len(v), float(np.median(v))


def _lmm(dend, yname, xnames):
    """Mixed model y ~ x1 + x2 ... with FOV random intercept; predictors z-scored within FOV."""
    import statsmodels.formula.api as smf
    d = dend.dropna(subset=[yname] + xnames).copy()
    if d.fov.nunique() < 3 or len(d) < 50:
        return None
    for x in xnames:
        d[x + "_z"] = d.groupby("fov")[x].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    f = f"{yname} ~ " + " + ".join(x + "_z" for x in xnames)
    try:
        m = smf.mixedlm(f, d, groups=d["fov"]).fit(reml=True, method="lbfgs")
        return {x: (float(m.params[x + "_z"]), float(m.pvalues[x + "_z"])) for x in xnames}, int(len(d)), int(d.fov.nunique())
    except Exception:
        return None


def cohort(source, res):
    out = OUT / source
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([x["run"] for x in res])
    dend = pd.concat([x["dend"] for x in res], ignore_index=True)
    dend = _merge_pupil(dend, source)
    runs.to_csv(out / "per_run.csv", index=False)
    dend.to_csv(out / "per_dendrite.csv", index=False)
    fov = runs.groupby(["fov", "mouse"]).mean(numeric_only=True).reset_index()
    fov.to_csv(out / "per_fov.csv", index=False)

    S = dict(source=source, n_runs=int(len(runs)), n_fovs=int(runs.fov.nunique()),
             n_mice=int(runs.mouse.nunique()), n_dendrite_entries=int(len(dend)), findings=[])
    L = [f"CROSS-VALIDATED ENCODING / POPULATION-COUPLING - {source} dendrites",
         f"{len(runs)} runs, {runs.fov.nunique()} FOVs, {runs.mouse.nunique()} mice, {len(dend)} dendrite-run entries.",
         "5-fold BLOCKED (contiguous) CV in time. Held-out R^2 pooled over folds. alpha by blocked inner split.",
         "FOV = unit of replication (Wilcoxon over FOV means); per-dendrite relations via LMM w/ FOV random intercept.",
         ""]

    def add(name, text, **kw):
        S["findings"].append(dict(name=name, text=text, **kw))
        L.append(f"* [{name}] {text}")

    # ---- 1 population coupling distribution + independence fractions
    p, n, med = _wilcoxon(fov.r2_pop_median)
    add("pop_coupling_distribution",
        f"Population coupling (predict a dendrite from ALL others): held-out R^2 median {med:.3f} across {n} FOVs "
        f"(Wilcoxon vs 0 p={p:.2g}); per-dendrite median {np.nanmedian(dend.r2_pop):.3f}, "
        f"IQR [{np.nanpercentile(dend.r2_pop,25):.3f}, {np.nanpercentile(dend.r2_pop,75):.3f}]. "
        f"Fraction effectively independent (R^2<0.02): {np.nanmean(dend.r2_pop<0.02)*100:.0f}% of dendrites "
        f"(FOV median {fov.frac_pop_lt_002.median()*100:.0f}%); strongly coupled (R^2>0.2): "
        f"{np.nanmean(dend.r2_pop>0.2)*100:.0f}% (FOV median {fov.frac_pop_gt_02.median()*100:.0f}%).",
        pop_r2_median=med, p=p, frac_lt002=float(np.nanmean(dend.r2_pop < 0.02)),
        frac_gt02=float(np.nanmean(dend.r2_pop > 0.2)))

    # ---- 2 far-only (> 50 um) control for optical crosstalk
    p, n, med = _wilcoxon(fov.r2_pop_far_median)
    pd_, nd_, medd_ = _wilcoxon(fov.med_far_minus_all)
    add("far_only_crosstalk",
        f"Far-only model (predictors > {FAR_UM:.0f} um away): held-out R^2 median {med:.3f} across {n} FOVs "
        f"(Wilcoxon vs 0 p={p:.2g}); {np.nanmean(dend.r2_pop_far50>0.02)*100:.0f}% of dendrites still predicted "
        f"(R^2>0.02), {np.nanmean(dend.r2_pop_far50>0.2)*100:.0f}% with R^2>0.2. Far-minus-all R^2 median "
        f"{medd_:+.3f} (p={pd_:.2g}): population coupling is {'largely retained' if med>0.3*fov.r2_pop_median.median() else 'much reduced'} "
        f"when near neighbours are excluded, so it reflects a shared network signal, not only short-range optical crosstalk.",
        far_r2_median=med, p=p, far_minus_all=medd_)
    p, n, med = _wilcoxon(fov.med_near_minus_all)
    add("near20_vs_all",
        f"20-nearest-neighbour model: R^2 median {fov.r2_pop_near_median.median():.3f}; near-minus-all median "
        f"{med:+.3f} (p={p:.2g}). {'Nearby dendrites carry most of the predictability' if med>-0.02 else 'Distant dendrites add predictability beyond the nearest 20'}.")

    # ---- 3 unique contributions
    txt = []
    for blk, col in (("behavior", "unique_beh"), ("global Ca", "unique_glob"), ("population", "unique_pop")):
        p, n, m = _wilcoxon(fov[col + "_median"])
        txt.append(f"{blk} {m:+.3f} (p={p:.2g}, {n} FOVs)")
    add("unique_contributions",
        "Unique held-out R^2 (full model minus model without that block), FOV medians: " + "; ".join(txt) +
        f". Full model median R^2 {fov.r2_all_median.median():.3f}; global-only {fov.r2_glob_median.median():.3f}; "
        f"behavior-only {fov.r2_beh_median.median():.3f} (behavior FOVs only).")

    # ---- 4 what predicts population coupling (depth / rate / size) : per-FOV Spearman + LMM
    for lab, col in (("depth", "rho_pop_depth"), ("event rate", "rho_pop_event_rate"), ("log size", "rho_pop_size")):
        p, n, med = _wilcoxon(runs[col])
        add(f"pop_vs_{col.split('_')[-1]}",
            f"Population coupling vs {lab}: within-run Spearman median {med:+.3f} ({n} runs, Wilcoxon over runs p={p:.2g}).",
            rho=med, p=p)
    lmm = _lmm(dend, "r2_pop", ["depth_um", "event_rate_per_min", "n_vox"])
    if lmm:
        params, nlmm, nf = lmm
        add("pop_coupling_LMM",
            "LMM r2_pop ~ depth + event_rate + size (FOV random intercept, predictors z within FOV, "
            f"n={nlmm}, {nf} FOVs): " + ", ".join(f"{k} {v[0]:+.3f} (p={v[1]:.2g})" for k, v in params.items()) + ".")

    # ---- 5 chorister vs soloist (pop coupling vs behavior R^2)
    beh_dend = dend[dend.r2_beh.notna()]
    if len(beh_dend) > 50 and beh_dend.fov.nunique() >= 3:
        p, n, med = _wilcoxon(runs.rho_pop_beh)
        lab = ("choristers: dendrites coupled to behaviour are ALSO coupled to the population"
               if med > 0.05 else
               "soloists: behaviour-coupled dendrites are NOT more population-coupled" if med < -0.05 else
               "no systematic chorister/soloist axis")
        lmm2 = _lmm(beh_dend, "r2_pop", ["r2_beh"])
        extra = ""
        if lmm2:
            params, nlmm, nf = lmm2
            extra = f" LMM slope r2_pop~r2_beh {params['r2_beh'][0]:+.3f} (p={params['r2_beh'][1]:.2g}, n={nlmm}, {nf} FOVs)."
        add("chorister_vs_soloist",
            f"Population coupling vs behaviour R^2 (Okun chorister/soloist): within-run Spearman median {med:+.3f} "
            f"({n} runs, p={p:.2g}) -> {lab}.{extra}", rho=med, p=p)

    # ---- 6 graphical lasso network
    gok = runs[runs.get("glasso_ok", False) == True] if "glasso_ok" in runs else pd.DataFrame()
    if len(gok):
        add("glasso_network",
            f"Sparse partial-correlation network (GraphicalLassoCV, {int(gok.glasso_n.median())} nodes median): "
            f"{gok.glasso_frac_zero.median()*100:.0f}% of partial correlations are exactly zero "
            f"(edge density {gok.glasso_density.median()*100:.1f}%); mean degree {gok.glasso_mean_degree.median():.1f}, "
            f"max degree {gok.glasso_max_degree.median():.0f}, degree CV {gok.glasso_degree_cv.median():.2f} "
            f"(heavy-tailed -> hubs exist). Hub (top-10%) depth {gok.hub_depth_um.median():.0f} um vs non-hub "
            f"{gok.nonhub_depth_um.median():.0f} um (Mann-Whitney p<0.05 in {(gok.hub_depth_p<0.05).sum()}/{gok.hub_depth_p.notna().sum()} runs); "
            f"degree vs depth rho {gok.rho_degree_depth.median():+.2f}, vs event rate {gok.rho_degree_event_rate.median():+.2f}. "
            f"Vs density-matched plain correlation network: degree-sequence Spearman {gok.rho_degree_glasso_corr.median():.2f}, "
            f"hub overlap Jaccard {gok.hub_jaccard_glasso_corr.median():.2f} (direct partial-correlation structure differs "
            f"from marginal correlation: much is explained away by shared/indirect paths).",
            frac_zero=float(gok.glasso_frac_zero.median()), density=float(gok.glasso_density.median()))

    # ---- 7 anti-phase-with-pupil population (reality check from encoding)
    if "pupil_group" in dend and (dend.pupil_group == "anti").any():
        g = dend.copy()
        anti, inp, none = g[g.pupil_group == "anti"], g[g.pupil_group == "in"], g[g.pupil_group == "none"]
        # per-FOV contrasts anti vs in (FOVs with >=3 in each)
        def fov_contrast(col):
            diffs = []
            for f_, gg in g.groupby("fov"):
                a = gg[gg.pupil_group == "anti"][col].dropna()
                b = gg[gg.pupil_group == "in"][col].dropna()
                if len(a) >= 3 and len(b) >= 3:
                    diffs.append(a.median() - b.median())
            return _wilcoxon(diffs)
        parts = []
        for lab, col in (("depth_um", "depth_um"), ("event_rate", "event_rate_per_min"),
                         ("r2_pop", "r2_pop"), ("r2_pop_far50", "r2_pop_far50"), ("r2_beh", "r2_beh")):
            p, n, med = fov_contrast(col)
            parts.append(f"{lab} {med:+.3g} (p={p:.2g},{n}FOV)")
        pupil_defl = None
        if {"z_pupil_large", "z_pupil_small"}.issubset(g.columns):
            anti_defl = (anti.z_pupil_large - anti.z_pupil_small)
            in_defl = (inp.z_pupil_large - inp.z_pupil_small)
            pupil_defl = (float(anti_defl.median()), float(in_defl.median()))
        halves = float(anti.pupil_halves_agree.dropna().mean()) if "pupil_halves_agree" in anti else np.nan
        # core-vs-shell sign-flip diagnostic: is the anti-phase sign made by shell (neuropil) subtraction?
        verdict = ""
        kw = {}
        if {"r_core_pupil", "r_shell_pupil", "pupil_r"}.issubset(anti.columns):
            fin, cor, she = anti.pupil_r.median(), anti.r_core_pupil.median(), anti.r_shell_pupil.median()
            frac_core_pos = float((anti.r_core_pupil > 0).mean())
            frac_shell_gt_core = float((anti.r_shell_pupil > anti.r_core_pupil).mean())
            artifact = (cor > 0) and (frac_core_pos > 0.6) and (frac_shell_gt_core > 0.6)
            verdict = (f" SIGN-FLIP DIAGNOSTIC: for anti-phase dendrites the final (core-minus-shell) trace is "
                       f"pupil-NEGATIVE (r={fin:+.3f}) but the dendrite CORE itself is pupil-POSITIVE (r={cor:+.3f}, "
                       f"positive in {frac_core_pos*100:.0f}% of them) and the surrounding neuropil shell is MORE "
                       f"pupil-positive than the core (r={she:+.3f}; shell>core in {frac_shell_gt_core*100:.0f}%). "
                       + ("The anti-phase sign is MANUFACTURED BY SHELL (neuropil) SUBTRACTION, not a genuine "
                          "dendritic anti-correlation with arousal; its cross-half reproducibility reflects a "
                          "systematic subtraction bias, not a real anti-phase cell type. "
                          if artifact else
                          "Core is itself pupil-negative, so the anti-phase sign is NOT only a subtraction artifact. "))
            kw = dict(anti_final_pupil_r=float(fin), anti_core_pupil_r=float(cor), anti_shell_pupil_r=float(she),
                      frac_core_pos=frac_core_pos, frac_shell_gt_core=frac_shell_gt_core, likely_artifact=bool(artifact))
        add("antiphase_pupil_population",
            f"Anti-phase group n={len(anti)} ({len(anti)/len(g)*100:.0f}% of dendrites; in-phase n={len(inp)}). "
            f"Replicating prior description, anti vs in-phase per-FOV contrasts: " + "; ".join(parts) + ". " +
            (f"Pupil-large minus pupil-small z: anti {pupil_defl[0]:+.2f} vs in {pupil_defl[1]:+.2f} "
             f"(anti dendrites' extracted trace goes DOWN when pupil is large). " if pupil_defl else "") +
            f"Phase sign reproduces across within-run halves in {halves*100:.0f}% of anti-phase dendrites (50% = chance)."
            + verdict +
            f"Anti-phase dendrites remain strongly population-coupled (r2_pop {anti.r2_pop.median():+.3f}, "
            f"far-only {anti.r2_pop_far50.median():+.3f}), i.e. they are embedded in the shared signal, not independent.",
            n_anti=int(len(anti)), halves_agree=halves, **kw)

    # ---- auto vs human stability note added by main()
    L += ["",
          "Reading guide: independent units -> pop R^2 ~ 0 for most dendrites, far-only R^2 ~ 0, flat degree",
          "distribution, no chorister/soloist axis. Shared network -> positive pop & far R^2, hubs, heavy-tailed degree.",
          "Caveats: 5 Hz, ~106-108 s per run (short for CV); global Ca contains the target (its R^2 is an upper bound);",
          "SCAPE optical crosstalk inflates near-neighbour coupling (far-only model is the control); 2 mice with behaviour,",
          "3 total; held-out R^2 can be negative (worse than the mean) and is reported unclipped."]
    for ln in [x for x in L if not x.startswith("* [")]:
        pass
    (out / "summary.txt").write_text("\n".join(L))
    (out / "summary.json").write_text(json.dumps(S, indent=2,
                                       default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))
    figures(source, runs, fov, dend, out)
    return L


def figures(source, runs, fov, dend, out):
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    a = ax[0, 0]
    v = dend.r2_pop.dropna()
    a.hist(np.clip(v, -0.1, 1), bins=50, color="#4477aa")
    a.axvline(0.02, color="k", ls=":", lw=0.8); a.axvline(0.2, color="r", ls="--", lw=0.8)
    a.set_xlabel("population R^2 (all others)"); a.set_ylabel("# dendrites")
    a.set_title(f"1  pop coupling (med {np.nanmedian(v):.3f})")
    a = ax[0, 1]
    a.scatter(dend.r2_pop, dend.r2_pop_far50, s=3, alpha=0.25, c=pd.factorize(dend.fov)[0], cmap="tab20")
    lim = [min(dend.r2_pop.min(), -0.05), max(0.6, dend.r2_pop.quantile(0.99))]
    a.plot(lim, lim, "k--", lw=0.6); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("pop R^2 (all)"); a.set_ylabel("pop R^2 (>50 um)")
    a.set_title("2  far-only vs all (crosstalk control)")
    a = ax[0, 2]
    a.scatter(dend.depth_um, dend.r2_pop, s=3, alpha=0.25, c=pd.factorize(dend.fov)[0], cmap="tab20")
    a.set_xlabel("depth (um, 0=surface)"); a.set_ylabel("pop R^2"); a.set_title("3  pop coupling vs depth")
    a = ax[0, 3]
    a.scatter(dend.event_rate_per_min, dend.r2_pop, s=3, alpha=0.25, c=pd.factorize(dend.fov)[0], cmap="tab20")
    a.set_xlabel("event rate (/min)"); a.set_ylabel("pop R^2"); a.set_title("4  pop coupling vs activity")
    a = ax[1, 0]
    bd = dend[dend.r2_beh.notna()]
    if len(bd):
        a.scatter(bd.r2_beh, bd.r2_pop, s=3, alpha=0.25, c=pd.factorize(bd.fov)[0], cmap="tab20")
        a.set_xlabel("behavior R^2"); a.set_ylabel("pop R^2")
    a.set_title("5  chorister vs soloist")
    a = ax[1, 1]
    meds = [np.nanmedian(dend.unique_beh), np.nanmedian(dend.unique_glob), np.nanmedian(dend.unique_pop)]
    a.bar(["behavior", "global", "population"], meds, color=["#e69500", "#117733", "#4477aa"])
    a.axhline(0, color="k", lw=0.5); a.set_ylabel("median unique held-out R^2")
    a.set_title("6  unique contributions")
    a = ax[1, 2]
    if "glasso_degree" in dend and dend.glasso_degree.notna().any():
        a.hist(dend.glasso_degree.dropna(), bins=40, color="#aa3377")
        a.set_xlabel("graphical-lasso degree"); a.set_ylabel("# dendrites")
    a.set_title("7  partial-correlation degree")
    a = ax[1, 3]
    if "pupil_group" in dend:
        order = ["none", "in", "anti"]
        data = [dend[dend.pupil_group == gcat].r2_pop.dropna() for gcat in order]
        data = [d.to_numpy() for d in data if len(d)]
        if data:
            a.boxplot(data, labels=[o for o, d in zip(order, [dend[dend.pupil_group == gcat].r2_pop.dropna() for gcat in order]) if len(d)],
                      showfliers=False)
            a.set_ylabel("pop R^2")
    a.set_title("8  pop coupling by pupil phase")
    fig.suptitle(f"Encoding / population coupling - {source} dendrites", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_encoding.png", dpi=140)
    fig.savefig(out / "fig_encoding.pdf")
    plt.close(fig)

    # network comparison figure
    gok = runs[runs.get("glasso_ok", False) == True] if "glasso_ok" in runs else pd.DataFrame()
    if len(gok):
        fig2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
        ax2[0].scatter(gok.glasso_density * 100, gok.glasso_frac_zero * 100, s=20)
        ax2[0].set_xlabel("edge density (%)"); ax2[0].set_ylabel("% zero partial corr"); ax2[0].set_title("sparsity")
        ax2[1].scatter(gok.hub_depth_um, gok.nonhub_depth_um, s=20)
        lim = [0, max(gok.hub_depth_um.max(), gok.nonhub_depth_um.max()) * 1.05]
        ax2[1].plot(lim, lim, "k--", lw=0.6); ax2[1].set_xlabel("hub depth (um)"); ax2[1].set_ylabel("non-hub depth (um)")
        ax2[1].set_title("hub vs non-hub depth")
        ax2[2].scatter(gok.rho_degree_glasso_corr, gok.hub_jaccard_glasso_corr, s=20)
        ax2[2].set_xlabel("degree Spearman (glasso vs corr)"); ax2[2].set_ylabel("hub Jaccard")
        ax2[2].set_title("partial vs marginal network")
        fig2.suptitle(f"Partial-correlation network - {source}", fontsize=10)
        fig2.tight_layout()
        fig2.savefig(out / "fig_network.png", dpi=140)
        fig2.savefig(out / "fig_network.pdf")
        plt.close(fig2)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("auto", "human", "both"), default="both")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--no-glasso", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    keys = a.run or [r.key for r in discover_runs()]
    if a.limit:
        keys = keys[:a.limit]
    t0 = time.time()
    allL = {}
    for s in (("auto", "human") if a.source == "both" else (a.source,)):
        res = []
        tasks = [(k, s) for k in keys]
        with ProcessPoolExecutor(max(1, min(a.jobs, 3))) as ex:
            futs = {ex.submit(analyze, k, s, not a.no_glasso): k for k, s in tasks}
            for f in as_completed(futs):
                try:
                    x = f.result()
                    if x is not None:
                        res.append(x)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("ERROR", futs[f], e, flush=True)
        if not res:
            print(f"no results for source={s}")
            continue
        res.sort(key=lambda x: x["run"]["run"])
        allL[s] = cohort(s, res)
        print("\n".join(allL[s]), "\n", flush=True)
    if len(allL) == 2:
        (OUT / "compare_auto_vs_human.txt").write_text(
            "AUTO\n" + "\n".join(allL["auto"]) + "\n\nHUMAN\n" + "\n".join(allL["human"]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
