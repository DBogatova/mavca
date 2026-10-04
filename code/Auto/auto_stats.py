#!/usr/bin/env python
"""auto_stats.py - cohort statistics for the SCAPE apical-dendrite pipeline (code/Auto/, v2).

Run per source (auto = automatic dendrites, human = hand-curated masks; both through the
same extractor) and then compared:

  per run   pairwise dendrite correlations vs a circular-shift null (BH-FDR per pair),
            correlation vs 3D centroid distance / depth difference, participation ratio,
            events (robust z > 3 for >= 2 frames), global-event fraction, dendrite and
            global-Ca coupling to pupil / whisker / accel (best lag within +-3 s, p from a
            circular-shift null that repeats the lag search = corrected for the search),
            accel-onset-triggered global Ca, quiet vs active (accel above its 75th pct).
  cohort    FOV is the unit of replication: runs that share a field of view share
            dendrites (human masks are literally the same), so run metrics are averaged
            within FOV before any across-sample test; per-dendrite models use mixed models
            with a random intercept per FOV (statsmodels MixedLM); mouse-level variance is
            reported as ICC. Same-FOV reproducibility: Spearman of pair-r across runs (human:
            same masks; auto: units matched across runs by overlap).
  compare   run-level metrics auto vs human (Spearman across runs, paired Wilcoxon over FOVs,
            sign agreement of each conclusion).

Time alignment: trace CSVs carry time_s = 0 at frame skip_s*frame_rate; load_behavior()
returns behavior on the same clock. (v1 of this script subtracted skip_s a second time,
which misaligned behavior by 12-14 s; fixed here.)

Outputs: scape-auto/stats/<source>/{per_run.csv, per_fov.csv, per_dendrite.csv,
         stats_summary.txt, stats_summary.json, fig_*.png/pdf}
         scape-auto/stats/compare_auto_vs_human.{csv,txt,json,png,pdf}
CLI: auto_stats.py [--source auto|human|both] [--all | --run KEY ...] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import stats as sp
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, fov_group, load_behavior, resample_to, human_masks, human_masks_valid, VOXEL_ZYX, AUTO_ROOT,
)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore")
STATS = AUTO_ROOT / "stats"
EV_Z, EV_MIN = 3.0, 2
GLOBAL_FRAC = 0.2
MAX_LAG_S = 3.0
N_SHIFT = 200
BEHS = ("pupil", "whisker", "accel")
__version__ = "2.0.0"


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


def events(z):
    """Peaks of runs with z > EV_Z lasting >= EV_MIN frames. Returns list of arrays per column."""
    out = []
    for j in range(z.shape[1]):
        a = np.r_[False, z[:, j] > EV_Z, False]
        d = np.diff(a.astype(int))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        pk = [s + int(np.argmax(z[s:e, j])) for s, e in zip(st, en) if e - s >= EV_MIN]
        out.append(np.array(pk, int))
    return out


def lagged_corr_max(M, y, maxlag):
    """max over lags of |r| between columns of M and y shifted by lag (y leads for lag>0).
    Returns (r_at_best, best_lag) per column."""
    T = len(y)
    best = np.zeros(M.shape[1])
    blag = np.zeros(M.shape[1], int)
    for L in range(-maxlag, maxlag + 1):
        if L >= 0:
            a, b = M[L:], y[:T - L]
        else:
            a, b = M[:T + L], y[-L:]
        r = (zc(a) * zc(b[:, None])).sum(0)
        upd = np.abs(r) > np.abs(best)
        best[upd] = r[upd]
        blag[upd] = L
    return best, blag


def coupling(M, y, fr, rng):
    maxlag = int(round(MAX_LAG_S * fr))
    r0, lag = lagged_corr_max(M, y, maxlag)
    ms = int(10 * fr)
    null = np.stack([np.abs(lagged_corr_max(M, np.roll(y, s), maxlag)[0])
                     for s in rng.integers(ms, len(y) - ms, N_SHIFT)])
    p = (1 + (null >= np.abs(r0)[None]).sum(0)) / (N_SHIFT + 1)
    return r0, lag / fr, p


def load_source(r: Run, source: str):
    if source == "auto":
        csv = r.out / "traces" / "dff_auto.csv"
        lp = r.out / "masks" / "auto_labelmap_reviewed.tif"
        lp = lp if lp.exists() else r.out / "masks" / "auto_labelmap.tif"
        if not (csv.exists() and lp.exists()):
            return None
        lab = tifffile.imread(str(lp))
        df = pd.read_csv(csv)
        names = [c for c in df.columns if c.startswith("dend_")]
        units = {f"dend_{k - 1:03d}": np.flatnonzero(lab.ravel() == k) for k in range(1, int(lab.max()) + 1)}
        shape = lab.shape
    else:
        csv = r.out / "traces" / "dff_human_sameextractor.csv"
        if not csv.exists() or not human_masks_valid(r):
            return None
        df = pd.read_csv(csv)
        names = [c for c in df.columns if c.startswith("dend_")]
        import tifffile as tf
        shape = tf.memmap(str(r.raw), mode="r").shape[1:]
        units = {n: np.flatnonzero(m.ravel()) for n, m in human_masks(r, shape)}
    names = [n for n in names if n in units and units[n].size]
    return df, names, units, shape


# ----------------------------------------------------------------------------- per run
def analyze_run(key: str, source: str) -> dict | None:
    r = get_run(key)
    got = load_source(r, source)
    if got is None:
        return None
    df, names, units, shape = got
    fr = r.frame_rate
    t = df["time_s"].to_numpy()
    M = df[names].to_numpy(float)
    T, N = M.shape
    rng = np.random.default_rng(0)
    cent = np.array([np.mean(np.unravel_index(units[n], shape), axis=1) * np.array(VOXEL_ZYX) for n in names])
    res = dict(run=key, source=source, date=r.date, mouse=r.mouse, fov=fov_group(r), frame_rate=fr,
               n_frames=T, duration_min=T / fr / 60, n_dendrites=N)

    # pairwise + null (independent circular shifts of each dendrite)
    C = zc(M).T @ zc(M)
    iu = np.triu_indices(N, 1)
    pr = C[iu]
    ms = int(10 * fr)
    null = []
    for _ in range(20):
        Mr = np.stack([np.roll(M[:, j], rng.integers(ms, T - ms)) for j in range(N)], 1)
        null.append((zc(Mr).T @ zc(Mr))[iu])
    null = np.sort(np.abs(np.concatenate(null)))
    p_pair = 1 - np.searchsorted(null, np.abs(pr)) / (null.size + 1)
    q_pair = bh(np.maximum(p_pair, 1 / (null.size + 1)))
    dist = np.linalg.norm(cent[iu[0]] - cent[iu[1]], axis=1)
    ddep = np.abs(cent[iu[0], 1] - cent[iu[1], 1])
    ev = np.linalg.eigvalsh(np.cov(M.T))
    ev = ev[ev > 0]
    res.update(mean_pair_r=float(pr.mean()), median_pair_r=float(np.median(pr)),
               frac_pairs_sig=float((q_pair < 0.05).mean()), frac_pairs_pos_sig=float(((q_pair < 0.05) & (pr > 0)).mean()),
               rho_r_distance=float(sp.spearmanr(dist, pr)[0]), rho_r_depthdiff=float(sp.spearmanr(ddep, pr)[0]),
               r_near_lt30um=float(pr[dist < 30].mean()) if (dist < 30).any() else np.nan,
               r_far_gt100um=float(pr[dist > 100].mean()) if (dist > 100).any() else np.nan,
               pr_over_n=float(ev.sum() ** 2 / (ev ** 2).sum() / N),
               participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()))

    # events
    Z = rz(M)
    evs = events(Z)
    nact = (Z > EV_Z).sum(1)
    glob_frames = np.convolve(nact >= max(2, GLOBAL_FRAC * N), np.ones(3), "same") > 0
    n_ev = np.array([len(e) for e in evs])
    n_glob = sum(int(glob_frames[e].sum()) for e in evs)
    amp = np.array([M[e, j].mean() if len(e) else np.nan for j, e in enumerate(evs)])
    res.update(event_rate_per_min=float(n_ev.mean() / res["duration_min"]),
               frac_global_events=float(n_glob / max(1, n_ev.sum())),
               frac_frames_global=float(glob_frames.mean()))

    # behavior
    beh = load_behavior(r)
    B = {}
    for k in BEHS:
        if beh.get(k) is not None:
            y = resample_to(beh[f"{k}_t"], beh[k], t)
            if np.isfinite(y).mean() > 0.9:
                B[k] = np.where(np.isfinite(y), y, np.nanmedian(y))
    gpath = r.out / "traces" / "global_ca.csv"
    gy = None
    if gpath.exists():
        g = pd.read_csv(gpath)
        gy = resample_to(g["time_s"].to_numpy(), g["global_dff"].to_numpy(), t)
        gy = np.where(np.isfinite(gy), gy, np.nanmedian(gy))
    dend = pd.DataFrame(dict(run=key, source=source, mouse=r.mouse, fov=fov_group(r), dendrite=names,
                             depth_um=cent[:, 1], x_um=cent[:, 2], z_um=cent[:, 0], n_vox=[units[n].size for n in names],
                             event_rate_per_min=n_ev / res["duration_min"], event_amp=amp))
    res["has_behavior"] = bool(B)
    for k, y in B.items():
        rr, lag, p = coupling(M, y, fr, rng)
        q = bh(p)
        dend[f"r_{k}"], dend[f"lag_{k}"], dend[f"q_{k}"] = rr, lag, q
        res[f"frac_coupled_{k}"] = float((q < 0.05).mean())
        res[f"frac_pos_coupled_{k}"] = float(((q < 0.05) & (rr > 0)).mean())
        res[f"frac_neg_coupled_{k}"] = float(((q < 0.05) & (rr < 0)).mean())
        res[f"median_r_{k}"] = float(np.median(rr))
        if gy is not None:
            gr, gl, gp = coupling(gy[:, None], y, fr, rng)
            res[f"global_r_{k}"], res[f"global_lag_{k}"], res[f"global_p_{k}"] = float(gr[0]), float(gl[0]), float(gp[0])
    if "accel" in B:
        a = B["accel"]
        act = a > np.percentile(a, 75)
        if act.sum() > 20 and (~act).sum() > 20:
            for nm, sel in (("active", act), ("quiet", ~act)):
                Cs = zc(M[sel]).T @ zc(M[sel])
                res[f"pair_r_{nm}"] = float(Cs[iu].mean())
                res[f"event_rate_{nm}"] = float(np.mean([np.isin(e, np.flatnonzero(sel)).sum() for e in evs]) / (sel.sum() / fr / 60))
        if gy is not None:
            med = np.median(a)
            mad = np.median(np.abs(a - med)) * 1.4826 + 1e-12
            hi = a > med + 3 * mad
            w = int(3 * fr)
            on = [i for i in np.flatnonzero(hi[1:] & ~hi[:-1]) + 1 if i >= w and i < T - w and not hi[i - int(2 * fr):i].any()]
            if len(on) >= 3:
                seg = np.stack([gy[i - w:i + w + 1] - gy[i - w:i].mean() for i in on])
                post = slice(w + 1, w + 1 + int(1.5 * fr))
                obs = seg[:, post].mean()
                sh = []
                for _ in range(500):
                    rnd = rng.integers(w, T - w, len(on))
                    sh.append(np.mean([gy[i + 1:i + 1 + int(1.5 * fr)].mean() - gy[i - w:i].mean() for i in rnd]))
                sh = np.array(sh)
                res.update(n_accel_onsets=len(on), onset_global_change=float(obs),
                           onset_p_two_sided=float((1 + (np.abs(sh - sh.mean()) >= abs(obs - sh.mean())).sum()) / 501))
    pairs = pd.DataFrame(dict(run=key, fov=fov_group(r), a=np.array(names)[iu[0]], b=np.array(names)[iu[1]], r=pr, dist_um=dist))
    return dict(run=res, dend=dend, pairs=pairs)


# ----------------------------------------------------------------------------- cohort
def mixed(formula, data, group):
    import statsmodels.formula.api as smf
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(formula, data, groups=data[group]).fit(reml=True, method="lbfgs")
        return m
    except Exception:
        return None


def icc(df, col):
    """eta^2 = share of run-level sum of squares explained by mouse, and by FOV (0-1)."""
    d = df[["mouse", "fov", col]].dropna()
    if d.mouse.nunique() < 2 or len(d) < 4:
        return None
    ss = ((d[col] - d[col].mean()) ** 2).sum()
    if ss <= 0:
        return None
    def eta(g):
        gm = d.groupby(g)[col].transform("mean")
        return float(((gm - d[col].mean()) ** 2).sum() / ss)
    return dict(mouse=eta("mouse"), fov=eta("fov"))


def auto_reproducibility(runs_by_fov: dict) -> list[dict]:
    """For FOVs with several runs: match auto units across runs (overlap coefficient >= 0.5) and
    correlate their pair-r structure."""
    out = []
    for fov, keys in runs_by_fov.items():
        if len(keys) < 2:
            continue
        labs = {}
        for k in keys:
            p = get_run(k).out / "masks" / "auto_labelmap_perrun.tif"   # each run's OWN detection
            if p.exists():
                labs[k] = tifffile.imread(str(p)).astype(np.int32)
        ks = sorted(labs)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                A, Bm = labs[ks[i]], labs[ks[j]]
                if A.shape != Bm.shape:
                    continue
                na, nb = A.max(), Bm.max()
                I = np.zeros((na + 1, nb + 1))
                np.add.at(I, (A.ravel(), Bm.ravel()), 1)
                sa, sb = I.sum(1), I.sum(0)
                I = I[1:, 1:]
                oc = I / np.maximum(np.minimum(sa[1:, None], sb[None, 1:]), 1)
                m_a = oc.argmax(1)
                good = oc[np.arange(na), m_a] >= 0.5
                out.append(dict(fov=fov, run_a=ks[i], run_b=ks[j], n_a=int(na), n_b=int(nb),
                                frac_a_matched_in_b=float(good.mean()),
                                pairs=[(int(x) + 1, int(m_a[x]) + 1) for x in np.flatnonzero(good)]))
    return out


def cohort(source: str, results: list[dict], out: Path) -> dict:
    runs = pd.DataFrame([x["run"] for x in results])
    dend = pd.concat([x["dend"] for x in results], ignore_index=True)
    pairs = pd.concat([x["pairs"] for x in results], ignore_index=True)
    runs.to_csv(out / "per_run.csv", index=False)
    dend.to_csv(out / "per_dendrite.csv", index=False)
    num = runs.select_dtypes("number").columns
    fov = runs.groupby(["fov", "mouse"])[list(num)].mean().reset_index()
    fov["n_runs"] = runs.groupby("fov").size().reindex(fov.fov).to_numpy()
    fov.to_csv(out / "per_fov.csv", index=False)
    S = dict(source=source, version=__version__, timestamp=datetime.now(timezone.utc).isoformat(),
             n_runs=len(runs), n_fovs=int(runs.fov.nunique()), n_mice=int(runs.mouse.nunique()),
             n_dendrites=int(len(dend)), tests=[])
    T = S["tests"]

    def wil(col, name, desc):
        v = fov[col].dropna()
        if len(v) < 4:
            return
        w = sp.wilcoxon(v)
        T.append(dict(name=name, level="FOV", n=int(len(v)), median=float(v.median()), mean=float(v.mean()),
                      iqr=[float(v.quantile(.25)), float(v.quantile(.75))], test="Wilcoxon signed-rank vs 0",
                      p=float(w.pvalue), conclusion=desc.format(med=v.median(), p=w.pvalue, n=len(v))))

    # 1 correlations
    wil("mean_pair_r", "mean_pairwise_r",
        "Mean dendrite-dendrite r per FOV: median {med:.3f} across {n} FOVs (p={p:.2g} vs 0).")
    T.append(dict(name="frac_pairs_significant", level="FOV", n=int(fov.frac_pairs_sig.notna().sum()),
                  median=float(fov.frac_pairs_sig.median()),
                  conclusion=f"Median {fov.frac_pairs_sig.median() * 100:.0f}% of dendrite pairs per FOV exceed the circular-shift null (BH q<0.05); the rest are statistically independent."))
    wil("rho_r_distance", "corr_vs_distance",
        "Spearman rho(pair r, 3D distance) per FOV: median {med:.3f} (p={p:.2g}); negative = nearby dendrites more correlated.")
    wil("pr_over_n", "dimensionality", "Participation ratio / N: median {med:.2f} across {n} FOVs (1 = fully independent).")
    T.append(dict(name="global_events", level="FOV", n=int(fov.frac_global_events.notna().sum()),
                  median=float(fov.frac_global_events.median()),
                  range=[float(fov.frac_global_events.min()), float(fov.frac_global_events.max())],
                  conclusion=f"Fraction of dendritic events that are part of a global (>= {GLOBAL_FRAC * 100:.0f}% co-active) event: median {fov.frac_global_events.median():.2f}, range {fov.frac_global_events.min():.2f}-{fov.frac_global_events.max():.2f} across FOVs (FOV-specific regimes)."))
    # 2 within vs across mouse
    S["icc"] = {c: icc(runs, c) for c in ("mean_pair_r", "event_rate_per_min", "frac_global_events", "pr_over_n")}
    rep = []
    # pair-correlation structure across runs of one FOV: both sources share dendrite identities
    # within a FOV (human: same masks; auto: FOV-union masks), so pairs are matched by name
    for f, g in pairs.groupby("fov"):
        ks = sorted(g.run.unique())
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a = g[g.run == ks[i]].set_index(["a", "b"]).r
                b = g[g.run == ks[j]].set_index(["a", "b"]).r
                c = a.index.intersection(b.index)
                if len(c) > 20:
                    rep.append(dict(fov=f, run_a=ks[i], run_b=ks[j], n_pairs=len(c), rho=float(sp.spearmanr(a[c], b[c])[0])))
    redetect = []
    if source == "auto":    # do independent per-run detections find the same dendrites?
        byfov = runs.groupby("fov").run.apply(list).to_dict()
        redetect = [{k: v for k, v in m.items() if k != "pairs"} for m in auto_reproducibility(byfov)]
    S["auto_redetection_across_runs"] = redetect
    S["same_fov_reproducibility"] = rep
    if rep:
        rh = [x["rho"] for x in rep]
        T.append(dict(name="same_fov_reproducibility", level="run pair", n=len(rh), median=float(np.median(rh)),
                      conclusion=f"Pair-correlation structure is reproducible across runs of the same FOV: median Spearman rho {np.median(rh):.2f} over {len(rh)} run pairs."))
    if redetect:
        fr_ = [x["frac_a_matched_in_b"] for x in redetect]
        T.append(dict(name="auto_redetection", level="run pair", n=len(fr_), median=float(np.median(fr_)),
                      conclusion=f"Independent per-run detections of the same FOV re-find each other's dendrites (overlap >= 0.5): median {np.median(fr_) * 100:.0f}% (range {min(fr_) * 100:.0f}-{max(fr_) * 100:.0f}%) over {len(fr_)} run pairs; the rest are dendrites active in one run only."))
    # 3 behavior
    for k in BEHS:
        col = f"r_{k}"
        if col not in dend or dend[col].notna().sum() < 20:
            continue
        d = dend.dropna(subset=[col]).copy()
        d["fz"] = np.arctanh(np.clip(d[col], -0.999, 0.999))
        m = mixed("fz ~ 1", d, "fov")
        fc = fov[f"frac_coupled_{k}"].dropna()
        T.append(dict(name=f"dendrite_coupling_{k}", level="dendrite (LMM, random intercept FOV)", n_dendrites=int(len(d)),
                      n_fovs=int(d.fov.nunique()),
                      mean_r=float(np.tanh(m.params["Intercept"])) if m else None, p=float(m.pvalues["Intercept"]) if m else None,
                      frac_coupled_median_fov=float(fc.median()),
                      frac_pos=float(fov[f"frac_pos_coupled_{k}"].median()), frac_neg=float(fov[f"frac_neg_coupled_{k}"].median()),
                      conclusion=(f"Dendrite vs {k}: LMM mean r = {np.tanh(m.params['Intercept']):.3f} (p={m.pvalues['Intercept']:.2g}); "
                                  f"median {fc.median() * 100:.0f}% of dendrites per FOV significantly coupled (best lag +-{MAX_LAG_S:g} s, BH q<0.05; "
                                  f"{fov[f'frac_pos_coupled_{k}'].median() * 100:.0f}% positive, {fov[f'frac_neg_coupled_{k}'].median() * 100:.0f}% negative).") if m else "model failed"))
        gcol = f"global_r_{k}"
        if gcol in runs:
            g = runs.dropna(subset=[gcol])
            nsig = int((g[f"global_p_{k}"] < 0.05).sum())
            wil(gcol, f"global_coupling_{k}", f"Global Ca vs {k}: median r {{med:.3f}} across {{n}} FOVs (p={{p:.2g}}); {nsig}/{len(g)} runs significant individually.")
    if "pair_r_active" in fov:
        d = fov.dropna(subset=["pair_r_active", "pair_r_quiet"])
        if len(d) >= 4:
            w = sp.wilcoxon(d.pair_r_active, d.pair_r_quiet)
            w2 = sp.wilcoxon(d.event_rate_active, d.event_rate_quiet)
            T.append(dict(name="state_dependence", level="FOV", n=int(len(d)),
                          pair_r_active=float(d.pair_r_active.median()), pair_r_quiet=float(d.pair_r_quiet.median()), p_pair_r=float(w.pvalue),
                          rate_active=float(d.event_rate_active.median()), rate_quiet=float(d.event_rate_quiet.median()), p_rate=float(w2.pvalue),
                          conclusion=f"Moving (accel > 75th pct) vs still: pair r {d.pair_r_active.median():.3f} vs {d.pair_r_quiet.median():.3f} (p={w.pvalue:.2g}); event rate {d.event_rate_active.median():.2f} vs {d.event_rate_quiet.median():.2f}/min (p={w2.pvalue:.2g}); Wilcoxon over {len(d)} FOVs."))
    if "onset_global_change" in fov:
        wil("onset_global_change", "accel_onset_global_ca", "Global Ca change in the 1.5 s after locomotion onsets: median {med:.2f}% dF/F across {n} FOVs (p={p:.2g}).")
    # 4 depth
    d = dend.dropna(subset=["depth_um", "event_rate_per_min"]).copy()
    d["depth_z"] = d.groupby("fov").depth_um.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    d["rate_z"] = d.groupby("fov").event_rate_per_min.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    m = mixed("rate_z ~ depth_z", d, "fov")
    if m:
        T.append(dict(name="depth_vs_event_rate", level="dendrite (LMM, within-FOV z, random intercept FOV)", n_dendrites=int(len(d)),
                      coef=float(m.params["depth_z"]), p=float(m.pvalues["depth_z"]),
                      per_fov_rho=d.groupby("fov").apply(lambda g: sp.spearmanr(g.depth_um, g.event_rate_per_min)[0]).round(3).to_dict(),
                      conclusion=f"Event rate vs depth: coef {m.params['depth_z']:.3f} SD per SD (p={m.pvalues['depth_z']:.2g}); " + ((("superficial" if m.params['depth_z'] < 0 else "deeper") + " dendrites fire more.") if m.pvalues['depth_z'] < 0.05 else "no significant depth dependence pooled over FOVs (see per_fov_rho for FOV-specific effects).")))
    d2 = dend.dropna(subset=["depth_um", "event_amp"]).copy()
    d2["depth_z"] = d2.groupby("fov").depth_um.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    d2["amp_z"] = d2.groupby("fov").event_amp.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    m = mixed("amp_z ~ depth_z", d2, "fov")
    if m:
        T.append(dict(name="depth_vs_amplitude", level="dendrite (LMM)", n_dendrites=int(len(d2)), coef=float(m.params["depth_z"]),
                      p=float(m.pvalues["depth_z"]), conclusion=f"Event amplitude vs depth: coef {m.params['depth_z']:.3f} (p={m.pvalues['depth_z']:.2g})."))
    (out / "stats_summary.json").write_text(json.dumps(S, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else str(o)))
    lines = [f"SCAPE apical dendrites - cohort statistics ({source} dendrites), {S['timestamp']}",
             f"{S['n_runs']} runs, {S['n_fovs']} FOVs, {S['n_mice']} mice, {S['n_dendrites']} dendrite-run entries.",
             "FOV = unit of replication (runs sharing a field of view are averaged first).", ""]
    for x in T:
        lines.append(f"* [{x['name']}] {x.get('conclusion', '')}")
    lines.append("")
    lines.append("Share of run-level variance (eta^2) explained by mouse / by FOV: " +
                 "; ".join(f"{k}: mouse {v['mouse']:.2f}, FOV {v['fov']:.2f}" for k, v in S["icc"].items() if v))
    lines += ["", "Caveats: 3 mice (rbp4_132, rbp4_139, rbp4cre_139 - treated as separate animals as labelled), so mouse-level",
              "inference is weak; FOV-level tests have n = number of FOVs above (~10). 136/138 6 Hz dual-channel sessions have no raw stack locally and",
              "are not included. Behavior coupling uses full-trace lagged correlation plus onset-triggered averages."]
    (out / "stats_summary.txt").write_text("\n".join(lines))
    figures(source, runs, fov, dend, pairs, out)
    return S


def figures(source, runs, fov, dend, pairs, out):
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    # a pairwise r per FOV
    a = ax[0, 0]
    order = fov.sort_values("mean_pair_r").fov.tolist()
    for i, f in enumerate(order):
        v = pairs[pairs.fov == f].r
        a.boxplot(v, positions=[i], widths=0.6, showfliers=False)
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks(range(len(order)), [f.split("/")[0][5:] + " " + f.split("/")[-1] for f in order], rotation=60, fontsize=6)
    a.set_ylabel("pairwise r")
    a.set_title("a  dendrite-dendrite r per FOV")
    # b r vs distance
    a = ax[0, 1]
    bins = np.arange(0, 400, 25)
    for f, g in pairs.groupby("fov"):
        idx = np.digitize(g.dist_um, bins)
        m = [g.r[idx == k].mean() if (idx == k).sum() > 10 else np.nan for k in range(1, len(bins))]
        a.plot(bins[:-1] + 12.5, m, lw=0.8, alpha=0.7)
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("centroid distance (um)")
    a.set_ylabel("mean pair r")
    a.set_title("b  correlation vs distance (one line per FOV)")
    # c global fraction
    a = ax[0, 2]
    a.bar(range(len(fov)), fov.frac_global_events, color="#555")
    a.set_xticks(range(len(fov)), [f.split("/")[0][5:] + " " + f.split("/")[-1] for f in fov.fov], rotation=60, fontsize=6)
    a.set_ylabel("fraction of events that are global")
    a.set_title("c  global vs local events")
    # d coupling fractions
    a = ax[1, 0]
    for i, k in enumerate(BEHS):
        if f"frac_pos_coupled_{k}" in fov:
            a.scatter(np.full(len(fov), i - 0.12) + np.random.default_rng(i).normal(0, 0.03, len(fov)), fov[f"frac_pos_coupled_{k}"], color="tab:red", s=12)
            a.scatter(np.full(len(fov), i + 0.12) + np.random.default_rng(i + 9).normal(0, 0.03, len(fov)), fov[f"frac_neg_coupled_{k}"], color="tab:blue", s=12)
    a.set_xticks(range(3), BEHS)
    a.set_ylabel("fraction of dendrites (per FOV)")
    a.set_title("d  significantly coupled: red +, blue -")
    # e global r
    a = ax[1, 1]
    for i, k in enumerate(BEHS):
        if f"global_r_{k}" in runs:
            v = runs[f"global_r_{k}"]
            sig = runs[f"global_p_{k}"] < 0.05
            x = np.full(len(v), i) + np.random.default_rng(i).normal(0, 0.05, len(v))
            a.scatter(x[sig], v[sig], color="k", s=12)
            a.scatter(x[~sig], v[~sig], facecolors="none", edgecolors="k", s=12)
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks(range(3), BEHS)
    a.set_ylabel("global Ca r (best lag)")
    a.set_title("e  global Ca vs behavior per run (filled p<0.05)")
    # f depth
    a = ax[1, 2]
    d = dend.dropna(subset=["depth_um"])
    a.scatter(d.depth_um, d.event_rate_per_min, s=3, alpha=0.4, c=pd.factorize(d.fov)[0], cmap="tab20")
    a.set_xlabel("depth below top of FOV (um)")
    a.set_ylabel("event rate (/min)")
    a.set_title("f  event rate vs depth")
    fig.suptitle(f"Cohort statistics - {source} dendrites ({len(runs)} runs, {runs.fov.nunique()} FOVs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_cohort.png", dpi=140)
    fig.savefig(out / "fig_cohort.pdf")
    plt.close(fig)


def compare(S_auto, S_human):
    ra = pd.read_csv(STATS / "auto" / "per_run.csv")
    rh = pd.read_csv(STATS / "human" / "per_run.csv")
    m = ra.merge(rh, on=["run", "fov", "mouse"], suffixes=("_auto", "_human"))
    cols = ["n_dendrites", "mean_pair_r", "frac_pairs_sig", "rho_r_distance", "pr_over_n", "participation_ratio", "event_rate_per_min",
            "frac_global_events", "global_r_accel", "global_r_pupil", "global_r_whisker", "frac_coupled_accel",
            "frac_coupled_pupil", "frac_coupled_whisker", "median_r_accel", "median_r_pupil", "median_r_whisker",
            "pair_r_active", "pair_r_quiet"]
    rows = []
    fa = m.groupby("fov").mean(numeric_only=True)
    for c in cols:
        a, h = f"{c}_auto", f"{c}_human"
        if a not in m or h not in m:
            continue
        d = m[[a, h]].dropna()
        df_ = fa[[a, h]].dropna()
        if len(d) < 4:
            continue
        rho = sp.spearmanr(d[a], d[h])[0]
        w = sp.wilcoxon(df_[a], df_[h]).pvalue if len(df_) >= 4 and (df_[a] != df_[h]).any() else np.nan
        rows.append(dict(metric=c, n_runs=len(d), n_fovs=len(df_), auto_median=float(d[a].median()), human_median=float(d[h].median()),
                         spearman_across_runs=float(rho), wilcoxon_p_fov=float(w),
                         same_sign_frac=float((np.sign(d[a]) == np.sign(d[h])).mean())))
    cmp = pd.DataFrame(rows)
    cmp.to_csv(STATS / "compare_auto_vs_human.csv", index=False)
    # conclusions side by side
    ta = {t["name"]: t for t in S_auto["tests"]}
    th = {t["name"]: t for t in S_human["tests"]}
    lines = ["Automatic vs human-curated dendrites: do they lead to the same conclusions?", ""]
    for k in th:
        if k in ta:
            lines += [f"[{k}]", f"  human: {th[k].get('conclusion', '')}", f"  auto : {ta[k].get('conclusion', '')}", ""]
    lines += ["Run-level agreement (Spearman across runs; paired Wilcoxon over FOVs).",
              "Note: auto finds ~3-4x more dendrites per run; pr_over_n falls with N by construction, so compare",
              "participation_ratio and the FOV-level conclusions rather than pr_over_n. global_r_* use the same",
              "whole-field trace for both sources (identical by design).", ""]
    for _, x in cmp.iterrows():
        lines.append(f"  {x.metric:22s} auto {x.auto_median:8.3f}  human {x.human_median:8.3f}  rho {x.spearman_across_runs:5.2f}  p_diff {x.wilcoxon_p_fov:.2g}")
    (STATS / "compare_auto_vs_human.txt").write_text("\n".join(lines))
    (STATS / "compare_auto_vs_human.json").write_text(cmp.to_json(orient="records", indent=2))
    show = [c for c in ("mean_pair_r", "event_rate_per_min", "frac_global_events", "pr_over_n", "global_r_accel", "frac_coupled_accel", "median_r_whisker", "rho_r_distance") if f"{c}_auto" in m]
    fig, ax = plt.subplots(2, 4, figsize=(13, 6))
    for a, c in zip(ax.ravel(), show):
        a.scatter(m[f"{c}_human"], m[f"{c}_auto"], s=14, c=pd.factorize(m.fov)[0], cmap="tab20")
        lo = np.nanmin(m[[f"{c}_human", f"{c}_auto"]].to_numpy())
        hi = np.nanmax(m[[f"{c}_human", f"{c}_auto"]].to_numpy())
        a.plot([lo, hi], [lo, hi], "k--", lw=0.6)
        rr = cmp.set_index("metric").spearman_across_runs.get(c, np.nan)
        a.set_title(f"{c}  (rho {rr:.2f})", fontsize=8)
        a.set_xlabel("human")
        a.set_ylabel("auto")
    fig.suptitle("Run-level statistics: automatic vs human-curated dendrites (colour = FOV)")
    fig.tight_layout()
    fig.savefig(STATS / "compare_auto_vs_human.png", dpi=140)
    fig.savefig(STATS / "compare_auto_vs_human.pdf")
    plt.close(fig)
    return cmp


def run_source(source, keys, jobs):
    out = STATS / source
    out.mkdir(parents=True, exist_ok=True)
    res = []
    with ProcessPoolExecutor(max(1, min(jobs, 6))) as ex:
        futs = {ex.submit(analyze_run, k, source): k for k in keys}
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
    print(f"[{source}] analysed {len(res)} runs")
    return cohort(source, res, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("auto", "human", "both"), default="both")
    ap.add_argument("--all", action="store_true", help="all runs (default when no --run)")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--force", action="store_true", help="accepted for CLI symmetry; stats are always recomputed")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    keys = a.run or [r.key for r in discover_runs()]
    t0 = time.time()
    S = {}
    for s in (("auto", "human") if a.source == "both" else (a.source,)):
        S[s] = run_source(s, keys, a.jobs)
        print((STATS / s / "stats_summary.txt").read_text())
    if len(S) == 2:
        compare(S["auto"], S["human"])
        print((STATS / "compare_auto_vs_human.txt").read_text())
    print(f"stats done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
