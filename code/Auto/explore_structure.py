#!/usr/bin/env python
"""explore_structure.py - spatial / anatomical structure & behavior angle on the question
"are L5 pyramidal apical dendrites independent computational units?" (code/Auto/).

This is a stand-alone exploratory battery that is orthogonal (where possible) to the existing
independence_tests.py / coherence_behavior.py.  It asks whether the COUPLING and the BEHAVIOR
TUNING of dendrites are organised by anatomy in ways a bag of independent units would not predict:

  S1 trunk/column    : are vertically aligned dendrites (same lateral x-z column, different
                       cortical depth = candidate segments of one apical trunk) more coupled
                       than distance-matched non-aligned pairs?  Is their coupling SEQUENTIAL
                       (peak cross-correlation at a non-zero lag), and does the lag sign reveal
                       a propagation direction along the trunk (surface<->deep)?
  S2 depth bands     : split each FOV into superficial / middle / deep thirds and compare event
                       rate, event amplitude, behaviour coupling (accel/whisker/pupil, best lag
                       within +-1 s), global-event participation, and within- vs between-band r.
  S3 behaviour-spec. : conditional / partial coupling - are there dendrites coupled to whisking
                       but NOT locomotion (or vice-versa) beyond what shared arousal predicts?
                       Ca vs whisker during still frames only; Ca vs pupil during no-whisk frames;
                       partial correlation of each dendrite with each behaviour given the others.
  S4 lateral grad.   : event rate / behaviour coupling vs lateral x position (gradients, L-R).
  S5 negative pairs  : pairs significantly anti-correlated below the shift null - distance, depth,
                       behaviour-tuning similarity (mutual inhibition / competition would surprise).
  S6 func. clusters  : cluster residual traces (global Ca + behaviour regressed out); are clusters
                       spatially compact vs a label-permutation null?

Replication: FOV (scape_common.fov_group) is the unit; runs of one FOV are averaged before the
across-FOV test (Wilcoxon signed-rank unless noted).  Nulls: independent circular shifts >= 10 s.
BH-FDR within families of many per-dendrite / per-pair tests.  5 Hz sampling: lags < 0.2 s unresolved.

Outputs: scape-auto/stats/explore_structure/<source>/{*.csv, summary.txt, fig_*.png/pdf}
CLI: explore_structure.py [--source auto|human]   (default auto)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, fov_group, load_behavior, resample_to, human_masks, human_masks_valid,
    open_stack, VOXEL_ZYX, AUTO_ROOT,
)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore")

FR = 5.0                 # Hz (all runs)
MINSHIFT = int(round(10 * FR))   # >= 10 s circular shift = 50 frames
NSH_PAIR = 200           # shifts for per-pair significance
NSH_BEH = 200            # shifts for per-dendrite behaviour significance
NPERM = 500              # label permutations for cluster compactness
ZTHR = 3.0               # robust-z event threshold
MAXLAG = int(round(3.0 * FR))    # +-3 s for cross-correlation
BEHLAG = int(round(1.0 * FR))    # +-1 s for behaviour best-lag
COL_LAT = 15.0           # um: lateral radius defining a vertical column (same-trunk candidate)
BEHAVIORS = ("accel", "whisker", "pupil")


# ----------------------------------------------------------------------------- small helpers
def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pp = p[ok]
    n = pp.size
    if n == 0:
        return q
    o = np.argsort(pp)
    adj = np.minimum.accumulate((pp[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1)
    q[ok] = out
    return q


def rz(M):
    """robust z-score per column (median / MAD)."""
    med = np.median(M, 0)
    mad = np.median(np.abs(M - med), 0) * 1.4826 + 1e-9
    return (M - med) / mad


def zc(M):
    """zero-mean unit-norm columns (for correlation via matmul)."""
    M = M - M.mean(0)
    return M / (np.linalg.norm(M, axis=0) + 1e-12)


def corr_cols(M):
    Z = zc(M)
    return Z.T @ Z


def shift_cols(M, rng):
    """independent circular shift (>= MINSHIFT) of every column."""
    T = M.shape[0]
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        out[:, j] = np.roll(M[:, j], rng.integers(MINSHIFT, T - MINSHIFT))
    return out


def event_onsets(z):
    """z: (T,) robust-z.  onsets of runs (z>ZTHR) lasting >= 2 frames."""
    a = np.r_[False, z > ZTHR, False]
    d = np.diff(a.astype(int))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return st[(en - st) >= 2]


def wilcoxon(x, mu=0.0):
    """Wilcoxon signed-rank of (x-mu) vs 0; returns (median, p, n)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    d = x - mu
    d = d[d != 0]
    if d.size < 1:
        return (float(np.median(x)) if n else np.nan, np.nan, n)
    if d.size < 2:
        return (float(np.median(x)), np.nan, n)
    try:
        p = sp.wilcoxon(d)[1]
    except Exception:
        p = np.nan
    return (float(np.median(x)), float(p), n)


def best_lag_corr(y, x, maxlag):
    """max |Pearson| of y vs x over integer lags in [-maxlag, maxlag]; returns (r_at_best, lag)."""
    y = y - np.nanmean(y)
    x = x - np.nanmean(x)
    best_r, best_l = 0.0, 0
    for L in range(-maxlag, maxlag + 1):
        if L >= 0:
            a, b = y[L:], x[:len(x) - L] if L > 0 else x
        else:
            a, b = y[:L], x[-L:]
        if a.size < 10:
            continue
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        r = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
        if abs(r) > abs(best_r):
            best_r, best_l = r, L
    return best_r, best_l


def partial_resid(Y, C):
    """residual of columns of Y after regressing out design C (T,k) incl. constant handled here."""
    D = np.c_[np.ones(len(Y)), C]
    beta = np.linalg.lstsq(D, Y, rcond=None)[0]
    return Y - D @ beta


# ----------------------------------------------------------------------------- per-run loading
class RunData:
    pass


def load_run(r, source):
    """Return RunData with traces, centroids (um), behaviour & global Ca on the Ca clock, events."""
    if source == "auto":
        df = pd.read_csv(r.out / "traces" / "dff_auto.csv")
        mk = pd.read_csv(r.out / "masks" / "auto_masks.csv")
        cen = {row["name"]: (row["cz"] * VOXEL_ZYX[0], row["cy"] * VOXEL_ZYX[1], row["cx"] * VOXEL_ZYX[2])
               for _, row in mk.iterrows()}
    else:
        if not human_masks_valid(r):
            return None
        df = pd.read_csv(r.out / "traces" / "dff_human_sameextractor.csv")
        shp = open_stack(r).shape[1:]
        cen = {}
        for name, m in human_masks(r, shp):
            idx = np.argwhere(m)
            if idx.size:
                c = idx.mean(0)
                cen[name] = (c[0] * VOXEL_ZYX[0], c[1] * VOXEL_ZYX[1], c[2] * VOXEL_ZYX[2])

    tcols = [c for c in df.columns if c.startswith("dend_") and c in cen]
    if len(tcols) < 6:
        return None
    M = df[tcols].to_numpy(float)
    good = np.isfinite(M).all(0) & (M.std(0) > 1e-6)
    tcols = [c for c, g in zip(tcols, good) if g]
    M = M[:, good]
    ca_t = df["time_s"].to_numpy(float)
    T, N = M.shape
    if N < 6 or T < 100:
        return None

    cz = np.array([cen[c][0] for c in tcols])
    cy = np.array([cen[c][1] for c in tcols])   # cortical depth (um); row 0 = surface
    cx = np.array([cen[c][2] for c in tcols])   # lateral (um)

    # global Ca
    g = pd.read_csv(r.out / "traces" / "global_ca.csv")
    glob = resample_to(g["time_s"].to_numpy(float), g["global_dff"].to_numpy(float), ca_t)
    if glob is None or not np.isfinite(glob).all():
        glob = np.nan_to_num(glob, nan=np.nanmean(glob)) if glob is not None else M.mean(1)

    # behaviour
    beh = {}
    b = load_behavior(r)
    for key in BEHAVIORS:
        v = resample_to(b[key + "_t"], b[key], ca_t)
        if v is not None and np.isfinite(v).sum() > 0.8 * T:
            v = np.where(np.isfinite(v), v, np.nanmean(v))
            beh[key] = v
    has_beh = len(beh) == 3

    # events
    Z = rz(M)
    onsets = [event_onsets(Z[:, j]) for j in range(N)]
    dur_min = T / FR / 60.0
    ev_rate = np.array([len(o) / dur_min for o in onsets])
    ev_amp = np.array([float(np.mean([Z[o, j].max() if o.size else 0 for o in [onsets[j]]])) if onsets[j].size
                       else 0.0 for j in range(N)])
    # amplitude = mean peak robust-z across that dendrite's events
    ev_amp = np.array([float(np.mean([Z[s:s + 3, j].max() for s in onsets[j]])) if onsets[j].size else 0.0
                       for j in range(N)])
    active = Z > ZTHR

    # global events
    gz = rz(glob[:, None])[:, 0]
    gons = event_onsets(gz)

    rd = RunData()
    rd.key = r.key
    rd.fov = fov_group(r)
    rd.mouse = r.mouse
    rd.names = tcols
    rd.M = M
    rd.Z = Z
    rd.active = active
    rd.cz, rd.cy, rd.cx = cz, cy, cx
    rd.glob = glob
    rd.gons = gons
    rd.beh = beh
    rd.has_beh = has_beh
    rd.onsets = onsets
    rd.ev_rate = ev_rate
    rd.ev_amp = ev_amp
    rd.T, rd.N = T, N
    return rd


def pair_geometry(rd):
    """return (iu, ju, dist3d, latdist, depthgap) for upper-triangle pairs (um)."""
    N = rd.N
    iu, ju = np.triu_indices(N, 1)
    dz = rd.cz[iu] - rd.cz[ju]
    dy = rd.cy[iu] - rd.cy[ju]
    dx = rd.cx[iu] - rd.cx[ju]
    dist3d = np.sqrt(dz**2 + dy**2 + dx**2)
    latdist = np.sqrt(dz**2 + dx**2)   # perpendicular to cortical-depth axis
    depthgap = np.abs(dy)
    return iu, ju, dist3d, latdist, depthgap


def pair_corr_and_null(rd, seed):
    """observed upper-tri r and one-sided shift-null p for positive & negative coupling."""
    rng = np.random.default_rng(seed)
    N = rd.N
    iu, ju = np.triu_indices(N, 1)
    C = corr_cols(rd.M)
    obs = C[iu, ju]
    ge = np.zeros(obs.size)   # null >= obs  (positive)
    le = np.zeros(obs.size)   # null <= obs  (negative)
    snull = np.zeros(obs.size)
    for _ in range(NSH_PAIR):
        Cs = corr_cols(shift_cols(rd.M, rng))
        n = Cs[iu, ju]
        ge += n >= obs
        le += n <= obs
        snull += n
    p_pos = (ge + 1) / (NSH_PAIR + 1)
    p_neg = (le + 1) / (NSH_PAIR + 1)
    null_mean = snull / NSH_PAIR
    return obs, null_mean, p_pos, p_neg


# =============================================================================  S1 trunk/column
def analysis_trunk(rds, lines, outdir, source):
    rows = []
    per_fov_diff = defaultdict(list)       # distance-matched aligned-minus-nonaligned r
    per_fov_trunk_excess = defaultdict(list)
    seq_deeper_leads = []                  # pooled across trunk sequential pairs (descriptive)
    seq_fov = defaultdict(list)            # per-FOV list of deeper-leads bools (replicated test)
    seq_speeds = []
    bins = np.arange(0, 300, 25.0)
    for rd in rds:
        iu, ju, dist3d, latdist, depthgap = pair_geometry(rd)
        C = corr_cols(rd.M)
        r = C[iu, ju]
        aligned = latdist < COL_LAT          # vertically aligned (column / trunk candidate)
        trunk = aligned & (depthgap > 30)
        # distance-matched aligned vs non-aligned
        diffs = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            inb = (dist3d >= b0) & (dist3d < b1)
            a = r[inb & aligned]
            na = r[inb & ~aligned]
            if a.size >= 3 and na.size >= 10:
                diffs.append(np.nanmean(a) - np.nanmean(na))
        if diffs:
            per_fov_diff[rd.fov].append(float(np.nanmean(diffs)))
        # trunk-pair mean r vs same-depth lateral neighbour mean r (both near-range)
        latnb = (depthgap < 15) & (latdist >= COL_LAT) & (latdist < 60)
        if trunk.sum() >= 3:
            per_fov_trunk_excess[rd.fov].append(float(np.nanmean(r[trunk]) -
                                                      (np.nanmean(r[latnb]) if latnb.sum() >= 3 else np.nanmean(r))))
        # sequential propagation among trunk pairs (peak lag != 0)
        tp = np.flatnonzero(trunk)
        if tp.size:
            Zc = zc(rd.M)
            for idx in tp:
                i, j = iu[idx], ju[idx]
                best_r, best_l = 0.0, 0
                for L in range(-MAXLAG, MAXLAG + 1):
                    if L >= 0:
                        a, b = Zc[L:, i], Zc[:rd.T - L, j]
                    else:
                        a, b = Zc[:rd.T + L, i], Zc[-L:, j]
                    cc = float(a @ b)
                    if abs(cc) > abs(best_r):
                        best_r, best_l = cc, L
                if abs(best_r) > 0.2 and best_l != 0:
                    # positive lag L means col j leads col i (i(t) ~ j(t-? )): we aligned a=i[L:], b=j[:T-L]
                    # i.e. i lagged by L vs j -> j leads i by L frames when L>0.
                    leader, follower = (j, i) if best_l > 0 else (i, j)
                    deeper_leads = rd.cy[leader] > rd.cy[follower]   # larger cy = deeper
                    seq_deeper_leads.append(bool(deeper_leads))
                    seq_fov[rd.fov].append(bool(deeper_leads))
                    seq_speeds.append(depthgap[idx] / (abs(best_l) / FR))
        rows.append(dict(run=rd.key, fov=rd.fov, n_trunk=int(trunk.sum()),
                         r_trunk=float(np.nanmean(r[trunk])) if trunk.sum() else np.nan,
                         r_latnb=float(np.nanmean(r[latnb])) if latnb.sum() else np.nan,
                         r_all=float(np.nanmean(r))))
    pd.DataFrame(rows).to_csv(outdir / "s1_trunk_per_run.csv", index=False)

    diff_fov = np.array([np.mean(v) for v in per_fov_diff.values()])
    med, p, n = wilcoxon(diff_fov, 0)
    lines.append(f"[S1_column_coupling] distance-matched (aligned lat<{COL_LAT:.0f}um) minus non-aligned r: "
                 f"median {med:+.4f} over {n} FOVs (Wilcoxon p={p:.3g}). "
                 f"{'Vertically aligned dendrites are MORE coupled at the same 3D distance.' if (p==p and p<0.05 and med>0) else 'No excess coupling for vertical alignment beyond distance.'}")
    tex = np.array([np.mean(v) for v in per_fov_trunk_excess.values()])
    med2, p2, n2 = wilcoxon(tex, 0)
    lines.append(f"[S1_trunk_vs_lateral] trunk pairs (lat<{COL_LAT:.0f}um, depthgap>30um) r minus same-depth "
                 f"lateral-neighbour r: median {med2:+.4f} over {n2} FOVs (Wilcoxon p={p2:.3g}).")
    if seq_deeper_leads:
        k = int(np.sum(seq_deeper_leads)); tot = len(seq_deeper_leads)
        fov_frac = np.array([np.mean(v) for v in seq_fov.values() if len(v) >= 5])
        med, p, n = wilcoxon(fov_frac, 0.5)
        lines.append(f"[S1_propagation] {tot} trunk pairs are sequential (|peak xcorr|>0.2 at non-zero lag); "
                     f"deeper segment leads in {k}/{tot} pooled ({100*k/tot:.0f}%); per-FOV deeper-leads fraction "
                     f"median {med:.2f} over {n} FOVs (Wilcoxon vs 0.5 p={p:.3g}). "
                     f"median apparent speed {np.median(seq_speeds):.0f} um/s. "
                     f"{'Superficial segment tends to lead (top-down) along candidate trunks.' if (p==p and p<0.05 and med<0.5) else ('Deeper segment tends to lead (bottom-up).' if (p==p and p<0.05 and med>0.5) else 'No consistent up/down direction across FOVs.')}")
    else:
        lines.append("[S1_propagation] no sequential trunk pairs above threshold (null).")

    # figure
    try:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].axhline(0, color="k", lw=.5)
        ax[0].bar(range(len(diff_fov)), np.sort(diff_fov), color="#4477aa")
        ax[0].set(title="aligned-minus-nonaligned r\n(distance matched, per FOV)", xlabel="FOV", ylabel="Δr")
        if seq_deeper_leads:
            ax[1].bar(["deeper\nleads", "superf.\nleads"], [np.sum(seq_deeper_leads),
                       len(seq_deeper_leads) - np.sum(seq_deeper_leads)], color=["#cc6677", "#88ccee"])
            ax[1].set(title="trunk sequential pairs: leader depth", ylabel="n pairs")
        fig.tight_layout(); fig.savefig(outdir / "fig_s1_trunk.png", dpi=120); fig.savefig(outdir / "fig_s1_trunk.pdf"); plt.close(fig)
    except Exception as e:
        lines.append(f"[S1 figure error] {e}")


# =============================================================================  S2 depth bands
def analysis_depth(rds, lines, outdir, source):
    rows = []
    # continuous per-FOV spearman of metric vs depth
    metrics = ["ev_rate", "ev_amp", "participation", "r_accel", "r_whisker", "r_pupil"]
    sp_fov = {m: defaultdict(list) for m in metrics}
    within_between = defaultdict(list)
    band_vals = {m: {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list)} for m in metrics}
    for rd in rds:
        N = rd.N
        depth = rd.cy
        # participation in global events
        part = np.zeros(N)
        if rd.gons.size:
            win = 2
            for j in range(N):
                hit = 0
                for s in rd.gons:
                    lo, hi = max(0, s - win), min(rd.T, s + win + 1)
                    if rd.active[lo:hi, j].any():
                        hit += 1
                part[j] = hit / rd.gons.size
        # behaviour best-lag corr magnitude per dendrite
        rbeh = {k: np.full(N, np.nan) for k in BEHAVIORS}
        for k, v in rd.beh.items():
            for j in range(N):
                rbeh[k][j] = abs(best_lag_corr(rd.M[:, j], v, BEHLAG)[0])
        mvals = dict(ev_rate=rd.ev_rate, ev_amp=rd.ev_amp, participation=part,
                     r_accel=rbeh["accel"], r_whisker=rbeh["whisker"], r_pupil=rbeh["pupil"])
        for m in metrics:
            y = mvals[m]
            ok = np.isfinite(y)
            if ok.sum() >= 6 and np.nanstd(y[ok]) > 0:
                rho = sp.spearmanr(depth[ok], y[ok])[0]
                sp_fov[m][rd.fov].append(rho)
            # band means (tertiles)
            q = np.quantile(depth, [1/3, 2/3])
            band = np.digitize(depth, q)
            for bb in (0, 1, 2):
                sel = (band == bb) & ok
                if sel.sum():
                    band_vals[m][bb][rd.fov].append(float(np.nanmean(y[sel])))
        # within vs between band r
        C = corr_cols(rd.M)
        q = np.quantile(depth, [1/3, 2/3])
        band = np.digitize(depth, q)
        iu, ju = np.triu_indices(N, 1)
        same = band[iu] == band[ju]
        within_between[rd.fov].append(float(np.nanmean(C[iu, ju][same]) - np.nanmean(C[iu, ju][~same])))
        rows.append(dict(run=rd.key, fov=rd.fov,
                         rate_sup=np.nanmean(rd.ev_rate[band == 0]), rate_deep=np.nanmean(rd.ev_rate[band == 2]),
                         amp_sup=np.nanmean(rd.ev_amp[band == 0]), amp_deep=np.nanmean(rd.ev_amp[band == 2])))
    pd.DataFrame(rows).to_csv(outdir / "s2_depth_per_run.csv", index=False)

    ps = []
    stmts = []
    for m in metrics:
        vals = np.array([np.mean(v) for v in sp_fov[m].values()]) if sp_fov[m] else np.array([])
        med, p, n = wilcoxon(vals, 0)
        ps.append(p)
        stmts.append((m, med, p, n))
    q = bh([s[2] for s in stmts])
    lines.append(f"[S2_depth_gradients] Spearman(metric vs cortical depth) per FOV, Wilcoxon vs 0 (BH over {len(stmts)} metrics):")
    for (m, med, p, n), qq in zip(stmts, q):
        sign = "deeper↑" if med > 0 else "deeper↓"
        sig = "sig" if (qq == qq and qq < 0.05) else "ns"
        lines.append(f"    {m:14s} rho_med={med:+.3f} ({sign}), p={p:.3g}, BHq={qq:.3g} [{sig}], {n} FOVs")
    wb = np.array([np.mean(v) for v in within_between.values()])
    med, p, n = wilcoxon(wb, 0)
    lines.append(f"[S2_within_vs_between_band] within-depth-band r minus between-band r: median {med:+.4f} "
                 f"over {n} FOVs (Wilcoxon p={p:.3g}). "
                 f"{'Coupling is organised by depth band.' if (p==p and p<0.05 and med>0) else 'No depth-band organisation of coupling.'}")

    # figure: band means for key metrics
    try:
        keym = ["ev_rate", "ev_amp", "participation", "r_whisker"]
        fig, axs = plt.subplots(1, len(keym), figsize=(3 * len(keym), 3))
        for ax, m in zip(axs, keym):
            means = []
            for bb in (0, 1, 2):
                fovm = [np.mean(v) for v in band_vals[m][bb].values()]
                means.append(fovm)
            bp_data = [np.array(x) for x in means]
            ax.boxplot(bp_data, labels=["sup", "mid", "deep"], showmeans=True)
            ax.set(title=m)
        fig.suptitle("depth bands (per-FOV means)")
        fig.tight_layout(); fig.savefig(outdir / "fig_s2_depth.png", dpi=120); fig.savefig(outdir / "fig_s2_depth.pdf"); plt.close(fig)
    except Exception as e:
        lines.append(f"[S2 figure error] {e}")


# =============================================================================  S3 behaviour-specific
def corr_shift_pvals(Yfull, xfull, mask, seed):
    """obs corr of each column of Yfull with xfull on masked rows; two-sided shift-null p."""
    rng = np.random.default_rng(seed)
    Y = Yfull[mask]
    x = xfull[mask]
    Zy = zc(Y)
    zx = (x - x.mean()); zx /= (np.linalg.norm(zx) + 1e-12)
    obs = Zy.T @ zx
    cnt = np.zeros(obs.size)
    for _ in range(NSH_BEH):
        xs = np.roll(xfull, rng.integers(MINSHIFT, len(xfull) - MINSHIFT))[mask]
        zxs = xs - xs.mean(); zxs /= (np.linalg.norm(zxs) + 1e-12)
        n = Zy.T @ zxs
        cnt += np.abs(n) >= np.abs(obs)
    p = (cnt + 1) / (NSH_BEH + 1)
    return obs, p


def analysis_behavior(rds, lines, outdir, source):
    rds = [rd for rd in rds if rd.has_beh]
    if not rds:
        lines.append("[S3_behaviour] no runs with all three behaviours (null).")
        return
    rows = []
    frac_whisk_survive = defaultdict(list)   # whisk-coupled dendrites still sig in still-only frames
    frac_pupil_survive = defaultdict(list)   # pupil-coupled still sig in no-whisk frames
    frac_whisk_spec = defaultdict(list)      # whisk-specific (partial whisk sig, partial accel ns)
    frac_accel_spec = defaultdict(list)
    frac_pupil_spec = defaultdict(list)
    for ri, rd in enumerate(rds):
        T, N = rd.T, rd.N
        allmask = np.ones(T, bool)
        accel, whisk, pupil = rd.beh["accel"], rd.beh["whisker"], rd.beh["pupil"]
        # unconditional coupling + sig
        r_w, p_w = corr_shift_pvals(rd.M, whisk, allmask, 100 + ri)
        r_a, p_a = corr_shift_pvals(rd.M, accel, allmask, 200 + ri)
        r_p, p_p = corr_shift_pvals(rd.M, pupil, allmask, 300 + ri)
        q_w, q_a, q_p = bh(p_w), bh(p_a), bh(p_p)
        # conditional: still frames (accel low third); whisker coupling there
        still = accel < np.quantile(accel, 1/3)
        r_w_still, p_w_still = corr_shift_pvals(rd.M, whisk, still, 400 + ri)
        q_w_still = bh(p_w_still)
        # conditional: no-whisk frames (whisker low third); pupil coupling there
        nowhisk = whisk < np.quantile(whisk, 1/3)
        r_p_nw, p_p_nw = corr_shift_pvals(rd.M, pupil, nowhisk, 500 + ri)
        q_p_nw = bh(p_p_nw)
        # partial correlations (each behaviour given the other two)
        res_w = partial_resid(whisk[:, None], np.c_[accel, pupil])[:, 0]
        res_a = partial_resid(accel[:, None], np.c_[whisk, pupil])[:, 0]
        res_p = partial_resid(pupil[:, None], np.c_[accel, whisk])[:, 0]
        Yw = partial_resid(rd.M, np.c_[accel, pupil])
        Ya = partial_resid(rd.M, np.c_[whisk, pupil])
        Yp = partial_resid(rd.M, np.c_[accel, whisk])
        pr_w, pp_w = corr_shift_pvals(Yw, res_w, allmask, 600 + ri)
        pr_a, pp_a = corr_shift_pvals(Ya, res_a, allmask, 700 + ri)
        pr_p, pp_p = corr_shift_pvals(Yp, res_p, allmask, 800 + ri)
        qpw, qpa, qpp = bh(pp_w), bh(pp_a), bh(pp_p)

        wcoup = q_w < 0.05
        if wcoup.sum() >= 3:
            frac_whisk_survive[rd.fov].append(float(np.mean(q_w_still[wcoup] < 0.05)))
        pcoup = q_p < 0.05
        if pcoup.sum() >= 3:
            frac_pupil_survive[rd.fov].append(float(np.mean(q_p_nw[pcoup] < 0.05)))
        wspec = (qpw < 0.05) & (qpa >= 0.05)
        aspec = (qpa < 0.05) & (qpw >= 0.05)
        pspec = (qpp < 0.05) & (qpw >= 0.05) & (qpa >= 0.05)
        frac_whisk_spec[rd.fov].append(float(np.mean(wspec)))
        frac_accel_spec[rd.fov].append(float(np.mean(aspec)))
        frac_pupil_spec[rd.fov].append(float(np.mean(pspec)))
        for j in range(N):
            rows.append(dict(run=rd.key, fov=rd.fov, dend=rd.names[j], depth_um=rd.cy[j], latx_um=rd.cx[j],
                             r_whisk=r_w[j], q_whisk=q_w[j], r_accel=r_a[j], q_accel=q_a[j],
                             r_pupil=r_p[j], q_pupil=q_p[j],
                             partial_whisk=pr_w[j], q_partial_whisk=qpw[j],
                             partial_accel=pr_a[j], q_partial_accel=qpa[j],
                             partial_pupil=pr_p[j], q_partial_pupil=qpp[j]))
    pd.DataFrame(rows).to_csv(outdir / "s3_behavior_per_dendrite.csv", index=False)

    def rep(d):
        return np.array([np.mean(v) for v in d.values()])
    for name, d, msg in [
        ("whisk_survives_still", frac_whisk_survive,
         "whisking-coupled dendrites still significant when only STILL (low-accel) frames are used"),
        ("pupil_survives_nowhisk", frac_pupil_survive,
         "pupil-coupled dendrites still significant when only NO-WHISK (low-whisker) frames are used")]:
        vals = rep(d)
        med, p, n = wilcoxon(vals, 0)
        lines.append(f"[S3_{name}] fraction of {msg}: median {100*med:.0f}% over {n} FOVs. "
                     f"{'Coupling is NOT fully explained by the other behaviour (survives conditioning).' if med>0.3 else 'Coupling largely vanishes under conditioning (shared-arousal driven).'}")
    for name, d, lab in [("whisk_specific", frac_whisk_spec, "whisking but not locomotion"),
                         ("accel_specific", frac_accel_spec, "locomotion but not whisking"),
                         ("pupil_specific", frac_pupil_spec, "pupil but neither whisk nor accel")]:
        vals = rep(d)
        med, p, n = wilcoxon(vals, 0)
        lines.append(f"[S3_{name}] fraction of dendrites with PARTIAL coupling to {lab}: "
                     f"median {100*med:.1f}% over {n} FOVs (Wilcoxon vs 0 p={p:.3g}).")

    # figure: partial whisk vs partial accel scatter (pooled)
    try:
        df = pd.read_csv(outdir / "s3_behavior_per_dendrite.csv")
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
        ax[0].axhline(0, color="k", lw=.4); ax[0].axvline(0, color="k", lw=.4)
        ax[0].scatter(df["partial_accel"], df["partial_whisk"], s=6, alpha=.4,
                      c=(df["q_partial_whisk"] < 0.05).map({True: "#cc3311", False: "#bbbbbb"}))
        ax[0].set(xlabel="partial r: locomotion", ylabel="partial r: whisking",
                  title="behaviour-specific coupling\n(red = whisk-partial sig)")
        wspec = ((df["q_partial_whisk"] < .05) & (df["q_partial_accel"] >= .05)).mean()
        aspec = ((df["q_partial_accel"] < .05) & (df["q_partial_whisk"] >= .05)).mean()
        both = ((df["q_partial_accel"] < .05) & (df["q_partial_whisk"] < .05)).mean()
        ax[1].bar(["whisk\nonly", "loco\nonly", "both"], [wspec, aspec, both],
                  color=["#ee6677", "#4477aa", "#228833"])
        ax[1].set(ylabel="fraction of dendrites", title="partial-coupling specificity")
        fig.tight_layout(); fig.savefig(outdir / "fig_s3_behavior.png", dpi=120); fig.savefig(outdir / "fig_s3_behavior.pdf"); plt.close(fig)
    except Exception as e:
        lines.append(f"[S3 figure error] {e}")


# =============================================================================  S4 lateral gradient
def analysis_lateral(rds, lines, outdir, source):
    rows = []
    sp_rate = defaultdict(list)
    sp_beh = defaultdict(list)
    for rd in rds:
        ok = np.isfinite(rd.ev_rate)
        if ok.sum() >= 6 and np.nanstd(rd.cx[ok]) > 0:
            sp_rate[rd.fov].append(sp.spearmanr(rd.cx[ok], rd.ev_rate[ok])[0])
        if rd.has_beh:
            acc = rd.beh["accel"]
            rb = np.array([abs(best_lag_corr(rd.M[:, j], acc, BEHLAG)[0]) for j in range(rd.N)])
            if np.nanstd(rb) > 0:
                sp_beh[rd.fov].append(sp.spearmanr(rd.cx, rb)[0])
        rows.append(dict(run=rd.key, fov=rd.fov, latx_mean=float(np.mean(rd.cx)),
                         latx_range=float(rd.cx.max() - rd.cx.min())))
    pd.DataFrame(rows).to_csv(outdir / "s4_lateral_per_run.csv", index=False)
    for name, d, lab in [("rate_vs_latx", sp_rate, "event rate vs lateral x"),
                         ("accelcoup_vs_latx", sp_beh, "locomotion coupling vs lateral x")]:
        vals = np.array([np.mean(v) for v in d.values()]) if d else np.array([])
        med, p, n = wilcoxon(vals, 0)
        lines.append(f"[S4_{name}] Spearman {lab}: median rho {med:+.3f} over {n} FOVs (Wilcoxon p={p:.3g}). "
                     f"{'A lateral gradient is present.' if (p==p and p<0.05) else 'No lateral gradient.'}")


# =============================================================================  S5 negative pairs
def analysis_negative(rds, lines, outdir, source, cache):
    rows = []
    frac_neg_fov = defaultdict(list)
    dist_neg, dist_pos = [], []
    depth_neg, depth_pos = [], []
    behsim_neg, behsim_pos = [], []
    # per-FOV paired medians for replicated tests
    fov_dist = defaultdict(lambda: {"neg": [], "pos": []})
    fov_bsim = defaultdict(lambda: {"neg": [], "pos": []})
    for rd in rds:
        obs, null_mean, p_pos, p_neg = cache[rd.key]
        iu, ju, dist3d, latdist, depthgap = pair_geometry(rd)
        sig_neg = (p_neg < 0.025) & (obs < 0)
        sig_pos = (p_pos < 0.025) & (obs > 0)
        frac_neg_fov[rd.fov].append(float(np.mean(sig_neg)))
        dist_neg += list(dist3d[sig_neg]); dist_pos += list(dist3d[sig_pos])
        depth_neg += list(depthgap[sig_neg]); depth_pos += list(depthgap[sig_pos])
        fov_dist[rd.fov]["neg"] += list(dist3d[sig_neg]); fov_dist[rd.fov]["pos"] += list(dist3d[sig_pos])
        # behaviour-tuning similarity of each pair (cosine of accel/whisk/pupil best-lag r vectors)
        if rd.has_beh:
            tune = np.zeros((rd.N, 3))
            for bi, k in enumerate(BEHAVIORS):
                v = rd.beh[k]
                tune[:, bi] = [best_lag_corr(rd.M[:, j], v, BEHLAG)[0] for j in range(rd.N)]
            tn = tune / (np.linalg.norm(tune, axis=1, keepdims=True) + 1e-9)
            cos = (tn[iu] * tn[ju]).sum(1)
            behsim_neg += list(cos[sig_neg]); behsim_pos += list(cos[sig_pos])
            fov_bsim[rd.fov]["neg"] += list(cos[sig_neg]); fov_bsim[rd.fov]["pos"] += list(cos[sig_pos])
        rows.append(dict(run=rd.key, fov=rd.fov, n_pairs=int(obs.size),
                         frac_sig_neg=float(np.mean(sig_neg)), frac_sig_pos=float(np.mean(sig_pos))))
    pd.DataFrame(rows).to_csv(outdir / "s5_negative_per_run.csv", index=False)
    vals = np.array([np.mean(v) for v in frac_neg_fov.values()])
    med, p, n = wilcoxon(vals, 0.025)
    lines.append(f"[S5_negative_fraction] significantly anti-correlated pairs (shift-null, 1-sided 2.5%): "
                 f"median {100*med:.1f}% of pairs over {n} FOVs vs 2.5% expected (Wilcoxon vs 0.025 p={p:.3g}). "
                 f"{'Excess of negative pairs beyond chance.' if (p==p and p<0.05 and med>0.025) else 'Not above chance.'}")
    # FOV-level paired test: within each FOV, median distance of neg vs pos pairs
    dn = np.array([np.median(v["neg"]) for v in fov_dist.values() if len(v["neg"]) >= 5 and len(v["pos"]) >= 5])
    dp = np.array([np.median(v["pos"]) for v in fov_dist.values() if len(v["neg"]) >= 5 and len(v["pos"]) >= 5])
    if dn.size >= 2:
        try:
            pw = sp.wilcoxon(dn - dp)[1]
        except Exception:
            pw = np.nan
        lines.append(f"[S5_negative_distance] within-FOV median 3D distance, negative vs positive sig pairs: "
                     f"{np.median(dn):.0f} vs {np.median(dp):.0f} um over {dn.size} FOVs (paired Wilcoxon p={pw:.3g}); "
                     f"pooled {np.median(dist_neg):.0f} vs {np.median(dist_pos):.0f} um, depth gap {np.median(depth_neg):.0f} vs {np.median(depth_pos):.0f} um (descriptive).")
    bn = np.array([np.median(v["neg"]) for v in fov_bsim.values() if len(v["neg"]) >= 5 and len(v["pos"]) >= 5])
    bp = np.array([np.median(v["pos"]) for v in fov_bsim.values() if len(v["neg"]) >= 5 and len(v["pos"]) >= 5])
    if bn.size >= 2:
        try:
            pw = sp.wilcoxon(bn - bp)[1]
        except Exception:
            pw = np.nan
        lines.append(f"[S5_negative_behtuning] within-FOV median behaviour-tuning cosine, negative vs positive sig "
                     f"pairs: {np.median(bn):+.2f} vs {np.median(bp):+.2f} over {bn.size} FOVs (paired Wilcoxon p={pw:.3g}). "
                     f"Negative pairs tend to have OPPOSITE behaviour tuning. NB this is partly expected because "
                     f"behaviour drives much of the Ca signal (anti-correlated pairs must have opposing arousal "
                     f"tuning); consistent with a minority of arousal-SUPPRESSED dendrites (the known anti-phase group) "
                     f"rather than direct mutual inhibition.")
    try:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].hist([dist_pos, dist_neg], bins=15, label=["positive", "negative"], color=["#4477aa", "#cc6677"])
        ax[0].set(xlabel="3D distance (um)", ylabel="n pairs", title="sig pairs by distance"); ax[0].legend()
        if behsim_neg and behsim_pos:
            ax[1].hist([behsim_pos, behsim_neg], bins=15, label=["positive", "negative"], color=["#4477aa", "#cc6677"])
            ax[1].set(xlabel="behaviour-tuning cosine", title="sig pairs by behaviour tuning"); ax[1].legend()
        fig.tight_layout(); fig.savefig(outdir / "fig_s5_negative.png", dpi=120); fig.savefig(outdir / "fig_s5_negative.pdf"); plt.close(fig)
    except Exception as e:
        lines.append(f"[S5 figure error] {e}")


# =============================================================================  S6 functional clusters
def analysis_clusters(rds, lines, outdir, source):
    rows = []
    zs = []
    z_fov = defaultdict(list)
    nsig = 0
    for rd in rds:
        C = np.c_[rd.glob]
        for k in BEHAVIORS:
            if k in rd.beh:
                C = np.c_[C, rd.beh[k]]
        R = partial_resid(rd.M, C)
        CR = corr_cols(R)
        np.fill_diagonal(CR, 0)
        D = 1 - CR
        iu = np.triu_indices(rd.N, 1)
        try:
            L = linkage(D[iu], "average")
        except Exception:
            continue
        # pick k by modularity
        best_q, best_lab = -1, None
        W = np.clip(CR, 0, None); m = W.sum() / 2
        kdeg = W.sum(1)
        for kk in range(2, min(10, rd.N - 1) + 1):
            lab = fcluster(L, kk, "maxclust")
            same = lab[:, None] == lab[None, :]
            q = ((W - np.outer(kdeg, kdeg) / (2 * m)) * same).sum() / (2 * m) if m > 0 else 0
            if q > best_q:
                best_q, best_lab = q, lab
        if best_lab is None:
            continue
        # spatial compactness: mean within-cluster 3D distance vs label permutation
        pts = np.c_[rd.cz, rd.cy, rd.cx]
        iuu, juu = np.triu_indices(rd.N, 1)
        dall = np.sqrt(((pts[iuu] - pts[juu]) ** 2).sum(1))
        same = best_lab[iuu] == best_lab[juu]
        if same.sum() < 3:
            continue
        obs_wd = dall[same].mean()
        rng = np.random.default_rng(hash(rd.key) % 2**32)
        null = np.empty(NPERM)
        for s in range(NPERM):
            lab = best_lab[rng.permutation(rd.N)]
            sm = lab[iuu] == lab[juu]
            null[s] = dall[sm].mean() if sm.sum() else np.nan
        z = (obs_wd - np.nanmean(null)) / (np.nanstd(null) + 1e-9)
        p = (np.sum(null <= obs_wd) + 1) / (NPERM + 1)
        zs.append(z)
        z_fov[rd.fov].append(z)
        if p < 0.05:
            nsig += 1
        rows.append(dict(run=rd.key, fov=rd.fov, n_clusters=int(best_lab.max()), modularity=float(best_q),
                         within_dist_um=float(obs_wd), null_dist_um=float(np.nanmean(null)), z=float(z), p=float(p)))
    pd.DataFrame(rows).to_csv(outdir / "s6_clusters_per_run.csv", index=False)
    if zs:
        zf = np.array([np.mean(v) for v in z_fov.values()])
        med, p, n = wilcoxon(zf, 0)
        lines.append(f"[S6_cluster_compactness] residual functional clusters (global Ca + behaviour regressed out): "
                     f"within-cluster 3D distance z vs label-permutation null, per-FOV median z={med:+.2f} over {n} "
                     f"FOVs (Wilcoxon p={p:.3g}); {nsig}/{len(zs)} runs compact at p<0.05 (negative z = more compact "
                     f"than chance). "
                     f"{'Functional clusters are spatially compact beyond chance.' if (p==p and p<0.05 and med<0) else 'Functional clusters are NOT clearly spatially compact.'}")
    else:
        lines.append("[S6_cluster_compactness] no clusterable runs (null).")


# =============================================================================  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "human"], default="auto")
    args = ap.parse_args()
    t0 = time.time()

    outdir = AUTO_ROOT / "stats" / "explore_structure" / args.source
    outdir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    rds = []
    for r in runs:
        try:
            rd = load_run(r, args.source)
        except Exception as e:
            print(f"  skip {r.key}: {e}")
            rd = None
        if rd is not None:
            rds.append(rd)
    fovs = sorted({rd.fov for rd in rds})
    beh_fovs = sorted({rd.fov for rd in rds if rd.has_beh})
    print(f"[{args.source}] loaded {len(rds)} runs, {len(fovs)} FOVs, {len(beh_fovs)} with behaviour "
          f"({time.time()-t0:.0f}s)")

    # shared pairwise corr+null (used by S5)
    cache = {}
    for i, rd in enumerate(rds):
        cache[rd.key] = pair_corr_and_null(rd, seed=1000 + i)
    print(f"  pairwise nulls done ({time.time()-t0:.0f}s)")

    lines = []
    lines.append(f"SPATIAL / ANATOMICAL STRUCTURE & BEHAVIOUR  -  {args.source} dendrites")
    tot_dend = sum(rd.N for rd in rds)
    tot_pairs = sum(rd.N * (rd.N - 1) // 2 for rd in rds)
    lines.append(f"{len(rds)} runs, {len(fovs)} FOVs ({len(beh_fovs)} with behaviour), "
                 f"{len({rd.mouse for rd in rds})} mice, {tot_dend} dendrite-runs, {tot_pairs} pairs.")
    lines.append(f"Nulls: independent circular shifts >= {MINSHIFT/FR:.0f} s. FOV = replication unit "
                 f"(runs averaged, then Wilcoxon signed-rank). BH-FDR within families. 5 Hz: lags < 0.2 s unresolved.")
    lines.append("")

    analysis_trunk(rds, lines, outdir, args.source);        print(f"  S1 done ({time.time()-t0:.0f}s)")
    analysis_depth(rds, lines, outdir, args.source);        print(f"  S2 done ({time.time()-t0:.0f}s)")
    analysis_behavior(rds, lines, outdir, args.source);     print(f"  S3 done ({time.time()-t0:.0f}s)")
    analysis_lateral(rds, lines, outdir, args.source);      print(f"  S4 done ({time.time()-t0:.0f}s)")
    analysis_negative(rds, lines, outdir, args.source, cache); print(f"  S5 done ({time.time()-t0:.0f}s)")
    analysis_clusters(rds, lines, outdir, args.source);     print(f"  S6 done ({time.time()-t0:.0f}s)")

    lines.append("")
    lines.append("Reading guide: independent units predict no excess coupling for vertical alignment (S1), no "
                 "depth/lateral gradients and no depth-band organisation of coupling (S2,S4), no behaviour-specific "
                 "dendrites beyond shared arousal (S3), negative pairs at chance 2.5% (S5), spatially random "
                 "functional clusters (S6).")
    lines.append("Caveats: 5 Hz (lags < 0.2 s unresolved), ~106 s / run, 2 mice with behaviour (3 total); SCAPE "
                 "optical crosstalk between neighbouring dendrites inflates short-range / same-column correlations "
                 "(core-shell background subtraction reduces but does not remove it) - so S1 column effects and the "
                 "shortest-distance part of S2/S5 should be read as upper bounds.")
    (outdir / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[done {args.source}] {time.time()-t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    main()
