#!/usr/bin/env python
"""compare_genotype.py - Ai162 (transgenic GCaMP6s) vs viral (AAV GCaMP7s) comparison.

Reads the per-run / per-dendrite tables already written by the automatic pipeline
(auto_stats.py, independence_tests.py, explore_encoding.py, explore_factors.py), plus the
per-run artefacts (validation/validation.json, masks/auto_masks.csv) and the dF/F traces
(traces/dff_auto.csv), attaches a genotype and a field-of-view (FOV) label to every run,
and asks whether the two genotype groups differ.

Design decisions (all driven by the small-n, partial-data reality of this cohort)
--------------------------------------------------------------------------------
* FOV is the unit of analysis. Runs that image the same FOV are averaged first, so a
  session imaged five times does not count as five independent samples. FOV comes from the
  `fov` column the pipeline already wrote (scape_common.fov_group); for runs not yet in any
  stats table (e.g. Ai162 runs still being processed) the FOV is resolved with fov_group()
  ONLY when every intra-session image-similarity pair is already cached in
  scape-auto/fov_check.json, so this script never opens a raw stack and never writes that
  cache (the pipeline runner owns it). If a pair is missing, the run is left as its own FOV
  and that is recorded.
* Genotype is get_run(key).genotype ('Ai162' | 'viral'). scape_common marks
  rbp4cre_139_phpeb as viral(uncertain); every comparison is run twice - with and without
  that mouse - and both are reported.
* FOV-level group comparison: Mann-Whitney U (two-sided) per metric, Benjamini-Hochberg FDR
  across the metrics of each family, Cliff's delta and Hedges g as effect sizes.
* Per-dendrite quantities additionally get a statsmodels MixedLM: value ~ genotype with a
  random intercept per FOV (dendrites nested in FOV).
* A mouse-level view (per-mouse medians) is reported too, with the plain warning that there
  are only ~2-3 mice per group.
* "Basic signal properties" that could differ purely because GCaMP6s (Ai162) and GCaMP7s
  (viral) are different indicators are computed directly from dff_auto.csv / auto_masks.csv /
  validation.json so they are available for the Ai162 runs *before* the heavy stats tables
  are regenerated: dendrite count, size (n_vox), length, y-span, peak SNR, event rate, event
  amplitude, event decay time (peak -> half-max, in seconds via r.frame_rate) and fraction
  of time active. Time-based quantities use r.frame_rate; see the frame-rate caveat in
  summary.txt.

Confound reporting: for every metric whose genotype difference survives FDR, the script
reports the Spearman correlation of that metric (across FOVs) with the obvious confounds -
frame rate, number of dendrites, cortical-depth span and z-extent (a proxy for the number of
Z planes: Ai162 42-69 planes, viral 30) - and flags the metric when |rho| >= 0.5.

Reads read-only from scape-auto/, scape-data/ and /Volumes/IMAC; writes ONLY under
scape-auto/stats/genotype/. Idempotent; well under 2 min; handles partial data and reports
null (not 0, not a guess) wherever a value could not be computed.

Usage
-----
    PROJ=/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025
    $PROJ/.venv311/bin/python $PROJ/code/Auto/compare_genotype.py            # source=both
    $PROJ/.venv311/bin/python $PROJ/code/Auto/compare_genotype.py --source auto
Options: --source {auto,human,both} (default both), --out DIR, --min-n N (minimum FOVs per
group to attempt a test; default 2).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sp

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from Auto.scape_common import (  # noqa: E402
    AUTO_ROOT, GENOTYPE, discover_runs, fov_group, genotype, get_run,
)

STATS = AUTO_ROOT / "stats"
EV_Z, EV_MIN = 3.0, 2          # same event definition as auto_stats.py
VOXEL_Z_UM = 3.9               # Z step; z_extent_um / 3.9 ~ number of Z planes

# Frame rates per the task / scape_common.AUTO_SESSION_PARAMS. All viral runs are 5.0 Hz;
# Ai162 sessions were measured from trigger spans and are approximate (+-10%).
AI162_RATE_RANGE = (4.71, 6.05)

# ---- per-run metric families: which per_run.csv to read and where it lives --------------
# builder(source) -> Path, where source in {"auto","human"}.
RUN_TABLES = {
    "auto_stats":  lambda s: STATS / s / "per_run.csv",
    "independence": lambda s: STATS / "independence" / s / "per_run.csv",
    "encoding":    lambda s: STATS / "explore_encoding" / s / "per_run.csv",
    "factors":     lambda s: STATS / "explore_factors" / s / "per_run.csv",
}
DEND_TABLES = {
    "auto_stats":  lambda s: STATS / s / "per_dendrite.csv",
    "independence": lambda s: STATS / "independence" / s / "per_dendrite.csv",
    "encoding":    lambda s: STATS / "explore_encoding" / s / "per_dendrite.csv",
    "factors":     lambda s: STATS / "explore_factors" / s / "per_dendrite.csv",
}

# Columns that are identifiers / bookkeeping, never compared as metrics.
ID_COLS = {"run", "source", "date", "mouse", "fov", "dendrite", "cluster", "id", "name",
           "origin_run", "origin_id"}
NONMETRIC = {"has_behavior", "has_beh", "glasso_ok", "kstar_reached", "n_frames", "n", "T",
             "fr", "frame_rate", "co_k", "n_accel_onsets", "glasso_n", "dendrite_like"}
import re  # noqa: E402
DROP_RE = re.compile(r"^(frac|null|excess|ratio)_frames_ge\d+$")   # session-specific count thresholds

# Per-dendrite metrics given to the mixed model (kept short so the fit stays fast).
DEND_METRICS = {
    "signal":      ["n_vox", "length_um", "yspan_um", "peak_snr", "event_rate_per_min",
                    "event_amp", "decay_time_s", "frac_time_active"],
    "auto_stats":  ["event_rate_per_min", "event_amp", "n_vox", "depth_um"],
    "independence": ["iei_cv", "fano_10s", "global_participation", "r_pupil", "r_whisker", "r_accel"],
    "encoding":    ["r2_pop", "r2_pop_far50", "r2_beh", "r2_glob", "glasso_degree", "corr_degree"],
    "factors":     ["private_var_frac", "load_f1"],
}

# Metrics highlighted in the figure (used if both groups have >=1 FOV with the value).
FIG_METRICS = [
    ("signal", "n_masks"), ("signal", "med_length_um"), ("signal", "med_peak_snr"),
    ("signal", "sig_event_rate_per_min"), ("signal", "sig_event_amp"),
    ("signal", "sig_decay_time_s"), ("signal", "sig_frac_time_active"),
    ("signal", "val_median_best_r"), ("signal", "val_reliability_auto_median"),
    ("auto_stats", "mean_pair_r"), ("auto_stats", "event_rate_per_min"),
    ("auto_stats", "frac_global_events"), ("auto_stats", "pr_over_n"),
    ("encoding", "r2_pop_median"), ("independence", "pc1_var_frac"),
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]
COLOR = {"Ai162": "#d1495b", "viral": "#2e78b6"}


# =============================================================================== utilities
def is_uncertain(mouse: str) -> bool:
    return "(uncertain)" in GENOTYPE.get(mouse, "")


def mouse_of(key: str) -> str:
    return key.split("/")[1] if "/" in key else key


def bh(pvals):
    """Benjamini-Hochberg q-values; NaN p's are ignored and returned as NaN."""
    p = np.asarray(pvals, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    pp = p[ok]
    o = np.argsort(pp)
    qq = np.empty(m)
    qq[o] = np.minimum.accumulate((pp[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    q[ok] = np.minimum(qq, 1.0)
    return q


def cliffs_delta(x, y):
    """Cliff's delta for x vs y: P(x>y) - P(x<y). Positive => x (Ai162) tends larger."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (x.size * y.size)


def hedges_g(x, y):
    """Hedges g for x (Ai162) minus y (viral)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if sp2 <= 0:
        return np.nan
    d = (x.mean() - y.mean()) / np.sqrt(sp2)
    J = 1 - 3 / (4 * (nx + ny) - 9)      # small-sample correction
    return d * J


def mannwhitney(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size < 1 or y.size < 1 or (x.size + y.size) < 3:
        return np.nan, np.nan
    if np.ptp(np.r_[x, y]) == 0:
        return np.nan, np.nan
    try:
        u, p = sp.mannwhitneyu(x, y, alternative="two-sided")
        return float(u), float(p)
    except ValueError:
        return np.nan, np.nan


def jnull(x):
    """JSON-safe scalar: NaN/inf -> None (reported as null)."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    return xf if np.isfinite(xf) else None


# =============================================================================== FOV lookup
def build_fov_lookup():
    """run_key -> fov label, taken from whatever stats table already has it (read-only)."""
    look = {}
    for build in RUN_TABLES.values():
        for s in ("auto", "human"):
            p = build(s)
            if p.exists():
                try:
                    df = pd.read_csv(p, usecols=["run", "fov"])
                except Exception:
                    continue
                for k, f in zip(df["run"], df["fov"]):
                    look.setdefault(str(k), str(f))
    return look


def fov_cache_pairs():
    p = AUTO_ROOT / "fov_check.json"
    if p.exists():
        try:
            return set(json.loads(p.read_text()).keys())
        except Exception:
            return set()
    return set()


def resolve_fov(key, table_fov, cache_pairs, incomplete):
    """FOV for a run, read-only. Prefer the stats-table value; else use fov_group() only when
    every intra-session similarity pair is cached (no stack open, no cache write)."""
    if key in table_fov:
        return table_fov[key]
    try:
        r = get_run(key)
    except Exception:
        return key
    sess = [x for x in discover_runs() if x.date == r.date and x.mouse == r.mouse]
    if len(sess) > 1:
        for i in range(len(sess)):
            for j in range(i + 1, len(sess)):
                pk = "|".join(sorted((sess[i].key, sess[j].key)))
                if pk not in cache_pairs:
                    incomplete.add(key)
                    return key              # fallback: its own FOV
    try:
        return fov_group(r)                 # pure cache read
    except Exception:
        incomplete.add(key)
        return key


# ========================================================= signal props straight from files
def robust_z(M):
    med = np.median(M, 0)
    mad = np.median(np.abs(M - med), 0) * 1.4826 + 1e-9
    return (M - med) / mad


def find_events(z):
    out = []
    for j in range(z.shape[1]):
        a = np.r_[False, z[:, j] > EV_Z, False]
        d = np.diff(a.astype(int))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        pk = [s + int(np.argmax(z[s:e, j])) for s, e in zip(st, en) if e - s >= EV_MIN]
        out.append(np.array(pk, int))
    return out


def decay_half_time(trace, peaks, fr):
    """Median time (s) from each event peak to the first crossing of half the peak dF/F.
    Sub-frame linear interpolation; right-censored events (never recross) are dropped."""
    T = trace.size
    times = []
    for p in peaks:
        pk = trace[p]
        if pk <= 0:
            continue
        half = pk * 0.5
        t = p
        while t + 1 < T and trace[t + 1] > half:
            t += 1
        if t + 1 >= T or trace[t + 1] > half:
            continue                                   # censored
        a, b = trace[t], trace[t + 1]
        frac = (a - half) / (a - b) if a != b else 0.0
        times.append((t - p + frac) / fr)
    return float(np.median(times)) if times else np.nan


def signal_props_from_files(fov_look, cache_pairs, incomplete):
    """Per-run and per-dendrite signal properties for every run with auto outputs on disk."""
    run_rows, dend_rows = [], []
    seen = set()
    # union of run keys that have a dff_auto.csv and are known to discover_runs
    for r in discover_runs(only_with_raw=False):
        dff = r.out / "traces" / "dff_auto.csv"
        if not dff.exists() or r.key in seen:
            continue
        seen.add(r.key)
        geno = r.genotype
        if geno not in ("Ai162", "viral"):
            continue
        fov = resolve_fov(r.key, fov_look, cache_pairs, incomplete)
        fr = float(r.frame_rate)
        try:
            df = pd.read_csv(dff)
        except Exception:
            continue
        names = [c for c in df.columns if c.startswith("dend_")]
        if not names:
            continue
        M = df[names].to_numpy(float)
        Tn = M.shape[0]
        dur_min = Tn / fr / 60.0
        Z = robust_z(M)
        evs = find_events(Z)
        frac_active = (Z > EV_Z).mean(0)
        # masks geometry
        mk = r.out / "masks" / "auto_masks.csv"
        geom = {}
        if mk.exists():
            try:
                mdf = pd.read_csv(mk)
                for _, row in mdf.iterrows():
                    geom[str(row.get("name"))] = row
            except Exception:
                pass
        d_rate, d_amp, d_decay, d_active = [], [], [], []
        for j, nm in enumerate(names):
            pk = evs[j]
            rate = len(pk) / dur_min if dur_min > 0 else np.nan
            amp = float(M[pk, j].mean()) if len(pk) else np.nan
            dec = decay_half_time(M[:, j], pk, fr)
            fa = float(frac_active[j])
            d_rate.append(rate); d_amp.append(amp); d_decay.append(dec); d_active.append(fa)
            g = geom.get(nm, {})
            dend_rows.append(dict(
                run=r.key, mouse=r.mouse, fov=fov, genotype=geno, uncertain=is_uncertain(r.mouse),
                frame_rate=fr, dendrite=nm,
                n_vox=jnull(g.get("n_vox")), length_um=jnull(g.get("length_um")),
                yspan_um=jnull(g.get("yspan_um")), peak_snr=jnull(g.get("peak_snr")),
                event_rate_per_min=jnull(rate), event_amp=jnull(amp),
                decay_time_s=jnull(dec), frac_time_active=jnull(fa)))
        # validation.json
        val = {}
        vp = r.out / "validation" / "validation.json"
        if vp.exists():
            try:
                vj = json.loads(vp.read_text())
                rel = vj.get("reliability", {}) or {}
                fn = vj.get("functional", {}) or {}
                val = dict(
                    val_n_auto=vj.get("n_auto"), val_n_human=vj.get("n_human"),
                    val_human_recall=fn.get("human_recall"), val_median_best_r=fn.get("median_best_r"),
                    val_reliability_auto_median=rel.get("auto_median"),
                    val_reliability_human_median=rel.get("human_median"))
            except Exception:
                pass
        mdf_n = len(geom) if geom else len(names)

        def med(col):
            if not geom:
                return np.nan
            vals = pd.to_numeric(pd.Series([g.get(col) for g in geom.values()]), errors="coerce")
            return float(vals.median()) if vals.notna().any() else np.nan
        row = dict(run=r.key, mouse=r.mouse, fov=fov, genotype=geno, uncertain=is_uncertain(r.mouse),
                   frame_rate=fr, n_masks=mdf_n, n_frames=Tn, duration_min=dur_min,
                   med_n_vox=med("n_vox"), med_length_um=med("length_um"),
                   med_yspan_um=med("yspan_um"), med_peak_snr=med("peak_snr"),
                   sig_event_rate_per_min=float(np.nanmean(d_rate)) if np.isfinite(d_rate).any() else np.nan,
                   sig_event_amp=float(np.nanmedian(d_amp)) if np.isfinite(d_amp).any() else np.nan,
                   sig_decay_time_s=float(np.nanmedian(d_decay)) if np.isfinite(d_decay).any() else np.nan,
                   sig_frac_time_active=float(np.nanmedian(d_active)) if np.isfinite(d_active).any() else np.nan)
        row.update(val)
        run_rows.append(row)
    return pd.DataFrame(run_rows), pd.DataFrame(dend_rows)


# =============================================================================== assembly
def numeric_metric_cols(df):
    cols = []
    for c in df.columns:
        if c in ID_COLS or c in NONMETRIC or DROP_RE.match(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def load_run_long(source):
    """Long per-run frame: run, mouse, fov, genotype, uncertain, frame_rate, family, metric, value."""
    frames = []
    for family, build in RUN_TABLES.items():
        p = build(source)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "run" not in df.columns or "fov" not in df.columns:
            continue
        if "mouse" not in df.columns:
            df["mouse"] = df["run"].map(mouse_of)
        df["genotype"] = df["mouse"].map(genotype)
        df["uncertain"] = df["mouse"].map(is_uncertain)
        fr = df["frame_rate"] if "frame_rate" in df.columns else (df["fr"] if "fr" in df.columns else np.nan)
        df["_frame_rate"] = fr
        mcols = numeric_metric_cols(df)
        if not mcols:
            continue
        long = df.melt(id_vars=["run", "mouse", "fov", "genotype", "uncertain", "_frame_rate"],
                       value_vars=mcols, var_name="metric", value_name="value")
        long["family"] = family
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["run", "mouse", "fov", "genotype", "uncertain",
                                     "_frame_rate", "metric", "value", "family"])
    return pd.concat(frames, ignore_index=True)


def signal_run_long(sig_run):
    if sig_run.empty:
        return pd.DataFrame(columns=["run", "mouse", "fov", "genotype", "uncertain",
                                     "_frame_rate", "metric", "value", "family"])
    mcols = [c for c in sig_run.columns if c not in
             {"run", "mouse", "fov", "genotype", "uncertain", "frame_rate", "n_frames"}]
    df = sig_run.rename(columns={"frame_rate": "_frame_rate"})
    long = df.melt(id_vars=["run", "mouse", "fov", "genotype", "uncertain", "_frame_rate"],
                   value_vars=mcols, var_name="metric", value_name="value")
    long["family"] = "signal"
    return long


def covariates_per_fov(sig_run, dend_auto):
    """Per-FOV confound covariates: frame_rate, n_dendrites, depth span, z-extent (~#planes)."""
    cov = {}
    if not sig_run.empty:
        g = sig_run.groupby("fov")
        for fov, sub in g:
            cov[fov] = dict(frame_rate=float(sub["frame_rate"].mean()),
                            n_dendrites=float(sub["n_masks"].mean()))
    # depth / z extents from the auto per-dendrite table (microns)
    if dend_auto is not None and not dend_auto.empty and "fov" in dend_auto.columns:
        for fov, sub in dend_auto.groupby("fov"):
            d = cov.setdefault(fov, {})
            if "depth_um" in sub:
                dd = pd.to_numeric(sub["depth_um"], errors="coerce").dropna()
                if len(dd):
                    d["depth_span_um"] = float(dd.max() - dd.min())
            if "z_um" in sub:
                zz = pd.to_numeric(sub["z_um"], errors="coerce").dropna()
                if len(zz):
                    d["z_extent_um"] = float(zz.max() - zz.min())
                    d["approx_n_planes"] = float((zz.max() - zz.min()) / VOXEL_Z_UM + 1)
    return cov


# =============================================================================== comparison
def fov_level_table(long, cohort):
    """Average runs within FOV first. Returns tidy FOV-level long frame for one cohort."""
    df = long.copy()
    if cohort == "without_uncertain":
        df = df[~df["uncertain"]]
    df = df.dropna(subset=["value"])
    if df.empty:
        return df.assign(fov_value=[])
    agg = (df.groupby(["family", "metric", "fov", "mouse", "genotype"], as_index=False)
             .agg(fov_value=("value", "mean")))
    return agg


def compare_fov(agg, cov, min_n):
    """Mann-Whitney + effect sizes per (family, metric) at FOV level; BH-FDR across metrics."""
    rows = []
    for (family, metric), sub in agg.groupby(["family", "metric"]):
        ai = sub.loc[sub.genotype == "Ai162", "fov_value"].to_numpy(float)
        vi = sub.loc[sub.genotype == "viral", "fov_value"].to_numpy(float)
        ai = ai[np.isfinite(ai)]; vi = vi[np.isfinite(vi)]
        n_ai, n_vi = ai.size, vi.size
        row = dict(family=family, metric=metric, level="fov", test="mannwhitney_fov",
                   n_ai162=n_ai, n_viral=n_vi,
                   ai162_median=jnull(np.median(ai)) if n_ai else None,
                   viral_median=jnull(np.median(vi)) if n_vi else None,
                   ai162_mean=jnull(np.mean(ai)) if n_ai else None,
                   viral_mean=jnull(np.mean(vi)) if n_vi else None,
                   cliffs_delta=None, hedges_g=None, U=None, p=None, q=None)
        if n_ai >= min_n and n_vi >= min_n:
            u, p = mannwhitney(ai, vi)
            row.update(U=jnull(u), p=jnull(p),
                       cliffs_delta=jnull(cliffs_delta(ai, vi)), hedges_g=jnull(hedges_g(ai, vi)))
            # confounds: correlate the FOV metric with each covariate across all FOVs present
            conf = {}
            vals = sub.set_index("fov")["fov_value"]
            for cvname in ("frame_rate", "n_dendrites", "depth_span_um", "z_extent_um"):
                xy = [(cov.get(f, {}).get(cvname), v) for f, v in vals.items()]
                xy = [(a, b) for a, b in xy if a is not None and np.isfinite(a) and np.isfinite(b)]
                if len(xy) >= 4:
                    a = np.array([p0 for p0, _ in xy]); b = np.array([p1 for _, p1 in xy])
                    if np.ptp(a) > 0 and np.ptp(b) > 0:
                        rho = sp.spearmanr(a, b)[0]
                        conf[cvname] = jnull(rho)
            row["confound_rho"] = conf
            row["low_power"] = bool(min(n_ai, n_vi) < 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and out["p"].notna().any():
        out["q"] = bh(out["p"].to_numpy(float))
    return out


def mixedlm_dendrite(dend_long, cohort, min_fov):
    """value ~ genotype with random intercept per FOV, per (family, metric)."""
    import statsmodels.formula.api as smf
    df = dend_long.copy()
    if cohort == "without_uncertain":
        df = df[~df["uncertain"]]
    rows = []
    for (family, metric), sub in df.groupby(["family", "metric"]):
        s = sub.dropna(subset=["value"]).copy()
        if s.empty:
            continue
        n_ai_fov = s.loc[s.genotype == "Ai162", "fov"].nunique()
        n_vi_fov = s.loc[s.genotype == "viral", "fov"].nunique()
        n_ai = int((s.genotype == "Ai162").sum()); n_vi = int((s.genotype == "viral").sum())
        row = dict(family=family, metric=metric, level="dendrite", test="mixedlm_dendrite",
                   n_ai162=n_ai, n_viral=n_vi, n_ai162_fov=n_ai_fov, n_viral_fov=n_vi_fov,
                   ai162_median=jnull(s.loc[s.genotype == "Ai162", "value"].median()) if n_ai else None,
                   viral_median=jnull(s.loc[s.genotype == "viral", "value"].median()) if n_vi else None,
                   beta=None, p=None, q=None)
        if n_ai_fov >= min_fov and n_vi_fov >= min_fov and np.ptp(s["value"]) > 0:
            s["g"] = (s.genotype == "Ai162").astype(float)   # beta = Ai162 - viral
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = smf.mixedlm("value ~ g", s, groups=s["fov"]).fit(reml=True, method="lbfgs")
                row.update(beta=jnull(m.params.get("g")), p=jnull(m.pvalues.get("g")))
            except Exception:
                pass
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and out["p"].notna().any():
        out["q"] = bh(out["p"].to_numpy(float))
    return out


def build_dend_long(source, sig_dend):
    """Long per-dendrite frame limited to DEND_METRICS, over every available dendrite table."""
    frames = []
    # file-computed signal per-dendrite (auto only)
    if source == "auto" and sig_dend is not None and not sig_dend.empty:
        cols = [c for c in DEND_METRICS["signal"] if c in sig_dend.columns]
        if cols:
            long = sig_dend.melt(id_vars=["run", "mouse", "fov", "genotype", "uncertain"],
                                 value_vars=cols, var_name="metric", value_name="value")
            long["family"] = "signal"
            frames.append(long)
    for family, build in DEND_TABLES.items():
        p = build(source)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "run" not in df.columns:
            continue
        if "mouse" not in df.columns:
            df["mouse"] = df["run"].map(mouse_of)
        if "fov" not in df.columns:
            continue
        df["genotype"] = df["mouse"].map(genotype)
        df["uncertain"] = df["mouse"].map(is_uncertain)
        cols = [c for c in DEND_METRICS.get(family, []) if c in df.columns]
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            continue
        long = df.melt(id_vars=["run", "mouse", "fov", "genotype", "uncertain"],
                       value_vars=cols, var_name="metric", value_name="value")
        long["family"] = family
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["run", "mouse", "fov", "genotype", "uncertain",
                                     "metric", "value", "family"])
    return pd.concat(frames, ignore_index=True)


# =============================================================================== outputs
def per_fov_wide(agg_all):
    """agg_all: concat of FOV-level long over mask sources, with a 'mask_source' col."""
    if agg_all.empty:
        return pd.DataFrame()
    agg_all = agg_all.copy()
    agg_all["col"] = agg_all["mask_source"] + ":" + agg_all["family"] + ":" + agg_all["metric"]
    wide = agg_all.pivot_table(index=["fov", "mouse", "genotype"], columns="col",
                               values="fov_value", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide


def per_mouse_table(agg_all):
    if agg_all.empty:
        return pd.DataFrame()
    rows = []
    for (ms, mouse, geno), sub in agg_all.groupby(["mask_source", "mouse", "genotype"]):
        rec = dict(mask_source=ms, mouse=mouse, genotype=geno,
                   uncertain=is_uncertain(mouse), n_fov=sub["fov"].nunique())
        for (fam, met), s2 in sub.groupby(["family", "metric"]):
            rec[f"{fam}:{met}"] = jnull(s2["fov_value"].median())
        rows.append(rec)
    return pd.DataFrame(rows)


def make_figure(agg_fov_auto, sig_run, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # metrics with data in both groups (with-uncertain cohort, auto mask set)
    avail = []
    for fam, met in FIG_METRICS:
        sub = agg_fov_auto[(agg_fov_auto.family == fam) & (agg_fov_auto.metric == met)]
        if sub.empty:
            continue
        if (sub.genotype == "Ai162").sum() >= 1 and (sub.genotype == "viral").sum() >= 1:
            avail.append((fam, met, sub))
    mice = sorted(agg_fov_auto["mouse"].unique())
    mk = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(mice)}

    if not avail:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No metric yet has both an Ai162 and a viral FOV.\n"
                "(Ai162 stats tables are regenerated after the pipeline runner finishes;\n"
                "signal properties from dff_auto.csv appear as soon as a run is processed.)",
                ha="center", va="center", wrap=True)
    else:
        ncol = 3
        nrow = int(np.ceil(len(avail) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), squeeze=False)
        rng = np.random.default_rng(0)
        for idx, (fam, met, sub) in enumerate(avail):
            ax = axes[idx // ncol][idx % ncol]
            for geno in ("viral", "Ai162"):
                g = sub[sub.genotype == geno]
                x0 = 0 if geno == "viral" else 1
                for _, row in g.iterrows():
                    jx = x0 + rng.uniform(-0.12, 0.12)
                    ax.scatter(jx, row["fov_value"], marker=mk.get(row["mouse"], "o"),
                               s=55, color=COLOR[geno],
                               edgecolors="k" if is_uncertain(row["mouse"]) else "none",
                               linewidths=1.1, alpha=0.9, zorder=3)
                if len(g):
                    ax.hlines(g["fov_value"].median(), x0 - 0.25, x0 + 0.25,
                              color=COLOR[geno], lw=2.5, zorder=2)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["viral", "Ai162"])
            ax.set_xlim(-0.5, 1.5)
            ax.set_title(f"{fam}:{met}", fontsize=9)
            ax.tick_params(labelsize=8)
        for j in range(len(avail), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        handles = [Line2D([0], [0], marker=mk[m], color="w", markerfacecolor="#888",
                          markeredgecolor="k", markersize=8, label=m) for m in mice]
        handles += [Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR["Ai162"], markersize=9, label="Ai162"),
                    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR["viral"], markersize=9, label="viral"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="#ccc", markeredgecolor="k", markersize=9,
                           label="black edge = uncertain mouse")]
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Ai162 (GCaMP6s, transgenic) vs viral (GCaMP7s) - one point per FOV, auto masks",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out / "fig_genotype.png", dpi=130, bbox_inches="tight")
    fig.savefig(out / "fig_genotype.pdf", bbox_inches="tight")
    plt.close(fig)
    return [f"{fam}:{met}" for fam, met, _ in avail]


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "human", "both"], default="both",
                    help="mask set to compare (default both)")
    ap.add_argument("--out", default=str(STATS / "genotype"))
    ap.add_argument("--min-n", type=int, default=2, help="min FOVs per group to run a test")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = ["auto", "human"] if args.source == "both" else [args.source]

    fov_look = build_fov_lookup()
    cache_pairs = fov_cache_pairs()
    incomplete = set()

    # file-based signal properties (auto masks only; never opens a raw stack)
    sig_run, sig_dend = signal_props_from_files(fov_look, cache_pairs, incomplete)

    comparison_rows = []
    agg_store = []          # FOV-level long across sources+cohorts for per_fov / per_mouse
    agg_fov_auto_withunc = pd.DataFrame()
    cov_auto = {}

    for source in sources:
        run_long = load_run_long(source)
        if source == "auto":
            run_long = pd.concat([run_long, signal_run_long(sig_run)], ignore_index=True)
        dend_long = build_dend_long(source, sig_dend)
        dend_auto_tbl = None
        p_auto_dend = DEND_TABLES["auto_stats"](source)
        if p_auto_dend.exists():
            try:
                dend_auto_tbl = pd.read_csv(p_auto_dend)
            except Exception:
                dend_auto_tbl = None
        cov = covariates_per_fov(sig_run if source == "auto" else pd.DataFrame(), dend_auto_tbl)
        if source == "auto":
            cov_auto = cov

        for cohort in ("with_uncertain", "without_uncertain"):
            agg = fov_level_table(run_long, cohort)
            if not agg.empty:
                agg2 = agg.copy()
                agg2["mask_source"] = source
                agg2["cohort"] = cohort
                agg_store.append(agg2)
                if source == "auto" and cohort == "with_uncertain":
                    agg_fov_auto_withunc = agg.copy()
            cmp_fov = compare_fov(agg, cov, args.min_n)
            cmp_dend = mixedlm_dendrite(dend_long, cohort, max(2, args.min_n))
            for cdf in (cmp_fov, cmp_dend):
                if cdf.empty:
                    continue
                cdf = cdf.copy()
                cdf["mask_source"] = source
                cdf["cohort"] = cohort
                comparison_rows.append(cdf)

    comparison = pd.concat(comparison_rows, ignore_index=True) if comparison_rows else pd.DataFrame()
    agg_all = pd.concat(agg_store, ignore_index=True) if agg_store else pd.DataFrame()
    # per_fov / per_mouse use the with_uncertain cohort (full cohort); keep both mask sources
    agg_full = agg_all[agg_all.cohort == "with_uncertain"] if not agg_all.empty else agg_all

    # ---- write tables
    if not comparison.empty:
        lead = ["mask_source", "cohort", "level", "family", "metric", "test",
                "n_ai162", "n_viral", "ai162_median", "viral_median",
                "cliffs_delta", "hedges_g", "beta", "U", "p", "q"]
        cols = [c for c in lead if c in comparison.columns] + \
               [c for c in comparison.columns if c not in lead]
        comparison = comparison[cols]
        comparison["confound_rho"] = comparison.get(
            "confound_rho", pd.Series([None] * len(comparison))).map(
            lambda d: json.dumps(d) if isinstance(d, dict) else "")
        comparison.to_csv(out / "comparison.csv", index=False)
    else:
        pd.DataFrame(columns=["mask_source", "cohort", "level", "family", "metric"]).to_csv(
            out / "comparison.csv", index=False)

    per_fov_wide(agg_full).to_csv(out / "per_fov.csv", index=False)
    per_mouse_table(agg_full).to_csv(out / "per_mouse.csv", index=False)

    fig_metrics = make_figure(agg_fov_auto_withunc, sig_run, out)

    # ---- counts for the plain-language header
    def group_counts(df, geno):
        sub = df[df.genotype == geno]
        return sub["fov"].nunique(), sub["mouse"].nunique(), sorted(sub["mouse"].unique())
    if not agg_full.empty:
        auto_full = agg_full[agg_full.mask_source == "auto"]
    else:
        auto_full = pd.DataFrame(columns=["fov", "mouse", "genotype"])

    summary = build_summary(comparison, agg_all, cov_auto, incomplete, fig_metrics,
                            sources, args.min_n)
    (out / "summary.txt").write_text(summary["text"])
    (out / "summary.json").write_text(json.dumps(summary["json"], indent=2, default=jnull))

    print(f"[compare_genotype] wrote {out}")
    print(f"  comparison rows: {len(comparison)}; FOV-level metrics tested: "
          f"{int((comparison.test == 'mannwhitney_fov').sum()) if not comparison.empty else 0}; "
          f"mixedlm metrics: {int((comparison.test == 'mixedlm_dendrite').sum()) if not comparison.empty else 0}")
    print(f"  figure panels (both groups present): {len(fig_metrics)}")
    if incomplete:
        print(f"  NOTE: FOV left as singleton (uncached similarity) for: {sorted(incomplete)}")


def build_summary(comparison, agg_all, cov, incomplete, fig_metrics, sources, min_n):
    L = []
    J = {"generated_by": "compare_genotype.py", "sources": sources, "min_n": min_n}
    L.append("=" * 86)
    L.append("Ai162 (Rbp4-Cre x Ai162, transgenic GCaMP6s) vs viral (Rbp4-Cre + AAV-PHP.eB GCaMP7s)")
    L.append("=" * 86)
    L.append("")
    L.append("Unit of analysis = FOV (runs averaged within an FOV first). Group test = Mann-Whitney U")
    L.append("at the FOV level; effect sizes Cliff's delta and Hedges g; BH-FDR across the metrics of")
    L.append("each family. Per-dendrite quantities also get a MixedLM value ~ genotype, random")
    L.append("intercept per FOV. Every comparison is reported with and without the uncertain mouse")
    L.append("rbp4cre_139_phpeb (scape_common lists it viral(uncertain)).")
    L.append("")

    # cohort / group census (auto, with_uncertain)
    if not agg_all.empty:
        base = agg_all[(agg_all.mask_source == "auto") & (agg_all.cohort == "with_uncertain")]
    else:
        base = pd.DataFrame(columns=["fov", "mouse", "genotype"])
    for geno in ("Ai162", "viral"):
        sub = base[base.genotype == geno] if not base.empty else base
        nf = sub["fov"].nunique() if not sub.empty else 0
        mice = sorted(sub["mouse"].unique()) if not sub.empty else []
        L.append(f"  {geno:6s}: {nf} FOV(s), {len(mice)} mouse(mice): {', '.join(mice) if mice else 'none yet'}")
        J[f"{geno}_fovs_auto"] = int(nf)
        J[f"{geno}_mice_auto"] = mice
    L.append("")
    L.append("SMALL-n WARNING: there are only a few mice per genotype (see counts above). FOV-level p-values are")
    L.append("under-powered and every 'significant' difference below is dominated by between-mouse")
    L.append("variance. Treat the mouse-level medians in per_mouse.csv as the honest resolution.")
    L.append("")
    if incomplete:
        L.append(f"FOV note: resolved as singletons (image-similarity pair not cached): {sorted(incomplete)}")
        L.append("")

    partial = base.empty or (base.genotype == "Ai162").sum() == 0 or (base.genotype == "viral").sum() == 0
    # Detect whether the heavy stats tables already contain Ai162 (vs only signal props)
    if not comparison.empty:
        tbl_fam = comparison[(comparison.mask_source == "auto") & (comparison.cohort == "with_uncertain")
                             & (comparison.family != "signal") & (comparison.level == "fov")]
        tbl_has_ai = (tbl_fam["n_ai162"].fillna(0) > 0).any() if not tbl_fam.empty else False
    else:
        tbl_has_ai = False
    J["stats_tables_have_ai162"] = bool(tbl_has_ai)
    if not tbl_has_ai:
        L.append("PARTIAL DATA: the aggregated stats tables (auto_stats / independence / encoding /")
        L.append("factors) do not yet contain the Ai162 runs - they are regenerated by the orchestrator")
        L.append("after the pipeline runner finishes (auto_stats.py --source both; independence_tests.py")
        L.append("--source both; explore_encoding.py). Until then only the 'signal' family (computed")
        L.append("directly from dff_auto.csv / auto_masks.csv / validation.json) has both genotypes;")
        L.append("those rows ARE a real comparison now. Re-run this script after the tables refresh for")
        L.append("the full set. Metrics with no Ai162 FOV are reported with n_ai162=0 and p=q=null.")
        L.append("")

    # ---------- per-metric block (auto, both cohorts) ----------
    def fmt(x, nd=3):
        return "null" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

    J["metrics"] = []
    if not comparison.empty:
        for cohort in ("with_uncertain", "without_uncertain"):
            L.append("-" * 86)
            L.append(f"COHORT: {cohort}  (mask set: auto)")
            L.append("-" * 86)
            sel = comparison[(comparison.mask_source == "auto") & (comparison.cohort == cohort)]
            sel = sel.sort_values(["level", "family", "metric"])
            for _, r in sel.iterrows():
                nai, nvi = int(r.get("n_ai162") or 0), int(r.get("n_viral") or 0)
                if r["test"] == "mannwhitney_fov":
                    es = f"Cliff d={fmt(r.get('cliffs_delta'))}, g={fmt(r.get('hedges_g'))}"
                    stat = f"p={fmt(r.get('p'))}, q={fmt(r.get('q'))}"
                else:
                    es = f"beta(Ai162-viral)={fmt(r.get('beta'))}"
                    stat = f"p={fmt(r.get('p'))}, q={fmt(r.get('q'))} (MixedLM)"
                q = r.get("q")
                sig = "**FDR-sig**" if (q is not None and np.isfinite(q) and q < 0.05) else (
                    "n.s." if (r.get("p") is not None and np.isfinite(r.get("p"))) else "untested")
                concl = plain_conclusion(r, sig)
                L.append(f"[{r['level']:8s}] {r['family']}:{r['metric']}")
                L.append(f"    Ai162 med={fmt(r.get('ai162_median'))} (n={nai})  "
                         f"viral med={fmt(r.get('viral_median'))} (n={nvi})  {es}  {stat}  -> {sig}")
                L.append(f"    {concl}")
                if cohort == "with_uncertain":
                    J["metrics"].append({
                        "family": r["family"], "metric": r["metric"], "level": r["level"],
                        "n_ai162": nai, "n_viral": nvi,
                        "ai162_median": jnull(r.get("ai162_median")),
                        "viral_median": jnull(r.get("viral_median")),
                        "cliffs_delta": jnull(r.get("cliffs_delta")), "hedges_g": jnull(r.get("hedges_g")),
                        "beta": jnull(r.get("beta")), "p": jnull(r.get("p")), "q": jnull(r.get("q")),
                        "fdr_significant": bool(q is not None and np.isfinite(q) and q < 0.05)})
            L.append("")

    # ---------- which survive FDR + confound statement ----------
    def parse_conf(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str) and v:
            try:
                return json.loads(v)
            except Exception:
                return {}
        return {}

    L.append("=" * 86)
    L.append("WHAT SURVIVES FDR, AND WHAT IS CONFOUNDED")
    L.append("=" * 86)
    J["fdr_survivors"] = {}
    any_survivor = False
    for cohort in ("with_uncertain", "without_uncertain"):
        survivors = []
        if not comparison.empty:
            s = comparison[(comparison.mask_source == "auto") & (comparison.cohort == cohort)]
            for _, r in s.iterrows():
                q = r.get("q")
                if q is not None and np.isfinite(q) and q < 0.05:
                    survivors.append(r)
        J["fdr_survivors"][cohort] = []
        if not survivors:
            L.append(f"[{cohort}] No metric survives BH-FDR (q<0.05) with the data present "
                     "(expected while the Ai162 side is still mostly the signal family and n is tiny).")
        else:
            any_survivor = True
            L.append(f"[{cohort}] metrics with q<0.05 (auto):")
            for r in survivors:
                conf = parse_conf(r.get("confound_rho"))
                flags = [f"{k} (rho={v:+.2f})" for k, v in conf.items()
                         if v is not None and np.isfinite(v) and abs(v) >= 0.5]
                am, vm = r.get("ai162_median"), r.get("viral_median")
                direction = "Ai162>viral" if (am is not None and vm is not None and am > vm) else "Ai162<viral"
                line = f"    [{r['level']}] {r['family']}:{r['metric']} ({direction}; q={r.get('q'):.3f})"
                if flags:
                    line += "  -- CONFOUNDED by: " + "; ".join(flags)
                else:
                    line += "  -- no strong (|rho|>=0.5) tie to frame rate / n_dendrites / depth / z-extent"
                L.append(line)
                J["fdr_survivors"][cohort].append(
                    {"family": r["family"], "metric": r["metric"], "level": r["level"],
                     "q": jnull(r.get("q")), "direction": direction, "confounds": flags})
        L.append("")
    if not any_survivor:
        L.append("(Nothing survives FDR in either cohort yet - re-run after the stats tables refresh.)")
        L.append("")
    L.append("Confound caveats that apply to the whole comparison, independent of the numbers above:")
    L.append(f"  * FRAME RATE: viral = 5.0 Hz exactly; Ai162 = {AI162_RATE_RANGE[0]}-{AI162_RATE_RANGE[1]} Hz,")
    L.append("    measured from trigger spans and only accurate to ~+-10%. Per-minute rates are already")
    L.append("    rate-normalised; decay_time_s is in seconds but its *resolution* is one frame, so a")
    L.append("    slower Ai162 run cannot resolve fast decays - a lower apparent decay is partly a")
    L.append("    sampling-rate artefact, not necessarily GCaMP6s vs GCaMP7s kinetics.")
    L.append("  * Z PLANES / DEPTH: Ai162 stacks have 42-69 Z planes (some dual-channel) vs 30 for viral,")
    L.append("    so z-extent and depth span differ by construction; any metric correlating with")
    L.append("    z_extent_um or depth_span_um above is depth/volume-confounded, not genotype-specific.")
    L.append("  * N DENDRITES: detector yield differs between a dim transgenic GCaMP6s and bright viral")
    L.append("    GCaMP7s; metrics that scale with n_dendrites (e.g. pr_over_n, pc1_var_frac,")
    L.append("    frac_pairs_sig) move with detector count, which is itself genotype-linked - so it is a")
    L.append("    mediator as much as a confound. Reported both as a metric and as a covariate.")
    L.append("")
    L.append(f"Figure panels (metrics with >=1 FOV in both groups): "
             f"{', '.join(fig_metrics) if fig_metrics else 'none yet'}")
    J["figure_panels"] = fig_metrics
    return {"text": "\n".join(L) + "\n", "json": J}


def plain_conclusion(r, sig):
    nai, nvi = int(r.get("n_ai162") or 0), int(r.get("n_viral") or 0)
    if nai == 0 or nvi == 0:
        miss = "Ai162" if nai == 0 else "viral"
        return f"No {miss} FOV present yet -> null (not comparable until the tables include it)."
    am, vm = r.get("ai162_median"), r.get("viral_median")
    if am is None or vm is None:
        return "Insufficient finite values -> null."
    rel = "higher" if am > vm else ("lower" if am < vm else "equal")
    if sig == "**FDR-sig**":
        return f"Ai162 is {rel} than viral and survives FDR, but see confound/power caveats below."
    if sig == "n.s.":
        return f"Ai162 is {rel} than viral but not significant after FDR (small n)."
    return "Reported as null (untestable with the FOVs available)."


if __name__ == "__main__":
    main()
