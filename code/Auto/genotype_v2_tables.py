#!/usr/bin/env python
"""
genotype_v2_tables.py
=====================
Small supporting tables for REPORT_independence.md v2. Reads existing outputs only.

Writes (all under scape-auto/stats/genotype/):
  within_genotype.csv     key independence metrics tested WITHIN each genotype separately
                          (FOV = unit; Wilcoxon signed-rank of obs vs its null, or vs 0 / 1),
                          plus the number of FOVs in which obs exceeds null.
  mouse_level_tests.csv   every FOV-level metric of comparison.csv (auto, with_uncertain)
                          re-tested at the MOUSE level: per-mouse median of FOV values, exact
                          Mann-Whitney 5 Ai162 vs 3 viral mice; also with rbp4cre_139_phpeb
                          merged into rbp4_139_phpeb (probably the same animal) -> 5 vs 2.
  validation_per_run.csv  validation.json numbers + detection thresholds per run (all 29 runs).
"""
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

PROJ = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025"
sys.path.insert(0, os.path.join(PROJ, "code"))
from Auto.scape_common import discover_runs, genotype, GENOTYPE  # noqa: E402

STATS = os.path.join(PROJ, "scape-auto", "stats")
GDIR = os.path.join(STATS, "genotype")
ALIAS = {"rbp4cre_139_phpeb": "rbp4_139_phpeb"}


def wsr(x, ref=None, alt="greater"):
    x = np.asarray(x, float)
    if ref is not None:
        x = x - np.asarray(ref, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.all(x == 0):
        return np.nan, len(x)
    return float(wilcoxon(x, alternative=alt).pvalue), len(x)


# ------------------------------------------------------------------ within-genotype
ind = pd.read_csv(os.path.join(STATS, "independence/auto/per_fov.csv"))
enc = pd.read_csv(os.path.join(STATS, "explore_encoding/auto/per_run.csv")).groupby("fov").mean(numeric_only=True).reset_index()
fac = pd.read_csv(os.path.join(STATS, "explore_factors/auto/per_fov.csv"))
dyn = pd.read_csv(os.path.join(STATS, "explore_dynamics/auto/per_fov.csv"))
mouse_of = ind.set_index("fov")["mouse"].to_dict()

# (label, table, obs col, null col or constant, alternative, source)
TESTS = [
    ("mean pairwise r > 0",                      ind, "mean_r", 0.0, "greater", "independence/auto/per_fov.csv"),
    ("silent frames > null",                     ind, "frac_frames_silent", "frac_frames_silent_null", "greater", "independence/auto/per_fov.csv"),
    ("co-activity Fano > null",                  ind, "coactivity_fano", "coactivity_fano_null", "greater", "independence/auto/per_fov.csv"),
    (">=10% co-active frames > null",            ind, "sync_10pct_obs", "sync_10pct_null", "greater", "independence/auto/per_fov.csv"),
    ("PC1 variance > null",                      ind, "pc1_var_frac", "pc1_var_frac_null", "greater", "independence/auto/per_fov.csv"),
    ("residual modularity > null",               ind, "resid_modularity", "resid_modularity_null", "greater", "independence/auto/per_fov.csv"),
    ("IEI CV < 1 (sub-Poisson)",                 ind, "iei_cv_median", 1.0, "less", "independence/auto/per_fov.csv"),
    ("population R2 > 0",                        enc, "r2_pop_median", 0.0, "greater", "explore_encoding/auto/per_run.csv (FOV mean)"),
    ("far-only R2 > 0",                          enc, "r2_pop_far_median", 0.0, "greater", "explore_encoding/auto/per_run.csv (FOV mean)"),
    ("unique population R2 > 0",                 enc, "unique_pop_median", 0.0, "greater", "explore_encoding/auto/per_run.csv (FOV mean)"),
    ("frac. dendrites independent (R2<0.02)",    enc, "frac_pop_lt_002", None, None, "explore_encoding/auto/per_run.csv (FOV mean)"),
    ("k=20 residual sig-pair frac > null rate",  fac, "kstar_frac_sig", "kstar_null_rate", "greater", "explore_factors/auto/per_fov.csv"),
    ("k* (shared factors to independence)",      fac, "kstar", None, None, "explore_factors/auto/per_fov.csv"),
    ("private variance fraction (k=20)",         fac, "private_var_median", None, None, "explore_factors/auto/per_fov.csv"),
    ("slow local-rate r > null",                 dyn, "slow_r_obs", "slow_r_null", "greater", "explore_dynamics/auto/per_fov.csv"),
]
rows = []
for g in ("Ai162", "viral"):
    for label, tab, oc, nc, alt, src in TESTS:
        t = tab.copy()
        t["geno"] = t["fov"].map(lambda f: genotype(mouse_of.get(f, f.split("/")[1])))
        t = t[t.geno == g]
        obs = t[oc].to_numpy(float)
        if isinstance(nc, str):
            null = t[nc].to_numpy(float)
            p, n = wsr(obs, null, alt)
            k = int(np.nansum((obs > null) if alt == "greater" else (obs < null)))
            nullmed = float(np.nanmedian(null))
        elif nc is None:
            p, n, k, nullmed = np.nan, int(np.isfinite(obs).sum()), np.nan, np.nan
        else:
            p, n = wsr(obs, np.full_like(obs, nc), alt)
            k = int(np.nansum((obs > nc) if alt == "greater" else (obs < nc)))
            nullmed = nc
        extra = ""
        if oc == "kstar":
            extra = (f"reached (<=20) in >=1 run of {int((t['kstar_reached'] > 0).sum())}/{len(t)} FOVs "
                     f"(kstar_reached is the run-average within a FOV)")
        rows.append(dict(genotype=g, test=label, obs_median=float(np.nanmedian(obs)), null_or_ref=nullmed,
                         n_fov=n, n_mice=t["fov"].map(mouse_of).nunique(), n_fov_in_direction=k,
                         p_wilcoxon_one_sided=p, p_min_attainable=(0.5 ** n if n else np.nan),
                         note=extra, source=src))
wg = pd.DataFrame(rows)
wg.to_csv(os.path.join(GDIR, "within_genotype.csv"), index=False)

# ------------------------------------------------------------------ mouse-level tests
pf = pd.read_csv(os.path.join(GDIR, "per_fov.csv"))
cmp_ = pd.read_csv(os.path.join(GDIR, "comparison.csv"))
cmp_ = cmp_[(cmp_.mask_source == "auto") & (cmp_.cohort == "with_uncertain") & (cmp_.level == "fov")]
mrows = []
for _, c in cmp_.iterrows():
    col = f"auto:{c.family}:{c.metric}"
    if col not in pf:
        continue
    d = pf[["mouse", "genotype", col]].dropna()
    out = dict(family=c.family, metric=c.metric, p_fov=c.p, q_fov=c.q,
               ai162_median_fov=c.ai162_median, viral_median_fov=c.viral_median, cliffs_delta_fov=c.cliffs_delta)
    for tag, mm in (("", d["mouse"]), ("_alias", d["mouse"].map(lambda m: ALIAS.get(m, m)))):
        dm = d.assign(m=mm).groupby(["m", "genotype"])[col].median().reset_index()
        a = dm.loc[dm.genotype == "Ai162", col].to_numpy()
        v = dm.loc[dm.genotype == "viral", col].to_numpy()
        p = float(mannwhitneyu(a, v, alternative="two-sided").pvalue) if len(a) >= 2 and len(v) >= 2 else np.nan
        out.update({f"n_mice_ai162{tag}": len(a), f"n_mice_viral{tag}": len(v),
                    f"ai162_median_mouse{tag}": float(np.median(a)) if len(a) else np.nan,
                    f"viral_median_mouse{tag}": float(np.median(v)) if len(v) else np.nan,
                    f"p_mouse{tag}": p,
                    f"p_min_attainable{tag}": 2 / comb(len(a) + len(v), len(v)) if len(a) and len(v) else np.nan,
                    f"all_ai162_below_or_above{tag}": bool(len(a) and len(v) and (a.max() < v.min() or a.min() > v.max()))})
    mrows.append(out)
ml = pd.DataFrame(mrows)
ml.to_csv(os.path.join(GDIR, "mouse_level_tests.csv"), index=False)

# ------------------------------------------------------------------ validation / thresholds
vrows = []
for r in discover_runs():
    o = r.out
    m = json.load(open(o / "metrics.json")) if (o / "metrics.json").exists() else {}
    vj = o / "validation" / "validation.json"
    v = json.load(open(vj)) if vj.exists() else {}
    fn = v.get("functional", {})
    am = pd.read_csv(o / "masks" / "auto_masks.csv")
    thr = am.groupby(["fg_z", "seed_z", "grow_r"]).size()
    lowered = am[(am.fg_z < 4.5) | (am.seed_z < 7.0)]
    try:
        import tifffile
        with tifffile.TiffFile(o / "reference" / "ref_mean.tif") as t:
            nz = t.series[0].shape[0]
    except Exception:
        nz = np.nan
    vrows.append(dict(run=r.key, mouse=r.mouse, genotype=genotype(r.mouse),
                      genotype_label=GENOTYPE.get(r.mouse, "?"), n_planes=nz, frame_rate_hz=r.frame_rate,
                      raw_root=("/Volumes/IMAC/data" if str(r.raw).startswith("/Volumes/IMAC") else "scape-data"),
                      has_behavior=r.behavior_mat is not None,
                      n_auto=m.get("detect", {}).get("n_auto"), n_human=v.get("n_human"),
                      human_recall=fn.get("human_recall"), recall_r05=fn.get("recall_r05"),
                      median_best_r=fn.get("median_best_r"),
                      voxel_coverage_of_human=v.get("voxel_coverage_of_human"),
                      reliability_auto=v.get("reliability", {}).get("auto_median"),
                      reliability_null=v.get("reliability", {}).get("null_shifted_median"),
                      reliability_human=v.get("reliability", {}).get("human_median"),
                      thresholds="; ".join(f"FG{a:.2f}/SEED{b:.2f}/GROW{c:.2f}:{n}" for (a, b, c), n in thr.items()),
                      frac_units_lowered_threshold=len(lowered) / len(am) if len(am) else np.nan))
vt = pd.DataFrame(vrows)
vt.to_csv(os.path.join(GDIR, "validation_per_run.csv"), index=False)

with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70):
    print(wg.round(4).to_string(index=False))
    print()
    keep = ["family", "metric", "p_fov", "q_fov", "ai162_median_mouse", "viral_median_mouse", "p_mouse",
            "all_ai162_below_or_above", "p_mouse_alias"]
    print(ml[ml.p_mouse < 0.1][keep].round(4).to_string(index=False))
    print("mouse-level p<0.05:", int((ml.p_mouse < 0.05).sum()), "of", len(ml),
          "; alias p<0.05:", int((ml.p_mouse_alias < 0.05).sum()))
    print()
    print(vt.round(3).to_string(index=False))
