#!/usr/bin/env python
"""
make_genotype_figure.py
=======================
Ai162 (Rbp4-Cre x Ai162, transgenic GCaMP6s + PHP.eB red ACh sensor) vs viral
(Rbp4-Cre + AAV-PHP.eB GCaMP7s): one point per FOV, one marker shape per mouse,
for the most informative signal / independence metrics. Built only from tables
that already exist (nothing heavy is re-run):

  scape-auto/stats/genotype/per_fov.csv      FOV values ('auto:<family>:<metric>')
  scape-auto/stats/genotype/comparison.csv   FOV-level Mann-Whitney p and BH q
                                             (mask_source=auto, cohort=with_uncertain)
  scape-auto/stats/coherence/auto/per_fov.csv  global Ca vs behaviour coherence (slow band);
                                             NOT in comparison.csv, so its Mann-Whitney p is
                                             computed here and labelled 'p(here)', no FDR.

Each panel title gives the FOV-level Mann-Whitney p and q from comparison.csv and,
on a second line, a MOUSE-level Mann-Whitney p computed here on the per-mouse
medians of the FOV values (5 Ai162 vs 3 viral mice; the smallest attainable
two-sided exact p is 2/C(8,3) = 0.036). FOVs are nested in mice, so the FOV-level
p overstates certainty; the mouse-level p is the honest resolution.

The uncertain mouse rbp4cre_139_phpeb (scape_common GENOTYPE 'viral(uncertain)',
probably the same animal as rbp4_139_phpeb) is drawn with open markers.

Writes:
  scape-auto/stats/FIGURE_genotype.png / .pdf
  scape-auto/stats/genotype/figure_panels.csv   (the numbers shown in each panel)
"""
import os
import sys
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

PROJ = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025"
sys.path.insert(0, os.path.join(PROJ, "code"))
from Auto.scape_common import GENOTYPE  # noqa: E402

STATS = os.path.join(PROJ, "scape-auto", "stats")
GDIR = os.path.join(STATS, "genotype")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
_have_arial = any("Arial" in f.name for f in fm.fontManager.ttflist)
plt.rcParams["font.family"] = "Arial" if _have_arial else "DejaVu Sans"
plt.rcParams["font.size"] = 8.5
plt.rcParams["axes.linewidth"] = 0.8

C_GENO = {"viral": "#1f6feb", "Ai162": "#2e9e44"}
GROUP_X = {"viral": 0.0, "Ai162": 1.0}

# (per_fov column / comparison key, panel label, y-axis label, source)
# source 'cmp'  -> p/q from comparison.csv ; 'coh' -> coherence per_fov, p computed here
PANELS = [
    ("signal:n_masks",               "dendrites detected per FOV",          "n dendrites",              "cmp"),
    ("signal:med_length_um",         "dendrite size (median length)",       "length (um)",              "cmp"),
    ("signal:med_peak_snr",          "dendrite peak SNR (median)",          "peak robust z",            "cmp"),
    ("signal:sig_event_rate_per_min", "event rate (median)",                "events / min",             "cmp"),
    ("signal:sig_event_amp",         "event amplitude (median)",            "dF/F (%)",                 "cmp"),
    ("signal:sig_decay_time_s",      "event decay time (median)",           "decay (s)",                "cmp"),
    ("auto_stats:mean_pair_r",       "mean pairwise r",                     "r",                        "cmp"),
    ("auto_stats:frac_pairs_sig",    "fraction of pairs beyond shift null", "fraction",                 "cmp"),
    ("auto_stats:frac_global_events", "global-event fraction",              "fraction of events",       "cmp"),
    ("encoding:r2_pop_median",       "population-coupling R$^2$",           "held-out R$^2$ (median)",  "cmp"),
    ("encoding:r2_pop_far_median",   "far-only (>50 um) R$^2$",             "held-out R$^2$ (median)",  "cmp"),
    ("independence:pc1_var_frac",    "PC1 variance fraction",               "fraction",                 "cmp"),
    ("independence:frac_frames_silent", "silent-frame fraction",            "fraction of frames",       "cmp"),
    ("global_coh_accel_slow_excess", "global Ca-locomotion coherence",      "slow-band excess coh.",    "coh"),
    ("global_coh_whisker_slow_excess", "global Ca-whisker coherence",       "slow-band excess coh.",    "coh"),
    ("global_coh_pupil_slow_excess", "global Ca-pupil coherence",           "slow-band excess coh.",    "coh"),
]


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


# ----------------------------------------------------------------------------- load
pf = pd.read_csv(os.path.join(GDIR, "per_fov.csv"))
cmp_ = pd.read_csv(os.path.join(GDIR, "comparison.csv"))
cmp_ = cmp_[(cmp_.mask_source == "auto") & (cmp_.cohort == "with_uncertain") & (cmp_.level == "fov")]
coh = pd.read_csv(os.path.join(STATS, "coherence", "auto", "per_fov.csv"))
pf = pf.merge(coh[["fov"] + [p[0] for p in PANELS if p[3] == "coh"]], on="fov", how="left")

mice = sorted(pf["mouse"].unique(), key=lambda m: (pf.loc[pf.mouse == m, "genotype"].iloc[0] != "viral", m))
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
mk = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(mice)}
n_mouse = {g: pf.loc[pf.genotype == g, "mouse"].nunique() for g in ("Ai162", "viral")}
n_fov = {g: int((pf.genotype == g).sum()) for g in ("Ai162", "viral")}
p_min_mouse = 2 / comb(n_mouse["Ai162"] + n_mouse["viral"], n_mouse["viral"])

# ----------------------------------------------------------------------------- figure
ncol = 4
nrow = int(np.ceil(len(PANELS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.35 * nrow))
axes = axes.ravel()
rng = np.random.default_rng(0)
rows = []
for ax, (key, title, ylab, src) in zip(axes, PANELS):
    col = f"auto:{key}" if src == "cmp" else key
    d = pf[["fov", "mouse", "genotype", col]].rename(columns={col: "v"}).dropna(subset=["v"])
    # FOV-level p / q
    if src == "cmp":
        fam, met = key.split(":")
        c = cmp_[(cmp_.family == fam) & (cmp_.metric == met)]
        p_fov = float(c["p"].iloc[0]) if len(c) else np.nan
        q_fov = float(c["q"].iloc[0]) if len(c) else np.nan
        ptxt = f"FOV MW p={fmt_p(p_fov)}, q={fmt_p(q_fov)}"
    else:
        p_fov = mw(d.loc[d.genotype == "Ai162", "v"], d.loc[d.genotype == "viral", "v"])
        q_fov = np.nan
        ptxt = f"FOV MW p(here)={fmt_p(p_fov)}, no FDR"
    # mouse-level: median of FOV values per mouse, then MW across mice
    dm = d.groupby(["mouse", "genotype"])["v"].median().reset_index()
    p_mouse = mw(dm.loc[dm.genotype == "Ai162", "v"], dm.loc[dm.genotype == "viral", "v"])
    nm = dm.groupby("genotype").size().to_dict()
    nf = d.groupby("genotype").size().to_dict()
    for g in ("viral", "Ai162"):
        dg = d[d.genotype == g]
        if dg.empty:
            continue
        for _, r in dg.iterrows():
            x = GROUP_X[g] + rng.uniform(-0.17, 0.17)
            unc = GENOTYPE.get(r["mouse"], "").endswith("(uncertain)")
            ax.scatter(x, r["v"], marker=mk[r["mouse"]], s=42, zorder=3,
                       facecolor="white" if unc else C_GENO[g], edgecolor=C_GENO[g], linewidth=1.1, alpha=0.9)
        med = dg["v"].median()
        ax.plot([GROUP_X[g] - 0.3, GROUP_X[g] + 0.3], [med, med], color="k", lw=1.8, zorder=4)
        # mouse medians as short grey ticks to the right of the group
        for _, r in dm[dm.genotype == g].iterrows():
            ax.plot([GROUP_X[g] + 0.33, GROUP_X[g] + 0.43], [r["v"], r["v"]], color="#777777", lw=1.4, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"viral\n{nf.get('viral', 0)} FOV / {nm.get('viral', 0)} mice",
                        f"Ai162\n{nf.get('Ai162', 0)} FOV / {nm.get('Ai162', 0)} mice"], fontsize=7.5)
    ax.set_xlim(-0.55, 1.6)
    ax.set_ylabel(ylab)
    ax.set_title(f"{title}\n{ptxt}\nmouse-level MW p={fmt_p(p_mouse)}", fontsize=8.5, loc="left")
    rows.append(dict(metric=key, source=("comparison.csv" if src == "cmp" else "coherence/auto/per_fov.csv"),
                     ai162_median=d.loc[d.genotype == "Ai162", "v"].median(),
                     viral_median=d.loc[d.genotype == "viral", "v"].median(),
                     n_fov_ai162=nf.get("Ai162", 0), n_fov_viral=nf.get("viral", 0),
                     n_mouse_ai162=nm.get("Ai162", 0), n_mouse_viral=nm.get("viral", 0),
                     p_fov=p_fov, q_fov=q_fov, p_mouse=p_mouse))
for ax in axes[len(PANELS):]:
    ax.axis("off")

handles = [Line2D([], [], ls="", marker=mk[m], markersize=7,
                  markerfacecolor=("white" if GENOTYPE.get(m, "").endswith("(uncertain)")
                                   else C_GENO[pf.loc[pf.mouse == m, "genotype"].iloc[0]]),
                  markeredgecolor=C_GENO[pf.loc[pf.mouse == m, "genotype"].iloc[0]],
                  label=m + (" (genotype uncertain)" if GENOTYPE.get(m, "").endswith("(uncertain)") else ""))
           for m in mice]
handles += [Line2D([], [], color="k", lw=1.8, label="group median (FOVs)"),
            Line2D([], [], color="#777777", lw=1.4, label="mouse median")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Ai162 (transgenic GCaMP6s, {n_mouse['Ai162']} mice / {n_fov['Ai162']} FOVs) vs viral "
             f"(PHP.eB GCaMP7s, {n_mouse['viral']} mice / {n_fov['viral']} FOVs): one point per FOV, auto dendrites.  "
             f"No panel metric survives FDR (only run duration does); mouse-level p cannot go below {p_min_mouse:.3f} with "
             f"{n_mouse['Ai162']} vs {n_mouse['viral']} mice.",
             fontsize=10.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=1.6, w_pad=1.2)

out_png = os.path.join(STATS, "FIGURE_genotype.png")
out_pdf = os.path.join(STATS, "FIGURE_genotype.pdf")
fig.savefig(out_png, dpi=200)
fig.savefig(out_pdf)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(GDIR, "figure_panels.csv"), index=False)
print("wrote", out_png)
print("wrote", out_pdf)
print("wrote", os.path.join(GDIR, "figure_panels.csv"))
with pd.option_context("display.width", 220):
    print(tab.round(4).to_string(index=False))
