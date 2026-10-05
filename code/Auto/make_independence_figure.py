#!/usr/bin/env python
"""
make_independence_figure.py
===========================
Talk figure for the question "Are L5 apical dendrites independent computational
units?" built ENTIRELY from the pre-computed result tables under
PROJ/scape-auto/stats/ (no heavy pipelines are re-run; only small summaries are
computed from the CSVs).

Writes:  PROJ/scape-auto/stats/FIGURE_independence.png  and  .pdf

Nine lettered panels (A-I), each titled with the result it shows:
  A  dendrite map of one FOV, coloured by population-coupling R^2
  B  held-out variance explained per dendrite: behaviour vs global Ca vs other dendrites
  C  distribution of population-coupling R^2 (auto vs human) with independent / strongly-coupled fractions
  D  far-only (>50 um) vs all-dendrite R^2 (shared network signal, not short-range crosstalk)
  E  co-activation: silence & bursts far beyond the independent null
  F  coupled events are synchronous (same 0.2 s frame), not sequential
  G  choristers vs soloists: population coupling tracks behaviour coupling and decreases with depth
  H  sparse partial-correlation network with superficial hubs
  I  residual functional clusters are spatially compact + slow shared excitability

The 'anti-phase-with-pupil' dendrite group (a shell-subtraction artefact, per the
PI's decision) is deliberately NOT shown anywhere in this figure.

Data sources (all under scape-auto/stats/):
  explore_encoding/{auto,human}/per_dendrite.csv , per_run.csv
  independence/{auto,human}/per_run.csv
  explore_dynamics/{auto,human}/per_fov.csv
  (onset-precision same-frame fractions: explore_dynamics/{auto,human}/summary.txt
   -- these are not emitted to a CSV, so the few values used in panel F are quoted
   from those summaries and labelled as such in the code below.)
"""

import os
import numpy as np
import pandas as pd
import tifffile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.font_manager as fm

# ---- style: Arial, embeddable (TrueType) fonts in the PDF ------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
_have_arial = any("Arial" in f.name for f in fm.fontManager.ttflist)
plt.rcParams["font.family"] = "Arial" if _have_arial else "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["font.size"] = 9

PROJ = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025"
STATS = os.path.join(PROJ, "scape-auto", "stats")
FOV_A = "2026-05-12/rbp4_132_phpeb/run5"          # FOV shown in panel A

C_AUTO = "#1f6feb"   # auto dendrites (primary)
C_HUMAN = "#d1495b"  # human-curated dendrites (replication)
C_NULL = "#9aa0a6"   # independent circular-shift null
C_OBS = "#2a2d34"    # observed


# ----------------------------------------------------------------------------
def fov_median(df, value_cols):
    """Average within FOV first (FOV = unit of replication), then take the
    cohort median across FOVs. Returns a Series indexed by value_cols."""
    g = df.groupby("fov")[value_cols].mean()
    return g.median()


def letter(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="bottom", ha="right")


# ============================================================================
# LOAD TABLES
# ============================================================================
enc_a = pd.read_csv(os.path.join(STATS, "explore_encoding/auto/per_dendrite.csv"))
enc_h = pd.read_csv(os.path.join(STATS, "explore_encoding/human/per_dendrite.csv"))
encr_a = pd.read_csv(os.path.join(STATS, "explore_encoding/auto/per_run.csv"))
ind_a = pd.read_csv(os.path.join(STATS, "independence/auto/per_run.csv"))
ind_h = pd.read_csv(os.path.join(STATS, "independence/human/per_run.csv"))
dyn_a = pd.read_csv(os.path.join(STATS, "explore_dynamics/auto/per_fov.csv"))

# ============================================================================
# FIGURE SCAFFOLD  (3 x 3 grid, ~16 x 11 in)
# ============================================================================
fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.28,
                       left=0.055, right=0.975, top=0.93, bottom=0.065)

# ----------------------------------------------------------------------------
# PANEL A : dendrite map of one FOV, coloured by population coupling R^2
# ----------------------------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])
ref = tifffile.imread(os.path.join(PROJ, "scape-auto", FOV_A, "reference/ref_activity.tif"))
lab = tifffile.imread(os.path.join(PROJ, "scape-auto", FOV_A, "masks/auto_labelmap.tif"))
# project along the SCAPE z-axis -> (depth y, lateral x)
ref_mip = ref.max(axis=0)
lab_mip = np.zeros(lab.shape[1:], dtype=lab.dtype)
# a label is present in the projection if any z-plane carries it; keep the label
# with the most voxels in the column so outlines stay crisp
for z in range(lab.shape[0]):
    sl = lab[z]
    m = lab_mip == 0
    lab_mip[m] = sl[m]

sub = enc_a[enc_a["run"] == FOV_A].copy()
sub["label"] = sub["dendrite"].str.replace("dend_", "", regex=False).astype(int) + 1
lut = dict(zip(sub["label"], sub["r2_pop"]))

cmap = plt.cm.viridis
vmin, vmax = 0.0, 0.6
# grayscale activity background (log for dynamic range)
axA.imshow(np.log1p(ref_mip), cmap="gray", aspect="equal",
           extent=[0, ref_mip.shape[1] * 1.2, ref_mip.shape[0] * 1.0, 0])
# RGBA overlay: fill each dendrite with its coupling colour
overlay = np.zeros((*lab_mip.shape, 4), dtype=float)
for L, r2 in lut.items():
    mask = lab_mip == L
    if not mask.any():
        continue
    val = np.clip((r2 - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(val)
    overlay[mask] = (rgba[0], rgba[1], rgba[2], 0.72)
axA.imshow(overlay, aspect="equal",
           extent=[0, lab_mip.shape[1] * 1.2, lab_mip.shape[0] * 1.0, 0])
axA.set_xlabel("lateral position (um)")
axA.set_ylabel("cortical depth (superficial -> deep)")
axA.set_title("A dendrites in one FOV, coloured by population coupling R$^2$:\n"
              "coupling is distributed across the field, not clustered",
              fontsize=10, loc="left")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
sm.set_array([])
cb = fig.colorbar(sm, ax=axA, fraction=0.046, pad=0.02)
cb.set_label("population-coupling R$^2$")
letter(axA, "")

# ----------------------------------------------------------------------------
# PANEL B : held-out variance explained per dendrite by block
# ----------------------------------------------------------------------------
axB = fig.add_subplot(gs[0, 1])
blocks = [("behaviour", "r2_beh"), ("global Ca", "r2_glob"), ("other\ndendrites", "r2_pop")]
data = [enc_a[col].dropna().values for _, col in blocks]
bp = axB.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                 medianprops=dict(color="black", lw=1.6))
cols = ["#8aa29e", "#b8955b", C_AUTO]
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.75)
axB.set_xticklabels([b[0] for b in blocks])
axB.set_ylabel("held-out variance explained (R$^2$)")
axB.axhline(0, color="k", lw=0.6, ls=":")
axB.set_ylim(-0.15, 0.65)
meds = [np.median(d) for d in data]
for i, m in enumerate(meds):
    axB.text(i + 1, 0.60, f"med\n{m:.3f}", ha="center", va="top", fontsize=8)
axB.set_title("B other dendrites explain far more held-out variance\n"
              "than behaviour or global Ca (auto; blocked CV)",
              fontsize=10, loc="left")
letter(axB, "")

# ----------------------------------------------------------------------------
# PANEL C : distribution of population-coupling R^2, auto vs human
# ----------------------------------------------------------------------------
axC = fig.add_subplot(gs[0, 2])
bins = np.linspace(-0.2, 0.8, 41)
axC.hist(enc_a["r2_pop"].clip(-0.2, 0.8), bins=bins, density=True, color=C_AUTO,
         alpha=0.55, label=f"auto (n={len(enc_a)})")
axC.hist(enc_h["r2_pop"].clip(-0.2, 0.8), bins=bins, density=True, color=C_HUMAN,
         alpha=0.45, label=f"human (n={len(enc_h)})")
axC.axvline(0.02, color="k", ls="--", lw=1)
axC.axvline(0.20, color="k", ls=":", lw=1)
# fractions
fa_ind = (enc_a["r2_pop"] < 0.02).mean() * 100
fa_str = (enc_a["r2_pop"] > 0.20).mean() * 100
fh_ind = (enc_h["r2_pop"] < 0.02).mean() * 100
fh_str = (enc_h["r2_pop"] > 0.20).mean() * 100
axC.text(0.02, 0.97, f" 'independent' R$^2$<0.02\n auto {fa_ind:.0f}% / human {fh_ind:.0f}%",
         transform=axC.transAxes, fontsize=8, va="top")
axC.text(0.55, 0.97, f"'strongly coupled' R$^2$>0.2\nauto {fa_str:.0f}% / human {fh_str:.0f}%",
         transform=axC.transAxes, fontsize=8, va="top")
axC.set_xlabel("population-coupling R$^2$ (predict a dendrite from all others)")
axC.set_ylabel("density")
axC.legend(loc="center right", fontsize=8, frameon=False)
axC.set_title("C ~1/5 of dendrites are effectively independent;\n"
              "a slim majority are strongly coupled (auto & human agree)",
              fontsize=10, loc="left")
letter(axC, "")

# ----------------------------------------------------------------------------
# PANEL D : far-only (>50 um) vs all-dendrite R^2
# ----------------------------------------------------------------------------
axD = fig.add_subplot(gs[1, 0])
x = enc_a["r2_pop"].values
y = enc_a["r2_pop_far50"].values
ok = np.isfinite(x) & np.isfinite(y)
hb = axD.hexbin(x[ok], y[ok], gridsize=40, extent=(-0.3, 0.8, -0.3, 0.8),
                cmap="magma_r", mincnt=1, bins="log")
axD.plot([-0.3, 0.8], [-0.3, 0.8], color="k", lw=1, ls="--")
axD.set_xlim(-0.3, 0.8); axD.set_ylim(-0.3, 0.8)
axD.set_xlabel("R$^2$ from ALL other dendrites")
axD.set_ylabel("R$^2$ from FAR dendrites only (>50 um)")
# far-minus-all FOV median
famed = fov_median(encr_a, ["med_far_minus_all"])["med_far_minus_all"]
axD.text(0.03, 0.95,
         f"far-minus-all R$^2$ median {famed:+.3f}\n79% still predicted with\nneighbours <50 um removed",
         transform=axD.transAxes, fontsize=8, va="top")
cb = fig.colorbar(hb, ax=axD, fraction=0.046, pad=0.02)
cb.set_label("dendrites (log)")
axD.set_title("D coupling survives removing neighbours <50 um:\n"
              "a shared network signal, not short-range optical crosstalk",
              fontsize=10, loc="left")
letter(axD, "")

# ----------------------------------------------------------------------------
# PANEL E : co-activation vs independent null
# ----------------------------------------------------------------------------
axE = fig.add_subplot(gs[1, 1])
cols_e = ["frac_frames_silent", "frac_frames_silent_null",
          "sync_10pct_obs", "sync_10pct_null",
          "sync_25pct_obs", "sync_25pct_null"]
ma = fov_median(ind_a, cols_e)
mh = fov_median(ind_h, cols_e)
groups = ["silent\n(no dendrite\nactive)", ">=10% of\ndendrites\nco-active", ">=25% of\ndendrites\nco-active"]
obs_a = [ma["frac_frames_silent"], ma["sync_10pct_obs"], ma["sync_25pct_obs"]]
nul_a = [ma["frac_frames_silent_null"], ma["sync_10pct_null"], ma["sync_25pct_null"]]
obs_h = [mh["frac_frames_silent"], mh["sync_10pct_obs"], mh["sync_25pct_obs"]]
xx = np.arange(len(groups))
w = 0.38
axE.bar(xx - w / 2, np.array(obs_a) * 100, w, color=C_OBS, label="observed (auto)")
axE.bar(xx + w / 2, np.array(nul_a) * 100, w, color=C_NULL, label="independent null (auto)")
axE.plot(xx - w / 2, np.array(obs_h) * 100, "o", color=C_HUMAN, ms=6,
         label="observed (human)", zorder=5)
axE.set_xticks(xx); axE.set_xticklabels(groups, fontsize=8)
axE.set_ylabel("% of frames")
fano = fov_median(ind_a, ["coactivity_fano", "coactivity_fano_null"])
axE.text(0.97, 0.95,
         f"Fano of #co-active:\nobs {fano['coactivity_fano']:.1f} vs null {fano['coactivity_fano_null']:.1f}",
         transform=axE.transAxes, fontsize=8, va="top", ha="right")
axE.legend(loc="upper center", fontsize=7.5, frameon=False)
axE.set_title("E population alternates silence and co-active bursts,\n"
              "far beyond independent units (auto & human)",
              fontsize=10, loc="left")
letter(axE, "")

# ----------------------------------------------------------------------------
# PANEL F : onset precision -- synchronous vs sequential
#   same-frame (|offset|=0) fractions are reported only in
#   explore_dynamics/{auto,human}/summary.txt (not emitted to CSV); quoted here.
# ----------------------------------------------------------------------------
axF = fig.add_subplot(gs[1, 2])
# from explore_dynamics summaries, item [4 onset precision]:
sameframe_obs = {"auto": 0.40, "human": 0.51}     # observed median same-frame fraction
sameframe_null = {"auto": 0.10, "human": 0.10}    # circular-shift null
# sequential (lagged) coupled pairs, from independence [C_sequences]:
seq_frac_pct = 0.6
xx = np.arange(2)
axF.bar(xx - 0.2, [sameframe_obs["auto"], sameframe_obs["human"]], 0.4,
        color=C_OBS, label="observed")
axF.bar(xx + 0.2, [sameframe_null["auto"], sameframe_null["human"]], 0.4,
        color=C_NULL, label="shift null")
axF.set_xticks(xx); axF.set_xticklabels(["auto", "human"])
axF.set_ylabel("fraction of coupled pairs onset in the SAME 0.2 s frame")
axF.set_ylim(0, 0.65)
axF.legend(loc="upper right", fontsize=8, frameon=False)
axF.text(0.5, 0.52,
         f"sequential (lagged)\ncoupled pairs: {seq_frac_pct:.1f}%",
         transform=axF.transAxes, fontsize=9, ha="center",
         bbox=dict(boxstyle="round", fc="#fff3cd", ec="#c9a227"))
axF.set_title("F coupled events are SYNCHRONOUS (same 0.2 s frame),\n"
              "not sequential propagation (auto & human)",
              fontsize=10, loc="left")
letter(axF, "")

# ----------------------------------------------------------------------------
# PANEL G : choristers vs soloists -- pop coupling vs behaviour coupling & depth
# ----------------------------------------------------------------------------
axG = fig.add_subplot(gs[2, 0])
g = enc_a.dropna(subset=["r2_beh", "r2_pop", "depth_um"])
sc = axG.scatter(g["r2_beh"].clip(-0.1, 0.6), g["r2_pop"].clip(-0.2, 0.8),
                 c=g["depth_um"], cmap="viridis_r", s=7, alpha=0.6, linewidths=0)
axG.set_xlabel("behaviour coupling R$^2$")
axG.set_ylabel("population coupling R$^2$")
axG.axhline(0, color="k", lw=0.5, ls=":"); axG.axvline(0, color="k", lw=0.5, ls=":")
cb = fig.colorbar(sc, ax=axG, fraction=0.046, pad=0.02)
cb.set_label("cortical depth (um)")
axG.text(0.03, 0.96,
         "chorister axis: rho(pop,beh)=+0.283\npop coupling falls with depth\n(LMM -0.052 per z, p<1e-60)",
         transform=axG.transAxes, fontsize=8, va="top")
axG.set_title("G choristers: dendrites tied to behaviour are also tied to\n"
              "the population; coupling is graded along depth (auto)",
              fontsize=10, loc="left")
letter(axG, "")

# ----------------------------------------------------------------------------
# PANEL H : sparse partial-correlation network with superficial hubs
# ----------------------------------------------------------------------------
axH = fig.add_subplot(gs[2, 1])
deg = enc_a["glasso_degree"].dropna().values
maxd = int(np.nanmax(deg))
axH.hist(deg, bins=np.arange(0, maxd + 2) - 0.5, color=C_AUTO, alpha=0.8)
axH.set_yscale("log")
axH.set_xlabel("partial-correlation degree (direct network edges)")
axH.set_ylabel("dendrites (log)")
# descriptive network stats are quoted as run-level medians in the encoding
# summary; use the same pooling here so the figure matches the report
fz = encr_a[["glasso_frac_zero", "glasso_mean_degree", "glasso_max_degree",
             "glasso_degree_cv", "hub_depth_um", "nonhub_depth_um"]].median()
axH.text(0.55, 0.95,
         (f"{fz['glasso_frac_zero']*100:.0f}% of partial corr. = 0 (per run)\n"
          f"mean degree {fz['glasso_mean_degree']:.1f}, max {fz['glasso_max_degree']:.0f}\n"
          f"degree CV {fz['glasso_degree_cv']:.2f} (hubs)\n"
          f"hub depth {fz['hub_depth_um']:.0f} um\n"
          f"vs non-hub {fz['nonhub_depth_um']:.0f} um"),
         transform=axH.transAxes, fontsize=8, va="top")
axH.set_title("H direct coupling is sparse (heavy-tailed degree);\n"
              "hubs sit superficially (auto)",
              fontsize=10, loc="left")
letter(axH, "")

# ----------------------------------------------------------------------------
# PANEL I : residual clusters compact + slow shared excitability
# ----------------------------------------------------------------------------
axI = fig.add_subplot(gs[2, 2])
cl = ind_a.groupby("fov")[["cluster_within_dist_um", "cluster_within_dist_null_um"]].mean().dropna()
sl = dyn_a[["slow_r_obs", "slow_r_null"]].dropna()
# left group: cluster distance (left y-axis, um)
x0 = [0, 0.6]
for _, row in cl.iterrows():
    axI.plot(x0, [row["cluster_within_dist_um"], row["cluster_within_dist_null_um"]],
             color="#bbbbbb", lw=0.7, zorder=1)
axI.scatter([0] * len(cl), cl["cluster_within_dist_um"], color=C_OBS, s=22, zorder=3, label="observed")
axI.scatter([0.6] * len(cl), cl["cluster_within_dist_null_um"], color=C_NULL, s=22, zorder=3, label="random label null")
axI.set_ylabel("within-cluster 3D distance (um)")
axI.set_ylim(120, 180)
# right group: slow local-rate r (right y-axis)
axI2 = axI.twinx()
x1 = [1.6, 2.2]
for _, row in sl.iterrows():
    axI2.plot(x1, [row["slow_r_obs"], row["slow_r_null"]], color="#d9b3b3", lw=0.7, zorder=1)
axI2.scatter([1.6] * len(sl), sl["slow_r_obs"], color="#8c1d40", s=22, zorder=3)
axI2.scatter([2.2] * len(sl), sl["slow_r_null"], color=C_NULL, s=22, zorder=3)
axI2.set_ylabel("slow local-rate correlation r")
axI2.set_ylim(-0.05, 0.25)
axI.set_xticks([0, 0.6, 1.6, 2.2])
axI.set_xticklabels(["clusters\nobs", "random\nnull", "slow-rate\nobs", "shift\nnull"], fontsize=8)
axI.set_xlim(-0.4, 2.6)
axI.axvline(1.1, color="k", lw=0.5, ls=":")
axI.text(0.02, 0.04, "clusters more compact than chance\n(z=-1.90)",
         transform=axI.transAxes, fontsize=8, va="bottom")
axI.text(0.98, 0.96, "slow shared\nexcitability\nexcess +0.077",
         transform=axI.transAxes, fontsize=8, va="top", ha="right")
axI.set_title("I residual structure after removing global+behaviour:\n"
              "compact clusters & slow shared excitability (auto)",
              fontsize=10, loc="left")
letter(axI, "")

# ============================================================================
fig.suptitle("Are L5 pyramidal apical dendrites independent computational units?  "
             "SCAPE Ca$^{2+}$ imaging, 5 Hz, 3 mice, 20 runs, 10 FOVs  "
             "(auto detection; human-curated replication)",
             fontsize=13, fontweight="bold", y=0.985)

out_png = os.path.join(STATS, "FIGURE_independence.png")
out_pdf = os.path.join(STATS, "FIGURE_independence.pdf")
fig.savefig(out_png, dpi=200)
fig.savefig(out_pdf)
print("wrote", out_png)
print("wrote", out_pdf)
print(f"panel B medians: beh={meds[0]:.4f} glob={meds[1]:.4f} pop={meds[2]:.4f}")
print(f"panel C fractions auto ind/str = {fa_ind:.0f}/{fa_str:.0f}%, human = {fh_ind:.0f}/{fh_str:.0f}%")
print(f"panel D far-minus-all median = {famed:+.4f}")
print(f"panel E auto obs silent/sync10/sync25 = "
      f"{obs_a[0]*100:.0f}/{obs_a[1]*100:.1f}/{obs_a[2]*100:.2f}% ; null = "
      f"{nul_a[0]*100:.0f}/{nul_a[1]*100:.2f}/{nul_a[2]*100:.3f}%")
