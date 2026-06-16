# Results

## Dendritic calcium activity is movement-coupled at both global and branch-specific levels

We recorded volumetric calcium activity from L5 pyramidal apical dendrites (SCAPE, ~5 Hz)
and extracted ΔF/F traces from hundreds of curated dendritic-segment masks across multiple
fields of view (FOVs), with simultaneous accelerometer and pupil measurements. Because the
imaging rate is ~5 Hz (200 ms/frame), we restrict all claims to event rate, amplitude,
spatial co-activation, and behavioral coupling; we make no claims about fast electrical
propagation, backpropagating action potentials, dendritic spikes, or identification of the
apical nexus.

To separate whole-tree from branch-specific dynamics, we defined a global calcium signal as
the across-mask median trace and computed, for each mask, a global-regressed residual
(`residual_i = trace_i − β_i · global`). Behavioral coupling was assessed with event/peak-locked
analyses rather than full-trace correlation, because calcium and behavioral signals differ in
their temporal frequency content and can be transiently coupled with opposite sign.

In the behavior-rich rbp4_139 (0508) FOV (two runs sharing identical masks pooled; N = 62 masks,
46 movement onsets), **movement onset was associated with a clear increase in global dendritic
calcium** (Figure 1A; global ΔF/F vs accelerometer r = +0.24, circular-shift permutation
p = 0.002; coupling to pupil was not significant, r = +0.08, p = 0.23). However, movement coupling
was not entirely explained by this shared global mode. Detected branch calcium events were
preserved after regressing out the global signal (Figure 1B), and using a per-mask circular-shift
null (Figure 1C; n = 62 masks), **69% of branches were significantly movement-coupled in the raw
traces and 55% remained significantly coupled after global regression** (Figure 1D). These
residual movement-coupled branches were spatially distributed across the field (Figure 1E) and
were **not enriched by cortical depth (Mann–Whitney p = 0.31), branch family, or event class**
(global-participation fraction p = 0.27; Figure 1F), suggesting that branch-specific movement
coupling in this FOV was spatially distributed rather than confined to a single depth compartment.

Critically, this branch-specific coupling was not an artifact of detectability. Significantly
coupled and uncoupled branches did not differ in event rate, SNR, mask volume, or cortical depth
(all p > 0.3); in a linear mixed-effects model (per-mask power-adjusted coupling effect size ~
event rate + SNR + volume + depth, random intercept by branch family), none of these covariates
predicted coupling strength, while a strong baseline coupling remained (intercept = +2.9 SD above
the shuffled null, p < 1e-13). Thus residual movement coupling reflects genuine branch-specific
encoding rather than higher signal quality in active branches.

To move beyond a single FOV, we extended the analysis across all six shared-mask-group FOVs
using event-locked measures (movement-onset triggered averages and per-mask circular-shift
nulls) rather than full-trace correlation, which is unreliable across signals with different
frequency content (Figure 1B). **Branch-specific movement coupling was widespread: 40–70% of
branches (median 57%) were significantly coupled in every FOV** (Figure 1B-C). In contrast,
whole-tree *global* movement activation was FOV-specific: the global calcium signal rose at
movement onset only in the rbp4_139 (0508) FOV (+0.51 z at 0–0.6 s), whereas the remaining five
FOVs showed a small global decrease at movement onset (−0.08 to −0.26 z; Figure 1B-A,B),
consistent with a movement-associated dip rather than global activation. Substantial
whole-tree global events were likewise present only in 0508 (and weakly in one 0331 FOV;
Figure 1B-D). Thus branch-specific movement coupling is a population-level property, while
coordinated global movement-driven recruitment is the exception rather than the rule.

## Activity is graded by cortical depth, but not by reconstructed branch order

Across all FOVs (466 masks, 6 FOVs), we related per-mask event rate to morphology using
linear mixed-effects models with a random intercept for branch family (within-FOV z-scored
variables, to account for differing depth ranges and non-independence of masks within a cell/FOV).

**Cortical depth was a robust predictor of activity: superficial dendritic compartments showed
higher event rates than deeper compartments** (Figure 2.1–2.2; depth coefficient ≈ −0.15,
p = 0.002). This relationship was negative in 5 of 6 FOVs (Figure 2.3), consistent across the
dataset.

In contrast, **topology-derived predictors were not supported.** Reconstructing dendritic branch
families from spatial adjacency (root = deepest/most-proximal mask; branch order = graph hops from
root), neither branch order nor distance-to-trunk predicted event rate beyond depth in the mixed
model (branch order p = 0.14; distance-to-trunk p = 0.18; Figure 2 panel 6 table). Per-FOV
branch-order slopes flipped sign across FOVs (+0.48 to −0.14; Figure 2.4), and the apparent
branch-order effect in individual FOVs was unstable to the reconstruction threshold: in one FOV
the correlation collapsed from r = +0.32 to ≈ 0 as the adjacency distance increased, while another
remained negative throughout (Figure 2.5). We therefore do not interpret branch order or
distance-to-trunk as biological predictors with the current segment-based reconstruction; a
registered anatomical reconstruction would be required to test branch-order effects rigorously.

## Summary of robust findings

1. Global dendritic calcium is coupled to locomotion (movement), not to pupil, in mixed-regime FOVs.
2. In the behavior-rich 0508 FOV, movement-related dendritic calcium contained both a global
   (movement-activation) and a branch-specific component: 55% of branches retained significant
   movement coupling after global regression, and this survived controls for event rate, SNR,
   mask volume, and depth. Branch-level coupling was also present in other FOVs (40–70%), but
   whole-tree global movement-activation was specific to 0508; we frame the global result as
   FOV-specific rather than general.
3. Superficial dendritic compartments show higher activity than deeper compartments (robust across
   FOVs under family random effects).
4. Topology-derived branch-order and distance-to-trunk effects are **not** supported with the
   current reconstruction (inconsistent across FOVs and unstable to reconstruction parameters).

We make no claims of direct propagation, backpropagation, dendritic spikes, or apical-nexus
identification; the 5 Hz volumetric sampling does not resolve fast dendritic events, and SCAPE
acquisition order would confound any depth-latency analysis (not attempted here).

---
*Figures:* `scape-data/figures/fig1_behavior_coupling.png`, `scape-data/figures/fig1_controls.png`, `scape-data/figures/fig1b_population_behavior.png`, `scape-data/figures/fig2_morphology.png`.
*Mixed-effects coefficients:* `scape-data/figures/fig2_mixedmodel_coefs.csv`.
*Pooled per-mask metrics:* `scape-data/topology_pooled.csv`.
