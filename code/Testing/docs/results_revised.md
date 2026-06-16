# Results (revised — calcium, cross-mouse)

We extracted ΔF/F calcium traces from curated L5 apical dendritic-segment masks across **8 fields
of view (FOVs) from 4 mice** (rbp4_132, rbp4_139, rbp4cre_136, rbp4cre_138), spanning two
acquisition modes (single-channel 30-plane and dual-channel deeper volumes). One FOV (2026-04-16
run1/2/3) was excluded from biological claims as a contamination control (see below). Imaging is
~5 Hz; we make no claims about fast propagation, backpropagation, dendritic spikes, branch-order
effects, or apical-nexus identification.

## Main result: dendritic segments are high-dimensional, weakly correlated, and mostly locally active

Across all 8 FOVs and 4 mice, dendritic-segment activity was **high-dimensional and weakly
correlated** (Figure 6). Mean pairwise correlation was low (**r = 0.01–0.08**), the effective
dimensionality was high (**participation ratio / N = 0.22–0.62**), strongly-correlated pairs were
rare (**3–22% of pairs with |r|>0.3**), and individual events involved only a small fraction of
masks (**median participation 0.04–0.16**). PCA required many components to capture the variance
(no dominant shared mode). This held in **both acquisition modes and was, if anything, strongest in
the two newest mice** (rbp4cre_136, rbp4cre_138: eff_dim/N = 0.62, 0.56), establishing functional
independence as robust across mice, sessions, and acquisition modes.

This independence survived every control (Figure validation):
- **Signal quality:** restricting to high-SNR masks left correlation low (r = 0.03–0.16) and
  effective dimensionality high or higher (eff_dim/N = 0.28–0.69) — not a low-SNR artifact.
- **Amplitude/event periods:** restricting to high-amplitude (event) frames kept correlation low
  (r = 0.005–0.057, often *lower* than baseline) and dimensionality high (0.21–0.53) — masks do not
  co-fire even when active.
- **Time stability:** first- vs second-half effective dimensionality were comparable within each FOV.
- **Spatial dependence:** across 22,038 mask pairs (6 known-geometry FOVs), pairwise correlation was
  nearly independent of 3-D distance (r = −0.06) and of cortical-depth difference (r = −0.02); only a
  marginal proximity effect on coactivation (r = −0.11) was present. (A branch-family comparison was
  not interpretable because the segment-based family reconstruction over-merges; see Methods.)

**Contamination control.** A single FOV (2026-04-16 run1/2/3) showed the opposite of all of the above
— mean r ≈ 0.90, effective dimensionality ≈ rank-1 (eff_dim/N = 0.02), 93% event participation, and a
dominant mode decoupled from behavior — in an essentially immobile animal. Its global co-activation
survived local background subtraction but tracked a field-wide signal, indicating neuropil/hemodynamic
contamination rather than biology; it is excluded from all biological claims and shown only as a
negative control.

## Secondary: rare global co-activation regimes are FOV/condition-specific

Coordinated whole-tree activity was the exception. Only one FOV (rbp4_139, 0508) showed a substantial
global-event regime, and there the global mode was **movement-coupled** (global ΔF/F vs accelerometer
r = +0.24, p = 0.002, by event-locked permutation; not pupil); branch-specific movement coupling
persisted after global regression in ~55% of branches and survived controls for event rate, SNR,
volume, and depth. A second FOV (0331 run6/7) had a few global events but they were temporally
synchronous with no recruitment order. All remaining FOVs were local-dominated (global-event fraction
≈ 0). We therefore present global co-activation and its movement coupling as **FOV/condition-specific**,
not a general property.

## Depth gradient is heterogeneous (downgraded)

An earlier within-dataset observation that superficial segments are more active than deep ones did
**not generalize**. Event rate decreased with depth in 5/6 original FOVs (pooled mixed-effects depth
coefficient ≈ −0.15), but the two newest FOVs were flat (rbp4cre_136, r = +0.14, n.s.) or
significantly **reversed** (rbp4cre_138, r = +0.34, p = 0.022). We therefore report the depth–activity
relationship as **heterogeneous / condition-dependent**, not a robust cross-mouse gradient.

## What we do not claim
No fast propagation, backpropagation, dendritic spikes, branch-order/distance-to-trunk effects, or
apical-nexus identification. Branch-order and topology-derived metrics were unstable to reconstruction
parameters and inconsistent across FOVs (not supported). 5 Hz volumetric sampling limits all timing
claims; SCAPE acquisition order (Z = sequential galvo sweep; cortical depth Y captured simultaneously
per snapshot) was accounted for where relevant.

---
*Figures:* fig6_independence.png (main), validation_independence.csv, fig7_spatial.png (spatial),
fig1_behavior_coupling.png / fig1_controls.png (0508 movement coupling). Per-mask/FOV tables:
master_mask_table.csv, validation_independence.csv, spatial_pairs.csv.
