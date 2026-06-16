# Dendritic Calcium Analysis Plan (L5 apical dendrites, SCAPE, ~5 Hz)

Grounded in dendritic physiology; calibrated to 5 Hz volumetric imaging; tied to the
MAVCA pipeline outputs (curated masks + ΔF/F traces + accel/pupil behavior).

## Two framing constraints

**A. 5 Hz limit.** 1 frame = 200 ms. bAPs / dendritic spikes / electrical propagation are
unresolvable. Defensible: event rate, amplitude, spatial extent/synchrony, recruitment order
at ≥200 ms, behavioral coupling. Use "spatial recruitment order", "latency-gradient evidence",
"distal-to-proximal recruitment", "co-activation" — never "propagation velocity / bAP / spike".

**B. SCAPE scanning-order confound (critical, gates all latency analysis).** A SCAPE volume is
swept; one spatial axis is sampled sequentially within each ~200 ms volume. If cortical depth (Y)
correlates with the sweep axis, "top-down" latency can be a pure acquisition-timing artifact.
Before any latency claim: (1) recover per-voxel intra-volume acquisition time from scan geometry;
(2) subtract each mask's mean acquisition offset OR test whether the gradient aligns with the scan
axis; (3) null = shuffle depth labels within scan-time strata.

**Prerequisite gap: tree topology.** Branch order, path distance, distance-to-trunk, betweenness,
parent-branch angle need a connected dendritic graph + defined soma/trunk, which we do NOT have.
Available now: centroid xyz, depth (Y), volume, skeleton length, diameter proxy, tortuosity,
orientation, branchiness, and subtree identity (`find_dendrite_branches.py`). Reconstruct topology
before promising the rest. Sections below tagged [now] vs [needs topology].

## 1. Event detection & classification
Per-mask onset/peak/amplitude/duration on detrended ΔF/F (10th-pct F0, ≥1-frame crossing).
Data-driven classes via the **participation-fraction distribution** (fraction of masks active
±1 frame); fit GMM, cut at the antimode (NOT a round number):
- local (1 mask / small cluster), subtree (≥2 in same branch-family), global (≥ high mode in 1–2 fr),
  distal-first / proximal-first (onset-rank ~ depth, see §3), mixed/sequential.
Require subtree events to be in the same `find_dendrite_branches` family (else mere coincidence).

## 2. Global calcium signal (biology + nuisance)
Compute median / mean / PC1; median most robust. As nuisance: `resid_i = trace_i − β_i·global`,
redo analyses on residuals (branch-specific structure that survives = defensible local computation).
Identify PC1 by correlating with movement, pupil, out-of-mask neuropil (from RAW, not mean-sub
stack), frame brightness, motion. (Observed: movement-coupled in 0508/run5; quiet-state in
0331/run8; dominant-but-behavior-decoupled in 0416/run1 → contamination.)

## 3. Top-down vs bottom-up latency [gated on §B]
Depth bins (4–5 along Y) [now]; path-distance bins [needs topology]. Onset per chunk = first
≥1-frame crossing, parabolic sub-frame refine (±~50 ms). Per event regress onset vs depth/path;
classify by slope sign + |slope|·extent ≥1 frame. Permutation nulls: shuffle depth labels;
shuffle within scan-time strata. Pool slopes across events for the distribution.

## 4. Morphology ↔ activity (mixed-effects, mandatory)
Masks within a cell/FOV are not independent → random effects:
`event_rate ~ depth + branch_order + tortuosity + (1 + depth | cell)` (Poisson GLMM for counts);
`amplitude ~ volume + depth + branch_class + (1|cell)`;
`global_participation ~ distance_to_trunk + branch_order + (1|cell)`.
(Observed: event-rate ↓ with depth in all FOVs; effect is FOV-specific → use random slopes.
Tortuosity unrelated to activity.)

## 5. Functional hub / candidate integration zone
Functional: lead score, participation score, correlation-graph degree, dual-participation
(active in both subtree AND global events). Anatomical [needs topology]: branch-point proximity,
betweenness, bridges distal↔proximal. Combine (z-sum) → "candidate functional integration zone"
(never "the apical nexus"). NB ICA assemblies were spatially intermixed → expect no tight hub.

## 6. Behavior
Global coupling: event-triggered avg of global Ca around movement-onset/whisk/pupil. Local
coupling: repeat on global-regressed residuals. Encoding (behavior→trace) + decoding
(population→behavior), cross-validated, raw vs residual. Controls: movement+neuropil regressors;
time-shuffle null for every correlation.

## 7. PCA / ICA / NMF / seqNMF
PCA: `eff_dim/N` (effective dimensionality), PC1=global mode, back-project loadings onto 3D masks;
later PCs → tuft-vs-trunk / branch-family. ICA: modules but unstable (report only if seed-stable).
**NMF (preferred):** nonnegative, additive → motifs (global / distal-tuft / proximal-trunk /
branch-cluster); rank by error elbow + stability. seqNMF: repeated sequences (tuft→trunk→global),
riskiest at 5 Hz — exploratory only.

## 8. Frequency / temporal structure
OK: event rate/min (local/subtree/global), IEI distribution, burstiness, autocorrelation time,
slow-band (<1 Hz) compartment coherence, event-rate change around behavior. NOT OK: fast rhythms
(Nyquist 2.5 Hz; GCaMP low-passes). State limits.

## 9. Cell-level phenotyping [needs cell grouping; FOV-level for now]
Features: length/volume, #masks, complexity, event rate, local/subtree/global fractions,
synchrony (eff_dim/N), behavior coupling, distal/proximal-first fractions, compartmentalization
index (1 − mean residual-after-global r). Cluster → globally-active / compartmentalized /
behavior-coupled / tuft- / trunk-dominant / silent. Validate (silhouette, bootstrap).

## 10. Controls & artifact checks (priority by what bit us)
1. Neuropil/global contamination: inter-mask r before/after global+neuropil regression; PC1 vs
   out-of-mask ΔF/F on RAW (mean-sub stack forces spurious anti-correlation — do not use).
2. Scanning-order (§B) for every latency claim.
3. Motion/z-drift: per-frame metric, exclude frames, global-event vs motion-time test.
4. Bleaching: detrend; event-rate stationarity.
5. Mask-size & SNR: covariates; confirm effects survive (depth survived volume+length).
6. Non-independence: cell/FOV random effects.
7. Two nulls: preserve-rate/shuffle-space; preserve-space/shuffle-time.

## 11. Figures
3D masks colored by depth/family/phenotype; depth-sorted trace heatmap; participation-fraction
histogram w/ data-driven cutoffs; example events on morphology colored by onset (scan-order
annotated); latency–depth slope distribution vs null; PCA/NMF motifs on morphology; behavior-
triggered maps (raw vs residual); morphology vs event-rate w/ LMM fit; integration-zone map;
phenotype clustering.

## 12. Statistics → question
Latency → permutation (label + scan-strata). Morphology↔activity → LMM/GLMM (Poisson), random
intercept+slope on cell. Event-class probs → GLM + FDR. Encoding/decoding → CV R²/acc vs null.
Effects → bootstrap CIs. Clustering → silhouette + bootstrap.

## 13. Roadmap
**Quick exploratory:** (1) participation-fraction GMM → event classes; (2) eff_dim/N + PC1-
behavior-neuropil diagnostic per FOV; (3) global-regression → compartmentalization index.
**Robust/publication:** (4) reconstruct tree topology; (5) mixed-effects morphology↔activity;
(6) behavior coupling raw vs residual w/ controls; (7) latency gradients AFTER scan-order fix.
**Optional advanced:** NMF motifs on morphology; cell phenotyping.
**Risky (phrase carefully):** seqNMF; single-event distal/proximal calls; integration-zone claims.
**First:** resolve the 0416 run1/2/3 contamination (inflates global/synchrony stats).

## Central claim (conservative-but-exciting)
> Dendritic calcium events decompose into local, subtree, and global recruitment motifs whose
> proportions are structured by dendritic morphology (depth, branch identity) and behavioral
> state (movement-coupled global vs quiet-state local). After controlling for acquisition order,
> mask volume, and a global/neuropil component, a subset of events shows reproducible distal-to-
> proximal (or proximal-to-distal) recruitment ordering on a 200–600 ms timescale — consistent
> with distinct dendritic recruitment modes, though 5 Hz volumetric sampling precludes direct
> measurement of fast electrical propagation.
