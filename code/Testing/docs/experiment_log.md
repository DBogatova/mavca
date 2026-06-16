# Experiment Log — MAVCA dendritic calcium analysis

Registry of analyses performed, their outcome, and status so we do not repeat dead-ends.
Status key: **ROBUST** (use it) · **NULL** (tested, no effect — don't repeat) ·
**ARTIFACT/DATA** (quality issue) · **EXPLORATORY** (weak/needs more) · **BLOCKED** (needs prereq).
Constraints throughout: ~5 Hz (no propagation/bAP/spike/nexus claims); SCAPE scan-order confounds
any depth-latency analysis; mask non-independence → family/FOV random effects.

## 1. ROBUST findings (use these)
| finding | statistic | script | figure |
|---|---|---|---|
| Dendrites are largely independent units | mean pairwise r 0.01–0.08; eff_dim/N 0.22–0.49; compartmentalization (1−resid r after global regression) 0.97–0.99 | quick_explore.py, pca_ica_analysis.py | quick_explore_participation.png |
| Superficial > deep activity | event_rate~depth LMM coef −0.15, p=0.002 (negative in 5/6 FOVs); family RE | dendrite_topology.py, run_models.py, fig2_morphology.py | fig2_morphology.png |
| ...gradient carried by LOCAL events | rate_local~depth −0.14 p=0.003; subtree p=0.10, global p=0.22 (n.s.) | run_models.py | fig3_organization.png (D) |
| Event participation is bimodal → local vs global regimes | GMM modes 0.06 / 0.75 → data-driven global cutoff 0.17; regime is FOV-specific | quick_explore.py, event_typing_behavior.py | quick_explore_participation.png |
| 0508 FOV: global Ca movement-coupled | global~accel r=+0.24 p=0.002 (circular-shift); NOT pupil (r=0.08 p=0.23) | behavior_triggered.py | fig1_behavior_coupling.png (A) |
| 0508: branch-specific movement coupling beyond global, real | 55% branches sig after global regression; survives event_rate/SNR/volume/depth (intercept +2.9, p=1e-13); coupled vs not don't differ on any covariate | behavior_controls.py | fig1_controls.png |

## 2. NULL / negative results — DO NOT RE-RUN
| test | result | script |
|---|---|---|
| shape / tortuosity / elongation vs activity | null (r≈0) | mask_propagation_analysis.py, fig1_behavior (branchiness) |
| branch_order & distance-to-trunk vs event_rate | NOT supported: inconsistent across FOVs (+0.48..−0.14), threshold-sensitive (0508 +0.32→0); LMM p=0.14/0.18 | dendrite_topology.py, fig2 (4,5) |
| within-mask top-down propagation | unresolvable at 5 Hz: depth chunks synchronous (slope ≈ +0.02–0.04 fr/chunk) | depth_chunk_neuropil.py |
| residual movement coupling vs depth (0508) | null: LMM depth p=0.41 (0.50 w/ covariates) | run_models.py (Q1/Q2) |
| spatial / family clustering of movement-coupled branches (0508) | null: permutation p=0.93 (spatial), 0.60 (family) → distributed | run_models.py (Q4) |
| event-class enrichment of coupled branches (0508) | null: frac_local p=0.45, subtree p=0.10, global p=0.68 | run_models.py (Q3) |
| global Ca vs pupil (0508) | n.s. (r=+0.08, p=0.23) | behavior_triggered.py |
| depth vs peak ΔF/F as a GENERAL rule | FOV-specific only (strong 0416_run123 & 0331_run67; pooled-across-FOV ≈ 0) | mask_propagation_analysis.py |

## 3. ARTIFACTS & data-quality issues
- **0416 run1/2/3 = contaminated; EXCLUDE.** Whole-FOV co-activation r≈0.85–0.90 across all 3 runs;
  PC1 = 90% variance but decoupled from behavior; survives core-shell subtraction. Field signal
  (neuropil/hemodynamic) that shell subtraction fails to remove in this FOV. Animal was still.
- **stack_voxel_norm_mean_sub is mean-subtracted** → forces in-mask vs out-of-mask r≈−0.97
  (artifact). NEVER use it for neuropil/contamination tests — use the raw stack (in/out r≈0.99–1.0
  in all FOVs = normal field signal).
- **Behavior: full-trace Pearson is misleading** (Ca vs whisker/accel/pupil differ in frequency;
  can be event-locked with opposite sign, e.g. 0512 Ca rise → eye movement → pupil drop). Use
  event/peak-locked coupling (movement-onset triggered averages) only.
- **5/6 FOVs show a global Ca DROP at movement onset** (−0.08 to −0.26 z), not activation; only
  0508 shows positive global movement-activation. → global movement coupling is 0508-specific.
- **'subtree' event class depends on fragile family reconstruction** → trust family-independent
  local/global classes; treat subtree cautiously.
- 0331 run6 had a stale 33-dend trace (≠30 masks); regenerated via M4.

## 4. Methods constraints & decisions
- 5 Hz: defensible = event rate/amplitude/synchrony/recruitment-order ≥200 ms + behavior coupling.
- SCAPE scan-order: gate any depth-latency analysis (subtract per-voxel acquisition offset or test
  vs scan axis); not attempted.
- Tree topology (branch order / path distance / soma) needs real reconstruction; current
  spatial-adjacency families are fragile/threshold-sensitive.
- statsmodels installed (mixed-effects). Use family/FOV random effects always.
- F0 baseline = 10th percentile; background = core-shell with 1-voxel gap (pipeline convention).

## 5. Datasets & shared-mask (same-FOV) groups  [hash-verified]
Identical-mask groups (dend IDs match → poolable): 0416 run2/3 (80) & run5/6/7 (133);
0508 run5/6 (62); 0512 run5/6 (68) & run9/10 (122); 0331 run6/7 (30, masks in run7) &
run8/9/10 (51, masks in run8). 0416 run1 = same FOV as run2/3 but different curation (excluded).
0320 rbp4cre_139 run1/run3 = distinct FOVs. Per-dataset SKIP_FIRST_SECONDS: 0508=14 s, others=12 s.

## 6. Scripts (code/Extra/ unless noted) & outputs
| script | purpose | key output |
|---|---|---|
| mask_propagation_analysis.py | morphology + activity + within-mask propagation + per-FOV/group correlations | mask_metrics_*.csv |
| event_typing_behavior.py | event synchrony classes, global Ca, behavior, hubs, branchiness | event_behavior_summary.csv |
| pca_ica_analysis.py | eff_dim/N, PC1 behavior coupling, ICA assemblies | pca_ica_summary.csv |
| quick_explore.py | participation-fraction GMM + per-FOV diagnostic panel + contamination flag | quick_explore_*.csv/png |
| behavior_triggered.py | pooled movement-onset BTA, raw vs residual, per-mask null | behavior_triggered_pooled.png |
| behavior_controls.py | residual coupling survives event_rate/SNR/volume/depth | fig1_controls.png |
| screen_behavior.py | per-FOV behavior-coupling viability | behavior_screen.csv |
| dendrite_topology.py | branch families/order/dist-trunk across FOVs | topology_pooled.csv |
| build_table.py | master per-mask table (all FOVs) | master_mask_table.csv |
| run_models.py | Q1–Q5 mixed-effects + permutation | (stdout) |
| fig1_behavior.py, fig1b_population.py, fig2_morphology.py, fig3_organization.py | publication figures | scape-data/figures/ |
| Traces-STEP3/run_m4_batch.py | run M4 for a given run (overrides config) | per-run traces |
| Preprocessing-STEP1/deepcad_prep.py | split/merge 4D↔per-plane for DeepCAD-RT | — |

## 7. Tooling / infra
- M3 curation hotkeys added: `e` erase paint-dotted component, `c` keep largest component,
  `l`/`p` lasso keep/delete inside polygon (all Z), `t` toggle 2D/3D.
- DeepCAD-RT plan: M1 Max can't run it (CUDA); use BU SCC GPU (OnDemand Jupyter) or Colab;
  denoise per-Z-plane (deepcad_prep.py); feed denoised stack to M1/M2 only, keep M4 on raw.
- venv: use `.venv311/bin/python -m pip` (the `.venv311/bin/pip` shim points at python3.14).

## Central defensible claims (conservative)
1. L5 apical dendritic segments are largely functionally independent (low pairwise correlation,
   high effective dimensionality), with local events dominating.
2. Superficial compartments are more active than deep, a gradient carried by local events.
3. In a behavior-rich FOV (0508), dendritic calcium has both a movement-coupled global component
   and a spatially-distributed branch-specific movement-coupled component that survives controls;
   coordinated global movement-activation is FOV-specific, not general.
No claims of propagation, backpropagation, dendritic spikes, or apical-nexus identification.

## 8. SCAPE scan-order geometry (resolved 2026-06)
Raw stack (T,Z,Y,X)=(600,30,199,395); ImageJ meta images=18000=600 vol x 30 slices. Geometry:
each ~200 ms volume = 30 sequential camera snapshots, one per **Z galvo step** (Z=3.9 um coarse).
Within each snapshot, full **cortical depth (Y)** + X captured **simultaneously**.
- Top-down (depth/Y) recruitment is **scan-clean** (depths simultaneous per Z-plane) -> #7 viable along depth.
- Z (lateral) axis has intra-volume acquisition gradient ~200/30 = 6.7 ms/plane (full sweep ~200 ms)
  -> Z-direction recruitment is confounded; subtract per-mask Z-offset (z_centroid*6.7ms) before depth-order tests.
- AndorXylaTrigger is a sparse start/block trigger (1193 pulses), NOT per-frame; camera free-runs.
  Exact galvo waveform/sweep direction is in trigger/*_t1.mat (v7.3 HDF5, raw 1kHz channels) if needed.
- Viable #7: cross-mask recruitment order vs depth in co-active (global) events, Z-offset-corrected,
  scan-strata permutation null. Mostly testable in 0508 (global-rich); expect near-synchrony at 5 Hz.

## 9. #7 Depth recruitment order — PROMISING (needs replication)
Cross-mask recruitment order vs cortical depth in 0508 global events, scan-corrected
(depth/Y scan-clean; Z-acquisition offset z*6.7ms subtracted). recruitment_order.py / fig4_recruitment.png.
- SUPERFICIAL->DEEP (top-down) recruitment: pooled slope +1.33 ms/um (perm p=0.002); per-EVENT
  8/9 events +slope, median +1.23 ms/um, sign-test p=0.039. Onset spread ~3.8 frames -> SLOW
  (~150-200ms over depth ≈ ~1 frame), a recruitment-order/latency gradient, NOT fast propagation/bAP.
- Status: PROMISING but single FOV (0508), only 9 global events. Needs replication in additional
  global-event-rich recordings before any claim. This is the quantified version of the user's
  "top-down in videos" observation, now scan-controlled.

## 9b. #7 replication — FAILED (downgrade to 0508-specific)
Ran scan-corrected depth recruitment on 0331_r67, 0331_r8910, 0512_r56. NOT replicated:
- 0331_r67 (5 global events): slope +0.11 ms/um, perm p=0.71, 2/5 events, onset spread 0.3 frames
  (SYNCHRONOUS) -> no recruitment order.
- 0331_r8910, 0512_r56: 0 global multi-mask events spanning depth -> untestable.
Conclusion: the top-down recruitment order is 0508-SPECIFIC and requires EXTENDED global epochs
(0508 spread 3.8 fr ~0.8s; 0331_r67 brief/synchronous). Do NOT generalize. Frame strictly as a
single-FOV observation tied to 0508's extended movement-coupled global epochs; not replicated.

## 10. NMF spatial motifs (0508, EXPLORATORY) — nmf_motifs.py / fig5_nmf.png
V(masks x time)>=0 ~ W H, K=5 (weak elbow; recon err 104->70 over K2..8, gradual). Recovers:
- 1 GLOBAL motif: dominant for 38/62 masks, broad/depth-distributed, movement-coupled (H~accel r=+0.55).
- 4 LOCAL motifs: 3-11 masks each, spatially restricted, NOT movement-coupled (r~0.08-0.10).
Convergent with PCA PC1 + residual-coupling story (global movement mode + local branch modes).
Caveats: weak K elbow, single FOV, exploratory. Other FOVs (local-dominated) would likely yield
mostly local motifs.

## 11. External-drive rbp4cre Ca sessions added (2026-06)
Pulled (symlinked; traces local, archive untouched) from /Volumes/IMAC/data the NEW rbp4cre runs
with masks (not already in scape-data): 2026-02-09/rbp4cre_136_phpeb/run1, 2026-02-17/rbp4cre_138_phpeb/run7.
These are DUAL-CHANNEL deeper volumes: green=Ca, red=ACh; Z=69/55 planes (NOT 30; Z voxel unknown);
728-frame, NO behavior/trigger. Ca (green) traces regenerated via M4 (run_m4_batch.py with -green
override) -> aligned to masks (33/34, 45/45). characterize_new.py.
- INDEPENDENCE replicates across new mice: mean_r=0.04, eff_dim/N=0.56-0.62 (HIGHER than originals),
  frac_global 0.11-0.14 (local-dominated). Strengthens the independent-units result across 3 mice.
- DEPTH gradient does NOT replicate: 0209 r=+0.14 (n.s.), 0217 r=+0.34 p=0.022 (REVERSED, deep>superficial).
  => superficial>deep is FOV/mouse-heterogeneous, NOT universal. Report as condition-dependent.
- NOT pooled into depth LMM / topology (different Z planes, unknown Z voxel, dual-channel) to avoid errors.
- ACh (red channel) = FUTURE direction (more data coming): freq/correlation/coherence/PCA, dual-channel
  Ca-ACh coupling (ach_ca_traces_m8.py, dual_channel_coupling.py scaffolding exists).

## 12. Mask-level functional heterogeneity (2026-06, no topology) — Q1-Q5
Scripts: build_fingerprints.py, fig_hotspot_cluster.py, fig_footprints.py, fig_grammar_sign.py.
Tables: mask_fingerprint_table.csv, hotspot_summary.csv, event_footprint_table.csv, state_event_grammar.csv.
Figs: fig8_hotspots, fig9_funcclusters, fig10_footprints, fig11_grammar_sign.
Q1 HOTSPOTS: events NON-uniform (Gini 0.41-0.59); top10% masks=26-47% events, top20%=46-65%;
   hotspots moderately stable (half rho 0.20-0.64, across-run 0.23-0.64). => functional heterogeneity real.
Q2 FUNCTIONAL CLUSTERS (functional features only): 2 clusters (sil 0.32, 178 masks) = local-preferring
   (119) vs global-participating (59, frac_global 0.46); differ by depth only (KW p=0.02), NOT SNR/vol/
   orient; cluster~FOV AMI=0.30 (partly driven by global-rich 0508). Heterogeneity ~morphology-independent.
Q3 FOOTPRINTS: events mostly LOCAL (frac_local 0.53-0.67), global rare (0.01-0.12); BUT recurring
   multi-mask motifs ABOVE shuffled-mask null in ALL FOVs (recur 0.02-0.18 vs ~0, p=0.003) -> modest
   reproducible co-activation motifs (events not purely random).
Q4 STATE GRAMMAR (0508): movement strongly increases event rate (quiet 8 -> pre 118 -> onset 314 /min)
   AND shifts balance to GLOBAL (frac_global quiet 0.00 -> pre 0.48 -> onset 0.34 -> post 0.05); amp higher
   in movement; footprint diversity LOWER in movement (0.30 vs 0.60 quiet). 'sustained' underpopulated
   (short bouts). => movement reconfigures local<->global balance.
Q5 COUPLING SIGN (0508): pos=21/neg=14/unc=27. pos-coupled modestly more superficial (depth 81 vs neg 111um,
   p=0.046) + higher SNR (p=0.019); event_rate/lg_pref n.s. BUT sign poorly reproducible run5-vs-run6
   (41% agreement among ever-coupled) -> pos/neg NOT a stable mask identity; interpret cautiously.

## 13. UPDATE (2026-06-04): 0416 run1/2/3 "contamination" RESOLVED — extraction artifact, not biology
The earlier "contaminated FOV" (§3) was traces extracted WITHOUT core-shell background subtraction.
Test (run1, raw stack): core-only mean_r=0.91 (rank-1); core-SHELL mean_r=0.06 (independent);
aggressive core alone does NOT fix (0.82). So contamination = missing background subtraction; the
field is fine (v1 curation was already independent).
Fix: re-curated run1 (cropped 19 mixed-Y masks 199->196; removed dend_006/036/048; merged 11
find_dendrite_branches groups -> 54 masks), re-ran M4 core-shell for run1/2/3 (run2/3 via
MASK_SOURCE_RUN=run1). Result: all 54 dends, mean_r 0.03-0.07, eff_dim/N 0.22-0.35 = CLEAN.
=> 0416 run1/2/3 is now a usable clean FOV (mixed regime, ~40% global, NOT movement-coupled).
=> The "contamination negative control" in results_revised/paper_figures is now obsolete; reframe as
a methods illustration (background subtraction is essential) or drop. Backups: *_precrop_backup,
*_MERGEBACKUP_20260604_143020.

## 8. NMDA-spike vs bAP "vertical-streak" amplitude/morphology testing (2026-06)
Full write-up: **code/Testing/RESULTS.md**. Code/figures/tables: **code/Testing/** (isolated from pipeline).
Hypothesis: NMDA (localized) transients < bAP (extended/vertical-streak) transients in amplitude.

**Verdict: amplitude is NOT a robust discriminator at 5 Hz** (4 independent operationalizations). Don't re-run as a general claim.

NULL / ARTIFACT — DO NOT RE-RUN:
- static mask morphology (length/verticality/linearity) vs peak ΔF/F → null (vertical-streak vs localized masks p=0.48). `nmda_bap_amplitude.py`
- within-dendrite branch split, SNR-robust classifier → null (extended vs localized p=0.70). The earlier strong result (1.6×, p≈2e-19, noise-threshold classifier) was a **detection-threshold artifact** (REL_FRAC sweep: significant only at extreme cutoffs). `nmda_bap_split_v2.py`, `diag_relfrac_sweep.py`
- event kinetics (FWHM/decay/area) isolated vs whole-tree → null (FWHM 0.60 s both); NMDA plateau unresolvable at 5 Hz. `nmda_bap_features.py`
- active-region morphology (streak vs tilted) → splits 2 real shape classes but amplitude flat (p=0.35). `nmda_bap_morpho_classify.py`

EXPLORATORY (FOV-specific, weak):
- per-EVENT amplitude vs co-activation extent: isolated < whole-tree pooled p=3e-6 but FOV-specific; within-mask coupling control modest (7/10 masks, +2%, p=0.024; residual-field caveat). `nmda_bap_amplitude.py`, `nmda_bap_features.py`
- whole-mask trunk vs branch per-segment amplitude: trunk>branch only in 0508/run5 (p=0.02) & 0320/run3 (p=0.006), small (1.1–1.2×); null in 4 other FOVs. `nmda_wholemask_run7.py`

METHODS (add to §3/§4):
- **Illumination gradient** (dim-left→bright-right ~1.1–1.23×) removed by ΔF/F; classes matched on X & brightness → not a confound. `fov_geometry_diagnostics.py`
- **Pial surface tilted** (X-tilt ≤12°, ~55 µm Y-span): slope-correct DEPTH; do NOT slope-correct ORIENTATION (dendrites align ~equally to image-Y vs surface-normal). `diag_orientation.py`
- **"Silent" masks = core-shell cancellation** (core active, shell≈core r≈0.96–1.0), not inactive dendrites. `diag_silent_*.py`

TOOL (structural, not functional): trunk/branch/combined morphology classifier `nmda_bap_gallery2.py::decompose()`
(LIN_TRUNK=0.80, BRANCH_ANGLE=45°, BRANCH_MIN_FRAC=0.10, TRUNK_FRAC=0.30). Matches ~5/8 hand-checked masks;
trunk-vs-fork limit is topological (PCA can't see branch count). Use as anatomy, decoupled from amplitude.
