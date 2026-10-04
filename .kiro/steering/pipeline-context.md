---
inclusion: always
---

# MAVCA Pipeline — Working Context

This file is auto-loaded every session. It captures project conventions, dataset
configs, key decisions, and common tasks. See `README.md` for full pipeline docs.

## Project
- **MAVCA**: dendritic calcium analysis from 4D SCAPE imaging of L5 pyramidal cell
  apical dendrites (cortex). Goal: characterize local dendritic computation.
- Venv: `source .venv311/bin/activate` (Python 3.11).
- Physical voxel size: `(3.9, 1.0, 1.2)` µm (Z, Y, X). Frame rate: 5 Hz.
- SCAPE axes: **Y = cortical depth** (surface at top), X/Z = lateral. Movies show Y×X MIP.

## Pipeline order
M1 (find_events_m1) → M1.5 (pre_segmentation) → M2 (auto_mask) → M2b (split) →
M3 (filter_selected_masks, curation) → M4 (save_traces) → M5 analysis.
Config (DATE/MOUSE/RUN) is hardcoded at the top of each script.

## Workflow driver (added 2026-09-25, code/Workflow/) — PREFER THIS over hand-editing headers
- `mavca_status.py` — disk-driven stage table for every run under scape-data/; `--next`
  prints the one run to act on + exact command; `--next --run-it` executes it.
- `mavca_gui.py` — Qt button panel over it (`--selftest` = headless check).
- `run_stage.py SCRIPT --date D --mouse M --run R [--set NAME=VAL] [--raw-prefix runA|runB]`
  rewrites a script's header constants IN MEMORY and runs the unmodified file. Generalises
  run_m4_batch.py to every M-script. `--show` = dry run.
- Per-run params (5 vs 6 Hz, 12 vs 14 s skip, MASK_SOURCE_RUN, runA_/runB_) are in
  `SESSION_PARAMS` / `MASK_SOURCE` in mavca_status.py, cited to this file. A session not
  listed there is shown in orange and REFUSED by the panel. Add new sessions there first.
- `STALE-OWN-MASKS` flag: 0416 run2/run3 still hold 80 local curated masks but their traces
  use run1's 54 (post 2026-06-04 fix). Those local folders are dead weight.

## Key conventions & decisions (agreed during curation)
- **F0 baseline = 10th percentile** everywhere (M4 and M4.5). Not 20th.
- **Background shell = `dilation(ball(3)) & ~dilation(ball(2))`** (1-voxel gap) in both
  M4 and M4.5. Avoids PSF contamination from the dendrite itself.
- **Accelerometer y-limit = 0.25** in ALL plotting scripts (behavior_plots,
  behavior_plots_concat, combo_with_behavior, all_spikes_plot, accel_trace_movie).
  Use `accel_mag` column (raw), NOT `aligned_time_s` (that's the time axis).
- **Pupil/whisker timing**: must apply Basler→SCAPE offset (from trigger CSV
  `baslerExposureTrigger` vs `AndorXylaTrigger` rising edges) before cropping.
  Accelerometer uses `aligned_time_s` (already SCAPE-aligned).
- **M1 z-score** uses local rolling median + local MAD (not global median).
- **Mask cleanup**: overlapping masks that never fire at different times are merged
  via `merge_masks.py` (keeps first dend name, renumbers, logs to merge_log.csv).
  Deletions/merges logged in `labelmaps_curated_dynamic/{curation_log,merge_log}.csv`.
- Newer datasets use raw prefix `runA_` (older used `runB_`).

## Per-dataset SKIP_FIRST_SECONDS
- 2026-05-08 (rbp4_139_phpeb): **14 s**
- 2026-05-12 (rbp4_132_phpeb): **12 s**
- 2026-04-16 (rbp4_132_phpeb): 12 s
- Verify per run: raw_T − preprocessed_T frames ÷ 5 Hz.

## Datasets with shared masks / same FOV (verified by hashing curated mask sets 2026-06)
Byte-identical curated masks (same FOV + same curation; dend IDs match physically):
- 2026-04-16 rbp4_132: **run2/run3** (83 masks); **run5/run6/run7** (133).
  ⚠️ run1 (80) is the SAME FOV as run2/run3 but masks DIFFER (83→80 merge applied only to
  run1). Matched-ID pair is run2/run3; run1 stands alone.
- 2026-05-08 rbp4_139: **run5/run6** (62).
- 2026-05-12 rbp4_132: **run5/run6** (68). run9 (131) is its own curation.
- 2026-03-31 rbp4_132: masks stored in run7 (30) and run8 (51); run6 shares run7 FOV,
  run9/run10 share run8 FOV (traces extracted via MASK_SOURCE_RUN, no own mask folder).
- 2026-03-20 rbp4cre_139: run1 (43) and run3 (101) are distinct FOVs.
Cross-run analyses with matching dend IDs are only valid within an identical-mask group.

## Key scientific findings (this project)
- Calcium events are sparse, local, largely independent (mean pairwise r ≈ 0.01–0.07);
  no global co-activation. Supports dendrites-as-independent-units (NMDA spike) hypothesis.
- Reproducible propagation pairs across runs with shared masks (run2↔run3 normalized
  pair-strength Spearman ρ ≈ 0.5). Hub/initiator/follower functional clusters via PCA+KMeans.
- Propagation speed ~200–600 µm/s (consistent with IP3 calcium waves, not electrical).
- Calcium imaging only sees suprathreshold/coupled events; misses voltage-only events.

## Mask shape/amplitude/propagation analysis (added 2026-06, code/Extra/mask_propagation_analysis.py)
- Per-mask morphology+activity+within-mask propagation. Uses preprocessed/
  stack_voxel_norm_mean_sub.tif (mask-aligned, memmap) for sub-mask timing; aligns
  taller masks to stack via M4 convention (crop bottom/deep Y rows).
- depth vs peak ΔF/F: r=-0.48 in 132/0416/run1 (p=6e-6), holds controlling for
  volume+length (partial r=-0.46). Superficial dendrites = larger transients. In
  139/0508/run5 only depth vs event-rate replicates (r=-0.28; both mice ≈-0.25).
- Within-mask propagation mostly UNRESOLVABLE at 5 Hz (sub-frame); among masks with a
  consistent gradient, biased superficial→deep (~24:7 pooled). No depth/orient speed dep.
- ⚠️ 132/0416/run1 inter-mask trace r≈0.90 (0.79 detrended) — contradicts the r≈0.01–0.07
  independent-units finding; 139/run5 is r≈0.07 (matches). run1 may be a globally co-active
  state OR a neuropil/bg-subtraction issue — confirm on run2/run3 before trusting run1.

## Grouped (shared-mask) re-analysis (2026-06, GROUPS in mask_propagation_analysis.py)
Pooled events across runs sharing masks (run1/2/3 now all 80 & identical; etc.). Revises above:
- Trace co-activation is FOV-SPECIFIC, not run1-specific: 0416 run1/2/3 ALL ≈0.85–0.90;
  every other FOV (0416 run5/6/7=0.01, 0331=0.04–0.08, 0508=0.07, 0512=0.02) ≈independent.
  So 0416_run123 is an outlier FOV (all 3 runs co-active) — inspect it (shallow/tuft? neuropil?
  motion?), don't trust it for independence claims. run5/6/7 same mouse+day is independent.
- depth→amplitude is FOV-DEPENDENT, NOT universal: strong only in 0416_run123 (r=-0.48) and
  0331_run67 (depth-rate r=-0.64); null in the other 4 FOVs; pooled across FOVs it vanishes
  (r≈0). Pooling depth across FOVs is confounded (different depth ranges) — keep within-FOV.
- Within-mask propagation: pooling cut n/a; ~55% uniform (sub-frame at 5Hz). Among directional,
  only a mild superficial→deep lean pooled (58 deep:41 surface); the strong deep bias seen
  earlier was mostly 0416_run123. Per-FOV outputs: <mask_run>/mask_analysis/group_*_metrics.csv;
  pooled scape-data/mask_metrics_grouped.csv.
- Data note: 0331 run6 trace has 33 dends ≠ run7's 30 masks (extracted with a different mask
  set) → run6 excluded from 0331_run67 pool. run5/6 (0508,0512) & run9/10,run5/6/7 had only one
  trace CSV available, so those "groups" effectively single-run.

## Event-typing + behavior (2026-06, code/Extra/event_typing_behavior.py)
Per-event coincidence (#masks co-active ±1fr) -> whole-tree/global vs local; global Ca vs
accel(accel_mag/aligned_time_s)+pupil(10Hz,Basler->SCAPE offset). Findings:
- TWO event regimes, FOV-specific: most FOVs (0416 run7, 0331 run8, 0512 run9) are ALL local/
  isolated (frac_global=0, mean_r≈0.01–0.08) = independent NMDA-spike-like. 0508 run5 is MIXED
  (frac_global≈0.6). 0416 run1 is ~all global (frac_global≈1.0, mean_r=0.90).
- Behavior distinguishes them: 0508 run5 global Ca vs accel r=+0.43, pupil +0.27, and accel is
  higher during global than local/quiet frames -> whole-tree events are arousal/movement-coupled
  (bAP/arousal-like). 0331 run8 local activity ANTI-correlates with pupil (r=-0.39) -> local
  events favor quiet state (matches "fire when not moving").
- ⚠️ 0416 run1 is near-totally global BUT animal was essentially still (accel≈0.006 throughout,
  movement coupling weak) -> global co-activation without movement = red flag for shared-signal
  contamination (neuropil/hemodynamic/bg-sub), not genuine bAP storms. Inspect run1 extraction/raw.
- Superficial dendrites more active: event-rate vs depth negative in ALL 5 runs (r≈-0.1 to -0.3).
- Branchiness (skeleton tortuosity) vs activity ≈0 (no relation). Hub-degree vs depth weak/mixed.
TODO next: top-down 4–5 depth-chunk intensity profiles (needs 4D stack) + per-mask freq/PSD.

## Depth-chunk + neuropil test (2026-06, code/Extra/depth_chunk_neuropil.py)
- Top-down: per-mask 5 cortical-depth chunks, event-triggered chunk×time. Both 0416/run1 and
  0508/run5 are SYNCHRONOUS across depth (slope ≈ +0.02–0.04 fr/chunk, peaks within ~0.15 fr)
  -> no resolvable top-down at 5 Hz (tiny superficial-lead in run5 only).
- ⚠️ CONTAMINATION test caveat: stack_voxel_norm_mean_sub has per-frame spatial mean removed →
  forces in-mask vs out-of-mask r≈-0.97 (artifact; do NOT use it for neuropil tests).
  On RAW (per-region ΔF/F): in-mask vs out-of-mask r≈0.99–1.00 in BOTH run1 and run5 → a strong
  field-wide signal (hemodynamic/global) exists everywhere, NOT unique to run1.
- Refined run1 interpretation: the discriminator is POST core-shell subtraction — run5 removes
  the field signal (residual inter-mask r=0.07) but run1 does NOT (r=0.90). So run1's whole-tree
  co-activation = field signal that survived local shell subtraction → run1-specific extraction/
  FOV issue (likely dense mask packing → shell contamination, or spatially-structured global
  signal), not confirmed biology; can't fully exclude real global co-activation either.
  Next discriminating step: replicate core-shell on raw for run1, corr residual-global vs neuropil;
  compare mask packing density run1 vs run5.

## Common tasks / prompts the user asks
- "Remove dend_XXX and log it" → delete mask, renumber sequentially, append to curation_log.csv.
- "Merge these masks" → use merge_masks.py (set MERGE_GROUPS), keep first name, log it.
- "Find duplicates in labelmaps_split" → pairwise Dice > 0.3 check.
- "Regenerate plots for runs X" → re-run with correct per-run SKIP_FIRST_SECONDS.
- "Check pipeline for correctness" → audit math, data flow, file paths, hardcoded values.
- Always renumber masks dend_000.. sequentially after any delete/merge.
- Keep per-dendrite M4 preview as ONE figure (MIP left, trace right).

## M3 curation speed-ups (added 2026-06)
- ~48% of split masks are a real body + stray blob joined by a thin neck (1-voxel
  erosion separates them). Do NOT auto-cut necks offline — apical dendrites are only
  2–4 voxels wide and would be amputated. Keep the human in the loop.
- M3 hotkeys for fast erasing: `e` = erase the connected component the paint dot
  touches (dot blob + e); `c` = keep only largest component (after a 1-slice cut
  stroke + `x`, press `c`). Both autosave + refresh like other edits.
- Lasso (added 2026-06): `l` = keep inside drawn polygon(s) / delete outside (applied
  across all Z); `p` = delete inside; `t` = toggle 2D/3D (napari only draws polygons in
  2D). Polygon vertices rasterized via skimage polygon2mask on (Y,X), broadcast over Z.

## Topology + mixed-effects (2026-06, code/Extra/dendrite_topology.py; statsmodels installed)
- Topology = cKDTree spatial adjacency (ADJ_UM=4) -> branch families; root=deepest mask;
  branch_order=BFS hops; dist_trunk=um to root. Pooled scape-data/topology_pooled.csv (466 masks,
  6 FOVs, 69 families; many singletons). RECONSTRUCTION IS FRAGILE/threshold-sensitive.
- Per-FOV rate~branch_order INCONSISTENT: +0.57(0331_r67,n=16), -0.30(0512_r56), null rest.
  Earlier 0508 +0.40 was dilation-threshold-dependent (KDTree gives +0.18 n.s.). Not robust.
- MixedLM pooled (family RE, within-FOV z): DEPTH coef=-0.15 p=0.002 (superficial fires more,
  ROBUST); branch_order p=0.14, dist_trunk p=0.18 -> NOT predictive beyond depth.
  => Publishable morphology->activity result = depth only; branch-order claims need better recon.
- Behavior caveat (user): Ca vs whisker/accel/pupil differ in frequency + can be event-locked with
  opposite sign (0512: Ca sharp-rise -> eye movement -> pupil DROP). Use event/peak-locked coupling
  (behavior-triggered avgs), NOT full-trace Pearson.
- venv note: use `.venv311/bin/python -m pip` (the `.venv311/bin/pip` shim points at python3.14).

## Spatial/morphological organization of activity & movement coupling (2026-06)
Master table: scape-data/master_mask_table.csv (466 masks, all FOVs; 0508 has residual movement
coupling_z + sig). Scripts: build_table.py, run_models.py, fig3_organization.py. Figures: fig1_*,
fig1b_*, fig1_controls, fig2_morphology, fig3_organization (scape-data/figures/).
ROBUST: (Q1/Q2) residual movement coupling NOT depth-organized (LMM depth p=0.41; survives
event_rate+SNR+volume, intercept +2.84 p=1e-12). (Q4) coupled branches NOT spatially or family
clustered (perm p=0.93, 0.60) -> spatially distributed. (Q5, all 6 FOVs, family RE) superficial>deep
gradient carried by LOCAL events (rate_local~depth coef=-0.14 p=0.003); subtree (p=0.10) & global
(p=0.22) n.s.; overall event_rate~depth -0.15 p=0.002.
EXPLORATORY: (Q3) coupled branches no sig event-class enrichment (frac_subtree trend p=0.10).
CAVEAT: 'subtree' class depends on fragile family reconstruction; weight local/global (family-
independent) over subtree. Movement coupling 0508-specific (do not generalize).

## SCAPE scan-order (2026-06): Z=galvo sweep axis (30 steps, sequential, ~6.7ms/plane); depth(Y)+X
captured simultaneously per Z snapshot. => TOP-DOWN/depth latency is SCAN-CLEAN (analysis #7 viable
along depth); Z-direction recruitment is scan-confounded (subtract z_centroid*6.7ms). Within-mask
latency still sub-frame/dead at 5Hz. AndorXylaTrigger=sparse start trigger (not per-frame).

## #7 recruitment order: 0508-only, NOT replicated (0331_r67 synchronous p=0.71; 0331_r8910 & 0512_r56 no global events). Top-down +1.23ms/um seen ONLY in 0508 extended global epochs. Do not generalize. recruitment_order.py.
(top-down) recruitment, +1.23 ms/um, 8/9 events (sign-test p=0.039; pooled perm p=0.002). Slow
(~1 frame over depth), recruitment-order NOT propagation. PROMISING but n=9 events, 1 FOV -> needs
replication. Depth axis is scan-clean (see scan-order note); Z-offset subtracted. recruitment_order.py.

## Acquisition frame rate (IMPORTANT — varies by session)
- rbp4_132 / rbp4_139 single-channel 30-plane sessions: 5 Hz.
- rbp4cre_136 (2026-02-09) & rbp4cre_138 (2026-02-17) dual-channel deeper (Z=69/55) sessions: **6 Hz**.
behavior_plots.py FRAME_RATE is hardcoded 5 -> set to 6 for 136/138 runs (feeds Ca time axis, skip
frames, end-crop; wrong rate stretches Ca time and over-crops pupil/whisker).
Analysis scripts used FS=5 for all -> for 136/138 the FRAME-RATE-AGNOSTIC metrics (mean r, eff_dim/N,
participation, event-class fractions, independence/Fig1) are CORRECT; TIME-based metrics (event
rate/min, IEI, durations s) for 136/138 are off by 5/6 and need recompute at 6 Hz if reported.

## ============ SESSION STATE — RESUME HERE (2026-06-04) ============
### BIG UPDATE: 0416 run1/2/3 "contamination" RESOLVED — it was missing background subtraction.
- Diagnosis: current traces were core-only (NO core-shell bg subtraction). core(r1) no-shell mean_r=0.91
  (rank-1); core(r1)-SHELL mean_r=0.06 (independent). Aggressive core alone does NOT fix (0.82). v1
  curation was already clean. => NOT a bad field; a trace-extraction artifact.
- Fix applied to 0416 run1: (a) cropped 19 mixed-Y masks 199->196 (M1 Y_CROP=3 = bottom crop;
  backup labelmaps_curated_dynamic_precrop_backup). (b) removed dend_006/036/048. (c) merged the 11
  find_dendrite_branches groups (dendrite_summary.csv) -> 80 masks -> 54 masks; renumbered;
  backup labelmaps_curated_dynamic_MERGEBACKUP_20260604_143020; logged merge_log.csv + curation_log.csv.
- Re-ran M4 (core-shell) for run1 (own masks) and run2/run3 (MASK_SOURCE_RUN=run1) -> all 54 dends,
  600fr, mean_r 0.03-0.07, eff_dim/N 0.22-0.35. 0416 run1/2/3 is NOW A CLEAN FOV (54 masks, shared).
- run_m4_batch.py now takes optional rawname (6th arg) + mask_run (7th arg; '-' = none).

### CONSEQUENCE: docs/figures predate this fix and are STALE re 0416 run1/2/3:
- docs/results_revised.md, experiment_log.md, paper_structure.md, paper_figures/Figure1_independence,
  fig6_independence all still treat 0416 run1/2/3 as the "contaminated negative control". NEEDS UPDATE:
  it is now a clean, mixed-regime FOV. Either find a different neg-control framing (no-shell illustration)
  or drop it. Consider re-adding 0416 run1/2/3 (and run5/6/7) to pooled independence analyses.

### 5-FOV PATTERN ANALYSIS (0416/0508/0512 only; fov5_patterns.py, fov5_patterns.csv):
5 FOVs (4 mouse-132 + 1 mouse-139=0508), all 5Hz+behavior. Patterns (descriptive, n=5):
- eff_dim/N anti-correlates with frac_global (r=-0.86): global-rich FOVs have lower dimensionality.
- Global regime FOV-specific: 0416_r123 & 0508 ~40% global; 0416_r567/0512_r56/0512_r910 = 0%.
- DISSOCIATION: global events != movement-coupled. 0508 global mode movement-coupled (+0.41z @ onset);
  0416_r123 equally global but NOT movement-coupled (-0.10) -> >=2 flavors (movement-driven vs spontaneous).
- 4/5 FOVs show movement-onset global Ca DIP (only 0508 positive). depth~rate 4/5 negative, 1 reversed.
- CONFOUND: 0508 is the only mouse-139 FOV AND only movement-coupled -> can't separate FOV vs mouse.

### OPEN THREADS / NEXT:
- Update stale docs/figures for the 0416 fix (above).
- Optional: characterize 0416_r123 (non-movement global) vs 0508 (movement global) events.
- 136/138 dual-channel = 6Hz (not 5); ACh=red channel (future); behavior_plots FRAME_RATE per run.
- 136/0209 & 138/0217 have pupil/whisker .mat but NO accel CSV (need mat_to_csv on Run00X_t1.mat).
- behavior_plots.py APPLY_PUPIL_TRIGGER_OFFSET flag (currently False); end-crop tied to Ca.

### KEY DOCS: docs/{analysis_plan,results_revised,experiment_log,paper_structure}.md;
### paper_figures/Figure1-4; scape-data/{rbp4_data_inventory,fov5_patterns,master_mask_table,*}.csv
