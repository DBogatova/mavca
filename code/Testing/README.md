# NMDA-spike vs bAP "vertical-streak" — assumption testing (sandbox)

**Full results write-up: [`RESULTS.md`](RESULTS.md)** (test-by-test statistics + status). This README is the
code/usage guide. Registry entry also in `docs/experiment_log.md` §8.

**Self-contained exploratory analysis. Not part of the M1–M9 pipeline.** All scripts, figures,
and intermediate CSVs live in this folder; they read project data from `scape-data/<date>/<mouse>/<run>/`
(masks, traces, raw stacks) but write **only** to `code/Testing/figures/` and `code/Testing/data/`.

## Hypothesis tested
> NMDA-spike Ca²⁺ transients (localized to a branch) have **lower amplitude** than bAP
> "vertical streak" transients (signal extended along the apical trunk / cortical depth).

We operationalized "vertical streak / bAP" as Ca signal that is **spatially extended** (along depth,
whole-tree, or trunk) and "NMDA" as **localized** (single branch), then asked whether amplitude separates them.

## What we found (bottom line)
**Amplitude is not a robust discriminator of NMDA-like vs bAP-like activity in this 5 Hz SCAPE / GCaMP data.**
The hypothesis was tested four independent ways; it only appears under specific, non-robust conditions:

| Test | Scale | Result |
|---|---|---|
| **Test 3** – static mask morphology → amplitude | per mask | **Null** (peak ΔF/F vs length/verticality ≈ 0) |
| **Test 1** – co-activation extent → amplitude | per event, # co-active **masks** | Isolated (NMDA-like) < whole-tree (bAP-like), pooled p≈3e-6, **FOV-specific**; within-mask paired only modest (7/10 masks, p≈0.02) |
| **Test 2** – within-dendrite branch split → amplitude | per event, **segments** | Strong with a noise-threshold classifier (1.6×, p≈2e-19) but this was a **detection artifact**; with an SNR-robust (relative-amplitude) classifier the effect **vanishes** (pooled p≈0.70) and a REL_FRAC sweep shows it is significant only at extreme cutoffs |
| **Active-region morphology** (vertical streak vs tilted) | per event | Morphology splits two real shape populations but **amplitude is flat** across them (pooled p≈0.35) |
| **Event kinetics** (FWHM/decay/area) | per event | Do **not** distinguish isolated vs whole-tree (FWHM 0.60 s both); NMDA plateau not resolvable at 5 Hz |

The only handle that carries real signal is **behavior/movement coupling** (0508/run5's whole-tree mode),
not single-event amplitude or kinetics. Where a small amplitude effect does appear (per-segment trunk>branch),
it is FOV-specific (0508/run5, 0320/run3; ~1.1–1.2×) and modest.

## Key methodological findings (reusable)
- **Illumination gradient** (dim left → bright right, ~1.1–1.23×) is removed by per-voxel ΔF/F; it is *not*
  a confound — trunk/branch event classes are matched on X-centroid and baseline brightness.
- **Pial surface is tilted** (X-tilt up to ~12°; surface drops ~55 µm in Y across the FOV). → **Depth must be
  slope-corrected**; **orientation must NOT** (dendrites align about equally to image-Y vs the surface normal:
  the tilt is small vs the ~25–34° spread of dendrite orientations). See `diag_orientation.py`, `fov_geometry_diagnostics.py`.
- **"Silent" masks are an artifact**: every mask flagged silent had an active core but a shell that brightened
  identically (core↔shell r≈0.96–1.0), so core-shell subtraction cancels it. They are background/field-shared
  signals, not inactive dendrites. See `diag_silent_*.py`.
- **Within-segment SNR coupling**: splitting a dendrite into small segments makes per-segment detection threshold-
  dependent; bigger events cross threshold in more segments → look "extended". This inflated the Test-2 effect.

## Morphological trunk / branch / combined classifier
`nmda_bap_gallery2.py` (`decompose()`) is the structural classifier developed iteratively from user feedback.
It is **morphology only** (mask shape, no activity) — a structural labelling tool, not a functional NMDA/bAP call.

Logic: skeletonize → longest-path backbone → split at bends (RDP); the longest straight run (+ collinear runs)
is the **trunk**, divergent runs and off-backbone parts are candidate **branches**. A post-process keeps a branch
only if it (a) diverges from the trunk by ≥ `BRANCH_ANGLE` and (b) is ≥ `BRANCH_MIN_FRAC` of the mask; otherwise it
is folded into the trunk. If, after that, the (merged) trunk is < `TRUNK_FRAC` of voxels or not straight
(< `LIN_TRUNK`), the whole mask is a **branch**. Whole-mask class: `trunk` (only trunk), `branch` (no trunk),
`combined` (trunk + ≥1 real branch).

**Recommended parameters** (tuned against user-labelled cases; `compare_manual_auto.py`-style validation):
`LIN_TRUNK=0.80, BRANCH_ANGLE=45°, BRANCH_MIN_FRAC=0.10, TRUNK_FRAC=0.30, RDP_EPS=8, TRUNK_MIN=35µm`.
This matches ~5/8 hand-labelled cases. The residual misses are **fundamental**: trunk-vs-fork is a *topological*
distinction (number of branches), which PCA/elongation metrics cannot see — e.g. a straight trunk and a fork can
have near-identical linearity/aspect. A topological (skeleton endpoint-count) discriminator would be the next step.
Optional `DEPTH_TRUNK_Y` forces voxels deeper than a given Y to trunk (e.g. 0508 tried Y=110).

## Scripts
| script | purpose |
|---|---|
| `nmda_bap_amplitude.py` | Test 1 (event amplitude vs participation) + Test 3 (mask morphology vs amplitude), all FOVs |
| `nmda_bap_split.py` | Test 2 v1: naïve top/bottom (depth) half-split, per-half ΔF/F, amplitude |
| `nmda_bap_split_v2.py` | Test 2 v2: branch/bend segmentation; SNR-robust event classification; **also the shared helper module** (VOXEL, raw_path, fit_surface, corrected_depth, thresh, event_peaks, gidx, FOVS) imported by the others |
| `fov_geometry_diagnostics.py` | Illumination profile (X/Z) + sloped pial-surface plane fit |
| `diag_orientation.py` | Are dendrites aligned to image-Y or the surface normal? (→ don't slope-correct orientation) |
| `diag_silent_masks.py`, `diag_silent_extract.py` | Why some masks read "silent" (core-shell cancellation test) |
| `diag_nmda_label.py`, `diag_nmda_examples.py` | Why deep/straight masks were mis-labelled NMDA (segment-SNR artifact) |
| `diag_relfrac_sweep.py` | Sensitivity of the Test-2 effect to the localization threshold (shows non-robustness) |
| `nmda_bap_features.py` | Coupling control for Test 1 (within-mask) + kinetics (FWHM/decay/area) discriminators |
| `nmda_bap_morpho_classify.py` | Per-event active-region morphology (vertical streak vs tilted) + amplitude |
| `nmda_bap_cluster.py`, `nmda_bap_gallery.py` | Earlier hierarchical cluster decomposition + per-mask gallery (superseded by gallery2) |
| `nmda_bap_gallery2.py` | **Current** straightness-based trunk/branch/combined classifier + per-mask gallery (per FOV) |
| `nmda_bap_split_maps.py` | Per-FOV maps: mask segments + per-mask event-class colouring |
| `nmda_bap_wholemask_amp.py` | Whole-mask Ca(vertical+deep) vs NMDA(superficial/tilted) classification + amplitude |
| `nmda_wholemask_run7.py` | Per-FOV map drawn by trunk/branch segments + per-segment amplitude (trunk vs branch), uses gallery2's `decompose()` |
| `manual_trunk_branch.py` | Napari manual trunk/branch annotation (ground truth). 3D view + 2D MIP marking, broadcast over Z |
| `compare_manual_auto.py` | Score auto-classifier vs manual labels; sweep parameters to find the essential ones |

## Outputs
- **Figures** → `code/Testing/figures/` (e.g. `nmda_bap_amplitude.png`, `nmda_bap_split_v2.png`, `fov_geometry.png`,
  `nmda_gallery2_<fov>.png`, `nmda_wholemask_<fov>.png`, `nmda_bap_features.png`, `nmda_examples_<fov>.png`).
- **Tables** → `code/Testing/data/` (`nmda_bap_mask_metrics.csv`, `nmda_bap_event_table.csv`,
  `nmda_bap_split_v2_events.csv`/`_segments.csv`, `nmda_bap_features.csv`, `nmda_bap_morpho_events.csv`,
  `nmda_bap_cluster_table.csv`, `nmda_bap_split_events.csv`).
- Manual annotations (if used) → `scape-data/<date>/<mouse>/<run>/manual_trunk_branch/` (project-data location).

## How to run
```bash
source .venv311/bin/activate
python code/Testing/nmda_bap_amplitude.py        # Tests 1 & 3 (fast, all FOVs)
python code/Testing/nmda_bap_split_v2.py         # Test 2 (streams raw stacks; ~6 min)
python code/Testing/nmda_bap_gallery2.py         # trunk/branch/combined galleries per FOV
python code/Testing/nmda_wholemask_run7.py       # per-FOV segment map + amplitude
```
FOV lists and parameters are set at the top of each script. The classifier parameters live at the top of
`nmda_bap_gallery2.py` and are inherited by the gallery/map/whole-mask scripts.
