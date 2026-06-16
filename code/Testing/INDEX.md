# code/Testing/ — INDEX & steering

Sandbox for the **NMDA-spike vs bAP "vertical-streak"** assumption testing (2026-06). Isolated from the
M1–M9 pipeline. Reads project data from `scape-data/<date>/<mouse>/<run>/`; writes only to this folder.
- **What/why + findings:** `RESULTS.md`   ·   **how to run:** `README.md`   ·   **master log:** `docs/experiment_log.md` §8
- **Verdict:** amplitude is NOT a robust NMDA-vs-bAP discriminator at 5 Hz (tested 4 ways). The trunk/branch
  classifier is a *structural* tool only. Behavior coupling (0508) is the real handle.

## Layout
```
code/Testing/
  README.md        usage guide
  RESULTS.md       full results (test-by-test stats + status)
  INDEX.md         this file
  *.py             20 analysis scripts
  figures/         all output PNGs (39)
  data/            intermediate CSV tables (8)
```

## Run order (dependencies)
1. `nmda_bap_amplitude.py`  → writes `nmda_bap_mask_metrics.csv`, `nmda_bap_event_table.csv`
2. `nmda_bap_split_v2.py`   → writes `nmda_bap_split_v2_events.csv`, `_segments.csv`  (streams raw stacks, ~6 min)
   - The `diag_*` scripts and `nmda_bap_split_maps.py` READ the CSVs from 1–2, so run those first.
3. classifier/galleries/maps (`nmda_bap_gallery2.py`, `nmda_wholemask_run7.py`, …) — standalone, read masks/raw directly.

`nmda_bap_split_v2.py` is also the **shared helper module** (VOXEL, raw_path, fit_surface, corrected_depth,
thresh, event_peaks, gidx, FOVS) imported by most others; `nmda_bap_gallery2.py` exports `decompose()`
(the classifier) used by the gallery/map/whole-mask scripts.

## Script → role → outputs
### A. The amplitude tests (the hypothesis)
| script | role | figure | data (W=write R=read) | finding |
|---|---|---|---|---|
| `nmda_bap_amplitude.py` | Test 1 (amp vs participation) + Test 3 (mask morphology vs amp), all FOVs | `nmda_bap_amplitude.png` | W: `nmda_bap_mask_metrics.csv`, `nmda_bap_event_table.csv` | Test3 NULL; Test1 EXPLORATORY (FOV-specific) |
| `nmda_bap_split.py` | Test 2 v1: naïve top/bottom (depth) half-split | `nmda_bap_split.png` | W: `nmda_bap_split_events.csv` | strong but = ARTIFACT (see v2) |
| `nmda_bap_split_v2.py` | Test 2 v2: branch/bend split, SNR-robust classify (+ shared helpers) | `nmda_bap_split_v2.png` | W: `nmda_bap_split_v2_events.csv`, `_segments.csv` | NULL (effect vanishes) |
| `nmda_bap_features.py` | Test 1 within-mask coupling control + kinetics (FWHM/decay/area) | `nmda_bap_features.png` | W: `nmda_bap_features.csv` | kinetics NULL; coupling modest |
| `nmda_bap_morpho_classify.py` | per-event active-region morphology (streak vs tilted) + amp | `nmda_bap_morpho_classify.png` | W: `nmda_bap_morpho_events.csv` | morphology splits classes; amp NULL |
| `nmda_wholemask_run7.py` | per-FOV trunk/branch segment map + per-segment amplitude | `nmda_wholemask_<fov>.png` | (uses gallery2 `decompose`) | trunk>branch only 0508/0320r3 |
| `nmda_bap_wholemask_amp.py` | whole-mask Ca(vertical+deep) vs NMDA(superficial/tilted) + amp | `nmda_wholemask_0416r567.png` | — | NULL across FOVs |

### B. Morphology classifier & galleries/maps  (structural; recommended params in gallery2)
| script | role | figure |
|---|---|---|
| `nmda_bap_gallery2.py` | **current** trunk/branch/combined classifier (`decompose()`) + per-mask gallery | `nmda_gallery2_<fov>.png` |
| `nmda_bap_cluster.py` | earlier hierarchical cluster decomposition (superseded) | `nmda_cluster_<fov>.png` |
| `nmda_bap_gallery.py` | earlier per-mask gallery (uses cluster.py, superseded) | `nmda_gallery_<fov>.png` |
| `nmda_bap_split_maps.py` | per-FOV map: segments + per-mask event-class colour | `nmda_bap_map_<fov>.png` (R: `nmda_bap_split_v2_events.csv`) |

Recommended classifier params (top of `nmda_bap_gallery2.py`): `LIN_TRUNK=0.80, BRANCH_ANGLE=45°,
BRANCH_MIN_FRAC=0.10, TRUNK_FRAC=0.30, RDP_EPS=8, TRUNK_MIN=35µm`. Optional `DEPTH_TRUNK_Y` forces deep voxels to trunk.

### C. Geometry & diagnostics  (print/figure only — explain artifacts)
| script | role | output |
|---|---|---|
| `fov_geometry_diagnostics.py` | illumination profile (X/Z) + sloped pial-surface plane fit | `fov_geometry.png` |
| `diag_orientation.py` | dendrites align to image-Y vs surface-normal? → don't slope-correct orientation | stdout |
| `diag_silent_masks.py` | cross-check "silent" masks vs pipeline trace | stdout (R: split_v2 + mask_metrics csvs) |
| `diag_silent_extract.py` | core-shell cancellation test (silent = shell≈core) | stdout (R: split_v2 csvs) |
| `diag_nmda_label.py` | why deep/straight masks mislabeled (segment-SNR) | stdout (R: split_v2 + mask_metrics) |
| `diag_nmda_examples.py` | example segment traces of flagged masks | `nmda_examples_<fov>.png` |
| `diag_relfrac_sweep.py` | sensitivity of Test-2 effect to localization threshold → non-robust | stdout (R: split_v2_events) |

### D. Manual annotation (optional ground truth)
| script | role | output |
|---|---|---|
| `manual_trunk_branch.py` | Napari manual trunk/branch labelling (3D view + 2D MIP marking) | `scape-data/<…>/<run>/manual_trunk_branch/*_tb.tif` + `manual_labels.csv` |
| `compare_manual_auto.py` | score auto-classifier vs manual labels; parameter sweep | stdout (R: manual labels) |

## figures/ by producer (prefix → script)
`nmda_bap_amplitude.png`→amplitude · `nmda_bap_split.png`→split · `nmda_bap_split_v2.png`→split_v2 ·
`fov_geometry.png`→geometry · `nmda_bap_features.png`→features · `nmda_bap_morpho_classify.png`→morpho ·
`nmda_bap_map_*`→split_maps · `nmda_gallery2_*`→gallery2 · `nmda_gallery_*`/`nmda_cluster_*`→gallery/cluster ·
`nmda_examples_*`→diag_nmda_examples · `nmda_wholemask_*`→wholemask_run7 (+ `_0416r567`→wholemask_amp,
`_0508_run5_depth*`→depth-rule variants).
