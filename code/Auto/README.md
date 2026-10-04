# SCAPE automatic dendrite pipeline (`code/Auto/`)

Finds the apical dendrites in every SCAPE run without hand curation, extracts their
calcium traces, compares them with the human-curated masks, makes a behaviour combo
figure and a reference + dynamic volume movie per run, and runs cohort statistics.
The SCAPE counterpart of `femtonics-data/code/STEP7_workflow` + `STEP9_auto`.

`scape-data/` is only read. Everything is written to `scape-auto/<DATE>/<MOUSE>/<RUN>/`
(gitignored). The M1-M5 scripts and `code/Workflow/` are unchanged.

## Start

```bash
code/scape              # control panel (also: double-click "Scape Panel" on the Desktop)
code/scape status       # table of runs, stages and headline numbers
code/scape run --all    # process everything that is missing or out of date, then statistics
code/scape run 2026-05-08/rbp4_139_phpeb/run5 --force   # redo one run
code/scape review 2026-05-08/rbp4_139_phpeb/run5        # inspect / fix masks in napari
code/scape stats        # cohort statistics only
```

A stage is redone only when its inputs are newer than its outputs, so `scape run --all` is
cheap when nothing changed. Full run from scratch: ~55 min for 20 runs on this Mac (2 jobs).

## Stages (per run)

| stage | script | output in `scape-auto/<DATE>/<MOUSE>/<RUN>/` |
|---|---|---|
| detect | `auto_detect.py` | `masks/auto_labelmap.tif` (+ `_perrun.tif`), `masks/auto_masks.csv`, `reference/` |
| traces | `extract_traces.py` | `traces/dff_auto.csv`, `traces/dff_human_sameextractor.csv` |
| validate | `validate_vs_human.py` | `validation/validation.json`, `overlay.png`, `beyond_human.png` |
| global | `global_ca.py` | `traces/global_ca.csv` |
| combo | `combo_plot.py` | `figures/combo_auto.png/.pdf` (+ `combo_human` with `--source human`) |
| movie | `make_movie.py` | `movies/<run>_dual_behavior.mp4` |
| stats | `auto_stats.py` | `scape-auto/stats/{auto,human}/`, `scape-auto/stats/compare_auto_vs_human.*` |
| coherence | `coherence_behavior.py` | `figures/coherence_<source>.png` per run; `scape-auto/stats/coherence/` |

`run_auto.py` chains them; `auto_status.py` / `auto_gui.py` show where every run is.
`scape_common.py` holds the shared paths, per-run parameters (frame rate, skip from
`code/Workflow/mavca_status.py`), FOV grouping and the behaviour loader.

## How dendrites are found (auto_detect.py)

1. Activity volume: per-voxel F0 = 10th percentile after the skip, dF/F, spatial high-pass
   (removes the field-wide glow), robust z-score.
2. Seeded region growing: start at the most active voxel, add neighbouring voxels whose trace
   correlates with the unit's reference trace (r > 0.45), repeat. Touching units with
   r > 0.85 are merged; units < 400 voxels, < 20 um deep, or flat horizontal sheets are dropped.
3. Runs that image the same field of view are detected one by one and then share the union of
   their dendrites, so a dendrite silent in one run is still measured there. Same-FOV runs are
   found from the images themselves (`scape_common.session_fov_groups`, r >= 0.75), not only
   from the MASK_SOURCE table.

Parameters were tuned on three runs (`TRAIN_RUNS` in `auto_detect.py`); validation reports
those separately from the 16 held-out runs.

## Reviewing masks (napari)

`code/scape review KEY`: click a dendrite to plot its trace and the trace of the human mask it
overlaps most; Shift+click adds to the selection; `D` deletes, `M` merges the selection, `S`
saves, `R` resets. Saving writes `masks/auto_labelmap_reviewed.tif` (older versions go to
`masks/old/`); every later stage uses the reviewed file when it exists. Then
`code/scape run KEY` rebuilds traces, figures and movie.

## What the validation numbers mean

| number | meaning |
|---|---|
| human_recall | fraction of human masks overlapped by an auto unit (overlap >= 0.2 of the smaller) whose dF/F trace correlates r >= 0.7 |
| recall_r05 / r08 | the same at r >= 0.5 / 0.8 |
| median_best_r | per human mask, the best trace r among overlapping auto units (median) |
| signal_recall_R07 | human masks whose trace is reproduced (multiple R >= 0.7) by the auto units lying inside them (human masks are often wider than one dendrite) |
| spatial iou_* | one-to-one Hungarian matching; TP needs IoU >= threshold |
| reliability | split-half r: top vs bottom half of a unit, each background-subtracted; `null_shifted` = the same shapes moved 60 voxels sideways (noise level) |

Human masks are not a complete ground truth: curators kept a subset of the active dendrites,
so auto units without a human match are not automatically false positives. Look at
`beyond_human.png` and the reliability numbers to judge them.

## Ca vs behavior: correlation and coherence

Pearson r is a poor measure for pupil: Ca transients are sharp and the pupil response is slow and
smoothed, so a tight relation still gives a small r whose sign depends on slow drift.
`coherence_behavior.py` measures magnitude-squared coherence (Welch, 25.6 s segments), which does not
depend on waveform shape, against a circular-shift null in three bands (0.04-0.2, 0.2-0.5, 0.5-1 Hz).
It also reports the lag from the cross-spectrum phase and a Ca-event-triggered pupil average.
