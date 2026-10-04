# MAVCA workflow driver — status, next-step, GUI

The apical-dendrites-2025 counterpart of `femtonics-data/code/STEP7_workflow/`.
Same idea: one command answers "where is every run and what do I do next",
every stage is detected from what is on disk, and a button panel runs the
next step for a selected run with the command shown verbatim in a log.

```bash
cd /Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025
PY=.venv311/bin/python

$PY code/Workflow/mavca_status.py            # table of all runs + processing_status.csv
$PY code/Workflow/mavca_status.py --next     # the one run to act on + exact command
$PY code/Workflow/mavca_status.py --next --run-it   # run it (non-GUI stages)
$PY code/Workflow/mavca_gui.py               # button panel
$PY code/Workflow/mavca_gui.py --selftest    # headless check, no window
```

## The three files

| file | role |
|------|------|
| `run_stage.py` | Runs ONE M-script for ONE run without editing it. Rewrites the header constants (`DATE`, `MOUSE`, `RUN`, `SKIP_FIRST_SECONDS`, `FRAME_RATE`, `MASK_SOURCE_RUN`, …) and the `runA_`/`runB_` literal in memory, then executes the unmodified file as `__main__`. `--show` prints the rewrites and runs nothing. |
| `mavca_status.py` | Discovers runs by walking `scape-data/<DATE>/<MOUSE>/<runN>/`, detects the stage from artifacts, attaches per-run params, builds the exact `run_stage.py` command for the next step. Writes `processing_status.csv` at the project root. |
| `mavca_gui.py` | Qt panel over `mavca_status`. Table + `Run next automatic step` (chains M1→…→M5→behaviour until napari or complete) + `Open M3 curation (napari)` (detached) + `Refresh`. |

## The stage ladder

| stage | artifact that proves it | next script |
|-------|------------------------|-------------|
| `no_raw` | — | fetch `raw/run[AB]_<run>_<mouse>-reslice-bin.tif` from SCC |
| `raw` | that file (or `preprocessed/raw_clean.tif`) | `find_events_m1.py` |
| `m1_events` | `preprocessed/active_frames.npy` + `stack_voxel_norm_mean_sub.tif` | `pre_segmentation_m1.5.py` |
| `m1_5_frames` | `preprocessed/best_frames/` non-empty | `auto_mask_m2.py` — or straight to M4 for shared-FOV runs |
| `m2_masks` | `labelmaps/dend_*.tif` + `masks_manifest.csv` | `split_merged_masks_m2b.py` |
| `m2b_split` | `labelmaps_split/dend_*.tif` | `filter_selected_masks_m3.py` **(napari)** |
| `m3_curated` | `labelmaps_curated_dynamic/dend_*.tif` (own or source run's) | `save_traces_m4.py` |
| `m4_traces` | `traces/dff_traces_curated_bgsub.csv` | `analyze_traces_m5.py` |
| `m5_analysed` | `traces/traces_quality.csv` | `behavior_plots.py` (needs `behavior/*_behavior.mat`) |
| `behavior` | `behavior_combined_plot.{pdf,png}` | whichever of M4/M5 is still missing |
| `complete` | M4 + M5 + behaviour all present | — |

Furthest artifact wins. Shared-FOV runs (0416 run2 borrows run1's masks) legitimately
skip M2/M2b/M3; a strict stop-at-first-gap would mis-rank them.

## What is MAVCA-specific, and why it is not a straight copy

**No master CSV.** Femtonics has `ranked_runs.csv`. Here the run list *is* the
directory tree, so runs are discovered by walking it.

**Per-run parameters live in code, cited to the steering file.** Frame rate
(5 Hz, but **6 Hz** for the 136/138 dual-channel sessions), `SKIP_FIRST_SECONDS`
(12 s, but **14 s** for 0508) and which run's curated masks to borrow are in
`SESSION_PARAMS` / `MASK_SOURCE` in `mavca_status.py`, each entry traceable to
`.kiro/steering/pipeline-context.md`. Every generated command passes them
explicitly. A run whose session is not listed shows in orange and the panel
**refuses** to auto-run it — running a 6 Hz session at 5 Hz is the single most
expensive documented mistake, so the panel makes it impossible rather than visible.

**Scripts are not CLI tools.** Every M-script is configured by constants at the
top and derives all paths from them at import time. `run_stage.py` is the general
form of what `run_m4_batch.py` did for M4 alone. The script on disk is never modified.

**Raw prefix varies by session** (`runA_` for 0320/0331/0508/0512, `runB_` for
0416). Detected from disk and passed through, so a script with the wrong literal
still finds its file.

**`STALE-OWN-MASKS`** is shown when a shared-FOV run still carries an old local
`labelmaps_curated_dynamic/` that differs from the source run's. Currently 0416
run2/run3: 80 local masks, but their traces were built from run1's 54. Those
folders are dead weight, not a source of truth.

## Adding a new session

1. Put data at `scape-data/<YYYY-MM-DD>/<mouse>/<runN>/raw/runA_<runN>_<mouse>-reslice-bin.tif`.
2. Add `("<date>", "<mouse>"): {"skip_s": …, "frame_rate": …}` to `SESSION_PARAMS`.
   Verify skip per run: `raw_T − preprocessed_T` frames ÷ frame rate.
3. If a run shares a FOV with an already-curated run, add it to `MASK_SOURCE`.
4. `--selftest`, then `--next`.

## Verified 2026-09-25

`--selftest` passes on all 23 runs (19 commands built, 0 on default params, 2 stale
flags). `--next --run-it` executed M5 on 0512/run5 end-to-end and wrote
`traces_quality.csv` (68 dends) into the correct folder; the run advanced to
`complete`. The Qt window opens offscreen with 23×9 cells; pressing *Run next* on a
`no_raw` row logs the reason and touches nothing. `run_stage.py --show` on a fake run
creates no directories.
