#!/usr/bin/env python
"""
mavca_status.py - STATUS TRACKER + DRIVER for the MAVCA (SCAPE apical dendrite) pipeline.

The apical-dendrites-2025 counterpart of femtonics-data/code/STEP7_workflow/femto_status.py.
Same principle: every stage is detected BY DISK PRESENCE of the artifact it produces, so
the table can never disagree with reality. Same outputs: a table, processing_status.csv,
--next, --next --run-it.

WHAT IS DIFFERENT FROM THE FEMTONICS VERSION, AND WHY
-----------------------------------------------------
1. NO MASTER CSV. Femtonics has ranked_runs.csv (44 runs matched to behaviour by trigger
   fingerprint). MAVCA has no equivalent, and the run list IS the directory tree:
   scape-data/<DATE>/<MOUSE>/<runN>/. So runs are discovered by walking the tree.

2. STAGES ARE THE M-LADDER, not the Femtonics autoseg ladder:
     raw           raw/run[AB]_<run>_<mouse>-reslice-bin.tif
     m1_events     preprocessed/active_frames.npy + stack_voxel_norm_mean_sub.tif
     m1_5_frames   preprocessed/best_frames/  (non-empty)
     m2_masks      labelmaps/dend_*_labelmap.tif + masks_manifest.csv
     m2b_split     labelmaps_split/dend_*_labelmap.tif + masks_manifest_split.csv
     m3_curated    labelmaps_curated_dynamic/dend_*_labelmap.tif   <- napari GUI
     m4_traces     traces/dff_traces_curated_bgsub.csv
     m5_analysed   traces/traces_quality.csv
     behavior      behavior_combined_plot.{pdf,png}
   "Furthest artifact wins" (a later artifact implies the earlier work happened).
   Shared-mask runs (0416 run2 uses run1's masks) legitimately skip M2/M2b/M3 and
   go raw -> m4_traces; a strict stop-at-first-gap would mis-rank them.

3. PER-RUN PARAMETERS ARE NOT IN A CSV, THEY ARE IN THE STEERING FILE. The frame rate
   (5 Hz vs 6 Hz for 136/138), SKIP_FIRST_SECONDS (12 vs 14 s) and MASK_SOURCE_RUN
   (which run's curated masks to borrow) are documented in .kiro/steering/
   pipeline-context.md and in run_m4_batch.py, not in any machine-readable place.
   They live in RUN_PARAMS below, with the steering file as the citation. Anything
   not listed falls back to a default and is FLAGGED in the table, so a new session
   cannot silently run at the wrong rate.

4. SCRIPTS ARE NOT CLI TOOLS. Every M-script is configured by editing constants at
   the top. So commands here are built through code/Workflow/run_stage.py, which
   rewrites those constants in memory and runs the unmodified script.

5. RAW PREFIX VARIES BY SESSION (runA_ vs runB_) and is detected from disk, then
   passed to run_stage so scripts with the wrong literal still find the file.

NOTHING IS WRITTEN except processing_status.csv at the project root (and only when
asked). No run directory is ever touched.

CLI
---
  PY=.venv311/bin/python
  $PY code/Workflow/mavca_status.py                 # table + write processing_status.csv
  $PY code/Workflow/mavca_status.py --filter stage=m3_curated
  $PY code/Workflow/mavca_status.py --mouse rbp4_132_phpeb
  $PY code/Workflow/mavca_status.py --next          # single next run + exact command
  $PY code/Workflow/mavca_status.py --next --run-it # run it (non-GUI stages only)
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "scape-data"
CODE = PROJECT_ROOT / "code"
VENV_PY = str(PROJECT_ROOT / ".venv311" / "bin" / "python")
RUN_STAGE = str(CODE / "Workflow" / "run_stage.py")

STAGES = [
    "no_raw",
    "raw",
    "m1_events",
    "m1_5_frames",
    "m2_masks",
    "m2b_split",
    "m3_curated",
    "m4_traces",
    "m5_analysed",
    "behavior",
    "complete",
]
STAGE_IDX = {s: i for i, s in enumerate(STAGES)}

# ---------------------------------------------------------------------------
# per-run parameters. SOURCE: .kiro/steering/pipeline-context.md (sections
# "Per-dataset SKIP_FIRST_SECONDS", "Acquisition frame rate", "Datasets with
# shared masks") and code/Traces-STEP3/run_m4_batch.py. Keys are
# (DATE, MOUSE) for session-wide values, with per-run overrides where a run
# borrows another run's masks.
# ---------------------------------------------------------------------------
DEFAULT_FRAME_RATE = 5.0
DEFAULT_SKIP_S = 12.0

SESSION_PARAMS: dict[tuple[str, str], dict] = {
    ("2026-05-08", "rbp4_139_phpeb"): {"skip_s": 14.0, "frame_rate": 5.0},
    ("2026-05-12", "rbp4_132_phpeb"): {"skip_s": 12.0, "frame_rate": 5.0},
    ("2026-04-16", "rbp4_132_phpeb"): {"skip_s": 12.0, "frame_rate": 5.0},
    ("2026-03-31", "rbp4_132_phpeb"): {"skip_s": 12.0, "frame_rate": 5.0},
    ("2026-03-20", "rbp4cre_139_phpeb"): {"skip_s": 12.0, "frame_rate": 5.0},
    # dual-channel deeper sessions acquire at 6 Hz (steering: "Acquisition frame rate")
    ("2026-02-09", "rbp4cre_136_phpeb"): {"skip_s": 12.0, "frame_rate": 6.0, "has_ach": True},
    ("2026-02-17", "rbp4cre_138_phpeb"): {"skip_s": 12.0, "frame_rate": 6.0, "has_ach": True},
    ("2025-12-02", "rbp4cre_136_phpeb"): {"skip_s": 12.0, "frame_rate": 6.0, "has_ach": True},
}

# run -> run whose labelmaps_curated_dynamic it uses for M4 (steering: "Datasets
# with shared masks"; 0416 run1/2/3 all 54 identical after the 2026-06-04 fix)
MASK_SOURCE: dict[tuple[str, str, str], str] = {
    ("2026-04-16", "rbp4_132_phpeb", "run2"): "run1",
    ("2026-04-16", "rbp4_132_phpeb", "run3"): "run1",
    ("2026-04-16", "rbp4_132_phpeb", "run6"): "run5",
    ("2026-04-16", "rbp4_132_phpeb", "run7"): "run5",
    ("2026-03-31", "rbp4_132_phpeb", "run6"): "run7",
    ("2026-03-31", "rbp4_132_phpeb", "run9"): "run8",
    ("2026-03-31", "rbp4_132_phpeb", "run10"): "run8",
    ("2026-05-08", "rbp4_139_phpeb", "run6"): "run5",
    ("2026-05-12", "rbp4_132_phpeb", "run6"): "run5",
}


# ---------------------------------------------------------------------------
# run discovery
# ---------------------------------------------------------------------------
def _run_sort_key(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def discover_runs(data_root: Path = DATA_ROOT) -> list[dict]:
    runs = []
    if not data_root.is_dir():
        return runs
    for date_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}$", p.name)):
        for mouse_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            for run_dir in sorted((p for p in mouse_dir.iterdir() if p.is_dir() and p.name.startswith("run")),
                                  key=lambda p: _run_sort_key(p.name)):
                runs.append({
                    "date": date_dir.name, "mouse": mouse_dir.name, "run": run_dir.name,
                    "run_dir": run_dir,
                })
    return runs


# ---------------------------------------------------------------------------
# per-run parameters + raw detection
# ---------------------------------------------------------------------------
def attach_params(run: dict) -> None:
    key = (run["date"], run["mouse"])
    p = SESSION_PARAMS.get(key)
    run["params_known"] = p is not None
    p = p or {}
    run["frame_rate"] = p.get("frame_rate", DEFAULT_FRAME_RATE)
    run["skip_s"] = p.get("skip_s", DEFAULT_SKIP_S)
    run["has_ach"] = p.get("has_ach", False)
    run["mask_source"] = MASK_SOURCE.get((run["date"], run["mouse"], run["run"]))


def detect_raw(run: dict) -> None:
    """Find the raw 4D stack and its prefix. Prefers preprocessed/raw_clean.tif when
    present (M1 and M4 both do), else raw/run[AB]_<run>_<mouse>-reslice-bin.tif."""
    d, r, m = run["run_dir"], run["run"], run["mouse"]
    run["raw_prefix"] = None
    run["raw_path"] = None
    for pre in ("runA", "runB"):
        cand = d / "raw" / f"{pre}_{r}_{m}-reslice-bin.tif"
        if cand.is_file():
            run["raw_prefix"], run["raw_path"] = pre, cand
            break
    if run["raw_path"] is None:
        # dual-channel sessions name raw differently (green/red reslice)
        for pre in ("runA", "runB"):
            cand = d / "raw" / f"{pre}_{r}_{m}-reslice-green.tif"
            if cand.is_file():
                run["raw_prefix"], run["raw_path"] = pre, cand
                break
    clean = d / "preprocessed" / "raw_clean.tif"
    run["raw_clean"] = clean if clean.is_file() else None


# ---------------------------------------------------------------------------
# artifact detection (pure disk presence)
# ---------------------------------------------------------------------------
def _nonempty_glob(folder: Path, pattern: str) -> int:
    return sum(1 for _ in folder.glob(pattern)) if folder.is_dir() else 0


def detect_artifacts(run: dict) -> dict:
    d = run["run_dir"]
    pre = d / "preprocessed"
    a = {k: False for k in STAGES[1:]}  # raw .. complete
    a["raw"] = run["raw_path"] is not None or run["raw_clean"] is not None
    a["m1_events"] = (pre / "active_frames.npy").is_file() and (pre / "stack_voxel_norm_mean_sub.tif").is_file()
    a["m1_5_frames"] = _nonempty_glob(pre / "best_frames", "*") > 0
    run["n_masks_m2"] = _nonempty_glob(d / "labelmaps", "dend_*_labelmap.tif")
    a["m2_masks"] = run["n_masks_m2"] > 0 and (d / "masks_manifest.csv").is_file()
    run["n_masks_split"] = _nonempty_glob(d / "labelmaps_split", "dend_*_labelmap.tif")
    a["m2b_split"] = run["n_masks_split"] > 0
    run["n_masks_curated"] = _nonempty_glob(d / "labelmaps_curated_dynamic", "dend_*_labelmap.tif")
    a["m3_curated"] = run["n_masks_curated"] > 0
    # shared-FOV runs: the masks that matter live in the source run's folder
    ms = run["mask_source"]
    run["n_masks_effective"] = run["n_masks_curated"]
    run["stale_own_masks"] = False
    if ms:
        src_dir = d.parent / ms / "labelmaps_curated_dynamic"
        run["n_masks_effective"] = _nonempty_glob(src_dir, "dend_*_labelmap.tif")
        a["m3_curated"] = a["m3_curated"] or run["n_masks_effective"] > 0
        run["stale_own_masks"] = run["n_masks_curated"] > 0 and run["n_masks_curated"] != run["n_masks_effective"]
    a["m4_traces"] = (d / "traces" / "dff_traces_curated_bgsub.csv").is_file()
    a["m5_analysed"] = (d / "traces" / "traces_quality.csv").is_file()
    a["behavior"] = any((d / f"behavior_combined_plot.{ext}").is_file() for ext in ("pdf", "png"))
    a["complete"] = a["m4_traces"] and a["m5_analysed"] and a["behavior"]
    # behaviour inputs (needed for the behavior stage to be runnable)
    run["behavior_mat"] = next(iter((d / "behavior").glob("*_behavior.mat")), None) if (d / "behavior").is_dir() else None
    run["trigger_csv"] = next(iter((d / "trigger").glob("*_trigger.csv")), None) if (d / "trigger").is_dir() else None
    return a


def stage_from_artifacts(a: dict, run: dict) -> str:
    if not a["raw"]:
        return "no_raw"
    idx = STAGE_IDX["raw"]
    for name in ("m1_events", "m1_5_frames", "m2_masks", "m2b_split", "m3_curated",
                 "m4_traces", "m5_analysed", "behavior"):
        if a[name]:
            idx = max(idx, STAGE_IDX[name])
    if a["complete"]:
        idx = STAGE_IDX["complete"]
    return STAGES[idx]


def checklist_str(a: dict) -> str:
    f = lambda b: "+" if b else "-"
    return (f"raw{f(a['raw'])} m1{f(a['m1_events'])} m1.5{f(a['m1_5_frames'])} "
            f"m2{f(a['m2_masks'])} m2b{f(a['m2b_split'])} m3{f(a['m3_curated'])} "
            f"m4{f(a['m4_traces'])} m5{f(a['m5_analysed'])} beh{f(a['behavior'])}")


# ---------------------------------------------------------------------------
# next action -> exact command (via run_stage.py)
# ---------------------------------------------------------------------------
def _stage_cmd(run: dict, script: Path, extra_sets: dict | None = None, script_args: list[str] | None = None) -> list[str]:
    cmd = [VENV_PY, RUN_STAGE, str(script),
           "--date", run["date"], "--mouse", run["mouse"], "--run", run["run"]]
    sets = {"SKIP_FIRST_SECONDS": run["skip_s"], "M1_SKIP_SECONDS": run["skip_s"],
            "FRAME_RATE": run["frame_rate"], "FS_HZ": run["frame_rate"]}
    if extra_sets:
        sets.update(extra_sets)
    for k, v in sets.items():
        cmd += ["--set", f"{k}={v!r}"]
    if run["raw_prefix"]:
        cmd += ["--raw-prefix", run["raw_prefix"]]
    if script_args:
        cmd += ["--"] + script_args
    return cmd


def next_action(run: dict, stage: str) -> dict:
    """{'label','cmd'|None,'gui','runnable','blocked_by'}"""
    S = CODE
    ms = run["mask_source"]

    if stage == "no_raw":
        want = "reslice-green.tif" if run["has_ach"] else "reslice-bin.tif"
        return dict(label=f"raw missing: raw/run[AB]_{run['run']}_{run['mouse']}-{want}",
                    cmd=None, gui=False, runnable=False, blocked_by="fetch raw from SCC")
    if stage == "raw":
        return dict(label="M1 find events (active frames, normalised stack)", gui=False, runnable=True, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Preprocessing-STEP1/find_events_m1.py"))
    if stage == "m1_events":
        return dict(label="M1.5 pre-segmentation (best frames)", gui=False, runnable=True, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Preprocessing-STEP1/pre_segmentation_m1.5.py"))
    if stage == "m1_5_frames":
        if ms:
            # shared-FOV run: skip mask generation, borrow curated masks
            return dict(label=f"M4 traces using {ms}'s curated masks (shared FOV)", gui=False, runnable=True, blocked_by=None,
                        cmd=_stage_cmd(run, S / "Traces-STEP3/save_traces_m4.py", {"MASK_SOURCE_RUN": ms}))
        return dict(label="M2 auto-mask", gui=False, runnable=True, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Masks-STEP2/auto_mask_m2.py"))
    if stage == "m2_masks":
        return dict(label="M2b split merged masks", gui=False, runnable=True, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Masks-STEP2/split_merged_masks_m2b.py"))
    if stage == "m2b_split":
        return dict(label="M3 curate masks (napari GUI)", gui=True, runnable=False, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Masks-STEP2/filter_selected_masks_m3.py"))
    if stage == "m3_curated":
        extra = {"MASK_SOURCE_RUN": ms} if ms else {}
        return dict(label="M4 save traces (core-shell bg-sub)", gui=False, runnable=True, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Traces-STEP3/save_traces_m4.py", extra))
    if stage == "m4_traces":
        return dict(label="M5 analyse traces", gui=False, runnable=True, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Traces-STEP3/analyze_traces_m5.py"))
    if stage == "m5_analysed":
        if run["behavior_mat"] is None:
            return dict(label="behaviour plot - BLOCKED: no behavior/*_behavior.mat", cmd=None, gui=False,
                        runnable=False, blocked_by="publish behaviour .mat for this run")
        extra = {"HAS_ACH": run["has_ach"], "BEHAVIOR_MAT": str(run["behavior_mat"])}
        return dict(label="behaviour + Ca combined plot", gui=False, runnable=True, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Behavior-Analysis/behavior_plots.py", extra))
    if stage == "behavior":
        # behaviour done but m4/m5 missing (old runs) - point at whichever is missing
        a = run["artifacts"]
        if not a["m4_traces"]:
            extra = {"MASK_SOURCE_RUN": ms} if ms else {}
            return dict(label="M4 save traces (behaviour plot exists but traces missing)", gui=False, runnable=True,
                        blocked_by=None, cmd=_stage_cmd(run, S / "Traces-STEP3/save_traces_m4.py", extra))
        return dict(label="M5 analyse traces", gui=False, runnable=True, blocked_by=None,
                    cmd=_stage_cmd(run, S / "Traces-STEP3/analyze_traces_m5.py"))
    return dict(label="complete - nothing to do", cmd=None, gui=False, runnable=False, blocked_by=None)


def cmd_display(cmd: list[str] | None) -> str:
    if not cmd:
        return ""
    shown = ["$PY" if c == VENV_PY else ("run_stage.py" if c == RUN_STAGE else c) for c in cmd]
    shown = [str(Path(c).relative_to(PROJECT_ROOT)) if c.startswith(str(PROJECT_ROOT)) else c for c in shown]
    return f"cd {PROJECT_ROOT} && PY={VENV_PY}\n  " + " ".join(shown)


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def build_status(data_root: Path = DATA_ROOT) -> list[dict]:
    runs = discover_runs(data_root)
    for run in runs:
        attach_params(run)
        detect_raw(run)
        a = detect_artifacts(run)
        run["artifacts"] = a
        run["stage"] = stage_from_artifacts(a, run)
        run["checklist"] = checklist_str(a)
        run["next"] = next_action(run, run["stage"])
        run["label"] = f"{run['date']}/{run['mouse']}/{run['run']}"
        flags = []
        if not run["params_known"]:
            flags.append("PARAMS-DEFAULTED")
        if run["mask_source"]:
            flags.append(f"masks<-{run['mask_source']}")
        if run["has_ach"]:
            flags.append("ACh")
        if run["raw_clean"]:
            flags.append("raw_clean")
        if run["stale_own_masks"]:
            flags.append(f"STALE-OWN-MASKS({run['n_masks_curated']})")
        run["flags"] = " ".join(flags)
    return runs


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def _rel(p) -> str:
    if not p:
        return ""
    try:
        return str(Path(p).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def write_status_csv(runs: list[dict]) -> Path:
    out = PROJECT_ROOT / "processing_status.csv"
    cols = ["date", "mouse", "run", "stage", "checklist", "frame_rate_hz", "skip_first_s",
            "raw_prefix", "mask_source_run", "n_masks_m2", "n_masks_split", "n_masks_curated", "n_masks_effective",
            "behavior_mat", "flags", "next_action", "next_command"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            cmd = r["next"]["cmd"]
            w.writerow([r["date"], r["mouse"], r["run"], r["stage"], r["checklist"],
                        r["frame_rate"], r["skip_s"], r["raw_prefix"] or "", r["mask_source"] or "",
                        r["n_masks_m2"], r["n_masks_split"], r["n_masks_curated"], r["n_masks_effective"],
                        _rel(r["behavior_mat"]), r["flags"], r["next"]["label"],
                        " ".join(cmd) if cmd else ""])
    return out


def print_table(runs: list[dict]) -> None:
    hdr = f"{'date':<10} {'mouse':<18} {'run':<6} {'Hz':>3} {'skip':>4} {'stage':<12} {'masks':>5} {'flags':<22} next"
    print(hdr)
    print("-" * (len(hdr) + 30))
    for r in runs:
        print(f"{r['date']:<10} {r['mouse']:<18} {r['run']:<6} {r['frame_rate']:>3.0f} {r['skip_s']:>4.0f} "
              f"{r['stage']:<12} {r['n_masks_effective']:>5} {r['flags']:<22} {r['next']['label']}")


def cmd_next(runs: list[dict], run_it: bool) -> int:
    cand = [r for r in runs if r["stage"] not in ("no_raw", "complete") and r["next"]["cmd"]]
    if not cand:
        print("Nothing actionable: every run with raw data is complete or blocked.")
        return 0
    # furthest-along first: finishing a run beats starting one
    r = max(cand, key=lambda d: (STAGE_IDX[d["stage"]], d["date"]))
    nx = r["next"]
    print(f"NEXT: {r['label']}")
    print(f"  stage      : {r['stage']}   [{r['checklist']}]")
    print(f"  params     : {r['frame_rate']:.0f} Hz, skip {r['skip_s']:.0f} s, raw prefix {r['raw_prefix']}"
          + (f", masks from {r['mask_source']}" if r["mask_source"] else "")
          + ("   ** PARAMS DEFAULTED - verify in steering **" if not r["params_known"] else ""))
    print(f"  do next    : {nx['label']}" + ("   [napari GUI]" if nx["gui"] else ""))
    print(f"  command    :\n  {cmd_display(nx['cmd'])}\n")
    if not run_it:
        return 0
    if nx["gui"]:
        print("  --run-it: interactive napari step; launch it yourself (printed above).")
        return 0
    print(f"  --run-it: executing (cwd={PROJECT_ROOT}) ...\n")
    sys.stdout.flush()
    return subprocess.call(nx["cmd"], cwd=str(PROJECT_ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--filter", default=None, metavar="stage=NAME")
    ap.add_argument("--mouse", default=None)
    ap.add_argument("--next", action="store_true")
    ap.add_argument("--run-it", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.data_root).resolve() if args.data_root else DATA_ROOT
    runs = build_status(root)

    if args.next:
        return cmd_next(runs, args.run_it)

    view = runs
    if args.filter:
        if not args.filter.startswith("stage="):
            ap.error("--filter must look like stage=NAME")
        want = args.filter.split("=", 1)[1]
        if want not in STAGE_IDX:
            ap.error(f"unknown stage '{want}'; valid: {', '.join(STAGES)}")
        view = [r for r in view if r["stage"] == want]
    if args.mouse:
        view = [r for r in view if r["mouse"] == args.mouse]

    if not args.no_csv:
        out = write_status_csv(runs)
        print(f"wrote {out.relative_to(PROJECT_ROOT)}  ({len(runs)} runs)\n")
    print_table(view)
    tally: dict[str, int] = {}
    for r in runs:
        tally[r["stage"]] = tally.get(r["stage"], 0) + 1
    print(f"\nstage tally ({len(runs)} runs): " + "  ".join(f"{s}={tally[s]}" for s in STAGES if s in tally))
    nd = [r["label"] for r in runs if not r["params_known"]]
    if nd:
        print(f"\n** {len(nd)} run(s) using DEFAULT params (5 Hz, 12 s) - not in SESSION_PARAMS: " + ", ".join(nd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
