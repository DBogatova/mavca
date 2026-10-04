#!/usr/bin/env python3
"""auto_status.py - STATUS TRACKER for the automatic SCAPE apical-dendrite pipeline.

Scans all 20 runs from scape_common.discover_runs() and detects which pipeline
stage artifacts exist (detect, traces, validate, global, combo, movie) and
whether they are stale (older than inputs).

CLI:
  $PY code/Auto/auto_status.py             # full table + write scape-auto/auto_status.csv
  $PY code/Auto/auto_status.py --filter stage=detect
  $PY code/Auto/auto_status.py --mouse rbp4_132_phpeb
  $PY code/Auto/auto_status.py --next      # single next run + exact command
  $PY code/Auto/auto_status.py --next --run-it  # run it (if script exists)

Stages and their artifacts (OUT = scape-auto/<DATE>/<MOUSE>/<RUN>):
  detect   -> OUT/masks/auto_labelmap.tif + auto_masks.csv
  traces   -> OUT/traces/dff_auto.csv
  validate -> OUT/validation/validation.json + overlay.png
  global   -> OUT/traces/global_ca.csv
  combo    -> OUT/figures/combo_auto.png
  movie    -> OUT/movies/<run>_dual_behavior.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scape_common import PROJECT, discover_runs, AUTO_ROOT, PYTHON, get_run, Run

STAGES = ["no_raw", "raw", "detect", "traces", "validate", "global", "combo", "movie", "complete"]
STAGE_IDX = {s: i for i, s in enumerate(STAGES)}

SCRIPTS = {
    "detect": "auto_detect.py",
    "traces": "extract_traces.py",
    "validate": "validate_vs_human.py",
    "global": "global_ca.py",
    "combo": "combo_plot.py",
    "movie": "make_movie.py",
}


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def _exists_any(paths: list[Path]) -> bool:
    return any(p.exists() for p in paths)


def detect_artifacts(r: Run) -> dict:
    """Detect which stage artifacts exist and their staleness."""
    out = r.out
    a = {k: False for k in STAGES[1:]}  # raw .. complete
    a["stale"] = {}

    # raw
    a["raw"] = r.raw is not None and r.raw.exists()
    raw_mtime = _mtime(r.raw) if r.raw else 0

    # detect
    labelmap = out / "masks" / "auto_labelmap.tif"
    reviewed = out / "masks" / "auto_labelmap_reviewed.tif"
    masks_csv = out / "masks" / "auto_masks.csv"
    a["detect"] = labelmap.exists() and masks_csv.exists()
    detect_mtime = max(_mtime(labelmap), _mtime(reviewed)) if a["detect"] else 0
    a["stale"]["detect"] = a["detect"] and detect_mtime < raw_mtime

    # traces (prefer reviewed labelmap when it exists)
    dff_auto = out / "traces" / "dff_auto.csv"
    a["traces"] = dff_auto.exists()
    labelmap_used = reviewed if reviewed.exists() else labelmap
    a["stale"]["traces"] = a["traces"] and _mtime(dff_auto) < _mtime(labelmap_used)

    # validate (overlay.png is only created when human masks exist)
    val_json = out / "validation" / "validation.json"
    val_png = out / "validation" / "overlay.png"
    # Check if validation.json indicates skipped (no human masks) - then overlay not required
    val_skipped = False
    if val_json.exists():
        try:
            val_data = json.loads(val_json.read_text())
            val_skipped = val_data.get("skipped", False)
        except Exception:
            pass
    a["validate"] = val_json.exists() and (val_png.exists() or val_skipped)
    a["stale"]["validate"] = a["validate"] and _mtime(val_json) < _mtime(dff_auto)

    # global
    global_csv = out / "traces" / "global_ca.csv"
    a["global"] = global_csv.exists()
    a["stale"]["global"] = a["global"] and _mtime(global_csv) < raw_mtime

    # combo
    combo_auto_png = out / "figures" / "combo_auto.png"
    combo_auto_pdf = out / "figures" / "combo_auto.pdf"
    combo_human_png = out / "figures" / "combo_human.png"
    combo_human_pdf = out / "figures" / "combo_human.pdf"
    a["combo"] = combo_auto_png.exists() and combo_auto_pdf.exists()
    combo_mtime = min(_mtime(combo_auto_png), _mtime(combo_auto_pdf))
    a["stale"]["combo"] = a["combo"] and combo_mtime < max(_mtime(dff_auto), _mtime(global_csv))

    # movie
    movie = out / "movies" / f"{r.run}_dual_behavior.mp4"
    a["movie"] = movie.exists()
    a["stale"]["movie"] = a["movie"] and _mtime(movie) < max(raw_mtime, _mtime(labelmap_used))

    # complete = all stages done
    a["complete"] = all(a.get(s) for s in ["detect", "traces", "global", "combo", "movie"])

    return a


def stage_from_artifacts(a: dict) -> str:
    """Return the furthest stage whose artifact is present (later implies earlier)."""
    if not a["raw"]:
        return "no_raw"
    idx = STAGE_IDX["raw"]
    for name in ["detect", "traces", "validate", "global", "combo", "movie"]:
        if a.get(name):
            idx = max(idx, STAGE_IDX[name])
    if a.get("complete"):
        idx = STAGE_IDX["complete"]
    return STAGES[idx]


def checklist_str(a: dict) -> str:
    def f(name):
        if not a.get(name):
            return "-"
        return "!" if a.get("stale", {}).get(name) else "+"
    return (f"raw{f('raw')} det{f('detect')} tr{f('traces')} val{f('validate')} "
            f"glob{f('global')} cmb{f('combo')} mov{f('movie')}")


def load_metrics(r: Run) -> dict:
    """Load headline metrics from metrics.json if it exists."""
    p = r.out / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def next_action(r: Run, a: dict, stage: str) -> dict:
    """Return {'label', 'cmd', 'runnable', 'blocked_by'}."""
    key = r.key

    if stage == "no_raw":
        return dict(label="no raw stack found", cmd=None, runnable=False, blocked_by="fetch raw")

    # Find the first missing stage
    stage_order = ["detect", "traces", "validate", "global", "combo", "movie"]
    next_stage = None
    for s in stage_order:
        if not a.get(s) or a.get("stale", {}).get(s):
            next_stage = s
            break

    if next_stage is None:
        return dict(label="complete - nothing to do", cmd=None, runnable=False, blocked_by=None)

    script = SCRIPTS.get(next_stage)
    script_path = HERE / script if script else None

    if script_path is None or not script_path.exists():
        return dict(
            label=f"{next_stage} - script {script} not available yet",
            cmd=None,
            runnable=False,
            blocked_by=f"waiting for {script}",
        )

    stale = " (stale)" if a.get("stale", {}).get(next_stage) else ""
    cmd = [PYTHON, str(script_path), "--run", key]
    return dict(label=f"{next_stage}{stale}", cmd=cmd, runnable=True, blocked_by=None)


def build_status() -> list[dict]:
    """Build status for all runs."""
    runs_out = []
    for r in discover_runs(only_with_raw=False):
        a = detect_artifacts(r)
        stage = stage_from_artifacts(a)
        metrics = load_metrics(r)

        # Extract headline metrics
        detect_m = metrics.get("detect", {})
        val_m = metrics.get("validation", {})
        trace_m = metrics.get("traces", {})

        info = {
            "run": r,
            "key": r.key,
            "date": r.date,
            "mouse": r.mouse,
            "run_name": r.run,
            "frame_rate": r.frame_rate,
            "skip_s": r.skip_s,
            "stage": stage,
            "artifacts": a,
            "checklist": checklist_str(a),
            "next": next_action(r, a, stage),
            "has_human": r.human_mask_dir is not None,
            # Metrics
            # headline numbers written by auto_detect (detect) / validate_vs_human (validation)
            "n_auto": detect_m.get("n_auto", val_m.get("n_auto")),
            "n_human": val_m.get("n_human"),
            "recall": val_m.get("human_recall"),            # functional recall of human dendrites
            "precision": val_m.get("auto_matched_frac"),    # fraction of auto units matching a human mask
            "f1": val_m.get("f1_iou_0_2"),                  # spatial one-to-one F1 at IoU 0.2
            "trace_r": val_m.get("median_best_r"),          # median best trace r per human mask
        }
        runs_out.append(info)
    return runs_out


def write_status_csv(runs: list[dict]) -> Path:
    out = AUTO_ROOT / "auto_status.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "date", "mouse", "run", "stage", "checklist", "frame_rate_hz", "skip_s",
        "has_human", "n_auto", "n_human", "recall", "precision", "f1", "trace_r",
        "next_action", "next_command",
    ]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            cmd = r["next"]["cmd"]
            w.writerow([
                r["date"], r["mouse"], r["run_name"], r["stage"], r["checklist"],
                r["frame_rate"], r["skip_s"], "Y" if r["has_human"] else "",
                r["n_auto"] or "", r["n_human"] or "",
                f"{r['recall']:.2f}" if r["recall"] is not None else "",
                f"{r['precision']:.2f}" if r["precision"] is not None else "",
                f"{r['f1']:.2f}" if r["f1"] is not None else "",
                f"{r['trace_r']:.3f}" if r["trace_r"] is not None else "",
                r["next"]["label"],
                " ".join(cmd) if cmd else "",
            ])
    return out


def print_table(runs: list[dict]) -> None:
    hdr = (f"{'date':<10} {'mouse':<18} {'run':<6} {'Hz':>3} {'stage':<10} "
           f"{'auto':>4} {'human':>5} {'rec':>5} {'r':>5} next")
    print(hdr)
    print("-" * (len(hdr) + 20))
    for r in runs:
        n_auto = r["n_auto"] or ""
        n_human = r["n_human"] or ""
        f1 = f"{r['recall']:.2f}" if r["recall"] is not None else ""
        tr = f"{r['trace_r']:.2f}" if r["trace_r"] is not None else ""
        print(f"{r['date']:<10} {r['mouse']:<18} {r['run_name']:<6} {r['frame_rate']:>3.0f} "
              f"{r['stage']:<10} {n_auto:>4} {n_human:>5} {f1:>5} {tr:>5} {r['next']['label']}")


def cmd_next(runs: list[dict], run_it: bool) -> int:
    """Print or run the next action."""
    cand = [r for r in runs if r["next"]["runnable"]]
    if not cand:
        print("Nothing actionable: every run is complete or waiting for scripts.")
        return 0

    # Prioritize runs furthest along (to finish them first)
    r = max(cand, key=lambda d: (STAGE_IDX.get(d["stage"], 0), d["date"]))
    nx = r["next"]

    print(f"NEXT: {r['key']}")
    print(f"  stage      : {r['stage']}   [{r['checklist']}]")
    print(f"  params     : {r['frame_rate']:.0f} Hz, skip {r['skip_s']:.0f} s")
    print(f"  do next    : {nx['label']}")
    if nx["cmd"]:
        print(f"  command    : {' '.join(nx['cmd'])}")
    print()

    if not run_it:
        return 0
    if not nx["cmd"]:
        print("  --run-it: no command to run.")
        return 0
    print(f"  --run-it: executing...\n")
    sys.stdout.flush()
    return subprocess.call(nx["cmd"], cwd=str(PROJECT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", default=None, metavar="stage=NAME")
    ap.add_argument("--mouse", default=None)
    ap.add_argument("--next", action="store_true", help="print the single next run to act on")
    ap.add_argument("--run-it", action="store_true", help="with --next: execute the command")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args(argv)

    runs = build_status()

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
        print(f"wrote {out.relative_to(PROJECT)}  ({len(runs)} runs)\n")

    print_table(view)

    # Stage tally
    tally: dict[str, int] = {}
    for r in runs:
        tally[r["stage"]] = tally.get(r["stage"], 0) + 1
    print(f"\nstage tally ({len(runs)} runs): " + "  ".join(f"{s}={tally[s]}" for s in STAGES if s in tally))
    return 0


if __name__ == "__main__":
    sys.exit(main())
