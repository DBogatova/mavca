#!/usr/bin/env python3
"""run_auto.py - Run the automatic SCAPE pipeline stages in dependency order.

Runs: detect -> traces -> validate -> global -> combo -> movie (-> stats)

CLI:
  $PY code/Auto/run_auto.py --all                   # all runs, all stages
  $PY code/Auto/run_auto.py --run KEY [--run KEY2]  # specific runs
  $PY code/Auto/run_auto.py --all --stages detect traces  # subset of stages
  $PY code/Auto/run_auto.py --all --force           # rebuild even if up-to-date
  $PY code/Auto/run_auto.py --all --jobs 2          # max 2 parallel run-processes

Each stage is a subprocess call of the stage script (code/Auto/<script>.py --run KEY).
Logs JSON lines to scape-auto/logs/runner.jsonl and full stdout/stderr per run/stage
to scape-auto/logs/. Continues with other runs on failure and prints a final summary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scape_common import PROJECT, AUTO_ROOT, PYTHON, discover_runs, get_run

# Stage order: detect -> traces -> validate/global (parallel ok) -> combo -> movie
STAGE_ORDER = ["detect", "traces", "validate", "global", "combo", "movie"]
STAGE_SCRIPTS = {
    "detect": "auto_detect.py",
    "traces": "extract_traces.py",
    "validate": "validate_vs_human.py",
    "global": "global_ca.py",
    "combo": "combo_plot.py",
    "movie": "make_movie.py",
}
# Dependencies: a stage requires these to be done first
STAGE_DEPS = {
    "detect": [],
    "traces": ["detect"],
    "validate": ["traces"],
    "global": [],  # independent of detect/traces - uses raw
    "combo": ["traces", "global"],
    "movie": ["detect", "global"],  # shows the detected dendrites + global Ca
}

LOG_DIR = AUTO_ROOT / "logs"


def log_entry(stage: str, run_key: str, result: str, elapsed: float = 0):
    """Append a JSON line to runner.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "run": run_key,
        "result": result,
        "elapsed_s": round(elapsed, 1),
    }
    with open(LOG_DIR / "runner.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def script_exists(stage: str) -> bool:
    script = STAGE_SCRIPTS.get(stage)
    return script is not None and (HERE / script).exists()


def run_stage(run_key: str, stage: str, force: bool = False) -> dict:
    """Run a single stage for a single run. Returns result dict."""
    script = STAGE_SCRIPTS.get(stage)
    if not script:
        return {"run": run_key, "stage": stage, "success": False, "error": "unknown stage"}

    script_path = HERE / script
    if not script_path.exists():
        return {"run": run_key, "stage": stage, "success": False, "error": f"script {script} not available"}

    cmd = [PYTHON, str(script_path), "--run", run_key]
    if force:
        cmd.append("--force")

    # Create log directory for this run
    log_subdir = LOG_DIR / run_key.replace("/", "_")
    log_subdir.mkdir(parents=True, exist_ok=True)
    log_file = log_subdir / f"{stage}.log"

    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            lf.write(f"# {datetime.now().isoformat()} run={run_key} stage={stage}\n")
            lf.write(f"# cmd: {' '.join(cmd)}\n\n")
            lf.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT),
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=1800,  # 30 min timeout per stage
            )
        elapsed = time.time() - t0
        success = proc.returncode == 0
        return {
            "run": run_key,
            "stage": stage,
            "success": success,
            "elapsed": elapsed,
            "returncode": proc.returncode,
            "log": str(log_file),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {"run": run_key, "stage": stage, "success": False, "elapsed": elapsed, "error": "timeout (30 min)"}
    except Exception as e:
        elapsed = time.time() - t0
        return {"run": run_key, "stage": stage, "success": False, "elapsed": elapsed, "error": str(e)}


def run_pipeline_for_run(run_key: str, stages: list[str], force: bool) -> list[dict]:
    """Run the pipeline stages in order for a single run."""
    results = []
    completed_stages = set()

    for stage in STAGE_ORDER:
        if stage not in stages:
            continue

        # Check dependencies
        deps = STAGE_DEPS.get(stage, [])
        missing_deps = [d for d in deps if d in stages and d not in completed_stages]
        if missing_deps:
            # Skip - dependency not met (it either failed or wasn't requested)
            results.append({
                "run": run_key,
                "stage": stage,
                "success": False,
                "skipped": True,
                "error": f"dependency not met: {missing_deps}",
            })
            continue

        if not script_exists(stage):
            results.append({
                "run": run_key,
                "stage": stage,
                "success": False,
                "error": f"script not available",
            })
            continue

        result = run_stage(run_key, stage, force)
        results.append(result)
        log_entry(stage, run_key, "OK" if result["success"] else "FAIL", result.get("elapsed", 0))

        if result["success"]:
            completed_stages.add(stage)
        else:
            # Continue with other stages if possible (they might not depend on this one)
            pass

    return results


def run_stats():
    """Run auto_stats.py --all at the end."""
    script_path = HERE / "auto_stats.py"
    if not script_path.exists():
        print("auto_stats.py not available - skipping cohort statistics")
        return {"stage": "stats", "success": False, "error": "script not available"}

    cmd = [PYTHON, str(script_path), "--all"]
    log_file = LOG_DIR / "stats.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, cwd=str(PROJECT), stdout=lf, stderr=subprocess.STDOUT, timeout=600)
        elapsed = time.time() - t0
        return {"stage": "stats", "success": proc.returncode == 0, "elapsed": elapsed}
    except Exception as e:
        return {"stage": "stats", "success": False, "error": str(e)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", dest="runs", metavar="KEY",
                    help="run key (DATE/MOUSE/RUN), repeatable")
    ap.add_argument("--all", action="store_true", help="process all discovered runs")
    ap.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER,
                    help="subset of stages to run (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs are newer than inputs")
    ap.add_argument("--jobs", type=int, default=2,
                    help="max parallel run-processes (default: 2, memory-safe)")
    ap.add_argument("--no-stats", action="store_true", help="skip cohort statistics at the end")
    args = ap.parse_args(argv)

    if not args.all and not args.runs:
        ap.error("specify --all or --run KEY")

    # Build run list
    if args.all:
        run_keys = [r.key for r in discover_runs()]
    else:
        run_keys = args.runs

    print(f"SCAPE Auto Pipeline Runner")
    print(f"  runs:   {len(run_keys)}")
    print(f"  stages: {' -> '.join(args.stages)}")
    print(f"  jobs:   {args.jobs}")
    print(f"  force:  {args.force}")
    print()

    # Check which scripts are available
    available = {s: script_exists(s) for s in args.stages}
    missing = [s for s, ok in available.items() if not ok]
    if missing:
        print(f"WARNING: Scripts not available for stages: {missing}")
        print("         Those stages will be skipped.\n")

    t_start = time.time()
    log_entry("runner", "start", f"runs={len(run_keys)} stages={args.stages}")

    # Run pipeline for each run (with limited parallelism for memory safety)
    all_results = []
    if args.jobs == 1:
        # Sequential
        for i, key in enumerate(run_keys, 1):
            print(f"[{i}/{len(run_keys)}] {key}")
            results = run_pipeline_for_run(key, args.stages, args.force)
            for r in results:
                status = "OK" if r["success"] else ("SKIP" if r.get("skipped") else "FAIL")
                elapsed = f" ({r['elapsed']:.1f}s)" if "elapsed" in r else ""
                print(f"    {r['stage']}: {status}{elapsed}")
            all_results.extend(results)
    else:
        # Parallel with ThreadPoolExecutor (subprocess-based, so thread-safe)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_pipeline_for_run, key, args.stages, args.force): key
                for key in run_keys
            }
            for i, future in enumerate(as_completed(futures), 1):
                key = futures[future]
                try:
                    results = future.result()
                    print(f"[{i}/{len(run_keys)}] {key}")
                    for r in results:
                        status = "OK" if r["success"] else ("SKIP" if r.get("skipped") else "FAIL")
                        elapsed = f" ({r['elapsed']:.1f}s)" if "elapsed" in r else ""
                        print(f"    {r['stage']}: {status}{elapsed}")
                    all_results.extend(results)
                except Exception as e:
                    print(f"[{i}/{len(run_keys)}] {key}: ERROR {e}")
                    all_results.append({"run": key, "success": False, "error": str(e)})

    # Run cohort statistics
    if not args.no_stats:
        print("\nRunning cohort statistics...")
        stats_result = run_stats()
        if stats_result["success"]:
            print(f"  stats: OK ({stats_result.get('elapsed', 0):.1f}s)")
        else:
            print(f"  stats: FAIL - {stats_result.get('error', 'unknown')}")

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SUMMARY  (total: {total_time:.0f}s = {total_time/60:.1f}min)")
    print(f"{'='*60}")

    # Per-stage summary
    for stage in args.stages:
        stage_results = [r for r in all_results if r.get("stage") == stage]
        ok = sum(1 for r in stage_results if r.get("success"))
        print(f"  {stage:10s}: {ok}/{len(stage_results)}")

    # Failures
    failures = [r for r in all_results if not r.get("success") and not r.get("skipped")]
    if failures:
        print(f"\nFailed stages ({len(failures)}):")
        for r in failures:
            err = r.get("error", f"exit {r.get('returncode', '?')}")
            print(f"  {r['run']} / {r.get('stage', '?')}: {err}")
            if r.get("log"):
                print(f"    log: {r['log']}")

    # Save summary
    summary_path = LOG_DIR / "runner_summary.json"
    summary = {
        "time": datetime.now(timezone.utc).isoformat(),
        "total_seconds": total_time,
        "runs": run_keys,
        "stages": args.stages,
        "results": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    log_entry("runner", "done", f"{len([r for r in all_results if r.get('success')])}/{len(all_results)} OK")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
