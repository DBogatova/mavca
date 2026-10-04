#!/usr/bin/env python
"""run_stage.py - run ONE pipeline script for ONE run without editing the script.

WHY THIS EXISTS
---------------
Every MAVCA script (find_events_m1, auto_mask_m2, save_traces_m4, ...) is
configured by module-level constants at the top of the file:

    DATE = "2026-04-16"
    MOUSE = "rbp4_132_phpeb"
    RUN = "run1"
    SKIP_FIRST_SECONDS = 12.0

and every path is DERIVED from those at import time (BASE = ... / DATE / MOUSE / RUN,
then RAW_ORIG_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}-...", plus mkdir calls).
So you cannot `import` a script and then change RUN: the paths are already baked.
run_m4_batch.py works around that for M4 alone by re-deriving ~12 globals by hand.

This runner generalises that for every script with zero per-script knowledge:
it reads the script SOURCE, rewrites the assignment lines for the names you pass
(DATE, MOUSE, RUN, SKIP_FIRST_SECONDS, FRAME_RATE, MASK_SOURCE_RUN, ...) and the
raw prefix (runA_/runB_), then exec()s the patched source as __main__ with
__file__ pointing at the ORIGINAL path. The script's own derivation logic then
computes every path correctly, exactly as if you had edited the header.

The script on disk is never modified.

USAGE
-----
  run_stage.py SCRIPT --date D --mouse M --run R
               [--set NAME=VALUE ...]     override any top-level constant
               [--raw-prefix runA|runB]   rewrite run[AB]_{RUN} literals
               [--show]                   print the rewritten lines, do not run
               [-- ARGS...]               passed to the script as sys.argv[1:]

Values in --set are parsed as Python literals when possible (14 -> int, 6.0 ->
float, None, "run8" -> str); anything that fails to parse is kept as a string.
Only names that actually appear as `NAME = ...` at column 0 are rewritten; a
--set for a name the script does not define is reported and ignored, so one
override set can be passed to every stage safely (e.g. M1_SKIP_SECONDS only
exists in M3).
"""
from __future__ import annotations

import argparse
import ast
import re
import runpy
import sys
from pathlib import Path


def _literal(v: str):
    try:
        return ast.literal_eval(v)
    except Exception:
        return v


def patch_source(src: str, overrides: dict, raw_prefix: str | None):
    """Return (patched_source, applied: dict, ignored: list, prefix_hits: int)."""
    lines = src.splitlines(keepends=True)
    applied, seen = {}, set()
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)(\s+#.*)?$", line.rstrip("\n"))
        if not m:
            continue
        name = m.group(1)
        if name in overrides and name not in seen:
            comment = m.group(3) or ""
            lines[i] = f"{name} = {overrides[name]!r}{comment}\n"
            applied[name] = overrides[name]
            seen.add(name)
    ignored = [k for k in overrides if k not in applied]
    out = "".join(lines)
    hits = 0
    if raw_prefix:
        out, hits = re.subn(r"run[AB]_\{RUN\}", f"{raw_prefix}_{{RUN}}", out)
    return out, applied, ignored, hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script")
    ap.add_argument("--date", required=True)
    ap.add_argument("--mouse", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--raw-prefix", choices=["runA", "runB"], default=None)
    ap.add_argument("--show", action="store_true", help="print rewrites, do not execute")
    ap.add_argument("script_args", nargs="*", help="args forwarded to the script (after --)")
    a = ap.parse_args(argv)

    script = Path(a.script).resolve()
    if not script.is_file():
        print(f"run_stage: no such script {script}", file=sys.stderr)
        return 2

    overrides = {"DATE": a.date, "MOUSE": a.mouse, "RUN": a.run}
    for kv in a.set:
        if "=" not in kv:
            ap.error(f"--set needs NAME=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = _literal(v.strip())

    src = script.read_text()
    patched, applied, ignored, hits = patch_source(src, overrides, a.raw_prefix)

    print(f"run_stage: {script.name}  {a.date}/{a.mouse}/{a.run}")
    for k, v in applied.items():
        print(f"  set {k} = {v!r}")
    if a.raw_prefix:
        print(f"  raw prefix -> {a.raw_prefix}_  ({hits} literal(s) rewritten)")
    if ignored:
        print(f"  (not defined in this script, ignored: {', '.join(ignored)})")
    sys.stdout.flush()

    if a.show:
        return 0

    code = compile(patched, str(script), "exec")
    sys.argv = [str(script)] + list(a.script_args)
    sys.path.insert(0, str(script.parent))
    g = {"__name__": "__main__", "__file__": str(script), "__builtins__": __builtins__}
    exec(code, g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
