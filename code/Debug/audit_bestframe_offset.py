#!/usr/bin/env python3
"""
Audit: did M1.5 (pre_segmentation_m1.5.py) use the same frame offset that
M1 (find_events_m1.py) used?

WHY THIS EXISTS
---------------
M1 trims SKIP_FIRST_SECONDS from the start of the recording, so the frame
indices written to preprocessed/event_groups.csv are relative to the TRIMMED
stack. M1.5 reads the UNTRIMMED raw 4D stack, so it has to add the same number
of frames back:

    skip_offset = int(M1_SKIP_SECONDS * FS_HZ)
    cs          = crop_start + skip_offset
    global_peak = cs + argmax(scores)

M1.5 stores that constant separately from M1, so the two can disagree. When
they do, the "best frames" are sampled from the wrong part of the recording and
nothing complains.

This script is READ-ONLY. It writes nothing and deletes nothing. It recovers
the offset that was actually used from the best-frame FILENAMES, which encode
both the global peak and the local frame index, and compares it against the
offset implied by the stacks on disk.

HOW THE OFFSET IS RECOVERED
---------------------------
For each event, filenames give global_peak, and event_groups.csv gives
crop_start and crop_end. Since

    global_peak = crop_start + skip_offset + peak_local
    0 <= peak_local < (crop_end - crop_start)

each event constrains skip_offset to a half-open interval. Intersecting the
intervals over all events of a run pins it down, usually exactly.

Ground truth for comparison, per pipeline-context.md:
    skip_frames = T(stack M1 read) - T(preprocessed/stack_voxel_norm_mean_sub)

Usage:
    python code/Debug/audit_bestframe_offset.py
    python code/Debug/audit_bestframe_offset.py --root /path/to/scape-data
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "scape-data"

BESTFRAME_RE = re.compile(
    r"bestframe_event_group_(?P<eid>\d+)_peak(?P<peak>\d+)_t(?P<tloc>\d+)_rank(?P<rank>\d+)_3d\.tif$"
)


def read_event_groups(path: Path) -> dict[int, tuple[int, int]]:
    """event_id -> (crop_start, crop_end)."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["event_id"])] = (int(row["crop_start"]), int(row["crop_end"]))
    return out


def tiff_n_frames(path: Path) -> int | None:
    """Number of time points, without loading pixel data."""
    try:
        import tifffile
    except ImportError:
        return None
    try:
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
        return int(shape[0]) if len(shape) >= 3 else None
    except Exception:
        return None


def infer_offset(best_dir: Path, events: dict[int, tuple[int, int]]):
    """Intersect per-event constraints. Returns (lo, hi, n_events) or None."""
    lo, hi, used = None, None, 0
    seen: set[int] = set()
    for p in sorted(best_dir.glob("bestframe_*_rank??_3d.tif")):
        m = BESTFRAME_RE.search(p.name)
        if not m:
            continue
        eid = int(m.group("eid"))
        if eid in seen:
            continue          # one constraint per event; ranks share the peak
        if eid not in events:
            continue
        seen.add(eid)
        peak = int(m.group("peak"))
        cs, ce = events[eid]
        span = max(ce - cs, 1)
        # peak = cs + off + peak_local, 0 <= peak_local < span
        e_hi = peak - cs               # peak_local = 0
        e_lo = peak - cs - span + 1    # peak_local = span-1
        lo = e_lo if lo is None else max(lo, e_lo)
        hi = e_hi if hi is None else min(hi, e_hi)
        used += 1
    if used == 0:
        return None
    return lo, hi, used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--fs", type=float, default=5.0,
                    help="volume rate assumed when reporting seconds")
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"error: no such directory: {args.root}")
        return 2

    print(f"scape-data root : {args.root}")
    print(f"assumed rate    : {args.fs} Hz (only affects the seconds column)\n")

    hdr = (f"{'dataset':<46} {'best':>5} {'ev':>4} "
           f"{'offset used':>13} {'expected':>9} {'verdict':<10}")
    print(hdr)
    print("-" * len(hdr))

    flagged, checked, unknown = [], 0, 0

    for eg in sorted(args.root.glob("*/*/*/preprocessed/event_groups.csv")):
        pre = eg.parent
        run_dir = pre.parent
        best = pre / "best_frames"
        label = "/".join(run_dir.parts[-3:])

        n_best = len(list(best.glob("bestframe_*_rank??_3d.tif"))) if best.is_dir() else 0
        if n_best == 0:
            continue

        try:
            events = read_event_groups(eg)
        except Exception as exc:
            print(f"{label:<46} {n_best:>5} {'?':>4}  unreadable event_groups.csv: {exc}")
            continue

        got = infer_offset(best, events)
        if got is None:
            print(f"{label:<46} {n_best:>5} {len(events):>4} "
                  f"{'no match':>13} {'-':>9} {'SKIP':<10}")
            continue
        lo, hi, used = got
        off_str = f"{lo}" if lo == hi else f"{lo}..{hi}"

        # Ground truth: frames dropped between the stack M1 read and its output.
        norm = pre / "stack_voxel_norm_mean_sub.tif"
        raw_clean = pre / "raw_clean.tif"
        raw_dir = run_dir / "raw"
        raw_orig = next((p for p in sorted(raw_dir.glob("*-reslice-bin.tif"))), None)
        src = raw_clean if raw_clean.exists() else raw_orig

        t_norm = tiff_n_frames(norm) if norm.exists() else None
        t_src = tiff_n_frames(src) if src and src.exists() else None
        expected = (t_src - t_norm) if (t_norm is not None and t_src is not None) else None

        if expected is None:
            verdict, exp_str = "UNKNOWN", "-"
            unknown += 1
        elif lo <= expected <= hi:
            verdict, exp_str = "ok", str(expected)
            checked += 1
        else:
            verdict, exp_str = "MISMATCH", str(expected)
            flagged.append((label, off_str, expected, lo, hi, used, n_best))
            checked += 1

        print(f"{label:<46} {n_best:>5} {used:>4} {off_str:>13} {exp_str:>9} {verdict:<10}")

    print()
    if flagged:
        print(f"{len(flagged)} run(s) MISMATCHED — best frames were sampled from the "
              f"wrong frames:\n")
        for label, off_str, expected, lo, hi, used, n_best in flagged:
            delta = (lo if lo == hi else lo) - expected
            print(f"  {label}")
            print(f"      offset actually used : {off_str} frames "
                  f"({(lo)/args.fs:.1f} s at {args.fs} Hz)")
            print(f"      offset M1 implies    : {expected} frames "
                  f"({expected/args.fs:.1f} s)")
            print(f"      error                : {delta:+d} frames "
                  f"({delta/args.fs:+.1f} s), from {used} events, {n_best} best-frame files")
        print("\nThese best frames only feed M2 seeding (BESTFRAME_MODE). Existing")
        print("labelmaps, curated masks and traces are NOT changed by this finding.")
        print("Re-running M1.5 would fix the intermediates but would only alter results")
        print("if M2 were also re-run, which would discard manual curation.")
    else:
        print("No offset mismatches found among runs that could be verified.")

    if unknown:
        print(f"\n{unknown} run(s) could not be verified (missing stack, or tifffile "
              f"unavailable). Their inferred offset is still printed above.")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
