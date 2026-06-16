#!/usr/bin/env python
"""Run M4 (save_traces_m4) for runs that have curated masks but no trace.
Overrides the script's hardcoded config per run (raw prefix, per-dataset skip).
Usage: python code/Traces-STEP3/run_m4_batch.py DATE MOUSE RUN PREFIX SKIP
"""
import sys, importlib.util
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("m4", HERE/"save_traces_m4.py")
m4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m4)

def run_one(date, mouse, run, prefix, skip, rawname=None, mask_run=None):
    base = m4.PROJECT_ROOT/"scape-data"/date/mouse/run
    m4.DATE, m4.MOUSE, m4.RUN, m4.BASE = date, mouse, run, base
    m4.RAW_CLEAN_PATH = base/"preprocessed"/"raw_clean.tif"
    m4.RAW_ORIG_PATH  = base/"raw"/(rawname or f"{prefix}_{run}_{mouse}-reslice-bin.tif")
    m4.RAW_STACK_PATH = m4.RAW_CLEAN_PATH if m4.RAW_CLEAN_PATH.exists() else m4.RAW_ORIG_PATH
    m4.MASK_SOURCE_RUN = mask_run
    m4.MASK_FOLDER = (m4.PROJECT_ROOT/"scape-data"/date/mouse/mask_run/"labelmaps_curated_dynamic"
                      if mask_run else base/"labelmaps_curated_dynamic")
    m4.TRACE_FOLDER = base/"traces"; m4.TRACE_FOLDER.mkdir(exist_ok=True)
    m4.TRACE_PKL = m4.TRACE_FOLDER/"dff_traces_curated_bgsub.pkl"
    m4.TRACE_CSV = m4.TRACE_FOLDER/"dff_traces_curated_bgsub.csv"
    m4.PREVIEW_FOLDER = base/"trace_previews_curated"; m4.PREVIEW_FOLDER.mkdir(exist_ok=True)
    m4.SKIP_FIRST_SECONDS = float(skip)
    m4.PLOT_ALL_TRACES = False; m4.SELECTED_NAMES = []
    assert m4.RAW_STACK_PATH.exists(), f"raw missing: {m4.RAW_STACK_PATH}"
    print(f"\n##### M4 {date}/{mouse}/{run}  raw={m4.RAW_STACK_PATH.name} masks={m4.MASK_FOLDER.parts[-2]} skip={skip}s #####")
    m4.main()

if __name__ == "__main__":
    a = sys.argv[1:]
    raw = a[5] if len(a) > 5 and a[5] not in ('-', '') else None
    mr = a[6] if len(a) > 6 and a[6] not in ('-', '') else None
    run_one(a[0], a[1], a[2], a[3], a[4], raw, mr)
