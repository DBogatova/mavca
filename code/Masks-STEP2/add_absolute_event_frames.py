#!/usr/bin/env python
"""
Add whole-movie (absolute) event start/end to a mask manifest.

The M2 masks_manifest.csv stores event_t_start/event_t_end as CROP-LOCAL frames
(always 0..crop_len-1) of the event_group_XXXX.tif a mask was detected in. This
script reconstructs each event group's absolute frame window from active_frames.npy
(M1 detection) and writes a new masks_manifest_absolute.csv with absolute columns.

This script reconstructs each event group's absolute frame window from active_frames.npy
(M1 detection) and writes a new masks_manifest_absolute.csv with absolute columns in the
ANALYSIS TIMELINE (post-skip): all columns are suffixed *_postskip. This is the timeline
used by active_frames, the overlay/3D movie, and the cropped trace plots (M1 drops the
first SKIP_FIRST_SECONDS before detection). To get raw-acquisition frames instead, add
skip_frames (= SKIP_FIRST_SECONDS * FS_HZ).

Usage: python code/Masks-STEP2/add_absolute_event_frames.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ===== CONFIG =====
DATE = "2026-04-16"
MOUSE = "rbp4_132_phpeb"
RUN = "run7"

FS_HZ = 5.0
SKIP_FIRST_SECONDS = 12.0     # M1 drop before detection (per steering: 0416 = 12 s)
CROP_RADIUS = 5               # M1 frames of padding each side of an event
MAX_FRAME_GAP = 2             # M1 consecutive-frame grouping gap

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
# Manifest to adjust (override with argv[1], e.g. masks_manifest_split.csv)
MANIFEST_NAME = sys.argv[1] if len(sys.argv) > 1 else "masks_manifest.csv"
MANIFEST = BASE / MANIFEST_NAME
ACTIVE = BASE / "preprocessed" / "active_frames.npy"
CROPS = BASE / "preprocessed" / "event_crops"
OUT = BASE / MANIFEST_NAME.replace(".csv", "_absolute.csv")


def group_consecutive(frames, gap):
    groups, g = [], [int(frames[0])]
    for f in frames[1:]:
        f = int(f)
        if f - g[-1] <= gap:
            g.append(f)
        else:
            groups.append(g); g = [f]
    groups.append(g)
    return groups


def main():
    man = pd.read_csv(MANIFEST)
    af = np.load(ACTIVE)
    skip_frames = int(round(SKIP_FIRST_SECONDS * FS_HZ))
    groups = group_consecutive(af, MAX_FRAME_GAP)

    n_files = len(list(CROPS.glob("event_group_*.tif")))
    n_src = man["source_event_file"].nunique()
    print(f"{DATE}/{MOUSE}/{RUN} [{MANIFEST_NAME}]: {len(groups)} reconstructed groups | "
          f"{n_files} crop files | {n_src} manifest source files | skip={skip_frames}f")
    assert len(groups) == n_files, (
        "group count != crop-file count — reconstruction params (CROP_RADIUS/GAP/SKIP) may be wrong")

    # event index -> windows (post-skip analysis frames)
    info = {}
    for i, g in enumerate(groups):
        cs = max(g[0] - CROP_RADIUS, 0)
        info[f"event_group_{i:04d}.tif"] = {"evt_start": g[0], "evt_end": g[-1],
                                            "crop_start": cs}

    missing = set(man["source_event_file"]) - set(info)
    assert not missing, f"manifest references unknown event groups: {sorted(missing)[:5]}"

    evt_starts, evt_ends, crop_starts, crop_ends = [], [], [], []
    for _, r in man.iterrows():
        d = info[r["source_event_file"]]
        cs = d["crop_start"]
        evt_starts.append(d["evt_start"])                  # detected event window (postskip)
        evt_ends.append(d["evt_end"])
        crop_starts.append(cs + int(r["event_t_start"]))   # full crop window (postskip)
        crop_ends.append(cs + int(r["event_t_end"]))

    # Skip tag encoded in the column names so the post-skip reference isn't lost.
    skip_tag = (f"{int(SKIP_FIRST_SECONDS)}s" if float(SKIP_FIRST_SECONDS).is_integer()
                else f"{SKIP_FIRST_SECONDS:g}s")
    suf = f"postskip_{skip_tag}"
    c_es, c_ee = f"event_t_start_{suf}", f"event_t_end_{suf}"
    c_cs, c_ce = f"crop_start_f_{suf}", f"crop_end_f_{suf}"

    # Replace the crop-local event_t_start/end with the absolute post-skip DETECTED
    # EVENT window (real activity); the column name carries the skipped seconds.
    out = man.rename(columns={"event_t_start": c_es, "event_t_end": c_ee})
    out[c_es] = evt_starts
    out[c_ee] = evt_ends
    # Also keep the full padded-crop window (the direct re-reference of the old span)
    out[c_cs] = crop_starts
    out[c_ce] = crop_ends

    out.to_csv(OUT, index=False)
    print(f"✅ wrote {OUT}  ({len(out)} masks)")
    cols = ["dend_id", "source_event_file", c_es, c_ee, c_cs, c_ce]
    print(out[cols].drop_duplicates("source_event_file").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
