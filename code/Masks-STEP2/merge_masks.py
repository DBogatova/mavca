#!/usr/bin/env python
"""
Merge specified mask groups in labelmaps_curated_dynamic.

Each group is merged into a single mask named after the first dendrite.
Remaining masks are renumbered sequentially. All actions are logged.

Usage:
    python code/Masks-STEP2/merge_masks.py

Configure MERGE_GROUPS below with the groups you want to merge.
Set groups to [] to skip merging and only renumber.
"""

import numpy as np
import tifffile
import csv
from pathlib import Path
from datetime import datetime

# ===== CONFIG =====
DATE = "2026-05-12"
MOUSE = "rbp4_132_phpeb"
RUN = "run5"

# Each sublist = one group to merge. First name is kept, rest are merged into it.
# Leave empty [] to skip.
MERGE_GROUPS = [
    ["dend_021", "dend_044"],
    ["dend_005", "dend_071"],
    ["dend_033", "dend_073"],
    ["dend_023", "dend_024", "dend_045", "dend_064"],
    ["dend_006", "dend_026"],
    ["dend_004", "dend_060"],
]

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
LOG_PATH = MASK_FOLDER / "merge_log.csv"


def main():
    if not MASK_FOLDER.exists():
        print(f"Mask folder not found: {MASK_FOLDER}")
        return

    print(f"=== Merge Masks: {DATE}/{MOUSE}/{RUN} ===")
    print(f"  Folder: {MASK_FOLDER}")
    print(f"  Groups to merge: {len(MERGE_GROUPS)}\n")

    merge_log = []

    for group in MERGE_GROUPS:
        if len(group) < 2:
            continue
        keep_name = group[0]
        to_merge = group[1:]

        keep_path = MASK_FOLDER / f"{keep_name}_labelmap.tif"
        if not keep_path.exists():
            print(f"  SKIP: {keep_name} not found")
            continue

        merged = tifffile.imread(keep_path).astype(bool)

        for name in to_merge:
            path = MASK_FOLDER / f"{name}_labelmap.tif"
            if not path.exists():
                print(f"    WARNING: {name} not found, skipping")
                continue
            m = tifffile.imread(path).astype(bool)
            # Handle shape mismatch
            if m.shape != merged.shape:
                target = tuple(max(a, b) for a, b in zip(m.shape, merged.shape))
                if merged.shape != target:
                    new = np.zeros(target, dtype=bool)
                    new[:merged.shape[0], :merged.shape[1], :merged.shape[2]] = merged
                    merged = new
                if m.shape != target:
                    new = np.zeros(target, dtype=bool)
                    new[:m.shape[0], :m.shape[1], :m.shape[2]] = m
                    m = new
            merged |= m
            path.unlink()

        # Save merged
        label_val = int(keep_name.split("_")[1]) + 1
        tifffile.imwrite(str(keep_path), (merged.astype(np.uint16) * label_val))

        merge_log.append({
            "kept": keep_name,
            "merged_into_it": "+".join(to_merge),
            "group_size": len(group),
            "merged_volume": int(merged.sum()),
            "timestamp": datetime.now().isoformat(),
        })
        print(f"  {keep_name} ← {to_merge} (vol={merged.sum()})")

    # Renumber remaining masks sequentially
    all_masks = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    print(f"\nRenumbering {len(all_masks)} masks...")
    temp = []
    for p in all_masks:
        temp.append((p.name, tifffile.imread(p)))
        p.unlink()
    for i, (old_name, m) in enumerate(temp):
        new_name = f"dend_{i:03d}_labelmap.tif"
        binary = (m > 0).astype(np.uint16)
        tifffile.imwrite(str(MASK_FOLDER / new_name), binary * (i + 1))

    print(f"Done: {len(temp)} masks (dend_000 to dend_{len(temp)-1:03d})")

    # Write merge log
    if merge_log:
        write_header = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kept", "merged_into_it", "group_size", "merged_volume", "timestamp"])
            if write_header:
                w.writeheader()
            w.writerows(merge_log)
        print(f"\n✅ Logged {len(merge_log)} merges to {LOG_PATH}")
    else:
        print("\nNo merges performed (MERGE_GROUPS is empty).")


if __name__ == "__main__":
    main()
