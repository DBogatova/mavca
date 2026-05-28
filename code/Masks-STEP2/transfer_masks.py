#!/usr/bin/env python
"""
Transfer curated masks to a new run and find novel dendrites.

Workflow:
1. Load curated masks from a reference run (labelmaps_curated_dynamic/)
2. Load auto-detected masks from the new run (labelmaps/ or labelmaps_split/)
3. Match by Dice overlap — curated masks that overlap new detections are "transferred"
4. New detections with no match are "novel candidates" → save for quick review
5. Copy matched curated masks to the new run's labelmaps_curated_dynamic/

Usage:
    python transfer_masks.py

Then only curate the novel candidates (much fewer than starting from scratch).
"""

from pathlib import Path
import numpy as np
import tifffile
import shutil
import csv

# ===== CONFIG =====
DATE = "2026-05-12"
MOUSE = "rbp4_132_phpeb"

# Reference run (already curated)
REF_RUN = "run4"

# New run (same FOV, needs masks)
NEW_RUN = "run5"

# Matching threshold
DICE_THRESHOLD = 0.3  # masks with Dice >= this are considered the same dendrite

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE

REF_CURATED = BASE / REF_RUN / "labelmaps_curated_dynamic"
NEW_DETECTED = BASE / NEW_RUN / "labelmaps_split"
if not NEW_DETECTED.exists():
    NEW_DETECTED = BASE / NEW_RUN / "labelmaps"

NEW_CURATED = BASE / NEW_RUN / "labelmaps_curated_dynamic"
NEW_CURATED.mkdir(parents=True, exist_ok=True)

NOVEL_FOLDER = BASE / NEW_RUN / "labelmaps_novel_candidates"
NOVEL_FOLDER.mkdir(parents=True, exist_ok=True)


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return 2.0 * inter / total if total > 0 else 0.0


def load_masks(folder):
    """Load all masks from a folder, return dict of name -> binary 3D array."""
    masks = {}
    for p in sorted(folder.glob("dend_*_labelmap.tif")):
        name = p.stem.replace("_labelmap", "")
        masks[name] = tifffile.imread(p).astype(bool)
    return masks


def main():
    print(f"Reference (curated): {REF_CURATED}")
    print(f"New (auto-detected): {NEW_DETECTED}")
    print()

    ref_masks = load_masks(REF_CURATED)
    new_masks = load_masks(NEW_DETECTED)
    print(f"Reference curated masks: {len(ref_masks)}")
    print(f"New auto-detected masks: {len(new_masks)}")

    if not ref_masks:
        raise FileNotFoundError(f"No curated masks in {REF_CURATED}")
    if not new_masks:
        raise FileNotFoundError(f"No detected masks in {NEW_DETECTED}")

    # Ensure shapes match (pad if needed)
    ref_shape = next(iter(ref_masks.values())).shape
    new_shape = next(iter(new_masks.values())).shape
    if ref_shape != new_shape:
        print(f"  WARNING: shape mismatch ref={ref_shape} vs new={new_shape}")
        print(f"  Will pad/crop to match")

    # Match: for each new mask, find best-matching reference mask
    new_matched = {}  # new_name -> (ref_name, dice_score)
    ref_matched = set()

    for new_name, new_m in new_masks.items():
        best_ref = None
        best_dice = 0
        for ref_name, ref_m in ref_masks.items():
            # Handle shape mismatch
            if ref_m.shape != new_m.shape:
                min_shape = tuple(min(r, n) for r, n in zip(ref_m.shape, new_m.shape))
                d = dice(ref_m[:min_shape[0], :min_shape[1], :min_shape[2]],
                         new_m[:min_shape[0], :min_shape[1], :min_shape[2]])
            else:
                d = dice(ref_m, new_m)
            if d > best_dice:
                best_dice = d
                best_ref = ref_name
        if best_dice >= DICE_THRESHOLD:
            new_matched[new_name] = (best_ref, best_dice)
            ref_matched.add(best_ref)

    # Results
    novel_new = [n for n in new_masks if n not in new_matched]
    unmatched_ref = [r for r in ref_masks if r not in ref_matched]

    print(f"\n=== RESULTS ===")
    print(f"Matched (transferred):  {len(new_matched)} masks")
    print(f"Novel in new run:       {len(novel_new)} masks (need review)")
    print(f"Lost from reference:    {len(unmatched_ref)} masks (not active in new run)")

    # Copy curated reference masks to new run
    print(f"\nTransferring {len(ref_masks)} curated masks to {NEW_CURATED.name}/...")
    for ref_name in ref_masks:
        src = REF_CURATED / f"{ref_name}_labelmap.tif"
        dst = NEW_CURATED / f"{ref_name}_labelmap.tif"
        if src.exists():
            shutil.copy2(src, dst)

    # Save novel candidates separately for quick review
    print(f"Saving {len(novel_new)} novel candidates to {NOVEL_FOLDER.name}/...")
    for name in novel_new:
        src = NEW_DETECTED / f"{name}_labelmap.tif"
        dst = NOVEL_FOLDER / f"{name}_labelmap.tif"
        if src.exists():
            shutil.copy2(src, dst)

    # Write summary CSV
    summary_path = BASE / NEW_RUN / "mask_transfer_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_mask", "status", "matched_ref", "dice"])
        for name in sorted(new_masks.keys()):
            if name in new_matched:
                ref_name, d = new_matched[name]
                w.writerow([name, "matched", ref_name, f"{d:.3f}"])
            else:
                w.writerow([name, "novel", "", ""])
        for name in unmatched_ref:
            w.writerow(["", "lost_from_ref", name, ""])

    print(f"\nSummary: {summary_path}")
    print(f"\nNext steps:")
    print(f"  1. Review novel candidates in {NOVEL_FOLDER.name}/ (only {len(novel_new)} masks)")
    print(f"  2. Keep good ones → copy to {NEW_CURATED.name}/")
    print(f"  3. Run M4 (save_traces_m4.py) on the new run")


if __name__ == "__main__":
    main()
