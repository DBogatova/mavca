#!/usr/bin/env python
"""
Merge dendrite branches in-place for a shared-FOV run group, keeping the runs
byte-identical.

Procedure (matches the run1 merge convention):
  1. Back up each run's labelmaps_curated_dynamic -> ..._MERGEBACKUP_<ts>.
  2. Merge each group on SOURCE_RUN (keep first dend as representative, union
     the rest), renumber dend_000.. sequentially (uint16, value = index+1),
     write merge_log.csv (new_name, rep, merged, group_size, volume, timestamp).
  3. Substitute the merged dend_*.tif + merge_log.csv into the TARGET runs so all
     runs share byte-identical masks.
  4. Verify byte-identity across all runs.

GROUPS were computed once from the POOLED run5/6/7 traces (1800 frames) with the
find_dendrite_branches thresholds (spatial Jaccard > 0.05, temporal corr > 0.3
over co-active frames), so the same grouping is valid for all three shared-FOV runs.
"""

import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

# ===== CONFIG =====
ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data/2026-04-16/rbp4_132_phpeb")
SOURCE_RUN = "run7"
TARGET_RUNS = ["run5", "run6"]
SUBFOLDER = "labelmaps_curated_dynamic"

# First name in each group is kept (representative); the rest are merged into it.
GROUPS = [
    ["dend_016", "dend_079"],
    ["dend_021", "dend_148", "dend_155"],
    ["dend_023", "dend_087"],
    ["dend_026", "dend_033", "dend_037", "dend_052"],
    ["dend_049", "dend_059"],
    ["dend_051", "dend_137"],
    ["dend_091", "dend_138"],
    ["dend_096", "dend_149"],
    ["dend_100", "dend_133", "dend_153"],
    ["dend_113", "dend_124"],
]


def folder_md5(folder: Path):
    """md5 of each dend_*_labelmap.tif (name -> hash)."""
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob("dend_*_labelmap.tif"))}


def backup(folder: Path, ts: str) -> Path:
    dest = folder.parent / f"{folder.name}_MERGEBACKUP_{ts}"
    shutil.copytree(folder, dest)
    return dest


def merge_source(folder: Path, ts: str):
    """Merge GROUPS in `folder`, renumber sequentially, write merge_log.csv."""
    mask_files = sorted(folder.glob("dend_*_labelmap.tif"))
    masks = {p.stem.replace("_labelmap", ""): tifffile.imread(p).astype(bool)
             for p in mask_files}
    all_names = list(masks)

    merged_away = {n for g in GROUPS for n in g[1:]}
    rep_to_group = {g[0]: g for g in GROUPS}

    # Kept masks = singletons + group reps, ordered by original index
    kept = sorted([n for n in all_names if n not in merged_away],
                  key=lambda n: int(n.split("_")[1]))

    # Build final mask arrays (union for reps)
    final = {}
    for n in kept:
        if n in rep_to_group:
            union = np.zeros_like(masks[n])
            for member in rep_to_group[n]:
                union |= masks[member]
            final[n] = union
        else:
            final[n] = masks[n]

    # new sequential name for each kept original
    new_name_of = {n: f"dend_{i:03d}" for i, n in enumerate(kept)}

    # Rewrite folder: delete old dend tifs, write renumbered ones (value = i+1)
    for p in mask_files:
        p.unlink()
    for i, n in enumerate(kept):
        out = (final[n] > 0).astype(np.uint16) * (i + 1)
        tifffile.imwrite(str(folder / f"dend_{i:03d}_labelmap.tif"), out)

    # merge_log.csv
    log_rows = []
    for rep, group in rep_to_group.items():
        vol = int(np.sum(final[rep]))
        log_rows.append({
            "new_name": new_name_of[rep],
            "rep": rep,
            "merged": ";".join(group),
            "group_size": len(group),
            "volume": vol,
            "timestamp": ts,
        })
    log_rows.sort(key=lambda r: r["new_name"])
    log_path = folder / "merge_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["new_name", "rep", "merged",
                                          "group_size", "volume", "timestamp"])
        if write_header:
            w.writeheader()
        w.writerows(log_rows)

    print(f"  merged {len(GROUPS)} groups; {len(all_names)} -> {len(kept)} masks")
    return len(kept)


def substitute(src: Path, dst: Path):
    """Replace dst's dend_*.tif with src's, and copy merge_log.csv."""
    for p in dst.glob("dend_*_labelmap.tif"):
        p.unlink()
    for p in sorted(src.glob("dend_*_labelmap.tif")):
        shutil.copy2(p, dst / p.name)
    if (src / "merge_log.csv").exists():
        shutil.copy2(src / "merge_log.csv", dst / "merge_log.csv")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = ROOT / SOURCE_RUN / SUBFOLDER
    tgts = [ROOT / r / SUBFOLDER for r in TARGET_RUNS]

    print(f"=== Merge branches (shared FOV): source={SOURCE_RUN}, "
          f"targets={TARGET_RUNS} ===")

    # Pre-check identity
    h0 = folder_md5(src)
    for t, r in zip(tgts, TARGET_RUNS):
        assert folder_md5(t) == h0, f"{r} not identical to {SOURCE_RUN} pre-merge!"
    print(f"  pre-merge: all runs identical ({len(h0)} masks)")

    # 1. backups
    print("Backing up...")
    for folder in [src] + tgts:
        b = backup(folder, ts)
        print(f"  {b}")

    # 2. merge on source
    print(f"Merging on {SOURCE_RUN}...")
    n_final = merge_source(src, ts)

    # 3. substitute into targets
    print("Substituting into targets...")
    for t, r in zip(tgts, TARGET_RUNS):
        substitute(src, t)
        print(f"  {r} updated")

    # 4. verify byte-identity
    h_src = folder_md5(src)
    ok = True
    for t, r in zip(tgts, TARGET_RUNS):
        if folder_md5(t) != h_src:
            ok = False
            print(f"  ✗ {r} differs from {SOURCE_RUN}!")
    print(f"\nFinal: {n_final} masks/run | byte-identical across runs: {ok} "
          f"| backups tagged _{ts}")


if __name__ == "__main__":
    main()
