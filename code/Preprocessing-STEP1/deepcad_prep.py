#!/usr/bin/env python
"""Glue for DeepCAD-RT (volumetric SCAPE data).

DeepCAD-RT denoises 2D+time movies (T,Y,X). Our data is 4D (T,Z,Y,X), so we
denoise plane-by-plane: split into per-Z movies, train ONE model across all
planes, denoise each, then restack. Plane-wise never couples signal across Z.

Usage:
  python deepcad_prep.py split RAW_4D.tif  PLANES_DIR   # before DeepCAD
  python deepcad_prep.py merge DENOISED_DIR OUT_4D.tif  # after DeepCAD
"""
import sys, os, glob, re
import numpy as np
import tifffile

def split(path, out_dir, z_axis=1):
    stack = tifffile.imread(path)
    assert stack.ndim == 4, f"expected 4D (T,Z,Y,X), got {stack.shape}"
    stack = np.moveaxis(stack, z_axis, 1)  # ensure (T,Z,Y,X)
    os.makedirs(out_dir, exist_ok=True)
    T, Z = stack.shape[0], stack.shape[1]
    for z in range(Z):
        tifffile.imwrite(os.path.join(out_dir, f"plane_{z:03d}.tif"), stack[:, z])
    print(f"wrote {Z} planes ({T} frames each) -> {out_dir}")

def merge(in_dir, out_path):
    paths = glob.glob(os.path.join(in_dir, "*plane_*.tif"))
    assert paths, f"no *plane_*.tif in {in_dir}"
    paths.sort(key=lambda p: int(re.search(r"plane_(\d+)", os.path.basename(p)).group(1)))
    stack = np.stack([tifffile.imread(p) for p in paths], axis=1)  # (T,Z,Y,X)
    tifffile.imwrite(out_path, stack)
    print(f"merged {len(paths)} planes -> {out_path} {stack.shape}")

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "split":
        split(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3])
    else:
        sys.exit("use: split RAW_4D.tif PLANES_DIR | merge DENOISED_DIR OUT_4D.tif")
