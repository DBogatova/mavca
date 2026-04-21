#!/usr/bin/env python
"""
Static Reference Volume: max ΔF/F over time, masked to dendrites.

Produces a single 3D .tif where each dendrite voxel holds its peak ΔF/F
and everything outside masks is 0.  Memory-efficient: streams the 4D
stack in chunks rather than loading it all at once.

Usage:
    python code/Visual-STEP4/static_reference_volume.py
"""

from pathlib import Path
import numpy as np
import tifffile

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUN = "run7"

Y_CROP = 3  # crop bottom N rows of Y to match masks
SKIP_FIRST_SECONDS = 0.0  # drop first N seconds
FRAME_RATE = 5  # Hz
CHUNK_T = 50  # frames per chunk for memory efficiency

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
STACK_PATH = BASE / "overlays" / "chunk_01_0000-0545_dff.tif"
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"
OUTPUT_PATH = BASE / "overlays" / "static_masked_max_over_time.tif"

# Optional: restrict to specific masks (None = use all)
SELECTED_MASKS = None  # e.g. ["dend_000", "dend_003", "dend_010"]


def load_union_mask(folder, shape_zyx, selected=None):
    """Load mask TIFs and return boolean union (Z,Y,X)."""
    paths = sorted(Path(folder).glob("dend_*_labelmap.tif"))
    if not paths:
        raise FileNotFoundError(f"No masks in {folder}")

    if selected is not None:
        sel = set(selected)
        paths = [p for p in paths if p.stem.replace("_labelmap", "") in sel]
        print(f"  Using {len(paths)} selected masks")

    union = np.zeros(shape_zyx, dtype=bool)
    loaded = 0
    Z_t, Y_t, X_t = shape_zyx
    for p in paths:
        m = tifffile.imread(p) > 0
        mz, my, mx = m.shape
        # Handle Y mismatch (masks may be Y-cropped differently)
        if my < Y_t:
            m = np.pad(m, ((0,0),(0,Y_t-my),(0,0)), mode='constant')
        elif my > Y_t:
            m = m[:, :Y_t, :]
        if m.shape != shape_zyx:
            print(f"  [SKIP] {p.name}: shape {m.shape} != {shape_zyx}")
            continue
        union |= m
        loaded += 1
        del m

    print(f"  Loaded {loaded} masks, union covers {union.sum():,} voxels "
          f"({100 * union.mean():.2f}% of volume)")
    return union


def main():
    print(f"=== Static Reference Volume ===")
    print(f"  {DATE} | {MOUSE} | {RUN}\n")

    # --- Peek at stack shape ---
    print(f"Stack: {STACK_PATH.name}")
    store = tifffile.memmap(str(STACK_PATH), mode='r')
    T, Z, Y, X = store.shape
    print(f"  Shape: T={T}, Z={Z}, Y={Y}, X={X}")

    if Y_CROP > 0:
        Y_eff = Y - Y_CROP
        print(f"  Y-crop bottom {Y_CROP} → Y={Y_eff}")
    else:
        Y_eff = Y

    shape_zyx = (Z, Y_eff, X)

    # --- Load masks ---
    print("Loading masks...")
    union = load_union_mask(MASK_FOLDER, shape_zyx, selected=SELECTED_MASKS)

    # --- Compute temporal max in chunks (skip first seconds) ---
    skip_frames = int(SKIP_FIRST_SECONDS * FRAME_RATE) if SKIP_FIRST_SECONDS > 0 else 0
    t_start = skip_frames
    print(f"Computing temporal max (chunk-wise, skipping first {skip_frames} frames)...")
    max_vol = np.full(shape_zyx, -np.inf, dtype=np.float32)

    for t0 in range(t_start, T, CHUNK_T):
        t1 = min(t0 + CHUNK_T, T)
        chunk = np.asarray(store[t0:t1]).astype(np.float32)
        if Y_CROP > 0:
            chunk = chunk[:, :, :Y_eff, :]
        np.maximum(max_vol, chunk.max(axis=0), out=max_vol)
        del chunk
        print(f"  frames {t0:>5d}–{t1-1:<5d}  (max so far: {max_vol[union].max():.4f})")

    del store

    # --- Apply mask & normalize per dendrite ---
    print("Normalizing per mask...")
    out = np.zeros(shape_zyx, dtype=np.float32)
    mask_paths = sorted(Path(MASK_FOLDER).glob("dend_*_labelmap.tif"))
    if SELECTED_MASKS is not None:
        sel = set(SELECTED_MASKS)
        mask_paths = [p for p in mask_paths if p.stem.replace("_labelmap", "") in sel]
    for p in mask_paths:
        m = tifffile.imread(p) > 0
        mz, my, mx = m.shape
        if my < shape_zyx[1]:
            m = np.pad(m, ((0,0),(0,shape_zyx[1]-my),(0,0)), mode='constant')
        elif my > shape_zyx[1]:
            m = m[:, :shape_zyx[1], :]
        if m.shape != shape_zyx:
            continue
        vals = max_vol[m]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        if hi <= lo:
            hi = lo + 1e-6
        # Scale this mask's voxels to [0, 1]
        scaled = np.clip((max_vol - lo) / (hi - lo), 0, 1)
        out[m] = scaled[m]
        del m, vals, scaled
    del max_vol

    # --- Stats ---
    nz = out[out > 0]
    print(f"\nOutput shape: {out.shape}")
    print(f"  Non-zero voxels: {nz.size:,}")
    print(f"  Min / Max: {nz.min():.4f} / {nz.max():.4f}")
    print(f"  Percentiles (25/50/75/95): "
          f"{np.percentile(nz, 25):.4f} / {np.percentile(nz, 50):.4f} / "
          f"{np.percentile(nz, 75):.4f} / {np.percentile(nz, 95):.4f}")

    # --- Save ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(OUTPUT_PATH), out)
    print(f"\n✅ Saved: {OUTPUT_PATH}")

    # --- Show in Napari ---
    import napari
    v = napari.Viewer(ndisplay=3)
    v.add_image(out, name="max ΔF/F", scale=(3.9, 1.0, 1.2),
                colormap="green", rendering="attenuated_mip")
    v.scale_bar.visible = True
    v.scale_bar.unit = "µm"
    napari.run()


if __name__ == "__main__":
    main()
