#!/usr/bin/env python
"""
Module 2r — Rescue missed dendrites after auto_mask_m2.

Workflow:
  1) Pick an event crop that has missed dendrites
  2) Napari opens showing:
     - temporal-max MIP of the event crop (background)
     - existing masks from that event (colored overlays)
     - a "seeds" Shapes layer for you to draw on
  3) Draw lines marking missed dendrites (avoid existing masks)
  4) Press Ctrl+G to generate 3D masks from those seeds
  5) Review generated masks in the viewer
  6) Press Ctrl+S to save and append to existing M2 output

Usage:
    python code/Masks-STEP2/rescue_masks_m2r.py
"""

from pathlib import Path
import numpy as np
import tifffile
import napari
import csv
import cv2
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops
from skimage.draw import line as draw_line

# ===== CONFIG =====
DATE = "2026-03-31"
MOUSE = "rbp4_132_phpeb"
RUN = "run8"

VOXEL_SIZE = (3.9, 1.0, 1.2)  # Z, Y, X µm
VOXEL_VOL = float(np.prod(VOXEL_SIZE))

# Event crop to rescue from
EVENT_FILE = "event_group_0012.tif"

# Seed line width (pixels around each drawn line)
SEED_WIDTH = 5

# Thresholding on enhanced crop
RESCUE_PERCENTILE = 97.0

# Cleanup
SLICE_CLOSE_K = 5
SLICE_MIN_PIX = 5
MIN_VOL = 4000.0  # µm³

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
EVENT_CROPS = BASE / "preprocessed" / "event_crops"

# Try guided output first, fall back to standard
OUT_LABELS = BASE / "labelmaps_guided"
if not OUT_LABELS.exists():
    OUT_LABELS = BASE / "labelmaps"
OUT_BGS = BASE / "labelmap_backgrounds_guided"
if not OUT_BGS.exists():
    OUT_BGS = BASE / "labelmap_backgrounds"
MANIFEST = BASE / "masks_manifest_guided.csv"
if not MANIFEST.exists():
    MANIFEST = BASE / "masks_manifest.csv"


def enhance_crop(stack):
    """Same enhancement as auto_mask_m2."""
    sm = gaussian_filter(stack, sigma=(0.0, 0.5, 1.0, 1.0))
    bg = gaussian_filter(sm, sigma=(0.0, 4.0, 8.0, 8.0))
    enh = sm - bg
    enh[enh < 0] = 0.0
    vmax = np.nanpercentile(enh, 99.9)
    if vmax > 0:
        enh /= vmax
    return enh.astype(np.float32)


def lines_to_seed_mask(shapes_data, Y, X, width):
    """Convert napari Shapes lines to a 2D binary seed mask (Y, X)."""
    mask = np.zeros((Y, X), dtype=bool)
    for coords in shapes_data:
        # coords is Nx2 array of (row, col) = (y, x)
        pts = np.array(coords, dtype=int)
        for i in range(len(pts) - 1):
            rr, cc = draw_line(pts[i, 0], pts[i, 1], pts[i+1, 0], pts[i+1, 1])
            valid = (rr >= 0) & (rr < Y) & (cc >= 0) & (cc < X)
            mask[rr[valid], cc[valid]] = True
    # Dilate to width
    if width > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        mask = cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)
    return mask


def grow_masks_from_seeds(enh, seed_2d):
    """
    Given enhanced 4D crop and 2D seed mask, produce list of 3D masks.
    seed_2d is (Y,X), extruded through all Z.
    """
    T, Z, Y, X = enh.shape
    # Temporal max of enhanced
    tmax = enh.max(axis=0)  # (Z, Y, X)

    # Threshold
    thresh = np.percentile(tmax[tmax > 0], RESCUE_PERCENTILE)
    cand = tmax > thresh

    # Extrude seed through Z
    seed_3d = np.zeros((Z, Y, X), dtype=bool)
    for z in range(Z):
        seed_3d[z] = seed_2d

    # Keep only candidate voxels connected to seeds
    from skimage.morphology import label as sk_label
    lbl = sk_label(cand)
    touching = np.unique(lbl[seed_3d & cand])
    touching = touching[touching > 0]
    if len(touching) == 0:
        print("  ⚠️  No candidate voxels overlap seeds. Try lower RESCUE_PERCENTILE.")
        return []
    grown = np.isin(lbl, touching)

    # Per-slice cleanup
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (SLICE_CLOSE_K, SLICE_CLOSE_K))
    for z in range(Z):
        s = grown[z].astype(np.uint8)
        s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, se_close)
        grown[z] = s > 0

    # 3D connected components → separate masks
    lbl3d = sk_label(grown)
    masks = []
    for r in regionprops(lbl3d):
        vol_um3 = r.area * VOXEL_VOL
        if vol_um3 < MIN_VOL:
            continue
        m = (lbl3d == r.label).astype(np.uint8)
        masks.append(m)

    return masks


def load_existing_masks_for_event(event_file):
    """Load existing masks that came from this event (via manifest)."""
    if not MANIFEST.exists():
        return [], []
    with open(MANIFEST, "r") as f:
        rows = list(csv.DictReader(f))

    masks, names = [], []
    for row in rows:
        if row["source_event_file"] == event_file:
            p = Path(row["labelmap_path"])
            if p.exists():
                m = tifffile.imread(p) > 0
                masks.append(m.astype(np.uint8))
                names.append(f"dend_{int(row['dend_id']):03d}")
    return masks, names


def next_dend_id():
    """Find the next available dend_id from the manifest."""
    if not MANIFEST.exists():
        return 0
    with open(MANIFEST, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    return max(int(r["dend_id"]) for r in rows) + 1


def main():
    event_path = EVENT_CROPS / EVENT_FILE
    if not event_path.exists():
        print(f"Event crop not found: {event_path}")
        return

    print(f"Loading event crop: {EVENT_FILE}")
    crop = tifffile.imread(str(event_path)).astype(np.float32)
    T, Z, Y, X = crop.shape
    print(f"  Shape: T={T}, Z={Z}, Y={Y}, X={X}")

    # Temporal max MIP for background
    tmax_mip = crop.max(axis=0).max(axis=0)  # (Y, X)

    # Load existing masks from this event
    existing_masks, existing_names = load_existing_masks_for_event(EVENT_FILE)
    print(f"  Existing masks from this event: {len(existing_masks)}")

    # Build combined existing mask MIP for overlay
    if existing_masks:
        combined = np.zeros((Y, X), dtype=np.int32)
        for i, m in enumerate(existing_masks):
            mip = m.max(axis=0)  # Z-MIP
            # Handle Y mismatch
            my = mip.shape[0]
            if my > Y:
                mip = mip[:Y, :X]
            elif my < Y:
                tmp = np.zeros((Y, X), dtype=mip.dtype)
                tmp[:my, :X] = mip[:, :X]
                mip = tmp
            combined[mip > 0] = i + 1

    # Enhance for later mask generation
    print("  Enhancing crop...")
    enh = enhance_crop(crop)

    # ---- Napari ----
    v = napari.Viewer(title=f"Rescue — {EVENT_FILE}")

    # Background MIP
    lo, hi = np.percentile(tmax_mip, 2), np.percentile(tmax_mip, 99.5)
    v.add_image(tmax_mip, name="event MIP", colormap="gray",
                contrast_limits=(lo, hi))

    # Existing masks overlay
    if existing_masks:
        v.add_labels(combined, name="existing masks", opacity=0.4)
        for name in existing_names:
            print(f"    {name}")

    # Seeds layer — draw lines here
    v.add_shapes(name="seeds", shape_type="path",
                 edge_color="lime", edge_width=3)

    # State
    rescued = {"masks": [], "saved": False}

    @v.bind_key("Control-g")
    def _generate(viewer):
        shapes_layer = v.layers["seeds"]
        if len(shapes_layer.data) == 0:
            print("No seeds drawn. Draw lines on the 'seeds' layer first.")
            return

        print(f"Generating masks from {len(shapes_layer.data)} seed lines...")
        seed_2d = lines_to_seed_mask(shapes_layer.data, Y, X, SEED_WIDTH)
        print(f"  Seed pixels: {seed_2d.sum()}")

        masks = grow_masks_from_seeds(enh, seed_2d)
        print(f"  Generated {len(masks)} masks")

        # Remove old rescue layers
        for layer in list(v.layers):
            if layer.name.startswith("rescue_"):
                v.layers.remove(layer)

        # Show in viewer
        for i, m in enumerate(masks):
            mip = m.max(axis=0)
            my = mip.shape[0]
            if my > Y:
                mip = mip[:Y, :]
            elif my < Y:
                tmp = np.zeros((Y, X), dtype=mip.dtype)
                tmp[:my, :] = mip
                mip = tmp
            v.add_labels(mip.astype(np.int32) * (i + 1),
                         name=f"rescue_{i:02d}",
                         opacity=0.5)
            vol = m.sum() * VOXEL_VOL
            print(f"    rescue_{i:02d}: {vol:.0f} µm³")

        rescued["masks"] = masks

    @v.bind_key("Control-s")
    def _save(viewer):
        if not rescued["masks"]:
            print("No rescued masks. Press Ctrl+G first.")
            return
        if rescued["saved"]:
            print("Already saved.")
            return

        start_id = next_dend_id()
        print(f"Saving {len(rescued['masks'])} rescued masks starting at dend_{start_id:03d}...")

        # Read existing manifest
        existing_rows = []
        if MANIFEST.exists():
            with open(MANIFEST, "r") as f:
                existing_rows = list(csv.DictReader(f))

        new_rows = []
        for i, m in enumerate(rescued["masks"]):
            did = start_id + i
            mask_path = OUT_LABELS / f"dend_{did:03d}_labelmap.tif"
            tifffile.imwrite(str(mask_path), (m * (did + 1)).astype(np.uint16))

            # 2D background
            bg2d = crop.max(axis=0).max(axis=0).astype(np.float16)  # (Y,X)
            bg_path = OUT_BGS / f"dend_{did:03d}_background_2dMIP.tif"
            tifffile.imwrite(str(bg_path), bg2d)

            new_rows.append({
                "dend_id": did,
                "labelmap_path": str(mask_path),
                "background_path": str(bg_path),
                "source_event_file": EVENT_FILE,
                "event_t_start": 0,
                "event_t_end": T,
                "voxel_size_z": VOXEL_SIZE[0],
                "voxel_size_y": VOXEL_SIZE[1],
                "voxel_size_x": VOXEL_SIZE[2],
                "volume_um3": float(m.sum() * VOXEL_VOL),
            })
            print(f"    Saved dend_{did:03d} ({new_rows[-1]['volume_um3']:.0f} µm³)")

        # Append to manifest
        all_rows = existing_rows + new_rows
        fieldnames = list(all_rows[0].keys())
        with open(MANIFEST, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

        rescued["saved"] = True
        print(f"✅ Manifest updated: {MANIFEST}")

    print("\n=== INSTRUCTIONS ===")
    print("1. Draw lines on 'seeds' layer where dendrites are missing")
    print("   (existing masks shown in color overlay)")
    print("2. Ctrl+G = generate masks from seeds")
    print("3. Ctrl+S = save rescued masks")
    napari.run()


if __name__ == "__main__":
    main()
