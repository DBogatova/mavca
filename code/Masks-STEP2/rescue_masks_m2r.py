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
DATE = "2026-05-12"
MOUSE = "rbp4_132_phpeb"
RUN = "run9"

VOXEL_SIZE = (3.9, 1.0, 1.2)  # Z, Y, X µm
VOXEL_VOL = float(np.prod(VOXEL_SIZE))

# Event crops to rescue from (iterate one by one)
EVENT_FILES = [
    "event_group_0000.tif",
    "event_group_0001.tif",
    "event_group_0002.tif",
    "event_group_0003.tif",
    "event_group_0005.tif",
    "event_group_0007.tif",
    "event_group_0008.tif",
    "event_group_0009.tif",
    "event_group_0012.tif",
    "event_group_0014.tif",
    "event_group_0019.tif",
    "event_group_0023.tif",
    "event_group_0025.tif",
    "event_group_0027.tif",
]

# Seed line width (pixels around each drawn line)
SEED_WIDTH = 5

# Thresholding on enhanced crop
RESCUE_PERCENTILE = 95.0

# Cleanup
SLICE_CLOSE_K = 5
SLICE_MIN_PIX = 5
MIN_VOL = 2000.0  # µm³

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
EVENT_CROPS = BASE / "preprocessed" / "event_crops"
EVENT_GROUPS_CSV = BASE / "preprocessed" / "event_groups.csv"

# Raw 4D stack (for better detection)
RAW_4D_PATH = BASE / "raw" / f"runB_{RUN}_{MOUSE}-reslice-bin.tif"
# ΔF/F MIP (for display)
DFF_MIP_PATH = BASE / "raw" / f"runB_{RUN}_{MOUSE}-reslice-bin-dff.tif"
M1_SKIP_SECONDS = 12.0
FS_HZ = 5.0

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
    # Filter to events that exist
    events = [e for e in EVENT_FILES if (EVENT_CROPS / e).exists()]
    if not events:
        print("No event crops found.")
        return
    print(f"Rescuing from {len(events)} events: {events}")

    v = napari.Viewer(title="Rescue Masks")
    state = {"idx": 0, "crop": None, "enh": None, "masks": [],
             "event": None, "T": 0, "Z": 0, "Y": 0, "X": 0}

    def _load_event(idx):
        """Load event idx into the viewer."""
        v.layers.clear()
        ef = events[idx]
        state["event"] = ef
        state["masks"] = []

        print(f"\n=== Event {idx+1}/{len(events)}: {ef} ===")

        # Get frame range for this event
        skip_offset = int(M1_SKIP_SECONDS * FS_HZ)
        eg_num = int(ef.replace("event_group_", "").replace(".tif", ""))
        cs, ce = 0, 0
        if EVENT_GROUPS_CSV.exists():
            import csv as _csv
            with open(EVENT_GROUPS_CSV) as f:
                for row in _csv.DictReader(f):
                    if int(row["event_id"]) == eg_num:
                        cs = int(row["crop_start"]) + skip_offset
                        ce = int(row["crop_end"]) + skip_offset
                        break

        # Load raw 4D for this event's frames (for detection)
        if RAW_4D_PATH.exists() and ce > cs:
            print(f"  Loading raw 4D frames {cs}–{ce}...")
            raw_full = tifffile.imread(str(RAW_4D_PATH)).astype(np.float32)
            if raw_full.ndim == 3:
                raw_full = raw_full[:, np.newaxis, :, :]
            crop = raw_full[cs:min(ce, raw_full.shape[0])]
            del raw_full
        else:
            crop = tifffile.imread(str(EVENT_CROPS / ef)).astype(np.float32)

        T, Z, Y, X = crop.shape
        state["crop"] = crop
        state["T"], state["Z"], state["Y"], state["X"] = T, Z, Y, X
        print(f"  Shape: T={T}, Z={Z}, Y={Y}, X={X}")

        # Background: use ΔF/F MIP if available (sharp contrast)
        if DFF_MIP_PATH.exists() and ce > cs:
            dff_mip = tifffile.imread(str(DFF_MIP_PATH)).astype(np.float32)
            if dff_mip.ndim == 4:
                dff_mip = dff_mip.max(axis=1)
            # Get event range MIPs and subtract quietest
            event_mips = dff_mip[cs:min(ce, dff_mip.shape[0])]
            if event_mips.shape[0] > 2:
                frame_means = event_mips.mean(axis=(1, 2))
                quiet_idx = int(np.argmin(frame_means))
                event_mips_sub = event_mips - 0.8 * event_mips[quiet_idx:quiet_idx+1]
                event_mips_sub[event_mips_sub < 0] = 0
                tmax_mip = event_mips_sub.max(axis=0)
            else:
                tmax_mip = event_mips.max(axis=0)
            del dff_mip, event_mips
        else:
            tmax_mip = crop.max(axis=0).max(axis=0)

        # Existing masks from this event
        existing_masks, existing_names = load_existing_masks_for_event(ef)
        print(f"  Existing masks: {len(existing_masks)}")

        # Enhance (with temporal bg subtraction)
        print("  Enhancing...")
        if T > 2:
            frame_means = crop.mean(axis=(1, 2, 3))
            quiet_idx = int(np.argmin(frame_means))
            crop_sub = crop - 0.8 * crop[quiet_idx:quiet_idx+1]
            crop_sub[crop_sub < 0] = 0
            state["enh"] = enhance_crop(crop_sub)
            del crop_sub
        else:
            state["enh"] = enhance_crop(crop)

        # Background MIP
        lo, hi = np.percentile(tmax_mip, 1), np.percentile(tmax_mip, 99.7)
        print(f"  Background MIP: shape={tmax_mip.shape}, range=[{tmax_mip.min():.3f}, {tmax_mip.max():.3f}], contrast=[{lo:.3f}, {hi:.3f}]")
        v.add_image(tmax_mip, name="event MIP", colormap="gray",
                    contrast_limits=(lo, hi))

        # Existing masks overlay
        if existing_masks:
            combined = np.zeros((Y, X), dtype=np.int32)
            for i, m in enumerate(existing_masks):
                mip = m.max(axis=0)
                my = mip.shape[0]
                if my > Y:
                    mip = mip[:Y, :X]
                elif my < Y:
                    tmp = np.zeros((Y, X), dtype=mip.dtype)
                    tmp[:my, :X] = mip[:, :X]
                    mip = tmp
                combined[mip > 0] = i + 1
            v.add_labels(combined, name="existing masks", opacity=0.4)
            for name in existing_names:
                print(f"    {name}")

        # Seeds layer
        v.add_shapes(name="seeds", shape_type="path",
                     edge_color="lime", edge_width=3)

        v.title = f"Rescue — {ef}  ({idx+1}/{len(events)})"

    @v.bind_key("Control-g")
    def _generate(viewer):
        shapes_layer = v.layers["seeds"]
        if len(shapes_layer.data) == 0:
            print("No seeds drawn.")
            return

        Y, X = state["Y"], state["X"]
        print(f"Generating masks from {len(shapes_layer.data)} seed lines...")
        seed_2d = lines_to_seed_mask(shapes_layer.data, Y, X, SEED_WIDTH)
        print(f"  Seed pixels: {seed_2d.sum()}")

        masks = grow_masks_from_seeds(state["enh"], seed_2d)
        print(f"  Generated {len(masks)} masks")

        # Remove old rescue layers
        for layer in list(v.layers):
            if layer.name.startswith("rescue_"):
                v.layers.remove(layer)

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
                         name=f"rescue_{i:02d}", opacity=0.5)
            vol = m.sum() * VOXEL_VOL
            print(f"    rescue_{i:02d}: {vol:.0f} µm³")

        state["masks"] = masks

    @v.bind_key("Control-s")
    def _save_and_next(viewer):
        ef = state["event"]
        masks = state["masks"]
        crop = state["crop"]
        T, Y, X = state["T"], state["Y"], state["X"]

        if masks:
            start_id = next_dend_id()
            print(f"Saving {len(masks)} masks starting at dend_{start_id:03d}...")

            existing_rows = []
            if MANIFEST.exists():
                with open(MANIFEST, "r") as f:
                    existing_rows = list(csv.DictReader(f))

            new_rows = []
            for i, m in enumerate(masks):
                did = start_id + i
                mask_path = OUT_LABELS / f"dend_{did:03d}_labelmap.tif"
                tifffile.imwrite(str(mask_path), (m * (did + 1)).astype(np.uint16))

                bg2d = crop.max(axis=0).max(axis=0).astype(np.float16)
                bg_path = OUT_BGS / f"dend_{did:03d}_background_2dMIP.tif"
                tifffile.imwrite(str(bg_path), bg2d)

                new_rows.append({
                    "dend_id": did,
                    "labelmap_path": str(mask_path),
                    "background_path": str(bg_path),
                    "source_event_file": ef,
                    "event_t_start": 0,
                    "event_t_end": T,
                    "voxel_size_z": VOXEL_SIZE[0],
                    "voxel_size_y": VOXEL_SIZE[1],
                    "voxel_size_x": VOXEL_SIZE[2],
                    "volume_um3": float(m.sum() * VOXEL_VOL),
                })
                print(f"    Saved dend_{did:03d}")

            all_rows = existing_rows + new_rows
            fieldnames = list(all_rows[0].keys())
            with open(MANIFEST, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(all_rows)
            print(f"✅ Saved to manifest")
        else:
            print("No masks to save, skipping event.")

        # Advance to next event
        state["idx"] += 1
        if state["idx"] < len(events):
            _load_event(state["idx"])
        else:
            print(f"\n✅ Done — all {len(events)} events processed.")
            from napari.utils.notifications import show_info
            show_info(f"Done — all {len(events)} events processed.")

    print("\n=== INSTRUCTIONS ===")
    print("1. Draw lines on 'seeds' layer where dendrites are missing")
    print("2. Ctrl+G = generate masks from seeds")
    print("3. Ctrl+S = save & advance to next event (or skip if no seeds)")
    _load_event(0)
    napari.run()


if __name__ == "__main__":
    main()
