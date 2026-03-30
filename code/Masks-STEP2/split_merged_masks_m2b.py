#!/usr/bin/env python
"""
Module 2b — Split merged masks (auto watershed + interactive polyline cuts)

Runs after auto_mask_m2.py.  Two modes:

  python split_merged_masks_m2b.py            # auto-watershed batch mode
  python split_merged_masks_m2b.py --cut 3    # Napari polyline-cut for dend 3
  python split_merged_masks_m2b.py --cut all  # Napari polyline-cut, iterate all

Auto-watershed pipeline per mask:
  1) Load 3D binary mask
  2) Skip if volume < SPLIT_VOL_THRESHOLD
  3) EDT → peak_local_max seeds → watershed → filter pieces
  4) Re-save as separate dend_* masks, update manifest

Polyline-cut pipeline (interactive):
  1) Show Z-MIP of mask on background in Napari
  2) User draws polylines across bridges (Shapes layer, "path" mode)
  3) Press "Apply Cuts" → rasterise polylines → 2D CC → lift to 3D
  4) Filter pieces, save, advance to next mask
"""

import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import (
    distance_transform_edt,
    label as ndi_label,
    binary_dilation,
    generate_binary_structure,
)
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import label
from skimage.filters import sato
from skimage.measure import regionprops

# ================== CONFIG ==================
DATE = "2026-03-20"
MOUSE = "rbp4cre_139_phpeb"
RUN = "run1"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

# ---- Input (M2 output) ----
IN_LABELS    = BASE / "labelmaps_guided"
IN_BGS       = BASE / "labelmap_backgrounds_guided"
IN_MANIFEST  = BASE / "masks_manifest.csv"

# ---- Output (split results) ----
OUT_LABELS   = BASE / "labelmaps_split"
OUT_PREV     = BASE / "labelmap_previews_split"
OUT_BGS      = BASE / "labelmap_backgrounds_split"
MANIFEST     = BASE / "masks_manifest_split.csv"
for p in (OUT_LABELS, OUT_PREV, OUT_BGS):
    p.mkdir(parents=True, exist_ok=True)

# ---- Physical voxel size (must match M2) ----
VOXEL_SIZE = (3.9, 1.0, 1.2)       # (Z,Y,X) μm
VOXEL_VOL  = float(np.prod(VOXEL_SIZE))

# ---- Splitting parameters ----
SPLIT_VOL_THRESHOLD = 15000.0      # μm³ — only attempt split if mask volume exceeds this
PEAK_MIN_DISTANCE = 8              # min voxel distance between watershed seeds
PEAK_THRESHOLD_ABS = 2.0           # min EDT value for a seed (filters shallow peaks)
USE_WEIGHT_VOL = True              # weight seeds by vesselness so they fall on true trunks
WEIGHT_SATO_SIGMAS = (1, 2, 3)    # Sato sigmas for weighting volume

# ---- Piece filters (after split) ----
MIN_PIECE_VOL = 3000.0             # μm³ — discard tiny fragments
MIN_PIECE_Y_SPAN_FRAC = 0.04      # min Y span as fraction of mask Y extent

# ---- Geometry filters (same spirit as M2) ----
USE_GEOMETRY_FILTER = True
MIN_Y_SPAN_FRAC = 0.06
MIN_ASPECT_Y_OVER_X = 0.8
MIN_ASPECT_Y_OVER_Z = 0.8

MAX_DENDRITES_TOTAL = 255

# ---- Polyline-cut parameters (Napari interactive mode) ----
CUT_THICKNESS = 5                  # polyline rasterisation thickness (px)
CUT_DILATE = 1                     # extra dilation iterations on cut mask
ERASE_THICKNESS = 5                # eraser brush thickness (px)
POST_CLOSE_K = 3                   # morphological close kernel after edits (0 = off)

# ---- Batch cleanup parameters (--clean mode) ----
CLEAN_CLOSE_K = 5                  # morphological close to fill gaps
CLEAN_OPEN_K = 3                   # morphological open to remove spurs
CLEAN_MIN_SLICE_PX = 10            # remove tiny 2D fragments per slice


# ================== CORE ==================

def split_merged_mask(mask3d, weight_vol=None):
    """
    Distance-transform watershed to split a merged 3D mask.

    Parameters
    ----------
    mask3d : ndarray (Z,Y,X) bool/uint8
        Binary mask (nonzero = foreground).
    weight_vol : ndarray (Z,Y,X) float, optional
        If provided, seeds are placed at peaks of (dist * weight_vol)
        so they prefer true trunk centers over bridging regions.

    Returns
    -------
    labelmap : ndarray (Z,Y,X) int32
        0 = background, 1..N = individual pieces.
    n_pieces : int
    """
    binary = mask3d.astype(bool)

    # Anisotropic EDT respecting physical voxel spacing
    dist = distance_transform_edt(binary, sampling=VOXEL_SIZE)

    # Optionally weight by vesselness so seeds land on tubular structures
    if weight_vol is not None:
        seed_vol = dist * (weight_vol + 1e-8)
    else:
        seed_vol = dist

    # Find seed coordinates
    coords = peak_local_max(
        seed_vol,
        min_distance=PEAK_MIN_DISTANCE,
        threshold_abs=PEAK_THRESHOLD_ABS,
        exclude_border=False,
    )

    if len(coords) <= 1:
        # Nothing to split (0 or 1 seed)
        if binary.any():
            out = np.zeros_like(binary, dtype=np.int32)
            out[binary] = 1
            return out, 1
        return np.zeros_like(binary, dtype=np.int32), 0

    # Build marker volume
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, start=1):
        markers[z, y, x] = i

    # Watershed on inverted distance (ridges become barriers)
    labels = watershed(-dist, markers, mask=binary)

    n_pieces = int(labels.max())
    print(f"    watershed found {n_pieces} pieces from {len(coords)} seeds")
    return labels, n_pieces


def filter_piece(piece_mask, full_Y):
    """Check if a watershed piece passes volume + geometry gates."""
    voxels = int(piece_mask.sum())
    vol_um3 = voxels * VOXEL_VOL
    if vol_um3 < MIN_PIECE_VOL:
        return False, vol_um3

    if USE_GEOMETRY_FILTER:
        props = regionprops(piece_mask.astype(np.int32))
        if not props:
            return False, vol_um3
        r = props[0]
        z0, y0, x0, z1, y1, x1 = r.bbox
        span_y = y1 - y0
        span_x = max(1, x1 - x0)
        span_z = max(1, z1 - z0)
        if span_y < int(MIN_Y_SPAN_FRAC * full_Y):
            return False, vol_um3
        if (span_y / span_x) < MIN_ASPECT_Y_OVER_X:
            return False, vol_um3
        if (span_y / span_z) < MIN_ASPECT_Y_OVER_Z:
            return False, vol_um3

    return True, vol_um3


def save_preview(mask_uint8, bg2d, out_path, title):
    """Save a Z-MIP overlay preview."""
    mip_mask = np.max(mask_uint8, axis=0).astype(bool)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(bg2d.astype(np.float32), cmap="gray")
    edges = cv2.Canny((mip_mask.astype(np.uint8) * 255), 0, 1) > 0
    ax.imshow(np.ma.masked_where(~edges, edges), cmap="autumn", alpha=0.8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ================== POLYLINE CUT (Napari interactive) ==================

def rasterise_polylines(shapes_data, shape_YX, thickness, dilate_iters):
    """
    Rasterise polylines into a binary 2D cut mask.

    Parameters
    ----------
    shapes_data : list of ndarray, each (N,2) with (row, col) = (Y, X)
    shape_YX : (H, W)
    thickness : int — line thickness in pixels
    dilate_iters : int — extra safety dilation

    Returns
    -------
    cuts2d : ndarray (H, W) bool
    """
    H, W = shape_YX
    cuts = np.zeros((H, W), dtype=np.uint8)

    for poly in shapes_data:
        pts = np.round(poly).astype(np.int32)
        for i in range(len(pts) - 1):
            y0, x0 = int(pts[i, 0]), int(pts[i, 1])
            y1, x1 = int(pts[i + 1, 0]), int(pts[i + 1, 1])
            cv2.line(cuts, (x0, y0), (x1, y1), 255, thickness)

    cuts_bool = cuts > 0
    if dilate_iters > 0:
        st = generate_binary_structure(2, 2)
        for _ in range(dilate_iters):
            cuts_bool = binary_dilation(cuts_bool, structure=st)
    return cuts_bool


def split_by_cuts(mask3d, cuts2d):
    """
    Apply 2D polyline cuts to a 3D mask.

    1) mip = mask3d.max(axis=0)
    2) mip_cut = mip & ~cuts2d
    3) lab2d = 2D connected components on mip_cut
    4) lab3d[z,y,x] = lab2d[y,x] wherever mask3d is True

    Returns (lab3d int32, n_pieces int).
    """
    mip = mask3d.max(axis=0).astype(bool)
    mip_cut = mip & ~cuts2d
    lab2d = label(mip_cut)
    n = int(lab2d.max())

    if n <= 1:
        lab3d = np.zeros_like(mask3d, dtype=np.int32)
        lab3d[mask3d > 0] = 1
        return lab3d, 1

    Z = mask3d.shape[0]
    lab3d = np.zeros_like(mask3d, dtype=np.int32)
    for z in range(Z):
        m = mask3d[z] > 0
        lab3d[z][m] = lab2d[m]
    return lab3d, n




def erase_from_mask_3d(mask3d, erase_polys, thickness):
    """
    Erase voxels from a 3D mask using user-drawn 2D polylines.
    Removes the rasterised region from every Z slice.

    Returns (mask3d_erased, n_removed).
    """
    Z, Y, X = mask3d.shape
    erase2d = rasterise_polylines(erase_polys, (Y, X), thickness, dilate_iters=0)
    if not erase2d.any():
        return mask3d.copy(), 0

    out = mask3d.copy()
    n_before = out.sum()
    for z in range(Z):
        out[z] &= ~erase2d
    n_removed = int(n_before - out.sum())
    return out, n_removed


def post_edit_cleanup(mask3d, close_k):
    """
    Per-slice morphological close to fill small gaps and smooth edges
    after manual edits (erase/cut).  Skipped if close_k <= 0.
    """
    if close_k <= 0:
        return mask3d
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    out = mask3d.copy()
    Z = mask3d.shape[0]
    for z in range(Z):
        sl = (out[z].astype(np.uint8) * 255)
        sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, se)
        out[z] = sl > 0
    return out


def _save_cut_pieces(labels, n_pieces, mask3d, bg2d, row, out_idx):
    """Save filtered pieces from a cut/split. Returns (new_out_idx, out_rows)."""
    Z, Y, X = mask3d.shape
    rows = []
    dend_id = row["dend_id"]

    if n_pieces <= 1:
        # Nothing was actually separated — save as-is
        m = mask3d.astype(np.uint8)
        op = OUT_LABELS / f"dend_{out_idx:03d}_labelmap.tif"
        bp = OUT_BGS / f"dend_{out_idx:03d}_background_2dMIP.tif"
        tifffile.imwrite(op, (m * (out_idx + 1)).astype(np.uint16))
        tifffile.imwrite(bp, bg2d.astype(np.float16))
        save_preview(m, bg2d, OUT_PREV / f"dend_{out_idx:03d}_preview.png",
                     f"dend_{out_idx:03d} (no split)")
        rows.append(_make_row(out_idx, op, bp, row, mask3d.sum() * VOXEL_VOL, dend_id, False))
        return out_idx + 1, rows

    kept = 0
    for pid in range(1, n_pieces + 1):
        if out_idx >= MAX_DENDRITES_TOTAL:
            break
        piece = (labels == pid)
        ok, pvol = filter_piece(piece, Y)
        if not ok:
            print(f"    piece {pid}/{n_pieces}: rejected (vol={pvol:.0f} μm³)")
            continue
        m = piece.astype(np.uint8)
        op = OUT_LABELS / f"dend_{out_idx:03d}_labelmap.tif"
        bp = OUT_BGS / f"dend_{out_idx:03d}_background_2dMIP.tif"
        tifffile.imwrite(op, (m * (out_idx + 1)).astype(np.uint16))
        tifffile.imwrite(bp, bg2d.astype(np.float16))
        save_preview(m, bg2d, OUT_PREV / f"dend_{out_idx:03d}_preview.png",
                     f"dend_{out_idx:03d} (piece {pid}/{n_pieces})")
        rows.append(_make_row(out_idx, op, bp, row, pvol, dend_id, True))
        out_idx += 1
        kept += 1
        print(f"    piece {pid}/{n_pieces}: kept (vol={pvol:.0f} μm³)")
    print(f"    → kept {kept}/{n_pieces} pieces")
    return out_idx, rows


def _make_row(idx, mask_path, bg_path, src_row, vol, split_from, was_split):
    return {
        "dend_id": idx,
        "labelmap_path": str(mask_path),
        "background_path": str(bg_path),
        "source_event_file": src_row["source_event_file"],
        "event_t_start": src_row["event_t_start"],
        "event_t_end": src_row["event_t_end"],
        "voxel_size_z": src_row["voxel_size_z"],
        "voxel_size_y": src_row["voxel_size_y"],
        "voxel_size_x": src_row["voxel_size_x"],
        "volume_um3": vol,
        "split_from": split_from,
        "was_split": was_split,
    }


def _write_manifest(out_rows):
    with open(MANIFEST, "w", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "dend_id", "labelmap_path", "background_path",
                "source_event_file", "event_t_start", "event_t_end",
                "voxel_size_z", "voxel_size_y", "voxel_size_x",
                "volume_um3", "split_from", "was_split",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Manifest saved to: {MANIFEST}")


def napari_cut(dend_ids=None):
    """
    Interactive polyline-cut mode.

    Opens Napari for each mask.  User draws polylines across bridges,
    then clicks "Apply Cuts".  Results are saved and the next mask loads.

    Parameters
    ----------
    dend_ids : list[int] or None
        Which dend_ids to process.  None = all from manifest.
    """
    import napari
    from magicgui import magicgui

    if not IN_MANIFEST.exists():
        print(f"[error] M2 manifest not found: {IN_MANIFEST}")
        return

    with open(IN_MANIFEST, "r") as f:
        m2_rows = list(csv.DictReader(f))

    if dend_ids is not None:
        m2_rows = [r for r in m2_rows if int(r["dend_id"]) in dend_ids]

    if not m2_rows:
        print("No masks to process.")
        return

    out_rows = []
    out_idx = [0]          # mutable so the closure can update it
    queue = list(m2_rows)
    current = [0]          # index into queue
    state = {}             # stash current mask/bg/row between load and apply

    viewer = napari.Viewer(title="Module 2b — Polyline Cut")

    def _load_mask(idx):
        """Load mask idx into the viewer."""
        viewer.layers.clear()
        row = queue[idx]
        did = row["dend_id"]
        mask_path = Path(row["labelmap_path"])
        bg_path = Path(row["background_path"])

        mask_raw = tifffile.imread(mask_path)
        mask3d = (mask_raw > 0).astype(bool)
        Z, Y, X = mask3d.shape

        bg2d = (
            tifffile.imread(bg_path).astype(np.float32)
            if bg_path.exists()
            else np.zeros((Y, X), dtype=np.float32)
        )

        # Z-MIP of mask for overlay
        mip = mask3d.max(axis=0).astype(np.float32)

        viewer.add_image(bg2d, name="background", colormap="gray")
        viewer.add_image(mip, name=f"mask_mip (dend_{did})", colormap="green",
                         opacity=0.4, blending="additive")
        viewer.add_shapes(
            ndim=2,
            name="cut_lines",
            shape_type="path",
            edge_color="red",
            edge_width=1,
        )
        viewer.add_shapes(
            ndim=2,
            name="erase_lines",
            shape_type="path",
            edge_color="yellow",
            edge_width=1,
        )
        viewer.title = f"Module 2b — dend_{did}  ({idx + 1}/{len(queue)})"

        # Stash data for the callbacks
        state["mask3d"] = mask3d
        state["bg2d"] = bg2d
        state["row"] = row

    def _apply():
        row = state["row"]
        mask3d = state["mask3d"]
        bg2d = state["bg2d"]
        Z, Y, X = mask3d.shape
        did = row["dend_id"]

        cut_polys = viewer.layers["cut_lines"].data
        ers_polys = viewer.layers["erase_lines"].data

        has_ers = len(ers_polys) > 0
        has_cuts = len(cut_polys) > 0
        any_edit = has_ers or has_cuts

        # 1) Apply erases
        if has_ers:
            mask3d, n_removed = erase_from_mask_3d(mask3d, ers_polys, ERASE_THICKNESS)
            print(f"  dend_{did}: erased {n_removed} voxels")

        # 2) Post-edit cleanup (close small gaps, smooth edges)
        if any_edit and POST_CLOSE_K > 0:
            mask3d = post_edit_cleanup(mask3d, POST_CLOSE_K)

        state["mask3d"] = mask3d

        # 3) Apply cuts or save
        if not any_edit:
            print(f"  dend_{did}: no edits, passing through")
            m = mask3d.astype(np.uint8)
            op = OUT_LABELS / f"dend_{out_idx[0]:03d}_labelmap.tif"
            bp = OUT_BGS / f"dend_{out_idx[0]:03d}_background_2dMIP.tif"
            tifffile.imwrite(op, (m * (out_idx[0] + 1)).astype(np.uint16))
            tifffile.imwrite(bp, bg2d.astype(np.float16))
            save_preview(m, bg2d, OUT_PREV / f"dend_{out_idx[0]:03d}_preview.png",
                         f"dend_{out_idx[0]:03d} (passthrough)")
            out_rows.append(_make_row(out_idx[0], op, bp, row,
                                      mask3d.sum() * VOXEL_VOL, did, False))
            out_idx[0] += 1
        elif has_cuts:
            cuts2d = rasterise_polylines(cut_polys, (Y, X), CUT_THICKNESS, CUT_DILATE)
            labels, n_pieces = split_by_cuts(mask3d, cuts2d)
            print(f"  dend_{did}: {len(cut_polys)} cuts → {n_pieces} pieces")
            new_idx, new_rows = _save_cut_pieces(labels, n_pieces, mask3d, bg2d, row, out_idx[0])
            out_idx[0] = new_idx
            out_rows.extend(new_rows)
        else:
            # Erase only, no cuts — save the edited mask
            print(f"  dend_{did}: erased, saving")
            m = mask3d.astype(np.uint8)
            op = OUT_LABELS / f"dend_{out_idx[0]:03d}_labelmap.tif"
            bp = OUT_BGS / f"dend_{out_idx[0]:03d}_background_2dMIP.tif"
            tifffile.imwrite(op, (m * (out_idx[0] + 1)).astype(np.uint16))
            tifffile.imwrite(bp, bg2d.astype(np.float16))
            save_preview(m, bg2d, OUT_PREV / f"dend_{out_idx[0]:03d}_preview.png",
                         f"dend_{out_idx[0]:03d} (erased)")
            out_rows.append(_make_row(out_idx[0], op, bp, row,
                                      mask3d.sum() * VOXEL_VOL, did, False))
            out_idx[0] += 1

        # Advance to next mask or finish
        _advance()

    @magicgui(call_button="Apply Cuts")
    def apply_cuts_widget():
        _apply()

    @magicgui(call_button="Skip (passthrough)")
    def skip_widget():
        row = state["row"]
        mask3d = state["mask3d"]
        bg2d = state["bg2d"]
        did = row["dend_id"]
        print(f"  dend_{did}: skipped")
        m = mask3d.astype(np.uint8)
        op = OUT_LABELS / f"dend_{out_idx[0]:03d}_labelmap.tif"
        bp = OUT_BGS / f"dend_{out_idx[0]:03d}_background_2dMIP.tif"
        tifffile.imwrite(op, (m * (out_idx[0] + 1)).astype(np.uint16))
        tifffile.imwrite(bp, bg2d.astype(np.float16))
        save_preview(m, bg2d, OUT_PREV / f"dend_{out_idx[0]:03d}_preview.png",
                     f"dend_{out_idx[0]:03d} (passthrough)")
        out_rows.append(_make_row(out_idx[0], op, bp, row,
                                  mask3d.sum() * VOXEL_VOL, did, False))
        out_idx[0] += 1
        _advance()

    def _advance():
        current[0] += 1
        if current[0] < len(queue):
            _load_mask(current[0])
        else:
            _write_manifest(out_rows)
            print(f"\n[✓] Done — {len(out_rows)} masks saved.")
            from napari.utils.notifications import show_info
            show_info(f"Done — {len(out_rows)} masks saved to {OUT_LABELS}")

    @magicgui(call_button="Delete (discard mask)")
    def delete_widget():
        row = state["row"]
        did = row["dend_id"]
        print(f"  dend_{did}: DELETED — will not appear in output")
        _advance()

    viewer.window.add_dock_widget(apply_cuts_widget, name="Apply Cuts", area="right")
    viewer.window.add_dock_widget(skip_widget, name="Skip", area="right")
    viewer.window.add_dock_widget(delete_widget, name="Delete", area="right")

    _load_mask(0)
    napari.run()


# ================== MAIN (auto-watershed batch) ==================

def main():
    # Read M2 manifest
    if not IN_MANIFEST.exists():
        print(f"[error] M2 manifest not found: {IN_MANIFEST}")
        return

    with open(IN_MANIFEST, "r") as f:
        m2_rows = list(csv.DictReader(f))

    print(f"Loaded {len(m2_rows)} masks from M2 manifest.")

    out_rows = []
    out_idx = 0

    for row in m2_rows:
        if out_idx >= MAX_DENDRITES_TOTAL:
            break

        dend_id = row["dend_id"]
        mask_path = Path(row["labelmap_path"])
        bg_path = Path(row["background_path"])

        if not mask_path.exists():
            print(f"  [skip] mask not found: {mask_path}")
            continue

        mask_raw = tifffile.imread(mask_path)
        mask3d = (mask_raw > 0).astype(bool)
        Z, Y, X = mask3d.shape
        vol_um3 = float(row["volume_um3"])

        bg2d = tifffile.imread(bg_path).astype(np.float32) if bg_path.exists() else np.zeros((Y, X), dtype=np.float32)

        # --- Decide whether to split ---
        if vol_um3 < SPLIT_VOL_THRESHOLD:
            m = mask3d.astype(np.uint8)
            op = OUT_LABELS / f"dend_{out_idx:03d}_labelmap.tif"
            bp = OUT_BGS / f"dend_{out_idx:03d}_background_2dMIP.tif"
            tifffile.imwrite(op, (m * (out_idx + 1)).astype(np.uint16))
            tifffile.imwrite(bp, bg2d.astype(np.float16))
            save_preview(m, bg2d, OUT_PREV / f"dend_{out_idx:03d}_preview.png",
                         f"dend_{out_idx:03d} (passthrough)")
            out_rows.append(_make_row(out_idx, op, bp, row, vol_um3, dend_id, False))
            out_idx += 1
            print(f"  dend_{dend_id} → passthrough (vol={vol_um3:.0f} μm³)")
            continue

        # --- Attempt watershed split ---
        print(f"  dend_{dend_id} → attempting split (vol={vol_um3:.0f} μm³)")

        weight_vol = None
        if USE_WEIGHT_VOL:
            dist_for_vess = distance_transform_edt(mask3d, sampling=VOXEL_SIZE).astype(np.float32)
            weight_vol = sato(dist_for_vess, sigmas=WEIGHT_SATO_SIGMAS, black_ridges=False)

        labels, n_pieces = split_merged_mask(mask3d, weight_vol=weight_vol)
        out_idx, new_rows = _save_cut_pieces(labels, n_pieces, mask3d, bg2d, row, out_idx)
        out_rows.extend(new_rows)

    # ====== WRITE MANIFEST ======
    print(f"\n[✓] Total output masks: {len(out_rows)}")
    _write_manifest(out_rows)
    print("Module 2b (split merged masks) complete.")


def clean_masks():
    """
    Batch cleanup pass over split output masks.
    Per-slice close → open → remove tiny fragments → re-save in place.
    """
    if not MANIFEST.exists():
        print(f"[error] Split manifest not found: {MANIFEST}")
        print("  Run the split (default or --cut) first.")
        return

    with open(MANIFEST, "r") as f:
        rows = list(csv.DictReader(f))

    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLEAN_CLOSE_K, CLEAN_CLOSE_K))
    se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLEAN_OPEN_K, CLEAN_OPEN_K))

    print(f"Cleaning {len(rows)} masks (close={CLEAN_CLOSE_K}, open={CLEAN_OPEN_K}, min_px={CLEAN_MIN_SLICE_PX})")

    updated_rows = []
    for row in rows:
        mask_path = Path(row["labelmap_path"])
        bg_path = Path(row["background_path"])
        did = row["dend_id"]

        if not mask_path.exists():
            print(f"  dend_{did}: mask not found, skipping")
            updated_rows.append(row)
            continue

        mask_raw = tifffile.imread(mask_path)
        mask3d = (mask_raw > 0).astype(bool)
        Z, Y, X = mask3d.shape
        n_before = int(mask3d.sum())

        cleaned = np.zeros_like(mask3d, dtype=bool)
        for z in range(Z):
            sl = mask3d[z].astype(np.uint8) * 255
            if CLEAN_CLOSE_K > 0:
                sl = cv2.morphologyEx(sl, cv2.MORPH_CLOSE, se_close)
            if CLEAN_OPEN_K > 0:
                sl = cv2.morphologyEx(sl, cv2.MORPH_OPEN, se_open)
            slb = sl > 0
            # Remove tiny 2D fragments
            if CLEAN_MIN_SLICE_PX > 0:
                lbl2 = label(slb)
                keep = np.zeros_like(slb, dtype=bool)
                for i in range(1, int(lbl2.max()) + 1):
                    if (lbl2 == i).sum() >= CLEAN_MIN_SLICE_PX:
                        keep |= (lbl2 == i)
                slb = keep
            cleaned[z] = slb

        n_after = int(cleaned.sum())
        vol_um3 = n_after * VOXEL_VOL

        # Re-save
        label_val = int(did) + 1
        tifffile.imwrite(mask_path, (cleaned.astype(np.uint8) * label_val).astype(np.uint16))

        # Re-save preview
        prev_path = OUT_PREV / f"dend_{int(did):03d}_preview.png"
        bg2d = tifffile.imread(bg_path).astype(np.float32) if bg_path.exists() else np.zeros((Y, X), dtype=np.float32)
        save_preview(cleaned.astype(np.uint8), bg2d, prev_path, f"dend_{int(did):03d} (cleaned)")

        delta = n_after - n_before
        sign = "+" if delta >= 0 else ""
        print(f"  dend_{did}: {n_before} → {n_after} voxels ({sign}{delta})")

        row["volume_um3"] = vol_um3
        updated_rows.append(row)

    # Re-write manifest with updated volumes
    with open(MANIFEST, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=updated_rows[0].keys())
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"[✓] Cleanup complete. Manifest updated: {MANIFEST}")


if __name__ == "__main__":
    # --cut N   → interactive polyline-cut for dend N
    # --cut all → interactive polyline-cut for all masks
    # --clean   → batch cleanup pass on split output
    # (no args) → auto-watershed batch mode
    if "--clean" in sys.argv:
        clean_masks()
    elif "--cut" in sys.argv:
        idx = sys.argv.index("--cut")
        arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "all"
        if arg == "all":
            napari_cut(dend_ids=None)
        else:
            napari_cut(dend_ids=[int(x) for x in arg.split(",")])
    else:
        main()
