#!/usr/bin/env python
"""
Module 3 (2c): Curate 3D masks with neighbors + backgrounds (fixed)

Hotkeys:
  Left/Right       : navigate masks
  b                : toggle background visibility
  u / j            : neighbor count +1 / -1  (1..6)
  m                : merge PAINT into current
  x                : subtract PAINT from current
  1/2/3            : MERGE neighbor # → current
  Shift+1/2/3      : SUBTRACT neighbor # from current
  d                : delete current
  k                : keep current, goto next
  Ctrl+R           : reset paint layer
  Ctrl+S           : save all kept masks
"""

from pathlib import Path
import numpy as np
import tifffile
import napari
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
import csv

# ======= CONFIG =======
DATE = "2026-04-16"
MOUSE = "rbp4_132_phpeb"
RUN = "run7"

VOXEL_SIZE = (3.9, 1.0, 1.2)  # (Z,Y,X) μm
NEIGHBOR_K_DEFAULT = 3
NEIGHBOR_K_MAX = 6

# ======= PATHS =======
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

# Input: split output from M2b (change to "labelmaps" / "labelmap_backgrounds" to use raw M2 output)
LABELMAP_FOLDER = BASE / "labelmaps_split"
BGS_FOLDER      = BASE / "labelmap_backgrounds_split"

# 3D background: per-event temporal max from event crops
EVENT_CROPS_FOLDER = BASE / "preprocessed" / "event_crops"
# Try split manifest first, fall back to M2 manifest
MANIFEST_PATH = BASE / "masks_manifest_split.csv"
if not MANIFEST_PATH.exists():
    MANIFEST_PATH = BASE / "masks_manifest.csv"
USE_3D_BG       = True   # False = use 2D MIP backgrounds as before

OUTPUT_FOLDER   = BASE / "labelmaps_curated_dynamic"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_FOLDER / "curation_log.csv"

# ======= HELPERS =======
def stem_id(p: Path) -> str:
    s = p.stem
    return s.replace("_labelmap", "").replace("_background_2dMIP", "")

def load_data():
    mask_paths = sorted(LABELMAP_FOLDER.glob("dend_*_labelmap.tif"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks in {LABELMAP_FOLDER}")
    masks = [tifffile.imread(p).astype(np.uint8) for p in mask_paths]
    names = [stem_id(p) for p in mask_paths]

    # Connected component check: warn and keep largest if multiple pieces
    for i, (m, name) in enumerate(zip(masks, names)):
        labeled, n = label(m > 0, return_num=True)
        if n > 1:
            sizes = [(labeled == lbl).sum() for lbl in range(1, n + 1)]
            biggest = np.argmax(sizes) + 1
            print(f"  ⚠️  {name}: {n} disconnected components "
                  f"(sizes: {sorted(sizes, reverse=True)}), keeping largest")
            masks[i] = ((labeled == biggest) * 1).astype(np.uint8)

    # backgrounds: 2D MIP fallback
    bg_map = {}
    for p in BGS_FOLDER.glob("dend_*_background_2dMIP.tif"):
        bg_map[stem_id(p)] = tifffile.imread(p).astype(np.float32)  # (Y,X)

    # 3D background: per-event temporal max from event crops
    bg3d_map = {}  # name → (Z,Y,X) float32
    if USE_3D_BG and MANIFEST_PATH.exists() and EVENT_CROPS_FOLDER.exists():
        with open(MANIFEST_PATH, "r") as f:
            rows = list(csv.DictReader(f))
        # Map dend name → source event file
        name_to_event = {}
        for row in rows:
            dend_id = int(row["dend_id"])
            name = f"dend_{dend_id:03d}"
            name_to_event[name] = row["source_event_file"]
        # Load each unique event crop once, compute temporal max
        event_cache = {}  # event_file → (Z,Y,X) float32
        for name in names:
            ev_file = name_to_event.get(name)
            if ev_file is None:
                continue
            if ev_file not in event_cache:
                ev_path = EVENT_CROPS_FOLDER / ev_file
                if ev_path.exists():
                    crop = tifffile.imread(str(ev_path)).astype(np.float32)
                    # crop is (T,Z,Y,X) — take temporal max
                    event_cache[ev_file] = crop.max(axis=0)
                    del crop
                    print(f"  Loaded event bg: {ev_file} → {event_cache[ev_file].shape}")
            if ev_file in event_cache:
                bg3d_map[name] = event_cache[ev_file]
        print(f"  3D backgrounds: {len(bg3d_map)} masks from {len(event_cache)} events")
        # Print mapping for verification
        for name in names:
            ev = name_to_event.get(name, "???")
            has_bg = "✓" if name in bg3d_map else "✗"
            print(f"    {has_bg} {name} → {ev}")

    # centroids in μm for NN search
    cents = []
    for m in masks:
        rp = regionprops(label(m))
        if not rp:
            cents.append(np.array([0., 0., 0.]))
        else:
            cz, cy, cx = rp[0].centroid
            vz, vy, vx = VOXEL_SIZE
            cents.append(np.array([cz*vz, cy*vy, cx*vx], dtype=float))
    cents = np.vstack(cents)

    return masks, names, bg_map, cents, bg3d_map

def broadcast_bg_2d_to_3d(bg2d, Z):
    return np.repeat(bg2d[None, ...], Z, axis=0)  # (Z,Y,X)

def mask_union(a, b):
    return (a.astype(bool) | b.astype(bool)).astype(np.uint8)

def mask_subtract(a, b):
    return (a.astype(bool) & ~b.astype(bool)).astype(np.uint8)

def save_curated(masks, names, deleted, edited):
    # Clear old files from previous runs
    for old in OUTPUT_FOLDER.glob("dend_*_labelmap.tif"):
        old.unlink()

    count = 0
    rows = []
    for i, name in enumerate(names):
        if i in deleted:
            rows.append({"name": name, "kept": 0, "out": ""})
            continue
        m = edited.get(i, masks[i])
        out = OUTPUT_FOLDER / f"dend_{count:03d}_labelmap.tif"
        tifffile.imwrite(out, (m * (count + 1)).astype(np.uint16))
        rows.append({"name": name, "kept": 1, "out": str(out)})
        count += 1

    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "kept", "out"])
        w.writeheader(); w.writerows(rows)

    print(f"✅ Saved {count} masks to {OUTPUT_FOLDER}")
    print(f"📝 Log: {LOG_PATH}")

# ======= MAIN =======
def main():
    masks, names, bg_map, cents, bg3d_map = load_data()
    N = len(masks)
    print(f"Loaded {N} masks.")

    # State
    idx = [0]  # current index (mutable)
    deleted = set()
    edited  = {}
    neighbor_k = [NEIGHBOR_K_DEFAULT]
    bg_on = [True]

    v = napari.Viewer(ndisplay=3)

    def show_mask_layer(i, highlight=False, name=None):
        layer_name = name or names[i]
        data = edited.get(i, masks[i])
        if layer_name in v.layers:
            v.layers[layer_name].data = data
            v.layers[layer_name].opacity = 1.0 if highlight else 0.4
        else:
            v.add_labels(data, name=layer_name, opacity=1.0 if highlight else 0.4)

    def refresh_scene():
        """Focus current mask + k NN; show background first."""
        i = idx[0]
        Z, Y, X = masks[i].shape

        # --- background ---
        if bg_on[0]:
            bg3d = bg3d_map.get(names[i])
            if bg3d is not None:
                # Per-event 3D background (temporal max of event crop)
                lo = float(np.percentile(bg3d, 2.0))
                hi = float(np.percentile(bg3d, 99.5))
                if "bg" in v.layers:
                    img = v.layers["bg"]
                    img.data = bg3d
                    img.contrast_limits = (lo, hi)
                    img.visible = True
                    img.opacity = 1.0
                else:
                    v.add_image(
                        bg3d, name="bg", blending="additive", colormap="gray",
                        contrast_limits=(lo, hi), opacity=1.0,
                    )
            else:
                # Fallback: 2D MIP broadcast
                bg2d = bg_map.get(names[i])
                if bg2d is not None:
                    bg3d_fb = broadcast_bg_2d_to_3d(bg2d, Z)
                    lo = float(np.percentile(bg3d_fb, 2.0))
                    hi = float(np.percentile(bg3d_fb, 99.5))
                    if "bg" in v.layers:
                        img = v.layers["bg"]
                        img.data = bg3d_fb
                        img.contrast_limits = (lo, hi)
                        img.visible = True
                        img.opacity = 1.0
                    else:
                        v.add_image(
                            bg3d_fb, name="bg", blending="additive", colormap="gray",
                            contrast_limits=(lo, hi), opacity=1.0,
                        )
                else:
                    if "bg" in v.layers:
                        v.layers["bg"].visible = False
        else:
            if "bg" in v.layers:
                v.layers["bg"].visible = False

        # --- remove all dend_* label layers (not the draw layer) ---
        # v.layers yields Layer objects; check .name
        for layer in list(v.layers):
            if isinstance(layer, napari.layers.Labels) and layer.name.startswith("dend_") and layer.name != "draw":
                v.layers.remove(layer)

        # --- neighbors by distance (in μm) ---
        d = cdist([cents[i]], cents)[0]
        order = np.argsort(d)
        neigh = [j for j in order if j != i and j not in deleted][:neighbor_k[0]]

        # focused first (high opacity)
        show_mask_layer(i, highlight=True, name=names[i])

        # neighbors (low opacity)
        for j, nidx in enumerate(neigh, start=1):
            show_mask_layer(nidx, highlight=False, name=f"{names[nidx]}__nbr{j}")

        # paint layer sized to current mask
        if "draw" in v.layers and v.layers["draw"].data.shape != masks[i].shape:
            v.layers.remove("draw")
        if "draw" not in v.layers:
            v.add_labels(np.zeros_like(masks[i], np.uint8), name="draw", opacity=0.6)

        print(f"🔎 Focus: {names[i]} | neighbors: {', '.join([names[n] for n in neigh])}")

    def _autosave():
        """Save after every action so no work is lost."""
        save_curated(masks, names, deleted, edited)

    # --- Navigation ---
    @v.bind_key("Right")
    def _next(viewer):
        start = idx[0]
        while idx[0] < N - 1:
            idx[0] += 1
            if idx[0] not in deleted:
                break
        refresh_scene()

    @v.bind_key("Left")
    def _prev(viewer):
        start = idx[0]
        while idx[0] > 0:
            idx[0] -= 1
            if idx[0] not in deleted:
                break
        refresh_scene()

    # Toggle background
    @v.bind_key("b")
    def _toggle_bg(viewer):
        bg_on[0] = not bg_on[0]
        refresh_scene()

    # Neighbor count up/down
    @v.bind_key("u")
    def _more_neighbors(viewer):
        neighbor_k[0] = min(NEIGHBOR_K_MAX, neighbor_k[0] + 1)
        refresh_scene()

    @v.bind_key("j")
    def _fewer_neighbors(viewer):
        neighbor_k[0] = max(1, neighbor_k[0] - 1)
        refresh_scene()

    # Delete / Keep
    @v.bind_key("d")
    def _delete(viewer):
        deleted.add(idx[0])
        print(f"❌ Deleted: {names[idx[0]]}")
        _autosave()
        _next(viewer)

    @v.bind_key("k")
    def _keep(viewer):
        print(f"✅ Kept: {names[idx[0]]}")
        _autosave()
        _next(viewer)

    # Reset draw
    @v.bind_key("Control-R")
    def _reset_draw(viewer):
        if "draw" in v.layers:
            v.layers.remove("draw")
        v.add_labels(np.zeros_like(masks[idx[0]], np.uint8), name="draw", opacity=0.6)
        print("🎨 Reset draw.")

    # Merge/Subtract PAINT
    @v.bind_key("m")
    def _merge_paint(viewer):
        i = idx[0]
        base = (edited.get(i, masks[i]) > 0)
        draw = (v.layers["draw"].data > 0)
        edited[i] = mask_union(base, draw)
        print(f"🟣 Merged PAINT into {names[i]}")
        _autosave()
        refresh_scene()

    @v.bind_key("x")
    def _subtract_paint(viewer):
        i = idx[0]
        base = (edited.get(i, masks[i]) > 0)
        draw = (v.layers["draw"].data > 0)
        edited[i] = mask_subtract(base, draw)
        print(f"✂️ Subtracted PAINT from {names[i]}")
        _autosave()
        refresh_scene()

    # Neighbor helpers
    def neighbor_indices():
        i = idx[0]
        d = cdist([cents[i]], cents)[0]
        order = np.argsort(d)
        return [j for j in order if j != i and j not in deleted][:neighbor_k[0]]

    def merge_neighbor(n):
        i = idx[0]
        neigh = neighbor_indices()
        if n-1 >= len(neigh): return
        j = neigh[n-1]
        a = edited.get(i, masks[i])
        b = edited.get(j, masks[j])
        edited[i] = mask_union(a, b)
        deleted.add(j)  # Remove merged neighbor
        print(f"🧩 MERGE neighbor#{n} ({names[j]}) → {names[i]} (neighbor deleted)")
        _autosave()
        refresh_scene()

    def subtract_neighbor(n):
        i = idx[0]
        neigh = neighbor_indices()
        if n-1 >= len(neigh): return
        j = neigh[n-1]
        a = edited.get(i, masks[i])
        b = edited.get(j, masks[j])
        edited[i] = mask_subtract(a, b)
        print(f"➖ SUBTRACT neighbor#{n} ({names[j]}) from {names[i]}")
        _autosave()
        refresh_scene()

    # Merge neighbors with q/w/r keys
    @v.bind_key("q")
    def _merge_n1(viewer): merge_neighbor(1)
    @v.bind_key("w")
    def _merge_n2(viewer): merge_neighbor(2)
    @v.bind_key("r")
    def _merge_n3(viewer): merge_neighbor(3)

    # Subtract neighbors with a/s/f keys
    @v.bind_key("a")
    def _sub_n1(viewer): subtract_neighbor(1)
    @v.bind_key("s")
    def _sub_n2(viewer): subtract_neighbor(2)
    @v.bind_key("f")
    def _sub_n3(viewer): subtract_neighbor(3)

    # Save all
    @v.bind_key("Control-S")
    def _save(viewer):
        save_curated(masks, names, deleted, edited)

    # Instructions
    print("\n=== INSTRUCTIONS ===")
    print("Left/Right: navigate  |  b: bg on/off  |  u/j: +/- neighbors")
    print("m: merge PAINT  |  x: subtract PAINT")
    print("q/w/r: MERGE neighbor 1/2/3 → current   |   a/s/f: SUBTRACT neighbor 1/2/3")
    print("d: delete  |  k: keep next")
    print("Ctrl+R: reset paint  |  Ctrl+S: save all")

    refresh_scene()
    napari.run()

if __name__ == "__main__":
    main()
