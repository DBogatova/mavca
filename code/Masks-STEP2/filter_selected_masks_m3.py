#!/usr/bin/env python
"""
Module 3 (2c): Curate 3D masks with neighbors + backgrounds (fixed)

Hotkeys:
  Left/Right       : navigate masks
  b                : toggle background visibility
  u / j            : neighbor count +1 / -1  (1..6)
  m                : merge PAINT into current
  x                : subtract PAINT from current
  e                : erase the connected component the PAINT dot touches
  c                : keep only the largest connected component
  t                : toggle 2D/3D display (2D needed to draw lasso)
  l                : keep inside lasso polygon(s), delete outside (all Z)
  p                : delete inside lasso polygon(s) (all Z)
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
from skimage.draw import polygon2mask
from scipy.spatial.distance import cdist
import csv

# ======= CONFIG =======
DATE = "2026-05-12"
MOUSE = "rbp4_132_phpeb"
RUN = "run9"

VOXEL_SIZE = (3.9, 1.0, 1.2)  # (Z,Y,X) μm
NEIGHBOR_K_DEFAULT = 3
NEIGHBOR_K_MAX = 6

# ======= PATHS =======
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

# Input: split output from M2b (change to "labelmaps" / "labelmap_backgrounds" to use raw M2 output)
LABELMAP_FOLDER = BASE / "labelmaps_split"
BGS_FOLDER      = BASE / "labelmap_backgrounds_split"

# 3D background: temporal max from raw 4D stack (shows all dendrites)
RAW_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin.tif"
M1_SKIP_SECONDS = 12.0  # must match find_events_m1.py SKIP_FIRST_SECONDS
FS_HZ = 5.0
EVENT_CROPS_FOLDER = BASE / "preprocessed" / "event_crops"
# Try split manifest first, fall back to M2 manifest
MANIFEST_PATH = BASE / "masks_manifest_split.csv"
if not MANIFEST_PATH.exists():
    MANIFEST_PATH = BASE / "masks_manifest.csv"
USE_3D_BG       = True   # False = use 2D MIP backgrounds 

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
            mrows = list(csv.DictReader(f))
        # Map dend name → source event file
        name_to_event = {}
        for r in mrows:
            dend_id = int(r["dend_id"])
            name = f"dend_{dend_id:03d}"
            name_to_event[name] = r["source_event_file"]
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

    # Global temporal max of raw 4D stack (all dendrites that ever fired)
    bg3d_global_max = None
    if USE_3D_BG and RAW_STACK_PATH.exists():
        print(f"  Computing global temporal max from {RAW_STACK_PATH.name}...")
        tf = tifffile.TiffFile(str(RAW_STACK_PATH))
        T_raw = tf.series[0].shape[0]
        chunk_t = 50
        tmax = None
        for t0 in range(0, T_raw, chunk_t):
            t1 = min(t0 + chunk_t, T_raw)
            chunk = tf.series[0].asarray()[t0:t1].astype(np.float32)
            if tmax is None:
                tmax = chunk.max(axis=0)
            else:
                np.maximum(tmax, chunk.max(axis=0), out=tmax)
            del chunk
        tf.close()
        bg3d_global_max = tmax
        print(f"  Global max: {bg3d_global_max.shape}")

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

    return masks, names, bg_map, cents, bg3d_map, bg3d_global_max

def broadcast_bg_2d_to_3d(bg2d, Z):
    return np.repeat(bg2d[None, ...], Z, axis=0)  # (Z,Y,X)

def mask_union(a, b):
    return (a.astype(bool) | b.astype(bool)).astype(np.uint8)

def mask_subtract(a, b):
    return (a.astype(bool) & ~b.astype(bool)).astype(np.uint8)

def save_curated(masks, names, deleted, edited, visited):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Clear existing output — we rewrite the full curated set from current state
    for old in OUTPUT_FOLDER.glob("dend_*_labelmap.tif"):
        old.unlink()

    # Only save masks that have been reviewed (visited) and not deleted
    count = 0
    rows = []
    for i, name in enumerate(names):
        if i in deleted:
            rows.append({"name": name, "kept": 0, "out": ""})
            continue
        if i not in visited:
            # Not yet reviewed — don't save, don't log
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
    print(f"📝 Log: {LOG_PATH} ({len(rows)} reviewed)")

# ======= MAIN =======
def main():
    masks, names, bg_map, cents, bg3d_map, bg3d_global_max = load_data()
    N = len(masks)
    print(f"Loaded {N} masks.")

    # Resume: check curation log for already-processed masks
    deleted = set()
    edited = {}
    start_idx = 0

    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            rows = list(csv.DictReader(f))
        processed_names = set()
        for r in rows:
            processed_names.add(r["name"])
            if int(r["kept"]) == 0:
                # Find index of this name
                if r["name"] in names:
                    deleted.add(names.index(r["name"]))

        # Find first unprocessed mask
        for i, name in enumerate(names):
            if name not in processed_names:
                start_idx = i
                break
        else:
            start_idx = N  # all processed

        if start_idx > 0:
            print(f"📂 Resuming from mask {start_idx}/{N} ({names[start_idx] if start_idx < N else 'done'})")
            print(f"   ({len(processed_names)} already processed, {len(deleted)} deleted)")

    if start_idx >= N:
        print("All masks already processed. Delete curation_log.csv to start over.")
        return

    # State
    idx = [start_idx]
    visited = set(i for i, name in enumerate(names) if name in 
                  (set() if not LOG_PATH.exists() else 
                   {r["name"] for r in csv.DictReader(open(LOG_PATH))}))
    neighbor_k = [NEIGHBOR_K_DEFAULT]
    bg_on = [True]
    bg_mode = ["event"]  # "event" (per-event max from crops) or "global" (full stack max)

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
            if bg_mode[0] == "global" and bg3d_global_max is not None:
                Z_bg = min(bg3d_global_max.shape[0], Z)
                Y_bg = min(bg3d_global_max.shape[1], Y)
                X_bg = min(bg3d_global_max.shape[2], X)
                bg3d = bg3d_global_max[:Z_bg, :Y_bg, :X_bg]
            else:
                bg3d = bg3d_map.get(names[i])
            if bg3d is not None:
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

        # lasso layer for "keep/delete inside polygon" (draw polygons in 2D mode)
        if "lasso" not in v.layers:
            v.add_shapes(name="lasso", ndim=3, edge_color="yellow",
                         face_color="transparent", edge_width=2)

        print(f"🔎 Focus: {names[i]} | neighbors: {', '.join([names[n] for n in neigh])}")

    def _autosave():
        """Save after every action so no work is lost."""
        save_curated(masks, names, deleted, edited, visited)

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

    # Toggle background mode (per-event vs global max)
    @v.bind_key("n")
    def _toggle_bg_mode(viewer):
        bg_mode[0] = "global" if bg_mode[0] == "event" else "event"
        print(f"🔄 Background: {bg_mode[0]}")
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
        visited.add(idx[0])
        deleted.add(idx[0])
        print(f"❌ Deleted: {names[idx[0]]}")
        _autosave()
        _next(viewer)

    @v.bind_key("k")
    def _keep(viewer):
        visited.add(idx[0])
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

    # Erase the connected component(s) touched by the paint dot
    @v.bind_key("e")
    def _erase_component(viewer):
        i = idx[0]
        base = (edited.get(i, masks[i]) > 0)
        draw = (v.layers["draw"].data > 0)
        lab = label(base)
        hit = np.unique(lab[draw & (lab > 0)])
        if hit.size == 0:
            print("⚠️  Dot the unwanted blob first, then press 'e'")
            return
        edited[i] = (base & ~np.isin(lab, hit)).astype(np.uint8)
        v.layers["draw"].data = np.zeros_like(masks[i], np.uint8)
        print(f"🧽 Erased {hit.size} component(s) from {names[i]}")
        _autosave()
        refresh_scene()

    # Keep only the largest connected component (e.g. after a cut stroke + 'x')
    @v.bind_key("c")
    def _keep_largest(viewer):
        i = idx[0]
        base = (edited.get(i, masks[i]) > 0)
        lab, n = label(base, return_num=True)
        if n <= 1:
            print("✓ Already a single component")
            return
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        edited[i] = (lab == sizes.argmax()).astype(np.uint8)
        print(f"🪓 Kept largest of {n} components for {names[i]}")
        _autosave()
        refresh_scene()

    # Lasso: keep/delete everything inside drawn polygon(s), across all Z
    def _lasso_mask(shape_zyx):
        Z, Y, X = shape_zyx
        polys = v.layers["lasso"].data if "lasso" in v.layers else []
        if len(polys) == 0:
            return None
        keep2d = np.zeros((Y, X), bool)
        for poly in polys:
            keep2d |= polygon2mask((Y, X), np.asarray(poly)[:, -2:])
        return np.broadcast_to(keep2d, (Z, Y, X))

    @v.bind_key("l")
    def _lasso_keep(viewer):
        i = idx[0]
        inside = _lasso_mask(masks[i].shape)
        if inside is None:
            print("⚠️  Draw a loop around the cell to KEEP (polygon tool, 2D mode), then press 'l'")
            return
        base = (edited.get(i, masks[i]) > 0)
        edited[i] = (base & inside).astype(np.uint8)
        v.layers["lasso"].data = []
        print(f"⭕ Kept inside lasso for {names[i]} (outside deleted)")
        _autosave()
        refresh_scene()

    @v.bind_key("p")
    def _lasso_delete(viewer):
        i = idx[0]
        inside = _lasso_mask(masks[i].shape)
        if inside is None:
            print("⚠️  Draw a loop around the junk to DELETE, then press 'p'")
            return
        base = (edited.get(i, masks[i]) > 0)
        edited[i] = (base & ~inside).astype(np.uint8)
        v.layers["lasso"].data = []
        print(f"⭕ Deleted inside lasso for {names[i]}")
        _autosave()
        refresh_scene()

    # Toggle 2D/3D display (2D needed to draw lasso polygons)
    @v.bind_key("t")
    def _toggle_ndisplay(viewer):
        v.dims.ndisplay = 2 if v.dims.ndisplay == 3 else 3
        print(f"🖥️  display = {v.dims.ndisplay}D")

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
        save_curated(masks, names, deleted, edited, visited)

    # Instructions
    print("\n=== INSTRUCTIONS ===")
    print("Left/Right: navigate  |  b: bg on/off  |  n: event/global bg  |  u/j: +/- neighbors")
    print("m: merge PAINT  |  x: subtract PAINT")
    print("e: erase PAINT-dotted component  |  c: keep largest component")
    print("t: 2D/3D toggle  |  l: KEEP inside lasso (delete outside)  |  p: DELETE inside lasso")
    print("q/w/r: MERGE neighbor 1/2/3 → current   |   a/s/f: SUBTRACT neighbor 1/2/3")
    print("d: delete  |  k: keep next")
    print("Ctrl+R: reset paint  |  Ctrl+S: save all")

    refresh_scene()
    napari.run()


def check_curated():
    """Post-curation check: disconnected components + duplicate masks."""
    import sys

    mask_paths = sorted(OUTPUT_FOLDER.glob("dend_*_labelmap.tif"))
    if not mask_paths:
        print("No curated masks found.")
        return

    print(f"=== Post-curation check: {len(mask_paths)} masks ===\n")

    masks = []
    names_check = []
    for p in mask_paths:
        m = tifffile.imread(p).astype(bool)
        masks.append(m)
        names_check.append(p.stem.replace("_labelmap", ""))

    # 1) Disconnected components
    print("--- Disconnected components ---")
    n_disconnected = 0
    for i, (m, name) in enumerate(zip(masks, names_check)):
        labeled, n = label(m > 0, return_num=True)
        if n > 1:
            sizes = [(labeled == lbl).sum() for lbl in range(1, n + 1)]
            print(f"  ⚠️  {name}: {n} components (sizes: {sorted(sizes, reverse=True)})")
            n_disconnected += 1
    if n_disconnected == 0:
        print("  ✅ All masks are single connected components")

    # 2) Duplicate / highly overlapping masks
    print("\n--- Duplicate detection (Dice > 0.3) ---")
    n_duplicates = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if masks[i].shape != masks[j].shape:
                continue
            inter = (masks[i] & masks[j]).sum()
            if inter == 0:
                continue
            dice = 2 * inter / (masks[i].sum() + masks[j].sum())
            if dice > 0.3:
                print(f"  ⚠️  {names_check[i]} ↔ {names_check[j]}: Dice={dice:.3f}")
                n_duplicates += 1
    if n_duplicates == 0:
        print("  ✅ No duplicates found")

    # 3) Summary
    print(f"\n--- Summary ---")
    print(f"  Total masks: {len(masks)}")
    print(f"  Disconnected: {n_disconnected}")
    print(f"  Duplicate pairs: {n_duplicates}")

    if n_duplicates > 0:
        print("\n  To merge duplicates, use find_dendrite_branches.py or")
        print("  manually merge in M3 (navigate to one, press 'q' to merge neighbor)")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        check_curated()
    else:
        main()
