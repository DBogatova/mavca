#!/usr/bin/env python3
"""
Module 1.5.2 — Semi-manual trunk guides (Napari polyline on M1.5 bestframes)

What it does
------------
For each M1.5 bestframe (rank01 by default):
- shows the 2D MIP PNG (bestframe_*_rank01_mip.png)
- loads the matching 3D volume (bestframe_*_rank01_3d.tif) as context (optional)
- you draw a polyline along a trunk in XY (Napari Shapes layer, add "path")
- press:
    s = save guide for this event_group
    n = next item
    p = previous item
    c = clear current guide (for this item)
    q = quit (close viewer)

Outputs
-------
Saves a single JSON:
  <BASE>/preprocessed/guides/trunk_guides.json

Structure:
  {
    "event_group_0003": {
      "event_group": "event_group_0003",
      "mip_path": ".../bestframe_event_group_0003_..._rank01_mip.png",
      "vol_path": ".../bestframe_event_group_0003_..._rank01_3d.tif",
      "polyline_xy": [[y,x], [y,x], ...],
      "notes": ""
    },
    ...
  }

Notes
-----
- This script avoids napari.utils.io.imread (not available in your napari).
- Uses imageio for PNG reading.
- Keeps API usage stable for Apple Silicon / napari versions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import tifffile
import imageio.v2 as imageio
import napari


# =========================
# CONFIG
# =========================
DATE = "2025-12-25"
MOUSE = "rAi162_phpeb"
RUN = "run1"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

BESTFRAMES_DIR = BASE / "preprocessed" / "best_frames_test"   # <-- change if needed
MIP_GLOB = "bestframe_*_rank01_mip.png"

GUIDE_DIR = BASE / "preprocessed" / "guides"
GUIDE_DIR.mkdir(parents=True, exist_ok=True)
GUIDE_JSON = GUIDE_DIR / "trunk_guides.json"

# If True, show the 3D stack too (helps find Z context). If napari gets heavy, set False.
SHOW_3D_VOL = True
VOL_OPACITY = 0.35

# If you want multiple guides per event_group, set True.
# If False: only the first drawn path is saved.
ALLOW_MULTIPLE_GUIDES = True

# =========================
# Regex
# =========================
RE_EVENT_GROUP = re.compile(r"(event_group_\d{4})")
RE_RANK01_MIP = re.compile(r"(.*)_rank01_mip$", re.IGNORECASE)


def extract_event_group(stem: str) -> str:
    m = RE_EVENT_GROUP.search(stem)
    return m.group(1) if m else stem


def find_matching_rank01_3d(mip_path: Path) -> Optional[Path]:
    """
    Given bestframe_*_rank01_mip.png, find bestframe_*_rank01_3d.tif
    Works for names like:
      bestframe_event_group_0003_peak..._t00006_rank01_mip.png
      bestframe_event_group_0003_peak..._t00006_rank01_3d.tif
    """
    stem = mip_path.stem
    # Replace trailing _mip with _3d if present
    base = stem.replace("_mip", "")
    candidate = mip_path.with_name(base + "_3d.tif")
    if candidate.exists():
        return candidate

    # Fallback: search in folder for same prefix + rank01_3d.tif
    eg = extract_event_group(stem)
    hits = sorted(mip_path.parent.glob(f"*{eg}*rank01*3d.tif"))
    return hits[0] if hits else None


def load_guides() -> Dict[str, Any]:
    if GUIDE_JSON.exists():
        try:
            return json.loads(GUIDE_JSON.read_text())
        except Exception:
            return {}
    return {}


def save_guides(guides: Dict[str, Any]) -> None:
    GUIDE_JSON.write_text(json.dumps(guides, indent=2))
    print(f"[saved] {GUIDE_JSON}")


def main():
    mip_paths = sorted(BESTFRAMES_DIR.glob(MIP_GLOB))
    if not mip_paths:
        raise FileNotFoundError(f"No files matched {MIP_GLOB} in {BESTFRAMES_DIR}")

    guides = load_guides()

    idx = 0
    viewer = napari.Viewer()

    # We'll recreate layers on each item; keep a reference to Shapes layer.
    shapes = viewer.add_shapes(
        name="guide_polyline",
        shape_type="path",
        edge_width=3
    )
    shapes.mode = "add_path"

    def status_line(i: int, eg: str) -> str:
        return f"{i+1}/{len(mip_paths)}  |  {eg}  |  keys: double-click=finish path, s=save, n=next, p=prev, c=clear, q=quit"

    def set_shapes_from_saved(eg: str):
        shapes.data = []
        if eg in guides and "polyline_xy" in guides[eg] and guides[eg]["polyline_xy"]:
            if ALLOW_MULTIPLE_GUIDES and isinstance(guides[eg]["polyline_xy"][0][0], list):
                # list of polylines
                for poly in guides[eg]["polyline_xy"]:
                    shapes.add(np.array(poly, dtype=float), shape_type="path")
            else:
                # single polyline
                shapes.add(np.array(guides[eg]["polyline_xy"], dtype=float), shape_type="path")
        shapes.mode = "add_path"

    def get_current_polylines_xy() -> list:
        if len(shapes.data) == 0:
            return []
        if ALLOW_MULTIPLE_GUIDES:
            polys = []
            for d in shapes.data:
                pts = np.array(d, dtype=float)
                polys.append(pts.tolist())
            return polys
        else:
            pts = np.array(shapes.data[0], dtype=float)
            return pts.tolist()

    def clear_shapes():
        shapes.data = []
        shapes.mode = "add_path"

    def load_item(i: int):
        # Clear only image layers, keep shapes layer
        layers_to_remove = [layer for layer in viewer.layers if layer != shapes]
        for layer in layers_to_remove:
            viewer.layers.remove(layer)

        mip = mip_paths[i]
        eg = extract_event_group(mip.stem)
        vol = find_matching_rank01_3d(mip)

        # Load MIP PNG via imageio
        img = imageio.imread(mip)
        # If RGB, convert to grayscale-ish for display (napari can show RGB too, but gray is simpler)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., 0]

        viewer.add_image(img, name=f"{eg}_mip", colormap="gray")

        if SHOW_3D_VOL and vol is not None and vol.exists():
            v = tifffile.imread(vol).astype(np.float32)  # (Z,Y,X)
            viewer.add_image(
                v,
                name=f"{eg}_vol",
                colormap="gray",
                blending="additive",
                opacity=VOL_OPACITY
            )

        # Move shapes to top and ensure it's in add_path mode
        viewer.layers.move(viewer.layers.index(shapes), -1)
        shapes.data = []
        set_shapes_from_saved(eg)

        viewer.status = status_line(i, eg)
        print(f"[show] {eg}  |  mip={mip.name}  |  vol={(vol.name if vol else 'NONE')}")

    def save_current(_=None):
        nonlocal idx
        mip = mip_paths[idx]
        eg = extract_event_group(mip.stem)
        vol = find_matching_rank01_3d(mip)

        polys = get_current_polylines_xy()
        if not polys:
            print("[warn] No guide drawn; nothing saved.")
            return

        guides[eg] = {
            "event_group": eg,
            "mip_path": str(mip),
            "vol_path": str(vol) if vol else "",
            "polyline_xy": polys,
            "notes": guides.get(eg, {}).get("notes", ""),
        }
        save_guides(guides)
        print(f"[guide saved] {eg}  ({'multi' if ALLOW_MULTIPLE_GUIDES else 'single'})")

    def next_item(_=None):
        nonlocal idx
        idx = min(idx + 1, len(mip_paths) - 1)
        load_item(idx)

    def prev_item(_=None):
        nonlocal idx
        idx = max(idx - 1, 0)
        load_item(idx)

    def clear_current(_=None):
        clear_shapes()
        print("[cleared] current polyline(s) (not saved until you press s)")

    def quit_viewer(_=None):
        viewer.close()

    viewer.bind_key("s", save_current)
    viewer.bind_key("n", next_item)
    viewer.bind_key("p", prev_item)
    viewer.bind_key("c", clear_current)
    viewer.bind_key("q", quit_viewer)

    load_item(idx)
    napari.run()


if __name__ == "__main__":
    main()
