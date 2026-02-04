#!/usr/bin/env python3
"""
Use after M2
Draw trunk guides on structural-masks_m2.py preview PNGs


Workflow:
1. Run structural-masks_m2.py first (USE_GUIDES=False)
2. Run this script to draw guides on the preview PNGs
3. Re-run structural-masks_m2.py with USE_GUIDES=True

Keys:
- double-click: finish current path
- s or Ctrl+S: save guide for this event_group (also saves PNG preview)
- n: next preview
- p: previous preview
- c: clear all guides for current preview
- d: delete selected guide (click to select first)
- e: toggle erase mode (click guides to delete them)
- q: quit
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import imageio.v2 as imageio
import napari


# =========================
# CONFIG
# =========================
DATE = "2025-12-25"
MOUSE = "rAi162_phpeb"
RUN = "run1"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

PREVIEW_DIR = BASE / "labelmap_previews"
PREVIEW_GLOB = "dend_*_preview.png"

# Show preview without overlay edges (cleaner for drawing guides)
SHOW_CLEAN_BACKGROUND = True  # Set False to see the edge overlays

GUIDE_DIR = BASE / "preprocessed" / "guides"
GUIDE_DIR.mkdir(parents=True, exist_ok=True)
GUIDE_JSON = GUIDE_DIR / "trunk_guides.json"
GUIDE_PREVIEW_DIR = GUIDE_DIR / "guide_previews"
GUIDE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Regex
# =========================
RE_EVENT_GROUP = re.compile(r"\(([^)]+)\)")  # Extract event_group from title like "dend_001 (event_group_0003)"


def extract_event_group_from_title(png_path: Path) -> str:
    """Extract event_group from preview PNG by reading the title from the image."""
    # Read from manifest
    manifest = BASE / "masks_manifest.csv"
    if manifest.exists():
        import csv
        with open(manifest, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if f"dend_{int(row['trunk_id']):03d}_preview.png" == png_path.name:
                    return row['event_group']
    return png_path.stem.replace("_preview", "")


def get_background_for_preview(png_path: Path) -> Path:
    """Get the clean background image for this preview."""
    manifest = BASE / "masks_manifest.csv"
    if manifest.exists():
        import csv
        with open(manifest, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if f"dend_{int(row['trunk_id']):03d}_preview.png" == png_path.name:
                    return Path(row['background_path'])
    return None


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


def save_guide_preview(png_path: Path, eg: str, polylines: list) -> None:
    """Save a PNG preview showing the guides overlaid on the background."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Load background
    if SHOW_CLEAN_BACKGROUND:
        bg_path = get_background_for_preview(png_path)
        if bg_path and bg_path.exists():
            import tifffile
            img = tifffile.imread(bg_path)
        else:
            img = imageio.imread(png_path)
    else:
        img = imageio.imread(png_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    
    # Draw polylines
    colors = plt.cm.rainbow(np.linspace(0, 1, len(polylines)))
    for i, poly in enumerate(polylines):
        pts = np.array(poly)
        ax.plot(pts[:, 1], pts[:, 0], 'o-', color=colors[i], linewidth=3, markersize=5, label=f'Guide {i+1}')
    
    ax.set_title(f"{eg} - {len(polylines)} guide(s)")
    ax.axis('off')
    if len(polylines) > 0:
        ax.legend(loc='upper right')
    
    out_png = GUIDE_PREVIEW_DIR / f"{eg}_guides.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[preview saved] {out_png}")


def main():
    preview_paths = sorted(PREVIEW_DIR.glob(PREVIEW_GLOB))
    if not preview_paths:
        raise FileNotFoundError(f"No files matched {PREVIEW_GLOB} in {PREVIEW_DIR}")

    guides = load_guides()

    idx = 0
    viewer = napari.Viewer()

    shapes = viewer.add_shapes(
        name="guide_polyline",
        shape_type="path",
        edge_width=3,
        edge_color="cyan"
    )
    shapes.mode = "add_path"

    erase_mode = [False]  # Use list to allow modification in nested functions

    def status_line(i: int, eg: str) -> str:
        mode = "ERASE" if erase_mode[0] else "DRAW"
        return f"{i+1}/{len(preview_paths)}  |  {eg}  |  Mode: {mode}  |  keys: e=erase, d=delete, s=save, n=next, p=prev, c=clear, q=quit"

    def set_shapes_from_saved(eg: str):
        shapes.data = []
        if eg in guides and "polyline_xy" in guides[eg] and guides[eg]["polyline_xy"]:
            polylines = guides[eg]["polyline_xy"]
            # Handle both single and multiple polylines
            if not isinstance(polylines[0][0], list):
                polylines = [polylines]
            for poly in polylines:
                shapes.add(np.array(poly, dtype=float), shape_type="path")
        shapes.mode = "add_path"

    def get_current_polylines_xy() -> list:
        if len(shapes.data) == 0:
            return []
        polys = []
        for d in shapes.data:
            pts = np.array(d, dtype=float)
            polys.append(pts.tolist())
        return polys

    def load_item(i: int):
        # Clear only image layers
        layers_to_remove = [layer for layer in viewer.layers if layer != shapes]
        for layer in layers_to_remove:
            viewer.layers.remove(layer)

        png = preview_paths[i]
        eg = extract_event_group_from_title(png)

        # Load clean background or preview
        if SHOW_CLEAN_BACKGROUND:
            bg_path = get_background_for_preview(png)
            if bg_path and bg_path.exists():
                import tifffile
                img = tifffile.imread(bg_path)
                # Convert to RGB for display
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                viewer.add_image(img, name=f"{eg}_background", rgb=True)
            else:
                img = imageio.imread(png)
                viewer.add_image(img, name=f"{eg}_preview", rgb=True)
        else:
            img = imageio.imread(png)
            viewer.add_image(img, name=f"{eg}_preview", rgb=True)

        # Move shapes to top
        viewer.layers.move(viewer.layers.index(shapes), -1)
        shapes.data = []
        set_shapes_from_saved(eg)

        viewer.status = status_line(i, eg)
        print(f"[show] {eg}  |  preview={png.name}")

    def save_current(_=None):
        nonlocal idx
        png = preview_paths[idx]
        eg = extract_event_group_from_title(png)

        polys = get_current_polylines_xy()
        if not polys:
            print("[warn] No guide drawn; nothing saved.")
            return

        guides[eg] = {
            "event_group": eg,
            "mip_path": str(png),
            "vol_path": "",
            "polyline_xy": polys,
            "notes": guides.get(eg, {}).get("notes", ""),
        }
        save_guides(guides)
        save_guide_preview(png, eg, polys)
        print(f"[guide saved] {eg}  ({len(polys)} polyline(s))")

    def next_item(_=None):
        nonlocal idx
        idx = min(idx + 1, len(preview_paths) - 1)
        load_item(idx)

    def prev_item(_=None):
        nonlocal idx
        idx = max(idx - 1, 0)
        load_item(idx)

    def clear_current(_=None):
        shapes.data = []
        shapes.mode = "add_path"
        erase_mode[0] = False
        viewer.status = status_line(idx, extract_event_group_from_title(preview_paths[idx]))
        print("[cleared] all guides for current preview")

    def delete_selected(_=None):
        if len(shapes.selected_data) > 0:
            # Remove selected shapes
            indices_to_remove = sorted(shapes.selected_data, reverse=True)
            data_list = list(shapes.data)
            for i in indices_to_remove:
                if i < len(data_list):
                    data_list.pop(i)
            shapes.data = data_list
            shapes.selected_data = set()
            print(f"[deleted] {len(indices_to_remove)} guide(s)")
        else:
            print("[info] Select a guide first (click on it), then press 'd'")

    def toggle_erase(_=None):
        erase_mode[0] = not erase_mode[0]
        if erase_mode[0]:
            shapes.mode = "select"
            print("[ERASE MODE] Click guides to select, then press 'd' to delete")
        else:
            shapes.mode = "add_path"
            shapes.selected_data = set()
            print("[DRAW MODE] Draw new guides")
        viewer.status = status_line(idx, extract_event_group_from_title(preview_paths[idx]))

    def quit_viewer(_=None):
        viewer.close()

    viewer.bind_key("s", save_current)
    viewer.bind_key("Control-s", save_current)
    viewer.bind_key("n", next_item)
    viewer.bind_key("p", prev_item)
    viewer.bind_key("c", clear_current)
    viewer.bind_key("d", delete_selected)
    viewer.bind_key("e", toggle_erase)
    viewer.bind_key("q", quit_viewer)

    load_item(idx)
    napari.run()


if __name__ == "__main__":
    main()
