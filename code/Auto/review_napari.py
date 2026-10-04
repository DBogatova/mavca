#!/usr/bin/env python3
"""review_napari.py - napari viewer for reviewing auto-detected dendrite masks.

Displays:
  - Mean image (reference) as the base layer
  - Human masks (if available) as outlines/labels in one color
  - Auto masks as a separate label layer
  - Click on a dendrite to see its ID and trace (auto vs matched human) in a dock plot

Keys:
  click - select a dendrite (Shift+click adds to the selection); its auto trace and the
          trace of the human mask it overlaps most are plotted in the dock
  D - delete the selected auto mask(s)
  M - merge the selected auto masks into the first one selected
  S - save edits to auto_labelmap_reviewed.tif + edit log
  R - reset to original auto_labelmap.tif

Edits are saved as a NEW file OUT/masks/auto_labelmap_reviewed.tif. The original
auto_labelmap.tif is never overwritten. Old reviewed versions are moved to
OUT/masks/old/. Downstream stages (extract_traces.py) should prefer
auto_labelmap_reviewed.tif when present.

Usage:
  $PY code/Auto/review_napari.py --run DATE/MOUSE/RUN
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scape_common import PROJECT, get_run, open_stack, human_masks, human_labelmap, Run


def compute_mean_image(r: Run) -> np.ndarray:
    """Compute mean projection over time, max over Z."""
    stack = open_stack(r)
    T, Z, Y, X = stack.shape
    # Process in chunks to save memory
    chunk_size = 50
    mean_acc = np.zeros((Z, Y, X), dtype=np.float64)
    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)
        chunk = stack[t0:t1].astype(np.float32)
        mean_acc += chunk.sum(axis=0)
    mean_zyx = (mean_acc / T).astype(np.float32)
    # Max over Z for display
    return mean_zyx.max(axis=0)


def load_auto_labelmap(r: Run) -> tuple[np.ndarray | None, Path | None]:
    """Load the auto labelmap. Prefer reviewed version if it exists."""
    reviewed = r.out / "masks" / "auto_labelmap_reviewed.tif"
    original = r.out / "masks" / "auto_labelmap.tif"
    import tifffile
    if reviewed.exists():
        return tifffile.imread(str(reviewed)), reviewed
    if original.exists():
        return tifffile.imread(str(original)), original
    return None, None


def load_original_auto_labelmap(r: Run) -> np.ndarray | None:
    """Load the original (non-reviewed) auto labelmap."""
    original = r.out / "masks" / "auto_labelmap.tif"
    if not original.exists():
        return None
    import tifffile
    return tifffile.imread(str(original))


def load_dff_traces(r: Run, which: str = "auto") -> dict | None:
    """Load dF/F traces from CSV. Returns {name: (time_s, values)}."""
    import pandas as pd
    if which == "auto":
        path = r.out / "traces" / "dff_auto.csv"
    else:
        path = r.out / "traces" / "dff_human_sameextractor.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    traces = {}
    time_col = "time_s" if "time_s" in df.columns else None
    if time_col is None:
        return None
    time_s = df[time_col].values
    for col in df.columns:
        if col in ("Frame", "time_s"):
            continue
        traces[col] = (time_s, df[col].values)
    return traces


def save_reviewed_labelmap(r: Run, labelmap: np.ndarray, edit_log: list[dict]):
    """Save the reviewed labelmap and edit log."""
    import tifffile
    masks_dir = r.outdir("masks")
    old_dir = masks_dir / "old"
    old_dir.mkdir(exist_ok=True)

    reviewed_path = masks_dir / "auto_labelmap_reviewed.tif"
    log_path = masks_dir / "auto_labelmap_reviewed_log.json"

    # Move old versions to old/
    if reviewed_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(reviewed_path), str(old_dir / f"auto_labelmap_reviewed_{ts}.tif"))
    if log_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(log_path), str(old_dir / f"auto_labelmap_reviewed_log_{ts}.json"))

    # Save new
    tifffile.imwrite(str(reviewed_path), labelmap.astype(np.uint16), compression="zlib")
    log_path.write_text(json.dumps(edit_log, indent=2, default=str))
    return reviewed_path, log_path


def main_gui(r: Run):
    """Launch the napari viewer."""
    import napari
    from napari.utils.notifications import show_info
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from PyQt5 import QtWidgets

    print(f"Loading data for {r.key}...")

    # Load mean image
    mean_img = compute_mean_image(r)
    print(f"  mean image: {mean_img.shape}")

    # Load masks
    auto_lab, auto_path = load_auto_labelmap(r)
    original_lab = load_original_auto_labelmap(r)

    # Get stack shape for human mask padding
    stack = open_stack(r)
    shape_zyx = stack.shape[1:]

    human_lab = human_labelmap(r, shape_zyx) if r.human_mask_dir else None
    human_list = human_masks(r, shape_zyx) if r.human_mask_dir else []

    # Load traces
    auto_traces = load_dff_traces(r, "auto")
    human_traces = load_dff_traces(r, "human")

    # Create viewer
    viewer = napari.Viewer(title=f"Review masks - {r.key}")

    # Add mean image
    viewer.add_image(mean_img, name="mean", colormap="gray", blending="additive")

    # Add human masks if available (max projection for 2D display)
    if human_lab is not None:
        human_2d = human_lab.max(axis=0)
        viewer.add_labels(human_2d, name="human masks", opacity=0.3)
        print(f"  human masks: {len(human_list)} dendrites")

    # Add auto masks
    if auto_lab is not None:
        auto_2d = auto_lab.max(axis=0)
        auto_layer = viewer.add_labels(auto_2d, name="auto masks", opacity=0.5)
        print(f"  auto masks: {auto_lab.max()} dendrites")
    else:
        # never offer human masks as editable "auto" masks: saving them would silently
        # replace the automatic result with the human one downstream
        show_info("No auto masks for this run yet - run detection first (scape run KEY)")
        napari.run()
        return

    # Track edits
    edit_log = []
    current_labelmap = auto_lab.copy() if auto_lab is not None else None

    # Create dock widget for trace plot
    fig, ax = plt.subplots(figsize=(8, 3))
    canvas = FigureCanvas(fig)
    dock = viewer.window.add_dock_widget(canvas, name="Trace", area="bottom")

    human_names = [n for n, _ in human_list]
    selection: list[int] = []

    def best_human(label: int):
        """Human mask overlapping this auto unit the most (by voxels), or None."""
        if human_lab is None:
            return None
        hv = human_lab[current_labelmap == label]
        hv = hv[hv > 0]
        if not hv.size:
            return None
        return human_names[int(np.bincount(hv).argmax()) - 1]

    def update_plot():
        ax.clear()
        for k, lab_id in enumerate(selection[:4]):
            name = f"dend_{lab_id - 1:03d}"          # auto trace names follow label id - 1
            if auto_traces and name in auto_traces:
                t, v = auto_traces[name]
                ax.plot(t, v, lw=0.8, label=f"auto {name}")
            hn = best_human(lab_id)
            if hn and human_traces and hn in human_traces:
                t, v = human_traces[hn]
                ax.plot(t, v, "--", lw=0.8, label=f"human {hn} (best overlap)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("dF/F (%)")
        ax.set_title("selected: " + ", ".join(str(x) for x in selection) if selection else "click a dendrite")
        if selection:
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        canvas.draw()

    # Click = select one dendrite; Shift+click = add to the selection (for merge / delete)
    @auto_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        if event.type != "mouse_press":
            return
        coords = layer.world_to_data(event.position)
        y, x = int(round(coords[-2])), int(round(coords[-1]))
        if not (0 <= y < layer.data.shape[0] and 0 <= x < layer.data.shape[1]):
            return
        label = int(layer.data[y, x])
        if label <= 0:
            return
        if "Shift" in event.modifiers:
            if label not in selection:
                selection.append(label)
        else:
            selection[:] = [label]
        update_plot()
        show_info(f"selected {selection}")

    def refresh():
        auto_layer.data = current_labelmap.max(axis=0)
        selection.clear()
        update_plot()

    @viewer.bind_key("d", overwrite=True)
    def delete_mask(viewer):
        if not selection:
            show_info("select a dendrite first (click)")
            return
        for lab_id in selection:
            current_labelmap[current_labelmap == lab_id] = 0
            edit_log.append({"action": "delete", "label": int(lab_id), "time": datetime.now().isoformat()})
        show_info(f"deleted {selection} (press S to save)")
        refresh()

    @viewer.bind_key("m", overwrite=True)
    def merge_masks(viewer):
        if len(selection) < 2:
            show_info("Shift+click at least two dendrites, then press M")
            return
        target = selection[0]
        for lab_id in selection[1:]:
            current_labelmap[current_labelmap == lab_id] = target
            edit_log.append({"action": "merge", "from": int(lab_id), "to": int(target), "time": datetime.now().isoformat()})
        show_info(f"merged {selection[1:]} into {target} (press S to save)")
        refresh()

    @viewer.bind_key("s", overwrite=True)
    def save_edits(viewer):
        paths = save_reviewed_labelmap(r, current_labelmap, edit_log)
        show_info(f"Saved to {paths[0].name}; rerun: scape run {r.key} (traces onward)")
        print(f"Saved: {paths[0]}\nLog: {paths[1]}")

    @viewer.bind_key("r", overwrite=True)
    def reset_to_original(viewer):
        nonlocal current_labelmap, edit_log
        if original_lab is None:
            show_info("No original labelmap to reset to")
            return
        current_labelmap = original_lab.copy()
        edit_log = [{"action": "reset", "time": datetime.now().isoformat()}]
        show_info("Reset to original auto_labelmap.tif (press S to save)")
        refresh()

    # Show instructions
    show_info("Click = select (Shift+click adds). D = delete, M = merge selection, S = save, R = reset.")

    napari.run()


def main_offscreen(r: Run) -> int:
    """Offscreen test: verify data can be loaded without error."""
    print(f"Offscreen test for {r.key}...")

    # Check we can compute mean image
    mean_img = compute_mean_image(r)
    print(f"  mean image: {mean_img.shape}, dtype={mean_img.dtype}")

    # Check masks
    auto_lab, auto_path = load_auto_labelmap(r)
    if auto_lab is not None:
        print(f"  auto labelmap: {auto_lab.shape}, max label={auto_lab.max()}")
    else:
        print("  auto labelmap: not found")

    # Check human masks
    stack = open_stack(r)
    shape_zyx = stack.shape[1:]
    human_lab = human_labelmap(r, shape_zyx) if r.human_mask_dir else None
    if human_lab is not None:
        print(f"  human labelmap: {human_lab.shape}, max label={human_lab.max()}")
    else:
        print("  human labelmap: not found")

    # If no auto masks but human masks exist, we can use human as stand-in
    if auto_lab is None and human_lab is not None:
        print("  (will use human masks as stand-in for review)")
    elif auto_lab is None and human_lab is None:
        print("  WARNING: no masks available for this run")

    print("Offscreen test PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run key (DATE/MOUSE/RUN)")
    ap.add_argument("--offscreen", action="store_true", help="offscreen test (no GUI)")
    args = ap.parse_args(argv)

    try:
        r = get_run(args.run)
    except KeyError:
        print(f"ERROR: run '{args.run}' not found")
        return 1

    if r.raw is None:
        print(f"ERROR: no raw stack for {r.key}")
        return 1

    if args.offscreen:
        return main_offscreen(r)
    else:
        main_gui(r)
        return 0


if __name__ == "__main__":
    sys.exit(main())
