#!/usr/bin/env python
"""
Dual Napari viewer: 4D ΔF/F movie + static reference.
Two windows, synced camera, ±45° pendulum swing (no full rotation = no GPU crash).

Ctrl+O  = toggle auto-rotate
p       = play/pause 4D movie at 5 fps
"""

import napari
import tifffile
import numpy as np
import math
from pathlib import Path
from qtpy.QtCore import QTimer

# ---- Config ----
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data/2026-03-20/rbp4cre_139_phpeb/run3")

MOVIE_PATH  = BASE / "overlays" / "chunk_01_0000-0575_dff.tif"
STATIC_PATH = BASE / "overlays" / "static_masked_max_over_time.tif"

VOXEL_ZYX = (3.9, 1.0, 1.2)
NOISE_FLOOR = 0
FPS = 5

ROT_AMPLITUDE = 45.0  # degrees each direction
ROT_PERIOD = 20.0     # seconds per full swing cycle
ROT_INTERVAL = 50     # ms between updates

# ---- Load data ----
print("Loading 4D movie...")
movie = tifffile.imread(str(MOVIE_PATH)).astype(np.float32)
movie = np.clip(movie, NOISE_FLOOR, None) - NOISE_FLOOR
T = movie.shape[0]
p5, p95 = np.nanpercentile(movie, 5), np.nanpercentile(movie, 95)

print("Loading static reference...")
static = tifffile.imread(str(STATIC_PATH)).astype(np.float32)
nz = static[static > 0]
s_lo, s_hi = float(np.percentile(nz, 2)), float(np.percentile(nz, 98))

# ---- Viewer 1: 4D movie ----
v1 = napari.Viewer(ndisplay=3, title="ΔF/F Movie")
v1.add_image(movie, name="ΔF/F Movie", scale=(1.0, *VOXEL_ZYX),
             colormap="green", rendering="mip",
             contrast_limits=(p5, p95), gamma=0.80)
v1.camera.angles = (0, 0, 90)
v1.scale_bar.visible = True
v1.scale_bar.unit = "µm"

# ---- Viewer 2: static reference ----
v2 = napari.Viewer(ndisplay=3, title="Static Reference")
v2.add_image(static, name="Static Reference", scale=VOXEL_ZYX,
             colormap="green", rendering="mip",
             contrast_limits=(s_lo, s_hi))
v2.camera.angles = (0, 0, 90)
v2.scale_bar.visible = True
v2.scale_bar.unit = "µm"

# ---- Sync cameras (manual drag) ----
_syncing = False

def _make_sync(src_viewer, dst_viewer):
    def _on_change(event):
        global _syncing
        if _syncing:
            return
        _syncing = True
        try:
            dst_viewer.camera.angles = src_viewer.camera.angles
            dst_viewer.camera.zoom = src_viewer.camera.zoom
        finally:
            _syncing = False
    return _on_change

_sync_1to2 = _make_sync(v1, v2)
_sync_2to1 = _make_sync(v2, v1)

v1.camera.events.angles.connect(_sync_1to2)
v1.camera.events.zoom.connect(_sync_1to2)
v2.camera.events.angles.connect(_sync_2to1)
v2.camera.events.zoom.connect(_sync_2to1)

# ---- Auto-rotate (±45° pendulum) ----
_tick = [0]
_rot_timer = QTimer()
_rot_timer.setInterval(ROT_INTERVAL)

def _rotate():
    _tick[0] += 1
    t = _tick[0] * ROT_INTERVAL / 1000.0
    angle = ROT_AMPLITUDE * math.sin(2 * math.pi * t / ROT_PERIOD)
    v1.camera.angles = (0, angle, 90)
    # v2 follows via camera sync

_rot_timer.timeout.connect(_rotate)

@v1.bind_key("Control-o")
def _toggle_rotate(viewer):
    if _rot_timer.isActive():
        _rot_timer.stop()
        print("⏸ Rotate off")
    else:
        _rot_timer.start()
        print("🔄 Rotate on")

@v2.bind_key("Control-o")
def _toggle_rotate2(viewer):
    _toggle_rotate(viewer)

# ---- Playback (5 fps) ----
_play_timer = QTimer()
_play_timer.setInterval(int(1000 / FPS))

def _advance():
    t = v1.dims.current_step[0]
    v1.dims.set_current_step(0, (t + 1) % T)

_play_timer.timeout.connect(_advance)

@v1.bind_key("p")
def _toggle_play(viewer):
    if _play_timer.isActive():
        _play_timer.stop()
        print("⏸ Paused")
    else:
        _play_timer.start()
        print("▶ Playing at 5 fps")

@v2.bind_key("p")
def _toggle_play2(viewer):
    _toggle_play(viewer)

print("\nCtrl+O = toggle rotate | p = play/pause movie")
print("Drag either viewer to rotate both manually.")
napari.run()
