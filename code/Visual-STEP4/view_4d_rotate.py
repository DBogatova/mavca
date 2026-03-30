#!/usr/bin/env python
"""4D ΔF/F viewer with auto-rotate. Press Ctrl+O to start/stop rotation."""

import napari
import tifffile
import numpy as np
from pathlib import Path
from qtpy.QtCore import QTimer

# ---- Config ----
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data/2026-03-20/rbp4cre_139_phpeb/run3")
STACK_PATH = BASE / "overlays" / "chunk_01_0000-0575_dff.tif"
VOXEL_ZYX = (3.9, 1.0, 1.2)
NOISE_FLOOR = 0.00
FPS = 5
import math

TILT = 0  # no tilt
ROT_AMPLITUDE = 45.0  # degrees each direction
ROT_PERIOD = 20.0  # seconds for a full left-right-left cycle
ROT_INTERVAL = 50  # ms between updates

# ---- Load ----
print("Loading...")
stack = tifffile.imread(str(STACK_PATH)).astype(np.float32)
stack = np.clip(stack, NOISE_FLOOR, None) - NOISE_FLOOR
T = stack.shape[0]
p5, p95 = np.nanpercentile(stack, 5), np.nanpercentile(stack, 95)

# ---- Viewer ----
v = napari.Viewer(ndisplay=3)
v.add_image(stack, name="ΔF/F", scale=(1.0, *VOXEL_ZYX),
            colormap="green", rendering="mip",
            contrast_limits=(p5, p95), gamma=0.80)
v.camera.angles = (0, 0, 90)
v.scale_bar.visible = True
v.scale_bar.unit = "µm"

# ---- Auto-rotate ----
_tick = [0]
_rot_timer = QTimer()
_rot_timer.setInterval(ROT_INTERVAL)

def _rotate():
    _tick[0] += 1
    t = _tick[0] * ROT_INTERVAL / 1000.0  # seconds
    angle = ROT_AMPLITUDE * math.sin(2 * math.pi * t / ROT_PERIOD)
    v.camera.angles = (0, angle, 90)

_rot_timer.timeout.connect(_rotate)

# ---- Playback timer (5 fps) ----
_play_timer = QTimer()
_play_timer.setInterval(int(1000 / FPS))
_playing = [False]

def _advance():
    t = v.dims.current_step[0]
    v.dims.set_current_step(0, (t + 1) % T)

_play_timer.timeout.connect(_advance)

@v.bind_key("Control-o")
def _toggle_rotate(viewer):
    if _rot_timer.isActive():
        _rot_timer.stop()
        print("⏸ Rotate off")
    else:
        _rot_timer.start()
        print("🔄 Rotate on")

@v.bind_key("p")
def _toggle_play(viewer):
    if _play_timer.isActive():
        _play_timer.stop()
        print("⏸ Playback paused")
    else:
        _play_timer.start()
        print("▶ Playing at 5 fps")

print("Ctrl+O = toggle rotate | p = play/pause at 5 fps")
napari.run()
