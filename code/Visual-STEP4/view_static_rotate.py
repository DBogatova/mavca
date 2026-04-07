#!/usr/bin/env python
"""Static reference volume viewer with auto-rotate. Press Ctrl+O to start/stop."""

import napari
import tifffile
import numpy as np
from pathlib import Path
from qtpy.QtCore import QTimer

# ---- Config ----
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data/2026-03-31/rbp4_132_phpeb/run8")
STATIC_PATH = BASE / "overlays" / "static_masked_max_over_time.tif"
VOXEL_ZYX = (3.9, 1.0, 1.2)
import math

TILT = 0
ROT_AMPLITUDE = 45.0  # degrees each direction
ROT_PERIOD = 20.0  # seconds for a full left-right-left cycle
ROT_INTERVAL = 50  # ms between updates

# ---- Load ----
print("Loading...")
static = tifffile.imread(str(STATIC_PATH)).astype(np.float32)
nz = static[static > 0]
s_lo, s_hi = float(np.percentile(nz, 2)), float(np.percentile(nz, 98))

# ---- Viewer ----
v = napari.Viewer(ndisplay=3)
v.add_image(static, name="Static Reference", scale=VOXEL_ZYX,
            colormap="green", rendering="mip",
            contrast_limits=(s_lo, s_hi))
v.camera.angles = (0, 0, 90)
v.scale_bar.visible = True
v.scale_bar.unit = "µm"

# ---- Auto-rotate ----
_tick = [0]
_rot_timer = QTimer()
_rot_timer.setInterval(ROT_INTERVAL)

def _rotate():
    _tick[0] += 1
    t = _tick[0] * ROT_INTERVAL / 1000.0
    angle = ROT_AMPLITUDE * math.sin(2 * math.pi * t / ROT_PERIOD)
    v.camera.angles = (0, angle, 90)

_rot_timer.timeout.connect(_rotate)

@v.bind_key("Control-o")
def _toggle_rotate(viewer):
    if _rot_timer.isActive():
        _rot_timer.stop()
        print("⏸ Rotate off")
    else:
        _rot_timer.start()
        print("🔄 Rotate on")

print("Ctrl+O = toggle rotate")
napari.run()
