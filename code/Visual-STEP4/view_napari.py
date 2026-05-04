#!/usr/bin/env python
"""
Quick Napari viewer for 4D ΔF/F data with FOV box edges and scale bar (µm)
"""

import napari
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ---- Config ----
STACK_PATH = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data/2026-04-16/rbp4_132_phpeb/run7/overlays/chunk_01_0000-0535_dff.tif"

# (T, Z, Y, X): time left as frames; spatial voxels in µm
VOXEL_SCALE = (1.0, 3.9, 1.0, 1.2)  # T, Z, Y, X

print(f"Loading: {STACK_PATH}")
stack = tifffile.imread(STACK_PATH)
print(f"Stack shape: {stack.shape}")  # (T, Z, Y, X)

# Clip low values to black — removes noise speckles when nothing fires
NOISE_FLOOR = 0.00  # ΔF/F below this → 0 (adjust if needed)
stack = np.clip(stack, NOISE_FLOOR, None) - NOISE_FLOOR

_, Z, Y, X = stack.shape

# Print ΔF/F value range for colorbar reference
min_val = np.nanmin(stack)
max_val = np.nanmax(stack)
p5 = np.nanpercentile(stack, 5)
p95 = np.nanpercentile(stack, 95)
print(f"ΔF/F range: {min_val:.3f} to {max_val:.3f}")
print(f"ΔF/F 5-95%: {p5:.3f} to {p95:.3f}")

# Ensure valid contrast range
if p95 <= p5:
    contrast_min, contrast_max = min_val, max_val
else:
    contrast_min, contrast_max = p5, p95

# ---- Create separate colorbar figure ----
# Set CMU Serif font
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.serif'] = ['CMU Serif']

fig, ax = plt.subplots(figsize=(2, 6))
cmap = cm.get_cmap('turbo')
norm = plt.Normalize(vmin=contrast_min, vmax=contrast_max)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cb.set_label('ΔF/F (% change)', rotation=270, labelpad=20)
ax.remove()  # Remove the axes, keep only colorbar
plt.tight_layout()

# Save as vector formats
colorbar_path = Path(STACK_PATH).parent / "colorbar_dff"
fig.savefig(f"{colorbar_path}.svg", format='svg', bbox_inches='tight')
fig.savefig(f"{colorbar_path}.pdf", format='pdf', bbox_inches='tight')
print(f"Saved colorbar: {colorbar_path}.svg and {colorbar_path}.pdf")

plt.show()

# ---- Napari viewer ----
viewer = napari.Viewer(ndisplay=3)
layer = viewer.add_image(
    stack,
    name="ΔF/F Branches",
    scale=VOXEL_SCALE,
    colormap="turbo",
    rendering="attenuated_mip",
    contrast_limits=(contrast_min, contrast_max),
)

# ---- 3D Field of View edges ----
z0, z1 = 0, Z - 1
y0, y1 = 0, Y - 1
x0, x1 = 0, X - 1

C = {
    "000": np.array([z0, y0, x0]),
    "001": np.array([z0, y0, x1]),
    "010": np.array([z0, y1, x0]),
    "011": np.array([z0, y1, x1]),
    "100": np.array([z1, y0, x0]),
    "101": np.array([z1, y0, x1]),
    "110": np.array([z1, y1, x0]),
    "111": np.array([z1, y1, x1]),
}

edges = [
    np.stack([C["000"], C["001"]]),  # bottom rectangle
    np.stack([C["001"], C["011"]]),
    np.stack([C["011"], C["010"]]),
    np.stack([C["010"], C["000"]]),
    np.stack([C["100"], C["101"]]),  # top rectangle
    np.stack([C["101"], C["111"]]),
    np.stack([C["111"], C["110"]]),
    np.stack([C["110"], C["100"]]),
    np.stack([C["000"], C["100"]]),  # verticals
    np.stack([C["001"], C["101"]]),
    np.stack([C["010"], C["110"]]),
    np.stack([C["011"], C["111"]]),
]


# ---- Scale bar ----
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "µm"
viewer.scale_bar.position = "bottom_right"
viewer.scale_bar.color = "white"
viewer.scale_bar.ticks = True  # show tick marks
viewer.scale_bar.font_size = 10

# ---- Playback speed ----
# Napari default is 10 fps. Change to 5 in the play controls
# (click play button → adjust fps spinbox next to it)


print(f"\nColorbar range: {contrast_min:.1%} to {contrast_max:.1%} fluorescence change")
print("\nΔF/F meaning:")
print(f"  0.0 = no change (baseline)")
print(f"  0.1 = 10% increase (moderate calcium)")
print(f"  0.5 = 50% increase (strong calcium)")
print(f"  1.0 = 100% increase (very strong calcium)")
print(f" -0.1 = 10% decrease (below baseline)")

napari.run()
