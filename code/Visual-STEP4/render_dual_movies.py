#!/usr/bin/env python
"""
Render two MP4 movies (same duration, same camera angles):
  1) 4D ΔF/F movie — plays through time frames
  2) Static reference — rotates 360° over the same duration

Both use the same camera angles at each frame so they can be
placed side-by-side in a presentation.

Usage:
    python code/Visual-STEP4/render_dual_movies.py
"""

import numpy as np
import tifffile
import napari
from pathlib import Path
from tqdm import tqdm
import imageio

# ===== CONFIG =====
DATE = "2026-03-20"
MOUSE = "rbp4cre_139_phpeb"
RUN = "run3"

FPS = 5  # output video fps (matches acquisition)
TILT = 15.0  # elevation tilt (degrees)
NOISE_FLOOR = 0.00  # clip ΔF/F below this

VOXEL_ZYX = (3.9, 1.0, 1.2)  # µm

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

MOVIE_PATH  = BASE / "overlays" / "chunk_01_0000-0575_dff.tif"
STATIC_PATH = BASE / "overlays" / "static_masked_max_over_time.tif"

OUT_MOVIE  = BASE / "overlays" / "movie_4d_dff.mp4"
OUT_STATIC = BASE / "overlays" / "movie_static_rotate.mp4"


def render_video(viewer, n_frames, angle_func, frame_func, out_path, fps, app):
    """Render n_frames to an MP4 using napari screenshots."""
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                format="FFMPEG")
    for i in tqdm(range(n_frames), desc=f"Rendering {out_path.name}"):
        frame_func(i)
        az, el = angle_func(i, n_frames)
        viewer.camera.angles = (az, el, 0)
        app.processEvents()
        img = viewer.screenshot(canvas_only=True)
        writer.append_data(img)
    writer.close()
    print(f"✅ Saved: {out_path}")


def main():
    # ---- Load data ----
    print("Loading 4D movie...")
    movie = tifffile.imread(str(MOVIE_PATH)).astype(np.float32)
    movie = np.clip(movie, NOISE_FLOOR, None) - NOISE_FLOOR
    T = movie.shape[0]
    print(f"  Shape: {movie.shape}, {T} frames")

    print("Loading static reference...")
    static = tifffile.imread(str(STATIC_PATH)).astype(np.float32)
    print(f"  Shape: {static.shape}")

    # ---- Contrast ----
    p5, p95 = np.nanpercentile(movie, 5), np.nanpercentile(movie, 95)
    if p95 <= p5:
        p5, p95 = float(np.nanmin(movie)), float(np.nanmax(movie))

    nz = static[static > 0]
    s_lo, s_hi = float(np.percentile(nz, 2)), float(np.percentile(nz, 98))

    # Rotation: start from side view, rotate 360° around Y axis at 15° tilt
    def angle_fn(i, n):
        # Start from side (elevation=0), rotate full 360°
        el = (i / n) * 360.0
        return (TILT, el)

    # ---- Render 4D movie ----
    print(f"\n--- Rendering 4D movie ({T} frames) ---")
    v1 = napari.Viewer(ndisplay=3)
    v1.window.resize(800, 600)
    v1.add_image(movie, name="ΔF/F", scale=(1.0, *VOXEL_ZYX),
                 colormap="green", rendering="attenuated_mip",
                 contrast_limits=(p5, p95), gamma=0.80)

    def set_time(i):
        v1.dims.set_current_step(0, i)

    v1.camera.angles = (TILT, 0, 0)  # start from side view
    # Let the viewer initialize rendering
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()

    render_video(v1, T, angle_fn, set_time, OUT_MOVIE, FPS, app)
    v1.close()
    del v1, movie

    # ---- Render static rotation ----
    print(f"\n--- Rendering static rotation ({T} frames) ---")
    v2 = napari.Viewer(ndisplay=3)
    v2.window.resize(800, 600)
    v2.add_image(static, name="Static", scale=VOXEL_ZYX,
                 colormap="green", rendering="mip",
                 contrast_limits=(s_lo, s_hi))

    v2.camera.angles = (TILT, 0, 0)  # start from side view

    render_video(v2, T, angle_fn, lambda i: None, OUT_STATIC, FPS, app)
    v2.close()

    print(f"\nDone. Both videos are {T} frames at {FPS} fps = {T/FPS:.1f}s")
    print(f"  {OUT_MOVIE}")
    print(f"  {OUT_STATIC}")


if __name__ == "__main__":
    main()
