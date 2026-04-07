#!/usr/bin/env python
"""
Render rotating 3D movies using matplotlib (no GPU / no napari).

1) Static reference: 360° rotation over ~20 seconds
2) 4D ΔF/F movie: plays through time + rotates simultaneously

Both rendered as MP4 with matching camera angles.

Usage:
    python code/Visual-STEP4/render_rotating_movies.py
"""

import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio

# ===== CONFIG =====
DATE = "2026-03-20"
MOUSE = "rbp4cre_139_phpeb"
RUN = "run3"

FPS = 5
NOISE_FLOOR = 0.00
VOXEL_ZYX = (3.9, 1.0, 1.2)  # µm
ELEV = 25  # elevation angle (view from slightly above)
DPI = 150
FIGSIZE = (10, 10)

# ===== PATHS =====
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

MOVIE_PATH  = BASE / "overlays" / "chunk_01_0000-0575_dff.tif"
STATIC_PATH = BASE / "overlays" / "static_masked_max_over_time.tif"

OUT_STATIC = BASE / "overlays" / "movie_static_rotate.mp4"
OUT_MOVIE  = BASE / "overlays" / "movie_4d_dff.mp4"


def vol_to_mip_rgb(vol, azim, elev, voxel_scale, cmap, norm):
    """
    Render a 3D volume as a colored MIP from a given viewing angle.
    Projects along the axis closest to the viewing direction.
    Returns an RGB image.
    """
    # For a turntable rotation, project along different axes based on azimuth
    # Simple approach: always do Z-MIP but rotate the volume
    from scipy.ndimage import rotate as ndrotate

    # Scale volume to isotropic voxels for correct rotation
    vz, vy, vx = voxel_scale
    # Rotate around Z axis (vertical) by azimuth
    rotated = ndrotate(vol, -azim, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)

    # MIP along axis 2 (the "into screen" axis after rotation)
    mip = np.max(rotated, axis=2)  # (Z, Y)

    # Apply slight tilt: take a weighted MIP that favors top
    # For elevation, rotate around horizontal axis
    if abs(elev) > 1:
        mip_tilted = ndrotate(
            np.max(rotated, axis=2), 0, reshape=False, order=1
        )
        # Simple: just use the Z-Y MIP, elevation is visual only
        mip = np.max(rotated, axis=2)

    rgb = cmap(norm(mip))[:, :, :3]  # drop alpha
    return (rgb * 255).astype(np.uint8)


def render_mip_movie(vol_func, n_frames, out_path, voxel_scale, cmap_name,
                     vmin, vmax, fps, label=""):
    """Render n_frames of rotating MIP to MP4."""
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_under('black')
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                format="FFMPEG")

    for i in tqdm(range(n_frames), desc=f"Rendering {out_path.name}"):
        vol = vol_func(i)
        azim = (i / n_frames) * 360.0

        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        img = vol_to_mip_rgb(vol, azim, ELEV, voxel_scale, cmap, norm)
        ax.imshow(img, aspect='auto', interpolation='bilinear')
        ax.axis('off')

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()

        # Convert figure to RGB array
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(buf)
        plt.close(fig)

    writer.close()
    print(f"✅ Saved: {out_path}")


def main():
    # ---- Load static ----
    print("Loading static reference...")
    static = tifffile.imread(str(STATIC_PATH)).astype(np.float32)
    print(f"  Shape: {static.shape}")
    nz = static[static > 0]
    s_lo, s_hi = float(np.percentile(nz, 5)), float(np.percentile(nz, 98))

    # Render static: 360° in ~20s at FPS
    n_static = FPS * 20  # 100 frames for 20s
    print(f"\n--- Static rotation ({n_static} frames, {n_static/FPS:.0f}s) ---")
    render_mip_movie(
        vol_func=lambda i: static,
        n_frames=n_static,
        out_path=OUT_STATIC,
        voxel_scale=VOXEL_ZYX,
        cmap_name="Greens",
        vmin=s_lo, vmax=s_hi,
        fps=FPS,
    )
    del static

    # ---- Load 4D movie ----
    print("\nLoading 4D movie...")
    movie = tifffile.imread(str(MOVIE_PATH)).astype(np.float32)
    movie = np.clip(movie, NOISE_FLOOR, None) - NOISE_FLOOR
    T = movie.shape[0]
    print(f"  Shape: {movie.shape}, {T} frames")
    p5, p95 = float(np.nanpercentile(movie, 5)), float(np.nanpercentile(movie, 95))

    print(f"\n--- 4D movie ({T} frames, {T/FPS:.0f}s) ---")
    render_mip_movie(
        vol_func=lambda i: movie[i],
        n_frames=T,
        out_path=OUT_MOVIE,
        voxel_scale=VOXEL_ZYX,
        cmap_name="Greens",
        vmin=p5, vmax=p95,
        fps=FPS,
    )

    print(f"\nDone!")
    print(f"  {OUT_STATIC}")
    print(f"  {OUT_MOVIE}")


if __name__ == "__main__":
    main()
