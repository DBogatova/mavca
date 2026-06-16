#!/usr/bin/env python
"""
FOV geometry diagnostics: (a) illumination profile vs X (and Z), (b) sloped pial
surface Y_surf(X,Z) plane fit. Informs the improved split test.

Surface = first Y (from top=0, superficial) where the time-mean intensity in a
(Z,X) column crosses a fraction of that column's dynamic range. Plane fit gives
the surface tilt (deg) in the Y-X and Y-Z planes.
"""
from pathlib import Path
import numpy as np, tifffile, gc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
VOXEL = np.array([3.9, 1.0, 1.2])     # Z,Y,X um
FOVS = [
    ("0508",    "2026-05-08", "rbp4_139_phpeb", "run5", 14, 5.0),
    ("0416_r1", "2026-04-16", "rbp4_132_phpeb", "run1", 12, 5.0),
    ("0512_r9", "2026-05-12", "rbp4_132_phpeb", "run9", 12, 5.0),
]
NSAMP = 120

def raw_path(base, run):
    c = [p for p in sorted((base/run/"raw").glob("*reslice-bin.tif")) if "dff" not in p.name]
    return c[0] if c else None

def mean_volume(rawp, skip, fs):
    tf = tifffile.TiffFile(str(rawp)); T, Z, Y, X = tf.series[0].shape
    try: arr = tifffile.memmap(str(rawp))
    except Exception: arr = tf.series[0].asarray()
    idx = np.unique(np.linspace(int(skip*fs), T-1, NSAMP).astype(int))
    mv = np.asarray(arr[idx]).astype(np.float32).mean(0)   # (Z,Y,X)
    tf.close(); return mv, (T,Z,Y,X)

def surface_YofXZ(mv, frac=0.5, min_col=None):
    """For each (Z,X) column return first Y from top crossing frac*range; NaN if weak."""
    Z, Y, X = mv.shape
    base = np.percentile(mv, 20)
    col_max = mv.max(1)                          # (Z,X)
    if min_col is None: min_col = np.percentile(col_max, 40)
    thr = base + frac*(col_max - base)           # (Z,X)
    surf = np.full((Z, X), np.nan)
    for z in range(Z):
        for x in range(X):
            if col_max[z, x] < min_col: continue
            yy = np.where(mv[z, :, x] > thr[z, x])[0]
            if yy.size: surf[z, x] = yy[0]
    return surf

def fit_plane(surf):
    Z, X = surf.shape
    zz, xx = np.mgrid[0:Z, 0:X]
    m = np.isfinite(surf)
    A = np.column_stack([xx[m], zz[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, surf[m], rcond=None)   # Y = a*X + b*Z + c (voxels)
    a, b, c = coef
    pred = A @ coef
    resid = surf[m] - pred
    # slope in physical units: dY(um)/dX(um) = a*VOXEL_Y/VOXEL_X
    slope_x_deg = np.degrees(np.arctan2(a*VOXEL[1], VOXEL[2]))
    slope_z_deg = np.degrees(np.arctan2(b*VOXEL[1], VOXEL[0]))
    return dict(a=a, b=b, c=c, slope_x_deg=slope_x_deg, slope_z_deg=slope_z_deg,
                rms_vox=float(np.sqrt(np.mean(resid**2))), n=int(m.sum()))

def main():
    fig, axes = plt.subplots(len(FOVS), 3, figsize=(15, 4*len(FOVS)))
    for i, (lab, d, mo, run, skip, fs) in enumerate(FOVS):
        base = PR/"scape-data"/d/mo
        rawp = raw_path(base, run)
        mv, shp = mean_volume(rawp, skip, fs)
        Z, Y, X = mv.shape
        # illumination: mean over Z,Y (tissue only: Y below surface) vs X
        illum_x = mv.mean((0,1))                 # (X,)
        illum_z = mv.mean((1,2))                 # (Z,)
        surf = surface_YofXZ(mv)
        pf = fit_plane(surf)
        surf_x = np.nanmean(surf, axis=0)        # mean over Z -> Y_surf vs X
        lo, hi = illum_x[:X//4].mean(), illum_x[-X//4:].mean()
        print(f"[{lab}] shape(T,Z,Y,X)={shp}")
        print(f"   illumination X: left-quartile={lo:.0f} right-quartile={hi:.0f} "
              f"(right/left={hi/max(lo,1e-6):.2f}x)")
        print(f"   surface plane: Y_surf = {pf['a']:.3f}*X + {pf['b']:.3f}*Z + {pf['c']:.1f} (vox); "
              f"tilt_X={pf['slope_x_deg']:+.1f}deg tilt_Z={pf['slope_z_deg']:+.1f}deg "
              f"rms={pf['rms_vox']:.1f}vox (n={pf['n']})")
        print(f"   surface Y span across X: {np.nanmin(surf_x):.0f}..{np.nanmax(surf_x):.0f} vox "
              f"({(np.nanmax(surf_x)-np.nanmin(surf_x))*VOXEL[1]:.0f} um)")
        # plots
        ax = axes[i] if len(FOVS) > 1 else axes
        ax[0].plot(illum_x, color='tab:orange'); ax[0].set_title(f"{lab}: illumination vs X")
        ax[0].set_xlabel("X (lateral)"); ax[0].set_ylabel("mean intensity"); ax[0].grid(alpha=.3)
        ax[1].imshow(mv.max(0), cmap='gray', aspect='auto')   # Y x X MIP (like movie)
        ax[1].plot(np.arange(X), surf_x, 'r-', lw=1.2)
        ax[1].set_title(f"{lab}: Y×X MIP + surface (tilt_X={pf['slope_x_deg']:+.1f}°)")
        ax[1].set_xlabel("X"); ax[1].set_ylabel("Y (depth)")
        ax[2].plot(illum_z, color='tab:green'); ax[2].set_title(f"{lab}: illumination vs Z")
        ax[2].set_xlabel("Z (galvo)"); ax[2].set_ylabel("mean intensity"); ax[2].grid(alpha=.3)
        del mv; gc.collect()
    fig.tight_layout()
    (PR/"code"/"Testing"/"figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(PR/"code"/"Testing"/"figures"/"fov_geometry.png", dpi=140)
    print("\nSaved figures/fov_geometry.png")

if __name__ == "__main__":
    main()
