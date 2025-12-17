from pathlib import Path
import tifffile, numpy as np

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") \
       / "2025-12-02" / "rbp4cre_136_phpeb" / "run5"
SURF_TIF = BASE / "raw" / "surface_roi_2d.tif"
SURF_NPY = BASE / "raw" / "surface_roi_2d.npy"

surf = tifffile.imread(SURF_TIF).astype(bool)
np.save(SURF_NPY, surf)
print("Saved surface ROI:", SURF_NPY, surf.shape, surf.sum(), "pixels")
