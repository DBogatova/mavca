#!/usr/bin/env python
"""
Quick debug script to check where horizontal line artifacts come from
"""

import numpy as np
import tifffile
from pathlib import Path

# Config
DATE = "2025-10-29"
MOUSE = "rAi162_15"
RUN = "run1-crop"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_ORIG_PATH = BASE / "raw" / f"runB_run1_rAi162_15_v1_reslice-crop.tif"
RAW_CLEAN_PATH = BASE / "preprocessed" / "raw_clean.tif"

def check_artifacts(path, name):
    print(f"\n=== Checking {name}: {path} ===")
    
    with tifffile.TiffFile(path) as tif:
        # Load first few frames
        frame1 = tif.pages[0].asarray().astype(np.float32)
        frame2 = tif.pages[28].asarray().astype(np.float32)  # Assuming Z=28
        frame3 = tif.pages[56].asarray().astype(np.float32)
        
        print(f"Frame shape: {frame1.shape}")
        
        # Check for horizontal patterns
        for i, frame in enumerate([frame1, frame2, frame3], 1):
            # Row-wise variance (high variance = potential line artifacts)
            row_var = np.var(frame, axis=1)
            high_var_rows = np.sum(row_var > np.percentile(row_var, 95))
            
            print(f"Frame {i}: High variance rows: {high_var_rows}/{frame.shape[0]} ({100*high_var_rows/frame.shape[0]:.1f}%)")
            
            # Check for obvious horizontal lines (rows with extreme values)
            row_max = np.max(frame, axis=1)
            row_min = np.min(frame, axis=1)
            extreme_rows = np.sum((row_max > np.percentile(row_max, 99)) | (row_min < np.percentile(row_min, 1)))
            
            print(f"Frame {i}: Extreme value rows: {extreme_rows}/{frame.shape[0]} ({100*extreme_rows/frame.shape[0]:.1f}%)")

if __name__ == "__main__":
    # Check original raw data
    if RAW_ORIG_PATH.exists():
        check_artifacts(RAW_ORIG_PATH, "Original Raw")
    
    # Check motion-corrected data
    if RAW_CLEAN_PATH.exists():
        check_artifacts(RAW_CLEAN_PATH, "Motion Corrected")
    
    # Check ΔF/F stack if it exists
    dff_path = BASE / "preprocessed" / "dff_stack.tif"
    if dff_path.exists():
        check_artifacts(dff_path, "ΔF/F Stack")