#!/usr/bin/env python
"""
Fix mask by keeping only the largest connected component
"""
import numpy as np
import tifffile
from scipy.ndimage import label
from pathlib import Path

# Config
MASK_PATH = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data/2025-12-25/rAi162_phpeb/run1/labelmaps_curated_dynamic/dend_007_labelmap.tif"

def fix_mask(mask_path):
    # Load mask
    mask = tifffile.imread(mask_path).astype(bool)
    
    # Find connected components
    labeled, num_features = label(mask)
    
    if num_features <= 1:
        print("Mask already has only one component")
        return
    
    # Find largest component
    component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest_idx = np.argmax(component_sizes) + 1
    
    print(f"Found {num_features} components, keeping largest ({component_sizes[largest_idx-1]} voxels)")
    
    # Keep only largest component
    fixed_mask = (labeled == largest_idx).astype(np.uint16)
    
    # Save fixed mask
    tifffile.imwrite(mask_path, fixed_mask)
    print(f"Fixed mask saved to {mask_path}")

if __name__ == "__main__":
    fix_mask(MASK_PATH)