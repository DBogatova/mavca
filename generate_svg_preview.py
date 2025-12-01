#!/usr/bin/env python
"""
Generate SVG vector version of dendrite preview
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
from pathlib import Path

# Set CMU Serif font
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.serif'] = ['CMU Serif']

# Config
DATE = "2025-10-29"
MOUSE = "rAi162_15"
RUN = "run1-crop"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN

def main():
    # Load mask and background
    mask_path = BASE / "labelmaps" / "dend_000_labelmap.tif"
    bg_path = BASE / "labelmap_backgrounds" / "dend_000_background_2dMIP.tif"
    
    if not mask_path.exists() or not bg_path.exists():
        print(f"Files not found: {mask_path} or {bg_path}")
        return
    
    # Load data
    mask_3d = tifffile.imread(mask_path).astype(bool)
    bg_2d = tifffile.imread(bg_path).astype(np.float32)
    
    # Create Z-MIP of mask
    mask_mip = np.max(mask_3d, axis=0).astype(bool)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Show background
    ax.imshow(bg_2d, cmap="gray", alpha=0.8)
    
    # Get mask edges for vector outline
    edges = cv2.Canny((mask_mip.astype(np.uint8) * 255), 0, 1) > 0
    
    # Show edges as vector overlay
    ax.imshow(np.ma.masked_where(~edges, edges), cmap="autumn", alpha=0.9)
    
    ax.set_title("dend_000 (Vector SVG)", fontsize=14)
    ax.axis("off")
    
    # Save as SVG
    output_path = BASE / "dend_000_preview.svg"
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"SVG saved to: {output_path}")

if __name__ == "__main__":
    main()