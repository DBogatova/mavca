#!/usr/bin/env python
"""
Check raw Z-slices directly to see if horizontal lines are in original data
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

# Config
DATE = "2025-10-29"
MOUSE = "rAi162_15" 
RUN = "run1-crop"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_CLEAN_PATH = BASE / "preprocessed" / "raw_clean.tif"

def main():
    # Load raw data
    with tifffile.TiffFile(RAW_CLEAN_PATH) as tif:
        Z = 32  # Known Z dimension
        
        # Sample timepoints and Z-slices
        ts = [*np.arange(110, 130), *np.arange(590, 605)]
        zs = np.arange(20, 25)
        
        print(f"Checking {len(ts)} timepoints, {len(zs)} Z-slices")
        
        fig, axes = plt.subplots(len(ts), len(zs), figsize=(3*len(zs), 2*len(ts)))
        
        for i, t in enumerate(ts):
            for j, z in enumerate(zs):
                page_idx = t * Z + z
                if page_idx < len(tif.pages):
                    frame = tif.pages[page_idx].asarray().astype(np.float32)
                    
                    ax = axes[i, j] if len(ts) > 1 else axes[j]
                    ax.imshow(frame, vmin=np.percentile(frame, 1), vmax=np.percentile(frame, 99), cmap='gray')
                    
                    if i == 0:
                        ax.set_title(f'Z={z}')
                    if j == 0:
                        ax.set_ylabel(f'T={t}')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(BASE / "raw_slice_check.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved slice check to {BASE / 'raw_slice_check.png'}")

if __name__ == "__main__":
    main()