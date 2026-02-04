#!/usr/bin/env python
"""
Direct raw data viewer - no processing, just like your working code
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

# Config
DATE = "2025-10-29"
MOUSE = "rAi162_15"
RUN = "run1-crop"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
RAW_CLEAN_PATH = BASE / "preprocessed" / "raw_clean.tif"

def main():
    print("Loading raw data directly...")
    
    # Load raw data as 4D array
    with tifffile.TiffFile(RAW_CLEAN_PATH) as tif:
        Z = 32
        n_pages = len(tif.pages)
        T = n_pages // Z
        
        # Load into 4D array (T,Z,Y,X)
        print(f"Loading {T} timepoints, {Z} Z-slices...")
        
        # Load first chunk to get dimensions
        first_page = tif.pages[0].asarray()
        Y, X = first_page.shape
        
        # Create 4D array
        data = np.zeros((T, Z, Y, X), dtype=np.float32)
        
        for t in range(T):
            for z in range(Z):
                page_idx = t * Z + z
                if page_idx < n_pages:
                    data[t, z] = tif.pages[page_idx].asarray().astype(np.float32)
    
    print(f"Data shape: {data.shape}")
    
    # Your exact visualization code
    ts = [*np.arange(170, 200), *np.arange(590, 605)]
    zs = np.arange(15, 23)
    
    print(f"Showing {len(ts)} timepoints, {len(zs)} Z-slices")
    
    fig, axes = plt.subplots(len(ts), len(zs), figsize=(3*len(zs), 2*len(ts)))
    
    for i, t in enumerate(ts):
        for j, z in enumerate(zs):
            if t < data.shape[0] and z < data.shape[1]:
                ax = axes[i, j]
                ax.imshow(
                    data[t, z],
                    vmin=0, 
                    vmax=data.max(),
                    cmap='gray'
                )
                
                if i == 0:
                    ax.set_title(int(z))
                
                if j == 0:
                    ax.set_ylabel(int(t))
                
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(BASE / "raw_direct_view.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved to {BASE / 'raw_direct_view.png'}")

if __name__ == "__main__":
    main()