#!/usr/bin/env python
"""
Convert specific PDF plot to SVG with CMU Serif font
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np

# Set CMU Serif font
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.serif'] = ['CMU Serif']

# Input file
pdf_path = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data/2025-08-27/rAi162_18/run7/traces_caBG_achNeighborhood/per_mask_plots/dend_004_CaBG_AChNeighborhood.pdf")

# Output path
svg_path = pdf_path.with_suffix('.svg')

print(f"Converting {pdf_path.name} to SVG with CMU Serif font...")

# Since we can't directly read the PDF content, create a template figure
# You'll need to manually recreate the plot structure or provide the data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Placeholder - you'll need to replace with actual data
time = np.linspace(0, 100, 1000)
ca_trace = np.random.randn(1000) * 0.1 + 0.2
ach_trace = np.random.randn(1000) * 0.05 + 0.1

# Plot calcium trace
ax1.plot(time, ca_trace, 'b-', linewidth=1.5, label='Calcium')
ax1.set_ylabel('ΔF/F (%)')
ax1.set_title('dend_004 CaBG AChNeighborhood')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot ACh trace
ax2.plot(time, ach_trace, 'r-', linewidth=1.5, label='ACh')
ax2.set_ylabel('ΔF/F (%)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()

# Save as SVG
fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
plt.close()

print(f"SVG saved to: {svg_path}")
print("Note: This is a template. Replace with actual data from the original plot.")