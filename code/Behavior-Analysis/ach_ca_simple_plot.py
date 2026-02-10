#!/usr/bin/env python
"""
Plot only ACh and Ca signals (averaged)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATE = "2025-12-02"
MOUSE = "rbp4cre_136_phpeb"
RUN   = "run4"

# Manual frame shift (frames cropped from beginning)
MANUAL_FRAME_SHIFT = 34  # frames cropped manually
FRAME_RATE = 5  # Hz

# Crop initial decay (seconds)
CROP_INITIAL_SEC = 7.0  # Crop first 7 seconds for run5

# Paths
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
QUICKLOOK_CSV = BASE / "quicklook" / "dual_quicklook_traces.csv"
OUTPUT_PATH = BASE / "ach_ca_plot.png"
OUTPUT_CSV = BASE / "ach_ca_simple.csv"

def load_calcium_ach_data():
    """Load Ca and ACh traces from quicklook"""
    if not QUICKLOOK_CSV.exists():
        print(f"Quicklook CSV not found: {QUICKLOOK_CSV}")
        return None, None, None
    
    df = pd.read_csv(QUICKLOOK_CSV)
    # Apply manual frame shift
    time_shift = MANUAL_FRAME_SHIFT / FRAME_RATE  # Convert frames to seconds
    time = df.index.values / FRAME_RATE + time_shift
    ach_dff = df['ACh_dFF'].values * 100  # Convert to %
    ca_dff = df['Ca_dFF'].values * 100
    
    return time, ach_dff, ca_dff

def plot_ach_ca_signals():
    """Create plot of ACh and Ca signals"""
    # Load data
    time_ca, ach_dff, ca_dff = load_calcium_ach_data()
    
    if time_ca is None:
        print("No data to plot")
        return
    
    # Start time axis from 0
    time_ca_corrected = time_ca - time_ca[0]
    
    # Crop initial decay
    crop_idx = int(CROP_INITIAL_SEC * FRAME_RATE)
    time_ca_corrected = time_ca_corrected[crop_idx:]
    ach_dff = ach_dff[crop_idx:]
    ca_dff = ca_dff[crop_idx:]
    
    # Reset time to start from 0 after crop
    time_ca_corrected = time_ca_corrected - time_ca_corrected[0]
    
    print(f"Cropped first {CROP_INITIAL_SEC}s ({crop_idx} frames)")
    
    # Save to CSV
    df_out = pd.DataFrame({
        'time_sec': time_ca_corrected,
        'ACh_dFF_percent': ach_dff,
        'Ca_dFF_percent': ca_dff
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV: {OUTPUT_CSV}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    # Plot 1: ACh
    axes[0].plot(time_ca_corrected, ach_dff, color='red', linewidth=1.5)
    axes[0].set_ylabel('ACh ΔF/F (%)', fontsize=12)
    axes[0].set_title('ACh Signal', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Ca
    axes[1].plot(time_ca_corrected, ca_dff, color='green', linewidth=1.5)
    axes[1].set_ylabel('Ca ΔF/F (%)', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_title('Ca Signal', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'{MOUSE} - {DATE} - {RUN} (cropped first {CROP_INITIAL_SEC}s)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"Saved ACh/Ca plot: {OUTPUT_PATH}")

def main():
    print(f"Creating ACh/Ca plot for {MOUSE}-{DATE}-{RUN}")
    plot_ach_ca_signals()

if __name__ == "__main__":
    main()
