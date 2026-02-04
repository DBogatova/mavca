#!/usr/bin/env python
"""
Visualize Event Crops: Create colored Z-plane montages of event crops

Creates multi-panel figures showing:
- All Z-planes for each event crop
- Color-coded ΔF/F values using turbo colormap
- Time progression through the event
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tifffile
from pathlib import Path
import argparse

# ===== CONFIG =====
DATE = "2025-10-29"
MOUSE = "rAi162_15"
RUN = "run1-crop"

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/data") / DATE / MOUSE / RUN
EVENT_CROPS_FOLDER = BASE / "preprocessed" / "event_crops_scaled"
OUTPUT_FOLDER = BASE / "event_visualizations"
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Visualization settings
COLORMAP = "turbo"
FIGSIZE_PER_Z = 2  # inches per Z-plane
MAX_COLS = 7       # Maximum Z-planes per row

def visualize_event_crop(event_path, output_folder):
    """Create Z-plane montage for a single event crop"""
    print(f"Processing: {event_path.name}")
    
    # Load event crop (T,Z,Y,X)
    stack = tifffile.imread(event_path).astype(np.float32)
    T, Z, Y, X = stack.shape
    
    # Calculate global min/max for consistent color scaling
    vmin = np.nanpercentile(stack, 1)
    vmax = np.nanpercentile(stack, 99)
    
    # Create figure for Z-plane montage
    n_cols = min(Z, MAX_COLS)
    n_rows = int(np.ceil(Z / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * FIGSIZE_PER_Z, n_rows * FIGSIZE_PER_Z))
    if Z == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each Z-plane (time-averaged)
    for z in range(Z):
        ax = axes[z] if Z > 1 else axes[0]
        
        # Average over time for this Z-plane
        z_plane = np.nanmean(stack[:, z, :, :], axis=0)
        
        im = ax.imshow(z_plane, cmap=COLORMAP, vmin=vmin, vmax=vmax, 
                      interpolation='nearest', origin='lower')
        ax.set_title(f'Z={z}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for z in range(Z, len(axes)):
        axes[z].set_visible(False)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[:Z], shrink=0.8, aspect=20)
    cbar.set_label('ΔF/F', rotation=270, labelpad=15)
    
    # Add title with event info
    event_name = event_path.stem
    fig.suptitle(f'{event_name}\nTime-averaged Z-planes', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_folder / f"{event_name}_z_planes.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Also create time-lapse for peak Z-plane
    create_time_lapse(stack, event_path, output_folder, vmin, vmax)

def create_time_lapse(stack, event_path, output_folder, vmin, vmax):
    """Create time-lapse montage for the most active Z-plane"""
    T, Z, Y, X = stack.shape
    
    # Find Z-plane with highest activity
    z_activity = np.nanvar(stack, axis=(0, 2, 3))  # Variance over time and space
    peak_z = np.argmax(z_activity)
    
    # Get time series for peak Z-plane
    time_series = stack[:, peak_z, :, :]  # (T, Y, X)
    
    # Create time-lapse montage
    n_cols = min(T, 8)  # Max 8 time points per row
    n_rows = int(np.ceil(T / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if T == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for t in range(T):
        ax = axes[t] if T > 1 else axes[0]
        
        im = ax.imshow(time_series[t], cmap=COLORMAP, vmin=vmin, vmax=vmax,
                      interpolation='nearest', origin='lower')
        ax.set_title(f't={t}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for t in range(T, len(axes)):
        axes[t].set_visible(False)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[:T], shrink=0.8, aspect=20)
    cbar.set_label('ΔF/F', rotation=270, labelpad=15)
    
    # Add title
    event_name = event_path.stem
    fig.suptitle(f'{event_name}\nTime-lapse (Z={peak_z})', fontsize=12)
    
    plt.tight_layout()
    
    # Save time-lapse
    output_path = output_folder / f"{event_name}_timelapse_z{peak_z}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize event crops with colored Z-planes")
    parser.add_argument("--event", type=str, help="Specific event to visualize (e.g., 'event_group_0001')")
    parser.add_argument("--all", action="store_true", help="Visualize all events")
    args = parser.parse_args()
    
    # Find event crop files
    event_files = sorted(EVENT_CROPS_FOLDER.glob("event_group_*.tif"))
    
    if not event_files:
        print(f"No event crops found in {EVENT_CROPS_FOLDER}")
        return
    
    # Filter by specific event if requested
    if args.event:
        event_files = [f for f in event_files if args.event in f.name]
        if not event_files:
            print(f"Event '{args.event}' not found")
            return
    
    # Process events
    if args.all or args.event:
        print(f"Visualizing {len(event_files)} event crops...")
        for event_file in event_files:
            visualize_event_crop(event_file, OUTPUT_FOLDER)
    else:
        # Default: visualize first 3 events
        print(f"Visualizing first 3 events (use --all for all events)")
        for event_file in event_files[:3]:
            visualize_event_crop(event_file, OUTPUT_FOLDER)
    
    print(f"✅ Visualizations saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()