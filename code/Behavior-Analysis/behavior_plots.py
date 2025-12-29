#!/usr/bin/env python
"""
Combined behavior and calcium/ACh analysis plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
# h5py will be imported when needed
import tifffile

# Configuration
DATE = "2025-12-25"
MOUSE = "rAi162_phpeb"
RUN = "run1"

# Manual frame shift (frames cropped from beginning)
MANUAL_FRAME_SHIFT = 34  # frames cropped manually
FRAME_RATE = 5  # Hz

# Paths
BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
BEHAVIOR_MAT = BASE / "behavior" / f"{MOUSE}_25-12-26_Run001_behavior.mat"
TRIGGER_MAT = BASE / "trigger" / "Run004_t1.mat"
QUICKLOOK_CSV = BASE / "quicklook" / "dual_quicklook_traces.csv"
OUTPUT_PATH = BASE / "behavior_combined_plot.png"

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

def load_behavior_data():
    """Load pupil and whisker from behavior MAT file"""
    if not BEHAVIOR_MAT.exists():
        print(f"Behavior MAT not found: {BEHAVIOR_MAT}")
        return None, None, None
    
    mat_data = loadmat(BEHAVIOR_MAT)
    
    # Extract pupil data
    pupil_smooth = mat_data['pupil']['pupil_smooth'][0][0].flatten()
    
    # Extract whisker data (choose one type)
    whisker_smooth = mat_data['whisker']['whisker_smooth_long'][0][0].flatten()
    
    # Create time axis (assuming 10Hz for behavior)
    time_behavior = np.arange(len(pupil_smooth)) / 10.0
    
    return time_behavior, pupil_smooth, whisker_smooth

def load_accelerometer_data():
    """Load accelerometer from CSV files"""
    accel_csv = BASE / "trigger" / "Run001_t1_accel.csv"
    trigger_csv = BASE / "trigger" / "Run001_t1_trigger.csv"
    
    if not accel_csv.exists() or not trigger_csv.exists():
        print(f"CSV files not found: {accel_csv}, {trigger_csv}")
        return None, None, 0
    
    # Load accelerometer data
    accel_df = pd.read_csv(accel_csv)
    print(f"Accel CSV columns: {list(accel_df.columns)}")
    accel_data = accel_df.iloc[:, 1].values  # Assuming second column is data
    
    # Load trigger data
    trigger_df = pd.read_csv(trigger_csv)
    print(f"Trigger CSV columns: {list(trigger_df.columns)}")
    
    # Find the trigger column (might have different name)
    trigger_col = None
    for col in trigger_df.columns:
        if 'trigger' in col.lower() or 'andor' in col.lower():
            trigger_col = col
            break
    
    if trigger_col is None:
        trigger_col = trigger_df.columns[-1]  # Use last column as fallback
    
    trigger_data = trigger_df[trigger_col].values
    print(f"Using trigger column: {trigger_col}")
    
    # Find trigger offset (first time AndorXylaTrigger = 1)
    trigger_onset = np.where(trigger_data == 1)[0]
    offset_samples = trigger_onset[0] if len(trigger_onset) > 0 else 0
    
    print(f"Trigger onset at sample {offset_samples} ({offset_samples/1000.0:.3f}s)")
    
    # Create time axis (assuming 1kHz)
    time_accel = np.arange(len(accel_data)) / 1000.0
    
    return time_accel, accel_data, offset_samples

def plot_combined_signals():
    """Create combined plot of all signals"""
    # Load data
    time_ca, ach_dff, ca_dff = load_calcium_ach_data()
    time_behavior, pupil, whisker = load_behavior_data()
    time_accel, accel, offset = load_accelerometer_data()
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: ACh
    if time_ca is not None:
        # Start Ca/ACh time axis from 0
        time_ca_corrected = time_ca - time_ca[0]
        axes[0].plot(time_ca_corrected, ach_dff, color='red', linewidth=1.5)
        axes[0].set_ylabel('ACh ΔF/F (%)')
        axes[0].set_title('ACh Signal')
        axes[0].grid(alpha=0.3)
    
    # Plot 2: Ca
    if time_ca is not None:
        axes[1].plot(time_ca_corrected, ca_dff, color='green', linewidth=1.5)
        axes[1].set_ylabel('Ca ΔF/F (%)')
        axes[1].set_title('Ca Signal')
        axes[1].grid(alpha=0.3)
    
    # Crop behavior signals by 6.8 seconds (34 frames worth) to match Ca/ACh
    crop_time = MANUAL_FRAME_SHIFT / FRAME_RATE  # 6.8 seconds
    behavior_crop_samples = int(crop_time * 10)  # 68 samples at 10Hz
    accel_crop_samples = int(crop_time * 1000)   # 6800 samples at 1kHz
    
    # Use trigger onset + 2s for original offset
    scape_start_sec = (offset / 1000.0) + 2.0 if offset > 0 else 2.0
    scape_start_behavior_samples = int(scape_start_sec * 10)
    scape_start_accel_samples = int(scape_start_sec * 1000)
    
    # Plot 3: Pupil (crop 6.8s and match Ca/ACh duration)
    if time_behavior is not None and pupil is not None:
        try:
            mat_data = loadmat(BEHAVIOR_MAT)
            pupil_data = mat_data['pupil']['pupil_raw'][0][0].flatten()
        except:
            pupil_data = pupil
        
        # Crop from trigger start + additional 6.8s crop
        start_idx = scape_start_behavior_samples + behavior_crop_samples
        if start_idx < len(pupil_data):
            pupil_cropped = pupil_data[start_idx:]
        else:
            pupil_cropped = pupil_data
        
        # Match Ca/ACh duration and start from time 0
        if time_ca is not None:
            ca_duration = time_ca[-1] - time_ca[0]
            max_samples = int(ca_duration * 10)
            if len(pupil_cropped) > max_samples:
                pupil_cropped = pupil_cropped[:max_samples]
        
        time_cropped = np.arange(len(pupil_cropped)) / 10.0
        axes[2].plot(time_cropped, pupil_cropped, color='blue', linewidth=1.0)
        axes[2].set_ylabel('Pupil Dilation')
        axes[2].set_title('Pupil Signal')
        axes[2].grid(alpha=0.3)
    
    # Plot 4: Whisker (crop 6.8s and match Ca/ACh duration)
    if time_behavior is not None and whisker is not None:
        try:
            mat_data = loadmat(BEHAVIOR_MAT)
            whisker_data = mat_data['whisker']['whisker_raw_pad'][0][0].flatten()
        except:
            whisker_data = whisker
        
        # Crop from trigger start + additional 6.8s crop
        start_idx = scape_start_behavior_samples + behavior_crop_samples
        if start_idx < len(whisker_data):
            whisker_cropped = whisker_data[start_idx:]
        else:
            whisker_cropped = whisker_data
        
        # Match Ca/ACh duration and start from time 0
        if time_ca is not None:
            ca_duration = time_ca[-1] - time_ca[0]
            max_samples = int(ca_duration * 10)
            if len(whisker_cropped) > max_samples:
                whisker_cropped = whisker_cropped[:max_samples]
        
        time_cropped = np.arange(len(whisker_cropped)) / 10.0
        whisker_smooth = gaussian_filter1d(whisker_cropped, sigma=1.0)
        axes[3].plot(time_cropped, whisker_smooth, color='orange', linewidth=1.0)
        axes[3].set_ylabel('Whisker Motion')
        axes[3].set_title('Whisker Signal')
        axes[3].grid(alpha=0.3)
    
    # Plot 5: Accelerometer (crop 6.8s and match Ca/ACh duration)
    if time_accel is not None and accel is not None:
        # Crop from trigger start + additional 6.8s crop
        start_idx = scape_start_accel_samples + accel_crop_samples
        if start_idx < len(accel):
            accel_cropped = accel[start_idx:]
        else:
            accel_cropped = accel
        
        # Match Ca/ACh duration and start from time 0
        if time_ca is not None:
            ca_duration = time_ca[-1] - time_ca[0]
            max_samples = int(ca_duration * 1000)
            if len(accel_cropped) > max_samples:
                accel_cropped = accel_cropped[:max_samples]
        
        time_accel_cropped = np.arange(len(accel_cropped)) / 1000.0
        axes[4].plot(time_accel_cropped, accel_cropped, color='purple', linewidth=1.0)
        axes[4].set_ylabel('Acceleration')
        axes[4].set_title('Accelerometer')
        axes[4].set_xlabel('Time (s)')
        axes[4].grid(alpha=0.3)
    
    plt.suptitle(f'{MOUSE} - {DATE} - {RUN}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"Saved combined plot: {OUTPUT_PATH}")

def main():
    print(f"Creating combined behavior plot for {MOUSE}-{DATE}-{RUN}")
    plot_combined_signals()

if __name__ == "__main__":
    main()