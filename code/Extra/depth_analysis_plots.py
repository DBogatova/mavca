#!/usr/bin/env python3
"""
Depth Analysis Plots - Global Ca2+ by Y-depth chunks
Divides Y dimension into 4 chunks (top to bottom) and plots global calcium activity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from scipy.ndimage import gaussian_filter1d

# =================
# ===== CONFIG =====
# =================
DATE = "2025-12-02"
MOUSE = "rbp4cre_136_phpeb"
RUN = "run4"

FRAME_RATE = 5.0
CHUNK_T = 118
Y_CROP = 3  # crop bottom Y pixels to match masks

# Baseline
F0_NFRAMES = 30

# Artifact clamp / smoothing
SMOOTH_SIGMA = 0.5

# Plot / output
SAVE_FIG = True


# ================
# ===== PATHS =====
# ================
PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE = PROJECT_ROOT / "scape-data" / DATE / MOUSE / RUN

RAW_CLEAN_PATH = BASE / "preprocessed" / "raw_clean.tif"
RAW_ORIG_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}_binimagej_reslice_green.tif"
RAW_STACK_PATH = RAW_CLEAN_PATH if RAW_CLEAN_PATH.exists() else RAW_ORIG_PATH

OUTPUT_PATH = BASE / "traces"

def load_stack():
    """Load and crop the 4D stack"""
    print(f"Loading: {RAW_STACK_PATH}")
    stack = tifffile.imread(RAW_STACK_PATH).astype(np.float32)
    
    if stack.ndim != 4:
        raise ValueError(f"Expected 4D stack, got {stack.ndim}D")
    
    T, Z, Y, X = stack.shape
    print(f"Original shape: T={T}, Z={Z}, Y={Y}, X={X}")
    
    # Crop bottom Y pixels
    if Y_CROP > 0:
        stack = stack[:, :, :-Y_CROP, :]
        Y = Y - Y_CROP
        print(f"After Y-crop: T={T}, Z={Z}, Y={Y}, X={X}")
    
    return stack

def compute_f0(stack):
    """Compute F0 from last frames"""
    print(f"Computing F0 from last {F0_NFRAMES} frames")
    f0 = np.mean(stack[-F0_NFRAMES:], axis=0)
    return f0

def compute_depth_chunks(Y):
    """Divide Y into 4 equal chunks"""
    chunk_size = Y // 4
    chunks = []
    
    for i in range(4):
        start_y = i * chunk_size
        if i == 3:  # last chunk gets remainder
            end_y = Y
        else:
            end_y = (i + 1) * chunk_size
        chunks.append((start_y, end_y))
    
    print(f"Y chunks: {chunks}")
    return chunks

def compute_global_trace_for_chunk(stack, f0, y_start, y_end):
    """Compute global ΔF/F trace for a Y chunk"""
    T = stack.shape[0]
    
    # Extract chunk
    chunk_stack = stack[:, :, y_start:y_end, :]
    chunk_f0 = f0[:, y_start:y_end, :]
    
    # Compute global mean for each timepoint
    trace = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        frame = chunk_stack[t]
        # Global ΔF/F
        dff = (frame - chunk_f0) / (chunk_f0 + 1e-6)  # small epsilon to avoid division by zero
        trace[t] = np.mean(dff)
    
    return trace

def bleach_correct_trace(trace):
    """Apply exponential bleach correction"""
    from scipy.optimize import curve_fit
    
    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    t = np.arange(len(trace))
    try:
        popt, _ = curve_fit(exp_decay, t, trace, p0=[trace[0], 0.001, trace[-1]])
        bleach_trend = exp_decay(t, *popt)
        corrected = trace - bleach_trend + np.mean(trace)
        return corrected
    except:
        return trace
def smooth_trace(trace, sigma=SMOOTH_SIGMA):
    """Apply Gaussian smoothing"""
    if sigma > 0:
        return gaussian_filter1d(trace, sigma=sigma)
    return trace

def plot_depth_traces(time_s, traces, chunks, output_path=None):
    """Plot all depth traces together"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#E53E3E', '#FF8C00', '#38A169', '#3182CE']  # red, orange, green, blue
    depth_labels = ['Top (surface)', 'Upper middle', 'Lower middle', 'Bottom (deep)']
    
    for i, (trace, (y_start, y_end)) in enumerate(zip(traces, chunks)):
        offset_trace = trace + (3 - i) * 0.03
        ax.plot(time_s, offset_trace, color=colors[i], linewidth=1.5, 
                label=f'{depth_labels[i]} (Y: {y_start}-{y_end})')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.set_title(f'{DATE} | {MOUSE} | {RUN} - Global Ca²⁺ by Depth', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add scale bar
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    
    plt.show()

def main():
    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load data
    stack = load_stack()
    T, Z, Y, X = stack.shape
    
    # Compute F0
    f0 = compute_f0(stack)
    
    # Create time axis
    time_s = np.arange(T, dtype=np.float32) / FRAME_RATE
    
    # Divide Y into chunks
    chunks = compute_depth_chunks(Y)
    
    # Compute traces for each depth chunk
    print("\n=== Computing depth traces ===")
    traces = []
    for i, (y_start, y_end) in enumerate(chunks):
        print(f"Processing chunk {i+1}/4: Y={y_start}-{y_end}")
        trace = compute_global_trace_for_chunk(stack, f0, y_start, y_end)
        trace_corrected = bleach_correct_trace(trace)
        trace_smooth = smooth_trace(trace_corrected)
        traces.append(trace_smooth)
    
    # Plot results
    output_png = OUTPUT_PATH / "depth_analysis_global_ca.png" if SAVE_FIG else None
    plot_depth_traces(time_s, traces, chunks, output_png)
    
    # Save data
    output_data = {
        'time_s': time_s,
        'chunks': chunks,
        'traces': traces,
        'meta': {
            'DATE': DATE,
            'MOUSE': MOUSE, 
            'RUN': RUN,
            'FRAME_RATE': FRAME_RATE,
            'SMOOTH_SIGMA': SMOOTH_SIGMA,
            'F0_NFRAMES': F0_NFRAMES,
            'Y_CROP': Y_CROP,
            'stack_shape': (T, Z, Y, X)
        }
    }
    
    output_npy = OUTPUT_PATH / "depth_analysis_data.npy"
    np.save(output_npy, output_data, allow_pickle=True)
    print(f"Saved data: {output_npy}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()