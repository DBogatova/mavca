# 🧠 MAVCA -- Mask-Assisted Volumetric Calcium Analysis 
This repository contains a modular analysis pipeline for extracting and visualizing dendritic calcium activity from 4D imaging data. It supports frame selection, dendrite segmentation, functional branch clustering, ΔF/F computation, and Napari visualization.

---

## 📁 Folder Structure

Data is organized by date, mouse, and run in the following structure:

```
data/
└── yyyy-mm-dd/
    └── mouse_name/
        └── run_name/
            ├── raw/                           # Raw TIFFs from microscope
            │   ├── runA_run1_reslice-crop.tif                    # 4D raw stack (T,Z,Y,X)
            │   └── runA_run1_reslice-crop_processed.tif          # 3D MIP for motion detection
            ├── preprocessed/                  # Motion-corrected data (M1 output)
            │   ├── raw_clean.tif             # Motion-filtered 4D stack
            │   ├── event_crops/              # Selected mini-stacks
            │   ├── excluded_frames.npy       # Motion frame indices
            │   ├── frame_mapping.npy         # Original→clean frame mapping
            │   └── guides/                   # Optional trunk guides (M2.5)
            ├── labelmaps/                     # Initial dendrite masks (M2 output)
            ├── labelmaps_curated_dynamic/     # Curated masks (M3 output)
            ├── traces/                        # ΔF/F trace data (M4/M5 output)
            │   ├── dff_traces_curated_bgsub.csv
            │   ├── dff_traces_curated_bgsub_smooth.csv
            │   ├── depth_analysis_global_ca.png
            │   └── depth_analysis_data.npy
            ├── outside_mask_dynamics/         # Spatial dynamics analysis (M5)
            │   ├── globalCa_umrings.npy
            │   ├── globalCa_umrings_main.png
            │   └── globalCa_comparison.png
            ├── trace_previews_curated/        # Individual trace plots
            ├── trace_previews_curated_smooth/ # Smoothed trace plots
            └── overlays_curated/              # Mask overlays on background (M6)
```

---

## 🧩 Pipeline Overview

### Module 1 (M1): Preprocessing & Event Detection
| Script | Purpose |
|--------|---------|
| `find_events_m1.py` | Detect motion, compute ΔF/F, detect calcium events, extract mini-stacks |
| `ach_ca_plots.py` | (Optional) Preview two-channel data for quality control |

### Module 2 (M2): Initial Mask Creation
| Script | Purpose |
|--------|---------|
| `auto_mask_m2.py` | Segment dendrites from event crops using thresholding + clustering |

### Module 2.5 (M2.5): Optional Mask Refinement
| Script | Purpose |
|--------|---------|
| `mask_guide_m2.5.py` | (Optional) Draw trunk guides on preview PNGs for better segmentation |
| `auto_mask_m2.py` (re-run) | Re-run with `USE_GUIDES=True` to apply manual guides |

### Module 3 (M3): Mask Curation
| Script | Purpose |
|--------|---------|
| `filter_selected_masks_m3.py` | Interactive curation of masks in Napari with neighbor editing |

### Module 4 (M4): Trace Extraction
| Script | Purpose |
|--------|---------|
| `save_traces_m4.py` | Extract ΔF/F traces from curated masks with background subtraction |

### Module 5 (M5): Trace Analysis
| Script | Purpose |
|--------|---------|
| `downsample_traces.py` | Apply smoothing and/or decimation to traces |
| `plot_selected_traces.py` | Generate publication-quality stacked trace plots |
| `depth_analysis_plots.py` | Analyze global Ca²⁺ activity by Y-depth (4 layers, bleach-corrected) |
| `outside_mask_plot.py` | Compare inside vs outside mask dynamics with micron-based rings |

### Module 6 (M6): Visualization
| Script | Purpose |
|--------|---------|
| `create_3d_movie_m6.py` | Create 3D movie visualization of entire recording |
| `create_3d_movie_chunks_m6.py` | Create 3D movie in chunks (for long recordings) |

---

## ⚙️ How to Run the Pipeline

### 1. Setup Environment
```bash
cd mavca
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
Place your raw data in:
```
data/YYYY-MM-DD/MOUSE_ID/RUN_ID/raw/
```
with filenames:
- `runB_run4_reslice-crop.tif` (4D raw stack)
- `runB_run4_reslice-crop_processed.tif` (3D MIP for motion detection)

### 3. Run Pipeline

**Module 1 (M1): Preprocessing & Event Detection**
```bash
# Detect motion, compute ΔF/F, detect calcium events and create mini-stacks
python code/Preprocessing-STEP1/find_events_m1.py

# (Optional) Preview two-channel data for quality control
python code/Preprocessing-STEP1/ach_ca_plots.py
```

**Module 2 (M2): Initial Mask Creation**
```bash
# Auto-segment dendrites from events
python code/Masks-STEP2/auto_mask_m2.py
```

**Module 2.5 (M2.5): Optional Mask Refinement**
```bash
# Draw trunk guides on preview PNGs (optional, for better precision)
python code/Masks-STEP2/mask_guide_m2.5.py

# Re-run M2 with USE_GUIDES=True in auto_mask_m2.py
python code/Masks-STEP2/auto_mask_m2.py
```

**Module 3 (M3): Mask Curation**
```bash
# Interactively curate masks in Napari
python code/Masks-STEP2/filter_selected_masks_m3.py
```

**Module 4 (M4): Trace Extraction**
```bash
# Extract ΔF/F traces from curated masks
python code/Traces-STEP3/save_traces_m4.py
```

**Module 5 (M5): Trace Analysis**
```bash
# Apply smoothing (optional)
python code/Traces-STEP3/downsample_traces.py

# Generate publication plots
python code/Traces-STEP3/plot_selected_traces.py

# Analyze global Ca²⁺ by depth (Y-layers)
python code/Traces-STEP3/depth_analysis_plots.py

# Compare inside vs outside mask dynamics
python code/Traces-STEP3/outside_mask_plot.py
```

**Module 6 (M6): Visualization**
```bash
# Create 3D movie of entire recording
python code/Visual-STEP4/create_3d_movie_m6.py

# OR create 3D movie in chunks (if movie is too long)
python code/Visual-STEP4/create_3d_movie_chunks_m6.py
```

---

## 🔧 Configuration

All scripts use consistent configuration variables:
```python
DATE = "2025-08-06"
MOUSE = "organoid" 
RUN = "run4-crop"
```

Key parameters can be adjusted in each script:
- **Motion detection**: `K_MAD`, `TILES_YX`, `PAD_NEIGHBOR`
- **Event detection**: `Z_HI`, `Z_LO`, `TOP_FRAC`
- **Mask segmentation**: `INTENSITY_PERCENTILE`, `MIN_VOL`, `MAX_VOL`
- **Trace processing**: `SMOOTH_SIGMA`, `ARTIFACT_Z`

---

## 📦 Dependencies

Core packages:
```bash
pip install numpy scipy scikit-learn scikit-image tifffile matplotlib napari tqdm pandas
pip install "napari[pyqt5]"
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

---

## Expected Outputs

### Traces
- **CSV files**: `dff_traces_curated_bgsub.csv` with columns for each dendrite
- **Individual plots**: Per-dendrite trace previews with MIP overlays
- **Stacked plots**: Publication-ready multi-trace figures with color coding

### Masks
- **3D labelmaps**: `dend_XXX_labelmap.tif` files for each curated dendrite
- **Overlays**: Triplanar views (XY/XZ/YZ) with mask outlines on background

### Quality Control
- **Motion timeline**: Frame-by-frame motion scores and exclusions
- **Event timeline**: Activity detection and z-scores over time
- **Curation log**: Record of kept/deleted masks during interactive curation

---


## 🎯 Interactive Curation (Napari)

The mask curation interface (`filter_selected_masks_m3.py`) provides:

**Navigation**: Left/Right arrows, `b` (toggle background)
**Editing**: Paint tool + `m` (merge), `x` (subtract)  
**Neighbors**: `u`/`j` (adjust count), `q`/`w`/`r` (merge neighbors)
**Actions**: `d` (delete), `k` (keep), `Ctrl+S` (save all)

Masks are shown with nearest neighbors for context, enabling precise manual editing of segmentation boundaries.

---

## 📊 Trace Analysis Features

- **Background subtraction**: Core-shell approach with 3D morphological operations
- **Artifact correction**: Negative spike removal and smoothing
- **Bleach correction**: Exponential decay fitting for photobleaching compensation
- **Multiple output formats**: Raw traces, smoothed, decimated
- **Publication plots**: Color-coded stacked traces with scale bars
- **Flexible selection**: Choose specific dendrites for analysis
- **Depth analysis**: Global Ca²⁺ activity stratified by Y-depth (4 layers)
- **Spatial dynamics**: Inside vs outside mask comparisons with distance-based rings

---

## 🔍 Troubleshooting

**Common Issues:**

1. **"All scores are NaN"**: Check tissue mask creation, may need to lower detection thresholds
2. **Motion still visible**: Decrease `K_MAD` or increase `TILES_YX` for more sensitive detection  
3. **Too many frames removed**: Increase `K_MAD` or check MIP quality
4. **Mask misalignment**: Ensure consistent Y-cropping across modules
5. **Empty event crops**: Check event detection parameters (`Z_HI`, `Z_LO`)

**Debug Mode**: Most scripts include debug output showing frame counts, detection statistics, and processing steps.

---

## 📝 Citation

If you use this pipeline, please cite:
[Our future publication]

---

## 🤝 Contributing

This pipeline is designed for calcium imaging analysis of apical dendrites. For questions or contributions, please contact Daria Bogatova (daria@bu.edu), Anna Devor (adevor@bu.edu).
