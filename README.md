# 🧠 MAVCA -- Mask-Assisted Volumetric Calcium Analysis 
This repository contains a modular analysis pipeline for extracting and visualizing dendritic calcium activity from 4D imaging data. It supports frame selection, dendrite segmentation, functional branch clustering, ΔF/F computation, and Napari visualization.

---

## 📁 Folder Structure

Data is organized by date, mouse, and run in the following structure:

```
data/
└── 2025-08-06/
    └── organoid/
        └── run4-crop/
            ├── raw/                           # Raw TIFFs from microscope
            │   ├── runB_run4_reslice-crop.tif                    # 4D raw stack (T,Z,Y,X)
            │   └── runB_run4_reslice-crop_processed.tif          # 3D MIP for motion detection
            ├── preprocessed/                  # Motion-corrected data (STEP 1 output)
            │   ├── raw_clean.tif             # Motion-filtered 4D stack
            │   ├── dff_stack.tif             # ΔF/F stack from Module 1
            │   ├── event_crops/              # Selected mini-stacks
            │   ├── excluded_frames.npy       # Motion frame indices
            │   └── frame_mapping.npy         # Original→clean frame mapping
            ├── labelmaps/                     # Initial dendrite masks (STEP 2 output)
            ├── labelmaps_curated_dynamic/     # Curated masks (STEP 2 output)
            ├── traces/                        # ΔF/F trace data (STEP 3 output)
            │   ├── dff_traces_curated_bgsub.csv
            │   └── dff_traces_curated_bgsub_smooth.csv
            ├── trace_previews_curated/        # Individual trace plots
            ├── trace_previews_curated_smooth/ # Smoothed trace plots
            └── overlays_curated/              # Mask overlays on background
```

---

## 🧩 Pipeline Overview

### STEP 1: Preprocessing & Event Detection
| Script | Purpose |
|--------|---------|
| `remove_motion_frames_simple.py` | Detect and remove motion artifacts from raw 4D stack |
| `find_events_m1.py` | Compute ΔF/F, detect calcium events, extract mini-stacks |

### STEP 2: Mask Creation & Curation
| Script | Purpose |
|--------|---------|
| `auto_mask_m2.py` | Segment dendrites from event crops using thresholding + clustering |
| `filter_selected_masks_m3.py` | Interactive curation of masks in Napari with neighbor editing |

### STEP 3: Trace Extraction & Analysis
| Script | Purpose |
|--------|---------|
| `save_traces_m4.py` | Extract ΔF/F traces from curated masks with background subtraction |
| `downsample_traces.py` | Apply smoothing and/or decimation to traces |
| `plot_selected_traces.py` | Generate publication-quality stacked trace plots |

### Visualization & Quality Control
| Script | Purpose |
|--------|---------|
| `organoid_outline.py` | Create mask overlays on background MIPs (XY/XZ/YZ views) |
| `remove_trace_artifacts.py` | Remove frames with trace-based motion artifacts |

---

## ⚙️ How to Run the Pipeline

### 1. Setup Environment
```bash
cd apical-dendrites-2025
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

**STEP 1: Motion Correction & Event Detection**
```bash
# Remove motion artifacts
python code/Extra/remove_motion_frames_simple.py

# Detect calcium events and create mini-stacks
python code/Preprocessing-STEP1/find_events_m1.py
```

**STEP 2: Mask Creation & Curation**
```bash
# Auto-segment dendrites from events
python code/Masks-STEP2/auto_mask_m2.py

# Interactively curate masks in Napari
python code/Masks-STEP2/filter_selected_masks_m3.py
```

**STEP 3: Trace Extraction & Analysis**
```bash
# Extract ΔF/F traces from curated masks
python code/Traces-STEP3/save_traces_m4.py

# Apply smoothing (optional)
python code/Traces-STEP3/downsample_traces.py

# Generate publication plots
python code/Traces-STEP3/plot_selected_traces.py
```

**Visualization**
```bash
# Create mask overlays
python code/organoid_outline.py --select dend_001,dend_016,dend_018 --combined
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

## ✅ Expected Outputs

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

## 🧼 Motion Correction

The pipeline includes robust motion artifact removal:

1. **Correlation-based detection**: Tile-wise frame-to-frame correlation analysis
2. **Adaptive thresholds**: Rolling median + MAD for local threshold adaptation  
3. **Physical removal**: Motion frames are completely removed (not set to NaN)
4. **Frame mapping**: Original→clean frame indices preserved for reference

Motion parameters can be tuned for sensitivity:
- `K_MAD = 2.5`: Lower = more sensitive (remove more frames)
- `TILES_YX = (4,4)`: More tiles = better local motion detection
- `PAD_NEIGHBOR = 2`: Remove neighboring frames around detected motion

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
- **Multiple output formats**: Raw traces, smoothed, decimated
- **Publication plots**: Color-coded stacked traces with scale bars
- **Flexible selection**: Choose specific dendrites for analysis

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
