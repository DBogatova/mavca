# 🧠 MAVCA — Mask-Assisted Volumetric Calcium Analysis

Modular pipeline for extracting and visualizing dendritic calcium activity from 4D SCAPE imaging data. Supports event detection, dendrite segmentation, mask curation, ΔF/F trace extraction, and dual-channel coupling analysis.

---

## 📁 Folder Structure

```
scape-data/
└── yyyy-mm-dd/
    └── mouse_name/
        └── run_name/
            ├── raw/                           # Raw TIFFs from microscope
            │   ├── runA_run4_mouse_binimagej_reslice_green.tif
            │   └── runA_run4_mouse_binimagej_reslice_red.tif
            ├── preprocessed/
            │   ├── raw_clean.tif              # Motion-filtered 4D stack
            │   ├── stack_voxel_norm_mean_sub.tif
            │   ├── stack_smoothed.tif
            │   ├── active_frames.npy
            │   ├── activity_timeline.pdf
            │   ├── event_crops/               # Mini-stacks per event (M1)
            │   └── best_frames/               # Best frames per event (M1.5)
            ├── labelmaps/                     # Initial masks (M2)
            ├── labelmap_backgrounds/          # 2D backgrounds for viz
            ├── labelmap_previews/             # Mask preview PNGs
            ├── labelmaps_split/               # After watershed split (M2b)
            ├── labelmap_backgrounds_split/
            ├── labelmaps_curated_dynamic/     # Curated masks (M3)
            ├── traces/                        # ΔF/F traces (M4)
            │   ├── dff_traces_curated_bgsub.csv
            │   └── dff_traces_curated_bgsub_smooth.csv
            └── masks_manifest.csv
```

---

## 🧩 Pipeline (Steps 1–4)

### M1: Event Detection — `find_events_m1.py`

Loads raw 4D stack, normalizes voxels, subtracts frame mean, applies Gaussian smoothing. Detects calcium events using rolling-baseline detrending + hysteresis thresholding on robust z-scores. Saves event crops (mini-stacks around each event).

```bash
python code/Preprocessing-STEP1/find_events_m1.py
```

Key config: `FS_HZ`, `SKIP_FIRST_SECONDS`, `Y_CROP`, `START_THRESHOLD`, `END_THRESHOLD`, `MIN_EVENT_DURATION`

### M1.5: Best-Frame Selection — `pre_segmentation_m1.5.py`

For each event crop, scores frames by a "sparse bright" metric (high percentile minus mid percentile of Z-MIP). Selects top-K frames around the peak with minimum spacing. Saves 3D volumes and contrast-stretched MIP PNGs used downstream by M2.

```bash
python code/Preprocessing-STEP1/pre_segmentation_m1.5.py
```

Key config: `TOP_Z_PLANES`, `TOP_K`, `PEAK_HALF_WINDOW`, `MIN_SEP`

---

### M2: Auto Mask Detection — `auto_mask_m2.py`

Segments dendrites from event crops using multi-window hysteresis detection:
1. Enhances each crop (spatial high-pass, clamp positive)
2. Finds top-K intensity peaks across time, builds seed + candidate masks via percentile thresholds
3. Keeps connected components touching seeds (hysteresis)
4. Optional vesselness (Sato filter) for trunk capture
5. Optional intensity grow to fill trunk bodies
6. Best-frame guidance mode unions M1.5 seeds with auto-detected candidates
7. 2D/3D cleanup, volume gating, Dice deduplication

```bash
python code/Masks-STEP2/auto_mask_m2.py
```

Key config: `TOPK_PEAKS`, `SEED_PCT`, `CAND_PCT`, `USE_VESSELNESS`, `BESTFRAME_MODE`, `MIN_VOL`, `MAX_VOL`

Outputs: `labelmaps/dend_XXX_labelmap.tif`, `labelmap_backgrounds/`, `labelmap_previews/`, `masks_manifest.csv`

---

### M2b: Split Merged Masks — `split_merged_masks_m2b.py`

Splits masks that contain multiple dendrites merged together. Two modes:

- **Auto split**: Distance-transform watershed finds natural split points
- **Interactive Napari**: Polyline tools for manual cut and erase operations

```bash
# Interactive mode (opens Napari)
python code/Masks-STEP2/split_merged_masks_m2b.py

# Batch cleanup of small fragments
python code/Masks-STEP2/split_merged_masks_m2b.py --clean
```

Outputs: `labelmaps_split/`, `labelmap_backgrounds_split/`

---

### M3: Mask Curation — `filter_selected_masks_m3.py`

Interactive Napari-based curation. Reads from `labelmaps_split/` (or `labelmaps/` if no split step). Navigate masks one by one, keep/delete/edit with neighbor context.

```bash
python code/Masks-STEP2/filter_selected_masks_m3.py
```

Controls:
- Left/Right arrows: navigate masks
- `b`: toggle background
- `d`: delete, `k`: keep
- Paint tool + `m`: merge, `x`: subtract
- `u`/`j`: adjust neighbor count
- `q`/`w`/`r`: merge neighbors
- `Ctrl+S`: save all

Outputs: `labelmaps_curated_dynamic/dend_XXX_labelmap.tif`

---

### M4: Trace Extraction — `save_traces_m4.py`

Extracts per-dendrite ΔF/F traces from curated masks with background subtraction (core-shell morphological approach). Streams through time in chunks to manage memory.

```bash
python code/Traces-STEP3/save_traces_m4.py
```

Outputs: `traces/dff_traces_curated_bgsub.csv`, per-dendrite preview plots

---

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

## ⚙️ Setup

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

Core dependencies: numpy, scipy, scikit-learn, scikit-image, tifffile, matplotlib, napari, tqdm, pandas, opencv-python

---

## 🔧 Common Configuration

All scripts share these variables at the top:
```python
DATE = "2025-12-02"
MOUSE = "rbp4cre_136_phpeb"
RUN = "run4"
```

Physical voxel size: `(3.9, 1.0, 1.2)` µm (Z, Y, X), frame rate: 5.0 Hz

---

## 📝 Citation

If you use this pipeline, please cite: [Our future publication]

## 🤝 Contributing

For questions or contributions: Daria Bogatova (daria@bu.edu), Anna Devor (adevor@bu.edu)
