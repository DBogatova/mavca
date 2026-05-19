#!/usr/bin/env python3
"""
Mask Quality Classifier — learns from M3 curation decisions.

Training: scans curated runs, extracts spatial features from masks
and temporal features from M4 trace CSVs (fast, no raw stack reading).

Usage:
    python code/Masks-STEP2/mask_classifier.py --train
    python code/Masks-STEP2/mask_classifier.py --predict
"""

import sys
import csv
import pickle
import warnings
import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, binary_erosion, binary_dilation
from skimage.morphology import ball, label
from skimage.measure import regionprops

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
SCAPE_DATA = PROJECT_ROOT / "scape-data"
MODEL_PATH = PROJECT_ROOT / "code" / "Masks-STEP2" / "mask_classifier_model.pkl"

PREDICT_DATE = "2026-04-16"
PREDICT_MOUSE = "rbp4_132_phpeb"
PREDICT_RUN = "run8"

KNOWN_RUNS = [
    ("2025-12-02", "rbp4cre_136_phpeb", "run4", 5.0),
    ("2026-03-20", "rbp4cre_139_phpeb", "run1", 5.0),
    ("2026-03-20", "rbp4cre_139_phpeb", "run3", 5.0),
    ("2026-03-31", "rbp4_132_phpeb", "run7", 5.0),
    ("2026-03-31", "rbp4_132_phpeb", "run8", 5.0),
    ("2026-04-16", "rbp4_132_phpeb", "run1", 5.0),
    ("2026-04-16", "rbp4_132_phpeb", "run7", 5.0),
]

VOXEL_SIZE = (3.9, 1.0, 1.2)
VOXEL_VOL = float(np.prod(VOXEL_SIZE))


def extract_spatial_features(mask_3d):
    """Extract shape/geometry features from a 3D binary mask."""
    m = mask_3d > 0
    if not m.any():
        return None

    volume_vox = int(m.sum())
    volume_um3 = volume_vox * VOXEL_VOL
    Z, Y, X = m.shape

    rp = regionprops(label(m.astype(np.uint8)))
    if not rp:
        return None
    r = rp[0]
    bbox = r.bbox
    z_span = (bbox[3] - bbox[0]) * VOXEL_SIZE[0]
    y_span = (bbox[4] - bbox[1]) * VOXEL_SIZE[1]
    x_span = (bbox[5] - bbox[2]) * VOXEL_SIZE[2]

    aspect_yz = y_span / (z_span + 1e-6)
    aspect_yx = y_span / (x_span + 1e-6)
    aspect_zx = z_span / (x_span + 1e-6)

    try:
        solidity = r.solidity
    except Exception:
        solidity = 0.5

    try:
        elongation = r.axis_major_length / (r.axis_minor_length + 1e-6)
    except Exception:
        elongation = 1.0

    eroded = binary_erosion(m, structure=ball(1))
    surface_vox = volume_vox - int(eroded.sum())
    surface_ratio = surface_vox / (volume_vox + 1e-6)

    cz, cy, cx = r.centroid
    y_frac = cy / Y
    z_frac = cz / Z

    _, n_components = label(m, return_num=True)

    # Z-MIP fill ratio (how much of the bounding box is filled in projection)
    mip = m.max(axis=0)
    bbox_area = (bbox[4]-bbox[1]) * (bbox[5]-bbox[2])
    mip_fill = mip.sum() / (bbox_area + 1e-6)

    # Number of Z slices with mask voxels
    z_slices_active = int((m.sum(axis=(1,2)) > 0).sum())
    z_continuity = z_slices_active / (bbox[3] - bbox[0] + 1e-6)

    return {
        "volume_um3": volume_um3,
        "z_span_um": z_span,
        "y_span_um": y_span,
        "x_span_um": x_span,
        "aspect_yz": aspect_yz,
        "aspect_yx": aspect_yx,
        "aspect_zx": aspect_zx,
        "solidity": solidity,
        "elongation": elongation,
        "surface_ratio": surface_ratio,
        "y_frac": y_frac,
        "z_frac": z_frac,
        "n_components": n_components,
        "mip_fill": mip_fill,
        "z_continuity": z_continuity,
    }


def extract_all_temporal_features(mask_paths, mask_names, base, fps):
    """Extract temporal features for ALL masks in one pass through the raw stack."""
    # Find raw stack
    raw_path = None
    clean = base / "preprocessed" / "raw_clean.tif"
    if clean.exists():
        raw_path = clean
    else:
        raw_dir = base / "raw"
        if raw_dir.exists():
            for p in sorted(raw_dir.glob("run[AB]_*.*tif")):
                if "reslice" in p.name.lower():
                    raw_path = p
                    break

    if raw_path is None or not raw_path.exists():
        return {name: {} for name in mask_names}

    print(f"    Extracting temporal features from {raw_path.name}...")
    store = tifffile.memmap(str(raw_path), mode='r')
    T_raw, raw_Z, raw_Y, raw_X = store.shape
    skip = int(11.0 * fps)

    # Build core/shell indices for all masks
    rois = []
    for mp, name in zip(mask_paths, mask_names):
        m = tifffile.imread(mp).astype(bool)
        Z, Y, X = m.shape
        # Adjust to raw dimensions
        if Y != raw_Y or X != raw_X:
            m_adj = np.zeros((raw_Z, raw_Y, raw_X), dtype=bool)
            yz, xz = min(Y, raw_Y), min(X, raw_X)
            zz = min(Z, raw_Z)
            m_adj[:zz, :yz, :xz] = m[:zz, :yz, :xz]
            m = m_adj

        core = binary_erosion(m, structure=ball(1))
        if not core.any():
            core = m.copy()
        inner = binary_dilation(m, structure=ball(2))
        outer = binary_dilation(m, structure=ball(3))
        shell = outer & ~inner
        rois.append({
            "name": name,
            "core": np.flatnonzero(core.ravel()),
            "shell": np.flatnonzero(shell.ravel()) if shell.any() else np.array([], dtype=np.int64),
        })

    n_vox = raw_Z * raw_Y * raw_X

    # F0 baseline (from core of first mask with voxels, but global is fine)
    f0_vals = []
    for t in range(skip, min(skip + 200, T_raw)):
        frame = np.asarray(store[t]).astype(np.float32).ravel()
        f0_vals.append(frame.mean())
    f0_global = np.percentile(f0_vals, 10)

    # Single pass: extract all traces + global signal (every 3rd frame for speed)
    traces = {r["name"]: [] for r in rois}
    global_trace = []
    for t in range(skip, T_raw, 3):
        frame = np.asarray(store[t]).astype(np.float32).ravel()
        if frame.size != n_vox:
            continue
        dff = (frame - f0_global) / (f0_global + 1e-6)
        global_trace.append(dff.mean())
        for r in rois:
            continue
        dff = (frame - f0_global) / (f0_global + 1e-6)
        for r in rois:
            if r["core"].size == 0:
                traces[r["name"]].append(0.0)
                continue
            cv = dff[r["core"]].mean()
            sv = dff[r["shell"]].mean() if r["shell"].size > 0 else 0.0
            traces[r["name"]].append(cv - sv)

    del store

    # Global trace for correlation
    global_arr = np.array(global_trace, dtype=np.float32)

    # Compute features from traces
    from scipy.stats import kurtosis as sp_kurtosis
    from scipy.signal import find_peaks

    result = {}
    for name, tr in traces.items():
        tr = np.array(tr, dtype=np.float32) * 100
        if len(tr) < 10:
            result[name] = {}
            continue

        tr_s = gaussian_filter1d(tr, sigma=0.5)
        peak_dff = float(np.percentile(tr_s, 95))
        p5 = float(np.percentile(tr_s, 5))
        baseline_noise = float(np.median(np.abs(tr_s - np.median(tr_s)))) + 1e-6
        snr = peak_dff / baseline_noise
        kurt = float(sp_kurtosis(tr_s, fisher=True))
        trace_std = float(tr_s.std())
        trace_range = float(tr_s.max() - tr_s.min())
        thr = tr_s.mean() + 2 * tr_s.std()
        peaks, _ = find_peaks(tr_s, height=thr, distance=3)
        n_transients = len(peaks)
        frac_negative = float((tr_s < -0.5).mean())
        ac1 = float(np.corrcoef(tr_s[:-1], tr_s[1:])[0, 1]) if len(tr_s) > 2 else 0.0

        # Correlation with global signal (high = neuropil/artifact, low = unique dendrite)
        if len(global_arr) == len(tr_s):
            corr_global = float(np.corrcoef(tr_s, global_arr[:len(tr_s)] * 100)[0, 1])
        else:
            n = min(len(tr_s), len(global_arr))
            corr_global = float(np.corrcoef(tr_s[:n], global_arr[:n] * 100)[0, 1])

        result[name] = {
            "peak_dff": peak_dff,
            "p5_dff": p5,
            "snr": snr,
            "kurtosis": kurt,
            "trace_std": trace_std,
            "trace_range": trace_range,
            "n_transients": n_transients,
            "frac_negative": frac_negative,
            "autocorr_lag1": ac1,
            "corr_global": corr_global,
        }

    print(f"    Got temporal features for {sum(1 for v in result.values() if v)} / {len(result)} masks")
    return result


def find_trace_csv(base):
    """Find M4 trace CSV for a run."""
    for p in [base / "traces" / "dff_traces_curated_bgsub.csv",
              base / "traces_concat" / "dff_traces_concat.csv"]:
        if p.exists():
            return p
    return None


def collect_training_data():
    """Scan curated runs, extract features + labels."""
    all_features = []
    all_labels = []
    all_run_ids = []  # track which run each sample came from

    for date, mouse, run, fps in KNOWN_RUNS:
        base = SCAPE_DATA / date / mouse / run
        curation_log = base / "labelmaps_curated_dynamic" / "curation_log.csv"
        split_folder = base / "labelmaps_split"

        if not curation_log.exists():
            print(f"  [skip] No curation log: {date}/{mouse}/{run}")
            continue
        if not split_folder.exists():
            print(f"  [skip] No split folder: {date}/{mouse}/{run}")
            continue

        with open(curation_log) as f:
            rows = list(csv.DictReader(f))

        kept_names = {r["name"] for r in rows if int(r["kept"]) == 1}
        rejected_names = {r["name"] for r in rows if int(r["kept"]) == 0}

        print(f"\n  {date}/{mouse}/{run}: {len(kept_names)} kept, {len(rejected_names)} rejected")

        mask_paths = sorted(split_folder.glob("dend_*_labelmap.tif"))

        # Filter to masks in curation log
        valid_paths = []
        valid_names = []
        valid_labels = []
        for mp in mask_paths:
            name = mp.stem.replace("_labelmap", "")
            if name in kept_names:
                valid_paths.append(mp)
                valid_names.append(name)
                valid_labels.append(1)
            elif name in rejected_names:
                valid_paths.append(mp)
                valid_names.append(name)
                valid_labels.append(0)

        # Extract temporal features for ALL masks in one pass
        temporal_map = extract_all_temporal_features(valid_paths, valid_names, base, fps)

        n_pos, n_neg, n_temporal = 0, 0, 0
        for mp, name, lbl in zip(valid_paths, valid_names, valid_labels):
            try:
                mask = tifffile.imread(mp).astype(bool)
            except Exception:
                continue

            sf = extract_spatial_features(mask)
            if sf is None:
                continue

            tf = temporal_map.get(name, {})
            if tf:
                n_temporal += 1

            features = {**sf, **tf}
            all_features.append(features)
            all_labels.append(lbl)
            all_run_ids.append(f"{date}/{run}")
            if lbl == 1:
                n_pos += 1
            else:
                n_neg += 1

        print(f"    {n_pos} pos, {n_neg} neg, {n_temporal} with temporal")

    return all_features, all_labels, all_run_ids


def train_model():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print("=== Collecting training data ===")
    features, labels, run_ids = collect_training_data()

    if len(features) < 20:
        print(f"\nToo few samples ({len(features)}).")
        return

    # Get all possible feature names
    all_keys = set()
    for f in features:
        all_keys.update(f.keys())
    feature_names = sorted(all_keys)

    X = np.zeros((len(features), len(feature_names)), dtype=np.float32)
    for i, f in enumerate(features):
        for j, fn in enumerate(feature_names):
            X[i, j] = f.get(fn, 0.0)
    y = np.array(labels, dtype=int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n=== Training ===")
    print(f"  Samples: {len(y)} ({y.sum()} pos, {(1-y).sum()} neg)")
    print(f"  Features: {len(feature_names)}: {feature_names}")

    # Cross-validation: both stratified and leave-one-run-out
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Encode run IDs for group CV
    unique_runs = sorted(set(run_ids))
    groups = np.array([unique_runs.index(r) for r in run_ids])
    logo = LeaveOneGroupOut()

    lr_pipe = Pipeline([("scaler", StandardScaler()),
                        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                                   max_iter=1000, random_state=42))])
    lr_scores = cross_val_score(lr_pipe, X, y, cv=cv, scoring="accuracy")
    lr_logo = cross_val_score(lr_pipe, X, y, cv=logo, groups=groups, scoring="accuracy")
    print(f"\n  Logistic Regression:  5-fold={lr_scores.mean():.3f}  LORO={lr_logo.mean():.3f}")

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42,
                                 class_weight="balanced", min_samples_leaf=5)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    rf_logo = cross_val_score(rf, X, y, cv=logo, groups=groups, scoring="accuracy")
    print(f"  Random Forest:        5-fold={rf_scores.mean():.3f}  LORO={rf_logo.mean():.3f}")

    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                                     random_state=42, min_samples_leaf=5,
                                     subsample=0.8)
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring="accuracy")
    gb_logo = cross_val_score(gb, X, y, cv=logo, groups=groups, scoring="accuracy")
    print(f"  Gradient Boosting:    5-fold={gb_scores.mean():.3f}  LORO={gb_logo.mean():.3f}")

    # Pick best by LORO (more honest — tests generalization to new runs)
    candidates = [("LogisticRegression", lr_pipe, lr_logo.mean()),
                  ("RandomForest", rf, rf_logo.mean()),
                  ("GradientBoosting", gb, gb_logo.mean())]
    best_name, clf, best_score = max(candidates, key=lambda x: x[2])

    print(f"\n  Best: {best_name} ({best_score:.3f})")

    # Train on all data
    clf.fit(X, y)

    # Feature importance
    print(f"\n  Top features:")
    imp = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])
    for fn, fi in imp[:10]:
        print(f"    {fn}: {fi:.3f}")

    # Confusion matrix on training data
    from sklearn.metrics import classification_report
    y_pred = clf.predict(X)
    print(f"\n  Training set report:")
    print(classification_report(y, y_pred, target_names=["rejected", "kept"]))

    model_data = {
        "classifier": clf,
        "feature_names": feature_names,
        "model_type": best_name,
        "n_positive": int(y.sum()),
        "n_negative": int((1-y).sum()),
        "cv_accuracy": float(best_score),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"✅ Model saved: {MODEL_PATH}")


def predict_masks():
    if not MODEL_PATH.exists():
        print("No model. Run --train first.")
        return

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    clf = model_data["classifier"]
    feature_names = model_data["feature_names"]
    print(f"Model: {model_data['model_type']}, CV={model_data['cv_accuracy']:.3f}")

    base = SCAPE_DATA / PREDICT_DATE / PREDICT_MOUSE / PREDICT_RUN
    for folder_name in ["labelmaps_split", "labelmaps_guided", "labelmaps"]:
        split_folder = base / folder_name
        if split_folder.exists():
            break
    else:
        print(f"No masks in {base}")
        return

    trace_csv = find_trace_csv(base)
    mask_paths = sorted(split_folder.glob("dend_*_labelmap.tif"))
    print(f"Scoring {len(mask_paths)} masks from {split_folder.name}")

    results = []
    for mp in mask_paths:
        name = mp.stem.replace("_labelmap", "")
        mask = tifffile.imread(mp).astype(bool)
        sf = extract_spatial_features(mask)
        if sf is None:
            results.append((name, 0.0))
            continue
        tf = extract_temporal_features_from_csv(name, trace_csv)
        features = {**sf, **tf}
        x = np.array([[features.get(fn, 0.0) for fn in feature_names]], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        prob = clf.predict_proba(x)[0, 1]
        results.append((name, prob))

    results.sort(key=lambda r: -r[1])
    print(f"\n{'Name':<15} {'Score':>6} {'Verdict':>10}")
    print("-" * 35)
    for name, prob in results:
        v = "✅ KEEP" if prob > 0.5 else "❌ REJECT"
        print(f"{name:<15} {prob:>6.3f} {v:>10}")

    out_csv = base / "mask_quality_scores.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "quality_score", "predicted_keep"])
        for name, prob in results:
            w.writerow([name, f"{prob:.4f}", 1 if prob > 0.5 else 0])
    print(f"\n✅ {out_csv}")


if __name__ == "__main__":
    if "--train" in sys.argv:
        train_model()
    elif "--predict" in sys.argv:
        predict_masks()
    else:
        print("Usage: --train or --predict")
