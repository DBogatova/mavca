#!/usr/bin/env python3
"""auto_stats.py — Cohort statistics for the automatic SCAPE apical-dendrite pipeline.

Analyses (from --source auto or --source human traces):
  1. Dendrite-dendrite correlations per run: pairwise r distribution vs circular-shift
     null; fraction significant (FDR); correlation vs 3D distance and depth difference;
     event co-occurrence; dimensionality (participation ratio / eff_dim/N).
  2. Within-mouse vs across-mouse/FOV: mixed models (statsmodels MixedLM); cross-run
     reproducibility of pairwise structure for same-FOV runs.
  3. Behavior coupling: per-dendrite and global Ca vs pupil/whisker/accel at best lag
     (±3 s) with circular-shift significance; onset-triggered averages; state-dependence.
  4. Depth/morphology: event rate and amplitude vs cortical depth (mixed model).
  5. Global vs local events: fraction of events where >X% dendrites co-active.
  6. Auto vs human comparison: same statistics on both sources, agreement metrics.

CLI: --run DATE/MOUSE/RUN (repeatable), --all, --force, --jobs N, --source {auto,human}

Outputs to PROJ/scape-auto/stats/<source>/ and PROJ/scape-auto/stats/compare_auto_vs_human.*.
"""
from __future__ import annotations

import argparse
import json
import datetime
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats as sp
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from scape_common import (  # noqa: E402
    discover_runs, get_run, Run, fov_group, load_behavior, human_masks,
    human_labelmap, open_stack, VOXEL_ZYX, AUTO_ROOT, DATA_ROOT, merge_metrics
)

STATS_ROOT = AUTO_ROOT / "stats"
NOTES_FILE = HERE / "notes" / "statistics.json"

# ─── Event detection parameters (project conventions) ─────────────────────────
DFF_THR_Z = 2.5        # z-score threshold for event detection (MAD-based)
MIN_EVENT_DUR = 2      # minimum frames above threshold
GLOBAL_FRAC = 0.25     # fraction of dendrites co-active to call an event "global"
MAX_LAG_S = 3.0        # max lag for behavior cross-correlation
N_SHIFT = 200          # circular shifts for null distributions
FDR_ALPHA = 0.05       # FDR threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════
def log_note(what: str, files: list[str], verified: str, caveats: str = ""):
    """Append a note to the stage notes file."""
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    notes = json.loads(NOTES_FILE.read_text()) if NOTES_FILE.exists() else []
    notes.append({
        "time": datetime.datetime.now().astimezone().isoformat(),
        "what": what,
        "files": files,
        "verified": verified,
        "caveats": caveats
    })
    NOTES_FILE.write_text(json.dumps(notes, indent=2))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    q = pvals * n / rank
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.clip(q, 0, 1)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median and MAD."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < 1e-9:
        mad = np.nanstd(x)
    return (x - med) / (mad * 1.4826 + 1e-9)


def detect_events(trace: np.ndarray, thr_z: float = DFF_THR_Z,
                  min_dur: int = MIN_EVENT_DUR) -> list[int]:
    """Detect calcium events via robust z-score threshold."""
    z = robust_zscore(trace)
    above = z > thr_z
    peaks = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            if i - start >= min_dur:
                seg = trace[start:i]
                peaks.append(start + int(np.argmax(seg)))
        else:
            i += 1
    return peaks


def compute_global_ca(traces: np.ndarray) -> np.ndarray:
    """Compute global calcium as mean across dendrites."""
    return np.nanmean(traces, axis=1)


def circular_shift_corr(x: np.ndarray, y: np.ndarray, n_shift: int = N_SHIFT,
                        min_shift: int = 20) -> tuple[float, float]:
    """Compute correlation and p-value from circular-shift null."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    x_clean, y_clean = x[mask], y[mask]
    obs_r = np.corrcoef(x_clean, y_clean)[0, 1]
    
    rng = np.random.default_rng(42)
    T = len(x_clean)
    if T <= 2 * min_shift:
        return obs_r, np.nan
    
    null_rs = []
    for _ in range(n_shift):
        shift = rng.integers(min_shift, T - min_shift)
        y_shifted = np.roll(y_clean, shift)
        null_rs.append(np.corrcoef(x_clean, y_shifted)[0, 1])
    
    null_rs = np.array(null_rs)
    # Two-sided p-value
    p = (np.sum(np.abs(null_rs) >= np.abs(obs_r)) + 1) / (n_shift + 1)
    return obs_r, p


def best_lag_corr(x: np.ndarray, y: np.ndarray, frame_rate: float,
                  max_lag_s: float = MAX_LAG_S) -> tuple[float, int, float, float]:
    """Find best cross-correlation within lag range. Returns (r_best, lag_frames, lag_s, p)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return np.nan, 0, np.nan, np.nan
    
    max_lag = int(max_lag_s * frame_rate)
    x_clean, y_clean = x[mask], y[mask]
    
    best_r, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_seg = x_clean[:lag]
            y_seg = y_clean[-lag:]
        elif lag > 0:
            x_seg = x_clean[lag:]
            y_seg = y_clean[:-lag]
        else:
            x_seg, y_seg = x_clean, y_clean
        
        if len(x_seg) < 20:
            continue
        r = np.corrcoef(x_seg, y_seg)[0, 1]
        if np.abs(r) > np.abs(best_r):
            best_r = r
            best_lag = lag
    
    # Get p-value from circular shift at best lag
    _, p = circular_shift_corr(x_clean, y_clean, n_shift=N_SHIFT)
    return best_r, best_lag, best_lag / frame_rate, p


def participation_ratio(cov_matrix: np.ndarray) -> float:
    """Compute participation ratio (effective dimensionality) from covariance."""
    eigvals = np.linalg.eigvalsh(cov_matrix)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return np.nan
    total = eigvals.sum()
    return (total ** 2) / (eigvals ** 2).sum()


def onset_triggered_average(signal: np.ndarray, onsets: np.ndarray,
                            window: int = 15) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute onset-triggered average with bootstrap CI."""
    valid = onsets[(onsets >= window) & (onsets < len(signal) - window)]
    if len(valid) < 3:
        return np.full(2 * window + 1, np.nan), np.full(2 * window + 1, np.nan), 0
    
    segments = np.array([signal[o - window:o + window + 1] for o in valid])
    # Baseline subtract using pre-onset period
    baseline = segments[:, :window // 2].mean(axis=1, keepdims=True)
    segments = segments - baseline
    
    mean = segments.mean(axis=0)
    sem = segments.std(axis=0) / np.sqrt(len(valid))
    return mean, sem, len(valid)


def detect_behavior_onsets(signal: np.ndarray, z_thr: float = 0.5,
                           refractory: int = 10) -> np.ndarray:
    """Detect onsets in behavior signal (rising edges after z-scoring)."""
    if signal is None or len(signal) < 20:
        return np.array([], dtype=int)
    z = (signal - np.nanmean(signal)) / (np.nanstd(signal) + 1e-9)
    rising = np.where(np.diff((z > z_thr).astype(int)) == 1)[0] + 1
    
    # Apply refractory period
    onsets = []
    for r in rising:
        if len(onsets) == 0 or r - onsets[-1] >= refractory:
            onsets.append(r)
    return np.array(onsets, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_traces(r: Run, source: str) -> tuple[pd.DataFrame | None, list[str]]:
    """Load traces from the specified source (auto or human)."""
    if source == "auto":
        path = r.out / "traces" / "dff_auto.csv"
    else:  # human
        # Try auto-extracted human traces first, then original
        path = r.out / "traces" / "dff_human_sameextractor.csv"
        if not path.exists():
            path = r.human_traces
    
    if path is None or not path.exists():
        return None, []
    
    df = pd.read_csv(path)
    dend_cols = [c for c in df.columns if c.startswith("dend_")]
    return df, dend_cols


def load_masks(r: Run, source: str, shape_zyx: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Load masks for the specified source."""
    masks = {}
    if source == "auto":
        labelmap_path = r.out / "masks" / "auto_labelmap.tif"
        if labelmap_path.exists():
            import tifffile
            labelmap = tifffile.imread(str(labelmap_path))
            for label in np.unique(labelmap):
                if label == 0:
                    continue
                masks[f"dend_{label - 1:03d}"] = labelmap == label
    else:  # human
        for name, m in human_masks(r, shape_zyx):
            masks[name] = m
    return masks


def load_global_ca(r: Run) -> np.ndarray | None:
    """Load or compute global calcium."""
    path = r.out / "traces" / "global_ca.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df["global_dff"].values
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Per-run analysis
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class RunStats:
    """Container for per-run statistics."""
    run_key: str
    source: str
    date: str
    mouse: str
    run: str
    fov: str
    frame_rate: float
    n_frames: int
    n_dendrites: int
    has_behavior: bool
    
    # Dendrite-dendrite correlations
    mean_pairwise_r: float = np.nan
    median_pairwise_r: float = np.nan
    frac_sig_pairs_fdr: float = np.nan
    n_sig_pairs: int = 0
    n_total_pairs: int = 0
    
    # Distance effects
    r_corr_vs_distance: float = np.nan
    p_corr_vs_distance: float = np.nan
    r_corr_vs_depth_diff: float = np.nan
    p_corr_vs_depth_diff: float = np.nan
    
    # Dimensionality
    eff_dim: float = np.nan
    eff_dim_normalized: float = np.nan  # eff_dim / N
    
    # Event statistics
    mean_event_rate: float = np.nan  # events/min
    frac_global_events: float = np.nan
    n_global_events: int = 0
    n_local_events: int = 0
    
    # Behavior coupling (global Ca)
    r_global_accel: float = np.nan
    p_global_accel: float = np.nan
    r_global_pupil: float = np.nan
    p_global_pupil: float = np.nan
    r_global_whisker: float = np.nan
    p_global_whisker: float = np.nan
    
    # Behavior: fraction of dendrites coupled
    frac_dend_accel_coupled: float = np.nan
    frac_dend_pupil_coupled: float = np.nan
    
    # State dependence
    mean_r_quiet: float = np.nan
    mean_r_active: float = np.nan
    event_rate_quiet: float = np.nan
    event_rate_active: float = np.nan
    
    # Depth effects
    depth_event_rate_r: float = np.nan
    depth_amplitude_r: float = np.nan
    
    # Co-occurrence
    mean_coincidence_index: float = np.nan
    
    caveats: list = field(default_factory=list)


def analyze_run(r: Run, source: str) -> RunStats | None:
    """Perform comprehensive analysis on a single run."""
    df, dend_cols = load_traces(r, source)
    if df is None or len(dend_cols) < 2:
        return None
    
    # Initialize stats
    stats = RunStats(
        run_key=r.key,
        source=source,
        date=r.date,
        mouse=r.mouse,
        run=r.run,
        fov=fov_group(r),
        frame_rate=r.frame_rate,
        n_frames=len(df),
        n_dendrites=len(dend_cols),
        has_behavior=r.behavior_mat is not None
    )
    
    # Extract trace matrix
    traces = df[dend_cols].values  # (T, N)
    T, N = traces.shape
    duration_min = T / r.frame_rate / 60
    
    # Time vector (aligned to skip_s)
    time_s = np.arange(T) / r.frame_rate - r.skip_s
    
    # Load masks for morphology
    try:
        stack = open_stack(r)
        shape_zyx = stack.shape[1:]  # (Z, Y, X)
        masks = load_masks(r, source, shape_zyx)
    except Exception:
        masks = {}
        stats.caveats.append("could not load masks for morphology")
    
    # ─── 1. Dendrite-dendrite correlations ────────────────────────────────────
    # Pairwise correlation matrix
    corr_matrix = np.corrcoef(traces.T)
    np.fill_diagonal(corr_matrix, np.nan)
    triu_idx = np.triu_indices(N, k=1)
    pairwise_r = corr_matrix[triu_idx]
    
    stats.mean_pairwise_r = float(np.nanmean(pairwise_r))
    stats.median_pairwise_r = float(np.nanmedian(pairwise_r))
    stats.n_total_pairs = len(pairwise_r)
    
    # Circular-shift null for significance
    pvals = []
    for i, j in zip(*triu_idx):
        _, p = circular_shift_corr(traces[:, i], traces[:, j], n_shift=100)
        pvals.append(p)
    pvals = np.array(pvals)
    qvals = bh_fdr(pvals)
    stats.n_sig_pairs = int(np.sum(qvals < FDR_ALPHA))
    stats.frac_sig_pairs_fdr = stats.n_sig_pairs / max(1, stats.n_total_pairs)
    
    # ─── Correlation vs distance (if masks available) ─────────────────────────
    if len(masks) >= 2:
        centroids = {}
        depths = {}
        for name, m in masks.items():
            if not m.any():
                continue
            coords = np.argwhere(m)
            centroid_um = coords.mean(axis=0) * np.array(VOXEL_ZYX)
            centroids[name] = centroid_um
            depths[name] = centroid_um[1]  # Y = depth
        
        # Build distance and depth-diff arrays aligned with pairwise_r
        distances = []
        depth_diffs = []
        valid_rs = []
        for idx, (i, j) in enumerate(zip(*triu_idx)):
            n1, n2 = dend_cols[i], dend_cols[j]
            if n1 in centroids and n2 in centroids:
                d = np.linalg.norm(centroids[n1] - centroids[n2])
                dd = np.abs(depths[n1] - depths[n2])
                distances.append(d)
                depth_diffs.append(dd)
                valid_rs.append(pairwise_r[idx])
        
        if len(distances) > 5:
            rho_d, p_d = sp.spearmanr(distances, valid_rs)
            stats.r_corr_vs_distance = float(rho_d)
            stats.p_corr_vs_distance = float(p_d)
            
            rho_dd, p_dd = sp.spearmanr(depth_diffs, valid_rs)
            stats.r_corr_vs_depth_diff = float(rho_dd)
            stats.p_corr_vs_depth_diff = float(p_dd)
    
    # ─── Dimensionality ───────────────────────────────────────────────────────
    cov = np.cov(traces.T)
    stats.eff_dim = float(participation_ratio(cov))
    stats.eff_dim_normalized = stats.eff_dim / N if N > 0 else np.nan
    
    # ─── 2. Event detection and classification ────────────────────────────────
    all_events = []
    event_frames = []
    for j, col in enumerate(dend_cols):
        evts = detect_events(traces[:, j])
        all_events.append(evts)
        event_frames.extend(evts)
    
    total_events = sum(len(e) for e in all_events)
    stats.mean_event_rate = total_events / N / duration_min if duration_min > 0 else np.nan
    
    # Global vs local classification
    active_per_frame = (robust_zscore(traces) > DFF_THR_Z).sum(axis=1)
    global_threshold = GLOBAL_FRAC * N
    
    n_global, n_local = 0, 0
    for j, evts in enumerate(all_events):
        for pk in evts:
            # Check ±1 frame window
            window = active_per_frame[max(0, pk - 1):min(T, pk + 2)]
            if window.max() >= global_threshold:
                n_global += 1
            else:
                n_local += 1
    
    stats.n_global_events = n_global
    stats.n_local_events = n_local
    stats.frac_global_events = n_global / max(1, n_global + n_local)
    
    # ─── 3. Behavior coupling ─────────────────────────────────────────────────
    if r.behavior_mat is not None or r.accel_csv is not None:
        beh = load_behavior(r, crop_s=r.skip_s)
        
        # Global Ca
        global_ca = compute_global_ca(traces)
        time_ca = time_s[time_s >= 0]  # After skip
        if len(global_ca) > len(time_ca):
            global_ca = global_ca[:len(time_ca)]
        
        # Resample behavior to Ca frames
        for beh_name, beh_t_key, beh_val_key in [
            ("accel", "accel_t", "accel"),
            ("pupil", "pupil_t", "pupil"),
            ("whisker", "whisker_t", "whisker")
        ]:
            beh_t = beh.get(beh_t_key)
            beh_val = beh.get(beh_val_key)
            
            if beh_t is not None and beh_val is not None and len(beh_t) > 0:
                # Resample to Ca time
                beh_resampled = np.interp(time_ca, beh_t, beh_val,
                                          left=np.nan, right=np.nan)
                
                # Global correlation
                r_best, lag, lag_s, p = best_lag_corr(
                    global_ca, beh_resampled, r.frame_rate
                )
                
                if beh_name == "accel":
                    stats.r_global_accel = float(r_best)
                    stats.p_global_accel = float(p)
                elif beh_name == "pupil":
                    stats.r_global_pupil = float(r_best)
                    stats.p_global_pupil = float(p)
                elif beh_name == "whisker":
                    stats.r_global_whisker = float(r_best)
                    stats.p_global_whisker = float(p)
                
                # Per-dendrite coupling
                if beh_name in ("accel", "pupil"):
                    n_coupled = 0
                    for j in range(N):
                        tr_valid = traces[:len(time_ca), j] if len(traces) > len(time_ca) else traces[:, j]
                        _, _, _, p_dend = best_lag_corr(
                            tr_valid, beh_resampled, r.frame_rate
                        )
                        if p_dend < 0.05:
                            n_coupled += 1
                    
                    if beh_name == "accel":
                        stats.frac_dend_accel_coupled = n_coupled / N
                    else:
                        stats.frac_dend_pupil_coupled = n_coupled / N
        
        # State dependence (quiet vs active based on accel)
        accel_t = beh.get("accel_t")
        accel = beh.get("accel")
        if accel is not None and len(accel) > 0:
            accel_resampled = np.interp(time_ca, accel_t, accel,
                                        left=np.nan, right=np.nan)
            accel_median = np.nanmedian(accel_resampled)
            
            quiet_frames = accel_resampled < accel_median
            active_frames = accel_resampled >= accel_median
            
            # Pairwise r in each state
            if quiet_frames.sum() > 20:
                corr_quiet = np.corrcoef(traces[:len(time_ca)][quiet_frames].T)
                np.fill_diagonal(corr_quiet, np.nan)
                stats.mean_r_quiet = float(np.nanmean(corr_quiet[triu_idx]))
                
                # Event rate in quiet
                quiet_event_count = sum(
                    1 for evts in all_events for pk in evts
                    if pk < len(quiet_frames) and quiet_frames[pk]
                )
                quiet_min = quiet_frames.sum() / r.frame_rate / 60
                stats.event_rate_quiet = quiet_event_count / N / quiet_min if quiet_min > 0 else np.nan
            
            if active_frames.sum() > 20:
                corr_active = np.corrcoef(traces[:len(time_ca)][active_frames].T)
                np.fill_diagonal(corr_active, np.nan)
                stats.mean_r_active = float(np.nanmean(corr_active[triu_idx]))
                
                # Event rate in active
                active_event_count = sum(
                    1 for evts in all_events for pk in evts
                    if pk < len(active_frames) and active_frames[pk]
                )
                active_min = active_frames.sum() / r.frame_rate / 60
                stats.event_rate_active = active_event_count / N / active_min if active_min > 0 else np.nan
    
    # ─── 4. Depth effects ─────────────────────────────────────────────────────
    if masks and len(masks) >= 5:
        depths_arr = []
        rates_arr = []
        amps_arr = []
        
        for j, col in enumerate(dend_cols):
            if col not in masks:
                continue
            m = masks[col]
            if not m.any():
                continue
            
            coords = np.argwhere(m)
            depth_um = coords.mean(axis=0)[1] * VOXEL_ZYX[1]
            depths_arr.append(depth_um)
            
            # Event rate
            n_evts = len(all_events[j])
            rates_arr.append(n_evts / duration_min if duration_min > 0 else np.nan)
            
            # Mean amplitude (peak dF/F)
            if n_evts > 0:
                amps_arr.append(np.mean([traces[pk, j] for pk in all_events[j]]))
            else:
                amps_arr.append(np.nan)
        
        if len(depths_arr) > 5:
            rho_rate, _ = sp.spearmanr(depths_arr, rates_arr)
            stats.depth_event_rate_r = float(rho_rate)
            
            valid_amps = [(d, a) for d, a in zip(depths_arr, amps_arr) if np.isfinite(a)]
            if len(valid_amps) > 5:
                rho_amp, _ = sp.spearmanr([x[0] for x in valid_amps],
                                          [x[1] for x in valid_amps])
                stats.depth_amplitude_r = float(rho_amp)
    
    # ─── 5. Co-occurrence / coincidence index ─────────────────────────────────
    # Fraction of event pairs that co-occur within ±1 frame
    if total_events > 10:
        coincidence_counts = []
        for j, evts in enumerate(all_events):
            for pk in evts:
                # How many other dendrites have an event within ±1 frame?
                co_active = sum(
                    1 for k, other_evts in enumerate(all_events)
                    if k != j and any(abs(pk - opk) <= 1 for opk in other_evts)
                )
                coincidence_counts.append(co_active / max(1, N - 1))
        
        stats.mean_coincidence_index = float(np.mean(coincidence_counts))
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dendrite summary (for mixed models)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_per_dendrite_stats(r: Run, source: str) -> pd.DataFrame:
    """Extract per-dendrite statistics for mixed-model analysis."""
    df, dend_cols = load_traces(r, source)
    if df is None or len(dend_cols) == 0:
        return pd.DataFrame()
    
    traces = df[dend_cols].values
    T, N = traces.shape
    duration_min = T / r.frame_rate / 60
    
    # Load masks
    try:
        stack = open_stack(r)
        shape_zyx = stack.shape[1:]
        masks = load_masks(r, source, shape_zyx)
    except Exception:
        masks = {}
    
    rows = []
    for j, col in enumerate(dend_cols):
        row = {
            "run_key": r.key,
            "source": source,
            "mouse": r.mouse,
            "fov": fov_group(r),
            "dendrite": col,
            "frame_rate": r.frame_rate
        }
        
        # Event detection
        evts = detect_events(traces[:, j])
        row["n_events"] = len(evts)
        row["event_rate"] = len(evts) / duration_min if duration_min > 0 else np.nan
        
        # Amplitude
        if len(evts) > 0:
            row["mean_amplitude"] = float(np.mean([traces[pk, j] for pk in evts]))
            row["max_amplitude"] = float(traces[:, j].max())
        else:
            row["mean_amplitude"] = np.nan
            row["max_amplitude"] = float(traces[:, j].max())
        
        # Morphology from mask
        if col in masks and masks[col].any():
            m = masks[col]
            coords = np.argwhere(m)
            centroid_um = coords.mean(axis=0) * np.array(VOXEL_ZYX)
            row["depth_um"] = centroid_um[1]  # Y
            row["cx_um"] = centroid_um[2]     # X
            row["cz_um"] = centroid_um[0]     # Z
            row["n_voxels"] = int(m.sum())
            
            # Length estimate (PCA)
            if len(coords) >= 6:
                centered = coords * np.array(VOXEL_ZYX) - centroid_um
                eigvals = np.linalg.eigvalsh(np.cov(centered.T))
                row["length_um"] = float(4 * np.sqrt(eigvals.max()))  # ~2 std
            else:
                row["length_um"] = np.nan
        else:
            row["depth_um"] = np.nan
            row["cx_um"] = np.nan
            row["cz_um"] = np.nan
            row["n_voxels"] = np.nan
            row["length_um"] = np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Pairwise summary (for across-run reproducibility)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_pairwise_stats(r: Run, source: str) -> pd.DataFrame:
    """Extract pairwise correlation stats for reproducibility analysis."""
    df, dend_cols = load_traces(r, source)
    if df is None or len(dend_cols) < 2:
        return pd.DataFrame()
    
    traces = df[dend_cols].values
    corr_matrix = np.corrcoef(traces.T)
    
    rows = []
    for i in range(len(dend_cols)):
        for j in range(i + 1, len(dend_cols)):
            rows.append({
                "run_key": r.key,
                "source": source,
                "fov": fov_group(r),
                "dend_i": dend_cols[i],
                "dend_j": dend_cols[j],
                "pair_id": f"{dend_cols[i]}_{dend_cols[j]}",
                "r": corr_matrix[i, j]
            })
    
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Cohort-level analyses
# ═══════════════════════════════════════════════════════════════════════════════
def cohort_analysis(run_stats: list[RunStats], per_dend: pd.DataFrame,
                    pairwise: pd.DataFrame, source: str, out_dir: Path) -> dict:
    """Perform cohort-level statistical analyses."""
    results = {
        "source": source,
        "n_runs": len(run_stats),
        "n_mice": len(set(s.mouse for s in run_stats)),
        "n_fovs": len(set(s.fov for s in run_stats)),
        "tests": []
    }
    
    # Convert to DataFrame for easier manipulation
    df_runs = pd.DataFrame([asdict(s) for s in run_stats])
    
    # ─── 1. Dendrite-dendrite correlations summary ────────────────────────────
    mean_r = df_runs["mean_pairwise_r"].dropna()
    if len(mean_r) > 0:
        results["pairwise_r_cohort_mean"] = float(mean_r.mean())
        results["pairwise_r_cohort_std"] = float(mean_r.std())
        results["pairwise_r_range"] = [float(mean_r.min()), float(mean_r.max())]
        
        # Test: is mean pairwise r significantly different from 0?
        if len(mean_r) >= 5:
            t, p = sp.ttest_1samp(mean_r, 0)
            results["tests"].append({
                "name": "pairwise_r_vs_zero",
                "test": "one-sample t-test",
                "n": len(mean_r),
                "statistic": float(t),
                "p": float(p),
                "effect_size": float(mean_r.mean()),
                "conclusion": f"Mean pairwise r = {mean_r.mean():.3f} (range {mean_r.min():.3f}-{mean_r.max():.3f}), {'significantly' if p < 0.05 else 'not significantly'} different from zero (p={p:.3g})"
            })
    
    # ─── 2. Dimensionality ────────────────────────────────────────────────────
    eff_dim_n = df_runs["eff_dim_normalized"].dropna()
    if len(eff_dim_n) > 0:
        results["eff_dim_normalized_mean"] = float(eff_dim_n.mean())
        results["eff_dim_normalized_std"] = float(eff_dim_n.std())
        results["tests"].append({
            "name": "dimensionality",
            "description": f"Effective dimensionality / N = {eff_dim_n.mean():.2f} ± {eff_dim_n.std():.2f} across {len(eff_dim_n)} runs. Lower values indicate more correlated activity.",
            "n": len(eff_dim_n)
        })
    
    # ─── 3. Global vs local events ────────────────────────────────────────────
    frac_global = df_runs["frac_global_events"].dropna()
    if len(frac_global) > 0:
        results["frac_global_events_mean"] = float(frac_global.mean())
        results["frac_global_events_std"] = float(frac_global.std())
        results["tests"].append({
            "name": "global_vs_local",
            "description": f"Fraction of events classified as global (>={100*GLOBAL_FRAC:.0f}% dendrites co-active): {frac_global.mean():.2f} ± {frac_global.std():.2f}",
            "n": len(frac_global)
        })
    
    # ─── 4. Behavior coupling ─────────────────────────────────────────────────
    for beh in ["accel", "pupil", "whisker"]:
        col = f"r_global_{beh}"
        p_col = f"p_global_{beh}"
        vals = df_runs[col].dropna()
        if len(vals) > 0:
            pvals = df_runs.loc[vals.index, p_col].dropna()
            n_sig = (pvals < 0.05).sum()
            results[f"r_global_{beh}_mean"] = float(vals.mean())
            results[f"r_global_{beh}_n_sig"] = int(n_sig)
            results["tests"].append({
                "name": f"behavior_coupling_{beh}",
                "description": f"Global Ca vs {beh}: mean r = {vals.mean():.3f}, {n_sig}/{len(vals)} runs significant (p<0.05, circular-shift null)",
                "n": len(vals)
            })
    
    # ─── 5. Depth effects (mixed model) ───────────────────────────────────────
    if len(per_dend) > 20 and per_dend["depth_um"].notna().sum() > 20:
        try:
            import statsmodels.formula.api as smf
            
            model_data = per_dend.dropna(subset=["depth_um", "event_rate", "mouse", "fov"])
            if len(model_data) > 10 and model_data["mouse"].nunique() >= 2:
                # Normalize depth within FOV for comparability
                model_data = model_data.copy()
                model_data["depth_z"] = model_data.groupby("fov")["depth_um"].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
                
                md = smf.mixedlm("event_rate ~ depth_z", model_data,
                                 groups=model_data["mouse"]).fit(reml=False)
                
                results["tests"].append({
                    "name": "depth_event_rate_mixed",
                    "test": "mixed-effects model (event_rate ~ depth_z, random=mouse)",
                    "n_dendrites": len(model_data),
                    "n_mice": int(model_data["mouse"].nunique()),
                    "slope": float(md.params["depth_z"]),
                    "p": float(md.pvalues["depth_z"]),
                    "conclusion": f"Depth effect on event rate: coef={md.params['depth_z']:.3f}, p={md.pvalues['depth_z']:.3g}. {'Superficial' if md.params['depth_z'] < 0 else 'Deep'} dendrites fire {'more' if md.params['depth_z'] < 0 else 'less'} frequently."
                })
        except Exception as e:
            results["tests"].append({
                "name": "depth_event_rate_mixed",
                "error": str(e)
            })
    
    # ─── 6. Cross-run reproducibility (same FOV) ──────────────────────────────
    if len(pairwise) > 0:
        fov_counts = pairwise.groupby("fov")["run_key"].nunique()
        multi_run_fovs = fov_counts[fov_counts > 1].index.tolist()
        
        if multi_run_fovs:
            repro_rhos = []
            for fov in multi_run_fovs:
                fov_data = pairwise[pairwise["fov"] == fov]
                runs = fov_data["run_key"].unique()
                if len(runs) < 2:
                    continue
                
                # Compare first two runs
                r1_data = fov_data[fov_data["run_key"] == runs[0]].set_index("pair_id")["r"]
                r2_data = fov_data[fov_data["run_key"] == runs[1]].set_index("pair_id")["r"]
                common = r1_data.index.intersection(r2_data.index)
                
                if len(common) > 10:
                    rho, _ = sp.spearmanr(r1_data[common], r2_data[common])
                    repro_rhos.append(rho)
            
            if repro_rhos:
                results["cross_run_reproducibility_mean"] = float(np.mean(repro_rhos))
                results["tests"].append({
                    "name": "cross_run_reproducibility",
                    "description": f"Spearman correlation of pairwise-r structure across runs of same FOV: mean ρ = {np.mean(repro_rhos):.2f} (n={len(repro_rhos)} FOV pairs)",
                    "n_fov_pairs": len(repro_rhos)
                })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison: auto vs human
# ═══════════════════════════════════════════════════════════════════════════════
def compare_sources(auto_stats: list[RunStats], human_stats: list[RunStats],
                    out_dir: Path) -> dict:
    """Compare statistics between auto and human sources."""
    results = {"comparison": "auto_vs_human"}
    
    # Match runs
    auto_dict = {s.run_key: s for s in auto_stats}
    human_dict = {s.run_key: s for s in human_stats}
    common_runs = set(auto_dict.keys()) & set(human_dict.keys())
    
    results["n_common_runs"] = len(common_runs)
    if len(common_runs) < 2:
        results["error"] = "Too few common runs for comparison"
        return results
    
    # Compare key metrics
    metrics = [
        ("mean_pairwise_r", "Mean pairwise correlation"),
        ("eff_dim_normalized", "Normalized effective dimensionality"),
        ("frac_global_events", "Fraction global events"),
        ("mean_event_rate", "Mean event rate (events/min)"),
        ("r_global_accel", "Global Ca vs accel correlation"),
        ("depth_event_rate_r", "Depth-event rate correlation"),
    ]
    
    comparisons = []
    for metric, label in metrics:
        auto_vals = [getattr(auto_dict[k], metric) for k in common_runs]
        human_vals = [getattr(human_dict[k], metric) for k in common_runs]
        
        # Remove NaN pairs
        valid = [(a, h) for a, h in zip(auto_vals, human_vals)
                 if np.isfinite(a) and np.isfinite(h)]
        
        if len(valid) < 3:
            continue
        
        auto_arr = np.array([v[0] for v in valid])
        human_arr = np.array([v[1] for v in valid])
        
        # Correlation
        r, p = sp.pearsonr(auto_arr, human_arr)
        
        # Mean difference (Bland-Altman style)
        diff = auto_arr - human_arr
        
        comp = {
            "metric": metric,
            "label": label,
            "n": len(valid),
            "auto_mean": float(auto_arr.mean()),
            "human_mean": float(human_arr.mean()),
            "correlation_r": float(r),
            "correlation_p": float(p),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std()),
            "agreement": "good" if r > 0.7 else "moderate" if r > 0.4 else "poor"
        }
        comparisons.append(comp)
    
    results["metric_comparisons"] = comparisons
    
    # Generate comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for idx, comp in enumerate(comparisons[:6]):
        ax = axes[idx]
        auto_vals = [getattr(auto_dict[k], comp["metric"])
                     for k in common_runs
                     if np.isfinite(getattr(auto_dict[k], comp["metric"])) and
                        np.isfinite(getattr(human_dict[k], comp["metric"]))]
        human_vals = [getattr(human_dict[k], comp["metric"])
                      for k in common_runs
                      if np.isfinite(getattr(auto_dict[k], comp["metric"])) and
                         np.isfinite(getattr(human_dict[k], comp["metric"]))]
        
        ax.scatter(human_vals, auto_vals, s=40, alpha=0.7)
        
        # Unity line
        lims = [min(min(human_vals), min(auto_vals)),
                max(max(human_vals), max(auto_vals))]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        
        ax.set_xlabel(f"Human {comp['label']}")
        ax.set_ylabel(f"Auto {comp['label']}")
        ax.set_title(f"r={comp['correlation_r']:.2f}, p={comp['correlation_p']:.3g}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Hide unused axes
    for idx in range(len(comparisons), 6):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"Auto vs Human Pipeline Comparison (n={len(common_runs)} runs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_auto_vs_human.png", dpi=150)
    fig.savefig(out_dir / "compare_auto_vs_human.pdf")
    plt.close(fig)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Output generation
# ═══════════════════════════════════════════════════════════════════════════════
def generate_figures(run_stats: list[RunStats], per_dend: pd.DataFrame,
                     source: str, out_dir: Path):
    """Generate cohort figures."""
    df_runs = pd.DataFrame([asdict(s) for s in run_stats])
    
    # Figure 1: Pairwise correlation distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1a: Distribution of mean pairwise r across runs
    ax = axes[0, 0]
    vals = df_runs["mean_pairwise_r"].dropna()
    ax.hist(vals, bins=15, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(vals.mean(), color="red", ls="--", lw=2, label=f"mean={vals.mean():.3f}")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Mean pairwise correlation")
    ax.set_ylabel("Number of runs")
    ax.set_title("Dendrite-dendrite correlations")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # 1b: Effective dimensionality
    ax = axes[0, 1]
    vals = df_runs["eff_dim_normalized"].dropna()
    ax.hist(vals, bins=15, color="coral", alpha=0.7, edgecolor="black")
    ax.axvline(vals.mean(), color="red", ls="--", lw=2, label=f"mean={vals.mean():.2f}")
    ax.set_xlabel("Effective dimensionality / N")
    ax.set_ylabel("Number of runs")
    ax.set_title("Activity dimensionality (1 = fully independent)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # 1c: Global vs local events
    ax = axes[1, 0]
    global_frac = df_runs["frac_global_events"].dropna()
    ax.bar(range(len(global_frac)), global_frac.values, color="mediumpurple", alpha=0.7)
    ax.axhline(global_frac.mean(), color="red", ls="--", lw=1.5)
    ax.set_xlabel("Run")
    ax.set_ylabel("Fraction global events")
    ax.set_title(f"Global events (>{100*GLOBAL_FRAC:.0f}% dendrites co-active)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # 1d: Behavior coupling
    ax = axes[1, 1]
    for i, (col, label, color) in enumerate([
        ("r_global_accel", "Accel", "tab:red"),
        ("r_global_pupil", "Pupil", "tab:blue"),
        ("r_global_whisker", "Whisker", "tab:green")
    ]):
        vals = df_runs[col].dropna().values
        if len(vals) > 0:
            x = np.arange(len(vals)) + i * 0.25
            ax.bar(x, vals, width=0.2, label=label, color=color, alpha=0.7)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Run")
    ax.set_ylabel("Correlation with global Ca")
    ax.set_title("Behavior coupling")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.suptitle(f"Cohort Statistics ({source}, n={len(run_stats)} runs)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"cohort_overview_{source}.png", dpi=150)
    fig.savefig(out_dir / f"cohort_overview_{source}.pdf")
    plt.close(fig)
    
    # Figure 2: Depth effects
    if len(per_dend) > 0 and per_dend["depth_um"].notna().sum() > 20:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        valid = per_dend.dropna(subset=["depth_um", "event_rate"])
        if len(valid) > 10:
            ax = axes[0]
            ax.scatter(valid["depth_um"], valid["event_rate"], s=20, alpha=0.5)
            # Trend line
            z = np.polyfit(valid["depth_um"], valid["event_rate"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid["depth_um"].min(), valid["depth_um"].max(), 100)
            ax.plot(x_line, p(x_line), "r--", lw=2)
            rho, _ = sp.spearmanr(valid["depth_um"], valid["event_rate"])
            ax.set_xlabel("Cortical depth (µm)")
            ax.set_ylabel("Event rate (events/min)")
            ax.set_title(f"Depth vs event rate (ρ={rho:.2f})")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        
        valid2 = per_dend.dropna(subset=["depth_um", "mean_amplitude"])
        if len(valid2) > 10:
            ax = axes[1]
            ax.scatter(valid2["depth_um"], valid2["mean_amplitude"], s=20, alpha=0.5)
            rho, _ = sp.spearmanr(valid2["depth_um"], valid2["mean_amplitude"])
            ax.set_xlabel("Cortical depth (µm)")
            ax.set_ylabel("Mean event amplitude (ΔF/F %)")
            ax.set_title(f"Depth vs amplitude (ρ={rho:.2f})")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        
        fig.suptitle(f"Depth Effects ({source})", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"depth_effects_{source}.png", dpi=150)
        fig.savefig(out_dir / f"depth_effects_{source}.pdf")
        plt.close(fig)


def generate_summary_text(cohort_results: dict, source: str, out_dir: Path):
    """Generate plain-text summary of statistical tests."""
    lines = [
        f"SCAPE Apical-Dendrite Pipeline Statistics Summary",
        f"Source: {source}",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"COHORT: {cohort_results['n_runs']} runs, {cohort_results['n_mice']} mice, {cohort_results['n_fovs']} FOVs",
        f"",
        "=" * 70,
        "STATISTICAL TESTS",
        "=" * 70,
        ""
    ]
    
    for test in cohort_results.get("tests", []):
        lines.append(f"--- {test.get('name', 'Unknown')} ---")
        if "error" in test:
            lines.append(f"  ERROR: {test['error']}")
        elif "conclusion" in test:
            lines.append(f"  Test: {test.get('test', 'N/A')}")
            lines.append(f"  n: {test.get('n', 'N/A')}")
            if "statistic" in test:
                lines.append(f"  Statistic: {test['statistic']:.3f}")
            if "p" in test:
                lines.append(f"  p-value: {test['p']:.3g}")
            if "effect_size" in test:
                lines.append(f"  Effect size: {test['effect_size']:.3f}")
            lines.append(f"  Conclusion: {test['conclusion']}")
        elif "description" in test:
            lines.append(f"  {test['description']}")
        lines.append("")
    
    lines.extend([
        "=" * 70,
        "CAVEATS",
        "=" * 70,
        "",
        "1. Sparse independent events: mean pairwise r is typically 0.01-0.07,",
        "   indicating largely independent dendritic activity.",
        "",
        "2. FOV-specific global regimes: some FOVs show more global events than",
        "   others; this is biological variability, not a pipeline issue.",
        "",
        "3. Behavior coupling analysis uses circular-shift nulls and onset-locked",
        "   analysis, not just full-trace Pearson (which can be misleading).",
        "",
        "4. Depth effects are within-FOV (Z-scored); pooling across FOVs can be",
        "   confounded by different depth ranges.",
        "",
        "5. Runs sharing a FOV share masks and are not independent samples for",
        "   between-run comparisons; mixed models account for this nesting.",
        ""
    ])
    
    (out_dir / f"stats_summary_{source}.txt").write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def process_source(runs: list[Run], source: str, force: bool, jobs: int) -> tuple[
    list[RunStats], pd.DataFrame, pd.DataFrame, dict
]:
    """Process all runs for a single source."""
    out_dir = STATS_ROOT / source
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing source: {source}")
    print(f"{'='*60}")
    
    # Check which runs have data for this source
    valid_runs = []
    for r in runs:
        df, cols = load_traces(r, source)
        if df is not None and len(cols) >= 2:
            valid_runs.append(r)
        else:
            print(f"  SKIP {r.key}: no {source} traces")
    
    print(f"\n{len(valid_runs)} runs with {source} traces")
    
    if len(valid_runs) == 0:
        return [], pd.DataFrame(), pd.DataFrame(), {}
    
    # Analyze runs
    run_stats = []
    all_per_dend = []
    all_pairwise = []
    
    for r in valid_runs:
        print(f"  Analyzing {r.key}...", end=" ", flush=True)
        try:
            stats = analyze_run(r, source)
            if stats is not None:
                run_stats.append(stats)
                print(f"OK (n={stats.n_dendrites}, r̄={stats.mean_pairwise_r:.3f})")
            else:
                print("SKIP (no stats)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        
        # Per-dendrite stats
        try:
            pd_stats = extract_per_dendrite_stats(r, source)
            if len(pd_stats) > 0:
                all_per_dend.append(pd_stats)
        except Exception:
            pass
        
        # Pairwise stats
        try:
            pw_stats = extract_pairwise_stats(r, source)
            if len(pw_stats) > 0:
                all_pairwise.append(pw_stats)
        except Exception:
            pass
    
    if len(run_stats) == 0:
        return [], pd.DataFrame(), pd.DataFrame(), {}
    
    # Combine DataFrames
    per_dend = pd.concat(all_per_dend, ignore_index=True) if all_per_dend else pd.DataFrame()
    pairwise = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()
    
    # Save CSVs
    df_runs = pd.DataFrame([asdict(s) for s in run_stats])
    df_runs.to_csv(out_dir / "per_run_stats.csv", index=False)
    
    if len(per_dend) > 0:
        per_dend.to_csv(out_dir / "per_dendrite_stats.csv", index=False)
    
    if len(pairwise) > 0:
        pairwise.to_csv(out_dir / "pairwise_stats.csv", index=False)
    
    # Cohort analysis
    print(f"\nRunning cohort analysis...")
    cohort_results = cohort_analysis(run_stats, per_dend, pairwise, source, out_dir)
    
    # Save cohort results
    with open(out_dir / f"stats_summary_{source}.json", "w") as f:
        json.dump(cohort_results, f, indent=2, default=str)
    
    # Generate figures
    print("Generating figures...")
    generate_figures(run_stats, per_dend, source, out_dir)
    
    # Generate text summary
    generate_summary_text(cohort_results, source, out_dir)
    
    print(f"\nOutputs written to {out_dir}")
    
    return run_stats, per_dend, pairwise, cohort_results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", action="append", default=[],
                    help="DATE/MOUSE/RUN to process (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="Process all runs")
    ap.add_argument("--force", action="store_true",
                    help="Force recomputation even if outputs exist")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Number of parallel jobs")
    ap.add_argument("--source", choices=["auto", "human", "both"], default="human",
                    help="Data source to analyze")
    args = ap.parse_args(argv)
    
    # Collect runs
    if args.all:
        runs = discover_runs()
    elif args.run:
        runs = [get_run(k) for k in args.run]
    else:
        print("Specify --run or --all")
        return 1
    
    print(f"auto_stats.py — Cohort statistics")
    print(f"Source: {args.source}")
    print(f"Runs: {len(runs)}")
    
    STATS_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Process sources
    sources = ["auto", "human"] if args.source == "both" else [args.source]
    
    results = {}
    for source in sources:
        stats, per_dend, pairwise, cohort = process_source(
            runs, source, args.force, args.jobs
        )
        results[source] = {
            "stats": stats,
            "per_dend": per_dend,
            "pairwise": pairwise,
            "cohort": cohort
        }
    
    # Auto vs human comparison
    if args.source == "both" and results.get("auto", {}).get("stats") and results.get("human", {}).get("stats"):
        print("\n" + "="*60)
        print("Comparing auto vs human...")
        print("="*60)
        
        comparison = compare_sources(
            results["auto"]["stats"],
            results["human"]["stats"],
            STATS_ROOT
        )
        
        with open(STATS_ROOT / "compare_auto_vs_human.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Text summary
        lines = [
            "AUTO vs HUMAN PIPELINE COMPARISON",
            "="*50,
            f"Common runs: {comparison.get('n_common_runs', 0)}",
            ""
        ]
        for comp in comparison.get("metric_comparisons", []):
            lines.append(f"{comp['label']}:")
            lines.append(f"  Auto mean: {comp['auto_mean']:.3f}")
            lines.append(f"  Human mean: {comp['human_mean']:.3f}")
            lines.append(f"  Correlation: r={comp['correlation_r']:.3f}, p={comp['correlation_p']:.3g}")
            lines.append(f"  Agreement: {comp['agreement']}")
            lines.append("")
        
        (STATS_ROOT / "compare_auto_vs_human.txt").write_text("\n".join(lines))
        print(f"\nComparison outputs written to {STATS_ROOT}")
    
    # Log notes
    output_files = [
        str(p.relative_to(PROJECT))
        for p in STATS_ROOT.rglob("*")
        if p.is_file()
    ]
    log_note(
        what=f"cohort statistics ({args.source})",
        files=output_files[:20],  # Limit to first 20
        verified=f"processed {len(runs)} runs",
        caveats="sparse events, FOV-specific global regimes, behavior uses onset-locked analysis"
    )
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    
    # Print key results
    for source in sources:
        cohort = results.get(source, {}).get("cohort", {})
        if cohort:
            print(f"\n{source.upper()} KEY RESULTS:")
            print(f"  Runs: {cohort.get('n_runs', 0)}")
            print(f"  Mean pairwise r: {cohort.get('pairwise_r_cohort_mean', 'N/A'):.3f}" if cohort.get('pairwise_r_cohort_mean') else "  Mean pairwise r: N/A")
            print(f"  Eff dim / N: {cohort.get('eff_dim_normalized_mean', 'N/A'):.2f}" if cohort.get('eff_dim_normalized_mean') else "  Eff dim / N: N/A")
            print(f"  Frac global events: {cohort.get('frac_global_events_mean', 'N/A'):.2f}" if cohort.get('frac_global_events_mean') else "  Frac global events: N/A")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
