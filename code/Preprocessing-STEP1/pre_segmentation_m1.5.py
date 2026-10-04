#!/usr/bin/env python3
"""
Module 1.5: Best-frame selection using ΔF/F MIP for scoring, raw 4D for 3D output

Strategy:
  - Score frames using ΔF/F MIP stack (T, Y, X) — has sharp dendrite contrast
  - Save 3D volumes from raw 4D stack (T, Z, Y, X) — needed for M2 mask detection
  - Uses event_groups.csv from M1 to know which frame ranges are events
  - Falls back to event crop files if ΔF/F stack not available

Outputs:
  bestframe_event_group_####_peak<global>_t<local>_rank##_3d.tif   (Z,Y,X)
  bestframe_event_group_####_peak<global>_t<local>_rank##_mip.png  (annotated)
"""

from __future__ import annotations

import csv
import re
import gc
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# =========================
# CONFIG
# =========================
DATE = "2026-05-12"
MOUSE = "rbp4_132_phpeb"
RUN = "run9"
FS_HZ = 5.0

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN

# ΔF/F MIP stack (T, Y, X) — used for SCORING (sharp contrast)
DFF_STACK_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin-dff.tif"
# Raw 4D stack (T, Z, Y, X) — used for SAVING 3D best frames
RAW_4D_PATH = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin.tif"
# How many seconds M1 skipped (event_groups.csv indices are relative to the
# TRIMMED stack, while this script indexes the UNTRIMMED raw 4D stack).
#
# This is now DERIVED from the stacks on disk rather than typed in, because
# duplicating M1's SKIP_FIRST_SECONDS here was a silent-failure trap: the two
# constants drifted apart and the best frames for 2026-05-12 run5 and run9 were
# sampled 10 frames (2 s) after the events M1 actually detected. The offset is
# also genuinely per-run, not per-date -- on 2026-05-12, run6 used 14 s while
# run5 and run9 used 12 s -- so no single hardcoded value can be right.
#
# Leave as None to derive automatically:
#     skip_offset = T(raw 4D stack) - T(preprocessed/stack_voxel_norm_mean_sub)
# Set to a float only to override derivation (e.g. if the preprocessed stack is
# unavailable); it is then interpreted as seconds, as before.
M1_SKIP_SECONDS = None

# Stack M1 wrote, used to derive the offset above.
NORM_STACK_PATH = BASE / "preprocessed" / "stack_voxel_norm_mean_sub.tif"
# Fallback: event crops from M1
EVENT_FOLDER = BASE / "preprocessed" / "event_crops"
EVENT_GROUPS_CSV = BASE / "preprocessed" / "event_groups.csv"

# Output
OUT_FOLDER = BASE / "preprocessed" / "best_frames"
OUT_FOLDER.mkdir(exist_ok=True, parents=True)

# ---- scoring ----
P_HI = 99.9
P_MID = 60.0
SUBTRACT_SPATIAL_BG = True
SPATIAL_BG_SIGMA = 15.0
MIP_SMOOTH_SIGMA = 0.0

# ---- temporal baseline subtraction ----
# Subtract the quietest frame (scaled) from all frames before scoring.
# This removes static background and reveals transient dendrite activity.
SUBTRACT_TEMPORAL_BG = True
TEMPORAL_BG_SCALE = 0.8  # scale factor for quiet frame (< 1.0 catches dimmer cells)

# ---- Z-MIP for raw 4D ----
TOP_Z_PLANES = 15  # None = use all Z

# ---- selection ----
TOP_K = 5
MIN_SEP = 3
MIN_SCORE_RATIO = 0.5  # skip frames scoring below this fraction of the best frame's score

# ---- PNG ----
PNG_P_LO = 1.0
PNG_P_HI = 99.7
PNG_DPI = 200


# =========================
# Helpers
# =========================
def _n_frames(path: Path):
    """Number of time points in a TIFF series, without loading pixel data."""
    try:
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
    except Exception:
        return None
    return int(shape[0]) if len(shape) >= 3 else None


def resolve_skip_offset():
    """
    Frames to add to event_groups.csv indices to reach raw-stack indices.

    M1 trims SKIP_FIRST_SECONDS from the front of the stack it reads and writes
    stack_voxel_norm_mean_sub.tif, so the number of frames it dropped is simply
    the difference in length. Deriving it removes the need to keep a copy of
    M1's constant in sync with M1.

    Raises RuntimeError rather than guessing, because a wrong offset produces
    plausible-looking best frames taken from the wrong moment in the recording.
    """
    # Explicit override wins, but say so loudly.
    if M1_SKIP_SECONDS is not None:
        off = int(M1_SKIP_SECONDS * FS_HZ)
        print(f"Frame offset: +{off} frames "
              f"(MANUAL override M1_SKIP_SECONDS={M1_SKIP_SECONDS}s at {FS_HZ}Hz)")
        return off

    # M1 reads raw_clean.tif when it exists, otherwise the original raw stack.
    raw_clean = BASE / "preprocessed" / "raw_clean.tif"
    if raw_clean.exists():
        raise RuntimeError(
            f"Cannot derive the frame offset for this run.\n"
            f"  {raw_clean} exists, so M1 indexed the motion-cleaned stack while\n"
            f"  this script indexes {RAW_4D_PATH.name}. Removed motion frames mean the\n"
            f"  two are not related by a constant offset, so event_groups.csv indices\n"
            f"  cannot be mapped onto the raw stack by adding a number.\n"
            f"  Either point RAW_4D_PATH at raw_clean.tif, or re-run M1 on the raw\n"
            f"  stack, or set M1_SKIP_SECONDS explicitly if you know the mapping holds."
        )

    if not NORM_STACK_PATH.exists():
        raise RuntimeError(
            f"Cannot derive the frame offset: {NORM_STACK_PATH} not found.\n"
            f"  Run M1 (find_events_m1.py) first, or set M1_SKIP_SECONDS explicitly."
        )

    t_raw = _n_frames(RAW_4D_PATH)
    t_norm = _n_frames(NORM_STACK_PATH)
    if t_raw is None or t_norm is None:
        raise RuntimeError(
            f"Cannot read frame counts (raw={t_raw}, norm={t_norm}).\n"
            f"  Set M1_SKIP_SECONDS explicitly to bypass derivation."
        )

    off = t_raw - t_norm
    if off < 0:
        raise RuntimeError(
            f"Derived a negative frame offset ({off}): the preprocessed stack has MORE\n"
            f"  frames ({t_norm}) than the raw stack ({t_raw}). These two files do not\n"
            f"  belong to the same run."
        )

    print(f"Frame offset: +{off} frames ({off / FS_HZ:.1f}s at {FS_HZ}Hz), "
          f"derived from {t_raw} raw - {t_norm} preprocessed frames")
    return off


def mip_z(vol_zyx):
    """Z-MIP, optionally restricted to top Z planes."""
    v = vol_zyx
    if TOP_Z_PLANES is not None and TOP_Z_PLANES > 0:
        z0 = max(v.shape[0] - TOP_Z_PLANES, 0)
        v = v[z0:]
    return np.nanmax(v, axis=0)


def score_frame(mip_2d):
    """Sparseness score on a 2D MIP. Favors sharp bright dendrites."""
    m = mip_2d.astype(np.float32, copy=False)
    if MIP_SMOOTH_SIGMA and MIP_SMOOTH_SIGMA > 0:
        m = gaussian_filter(m, sigma=float(MIP_SMOOTH_SIGMA))
    if SUBTRACT_SPATIAL_BG:
        bg = gaussian_filter(m, sigma=SPATIAL_BG_SIGMA)
        m = m - bg
        m[m < 0] = 0
    return float(np.percentile(m, P_HI) - np.percentile(m, P_MID))


def contrast_stretch_01(img, p_lo, p_hi):
    m = img.astype(np.float32, copy=False)
    finite = np.isfinite(m)
    if not finite.any():
        return np.zeros_like(m, dtype=np.float32)
    lo, hi = np.percentile(m[finite], (p_lo, p_hi))
    if hi <= lo:
        return np.zeros_like(m, dtype=np.float32)
    return np.clip((m - lo) / (hi - lo), 0, 1).astype(np.float32)


def save_mip_png(path, mip, title=""):
    m01 = contrast_stretch_01(mip, PNG_P_LO, PNG_P_HI)
    plt.figure(figsize=(5, 5))
    plt.imshow(m01, cmap="gray", vmin=0, vmax=1)
    if title:
        plt.title(title, fontsize=9, color="white",
                  bbox=dict(facecolor="black", alpha=0.6, pad=2))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=PNG_DPI)
    plt.close()


def select_topk_with_spacing(n_frames, scores, k, min_sep):
    """Select top-k scoring frames with minimum spacing."""
    cands = np.arange(n_frames)
    order = cands[np.argsort(scores)[::-1]]
    chosen = []
    for t in order:
        t = int(t)
        if all(abs(t - c) >= min_sep for c in chosen):
            chosen.append(t)
        if len(chosen) >= k:
            break
    return chosen


# =========================
# MAIN
# =========================
def main():
    skip_offset = resolve_skip_offset()

    # --- Load ΔF/F MIP (T, Y, X) for scoring ---
    if DFF_STACK_PATH.exists():
        print(f"Loading ΔF/F MIP for scoring: {DFF_STACK_PATH.name}")
        dff_mip = tifffile.imread(str(DFF_STACK_PATH)).astype(np.float32)
        # Handle if accidentally 4D
        if dff_mip.ndim == 4:
            print(f"  Got 4D {dff_mip.shape}, taking Z-MIP...")
            dff_mip = dff_mip.max(axis=1)
        print(f"  Shape: {dff_mip.shape} (T, Y, X)")
    else:
        dff_mip = None
        print(f"ΔF/F MIP not found: {DFF_STACK_PATH}")

    # --- Load raw 4D stack for 3D output ---
    if RAW_4D_PATH.exists():
        print(f"Loading raw 4D stack: {RAW_4D_PATH.name}")
        raw4d = tifffile.imread(str(RAW_4D_PATH)).astype(np.float32)
        if raw4d.ndim == 3:
            raw4d = raw4d[:, np.newaxis, :, :]
        print(f"  Shape: {raw4d.shape} (T, Z, Y, X)")
    else:
        raw4d = None
        print(f"Raw 4D not found: {RAW_4D_PATH}")

    if dff_mip is None and raw4d is None:
        print("Neither ΔF/F nor raw 4D found. Cannot proceed.")
        return

    # --- Load event groups ---
    events = []
    if EVENT_GROUPS_CSV.exists():
        with open(EVENT_GROUPS_CSV) as f:
            for row in csv.DictReader(f):
                events.append({
                    "id": int(row["event_id"]),
                    "crop_start": int(row["crop_start"]),
                    "crop_end": int(row["crop_end"]),
                })
        print(f"Loaded {len(events)} events from {EVENT_GROUPS_CSV.name}")
    else:
        # Fallback: scan event crop files
        paths = sorted(EVENT_FOLDER.glob("*.tif"))
        if not paths:
            raise FileNotFoundError("No event_groups.csv and no event crops found.")
        for i, p in enumerate(paths):
            m = re.search(r"(\d{4})", p.stem)
            eid = int(m.group(1)) if m else i
            events.append({"id": eid, "path": p})
        print(f"Fallback: {len(events)} event crop files")

    print(f"Output: {OUT_FOLDER}")
    print(f"TOP_K={TOP_K}, MIN_SEP={MIN_SEP}")
    # (the frame offset is reported by resolve_skip_offset() above)

    # --- Process each event ---
    for ev in events:
        eid = ev["id"]
        eg_id = f"event_group_{eid:04d}"

        if "crop_start" in ev and (dff_mip is not None or raw4d is not None):
            # Use global stacks with offset
            cs = ev["crop_start"] + skip_offset
            ce = ev["crop_end"] + skip_offset

            # Score from ΔF/F MIP
            if dff_mip is not None:
                ce_s = min(ce, dff_mip.shape[0])
                if cs >= dff_mip.shape[0]:
                    continue
                mip_frames = dff_mip[cs:ce_s]  # (T_crop, Y, X)
            elif raw4d is not None:
                ce_s = min(ce, raw4d.shape[0])
                if cs >= raw4d.shape[0]:
                    continue
                # Compute MIP from raw 4D for scoring
                mip_frames = np.array([mip_z(raw4d[t]) for t in range(cs, ce_s)])
            else:
                continue

            T_crop = mip_frames.shape[0]
        elif "path" in ev:
            # Fallback: load event crop
            crop_4d = tifffile.imread(str(ev["path"])).astype(np.float32)
            if crop_4d.ndim == 3:
                crop_4d = crop_4d[:, np.newaxis, :, :]
            mip_frames = np.array([mip_z(crop_4d[t]) for t in range(crop_4d.shape[0])])
            T_crop = crop_4d.shape[0]
            cs = 0
        else:
            continue

        if T_crop < 1:
            continue

        # Temporal baseline subtraction: find quietest frame, subtract from all
        if SUBTRACT_TEMPORAL_BG and T_crop > 2:
            # Quick pre-score to find the quietest frame
            pre_scores = np.array([score_frame(mip_frames[t]) for t in range(T_crop)])
            quiet_idx = int(np.argmin(pre_scores))
            quiet_frame = mip_frames[quiet_idx].copy()
            # Subtract scaled quiet frame from all MIPs
            mip_frames_sub = mip_frames - TEMPORAL_BG_SCALE * quiet_frame[np.newaxis]
            mip_frames_sub[mip_frames_sub < 0] = 0
            # Score on subtracted frames
            scores = np.array([score_frame(mip_frames_sub[t]) for t in range(T_crop)], dtype=np.float32)
        else:
            mip_frames_sub = mip_frames
            scores = np.array([score_frame(mip_frames[t]) for t in range(T_crop)], dtype=np.float32)

        # Select top-K
        chosen = select_topk_with_spacing(T_crop, scores, TOP_K, MIN_SEP)
        chosen = sorted(chosen, key=lambda t: float(scores[t]), reverse=True)

        # Filter out noisy frames (score too low relative to best)
        if chosen:
            best_score = scores[chosen[0]]
            if best_score > 0:
                before = len(chosen)
                chosen = [t for t in chosen if scores[t] >= MIN_SCORE_RATIO * best_score]
                if len(chosen) < before:
                    print(f"    Filtered {before - len(chosen)} noisy frames "
                          f"(threshold={MIN_SCORE_RATIO*best_score:.2f}, best={best_score:.2f})")

        peak_local = int(np.argmax(scores))
        global_peak = cs + peak_local

        for rank, t_local in enumerate(chosen, start=1):
            global_frame = cs + t_local
            title = f"{eg_id} | frame {global_frame} ({global_frame/FS_HZ:.1f}s)"

            # Get 3D volume from raw 4D
            if raw4d is not None and global_frame < raw4d.shape[0]:
                vol = raw4d[global_frame]  # (Z, Y, X)
            elif "path" in ev:
                vol = crop_4d[t_local]
            else:
                vol = mip_frames[t_local][np.newaxis]  # (1, Y, X) fallback

            mip = mip_z(vol) if vol.ndim == 3 and vol.shape[0] > 1 else vol.squeeze()

            # For PNG, show the subtracted MIP if available (reveals dendrites better)
            if SUBTRACT_TEMPORAL_BG and T_crop > 2:
                mip_display = mip_frames_sub[t_local]
            else:
                mip_display = mip

            out3d = OUT_FOLDER / f"bestframe_{eg_id}_peak{global_peak}_t{t_local:05d}_rank{rank:02d}_3d.tif"
            outpng = OUT_FOLDER / f"bestframe_{eg_id}_peak{global_peak}_t{t_local:05d}_rank{rank:02d}_mip.png"

            tifffile.imwrite(out3d, vol.astype(np.float32), photometric="minisblack")
            save_mip_png(outpng, mip_display, title=title)

        print(f"  {eg_id}: T={T_crop}, peak=frame {global_peak}, saved {len(chosen)}")

    del dff_mip, raw4d
    gc.collect()
    print("Done ✅")


if __name__ == "__main__":
    main()
