#!/usr/bin/env python
"""make_movie.py - reference + dynamic volume movie with global Ca, pupil and accel (code/Auto/, v2).

Layout, 1920x1080, black background (style of run8-pupil-acc-mip-dual-view.mp4):
  top-left   REFERENCE volume: peak activity of every detected dendrite (static), MIP over Z
  top-right  DYNAMIC volume: the live dF/F of the same dendrites, frame by frame, MIP over Z
             (green; 100 um scale bar; time stamp). Both use the true um aspect (X 1.2, Y 1.0 um/px).
  bottom     global Ca (green), pupil (blue), accelerometer (purple, y-limit 0.25),
             one shared time axis, moving red cursor.

dF/F for display: per-voxel F0 = 10th percentile (after skip_s) of the lightly smoothed stack,
spatial high-pass (minus a Gaussian of sigma (1.5,8,8) vox, removes the field-wide glow so
individual dendrites stand out), 3-frame running mean, shown only inside the dendrite masks
(dilated by 1 voxel). Masks: OUT/masks/auto_labelmap_reviewed.tif if present, else
auto_labelmap.tif; --source human uses the human curated masks.

The trace panels are drawn once with matplotlib; each frame only pastes the two images, the
cursor and the clock, so a 530-frame run renders in ~2 min. Frames are piped into ffmpeg
(H.264, yuv420p, crf 18).

CLI: make_movie.py --run KEY | --all [--force] [--jobs N] [--speed 2] [--source auto|human]
Output: OUT/movies/<run>_dual_behavior.mp4 (auto) or <run>_dual_behavior_human.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, open_stack, load_behavior, human_labelmap, merge_metrics, VOXEL_ZYX,
)

FFMPEG = "/opt/homebrew/bin/ffmpeg"
W, H = 1920, 1080
PANEL_W = 930
GREEN = np.array([60, 255, 60], np.float32)   # RGB


def masks_for(r: Run, source: str, shape):
    if source == "human":
        lab = human_labelmap(r, shape)
        return lab, "human"
    for name in ("auto_labelmap_reviewed.tif", "auto_labelmap.tif"):
        p = r.out / "masks" / name
        if p.exists():
            return tifffile.imread(str(p)), name
    raise FileNotFoundError(f"{r.out / 'masks'}: no auto labelmap (run auto_detect first)")


def to_rgb(img01, gamma=0.7):
    v = np.clip(img01, 0, 1) ** gamma
    return (v[..., None] * GREEN[None, None, :]).astype(np.uint8)


def trace_canvas(r: Run, t_end: float, ph: int, global_t, global_y, beh):
    """Draw the bottom trace panel once. Returns (RGB uint8 array, x pixel of t=0, px per s)."""
    rows = [("Global Ca  dF/F (%)", global_t, global_y, "#4ddc4d", None)]
    if beh.get("pupil") is not None:
        p = beh["pupil"]
        lo, hi = np.nanpercentile(p, [1, 99.5])
        rows.append(("Pupil (norm.)", beh["pupil_t"], (p - lo) / (hi - lo + 1e-9), "#5b8cff", (-0.05, 1.1)))
    if beh.get("accel") is not None:
        rows.append(("Accel", beh["accel_t"], beh["accel"], "#b48cff", (0, 0.25)))
    fig = plt.figure(figsize=(W / 100, ph / 100), dpi=100, facecolor="black")
    n = len(rows)
    left, right, bot, top = 0.06, 0.985, 0.11, 0.97
    hgt = (top - bot) / n
    axes = []
    for i, (name, t, y, col, yl) in enumerate(rows):
        ax = fig.add_axes([left, top - (i + 1) * hgt + 0.012, right - left, hgt - 0.024], facecolor="black")
        k = (t >= 0) & (t <= t_end)
        ax.plot(t[k], y[k], color=col, lw=0.9)
        ax.set_xlim(0, t_end)
        if yl:
            ax.set_ylim(*yl)
        ax.set_ylabel(name, color=col, fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#888888")
        ax.grid(color="#333333", lw=0.5)
        if i < n - 1:
            ax.set_xticklabels([])
        axes.append(ax)
    axes[-1].set_xlabel("Time (s)", color="white", fontsize=10)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    x0 = left * W
    pps = (right - left) * W / t_end
    return img, x0, pps


def label_panel(canvas, text, x, y):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


def make_movie(key: str, speed=2.0, force=False, source="auto", verbose=True) -> dict:
    r = get_run(key)
    suffix = "" if source == "auto" else "_human"
    out = r.outdir("movies") / f"{r.run}_dual_behavior{suffix}.mp4"
    s = open_stack(r)
    T0, Z, Y, X = s.shape
    lab, lab_name = masks_for(r, source, (Z, Y, X))
    if lab.max() == 0:
        raise RuntimeError("no dendrites to show")
    src_mtime = (r.out / "masks" / lab_name).stat().st_mtime if source == "auto" else 0
    if (not force and out.exists() and out.stat().st_mtime > max(src_mtime, Path(__file__).stat().st_mtime)):
        if verbose:
            print(f"[{key}] movie up to date")
        return {"run": key, "status": "skipped"}
    t0 = time.time()
    sk = int(round(r.skip_s * r.frame_rate))
    T = T0 - sk
    fr = r.frame_rate
    show = ndi.binary_dilation(lab > 0, iterations=1)

    # F0 from a frame sample (10th percentile, project convention)
    samp = np.linspace(sk, T0 - 1, min(160, T), dtype=int)
    f0 = np.percentile(np.stack([ndi.gaussian_filter(np.asarray(s[i], np.float32), (0.5, 1, 1)) for i in samp]), 10, axis=0)
    f0 = ndi.gaussian_filter(f0, (0.5, 2, 2)) + 1.0

    def dff_frame(i):
        f = ndi.gaussian_filter(np.asarray(s[sk + i], np.float32), (0.5, 1, 1))
        d = (f - f0) / f0
        d -= ndi.gaussian_filter(d, (1.5, 8, 8))
        return np.where(show, d, 0).max(0)

    # pass 1: all MIPs (T x Y x X float32, ~170 MB)
    mips = np.empty((T, Y, X), np.float32)
    for i in range(T):
        mips[i] = dff_frame(i)
    mips = ndi.uniform_filter1d(mips, 3, axis=0)
    vals = mips[:, show.any(0)]
    lo = np.percentile(vals, 75)
    hi = np.percentile(vals, 99.7)
    ref = mips.max(0)
    rlo, rhi = np.percentile(ref[show.any(0)], [5, 99.5])

    # geometry
    ph_img = int(round(PANEL_W * (Y * VOXEL_ZYX[1]) / (X * VOXEL_ZYX[2])))
    top_y = 46
    xL, xR = 20, W - 20 - PANEL_W
    trace_y = top_y + ph_img + 30
    ph = H - trace_y

    # traces
    beh = load_behavior(r)
    gpath = r.out / "traces" / "global_ca.csv"
    if gpath.exists():
        g = pd.read_csv(gpath)
        gt, gy = g["time_s"].to_numpy(), g["global_dff"].to_numpy()
    else:   # fall back: mean raw over the field
        gm = np.array([np.asarray(s[sk + i], np.float32).mean() for i in range(T)])
        f0g = np.percentile(gm, 10)
        gt, gy = np.arange(T) / fr, (gm - f0g) / f0g * 100
    t_end = T / fr
    tr_img, x0, pps = trace_canvas(r, t_end, ph, gt, gy, beh)

    base = np.zeros((H, W, 3), np.uint8)
    base[trace_y:trace_y + tr_img.shape[0], :tr_img.shape[1]] = tr_img[:H - trace_y]
    ref_rgb = cv2.resize(to_rgb((ref - rlo) / (rhi - rlo + 1e-9)), (PANEL_W, ph_img), interpolation=cv2.INTER_LINEAR)
    base[top_y:top_y + ph_img, xL:xL + PANEL_W] = ref_rgb
    label_panel(base, f"Reference: peak activity, {lab.max()} dendrites ({source})", xL, 32)
    label_panel(base, "Dynamic: dF/F", xR, 32)
    title = f"{r.key}"
    cv2.putText(base, title, (W - 20 - 9 * len(title), H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    # 100 um scale bar on the dynamic panel
    px_per_um = PANEL_W / (X * VOXEL_ZYX[2])
    sb = int(round(100 * px_per_um))
    sbx, sby = xR + PANEL_W - sb - 30, top_y + ph_img - 22
    cv2.line(base, (sbx, sby), (sbx + sb, sby), (255, 255, 255), 3)
    cv2.putText(base, "100 um", (sbx + sb // 2 - 34, sby - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    fps = max(1, int(round(fr * speed)))
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-preset", "medium", str(out) + ".part.mp4"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(T):
            frame = base.copy()
            dyn = cv2.resize(to_rgb((mips[i] - lo) / (hi - lo + 1e-9)), (PANEL_W, ph_img), interpolation=cv2.INTER_LINEAR)
            sub = frame[top_y:top_y + ph_img, xR:xR + PANEL_W]
            np.maximum(sub, dyn, out=sub)
            t = i / fr
            x = int(round(x0 + t * pps))
            frame[trace_y + 4:H - int(0.11 * ph) + 4, max(0, x - 1):x + 2] = (230, 30, 30)
            cv2.putText(frame, f"{t:6.1f} s", (xR + PANEL_W - 130, top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except Exception:
        proc.kill()
        raise
    Path(str(out) + ".part.mp4").replace(out)
    el = time.time() - t0
    merge_metrics(r, f"movie_{source}", {"path": out.name, "frames": T, "fps": fps, "speed": speed,
                                         "n_dendrites": int(lab.max()), "masks": lab_name, "time_s": round(el, 1)})
    if verbose:
        print(f"[{key}] {out.name}: {T} frames @ {fps} fps in {el:.0f}s")
    return {"run": key, "status": "done", "time_s": el}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed vs real time (default 2x)")
    ap.add_argument("--source", choices=("auto", "human"), default="auto")
    a = ap.parse_args()
    keys = [r.key for r in discover_runs()] if a.all else a.run
    if not keys:
        ap.error("give --run or --all")
    res = []
    with ProcessPoolExecutor(max(1, min(a.jobs, 3))) as ex:
        futs = {ex.submit(make_movie, k, a.speed, a.force, a.source): k for k in keys}
        for f in as_completed(futs):
            try:
                res.append(f.result())
            except Exception as e:
                import traceback
                traceback.print_exc()
                res.append({"run": futs[f], "status": "error", "error": repr(e)})
    bad = [x for x in res if x["status"] == "error"]
    for b in bad:
        print("ERROR", b["run"], b["error"])
    print(f"done={sum(x['status'] == 'done' for x in res)} skipped={sum(x['status'] == 'skipped' for x in res)} errors={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
