#!/usr/bin/env python
"""auto_detect.py - automatic dendrite detection for SCAPE runs (code/Auto/, v2).

Idea: a dendrite is a set of voxels that light up *together*. Shape alone does not
separate them (the field is densely packed with tilted, crossing apical trunks) and
plain neighbour correlation does not either (spatial blur makes every neighbour pair
correlated). So detection is activity-first, in four steps:

1. Normalised activity volume z(t,z,y,x), streamed once from the raw memmap:
   F0 = per-voxel 10th percentile after skip_s (project convention), dF/F of the
   lightly smoothed stack, spatial high-pass (minus a Gaussian of sigma (1.5,8,8) vox)
   which removes the field-wide/neuropil signal, temporal smoothing (sigma 1 frame),
   then a robust per-voxel z-score (median / MAD).
2. Activity map = mean of the 3 largest z values per voxel; foreground = map > FG_Z.
3. Greedy seeded region growing (CNMF/suite2p-like): take the most active unassigned
   voxel, build a reference trace from its 3x3x3 neighbourhood, grow through 26-connected
   foreground voxels whose trace correlates with the *reference* (not with the neighbour)
   above GROW_R; the reference is refreshed from the region as it grows. Repeat.
4. Clean-up: merge touching units whose traces correlate > MERGE_R (fragments of one
   dendrite), keep the largest connected piece, fill small holes, and keep units that
   are big enough, span enough depth (Y) and have a clear peak.

FOV mode (default): runs that share a field of view (MASK_SOURCE in code/Workflow/mavca_status.py,
the table the human pipeline uses) are each detected on their own (auto_labelmap_perrun.tif) and
then all get the union of those detections as auto_labelmap.tif, so dendrite identities are shared
within a FOV and a dendrite silent in one run is still measured there. --per-run disables this.

Outputs (OUT = scape-auto/<DATE>/<MOUSE>/<RUN>):
    OUT/masks/auto_labelmap.tif   uint16 (Z,Y,X) at the raw stack shape, ids 1..N
    OUT/masks/auto_masks.csv      id,name,n_vox,cz,cy,cx,length_um,verticality,score,...
    OUT/reference/ref_mean.tif    mean raw image (float32, Z,Y,X)
    OUT/reference/ref_activity.tif  activity map (top-3 mean z, float32, Z,Y,X)

CLI: auto_detect.py --run DATE/MOUSE/RUN [--run ...] | --all [--force] [--jobs N]
     [--cache DIR]  reuse/keep the z volume as DIR/<key>/z.npy (tuning only; ~2.5 GB/run)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, open_stack, merge_metrics, VOXEL_ZYX,
)

__version__ = "2.1.0"

# Parameters. Tuned on 2026-05-08/rbp4_139_phpeb/run5, 2026-05-12/rbp4_132_phpeb/run5,
# 2026-03-31/rbp4_132_phpeb/run7 only (TRAIN_RUNS); every other run is held out.
TRAIN_RUNS = ("2026-05-08/rbp4_139_phpeb/run5", "2026-05-12/rbp4_132_phpeb/run5",
              "2026-03-31/rbp4_132_phpeb/run7")
P = dict(
    SMOOTH_SP=(0.5, 1.0, 1.0),   # light spatial smoothing before dF/F (vox, Z,Y,X)
    HP_SIGMA=(1.5, 8.0, 8.0),    # spatial high-pass: subtract this Gaussian of dF/F
    T_SIGMA=1.0,                 # temporal smoothing (frames)
    SEG_SMOOTH=(0.7, 2.5, 1.2),  # smoothing of z before segmentation (longer along Y = along dendrites)
    TOPK=3,                      # activity map = mean of top-k z per voxel
    FG_Z=4.5,                    # foreground threshold on activity map (upper bound; see ADAPTIVE)
    SEED_Z=7.0,                  # stop seeding below this activity (upper bound)
    # Adaptive thresholds (added 2026-10-04 after the low-SNR dual-channel Ai162 stacks were
    # tuned): the activity map has a noise floor ~3 set by the frame count, and on dim stacks
    # the true dendrites barely clear it (2026-02-09 run1: only 15% of human-mask voxels >
    # 4.5, 2.5% > 7 -> 18 units for 34 masks). Thresholds are therefore set from the map's
    # own quantiles: FG = floor + 0.44 (p90 - floor), SEED = floor + 0.30 (p99 - floor),
    # floor = median, each clipped to [lower, fixed value]. On the brightest 30-plane stacks the
    # rule returns the fixed values (4.5 / 7.0); 9 of the 20 viral runs do, the others get
    # slightly lower thresholds (median FG 4.36); dim stacks get much lower ones (FG ~3.2-3.7). GROW_R drops to 0.40 when FG falls below 4.0 (dim stack).
    ADAPTIVE=True,
    FG_Z_MIN=3.0, SEED_Z_MIN=4.0, FG_COEF=0.44, SEED_COEF=0.30, GROW_R_DIM=0.40,
    GROW_R=0.45,                 # min corr with the unit's reference trace to join
    MERGE_R=0.85,                # merge nearby units whose traces correlate above this
    MERGE_GAP=2,                 # 'nearby' = within this many voxels
    DILATE=0,                    # (tested 1-2: no recall gain, higher pair r) grow each final unit by this many voxels into unclaimed space
    MIN_VOX=400,                 # min unit size (voxels; 1 vox = 4.68 um^3)
    MIN_YSPAN_UM=20.0,           # min depth extent
    MIN_VERTICALITY=0.0,         # |cos| main axis vs Y. 0 = off: hard shape cuts removed 15-20% of human-matched units in tuning
    MIN_ELONGATION=0.0,          # sqrt ratio of the two largest PCA variances (tube-like)
    MAX_SHEET_RATIO=1.5,         # reject horizontal sheets: X extent > ratio * Y extent ...
    MIN_SHEET_X_UM=60.0,         # ... and X extent > this (the bright superficial band, not a dendrite)
    MIN_PEAK_SNR=0.0,            # unit trace (robust z of its own mean trace) must peak above this
)


def log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


# --------------------------------------------------------------------------- step 1
BIG_BYTES = 5e9     # above this (T*Z*Y*X*4) the activity volume is built in Z-blocks on a disk memmap
SCRATCH = Path("/tmp/scape_auto_scratch")


def _scratch(name, shape, dtype):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(SCRATCH / name, mode="w+", dtype=dtype, shape=shape)


def activity_volume(r: Run, verbose=True):
    """Return z (T,Z,Y,X float16; a disk memmap for big stacks), mean raw image (Z,Y,X float32)."""
    t0 = time.time()
    s = open_stack(r)
    sk = int(round(r.skip_s * r.frame_rate))
    T0, Z, Y, X = s.shape
    T = T0 - sk
    big = T * Z * Y * X * 4 > BIG_BYTES
    if not big:
        a = np.asarray(s[sk:], dtype=np.float32)
        mean = a.mean(0)
        f0 = np.empty((Z, Y, X), np.float32)
        for z in range(Z):
            f0[z] = np.percentile(a[:, z], 10, axis=0)
        f0 = ndi.gaussian_filter(f0, (0.5, 2, 2)) + 1.0
        for t in range(T):
            d = (ndi.gaussian_filter(a[t], P["SMOOTH_SP"]) - f0) / f0
            a[t] = d - ndi.gaussian_filter(d, P["HP_SIGMA"])
        ndi.gaussian_filter1d(a, P["T_SIGMA"], axis=0, output=a)
        med = np.median(a[::2], axis=0)
        mad = np.median(np.abs(a[::2] - med), axis=0) * 1.4826 + 1e-4
        z = np.empty((T, Z, Y, X), np.float16)
        for t in range(T):
            z[t] = (a[t] - med) / mad
        del a
        log(f"[{r.key}] activity volume {z.shape} in {time.time() - t0:.0f}s", verbose)
        return z, mean
    # ---- big stack: Z-blocks with margins, result on a disk memmap
    log(f"[{r.key}] big stack {(T, Z, Y, X)}: Z-block processing to {SCRATCH}", verbose)
    mean = np.zeros((Z, Y, X), np.float32)
    f0 = np.empty((Z, Y, X), np.float32)
    for z0 in range(0, Z, 4):
        a = np.asarray(s[sk:, z0:z0 + 4], np.float32)
        mean[z0:z0 + 4] = a.mean(0)
        f0[z0:z0 + 4] = np.percentile(a, 10, axis=0)
        del a
    f0 = ndi.gaussian_filter(f0, (0.5, 2, 2)) + 1.0
    z = _scratch(f"{r.key.replace('/', '_')}_z.npy", (T, Z, Y, X), np.float16)
    margin = 5                                    # covers HP_SIGMA z=1.5 (3 sigma) + SMOOTH_SP
    blk = max(2, int(BIG_BYTES // (T * Y * X * 4 * 1.5)) - 2 * margin)
    for z0 in range(0, Z, blk):
        z1 = min(Z, z0 + blk)
        lo, hi = max(0, z0 - margin), min(Z, z1 + margin)
        a = np.asarray(s[sk:, lo:hi], np.float32)
        f0b = f0[lo:hi]
        for t in range(T):
            d = (ndi.gaussian_filter(a[t], P["SMOOTH_SP"]) - f0b) / f0b
            a[t] = d - ndi.gaussian_filter(d, P["HP_SIGMA"])
        ndi.gaussian_filter1d(a, P["T_SIGMA"], axis=0, output=a)
        core = slice(z0 - lo, z1 - lo)
        med = np.median(a[::2, core], axis=0)
        mad = np.median(np.abs(a[::2, core] - med), axis=0) * 1.4826 + 1e-4
        for t in range(T):
            z[t, z0:z1] = (a[t, core] - med) / mad
        del a
        log(f"[{r.key}]   planes {z0}-{z1} done ({time.time() - t0:.0f}s)", verbose)
    z.flush()
    log(f"[{r.key}] activity volume {z.shape} (memmap) in {time.time() - t0:.0f}s", verbose)
    return z, mean


# --------------------------------------------------------------------------- step 2-4
def segment(z: np.ndarray, verbose=True, key="", params=None):
    p = dict(P, **(params or {}))
    T, Z, Y, X = z.shape
    V = Z * Y * X
    t0 = time.time()
    big = isinstance(z, np.memmap)
    zs = _scratch(f"{key.replace('/', '_')}_zs.npy", z.shape, np.float16) if big else np.empty_like(z)
    for t in range(T):
        zs[t] = ndi.gaussian_filter(np.asarray(z[t], np.float32), p["SEG_SMOOTH"])
    flat = zs.reshape(T, V)
    # re-normalise: smoothing shrinks the noise, so express it in robust z again
    for i in range(0, V, 400_000):
        blk = flat[:, i:i + 400_000].astype(np.float32)
        med = np.median(blk[::2], axis=0)
        mad = np.median(np.abs(blk[::2] - med), axis=0) * 1.4826 + 1e-4
        flat[:, i:i + 400_000] = (blk - med) / mad
    # top-k mean, chunked over voxels to bound memory
    top = np.empty(V, np.float32)
    for i in range(0, V, 400_000):
        blk = flat[:, i:i + 400_000].astype(np.float32)
        top[i:i + 400_000] = np.partition(blk, T - p["TOPK"], axis=0)[-p["TOPK"]:].mean(0)
    fg_z, seed_z, grow_r = p["FG_Z"], p["SEED_Z"], p["GROW_R"]
    if p["ADAPTIVE"]:
        floor, p90, p99 = np.percentile(top[::7], [50, 90, 99])
        fg_z = float(np.clip(floor + p["FG_COEF"] * (p90 - floor), p["FG_Z_MIN"], p["FG_Z"]))
        seed_z = float(np.clip(floor + p["SEED_COEF"] * (p99 - floor), p["SEED_Z_MIN"], p["SEED_Z"]))
        if fg_z < 4.0:
            grow_r = p["GROW_R_DIM"]
        log(f"[{key}] adaptive thresholds: floor={floor:.2f} p90={p90:.2f} p99={p99:.2f} -> FG={fg_z:.2f} SEED={seed_z:.2f} GROW_R={grow_r}", verbose)
    fg = top > fg_z
    idx = np.flatnonzero(fg)
    n = idx.size
    if big:   # gather foreground columns chunk-wise (fancy indexing on a memmap is slow)
        tr = np.empty((T, n), np.float32)
        for i in range(0, V, 400_000):
            sel = (idx >= i) & (idx < i + 400_000)
            if sel.any():
                tr[:, sel] = flat[:, i:i + 400_000][:, idx[sel] - i]
        tr = np.clip(tr, -2, None)
    else:
        tr = np.clip(flat[:, idx].astype(np.float32), -2, None)
    tr -= tr.mean(0)
    tr /= np.linalg.norm(tr, axis=0) + 1e-6
    pos = np.full(V, -1, np.int64)
    pos[idx] = np.arange(n)
    coords = np.array(np.unravel_index(idx, (Z, Y, X))).T
    dirs = np.array([d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)])
    nbr = np.full((n, 26), -1, np.int64)
    for k, d in enumerate(dirs):
        c2 = coords + d
        ok = np.all((c2 >= 0) & (c2 < [Z, Y, X]), 1)
        nbr[ok, k] = pos[np.ravel_multi_index(c2[ok].T, (Z, Y, X))]
    topv = top[idx]
    assigned = np.zeros(n, np.int32)
    units = []
    for s in np.argsort(-topv):
        if topv[s] < seed_z:
            break
        if assigned[s]:
            continue
        nb = nbr[s][nbr[s] >= 0]
        nb = nb[assigned[nb] == 0]
        ref = tr[:, np.r_[s, nb]].mean(1)
        ref /= np.linalg.norm(ref) + 1e-9
        region = [s]
        inreg = np.zeros(n, bool)
        inreg[s] = True
        frontier = np.array([s])
        it = 0
        while frontier.size:
            cand = nbr[frontier].ravel()
            cand = np.unique(cand[cand >= 0])
            cand = cand[(~inreg[cand]) & (assigned[cand] == 0)]
            if not cand.size:
                break
            new = cand[(ref @ tr[:, cand]) > grow_r]
            inreg[new] = True
            region.extend(new.tolist())
            frontier = new
            it += 1
            if it % 5 == 0:
                ref = tr[:, region].mean(1)
                ref /= np.linalg.norm(ref) + 1e-9
        units.append(np.asarray(region))
        assigned[units[-1]] = len(units)
    log(f"[{key}] fg={n} vox, {len(units)} raw units in {time.time() - t0:.0f}s", verbose)

    # ---- merge touching units with highly correlated traces
    lab = np.zeros(V, np.int32)
    for k, u in enumerate(units, 1):
        lab[idx[u]] = k
    lab = lab.reshape(Z, Y, X)
    K = len(units)
    if K == 0:
        return np.zeros((Z, Y, X), np.uint16), top.reshape(Z, Y, X), []
    U = np.stack([tr[:, u].mean(1) for u in units], 1)
    U -= U.mean(0)
    U /= np.linalg.norm(U, axis=0) + 1e-9
    C = U.T @ U
    # adjacency: unit labels that touch after a 1-voxel dilation
    adj = set()
    g = p["MERGE_GAP"]
    grown = ndi.grey_dilation(lab, size=(1 + 2 * min(g, 1), 1 + 2 * g, 1 + 2 * g))
    m = (lab > 0) & (grown > 0) & (grown != lab)
    for i, j in set(zip(lab[m].tolist(), grown[m].tolist())):
        adj.add((min(i, j), max(i, j)))
    grown2 = -ndi.grey_dilation(-np.where(lab > 0, lab, K + 1), size=(1 + 2 * min(g, 1), 1 + 2 * g, 1 + 2 * g))
    m = (lab > 0) & (grown2 <= K) & (grown2 != lab)
    for i, j in set(zip(lab[m].tolist(), grown2[m].tolist())):
        adj.add((min(i, j), max(i, j)))
    parent = list(range(K + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in sorted(adj, key=lambda e: -C[e[0] - 1, e[1] - 1]):
        if C[i - 1, j - 1] > p["MERGE_R"]:
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a
    roots = np.array([find(i) for i in range(K + 1)])
    lab = roots[lab]

    # ---- per-unit clean-up and filters
    out = np.zeros((Z, Y, X), np.uint16)
    info = []
    zflat = zs.reshape(T, V)
    objs = ndi.find_objects(lab)
    nid = 0
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = lab[sl] == k
        cc, ncc = ndi.label(m, np.ones((3, 3, 3), bool))
        if ncc > 1:
            m = cc == (np.bincount(cc.ravel())[1:].argmax() + 1)
        m = ndi.binary_fill_holes(m)
        nvox = int(m.sum())
        if nvox < p["MIN_VOX"]:
            continue
        zz, yy, xx = np.nonzero(m)
        zz, yy, xx = zz + sl[0].start, yy + sl[1].start, xx + sl[2].start
        yspan = (yy.max() - yy.min() + 1) * VOXEL_ZYX[1]
        if yspan < p["MIN_YSPAN_UM"]:
            continue
        xspan = (xx.max() - xx.min() + 1) * VOXEL_ZYX[2]
        if xspan > p["MIN_SHEET_X_UM"] and xspan > p["MAX_SHEET_RATIO"] * yspan:
            continue
        vid = np.ravel_multi_index((zz, yy, xx), (Z, Y, X))
        if big:   # memmap: gather the (sorted) columns in one read per contiguous range
            vs = np.sort(vid)
            trace = np.zeros(T, np.float32)
            cuts = np.flatnonzero(np.diff(vs) > 1) + 1
            for seg in np.split(vs, cuts):
                trace += zflat[:, seg[0]:seg[-1] + 1].astype(np.float32).sum(1)
            trace /= vs.size
        else:
            trace = zflat[:, vid].astype(np.float32).mean(1)
        med = np.median(trace)
        trz = (trace - med) / (1.4826 * np.median(np.abs(trace - med)) + 1e-6)
        peak = float(trz.max())
        if peak < p["MIN_PEAK_SNR"]:
            continue
        pts = np.c_[zz * VOXEL_ZYX[0], yy * VOXEL_ZYX[1], xx * VOXEL_ZYX[2]]
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        main = evec[:, -1]
        length = float(np.ptp(pts @ main))
        elong = float(np.sqrt(ev[-1] / max(ev[-2], 1e-6)))
        if abs(main[1]) < p["MIN_VERTICALITY"] or elong < p["MIN_ELONGATION"]:
            continue
        nid += 1
        sub = out[sl]
        sub[m & (sub == 0)] = nid
        info.append(dict(
            id=nid, name=f"dend_{nid - 1:03d}", n_vox=nvox,
            cz=float(zz.mean()), cy=float(yy.mean()), cx=float(xx.mean()),
            length_um=round(length, 1),
            verticality=round(float(abs(main[1])), 3),          # |cos| of main axis vs Y
            elongation=round(elong, 2),
            yspan_um=round(float(yspan), 1),
            peak_snr=round(peak, 2),
            n_events=int(((trz[1:-1] > 5) & (trz[1:-1] >= trz[:-2]) & (trz[1:-1] >= trz[2:])).sum()),
            score=round(peak * np.log10(nvox), 2),
            dendrite_like=bool(abs(main[1]) >= 0.5 and elong >= 1.8),
        ))
    for _ in range(p["DILATE"]):
        grown = ndi.grey_dilation(out, size=(3, 3, 3))
        add = (out == 0) & (grown > 0)
        out[add] = grown[add]
    for d in info:
        d["n_vox"] = int((out == d["id"]).sum())
    log(f"[{key}] {nid} units after merge/filters ({time.time() - t0:.0f}s)", verbose)
    for d in info:
        d.update(fg_z=round(fg_z, 2), seed_z=round(seed_z, 2), grow_r=grow_r)
    return out, top.reshape(Z, Y, X), info


# --------------------------------------------------------------------------- driver
def outputs(r: Run):
    return r.out / "masks" / "auto_labelmap.tif", r.out / "masks" / "auto_masks.csv"


def perrun_outputs(r: Run):
    return r.out / "masks" / "auto_labelmap_perrun.tif", r.out / "masks" / "auto_masks_perrun.csv"


def params_hash() -> str:
    import hashlib
    return hashlib.sha1((json.dumps(P, sort_keys=True, default=str) + __version__).encode()).hexdigest()[:12]


def json_get(r: Run, section: str) -> dict:
    p = r.out / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get(section, {}) or {}


def fov_members(r: Run) -> list[Run]:
    """All runs that share r's field of view (MASK_SOURCE in mavca_status.py), source first."""
    from Auto.scape_common import fov_group
    g = fov_group(r)
    runs = [x for x in discover_runs() if x.date == r.date and x.mouse == r.mouse and fov_group(x) == g]
    src = g.split("/")[-1]
    runs.sort(key=lambda x: (x.run != src, x.run))
    return runs


def _atomic_tif(path: Path, arr):
    tmp = path.with_suffix(".tmp.tif")
    tifffile.imwrite(str(tmp), arr, compression="zlib")
    tmp.replace(path)


def detect_own(r: Run, force=False, cache=None, verbose=True) -> bool:
    """Per-run detection -> auto_labelmap_perrun.tif. Returns True if (re)computed."""
    lab_p, csv_p = perrun_outputs(r)
    old_lab, old_csv = outputs(r)
    d = json_get(r, "detect")
    # migrate results of earlier versions that wrote the own detection to auto_labelmap.tif
    if not lab_p.exists() and old_lab.exists() and not d.get("borrowed_from") and not d.get("fov_members"):
        import shutil
        r.outdir("masks")
        shutil.copy2(old_lab, lab_p)
        shutil.copy2(old_csv, csv_p)
    if (not force and lab_p.exists() and csv_p.exists() and lab_p.stat().st_mtime > r.raw.stat().st_mtime
            and d.get("params_hash", params_hash()) == params_hash()):
        return False
    t0 = time.time()
    zc = Path(cache) / r.key.replace("/", "_") if cache else None
    if zc and (zc / "z.npy").exists() and (zc / "mean.npy").exists():
        z = np.load(zc / "z.npy")
        mean = np.load(zc / "mean.npy")
    else:
        z, mean = activity_volume(r, verbose)
        if zc:
            zc.mkdir(parents=True, exist_ok=True)
            np.save(zc / "z.npy", z)
            np.save(zc / "mean.npy", mean)
    try:
        lab, top, info = segment(z, verbose, r.key)
    finally:
        del z
        for f in SCRATCH.glob(f"{r.key.replace('/', '_')}_*.npy"):
            f.unlink(missing_ok=True)
    r.outdir("masks")
    _atomic_tif(lab_p, lab)
    pd.DataFrame(info).to_csv(csv_p, index=False)
    ref = r.outdir("reference")
    _atomic_tif(ref / "ref_mean.tif", mean.astype(np.float32))
    _atomic_tif(ref / "ref_activity.tif", top.astype(np.float32))
    merge_metrics(r, "detect_perrun", {"n_auto": len(info), "time_s": round(time.time() - t0, 1),
                                       "params_hash": params_hash()})
    log(f"[{r.key}] own detection: {len(info)} dendrites in {time.time() - t0:.0f}s", verbose)
    return True


def fov_union(members: list[Run], overlap_max=0.2):
    """Union of the per-run detections of all runs in a FOV. The source run's units come first;
    a unit from another run is added when < overlap_max of its voxels are already claimed,
    and only on unclaimed voxels. Returns (labelmap, info rows)."""
    lab = None
    rows = []
    for m in members:
        L = tifffile.imread(str(perrun_outputs(m)[0])).astype(np.int32)
        info = pd.read_csv(perrun_outputs(m)[1]).set_index("id") if perrun_outputs(m)[1].stat().st_size > 1 else pd.DataFrame()
        if lab is None:
            lab = np.zeros(L.shape, np.uint16)
        for k in range(1, int(L.max()) + 1):
            u = L == k
            n = int(u.sum())
            if n == 0:
                continue
            if (lab[u] > 0).mean() >= overlap_max:
                continue
            new = len(rows) + 1
            add = u & (lab == 0)
            lab[add] = new
            row = info.loc[k].to_dict() if k in info.index else {}
            row.update(id=new, name=f"dend_{new - 1:03d}", n_vox=int(add.sum()), origin_run=m.run, origin_id=k)
            rows.append(row)
    return lab, rows


def process_run(key: str, force=False, cache: str | None = None, verbose=True, fov_mode=True) -> dict:
    """Detect dendrites for one run.

    Singleton FOV (or --per-run): auto_labelmap.tif = this run's own detection.
    FOV shared by several runs (MASK_SOURCE): every member is detected on its own
    (auto_labelmap_perrun.tif), and all members get the same union of those detections as
    auto_labelmap.tif, so a dendrite that is silent in one run but active in another is kept
    and identities are shared within the FOV (the human curation shares masks the same way)."""
    import fcntl
    r = get_run(key)
    t0 = time.time()
    members = fov_members(r) if fov_mode else [r]
    changed = False
    for m in members:
        lock = m.outdir("masks") / ".detect.lock"
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)       # parallel jobs never detect the same run twice
            changed |= detect_own(m, force and m.key == r.key, cache, verbose)
    lab_p, csv_p = outputs(r)
    newest = max(perrun_outputs(m)[0].stat().st_mtime for m in members)
    d = json_get(r, "detect")
    if (not changed and not force and lab_p.exists() and lab_p.stat().st_mtime >= newest
            and d.get("fov_members") == [m.run for m in members] and d.get("params_hash") == params_hash()):
        log(f"[{key}] up to date", verbose)
        return {"run": key, "status": "skipped"}
    if len(members) == 1:
        lab = tifffile.imread(str(perrun_outputs(r)[0]))
        rows = pd.read_csv(perrun_outputs(r)[1]).to_dict("records") if perrun_outputs(r)[1].stat().st_size > 1 else []
        for x in rows:
            x.update(origin_run=r.run, origin_id=x["id"])
    else:
        lab, rows = fov_union(members)
        ref = r.outdir("reference")
        if members[0].key != r.key:     # reference images: the source run's (same FOV)
            import shutil
            for n in ("ref_mean.tif", "ref_activity.tif"):
                p = members[0].out / "reference" / n
                if p.exists():
                    shutil.copy2(p, ref / n)
    r.outdir("masks")
    same = lab_p.exists() and np.array_equal(tifffile.imread(str(lab_p)), lab.astype(np.uint16))
    if not same:           # keep the old mtime when nothing changed, so later stages are not redone
        _atomic_tif(lab_p, lab.astype(np.uint16))
    pd.DataFrame(rows).to_csv(csv_p, index=False)
    per_member = {m.run: int(sum(1 for x in rows if x.get("origin_run") == m.run)) for m in members}
    merge_metrics(r, "detect", {"n_auto": len(rows), "version": __version__, "params": P,
                                "params_hash": params_hash(), "fov_members": [m.run for m in members],
                                "units_from_each_run": per_member, "train_run": key in TRAIN_RUNS,
                                "time_s": round(time.time() - t0, 1),
                                "timestamp": datetime.now(timezone.utc).isoformat()})
    log(f"[{key}] {len(rows)} dendrites" + (f" (FOV union of {per_member})" if len(members) > 1 else ""), verbose)
    return {"run": key, "status": "done", "n": len(rows), "time_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cache", default=None, help="z-volume cache dir (tuning only)")
    ap.add_argument("--per-run", action="store_true",
                    help="detect every run on its own, even when it shares a FOV with another run")
    a = ap.parse_args()
    keys = [r.key for r in discover_runs()] if a.all else a.run
    if not keys:
        ap.error("give --run or --all")
    res = []
    if a.jobs <= 1:
        for k in keys:
            try:
                res.append(process_run(k, a.force, a.cache, fov_mode=not a.per_run))
            except Exception as e:  # keep going, report at the end
                import traceback
                traceback.print_exc()
                res.append({"run": k, "status": "error", "error": repr(e)})
    else:
        with ProcessPoolExecutor(min(a.jobs, 2)) as ex:   # ~8 GB per job
            futs = {ex.submit(process_run, k, a.force, a.cache, True, not a.per_run): k for k in keys}
            for f in as_completed(futs):
                try:
                    res.append(f.result())
                except Exception as e:
                    res.append({"run": futs[f], "status": "error", "error": repr(e)})
    bad = [x for x in res if x["status"] == "error"]
    print(f"done={sum(x['status'] == 'done' for x in res)} skipped={sum(x['status'] == 'skipped' for x in res)} errors={len(bad)}")
    for b in bad:
        print("  ERROR", b["run"], b["error"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
