#!/usr/bin/env python
"""scape_common.py - shared contract for the automatic SCAPE pipeline (code/Auto/).

Every Auto script imports from here so that paths, per-run parameters and the
behavior/accelerometer loaders are defined exactly once.

Rules
-----
* READ-ONLY on scape-data/. All outputs go to scape-auto/<DATE>/<MOUSE>/<RUN>/.
* Per-run frame rate / skip / mask-source come from code/Workflow/mavca_status.py
  (SESSION_PARAMS, MASK_SOURCE), which is cited to .kiro/steering/pipeline-context.md.
  Nothing here hardcodes a rate.
* Stack axes: raw stack is (T, Z, Y, X) uint16, Y = cortical depth (row 0 = surface).
  Human masks are (Z, Y-3, X): M1 trims Y_CROP=3 rows from the DEEP end. Auto masks
  are written at the full raw (Z, Y, X) shape; `human_union()` pads human masks back.
* Voxel size (Z, Y, X) = (3.9, 1.0, 1.2) um.

Output layout per run (OUT = scape-auto/<DATE>/<MOUSE>/<RUN>)
    OUT/masks/auto_labelmap.tif        uint16 (Z,Y,X), 0=bg, k=dendrite k (1..N)
    OUT/masks/auto_masks.csv           one row per dendrite (id, name, n_vox, centroid, length_um, verticality, score...)
    OUT/traces/dff_auto.csv            Frame, time_s, dend_000..  (dF/F, core-shell bg-sub, F0=10th pct)
    OUT/traces/dff_human_sameextractor.csv   same extractor on the human masks (if any)
    OUT/traces/global_ca.csv           time_s, global_dff (mean of live voxels, %)
    OUT/reference/ref_mean.tif, ref_max_dff.tif   (Z,Y,X) float32
    OUT/figures/*.png|pdf              combo plots
    OUT/movies/*.mp4
    OUT/metrics.json                   per-run summary (written by several stages, merged)
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "scape-data"
AUTO_ROOT = PROJECT / "scape-auto"
PYTHON = str(PROJECT / ".venv311" / "bin" / "python")
VOXEL_ZYX = (3.9, 1.0, 1.2)
HUMAN_Y_CROP = 3
CHANGELOG = PROJECT / "code" / "Auto" / "CHANGELOG_auto.json"

sys.path.insert(0, str(PROJECT / "code" / "Workflow"))
from mavca_status import SESSION_PARAMS, MASK_SOURCE, DEFAULT_FRAME_RATE, DEFAULT_SKIP_S  # noqa: E402


@dataclass
class Run:
    date: str
    mouse: str
    run: str
    raw: Path | None
    frame_rate: float
    skip_s: float
    has_ach: bool
    mask_source: str | None            # run whose human masks this run uses
    human_mask_dir: Path | None         # labelmaps_curated_dynamic actually used by the human traces
    human_traces: Path | None           # traces/dff_traces_curated_bgsub.csv
    behavior_mat: Path | None
    accel_csv: Path | None
    trigger_csv: Path | None
    known_params: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.date}/{self.mouse}/{self.run}"

    @property
    def src(self) -> Path:
        return DATA_ROOT / self.date / self.mouse / self.run

    @property
    def out(self) -> Path:
        return AUTO_ROOT / self.date / self.mouse / self.run

    @property
    def run_num(self) -> str:
        return re.sub(r"\D", "", self.run).zfill(3)

    def outdir(self, sub: str) -> Path:
        p = self.out / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def _first(paths):
    paths = sorted(paths)
    return paths[0] if paths else None


def find_raw(src: Path) -> Path | None:
    """Raw green/bin 4D stack (T,Z,Y,X). Excludes *-dff.tif and red channel."""
    cands = [p for p in (src / "raw").glob("run*_*-reslice-*.tif")
             if not p.name.endswith("-dff.tif") and "-red" not in p.name]
    return _first(cands)


def discover_runs(only_with_raw: bool = True) -> list[Run]:
    runs = []
    for d in sorted(DATA_ROOT.glob("*/*/run*")):
        if not d.is_dir():
            continue
        date, mouse, run = d.parts[-3], d.parts[-2], d.name
        raw = find_raw(d)
        if only_with_raw and raw is None:
            continue
        p = SESSION_PARAMS.get((date, mouse))
        ms = MASK_SOURCE.get((date, mouse, run))
        hdir = (d.parent / ms if ms else d) / "labelmaps_curated_dynamic"
        tr = d / "traces" / "dff_traces_curated_bgsub.csv"
        runs.append(Run(
            date=date, mouse=mouse, run=run, raw=raw,
            frame_rate=float(p["frame_rate"]) if p else DEFAULT_FRAME_RATE,
            skip_s=float(p["skip_s"]) if p else DEFAULT_SKIP_S,
            has_ach=bool(p.get("has_ach", False)) if p else False,
            mask_source=ms,
            human_mask_dir=hdir if any(hdir.glob("dend_*_labelmap.tif")) else None,
            human_traces=tr if tr.exists() else None,
            behavior_mat=_first((d / "behavior").glob("*_behavior.mat")),
            accel_csv=_first((d / "trigger").glob("Run*_t1_accel.csv")),
            trigger_csv=_first((d / "trigger").glob("Run*_t1_trigger.csv")),
            known_params=p is not None,
        ))
    runs.sort(key=lambda r: (r.date, r.mouse, int(re.sub(r"\D", "", r.run) or 0)))
    return runs


def get_run(key: str) -> Run:
    """key = 'DATE/MOUSE/RUN' or 'DATE/RUN' if unambiguous."""
    for r in discover_runs(only_with_raw=False):
        if r.key == key or f"{r.date}/{r.run}" == key:
            return r
    raise KeyError(key)


FOV_MIN_CORR = 0.75   # same-FOV run pairs score 0.80-0.87, different FOVs 0.51-0.56 (checked 2026-10-04)


def _fov_img(run: "Run"):
    from scipy import ndimage as ndi
    st = open_stack(run)
    a = np.asarray(st[100:min(400, st.shape[0]):10], np.float32).mean(0)
    return a - ndi.gaussian_filter(a, (1, 6, 6))


def fov_pair_similarity(a: "Run", b: "Run") -> float:
    """Correlation of the spatially high-passed mean images of two runs (cached in
    scape-auto/fov_check.json)."""
    cache = AUTO_ROOT / "fov_check.json"
    d = json.loads(cache.read_text()) if cache.exists() else {}
    k = "|".join(sorted((a.key, b.key)))
    if k not in d:
        A, B = _fov_img(a), _fov_img(b)
        if A.shape != B.shape:
            d[k] = 0.0
        else:
            d[k] = round(float(np.corrcoef(A[:, :, 10:-10].ravel(), B[:, :, 10:-10].ravel())[0, 1]), 4)
        AUTO_ROOT.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d, indent=1, sort_keys=True))
    return d[k]


_FOV_CACHE: dict = {}


def session_fov_groups(date: str, mouse: str) -> dict[str, str]:
    """run -> FOV label (a run name) for one session, from image similarity (not from the
    hand-maintained MASK_SOURCE table). Runs whose images correlate >= FOV_MIN_CORR are joined
    (single linkage). The label is the run that other runs take their human masks from when
    there is one in the group, else the first run."""
    key = (date, mouse)
    if key in _FOV_CACHE:
        return _FOV_CACHE[key]
    runs = [r for r in discover_runs() if r.date == date and r.mouse == mouse]
    parent = {r.run: r.run for r in runs}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            if fov_pair_similarity(runs[i], runs[j]) >= FOV_MIN_CORR:
                parent[find(runs[j].run)] = find(runs[i].run)
    comp: dict[str, list] = {}
    for r in runs:
        comp.setdefault(find(r.run), []).append(r)
    out = {}
    for members in comp.values():
        names = {m.run for m in members}
        srcs = [m.mask_source for m in members if m.mask_source in names]
        label = srcs[0] if srcs else members[0].run
        for m in members:
            out[m.run] = label
    _FOV_CACHE[key] = out
    return out


def fov_group(r: Run) -> str:
    """Field-of-view label 'DATE/MOUSE/RUN' shared by all runs imaging the same FOV."""
    return f"{r.date}/{r.mouse}/{session_fov_groups(r.date, r.mouse)[r.run]}"


def fov_similarity(r: Run) -> float | None:
    """Image similarity of r with its MASK_SOURCE run (None if it has none)."""
    if not r.mask_source:
        return None
    return fov_pair_similarity(r, get_run(f"{r.date}/{r.mouse}/{r.mask_source}"))


def split_half_reliability(r: Run, units: list[np.ndarray], shape, exclude: np.ndarray | None = None) -> np.ndarray:
    """units: list of flat voxel index arrays. Returns r(top half, bottom half) per unit.

    Each half's trace is background-subtracted with its own local shell (2-3 voxels out,
    excluding all units), as in the trace extractor, so the shared neuropil/field signal
    does not make unrelated voxels look reliable. Translated copies of the units serve as
    the null in validate()."""
    Z, Y, X = shape
    excl = np.zeros(Z * Y * X, bool) if exclude is None else exclude
    from scipy import ndimage as ndi
    st = ndi.generate_binary_structure(3, 1)

    def shell(vid):
        m = np.zeros(Z * Y * X, bool)
        m[vid] = True
        m = m.reshape(shape)
        sl = tuple(slice(max(0, int(c.min()) - 4), int(c.max()) + 5) for c in np.unravel_index(vid, shape))
        sub = m[sl]
        o = ndi.binary_dilation(sub, st, iterations=3) & ~ndi.binary_dilation(sub, st, iterations=1)
        full = np.zeros(shape, bool)
        full[sl] = o
        full = full.ravel() & ~excl
        return np.flatnonzero(full)

    halves = []
    for vid in units:
        yy = np.unravel_index(vid, shape)[1]
        med = np.median(yy)
        a, b = vid[yy < med - 1], vid[yy > med + 1]
        if a.size >= 20 and b.size >= 20:
            sa, sb = shell(a), shell(b)
            halves.append((a, b, sa, sb) if sa.size >= 10 and sb.size >= 10 else None)
        else:
            halves.append(None)
    s = open_stack(r)
    sk = int(round(r.skip_s * r.frame_rate))
    T = s.shape[0] - sk
    A = np.zeros((T, len(units)))
    B = np.zeros((T, len(units)))
    for t in range(T):
        f = np.asarray(s[t + sk], np.float32).ravel()
        for k, h in enumerate(halves):
            if h is not None:
                A[t, k] = f[h[0]].mean() - f[h[2]].mean()
                B[t, k] = f[h[1]].mean() - f[h[3]].mean()
    D = np.c_[np.ones(T), np.linspace(-1, 1, T)]
    proj = D @ np.linalg.pinv(D)
    A -= proj @ A
    B -= proj @ B
    out = np.full(len(units), np.nan)
    for k, h in enumerate(halves):
        if h is not None:
            out[k] = float(np.corrcoef(A[:, k], B[:, k])[0, 1])
    return out


def human_mask_home(r: Run) -> str | None:
    """Which run of the session do r's human masks actually fit? Split-half reliability of up to
    40 masks is measured on every run of the session; the best one is the masks' home.
    Cached in scape-auto/human_mask_check.json (keyed by the mask folder)."""
    if r.human_mask_dir is None:
        return None
    cache = AUTO_ROOT / "human_mask_check.json"
    d = json.loads(cache.read_text()) if cache.exists() else {}
    k = str(r.human_mask_dir.relative_to(DATA_ROOT))
    if k not in d:
        runs = [x for x in discover_runs() if x.date == r.date and x.mouse == r.mouse]
        shape = open_stack(r).shape[1:]
        hm = human_masks(r, shape)
        units = [np.flatnonzero(m.ravel()) for _, m in hm[:40]]
        excl = np.zeros(int(np.prod(shape)), bool)
        for u in units:
            excl[u] = True
        scores = {}
        for x in runs:
            if open_stack(x).shape[1:] == tuple(shape):
                scores[x.run] = round(float(np.nanmedian(split_half_reliability(x, units, shape, excl))), 4)
        d[k] = {"split_half_r_by_run": scores, "home": max(scores, key=scores.get)}
        AUTO_ROOT.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d, indent=1, sort_keys=True))
    return d[k]["home"]


def human_masks_valid(r: Run) -> bool:
    """Human masks exist AND fit this run's field of view: the run the masks fit best
    (human_mask_home) must be in the same FOV group as r. Example: the 119 masks in
    2026-04-16/run5/labelmaps_curated_dynamic fit run7/run8 (split-half r 0.62/0.49) and not
    run5/run6 (0.07/0.03, noise level): valid for run7 (the FOV they were drawn on), not for run5/run6."""
    home = human_mask_home(r)
    if home is None:
        return False
    return fov_group(r) == fov_group(get_run(f"{r.date}/{r.mouse}/{home}"))


# --------------------------------------------------------------------------- stacks
def open_stack(r: Run):
    import tifffile
    return tifffile.memmap(str(r.raw), mode="r")


def human_masks(r: Run, shape_zyx: tuple[int, int, int] | None = None) -> list[tuple[str, np.ndarray]]:
    """Human curated masks as boolean arrays, padded at the deep end of Y to shape_zyx."""
    import tifffile
    out = []
    if r.human_mask_dir is None:
        return out
    for p in sorted(r.human_mask_dir.glob("dend_*_labelmap.tif")):
        m = tifffile.imread(str(p)) > 0
        if shape_zyx is not None and m.shape != tuple(shape_zyx):
            dz, dy, dx = (s - t for s, t in zip(shape_zyx, m.shape))
            if dz != 0 or dx != 0 or not (0 <= dy <= HUMAN_Y_CROP):
                raise ValueError(f"{p}: mask {m.shape} vs stack {shape_zyx} not a deep-Y crop")
            m = np.pad(m, ((0, 0), (0, dy), (0, 0)))
        out.append((p.stem.replace("_labelmap", ""), m))
    return out


def human_labelmap(r: Run, shape_zyx) -> np.ndarray:
    lab = np.zeros(shape_zyx, np.uint16)
    for i, (_, m) in enumerate(human_masks(r, shape_zyx), 1):
        lab[m & (lab == 0)] = i
    return lab


# --------------------------------------------------------------------------- behavior
def load_behavior(r: Run, crop_s: float | None = None) -> dict:
    """Pupil, whisker (10 Hz camera) and accelerometer (1 kHz), all on the imaging clock.

    Ported from code/Behavior-Analysis/behavior_plots.py:
      * camera->imaging offset = -settings.aligned_time_s[0] from the run's .mat if present,
        else (AndorXyla first rise - basler first rise) from the run's OWN trigger CSV.
      * accel uses aligned_time_s (already imaging-referenced), |accel_mag|, gaussian sigma=10.
      * pupil sigma=2, whisker (whisker_smooth_long) sigma=3.
    Returned times are seconds since imaging frame 0 minus crop_s (default r.skip_s), so they
    share an axis with Ca time = frame/frame_rate - crop_s.
    Each entry is None when the source is missing.
    """
    import pandas as pd
    from scipy.io import loadmat
    from scipy.ndimage import gaussian_filter1d

    crop = r.skip_s if crop_s is None else crop_s
    out = {"pupil_t": None, "pupil": None, "whisker_t": None, "whisker": None,
           "accel_t": None, "accel": None, "offset_s": None, "offset_source": None}

    if r.behavior_mat is not None:
        m = loadmat(str(r.behavior_mat))
        pupil = gaussian_filter1d(m["pupil"]["pupil_raw"][0][0].ravel().astype(float), 2)
        whisk = gaussian_filter1d(m["whisker"]["whisker_smooth_long"][0][0].ravel().astype(float), 3)
        offset, src = None, None
        try:
            at = m["settings"]["aligned_time_s"][0][0].ravel()
            if at.size:
                offset, src = float(-at[0]), "mat.settings.aligned_time_s"
        except Exception:
            pass
        if offset is None and r.trigger_csv is not None:
            trig = pd.read_csv(r.trigger_csv, usecols=["time_s", "baslerExposureTrigger", "AndorXylaTrigger"])
            b = trig.loc[trig["baslerExposureTrigger"].diff() == 1, "time_s"]
            a = trig.loc[trig["AndorXylaTrigger"].diff() == 1, "time_s"]
            if len(a) and len(b):
                offset, src = float(a.iloc[0] - b.iloc[0]), f"trigger:{r.trigger_csv.name}"
        if offset is None:
            offset, src = 0.0, "none(assumed 0)"
        tp = np.arange(pupil.size) / 10.0 - offset - crop
        tw = np.arange(whisk.size) / 10.0 - offset - crop
        kp, kw = tp >= 0, tw >= 0
        out.update(pupil_t=tp[kp], pupil=pupil[kp], whisker_t=tw[kw], whisker=whisk[kw],
                   offset_s=offset, offset_source=src)

    if r.accel_csv is not None:
        df = pd.read_csv(r.accel_csv, usecols=lambda c: c in ("accel_mag", "aligned_time_s"))
        t = df["aligned_time_s"].to_numpy() - crop
        a = gaussian_filter1d(np.abs(df["accel_mag"].to_numpy()), 10)
        k = t >= 0
        out.update(accel_t=t[k], accel=a[k])
    return out


def resample_to(t_src, y_src, t_dst):
    """Linear interpolation onto t_dst; NaN outside the source range."""
    if t_src is None or y_src is None:
        return None
    return np.interp(t_dst, t_src, y_src, left=np.nan, right=np.nan)


# --------------------------------------------------------------------------- bookkeeping
def merge_metrics(r: Run, section: str, data: dict) -> Path:
    """Merge one stage's results into OUT/metrics.json under `section`."""
    p = r.outdir(".") / "metrics.json"
    cur = json.loads(p.read_text()) if p.exists() else {"run": r.key}
    cur[section] = data
    p.write_text(json.dumps(cur, indent=2, default=_json_default))
    return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


if __name__ == "__main__":
    for r in discover_runs():
        print(f"{r.key:42s} fr={r.frame_rate} skip={r.skip_s} src={r.mask_source or '-':5s} "
              f"human={'Y' if r.human_mask_dir else '-'} tr={'Y' if r.human_traces else '-'} "
              f"beh={'Y' if r.behavior_mat else '-'} acc={'Y' if r.accel_csv else '-'}")
