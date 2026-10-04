#!/usr/bin/env python
"""validate_vs_human.py - compare automatic dendrites with human curation (code/Auto/, v2).

Human masks are not an exhaustive ground truth (curators kept 30-130 of the active
structures per field and drew them wider than the activity), so three kinds of
evidence are reported per run:

1. Functional recall (headline): a human dendrite counts as recovered when some auto
   unit overlaps it (overlap coefficient |A&H|/min(|A|,|H|) >= OVERLAP_MIN) AND their dF/F
   traces (same extractor, OUT/traces/*.csv) correlate r >= TRACE_R_MIN.
2. Spatial agreement: one-to-one Hungarian matching on IoU; precision / recall / F1 at
   IoU 0.1/0.2/0.3/0.5 (a TP needs IoU >= threshold), voxel coverage of the human union.
3. Is an unmatched auto unit real? Split-half reliability: the unit's voxels are split at
   its median depth (2-voxel gap) and the two halves' raw traces are correlated after
   removing the field-wide mean signal. Noise gives r ~ 0; a coherent dendrite gives a
   high r. The same number is computed for the human masks as a reference.
Plus downstream quantities on both sets (n units, mean pairwise r, event rate per min).

Outputs: OUT/validation/validation.json, overlay.png, beyond_human.png;
         merge_metrics(r, 'validation', summary).
CLI: validate_vs_human.py --run KEY [--run ...] | --all [--force] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, open_stack, human_masks, human_masks_valid, merge_metrics, VOXEL_ZYX,
    split_half_reliability,
)
from Auto.auto_detect import TRAIN_RUNS  # noqa: E402

__version__ = "2.0.0"
IOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
OVERLAP_MIN = 0.2
TRACE_R_MIN = 0.7


def auto_labelmap_path(r: Run) -> Path:
    rev = r.out / "masks" / "auto_labelmap_reviewed.tif"
    return rev if rev.exists() else r.out / "masks" / "auto_labelmap.tif"


def load_trace_matrix(path: Path, names: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not all(n in df.columns for n in names):
        return None
    return df[names].to_numpy(np.float64)


def zcols(M):
    M = M - M.mean(0)
    return M / (np.linalg.norm(M, axis=0) + 1e-12)


def event_rate_per_min(M, fr, k=3.0):
    """Threshold crossings at median + k*MAD (robust), per minute."""
    med = np.median(M, 0)
    mad = np.median(np.abs(M - med), 0) * 1.4826 + 1e-9
    above = M > med + k * mad
    n = (np.diff(above.astype(int), axis=0) == 1).sum(0)
    return n / (M.shape[0] / fr / 60.0)


def mean_pairwise_r(M):
    if M is None or M.shape[1] < 2:
        return None
    C = np.corrcoef(M.T)
    return float(np.nanmean(C[np.triu_indices(len(C), 1)]))


def validate(r: Run, force=False, verbose=True) -> dict:
    out = r.outdir("validation")
    jpath = out / "validation.json"
    lpath = auto_labelmap_path(r)
    if not lpath.exists():
        raise FileNotFoundError(f"{lpath} (run auto_detect first)")
    auto_csv = r.out / "traces" / "dff_auto.csv"
    hum_csv = r.out / "traces" / "dff_human_sameextractor.csv"
    newest_in = max(p.stat().st_mtime for p in (lpath, auto_csv, Path(__file__)) if p.exists())
    if not force and jpath.exists() and jpath.stat().st_mtime > newest_in:
        if verbose:
            print(f"[{r.key}] validation up to date")
        return {"run": r.key, "status": "skipped"}
    t0 = time.time()
    lab0 = tifffile.imread(str(lpath)).astype(np.int32)
    shape = lab0.shape
    # ids may have gaps after napari review (deleted units); names follow the original ids
    # (same convention as extract_traces), `lab` is relabelled 1..na for the matrices below
    a_ids = np.unique(lab0)[1:]
    na = int(a_ids.size)
    a_names = [f"dend_{i - 1:03d}" for i in a_ids]
    lut = np.zeros(int(lab0.max()) + 1, np.int32)
    lut[a_ids] = np.arange(1, na + 1)
    lab = lut[lab0]
    a_ids = np.arange(1, na + 1)
    a_sizes = np.bincount(lab.ravel(), minlength=na + 1)[1:]
    hm = human_masks(r, shape) if human_masks_valid(r) else []
    no_human_reason = ("no human masks" if r.human_mask_dir is None else
                       f"human masks come from {r.mask_source}, which images a different FOV (scape_common.human_masks_valid)")
    res = {"run": r.key, "version": __version__, "labelmap": lpath.name,
           "train_run": r.key in TRAIN_RUNS, "n_auto": na, "n_human": len(hm),
           "timestamp": datetime.now(timezone.utc).isoformat()}
    Ma = load_trace_matrix(auto_csv, a_names)
    a_units = [np.flatnonzero(lab.ravel() == i) for i in a_ids]
    if not hm:
        rel_a = split_half_reliability(r, a_units, shape, lab.ravel() > 0) if na else np.array([])
        res.update(skipped_human=True, skip_reason=no_human_reason,
                   reliability_auto_median=float(np.nanmedian(rel_a)) if na else None,
                   auto_mean_pairwise_r=mean_pairwise_r(Ma))
        jpath.write_text(json.dumps(res, indent=2))
        merge_metrics(r, "validation", {k: res[k] for k in ("n_auto", "n_human", "reliability_auto_median", "auto_mean_pairwise_r")})
        make_overlay(r, lab, [], out / "overlay.png", {})
        return {"run": r.key, "status": "done", "note": "no human masks"}

    h_names = [n for n, _ in hm]
    nh = len(hm)
    # intersections auto x human (human masks may overlap each other, so per mask)
    I = np.zeros((na, nh))
    h_sizes = np.zeros(nh)
    h_units = []
    for j, (_, m) in enumerate(hm):
        vid = np.flatnonzero(m.ravel())
        h_units.append(vid)
        h_sizes[j] = vid.size
        I[:, j] = np.bincount(lab.ravel()[vid], minlength=na + 1)[1:]
    U = a_sizes[:, None] + h_sizes[None, :] - I
    iou = I / np.maximum(U, 1)
    oc = I / np.maximum(np.minimum(a_sizes[:, None], h_sizes[None, :]), 1)

    # 2. spatial one-to-one
    ri, ci = linear_sum_assignment(-iou) if na and nh else (np.array([], int), np.array([], int))
    spatial = {}
    for th in IOU_THRESHOLDS:
        tp = int((iou[ri, ci] >= th).sum())
        P = tp / na if na else 0.0
        R = tp / nh
        spatial[f"iou_{th}"] = dict(tp=tp, precision=round(P, 3), recall=round(R, 3),
                                    f1=round(2 * P * R / (P + R), 3) if P + R else 0.0)
    union_h = np.zeros(lab.size, bool)
    for vid in h_units:
        union_h[vid] = True
    coverage = float((union_h & (lab.ravel() > 0)).sum() / union_h.sum())

    # 1. functional
    Mh = load_trace_matrix(hum_csv, h_names)
    func = {}
    best_r = np.full(nh, np.nan)
    a_matched = np.zeros(na, bool)
    if Ma is not None and Mh is not None and na:
        C = zcols(Ma).T @ zcols(Mh)
        ok = (oc >= OVERLAP_MIN) & (C >= TRACE_R_MIN)
        best_r = np.where(oc >= OVERLAP_MIN, C, np.nan)
        best_r = np.array([np.nanmax(c) if np.isfinite(c).any() else np.nan for c in best_r.T])
        a_matched = ok.any(1)
        # signal coverage: can the auto units that sit inside a human mask reproduce its trace?
        # (human masks are often wider than one auto unit and can span 2-3 of them)
        multR = np.full(nh, np.nan)
        Za = zcols(Ma)
        Zh = zcols(Mh)
        for j in range(nh):
            inside = np.flatnonzero(I[:, j] / np.maximum(a_sizes, 1) >= 0.3)
            if inside.size:
                X = Za[:, inside[:8]]
                beta, *_ = np.linalg.lstsq(X, Zh[:, j], rcond=None)
                multR[j] = float(np.corrcoef(X @ beta, Zh[:, j])[0, 1])
        func = dict(human_recall=round(float(ok.any(0).mean()), 3),
                    signal_recall_R07=round(float(np.nanmean(np.where(np.isfinite(multR), multR, 0) >= 0.7)), 3),
                    median_signal_R=round(float(np.nanmedian(multR)), 3) if np.isfinite(multR).any() else None,
                    auto_matched_frac=round(float(a_matched.mean()), 3),
                    median_best_r=round(float(np.nanmedian(best_r)), 3) if np.isfinite(best_r).any() else None,
                    n_human_recovered=int(ok.any(0).sum()),
                    human_best_r=[None if not np.isfinite(x) else round(float(x), 3) for x in best_r],
                    # recall at other r thresholds, for transparency
                    recall_r05=round(float(((oc >= OVERLAP_MIN) & (C >= 0.5)).any(0).mean()), 3),
                    recall_r08=round(float(((oc >= OVERLAP_MIN) & (C >= 0.8)).any(0).mean()), 3))

    # 3. reliability
    def shifted(vid, dx=60):
        z_, y_, x_ = np.unravel_index(vid, shape)
        return np.ravel_multi_index((z_, y_, (x_ + dx) % shape[2]), shape)
    rng = np.random.default_rng(0)
    null_units = [shifted(a_units[k]) for k in rng.choice(na, min(na, 40), replace=False)] if na else []
    excl = lab.ravel() > 0
    for vid in h_units:
        excl[vid] = True
    rel = split_half_reliability(r, a_units + h_units + null_units, shape, excl)
    rel_a, rel_h, rel_null = rel[:na], rel[na:na + nh], rel[na + nh:]
    q10 = np.nanpercentile(rel_h, 10) if np.isfinite(rel_h).any() else np.nan
    reli = dict(auto_median=round(float(np.nanmedian(rel_a)), 3) if na else None,
                null_shifted_median=round(float(np.nanmedian(rel_null)), 3) if len(rel_null) else None,
                human_median=round(float(np.nanmedian(rel_h)), 3),
                human_q10=round(float(q10), 3),
                auto_frac_ge_human_q10=round(float(np.nanmean(rel_a >= q10)), 3) if na else None,
                unmatched_auto_median=round(float(np.nanmedian(rel_a[~a_matched])), 3) if (~a_matched).any() else None,
                unmatched_auto_frac_ge_human_q10=round(float(np.nanmean(rel_a[~a_matched] >= q10)), 3) if (~a_matched).any() else None)

    down = dict(auto_n=na, human_n=nh,
                auto_mean_pairwise_r=mean_pairwise_r(Ma), human_mean_pairwise_r=mean_pairwise_r(Mh),
                auto_event_rate_median=float(np.median(event_rate_per_min(Ma, r.frame_rate))) if Ma is not None and na else None,
                human_event_rate_median=float(np.median(event_rate_per_min(Mh, r.frame_rate))) if Mh is not None else None)
    res.update(functional=func, spatial=spatial, voxel_coverage_of_human=round(coverage, 3),
               reliability=reli, downstream=down,
               params=dict(OVERLAP_MIN=OVERLAP_MIN, TRACE_R_MIN=TRACE_R_MIN))
    jpath.write_text(json.dumps(res, indent=2, default=float))

    matched_pairs = {}
    if func:
        for j in range(nh):
            cand = np.flatnonzero((oc[:, j] >= OVERLAP_MIN) & (C[:, j] >= TRACE_R_MIN))
            if cand.size:
                matched_pairs[j] = int(cand[np.argmax(C[cand, j])]) + 1
    make_overlay(r, lab, hm, out / "overlay.png", matched_pairs)
    make_gallery(r, lab, a_matched, rel_a, Ma, out / "beyond_human.png", a_names)
    summary = dict(n_auto=na, n_human=nh, train_run=res["train_run"],
                   human_recall=func.get("human_recall"), auto_matched_frac=func.get("auto_matched_frac"),
                   median_best_r=func.get("median_best_r"), signal_recall_R07=func.get("signal_recall_R07"),
                   median_signal_R=func.get("median_signal_R"), recall_r05=func.get("recall_r05"),
                   recall_iou_0_1=spatial["iou_0.1"]["recall"], reliability_null=reli["null_shifted_median"], f1_iou_0_2=spatial["iou_0.2"]["f1"],
                   voxel_coverage=round(coverage, 3), reliability_auto=reli["auto_median"],
                   reliability_human=reli["human_median"],
                   auto_mean_pairwise_r=down["auto_mean_pairwise_r"], human_mean_pairwise_r=down["human_mean_pairwise_r"],
                   time_s=round(time.time() - t0, 1))
    merge_metrics(r, "validation", summary)
    if verbose:
        print(f"[{r.key}] auto={na} human={nh} recall={func.get('human_recall')} "
              f"best_r={func.get('median_best_r')} signalR>=.7={func.get('signal_recall_R07')} F1@IoU.2={spatial['iou_0.2']['f1']} "
              f"rel auto/human/null={reli['auto_median']}/{reli['human_median']}/{reli['null_shifted_median']} ({time.time() - t0:.0f}s)")
    return {"run": r.key, "status": "done", **summary}


def _bg(r: Run):
    p = r.out / "reference" / "ref_activity.tif"
    if p.exists():
        return np.clip(tifffile.imread(str(p)), 0, None).max(0), "activity (top-3 z)"
    s = open_stack(r)
    return np.asarray(s[s.shape[0] // 2], np.float32).max(0), "raw frame"


def _labmip(L):
    first = (L > 0).argmax(0)
    return np.take_along_axis(L, first[None], 0)[0]


def make_overlay(r: Run, lab, hm, path, matched):
    bg, bgname = _bg(r)
    asp = VOXEL_ZYX[1] / VOXEL_ZYX[2]
    fig, ax = plt.subplots(1, 3, figsize=(24, 5.5))
    v = np.percentile(bg, 99.5)
    for a in ax:
        a.imshow(bg, cmap="gray", vmin=0, vmax=v, aspect=asp)
        a.set_xticks([]), a.set_yticks([])
    rng = np.random.default_rng(0)
    ca = rng.random((lab.max() + 1, 3)) * 0.8 + 0.2
    ca[0] = 0
    A = _labmip(lab)
    ax[0].imshow(np.dstack([ca[A], (A > 0) * 0.55]), aspect=asp)
    ax[0].set_title(f"auto ({lab.max()} units)")
    H = np.zeros(lab.shape[1:], int)
    for j, (_, m) in enumerate(hm, 1):
        H[(H == 0) & m.any(0)] = j
    ch = rng.random((len(hm) + 1, 3)) * 0.8 + 0.2
    ch[0] = 0
    ax[1].imshow(np.dstack([ch[H], (H > 0) * 0.55]), aspect=asp)
    ax[1].set_title(f"human ({len(hm)} masks)")
    for j, (_, m) in enumerate(hm):
        col = "lime" if j in matched else "red"
        ax[2].contour(m.any(0), [0.5], colors=col, linewidths=0.6)
    ax[2].contour(lab.max(0) > 0, [0.5], colors="cyan", linewidths=0.4)
    ax[2].set_title(f"human outlines: green = recovered ({len(matched)}), red = missed; cyan = auto union")
    fig.suptitle(f"{r.key} - background: {bgname} MIP over Z", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_gallery(r: Run, lab, a_matched, rel, Ma, path, names=None, nmax=24):
    un = np.flatnonzero(~a_matched)
    if not un.size or Ma is None:
        return
    def snr(x):
        med = np.median(x)
        return (x.max() - med) / (1.4826 * np.median(np.abs(x - med)) + 1e-9)
    sc = np.array([snr(Ma[:, k]) for k in un])
    order = un[np.argsort(-sc)][:nmax]
    bg, _ = _bg(r)
    n = len(order)
    fig, axs = plt.subplots(n, 2, figsize=(12, 1.4 * n + 0.6), gridspec_kw={"width_ratios": [1, 4]}, squeeze=False)
    t = np.arange(Ma.shape[0]) / r.frame_rate
    for row, k in enumerate(order):
        m = lab == k + 1
        mip = m.any(0)
        ys, xs = np.nonzero(mip)
        x0, x1 = max(0, xs.min() - 15), min(mip.shape[1], xs.max() + 15)
        axs[row, 0].imshow(bg[:, x0:x1], cmap="gray", vmin=0, vmax=np.percentile(bg, 99.5), aspect="auto")
        axs[row, 0].contour(mip[:, x0:x1], [0.5], colors="lime", linewidths=0.7)
        axs[row, 0].set_axis_off()
        axs[row, 1].plot(t, Ma[:, k], lw=0.6, color="k")
        axs[row, 1].set_ylabel(names[k] if names else f"dend_{k:03d}", fontsize=7, rotation=0, ha="right")
        axs[row, 1].text(1.0, 0.8, f"SNR {sc[list(un).index(k)]:.0f}  split-half r {rel[k]:.2f}", transform=axs[row, 1].transAxes, ha="right", fontsize=7)
        axs[row, 1].tick_params(labelsize=6)
    axs[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{r.key}: auto units with no human match ({un.size}); top {n} by SNR. Trace = dF/F (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _job(key, force):
    return validate(get_run(key), force)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    keys = [r.key for r in discover_runs()] if a.all else a.run
    if not keys:
        ap.error("give --run or --all")
    res = []
    with ProcessPoolExecutor(max(1, min(a.jobs, 3))) as ex:
        futs = {ex.submit(_job, k, a.force): k for k in keys}
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
