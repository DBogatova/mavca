#!/usr/bin/env python
"""combo_plot.py - one figure per run tying dendritic activity to behavior (code/Auto/, v2).

Left column, one shared time axis:
  A  global Ca dF/F (whole field)                       B  all dendrites, heatmap of robust z
  C  fraction of dendrites active per frame (z > 3)     D  the 10 highest-SNR dendrites
  E  pupil, whisker, accelerometer (accel y-limit 0.25)
Right column:
  F  dendrite map (MIP over Z) coloured by each dendrite's correlation with locomotion (accel)
  G  dendrite x dendrite correlation matrix (clustered), mean r
  H  pairwise r distribution vs a circular-shift null
  I  per-dendrite r with pupil / whisker / accel; grey band = 95% circular-shift null;
     filled = significant after Benjamini-Hochberg (q < 0.05)
  J  behavior-onset-triggered global Ca (accel and whisking onsets; pupil dilation onsets),
     mean +- 95% bootstrap CI, with the shuffled-onset mean +- 95% band

Inputs: OUT/traces/dff_auto.csv + masks (reviewed labelmap preferred) for --source auto;
OUT/traces/dff_human_sameextractor.csv + human masks for --source human;
OUT/traces/global_ca.csv; behavior via scape_common.load_behavior (imaging clock).
Output: OUT/figures/combo_<source>.png and .pdf (vector, Arial, fonttype 42),
        OUT/figures/combo_<source>_dendrite_behavior.csv (per-dendrite r, p, q).
CLI: combo_plot.py --run KEY | --all [--source auto|human] [--force] [--jobs N]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy.cluster.hierarchy import linkage, leaves_list  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, load_behavior, resample_to, human_labelmap, human_masks_valid, open_stack, merge_metrics, VOXEL_ZYX,
)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
N_SHIFT = 500
BEH_COLORS = {"pupil": "#2f6fdf", "whisker": "#e69500", "accel": "#8a3fd1"}


def robust_z(M):
    med = np.median(M, 0)
    mad = np.median(np.abs(M - med), 0) * 1.4826 + 1e-9
    return (M - med) / mad


def corr_cols(M, v):
    Mz = (M - M.mean(0)) / (M.std(0) + 1e-12)
    vz = (v - v.mean()) / (v.std() + 1e-12)
    return (Mz * vz[:, None]).mean(0)


def bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def onsets(y, fr, k=3.0, refractory_s=2.0):
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med)) * 1.4826 + 1e-12
    above = y > med + k * mad
    idx = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    keep, last = [], -1e9
    ref = int(refractory_s * fr)
    for i in idx:
        if i - last >= ref and not above[max(0, i - ref):i].any():
            keep.append(i)
        last = i
    return np.array(keep, int)


def load_inputs(r: Run, source: str):
    if source == "auto":
        csv = r.out / "traces" / "dff_auto.csv"
        lp = r.out / "masks" / "auto_labelmap_reviewed.tif"
        lp = lp if lp.exists() else r.out / "masks" / "auto_labelmap.tif"
        if not csv.exists() or not lp.exists():
            raise FileNotFoundError("auto traces/masks missing (run auto_detect + extract_traces)")
        lab = tifffile.imread(str(lp))
    else:
        csv = r.out / "traces" / "dff_human_sameextractor.csv"
        if not csv.exists():
            raise FileNotFoundError("dff_human_sameextractor.csv missing (run extract_traces)")
        lab = human_labelmap(r, open_stack(r).shape[1:])
        lp = None
    df = pd.read_csv(csv)
    names = [c for c in df.columns if c.startswith("dend_")]
    return df, names, lab, csv, lp


def make_figure(key: str, source="auto", force=False, verbose=True) -> dict:
    r = get_run(key)
    out = r.outdir("figures")
    png, pdf = out / f"combo_{source}.png", out / f"combo_{source}.pdf"
    df, names, lab, csv, lp = load_inputs(r, source)
    if not force and png.exists() and png.stat().st_mtime > max(csv.stat().st_mtime, Path(__file__).stat().st_mtime):
        if verbose:
            print(f"[{key}] combo_{source} up to date")
        return {"run": key, "status": "skipped"}
    t0 = time.time()
    fr = r.frame_rate
    t = df["time_s"].to_numpy()
    M = df[names].to_numpy(float)
    Zs = robust_z(M)
    N = len(names)
    g = pd.read_csv(r.out / "traces" / "global_ca.csv") if (r.out / "traces" / "global_ca.csv").exists() else None
    gy = resample_to(g["time_s"].to_numpy(), g["global_dff"].to_numpy(), t) if g is not None else np.nanmean(M, 1)
    beh = load_behavior(r)
    B = {}
    for k in ("pupil", "whisker", "accel"):
        if beh.get(k) is not None:
            y = resample_to(beh[f"{k}_t"], beh[k], t)
            if np.isfinite(y).sum() > 0.8 * len(t):
                B[k] = np.where(np.isfinite(y), y, np.nanmedian(y))
    valid = np.ones(len(t), bool)

    # per-dendrite behavior correlations + circular-shift null + BH
    rng = np.random.default_rng(0)
    min_shift = int(10 * fr)
    shifts = rng.integers(min_shift, len(t) - min_shift, N_SHIFT)
    rows = []
    for k, y in B.items():
        rr = corr_cols(M, y)
        null = np.stack([corr_cols(M, np.roll(y, s)) for s in shifts])
        p = (1 + (np.abs(null) >= np.abs(rr)[None]).sum(0)) / (N_SHIFT + 1)
        q = bh(p)
        for i, n in enumerate(names):
            rows.append(dict(dendrite=n, behavior=k, r=rr[i], p=p[i], q=q[i],
                             null_lo=np.percentile(null[:, i], 2.5), null_hi=np.percentile(null[:, i], 97.5)))
    bdf = pd.DataFrame(rows)
    if len(bdf):
        bdf.to_csv(out / f"combo_{source}_dendrite_behavior.csv", index=False)

    # pairwise
    C = np.corrcoef(M.T) if N > 1 else np.ones((1, 1))
    iu = np.triu_indices(N, 1)
    pair_r = C[iu] if N > 1 else np.array([])
    Cn = []
    for s in shifts[:20]:
        Mr = np.stack([np.roll(M[:, i], rng.integers(min_shift, len(t) - min_shift)) for i in range(N)], 1)
        Cn.append(np.corrcoef(Mr.T)[iu])
    null_pair = np.concatenate(Cn) if Cn and N > 1 else np.array([])
    order = leaves_list(linkage(M.T, "average", metric="correlation")) if N > 2 else np.arange(N)

    # ---------------- figure
    fig = plt.figure(figsize=(17, 13.5))
    gs = GridSpec(8, 3, figure=fig, width_ratios=[2.6, 1, 1], hspace=0.55, wspace=0.32,
                  height_ratios=[0.8, 2.2, 0.6, 1.8, 0.55, 0.55, 0.55, 0.15])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0], sharex=axA)
    axC = fig.add_subplot(gs[2, 0], sharex=axA)
    axD = fig.add_subplot(gs[3, 0], sharex=axA)
    axE = [fig.add_subplot(gs[4 + i, 0], sharex=axA) for i in range(3)]
    axA.plot(t, gy, color="#1f9e3a", lw=0.8)
    axA.set_ylabel("global\ndF/F (%)")
    axA.set_title(f"{r.key}  -  {source} masks: {N} dendrites  ({fr:g} Hz, first {r.skip_s:g} s dropped)", loc="left", fontsize=10, fontweight="bold")
    im = axB.imshow(Zs[:, order].T, aspect="auto", cmap="magma", vmin=-1, vmax=6, interpolation="nearest",
                    extent=[t[0], t[-1] + 1 / fr, N, 0])
    axB.set_ylabel("dendrite\n(clustered)")
    cb = fig.colorbar(im, ax=axB, pad=0.005, fraction=0.02)
    cb.set_label("robust z", fontsize=7)
    frac = (Zs > 3).mean(1)
    axC.fill_between(t, frac, color="#555555", lw=0)
    axC.set_ylabel("frac.\nactive")
    axC.set_ylim(0, max(0.05, frac.max() * 1.1))
    snr = Zs.max(0)
    top = np.argsort(-snr)[:10]
    off = 0
    for i in top:
        y = M[:, i]
        y = y - np.median(y)
        axD.plot(t, y + off, lw=0.6)
        axD.text(t[-1] + 0.005 * (t[-1] - t[0]), off, names[i], fontsize=6, ha="left", va="bottom", clip_on=False)
        off += max(np.percentile(y, 99.5) * 1.1, 5)
    axD.set_yticks([])
    axD.set_ylabel("top-10 SNR\ndF/F (offset)")
    for ax, k in zip(axE, ("pupil", "whisker", "accel")):
        if k in B:
            ax.plot(t, B[k], color=BEH_COLORS[k], lw=0.7)
            if k == "accel":
                ax.set_ylim(0, 0.25)
        else:
            ax.text(0.5, 0.5, f"no {k} data", transform=ax.transAxes, ha="center", va="center", color="grey")
        ax.set_ylabel(k, color=BEH_COLORS[k])
    axE[-1].set_xlabel("time (s)")
    for ax in [axA, axB, axC, axD] + axE[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axA.set_xlim(t[0], t[-1])

    # F map
    axF = fig.add_subplot(gs[0:2, 1:3])
    refp = r.out / "reference" / "ref_activity.tif"
    if refp.exists():
        bg = np.clip(tifffile.imread(str(refp)), 0, None).max(0)
        axF.imshow(bg, cmap="gray", vmin=0, vmax=np.percentile(bg, 99.5), aspect=VOXEL_ZYX[1] / VOXEL_ZYX[2])
    first = (lab > 0).argmax(0)
    L2 = np.take_along_axis(lab, first[None], 0)[0].astype(int)
    key_b = "accel" if "accel" in B else (next(iter(B)) if B else None)
    if key_b:
        rv = bdf[bdf.behavior == key_b].set_index("dendrite")["r"]
        vals = np.full(lab.max() + 1, np.nan)
        for n in names:
            k = int(n.split("_")[1]) + 1 if source == "auto" else names.index(n) + 1
            if k < vals.size:
                vals[k] = rv.get(n, np.nan)
        lim = max(0.1, np.nanpercentile(np.abs(vals[1:]), 95))
        img = np.where(L2 > 0, vals[L2], np.nan)
        hm = axF.imshow(img, cmap="coolwarm", vmin=-lim, vmax=lim, aspect=VOXEL_ZYX[1] / VOXEL_ZYX[2], alpha=0.85)
        c2 = fig.colorbar(hm, ax=axF, fraction=0.025, pad=0.01)
        c2.set_label(f"r with {key_b}", fontsize=7)
        axF.set_title(f"F  dendrite map, coloured by r with {key_b} (MIP over Z; Y = depth)", loc="left")
    else:
        rng2 = np.random.default_rng(1)
        cols = rng2.random((lab.max() + 1, 3))
        cols[0] = 0
        axF.imshow(np.dstack([cols[L2], (L2 > 0) * 0.8]), aspect=VOXEL_ZYX[1] / VOXEL_ZYX[2])
        axF.set_title("F  dendrite map (no behavior for this run)", loc="left")
    sb = 100 / VOXEL_ZYX[2]
    axF.plot([lab.shape[2] - sb - 8, lab.shape[2] - 8], [lab.shape[1] - 8] * 2, color="w", lw=2)
    axF.text(lab.shape[2] - sb / 2 - 8, lab.shape[1] - 12, "100 um", color="w", ha="center", fontsize=7)
    axF.set_xticks([]), axF.set_yticks([])

    axG = fig.add_subplot(gs[2:4, 1])
    if N > 1:
        Cg = C[np.ix_(order, order)].copy()
        np.fill_diagonal(Cg, np.nan)
        hg = axG.imshow(Cg, cmap="RdBu_r", vmin=-0.6, vmax=0.6, interpolation="nearest")
        fig.colorbar(hg, ax=axG, fraction=0.045, pad=0.02)
    axG.set_title(f"G  dendrite x dendrite r (mean {np.nanmean(pair_r):.3f})", loc="left")
    axG.set_xticks([]), axG.set_yticks([])

    axH = fig.add_subplot(gs[2:4, 2])
    if pair_r.size:
        bins = np.linspace(-0.5, 1, 61)
        axH.hist(null_pair, bins, density=True, color="#bbbbbb", label="circular-shift null")
        axH.hist(pair_r, bins, density=True, histtype="step", color="k", lw=1.2, label="observed")
        hi = np.percentile(null_pair, 97.5)
        axH.axvline(hi, color="grey", ls="--", lw=0.8)
        axH.set_title(f"H  pairwise r: {(pair_r > hi).mean() * 100:.0f}% of pairs > null 97.5%", loc="left")
        axH.legend(fontsize=6, frameon=False)
        axH.set_xlabel("r")

    axI = fig.add_subplot(gs[4:7, 1])
    if len(bdf):
        for j, k in enumerate(("pupil", "whisker", "accel")):
            d = bdf[bdf.behavior == k]
            if not len(d):
                continue
            x = j + (rng.random(len(d)) - 0.5) * 0.5
            sig = d.q.to_numpy() < 0.05
            axI.fill_between([j - 0.35, j + 0.35], d.null_lo.median(), d.null_hi.median(), color="#dddddd", zorder=0)
            axI.scatter(x[~sig], d.r[~sig], s=9, facecolors="none", edgecolors=BEH_COLORS[k], lw=0.6)
            axI.scatter(x[sig], d.r[sig], s=9, color=BEH_COLORS[k])
            axI.text(j, 1.01, f"{sig.mean() * 100:.0f}% sig", ha="center", fontsize=7, transform=axI.get_xaxis_transform())
        axI.axhline(0, color="k", lw=0.5)
        axI.set_xticks([0, 1, 2], ["pupil", "whisker", "accel"])
        axI.set_ylabel("r (dendrite dF/F vs behavior)")
    axI.set_title("I  dendrite-behavior r (filled: BH q<0.05)", loc="left", pad=14)

    axJ = fig.add_subplot(gs[4:7, 2])
    win = int(3 * fr)
    lags = np.arange(-win, win + 1) / fr
    trig = {}
    for k in ("accel", "whisker", "pupil"):
        if k not in B:
            continue
        on = onsets(B[k], fr)
        on = on[(on >= win) & (on < len(t) - win)]
        if on.size < 3:
            continue
        seg = np.stack([gy[i - win:i + win + 1] - gy[i - win:i].mean() for i in on])
        boots = np.stack([seg[rng.integers(0, len(seg), len(seg))].mean(0) for _ in range(500)])
        sh = []
        for _ in range(300):
            rand = rng.integers(win, len(t) - win, on.size)
            sh.append(np.stack([gy[i - win:i + win + 1] - gy[i - win:i].mean() for i in rand]).mean(0))
        sh = np.array(sh)
        m = seg.mean(0)
        axJ.plot(lags, m, color=BEH_COLORS[k], lw=1.2, label=f"{k} onsets (n={on.size})")
        axJ.fill_between(lags, *np.percentile(boots, [2.5, 97.5], 0), color=BEH_COLORS[k], alpha=0.2, lw=0)
        post = (lags > 0) & (lags <= 1.5)
        trig[k] = dict(n=int(on.size), post_mean=float(m[post].mean()),
                       p_vs_shuffle=float((1 + (sh[:, post].mean(1) >= m[post].mean()).sum()) / (len(sh) + 1)))
        if k == "accel":
            axJ.fill_between(lags, *np.percentile(sh, [2.5, 97.5], 0), color="#cccccc", alpha=0.6, lw=0, label="shuffled onsets")
    axJ.axvline(0, color="k", lw=0.5, ls="--")
    axJ.set_xlabel("time from onset (s)")
    axJ.set_ylabel("global dF/F change (%)")
    axJ.legend(fontsize=6, frameon=False)
    axJ.set_title("J  onset-triggered global Ca", loc="left")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summ = dict(n=N, mean_pair_r=float(np.nanmean(pair_r)) if pair_r.size else None,
                frac_pairs_above_null=float((pair_r > np.percentile(null_pair, 97.5)).mean()) if pair_r.size else None,
                frac_sig_behavior={k: float((bdf[bdf.behavior == k].q < 0.05).mean()) for k in B} if len(bdf) else {},
                global_r_behavior={k: float(np.corrcoef(gy, B[k])[0, 1]) for k in B},
                triggered=trig, time_s=round(time.time() - t0, 1))
    merge_metrics(r, f"combo_{source}", summ)
    if verbose:
        print(f"[{key}] combo_{source}: {N} dendrites, mean pair r {summ['mean_pair_r']}, {time.time() - t0:.0f}s")
    return {"run": key, "status": "done"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--source", choices=("auto", "human"), default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    runs = discover_runs() if a.all else [get_run(k) for k in a.run]
    if not runs:
        ap.error("give --run or --all")
    if a.source == "human":
        runs = [r for r in runs if human_masks_valid(r)]
    res = []
    with ProcessPoolExecutor(max(1, min(a.jobs, 4))) as ex:
        futs = {ex.submit(make_figure, r.key, a.source, a.force): r.key for r in runs}
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
