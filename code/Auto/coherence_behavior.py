#!/usr/bin/env python
"""coherence_behavior.py - frequency-resolved coupling of dendritic Ca with behavior (code/Auto/).

Why: Ca transients are sharp, pupil (and whisking) responses are slow and smoothed. Even a
tight, consistent relation (pupil = Ca passed through a slow filter, with a lag) gives a small
Pearson r whose sign is easily set by slow drift. Magnitude-squared coherence is invariant to
any linear filter between the two signals, so it measures how consistently they co-vary at
each frequency regardless of waveform shape; the cross-spectrum phase gives the lag.

Per run (source auto or human; same extractor for both):
  * Welch coherence (25.6 s segments, 50% overlap, linear detrend; ~7 segments per run) of
    global Ca and of every dendrite with pupil, whisker and accel (behavior resampled onto the
    imaging frames by scape_common.load_behavior / resample_to).
  * Null: 200 circular shifts of the behavior trace (>= 10 s), which keep both spectra and
    break only their alignment. Band coherence p = fraction of shifts >= observed; BH-FDR
    across dendrites.
  * Bands: slow 0.04-0.2 Hz, mid 0.2-0.5 Hz, fast 0.5-1.0 Hz.
  * Lag in the slow band from the cross-spectrum phase (positive = behavior follows Ca).
  * Ca-event-triggered pupil: pupil around global-Ca event onsets and around local dendritic
    events (no global event within +-1 s), baseline-subtracted, vs randomly placed onsets.
Cohort: FOV is the unit of replication (runs sharing a FOV averaged first); Wilcoxon over FOVs
of (observed - null mean) band coherence.

Outputs: OUT/figures/coherence_<source>.png (per run)
         scape-auto/stats/coherence/<source>/{per_run.csv, per_dendrite.csv, summary.txt, summary.json,
                                             fig_coherence.png/.pdf}
         scape-auto/stats/coherence/compare_auto_vs_human.txt
CLI: coherence_behavior.py [--source auto|human|both] [--run KEY ...] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats as sp
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Auto.scape_common import (  # noqa: E402
    discover_runs, get_run, Run, fov_group, load_behavior, resample_to, human_masks_valid, AUTO_ROOT,
)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore")
OUT = AUTO_ROOT / "stats" / "coherence"
BEHS = ("pupil", "whisker", "accel")
BANDS = {"slow": (0.04, 0.2), "mid": (0.2, 0.5), "fast": (0.5, 1.0)}
SEG_S = 25.6
N_SHIFT = 200
COL = {"pupil": "#2f6fdf", "whisker": "#e69500", "accel": "#8a3fd1"}


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    n = p.size
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def coh(X, y, fr):
    """X (T, N), y (T,) -> f, C (N, F), Pxy (N, F)."""
    nper = int(round(SEG_S * fr))
    f, C = signal.coherence(X.T, y[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
    _, Pxy = signal.csd(X.T, y[None, :], fs=fr, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=-1)
    return f, C, Pxy


def band_mean(f, C, band):
    k = (f >= band[0]) & (f <= band[1])
    return C[..., k].mean(-1)


def slow_lag(f, Pxy):
    """Lag (s) of y relative to X from the cross-spectrum phase, weighted over the slow band.
    csd(x, y) phase = phase_y - phase_x; y delayed by tau gives phase -2 pi f tau."""
    k = (f >= BANDS["slow"][0]) & (f <= BANDS["slow"][1])
    ph = np.angle(Pxy[..., k])
    w = np.abs(Pxy[..., k])
    tau = -ph / (2 * np.pi * f[k])
    return (tau * w).sum(-1) / (w.sum(-1) + 1e-12)


def onsets(z, fr, thr=3.0, refr_s=3.0):
    above = z > thr
    idx = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= refr_s * fr:
            out.append(i)
            last = i
    return np.array(out, int)


def rz(x):
    med = np.median(x, 0)
    return (x - med) / (np.median(np.abs(x - med), 0) * 1.4826 + 1e-9)


def triggered(y, on, fr, pre=3.0, post=6.0):
    a, b = int(pre * fr), int(post * fr)
    on = on[(on >= a) & (on < len(y) - b)]
    if on.size == 0:
        return None, 0
    seg = np.stack([y[i - a:i + b + 1] - y[i - a:i].mean() for i in on])
    return seg, on.size


def load(r: Run, source: str):
    p = r.out / "traces" / ("dff_auto.csv" if source == "auto" else "dff_human_sameextractor.csv")
    if not p.exists() or (source == "human" and not human_masks_valid(r)):
        return None
    df = pd.read_csv(p)
    names = [c for c in df.columns if c.startswith("dend_")]
    return df, names


def analyze(key: str, source: str) -> dict | None:
    r = get_run(key)
    got = load(r, source)
    if got is None:
        return None
    df, names = got
    t = df["time_s"].to_numpy()
    fr = r.frame_rate
    M = df[names].to_numpy(float)
    g = pd.read_csv(r.out / "traces" / "global_ca.csv")
    gy = resample_to(g["time_s"].to_numpy(), g["global_dff"].to_numpy(), t)
    gy = np.where(np.isfinite(gy), gy, np.nanmedian(gy))
    beh = load_behavior(r)
    B = {}
    for k in BEHS:
        if beh.get(k) is not None:
            y = resample_to(beh[f"{k}_t"], beh[k], t)
            if np.isfinite(y).mean() > 0.9:
                B[k] = np.where(np.isfinite(y), y, np.nanmedian(y))
    if not B:
        return None
    rng = np.random.default_rng(0)
    ms = int(10 * fr)
    shifts = rng.integers(ms, len(t) - ms, N_SHIFT)
    run = dict(run=key, source=source, mouse=r.mouse, fov=fov_group(r), n_dendrites=len(names))
    dend = pd.DataFrame(dict(run=key, source=source, fov=fov_group(r), dendrite=names))
    spectra = {}
    X = np.c_[gy, M]
    for k, y in B.items():
        f, C, Pxy = coh(X, y, fr)
        null = np.stack([coh(X, np.roll(y, s), fr)[1] for s in shifts])         # (S, N+1, F)
        spectra[k] = dict(f=f, global_=C[0], null_lo=np.percentile(null[:, 0], 2.5, 0),
                          null_hi=np.percentile(null[:, 0], 97.5, 0), null_mean=null[:, 0].mean(0),
                          dend_mean=C[1:].mean(0), dend_null=null[:, 1:].mean((0, 1)))
        lag = slow_lag(f, Pxy)
        ks = (f >= BANDS["slow"][0]) & (f <= BANDS["slow"][1])
        phase = np.angle(Pxy[:, ks].sum(-1))          # |phase| < 90 deg = in phase (positive coupling)
        for bn, band in BANDS.items():
            obs = band_mean(f, C, band)
            nb = band_mean(f, null, band)                                         # (S, N+1)
            p = (1 + (nb >= obs[None]).sum(0)) / (N_SHIFT + 1)
            q = bh(p[1:])
            run[f"global_coh_{k}_{bn}"] = float(obs[0])
            run[f"global_coh_{k}_{bn}_excess"] = float(obs[0] - nb[:, 0].mean())
            run[f"global_coh_{k}_{bn}_p"] = float(p[0])
            run[f"dend_coh_{k}_{bn}_excess"] = float((obs[1:] - nb[:, 1:].mean(0)).mean())
            run[f"frac_dend_coh_{k}_{bn}"] = float((q < 0.05).mean())
            dend[f"coh_{k}_{bn}"] = obs[1:]
            dend[f"q_{k}_{bn}"] = q
        run[f"global_lag_{k}_slow_s"] = float(lag[0])
        run[f"global_inphase_{k}_slow"] = bool(abs(phase[0]) < np.pi / 2)
        dend[f"inphase_{k}_slow"] = np.abs(phase[1:]) < np.pi / 2
        sigs = dend[f"q_{k}_slow"].to_numpy() < 0.05 if f"q_{k}_slow" in dend else None
        dend[f"lag_{k}_slow_s"] = lag[1:]
        sig = dend[f"q_{k}_slow"] < 0.05
        run[f"frac_sig_dend_inphase_{k}_slow"] = float(dend.loc[sig, f"inphase_{k}_slow"].mean()) if sig.any() else np.nan
        run[f"median_lag_{k}_slow_sig_dend_s"] = float(np.median(lag[1:][sig])) if sig.any() else np.nan
        run[f"pearson_global_{k}"] = float(np.corrcoef(gy, y)[0, 1])
    # Ca-event-triggered pupil
    trig = {}
    if "pupil" in B:
        y = B["pupil"]
        yn = (y - np.median(y)) / (np.percentile(y, 95) - np.percentile(y, 5) + 1e-12)   # range-normalised
        zg = rz(gy)
        on_g = onsets(zg, fr)
        Z = rz(M)
        glob = np.convolve(zg > 3, np.ones(int(2 * fr) + 1), "same") > 0
        on_l = []
        for j in range(Z.shape[1]):
            on_l += [i for i in onsets(Z[:, j], fr) if not glob[i]]
        on_l = np.array(sorted(on_l), int)
        post = slice(int(3 * fr) + int(1 * fr), int(3 * fr) + int(4 * fr))     # 1-4 s after onset
        for nm, on in (("global", on_g), ("local", on_l)):
            seg, n = triggered(yn, on, fr)
            if seg is None or n < 3:
                continue
            nullm = []
            for _ in range(500):
                rnd = rng.integers(int(3 * fr), len(y) - int(6 * fr), n)
                nullm.append(triggered(yn, rnd, fr)[0][:, post].mean())
            nullm = np.array(nullm)
            obs = seg[:, post].mean()
            trig[nm] = dict(n=int(n), mean=seg.mean(0), obs=float(obs), fr=float(fr),
                            p=float((1 + (np.abs(nullm - nullm.mean()) >= abs(obs - nullm.mean())).sum()) / 501))
            run[f"pupil_after_{nm}_events"] = float(obs - nullm.mean())
            run[f"pupil_after_{nm}_events_p"] = trig[nm]["p"]
            run[f"n_{nm}_events"] = int(n)
    run_fig(r, source, spectra, trig, fr)
    return dict(run=run, dend=dend, spectra={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in spectra.items()},
                trig={k: dict(v, mean=v["mean"].tolist()) for k, v in trig.items()})


def run_fig(r, source, spectra, trig, fr):
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for a, k in zip(ax[:3], BEHS):
        if k not in spectra:
            a.set_axis_off()
            continue
        s = spectra[k]
        f = s["f"]
        a.fill_between(f, s["null_lo"], s["null_hi"], color="#dddddd", lw=0, label="null 95% (shifted)")
        a.plot(f, s["global_"], color=COL[k], lw=1.5, label="global Ca")
        a.plot(f, s["dend_mean"], color=COL[k], lw=1, ls="--", label="mean over dendrites")
        a.plot(f, s["dend_null"], color="grey", lw=0.8, ls=":", label="dendrite null mean")
        for b in BANDS.values():
            a.axvline(b[1], color="k", lw=0.3)
        a.set_xlim(0, 1.2)
        a.set_ylim(0, 1)
        a.set_xlabel("frequency (Hz)")
        a.set_ylabel("coherence")
        a.set_title(f"Ca vs {k}")
    ax[0].legend(fontsize=6, frameon=False)
    a = ax[3]
    for nm, c in (("global", "k"), ("local", "#2f6fdf")):
        if nm in trig:
            m = np.asarray(trig[nm]["mean"])
            tt = np.arange(m.size) / fr - 3
            a.plot(tt, m, color=c, label=f"{nm} Ca events (n={trig[nm]['n']}, p={trig[nm]['p']:.2g})")
    a.axvline(0, color="k", lw=0.5, ls="--")
    a.set_xlabel("time from Ca event onset (s)")
    a.set_ylabel("pupil (range-normalised)")
    a.set_title("Ca-event-triggered pupil")
    a.legend(fontsize=6, frameon=False)
    fig.suptitle(f"{r.key} ({source} dendrites): coherence with behavior", fontsize=9)
    fig.tight_layout()
    fig.savefig(r.outdir("figures") / f"coherence_{source}.png", dpi=130)
    plt.close(fig)


def cohort(source, res):
    out = OUT / source
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([x["run"] for x in res])
    dend = pd.concat([x["dend"] for x in res], ignore_index=True)
    runs.to_csv(out / "per_run.csv", index=False)
    dend.to_csv(out / "per_dendrite.csv", index=False)
    fov = runs.groupby(["fov", "mouse"]).mean(numeric_only=True).reset_index()
    fov.to_csv(out / "per_fov.csv", index=False)
    lines = [f"Ca-behavior coherence ({source} dendrites): {len(runs)} runs, {fov.fov.nunique()} FOVs, {runs.mouse.nunique()} mice.",
             "Excess = observed band coherence minus circular-shift null mean; tests = Wilcoxon over FOVs.", ""]
    S = {"source": source, "tests": []}
    for k in BEHS:
        for bn in BANDS:
            c = f"global_coh_{k}_{bn}_excess"
            if c not in fov:
                continue
            v = fov[c].dropna()
            p = sp.wilcoxon(v).pvalue if len(v) >= 4 else np.nan
            nsig = int((runs[f"global_coh_{k}_{bn}_p"] < 0.05).sum())
            fd = fov[f"frac_dend_coh_{k}_{bn}"].median()
            de = fov[f"dend_coh_{k}_{bn}_excess"].dropna()
            pd_ = sp.wilcoxon(de).pvalue if len(de) >= 4 else np.nan
            S["tests"].append(dict(behavior=k, band=bn, global_excess_median=float(v.median()), p_fov=float(p),
                                   runs_sig=nsig, n_runs=int(runs[c].notna().sum()),
                                   dend_excess_median=float(de.median()), p_dend_fov=float(pd_), frac_dend_sig_median=float(fd)))
            lines.append(f"{k:8s} {bn:4s} ({BANDS[bn][0]}-{BANDS[bn][1]} Hz): global Ca excess coherence {v.median():+.3f} "
                         f"(p={p:.2g}, {nsig}/{runs[c].notna().sum()} runs p<0.05); dendrites: excess {de.median():+.3f} "
                         f"(p={pd_:.2g}), median {fd * 100:.0f}% of dendrites significant (BH).")
        if f"global_lag_{k}_slow_s" in runs:
            lg = fov[f"global_lag_{k}_slow_s"].dropna()
            ld = fov[f"median_lag_{k}_slow_sig_dend_s"].dropna()
            ip = runs[f"global_inphase_{k}_slow"].mean()
            fi = fov[f"frac_sig_dend_inphase_{k}_slow"].dropna()
            lines.append(f"{k:8s} slow-band lag (+ = {k} follows Ca): global median {lg.median():+.2f} s; "
                         f"significant dendrites median {ld.median():+.2f} s.")
            lines.append(f"{k:8s} slow-band sign: global Ca in phase (positive) in {ip * 100:.0f}% of runs; of the coherent "
                         f"dendrites a median {fi.median() * 100:.0f}% per FOV are in phase, {100 - fi.median() * 100:.0f}% anti-phase.")
        lines.append("")
    for nm in ("global", "local"):
        c = f"pupil_after_{nm}_events"
        if c in fov:
            v = fov[c].dropna()
            p = sp.wilcoxon(v).pvalue if len(v) >= 4 else np.nan
            nsig = int((runs[f"{c}_p"] < 0.05).sum())
            S["tests"].append(dict(test=f"pupil_after_{nm}_Ca_events", median=float(v.median()), p_fov=float(p), runs_sig=nsig))
            lines.append(f"Pupil 1-4 s after {nm} Ca events (vs random onsets, range-normalised): median change {v.median():+.3f} "
                         f"across {len(v)} FOVs (p={p:.2g}); {nsig}/{runs[c].notna().sum()} runs p<0.05.")
    lines.append("")
    pr = fov["pearson_global_pupil"].dropna()
    lines.append(f"For comparison, Pearson r global Ca vs pupil (zero lag): median {pr.median():+.3f} across FOVs.")
    lines += ["", "Caveats: ~106 s per run gives ~7 Welch segments, so single-run coherence is noisy (chance level ~0.2-0.3);",
              "the circular-shift null accounts for that. Frequencies below 0.04 Hz are not resolved. Pupil blinks / lost",
              "tracking appear as sharp dips and add broadband noise (pupil detection was not re-checked here)."]
    (out / "summary.txt").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(S, indent=2))
    # cohort figure: global Ca spectra per FOV (excess over null) per behavior + triggered pupil
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    byfov = {}
    for x in res:
        byfov.setdefault(x["run"]["fov"], []).append(x)
    for a, k in zip(ax[:3], BEHS):
        allc = []
        for fv, xs in byfov.items():
            ss = [x["spectra"][k] for x in xs if k in x["spectra"]]
            if not ss:
                continue
            f = np.asarray(ss[0]["f"])
            ex = np.mean([np.asarray(s["global_"]) - np.asarray(s["null_mean"]) for s in ss], 0)
            exd = np.mean([np.asarray(s["dend_mean"]) - np.asarray(s["dend_null"]) for s in ss], 0)
            a.plot(f, ex, color=COL[k], lw=0.6, alpha=0.5)
            allc.append((f, ex, exd))
        if allc:
            f = allc[0][0]
            same = [c for c in allc if c[0].size == f.size]
            a.plot(f, np.median([c[1] for c in same], 0), color=COL[k], lw=2, label="global Ca (median FOV)")
            a.plot(f, np.median([c[2] for c in same], 0), color="k", lw=1.2, ls="--", label="dendrites (median FOV)")
        a.axhline(0, color="k", lw=0.5)
        a.set_xlim(0, 1.2)
        a.set_xlabel("frequency (Hz)")
        a.set_ylabel("coherence above shift null")
        a.set_title(f"Ca vs {k} (thin: one FOV)")
        a.legend(fontsize=6, frameon=False)
    a = ax[3]
    for nm, c in (("global", "k"), ("local", "#2f6fdf")):
        ms = [np.asarray(x["trig"][nm]["mean"]) for x in res if nm in x["trig"]]
        if ms:
            L = min(m.size for m in ms)
            M = np.stack([m[:L] for m in ms])
            frs = [x["trig"][nm]["fr"] for x in res if nm in x["trig"]]
            tt = np.arange(L) / frs[0] - 3          # runs with another frame rate would need resampling
            if len(set(frs)) > 1:
                print("warning: mixed frame rates in triggered average")
            a.plot(tt, M.mean(0), color=c, lw=1.5, label=f"{nm} Ca events ({len(ms)} runs)")
            a.fill_between(tt, M.mean(0) - M.std(0) / np.sqrt(len(ms)), M.mean(0) + M.std(0) / np.sqrt(len(ms)), color=c, alpha=0.2, lw=0)
    a.axvline(0, color="k", lw=0.5, ls="--")
    a.set_xlabel("time from Ca event onset (s)")
    a.set_ylabel("pupil (range-normalised), mean +- SEM over runs")
    a.set_title("Ca-event-triggered pupil")
    a.legend(fontsize=6, frameon=False)
    fig.suptitle(f"Ca-behavior coherence, {source} dendrites", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_coherence.png", dpi=140)
    fig.savefig(out / "fig_coherence.pdf")
    plt.close(fig)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("auto", "human", "both"), default="both")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="all runs (default when no --run)")
    ap.add_argument("--force", action="store_true", help="accepted for CLI symmetry; always recomputed")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    keys = a.run or [r.key for r in discover_runs()]
    t0 = time.time()
    summ = {}
    for s in (("auto", "human") if a.source == "both" else (a.source,)):
        res = []
        with ProcessPoolExecutor(max(1, min(a.jobs, 6))) as ex:
            futs = {ex.submit(analyze, k, s): k for k in keys}
            for f in as_completed(futs):
                try:
                    x = f.result()
                    if x is not None:
                        res.append(x)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("ERROR", futs[f], e)
        res.sort(key=lambda x: x["run"]["run"])
        summ[s] = cohort(s, res)
        print("\n".join(summ[s]))
        print()
    if len(summ) == 2:
        (OUT / "compare_auto_vs_human.txt").write_text(
            "AUTO\n" + "\n".join(summ["auto"]) + "\n\nHUMAN\n" + "\n".join(summ["human"]))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
