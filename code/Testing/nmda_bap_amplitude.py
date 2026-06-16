#!/usr/bin/env python
"""
NMDA-spike vs bAP "vertical streak" AMPLITUDE test  (Tests 1 & 3).

Hypothesis (user):
  NMDA-spike Ca transients (LOCALIZED to a branch) have LOWER amplitude than
  bAP "vertical streak" transients (signal EXTENDED along the apical trunk =
  along cortical depth Y; tend to co-activate a large vertical extent / whole tree).

Two cheap tests from EXISTING curated masks + ΔF/F trace CSVs (no re-extraction):

  Test 3 (morphology -> amplitude, per mask):
    Classify each curated mask by shape:
      vertical-streak-like (bAP candidate) = long + vertical (principal axis along
        cortical depth Y) + linear (single trunk segment)
      localized/branch-like (NMDA candidate) = short OR off-vertical OR low-linearity
    Compare per-mask amplitude (peak ΔF/F, mean event amplitude).
    Also continuous Spearman: amplitude vs length / verticality / y-extent / linearity.

  Test 1 (spatial extent -> amplitude, per event):
    For every detected event, participation = fraction of masks co-active (±1 frame).
    bAP/streak ≈ high participation (whole-tree, vertical); NMDA ≈ isolated (1 mask).
    Compare event amplitude (peak ΔF/F) of isolated vs whole-tree events, and
    correlate amplitude vs participation.

Outputs:
  scape-data/nmda_bap_mask_metrics.csv   (per-mask morphology+amplitude+FOV)
  scape-data/nmda_bap_event_table.csv    (per-event amplitude+participation+FOV)
  scape-data/figures/nmda_bap_amplitude.png

NOTE: trace CSVs are ΔF/F in PERCENT (M4 multiplies by 100). Amplitudes reported in %.
NOTE: amplitude across FOVs is confounded (expression/imaging) -> we report per-FOV
      stats and pool using within-FOV ranks / z-scores, not raw cross-FOV pooling.
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
VOXEL = np.array([3.9, 1.0, 1.2])      # Z,Y,X um ; Y = cortical depth (surface at top)

# event detection (on % ΔF/F traces)
K_MAD   = 3.5      # threshold = median + K*1.4826*MAD
FLOOR   = 1.5      # absolute floor (%) so flat traces don't fire on noise
MIN_DUR = 2        # frames above threshold

# morphology class thresholds
L_LONG   = 40.0    # um : "long"
L_SHORT  = 25.0    # um : "short"
ORI_VERT = 30.0    # deg from cortical-depth Y axis : "vertical"
ORI_LAT  = 50.0    # deg : "lateral/off-vertical"
LIN_HI   = 0.70    # linearity : "linear single segment"
LIN_LO   = 0.50

# (label, date, mouse, mask_run, [trace runs sharing those masks], FS)
FOVS = [
    ("0508",     "2026-05-08", "rbp4_139_phpeb",   "run5", ["run5","run6"],          5.0),
    ("0512_r56", "2026-05-12", "rbp4_132_phpeb",   "run5", ["run5","run6"],          5.0),
    ("0512_r910","2026-05-12", "rbp4_132_phpeb",   "run9", ["run9","run10"],         5.0),
    ("0416_r123","2026-04-16", "rbp4_132_phpeb",   "run1", ["run1","run2","run3"],   5.0),
    ("0416_r567","2026-04-16", "rbp4_132_phpeb",   "run5", ["run5","run6","run7"],   5.0),
    ("0331_r67", "2026-03-31", "rbp4_132_phpeb",   "run7", ["run6","run7"],          5.0),
    ("0331_r8910","2026-03-31","rbp4_132_phpeb",   "run8", ["run8","run9","run10"],  5.0),
    ("0320_r1",  "2026-03-20", "rbp4cre_139_phpeb","run1", ["run1"],                 5.0),
    ("0320_r3",  "2026-03-20", "rbp4cre_139_phpeb","run3", ["run3"],                 5.0),
    ("136_1202", "2025-12-02", "rbp4cre_136_phpeb","run4", ["run4"],                 6.0),
    ("136_0209", "2026-02-09", "rbp4cre_136_phpeb","run1", ["run1"],                 6.0),
    ("138_0217", "2026-02-17", "rbp4cre_138_phpeb","run7", ["run7"],                 6.0),
]

# ---------------- morphology ----------------
def morphology(mask):
    coords = np.argwhere(mask)                      # (N,3) ZYX voxels
    n = len(coords)
    if n == 0:
        return dict(n_vox=0, vol_um3=0.0, depth_um=np.nan, y_extent_um=0.0,
                    length_um=0.0, linearity=0.0, orient_deg=np.nan, vert_frac=np.nan,
                    shape_class="empty")
    pts = coords * VOXEL                            # um
    c = pts.mean(0)
    y_extent = (coords[:,1].max() - coords[:,1].min() + 1) * VOXEL[1]
    out = dict(n_vox=n, vol_um3=n*float(np.prod(VOXEL)), depth_um=float(c[1]),
               y_extent_um=float(y_extent))
    if n < 6:
        out.update(length_um=0.0, linearity=0.0, orient_deg=np.nan, vert_frac=np.nan,
                   shape_class="tiny")
        return out
    cov = np.cov((pts - c).T)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w[::-1], 1e-9, None); V = V[:, ::-1]
    pa = V[:, 0]
    if pa[1] < 0: pa = -pa
    proj = (pts - c) @ pa
    length_um = float(proj.max() - proj.min())
    linearity = float((w[0] - w[1]) / w[0])
    orient_deg = float(np.degrees(np.arccos(np.clip(abs(pa[1]), 0, 1))))  # 0=vertical(depth),90=lateral
    out.update(length_um=length_um, linearity=linearity, orient_deg=orient_deg,
               vert_frac=float(abs(pa[1])))         # |cos| of axis with Y: 1=vertical streak
    out["shape_class"] = ("tiny" if n < 6 else
                          "blob" if linearity < 0.4 else
                          "long" if length_um >= L_LONG else "short")
    return out

def streak_class(mo):
    """vertical-streak (bAP cand) / localized (NMDA cand) / mid."""
    if mo["shape_class"] == "tiny" or np.isnan(mo["orient_deg"]):
        return "localized"
    long_v = mo["length_um"] >= L_LONG
    short_v = mo["length_um"] < L_SHORT
    vert  = mo["orient_deg"] <= ORI_VERT
    lat   = mo["orient_deg"] >  ORI_LAT
    lin   = mo["linearity"]  >= LIN_HI
    nonlin= mo["linearity"]  <  LIN_LO
    if long_v and vert and lin:
        return "vertical_streak"
    if short_v or lat or nonlin:
        return "localized"
    return "mid"

# ---------------- events ----------------
def thresh(tr):
    m = np.median(tr); mad = np.median(np.abs(tr - m)) * 1.4826
    return max(FLOOR, m + K_MAD * mad)

def event_peaks(tr, thr=None):
    if thr is None: thr = thresh(tr)
    idx = np.where(tr > thr)[0]
    pk = []
    if idx.size:
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if run.size >= MIN_DUR:
                pk.append(int(run[np.argmax(tr[run])]))
    return pk

# ---------------- per-FOV driver ----------------
def load_traces(base, run, names):
    f = base/run/"traces"/"dff_traces_curated_bgsub.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
    if not all(n in df.columns for n in names): return None
    return df[names].astype(float)

def analyze_fov(label, date, mouse, mask_run, runs, fs):
    base = PR/"scape-data"/date/mouse
    mpaths = sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))
    names = [p.stem.replace("_labelmap","") for p in mpaths]
    morphs = {nm: morphology(tifffile.imread(p) > 0) for nm, p in zip(names, mpaths)}

    dfs = [d for d in (load_traces(base, r, names) for r in runs) if d is not None]
    if not dfs:
        print(f"[{label}] no usable trace runs"); return None, None
    N = len(names)

    # per-mask amplitude pooled across runs
    agg = {nm: dict(peaks=[], peak_max=0.0) for nm in names}
    ev_rows = []
    for df in dfs:
        thr = {nm: thresh(df[nm].values) for nm in names}
        act = np.column_stack([df[nm].values > thr[nm] for nm in names])  # (T,N)
        for i, nm in enumerate(names):
            tr = df[nm].values
            for p in event_peaks(tr, thr[nm]):
                amp = float(tr[p])
                agg[nm]["peaks"].append(amp)
                agg[nm]["peak_max"] = max(agg[nm]["peak_max"], amp)
                lo, hi = max(0, p-1), min(len(tr), p+2)
                participation = float(act[lo:hi].any(0).sum()) / N
                ev_rows.append(dict(fov=label, name=nm, amp=amp,
                                    participation=participation,
                                    n_coactive=int(round(participation*N))))
    # mask rows
    mrows = []
    for nm in names:
        mo = morphs[nm]; pk = agg[nm]["peaks"]
        mrows.append(dict(fov=label, name=nm, fs=fs,
            **{k: mo[k] for k in ['n_vox','vol_um3','depth_um','y_extent_um',
                                  'length_um','linearity','orient_deg','vert_frac','shape_class']},
            streak_class=streak_class(mo),
            n_events=len(pk),
            peak_dff=agg[nm]["peak_max"],
            mean_amp=float(np.mean(pk)) if pk else np.nan,
            med_amp=float(np.median(pk)) if pk else np.nan))
    mdf = pd.DataFrame(mrows); edf = pd.DataFrame(ev_rows)

    # quick per-FOV report
    vs = mdf[mdf.streak_class=="vertical_streak"]; lo = mdf[mdf.streak_class=="localized"]
    msg = f"[{label}] N={N} verticalStreak={len(vs)} localized={len(lo)} mid={ (mdf.streak_class=='mid').sum() }"
    print(msg)
    return mdf, edf

def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 6 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0: return np.nan, np.nan, int(m.sum())
    r, p = stats.spearmanr(x[m], y[m]); return r, p, int(m.sum())

def main():
    all_m, all_e = [], []
    for label, d, mo, mr, runs, fs in FOVS:
        mdf, edf = analyze_fov(label, d, mo, mr, runs, fs)
        if mdf is not None: all_m.append(mdf); all_e.append(edf)
    M = pd.concat(all_m, ignore_index=True); E = pd.concat(all_e, ignore_index=True)
    (PR/"scape-data").mkdir(exist_ok=True)
    M.to_csv(PR/"code"/"Testing"/"data"/"nmda_bap_mask_metrics.csv", index=False)
    E.to_csv(PR/"code"/"Testing"/"data"/"nmda_bap_event_table.csv", index=False)

    # ---- TEST 3: morphology -> amplitude (per FOV Spearman, then summarise) ----
    print("\n================ TEST 3: morphology -> per-mask amplitude ================")
    print("Prediction: amplitude HIGHER for long / vertical / high y-extent / linear masks.")
    feats = [("length_um", +1), ("vert_frac", +1), ("y_extent_um", +1),
             ("orient_deg", -1), ("linearity", +1)]
    for feat, _exp in feats:
        rs = []
        for fov, g in M.groupby("fov"):
            r, p, n = sp(g[feat].values, g["peak_dff"].values)
            if np.isfinite(r): rs.append(r)
        rs = np.array(rs)
        if rs.size:
            print(f"  peak_dff vs {feat:12s}: median Spearman r={np.median(rs):+.2f} "
                  f"({(rs>0).sum()}/{rs.size} FOVs positive)  per-FOV r={np.round(rs,2).tolist()}")
    # group compare vertical_streak vs localized (within-FOV z-scored peak)
    M["peak_z"] = M.groupby("fov")["peak_dff"].transform(lambda s: (s - s.mean())/(s.std()+1e-9))
    vs = M[M.streak_class=="vertical_streak"]["peak_z"].dropna()
    lo = M[M.streak_class=="localized"]["peak_z"].dropna()
    if len(vs) > 3 and len(lo) > 3:
        u, pu = stats.mannwhitneyu(vs, lo, alternative="greater")
        print(f"\n  vertical_streak vs localized (within-FOV z-scored peak ΔF/F):")
        print(f"    vertical_streak n={len(vs)} median_z={np.median(vs):+.2f} | "
              f"localized n={len(lo)} median_z={np.median(lo):+.2f}")
        print(f"    Mann-Whitney (streak>localized) p={pu:.3g}")

    # ---- TEST 1: spatial extent -> amplitude (per event) ----
    print("\n================ TEST 1: co-activation extent -> event amplitude ============")
    print("Prediction: whole-tree/high-participation (bAP streak) events HIGHER amplitude")
    print("            than isolated single-mask (NMDA) events.")
    E["amp_z"] = E.groupby("fov")["amp"].transform(lambda s: (s - s.mean())/(s.std()+1e-9))
    rs = []
    for fov, g in E.groupby("fov"):
        r, p, n = sp(g["participation"].values, g["amp"].values)
        if np.isfinite(r):
            rs.append(r)
            print(f"  [{fov}] amp vs participation: r={r:+.2f} p={p:.2g} (n_events={n}, "
                  f"max_participation={g.participation.max():.2f})")
    rs = np.array(rs)
    if rs.size:
        print(f"  -> median Spearman r={np.median(rs):+.2f} ({(rs>0).sum()}/{rs.size} FOVs positive)")
    iso = E[E.n_coactive<=1]["amp_z"].dropna()
    multi = E[E.participation>=0.17]["amp_z"].dropna()   # CUT=0.17 (whole-tree, per project)
    if len(iso) > 3 and len(multi) > 3:
        u, pu = stats.mannwhitneyu(multi, iso, alternative="greater")
        print(f"\n  whole-tree (participation>=0.17) vs isolated (1 mask) [within-FOV z amp]:")
        print(f"    whole-tree n={len(multi)} median_z={np.median(multi):+.2f} | "
              f"isolated n={len(iso)} median_z={np.median(iso):+.2f}")
        print(f"    Mann-Whitney (wholeTree>isolated) p={pu:.3g}")

    make_fig(M, E)
    print("\nSaved: scape-data/nmda_bap_mask_metrics.csv, nmda_bap_event_table.csv, "
          "figures/nmda_bap_amplitude.png")

def _scatter(ax, x, y, xl, yl, c='steelblue'):
    ax.scatter(x, y, s=12, alpha=.4, color=c, edgecolor='none')
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)
    r, p, n = sp(np.asarray(x,float), np.asarray(y,float))
    if np.isfinite(r):
        ax.text(.04,.95,f"ρ={r:+.2f}\np={p:.2g}", transform=ax.transAxes, va='top',
                fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=.8))

def make_fig(M, E):
    (PR/"code"/"Testing"/"figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    _scatter(ax[0,0], M.length_um, M.peak_dff, "mask length (um)", "peak ΔF/F (%)")
    _scatter(ax[0,1], M.orient_deg, M.peak_dff, "orient from depth-Y (deg, 0=vertical)", "peak ΔF/F (%)")
    _scatter(ax[0,2], M.y_extent_um, M.peak_dff, "vertical extent (um, along depth)", "peak ΔF/F (%)")
    # box vertical_streak vs localized (z)
    order = ["localized","mid","vertical_streak"]
    data = [M[M.streak_class==c]["peak_z"].dropna().values for c in order]
    ax[1,0].boxplot(data, labels=[f"{c}\n(n={len(d)})" for c,d in zip(order,data)], showfliers=False)
    ax[1,0].axhline(0, color='k', lw=.6); ax[1,0].set_ylabel("peak ΔF/F (within-FOV z)")
    ax[1,0].set_title("Test 3: mask morphology class"); ax[1,0].grid(alpha=.3)
    # event amp vs participation
    _scatter(ax[1,1], E.participation, E.amp, "event participation (frac masks co-active)",
             "event peak ΔF/F (%)", c='indianred')
    ax[1,1].set_title("Test 1: extent vs amplitude")
    # box isolated vs whole-tree (z)
    iso = E[E.n_coactive<=1]["amp_z"].dropna().values
    mid = E[(E.participation>0)&(E.participation<0.17)&(E.n_coactive>1)]["amp_z"].dropna().values
    multi = E[E.participation>=0.17]["amp_z"].dropna().values
    bx = [iso, mid, multi]
    ax[1,2].boxplot(bx, labels=[f"isolated\n(n={len(iso)})", f"small\n(n={len(mid)})",
                                f"whole-tree\n(n={len(multi)})"], showfliers=False)
    ax[1,2].axhline(0, color='k', lw=.6); ax[1,2].set_ylabel("event amp (within-FOV z)")
    ax[1,2].set_title("Test 1: isolated (NMDA) vs whole-tree (bAP)"); ax[1,2].grid(alpha=.3)
    fig.suptitle("NMDA (localized) vs bAP (vertical-streak/whole-tree) amplitude — Tests 1 & 3",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PR/"code"/"Testing"/"figures"/"nmda_bap_amplitude.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
