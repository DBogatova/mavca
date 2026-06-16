#!/usr/bin/env python
"""
Per-mask morphology x activity x within-mask propagation analysis.

For each curated dendrite mask computes:
  morphology/location : volume, principal-axis length, elongation(linearity),
                        orientation vs cortical depth (Y), centroid depth, shape class
  activity/amplitude  : event count/rate, peak dFF, mean event amplitude, active fraction
  propagation         : along the mask's long axis, does Ca peak first superficial or
                        deep? signed speed (um/s) + per-event peak-time spread (frames)

Then correlates metrics and saves plots + a per-mask CSV.

NOTE on temporal resolution: FS=5 Hz -> 200 ms/frame. Within-mask propagation over
tens of um is typically sub-frame, so most masks read as "uniform"; we quantify how
many show a resolvable (>=1 frame) gradient.
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
RUNS = [("2026-04-16", "rbp4_132_phpeb", "run1"),
        ("2026-05-08", "rbp4_139_phpeb", "run5")]
VOXEL = np.array([3.9, 1.0, 1.2])      # Z,Y,X um ; Y = cortical depth
FS = 5.0
NBINS = 6                              # bins along mask long axis
DFF_THR = 1.5                          # event detection on dFF trace
MIN_DUR = 2                            # min frames above threshold
WIN = 3                                # +/- frames around event peak for timing
MIN_BIN_VOX = 3                        # min voxels for a bin to count in timing

# ---------- morphology ----------
def align(m, Z, Y, X):
    """Match M4: crop bottom (deepest) Y rows if taller, pad bottom if shorter."""
    my = m.shape[1]
    if my > Y:   m = m[:, :-(my - Y), :]
    elif my < Y: m = np.pad(m, ((0,0),(0,Y-my),(0,0)), mode='constant')
    return m

def morphology(mask):
    coords = np.argwhere(mask)                     # (N,3) voxel ZYX
    n = len(coords)
    pts = coords * VOXEL                            # um
    c = pts.mean(0)
    out = dict(n_vox=n, vol_um3=n*float(np.prod(VOXEL)),
               depth_um=c[1], cz=c[0], cy=c[1], cx=c[2])
    if n < 6:
        out.update(length_um=0, linearity=0, orient_deg=np.nan, shape_class="tiny",
                   pa=np.array([0,1,0.]))
        return out, coords, pts
    cov = np.cov((pts - c).T)
    w, V = np.linalg.eigh(cov)                       # ascending
    w = np.clip(w[::-1], 1e-9, None); V = V[:, ::-1] # descending
    pa = V[:, 0]
    if pa[1] < 0: pa = -pa                            # orient +axis toward deep (+Y)
    proj = (pts - c) @ pa
    length_um = proj.max() - proj.min()
    linearity = (w[0] - w[1]) / w[0]
    orient_deg = np.degrees(np.arccos(np.clip(abs(pa[1]), 0, 1)))  # 0=radial,90=tangential
    if linearity < 0.4:        shape = "blob"
    elif length_um >= 40:      shape = "long"
    else:                      shape = "short"
    out.update(length_um=length_um, linearity=linearity, orient_deg=orient_deg,
               shape_class=shape, pa=pa)
    return out, coords, pts

# ---------- activity ----------
def events(trace):
    above = trace > DFF_THR
    idx = np.where(above)[0]
    peaks = []
    if idx.size:
        splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for run in splits:
            if run.size >= MIN_DUR:
                peaks.append(run[np.argmax(trace[run])])
    return peaks

def activity(trace):
    pk = events(trace)
    amps = [trace[p] for p in pk]
    dur_min = len(trace)/FS/60
    return dict(n_events=len(pk), event_rate=len(pk)/dur_min if dur_min else 0,
                peak_dff=float(np.nanmax(trace)) if trace.size else 0,
                mean_amp=float(np.mean(amps)) if amps else 0,
                active_frac=float(np.mean(trace > DFF_THR))), pk

# ---------- within-mask propagation ----------
def bin_traces(stack_path, mask_list, morph_list, T):
    """Return per (mask,bin) mean trace and bin axis-positions (um)."""
    Z, Y, X = mask_list[0].shape
    grp = np.zeros((Z, Y, X), np.int32)             # group id volume
    binpos = {}                                     # im -> array of bin positions (um)
    for im, (m, mo) in enumerate(zip(mask_list, morph_list)):
        coords = np.argwhere(m)
        if len(coords) < MIN_BIN_VOX: 
            binpos[im] = None; continue
        proj = ((coords*VOXEL) - np.array([mo['cz'],mo['cy'],mo['cx']])) @ mo['pa']
        lo, hi = proj.min(), proj.max()
        b = np.clip(((proj-lo)/(hi-lo+1e-9)*NBINS).astype(int), 0, NBINS-1)
        gid = im*NBINS + b + 1
        grp[coords[:,0], coords[:,1], coords[:,2]] = gid
        pos = np.array([proj[b==k].mean() if (b==k).any() else np.nan for k in range(NBINS)])
        binpos[im] = pos
    NG = len(mask_list)*NBINS
    sums = np.zeros((NG+1, T), np.float64)
    arr = tifffile.memmap(str(stack_path))          # (T,Z,Y,X) lazy
    gflat = grp.ravel()
    for t in range(T):
        sums[:, t] = np.bincount(gflat, weights=np.asarray(arr[t]).ravel().astype(np.float64),
                                 minlength=NG+1)
    counts = np.bincount(gflat, minlength=NG+1).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = sums / counts[:, None]
    return mean[1:], counts[1:], binpos       # mean: (NG,T)

def event_speeds(bins, pos, peaks):
    """Per-event signed axial speed (um/s, + = deep) and peak-time spread (frames)."""
    speeds, spreads = [], []
    for p in peaks:
        a, b = max(0, p-WIN), min(bins.shape[1], p+WIN+1)
        valid = [k for k in range(NBINS) if np.isfinite(pos[k]) and
                 np.nanmax(bins[k, a:b]) > 0 and np.ptp(bins[k, a:b]) > 0]
        if len(valid) < 3: continue
        tk = np.array([a + np.argmax(bins[k, a:b]) for k in valid]) / FS
        pk = np.array([pos[k] for k in valid])
        if np.ptp(tk) == 0:
            speeds.append(0.0); spreads.append(0.0); continue
        speeds.append(np.polyfit(tk, pk, 1)[0]); spreads.append(np.ptp(tk)*FS)
    return speeds, spreads

def classify_speeds(speeds, spreads):
    if len(speeds) < 2:
        return dict(prop_dir="n/a", n_prop=len(speeds), med_speed=np.nan,
                    med_spread=np.nan, consistency=np.nan)
    sp = np.array(speeds)
    med = float(np.median(sp)); spread = float(np.median(spreads))
    nz = sp[np.abs(sp) > 1.0]
    consistency = float(np.mean(np.sign(nz) == np.sign(med))) if nz.size else 0.0
    directional = (spread >= 1) and (abs(med) > 1.0) and (consistency >= 0.6)
    d = "uniform" if not directional else ("toward_deep" if med > 0 else "toward_surface")
    return dict(prop_dir=d, n_prop=len(speeds), med_speed=med,
                med_spread=spread, consistency=consistency)

def propagation(im, bins, pos, peaks):
    return classify_speeds(*event_speeds(bins, pos, peaks))

# ---------- per-run driver ----------
def analyze_run(date, mouse, run):
    base = PROJECT_ROOT/"scape-data"/date/mouse/run
    mask_dir = base/"labelmaps_curated_dynamic"
    tcsv = base/"traces"/"dff_traces_curated_bgsub.csv"
    stack = base/"preprocessed"/"stack_voxel_norm_mean_sub.tif"
    out = base/"mask_analysis"; out.mkdir(exist_ok=True)
    paths = sorted(mask_dir.glob("dend_*_labelmap.tif"))
    names = [p.stem.replace("_labelmap","") for p in paths]
    masks = [tifffile.imread(p) > 0 for p in paths]
    df = pd.read_csv(tcsv)
    if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
    sshape = tifffile.TiffFile(str(stack)).series[0].shape if stack.exists() else None
    T = min(len(df), sshape[0]) if sshape else len(df)
    if sshape:                                   # align masks to stack grid (M4 convention)
        Z, Y, X = sshape[1:]
        masks = [align(m, Z, Y, X) for m in masks]

    rows, morphs, peaks_all = [], [], []
    for nm, m in zip(names, masks):
        mo, _, _ = morphology(m)
        tr = df[nm].values[:T] if nm in df.columns else np.zeros(T)
        ac, pk = activity(tr)
        morphs.append(mo); peaks_all.append(pk)
        rows.append({**{k: mo[k] for k in
                        ['n_vox','vol_um3','depth_um','length_um','linearity','orient_deg','shape_class']},
                     'name': nm, **ac})
    # propagation (needs aligned stack)
    prop = {}
    if stack.exists():
        mean, counts, binpos = bin_traces(stack, masks, morphs, T)
        for im, nm in enumerate(names):
            if binpos[im] is None:
                prop[nm] = dict(prop_dir="n/a", n_prop=0, med_speed=np.nan,
                                med_spread=np.nan, consistency=np.nan)
            else:
                bins = mean[im*NBINS:(im+1)*NBINS]
                prop[nm] = propagation(im, bins, binpos[im], peaks_all[im])
    for r in rows:
        r.update(prop.get(r['name'], dict(prop_dir="n/a", n_prop=0,
                                          med_speed=np.nan, med_spread=np.nan,
                                          consistency=np.nan)))
    res = pd.DataFrame(rows)
    res.insert(0, 'run', f"{date}/{mouse}/{run}")
    res.to_csv(out/"mask_metrics.csv", index=False)
    plot_run(res, df, out, f"{mouse}/{run}")
    return res

# ---------- plots ----------
def _scatter(ax, x, y, xl, yl):
    ax.scatter(x, y, s=18, alpha=.6, color='steelblue', edgecolor='none')
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() > 2 and np.ptp(x[m]) > 0:
        r, p = stats.pearsonr(x[m], y[m])
        ax.text(.04, .92, f"r={r:.2f}\np={p:.3f}", transform=ax.transAxes,
                va='top', fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=.8))

def plot_run(res, traces, out, title):
    a = res
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    _scatter(ax[0,0], a.length_um.values, a.n_events.values, "length (um)", "n events")
    _scatter(ax[0,1], a.vol_um3.values, a.peak_dff.values, "volume (um^3)", "peak dF/F")
    _scatter(ax[0,2], a.depth_um.values, a.event_rate.values, "depth Y (um)", "events/min")
    _scatter(ax[1,0], a.depth_um.values, a.peak_dff.values, "depth Y (um)", "peak dF/F")
    _scatter(ax[1,1], a.linearity.values, a.mean_amp.values, "linearity", "mean event amp")
    # shape class vs activity
    order = ["blob","short","long","tiny"]
    cats = [c for c in order if (a['shape_class']==c).any()]
    ax[1,2].boxplot([a.peak_dff[a['shape_class']==c].values for c in cats], labels=cats)
    ax[1,2].set_xlabel("shape"); ax[1,2].set_ylabel("peak dF/F"); ax[1,2].grid(alpha=.3)
    fig.suptitle(f"Morphology x Activity — {title}", fontsize=12)
    fig.tight_layout(); fig.savefig(out/"correlations.png", dpi=160); plt.close(fig)

    # propagation panel
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    pr = a[a.n_prop > 0]
    dirs = pr.prop_dir.value_counts()
    ax[0].bar(dirs.index, dirs.values, color='indianred')
    ax[0].set_title("within-mask propagation direction"); ax[0].set_ylabel("n masks")
    for t in ax[0].get_xticklabels(): t.set_rotation(20)
    _scatter(ax[1], pr.depth_um.values, pr.med_speed.values, "depth Y (um)", "signed speed (um/s, +deep)")
    ax[1].axhline(0, color='k', lw=.6)
    _scatter(ax[2], pr.length_um.values, pr.med_spread.values, "length (um)", "peak spread (frames)")
    ax[2].axhline(1, color='r', lw=.8, ls='--')  # 1-frame resolvability line
    fig.suptitle(f"Within-mask propagation — {title}  (>=1 frame = resolvable)", fontsize=12)
    fig.tight_layout(); fig.savefig(out/"propagation.png", dpi=160); plt.close(fig)

    # trace correlation matrix
    if traces is None:
        return
    cm = traces.corr().values
    iu = np.triu_indices_from(cm, 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, vmin=-.3, vmax=.3, cmap='RdBu_r')
    ax.set_title(f"trace corr — mean off-diag r={np.nanmean(cm[iu]):.3f}")
    fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout()
    fig.savefig(out/"trace_corr.png", dpi=160); plt.close(fig)

# ---------- group (shared-mask) driver ----------
# (label, date, mouse, mask_run, [runs to pool])  -- masks are shared across the runs
GROUPS = [
    ("0416_run123", "2026-04-16", "rbp4_132_phpeb", "run1", ["run1","run2","run3"]),
    ("0416_run567", "2026-04-16", "rbp4_132_phpeb", "run5", ["run5","run6","run7"]),
    ("0331_run67",  "2026-03-31", "rbp4_132_phpeb", "run7", ["run6","run7"]),
    ("0331_run8910","2026-03-31", "rbp4_132_phpeb", "run8", ["run8","run9","run10"]),
    ("0508_run56",  "2026-05-08", "rbp4_139_phpeb", "run5", ["run5","run6"]),
    ("0512_run56",  "2026-05-12", "rbp4_132_phpeb", "run5", ["run5","run6"]),
    ("0512_run910", "2026-05-12", "rbp4_132_phpeb", "run9", ["run9","run10"]),
]

def analyze_group(label, date, mouse, mask_run, runs):
    base = PROJECT_ROOT/"scape-data"/date/mouse
    paths = sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))
    if not paths:
        print(f"  [{label}] no masks in {mask_run}"); return None
    names = [p.stem.replace("_labelmap","") for p in paths]
    masks = [tifffile.imread(p) > 0 for p in paths]; nmask = len(masks)
    # usable runs: trace exists and matches mask count
    used = []
    for r in runs:
        tc = base/r/"traces"/"dff_traces_curated_bgsub.csv"
        if not tc.exists(): continue
        df = pd.read_csv(tc)
        if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
        df = df[[c for c in df.columns if 'dend' in c]]
        if df.shape[1] != nmask:
            print(f"  [{label}] skip {r}: {df.shape[1]} trace dends != {nmask} masks"); continue
        st = base/r/"preprocessed"/"stack_voxel_norm_mean_sub.tif"
        used.append((r, df, st if st.exists() else None))
    if not used:
        print(f"  [{label}] no usable trace runs"); return None
    sshape = next((tifffile.TiffFile(str(st)).series[0].shape for _,_,st in used if st), None)
    if sshape:
        Z,Y,X = sshape[1:]; masks = [align(m,Z,Y,X) for m in masks]
    morphs = [morphology(m)[0] for m in masks]
    # pooled activity + per-run trace correlation
    agg = {nm: dict(n=0, amps=[], peak=0.0, act=[], mins=0.0) for nm in names}
    per_run_r = {}
    for r, df, st in used:
        cm = np.corrcoef(df.values.T); iu = np.triu_indices_from(cm,1)
        per_run_r[r] = float(np.nanmean(cm[iu]))
        mins = len(df)/FS/60
        for nm in names:
            tr = df[nm].values; pk = events(tr)
            a = agg[nm]; a['n'] += len(pk); a['amps'] += [tr[p] for p in pk]
            a['peak'] = max(a['peak'], float(np.nanmax(tr)) if tr.size else 0)
            a['act'].append(float(np.mean(tr > DFF_THR))); a['mins'] += mins
    # pooled propagation speeds across runs
    pooled = {nm: ([],[]) for nm in names}
    for r, df, st in used:
        if st is None: continue
        T = min(len(df), tifffile.TiffFile(str(st)).series[0].shape[0])
        mean, counts, binpos = bin_traces(st, masks, morphs, T)
        for im, nm in enumerate(names):
            if binpos[im] is None: continue
            s, sp = event_speeds(mean[im*NBINS:(im+1)*NBINS], binpos[im], events(df[nm].values[:T]))
            pooled[nm][0].extend(s); pooled[nm][1].extend(sp)
    rows = []
    for im, nm in enumerate(names):
        mo = morphs[im]; a = agg[nm]
        rows.append(dict(group=label, name=nm,
            **{k: mo[k] for k in ['n_vox','vol_um3','depth_um','length_um','linearity','orient_deg','shape_class']},
            n_events=a['n'], event_rate=a['n']/a['mins'] if a['mins'] else 0,
            peak_dff=a['peak'], mean_amp=float(np.mean(a['amps'])) if a['amps'] else 0,
            active_frac=float(np.mean(a['act'])) if a['act'] else 0,
            **classify_speeds(*pooled[nm])))
    res = pd.DataFrame(rows)
    out = base/mask_run/"mask_analysis"; out.mkdir(exist_ok=True)
    res.to_csv(out/f"group_{label}_metrics.csv", index=False)
    plot_run(res, None, out, label)
    nd = int(res.prop_dir.isin(['toward_deep','toward_surface']).sum())
    print(f"  [{label}] runs={[r for r,_,_ in used]} masks={nmask} "
          f"trace_mean_r={ {r: round(v,2) for r,v in per_run_r.items()} }")
    print(f"     pooled propagation: directional={nd} uniform={(res.prop_dir=='uniform').sum()} "
          f"n/a={(res.prop_dir=='n/a').sum()}")
    return res

def corr(a, x, y):
    m = np.isfinite(a[x]) & np.isfinite(a[y])
    if m.sum() < 4 or np.ptp(a[x][m]) == 0: return "n<4"
    r, p = stats.pearsonr(a[x][m], a[y][m]); return f"r={r:+.2f} p={p:.2g} (n={m.sum()})"

if __name__ == "__main__":
    allres = []
    for label, d, mo, mr, runs in GROUPS:
        print(f"\n=== {label} ===")
        r = analyze_group(label, d, mo, mr, runs)
        if r is None: continue
        allres.append(r)
        print("     depth vs peak_dff:", corr(r,'depth_um','peak_dff'),
              "| depth vs rate:", corr(r,'depth_um','event_rate'))
        print("     length vs n_events:", corr(r,'length_um','n_events'),
              "| direction:", r[r.n_prop>=2].prop_dir.value_counts().to_dict())
    if allres:
        pooled = pd.concat(allres, ignore_index=True)
        pooled.to_csv(PROJECT_ROOT/"scape-data"/"mask_metrics_grouped.csv", index=False)
        print("\n=== POOLED across groups ===")
        print("depth vs peak_dff:", corr(pooled,'depth_um','peak_dff'))
        print("depth vs event_rate:", corr(pooled,'depth_um','event_rate'))
        print("direction counts:", pooled[pooled.n_prop>=2].prop_dir.value_counts().to_dict())
