#!/usr/bin/env python
"""
Event typing + behavior coupling + hubs + morphology, from traces/behavior/masks.

Per run:
  global Ca   : mean dF/F across masks; population active-fraction per frame
  event type  : per event, coincidence = #masks co-active (+/-1 fr); whole-tree(global)
                vs local(isolated) by fraction of masks co-firing
  behavior    : align accel (accel_mag/aligned_time_s) + pupil (10Hz, Basler->SCAPE
                offset); corr(global Ca, accel/pupil); accel during global vs local events
  hubs        : correlation-network degree (#partners r>0.5) + global-event participation;
                spatial map (X vs cortical depth Y) -> look for clustering ("nexus")
  morphology  : depth, length, orientation, branchiness (skeleton tortuosity/branchpoints)
                joined to activity/hub score

5 Hz caveat: event kinetics (bAP vs NMDA) not resolvable; synchrony + amplitude + depth +
behavior coupling are the available proxies.
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, convolve
from skimage.morphology import skeletonize

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
VOXEL = np.array([3.9, 1.0, 1.2]); FS = 5.0
DFF_THR = 1.5; MIN_DUR = 2; GLOBAL_FRAC = 0.30   # event "global" if >=30% masks co-active
RUNS = [("2026-04-16","rbp4_132_phpeb","run1"),   # outlier co-active FOV
        ("2026-04-16","rbp4_132_phpeb","run7"),   # independent, same mouse/day
        ("2026-03-31","rbp4_132_phpeb","run8"),
        ("2026-05-08","rbp4_139_phpeb","run5"),
        ("2026-05-12","rbp4_132_phpeb","run9")]

def events(tr):
    idx = np.where(tr > DFF_THR)[0]; pk = []
    if idx.size:
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0]+1):
            if run.size >= MIN_DUR: pk.append(run[np.argmax(tr[run])])
    return pk

def morphology(m):
    c = np.argwhere(m); pts = c*VOXEL; ctr = pts.mean(0)
    out = dict(depth_um=ctr[1], cx=ctr[2]*1.0, cy=ctr[1], n_vox=len(c))
    if len(c) >= 6:
        w, V = np.linalg.eigh(np.cov((pts-ctr).T)); w = np.clip(w[::-1],1e-9,None); pa = V[:,::-1][:,0]
        proj = (pts-ctr)@pa
        out.update(length_um=float(proj.max()-proj.min()), linearity=float((w[0]-w[1])/w[0]),
                   orient_deg=float(np.degrees(np.arccos(min(abs(pa[1]),1)))))
    else: out.update(length_um=0, linearity=0, orient_deg=np.nan)
    # branchiness from MIP skeleton
    mip = m.max(0); sk = skeletonize(mip)
    if sk.sum() > 3:
        nb = convolve(sk.astype(int), np.ones((3,3)), mode='constant') - sk
        bp = int((sk & (nb >= 3)).sum())
        sc = np.argwhere(sk); d = np.linalg.norm(sc[:,None]-sc[None], axis=2)
        tort = sk.sum()/(d.max()+1e-6)              # path length / max extent
    else: bp, tort = 0, 1.0
    out.update(branchpoints=bp, tortuosity=float(tort))
    return out

def load_behavior(base, run, T):
    """Return accel, pupil resampled to T Ca-frames (NaN if missing). Time 0 = trigger."""
    t_ca = np.arange(T)/FS
    accel = pupil = None
    try:
        ac = pd.read_csv(next((base/"trigger").glob("*_accel.csv")))
        a = np.abs(ac['accel_mag'].values); ta = ac['aligned_time_s'].values
        accel = np.interp(t_ca, ta, gaussian_filter1d(a, 10))
    except Exception as e: print("   accel:", e)
    try:
        mat = loadmat(next((base/"behavior").glob("*behavior.mat")))
        p = gaussian_filter1d(mat['pupil']['pupil_raw'][0][0].flatten(), 2); tp = np.arange(len(p))/10.0
        trig = pd.read_csv(next((base/"trigger").glob("*_trigger.csv")))
        off = trig.loc[trig['AndorXylaTrigger'].diff()==1,'time_s'].iloc[0] - \
              trig.loc[trig['baslerExposureTrigger'].diff()==1,'time_s'].iloc[0]
        pupil = np.interp(t_ca, tp-off, p)
    except Exception as e: print("   pupil:", e)
    return accel, pupil

def analyze(date, mouse, run):
    base = PR/"scape-data"/date/mouse/run
    df = pd.read_csv(base/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
    df = df[[c for c in df.columns if 'dend' in c]]
    X = df.values; T, N = X.shape
    names = list(df.columns)
    gca = X.mean(1)                                  # global calcium
    active = X > DFF_THR; acount = active.sum(1)      # masks active per frame
    # event typing
    glob, loc, allpk = 0, 0, np.zeros(T, bool)
    permask_glob = {n:0 for n in names}
    for j, n in enumerate(names):
        for p in events(X[:, j]):
            allpk[p] = True
            co = acount[max(0,p-1):p+2].max()
            if co >= GLOBAL_FRAC*N: glob += 1; permask_glob[n] += 1
            elif co <= 2: loc += 1
    # hubs: correlation degree
    cm = np.corrcoef(X.T); np.fill_diagonal(cm, 0)
    degree = (np.abs(cm) > 0.5).sum(1)
    mean_r = cm[np.triu_indices(N,1)].mean()
    # behavior
    accel, pupil = load_behavior(base, run, T)
    def rcorr(b):
        if b is None: return np.nan
        m = np.isfinite(b); return stats.pearsonr(gca[m], b[m])[0]
    r_acc, r_pup = rcorr(accel), rcorr(pupil)
    # accel during global vs local-active vs quiet frames
    gframes = acount >= GLOBAL_FRAC*N
    aq = {}
    if accel is not None:
        aq = dict(accel_global=float(np.nanmean(accel[gframes])) if gframes.any() else np.nan,
                  accel_active=float(np.nanmean(accel[(acount>0)&~gframes])),
                  accel_quiet=float(np.nanmean(accel[acount==0])) if (acount==0).any() else np.nan)
    # morphology
    mp = sorted((base/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))
    morph = {p.stem.replace("_labelmap",""): morphology(tifffile.imread(p)>0) for p in mp}
    rows = []
    for j, n in enumerate(names):
        mo = morph.get(n, {})
        rows.append(dict(run=f"{date}/{run}", name=n, n_events=len(events(X[:,j])),
                         peak=float(X[:,j].max()), degree=int(degree[j]), glob_part=permask_glob[n],
                         **{k: mo.get(k, np.nan) for k in
                            ['depth_um','length_um','linearity','orient_deg','branchpoints','tortuosity','cx','cy']}))
    res = pd.DataFrame(rows)
    out = base/"mask_analysis"; out.mkdir(exist_ok=True); res.to_csv(out/"event_behavior_metrics.csv", index=False)
    plot(res, gca, accel, pupil, acount, cm, f"{mouse}/{run}", out)
    summary = dict(run=f"{date}/{run}", N=N, mean_r=mean_r, frac_global=glob/max(glob+loc,1),
                   n_global=glob, n_local=loc, r_gca_accel=r_acc, r_gca_pupil=r_pup, **aq,
                   hub_depth_r=_c(res,'degree','depth_um'), evt_depth_r=_c(res,'n_events','depth_um'),
                   evt_tort_r=_c(res,'n_events','tortuosity'), deg_branch_r=_c(res,'degree','branchpoints'))
    return res, summary

def _c(a, x, y):
    m = np.isfinite(a[x]) & np.isfinite(a[y])
    return round(stats.pearsonr(a[x][m], a[y][m])[0], 2) if m.sum() > 4 and np.ptp(a[x][m])>0 else np.nan

def plot(res, gca, accel, pupil, acount, cm, title, out):
    t = np.arange(len(gca))/FS
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(3,1,1)
    ax.plot(t, gca, 'k', lw=.8, label='global Ca'); ax.set_ylabel('global dF/F'); ax.legend(loc='upper left', fontsize=7)
    axb = ax.twinx()
    if accel is not None: axb.plot(t, accel, 'tab:red', lw=.7, alpha=.7, label='accel')
    if pupil is not None: axb.plot(t, pupil/ (np.nanmax(pupil)+1e-9), 'tab:blue', lw=.7, alpha=.6, label='pupil(norm)')
    axb.set_ylim(0, 0.25 if accel is not None else 1); axb.legend(loc='upper right', fontsize=7)
    ax.set_title(f"{title}  global Ca + behavior")
    ax2 = fig.add_subplot(3,3,4); ax2.plot(t, acount, 'g', lw=.6); ax2.set_title("masks active / frame"); ax2.set_xlabel("s")
    ax3 = fig.add_subplot(3,3,5)
    im = ax3.imshow(cm, vmin=-.3, vmax=.3, cmap='RdBu_r'); ax3.set_title(f"trace corr (mean off-diag {cm[np.triu_indices(len(cm),1)].mean():.2f})"); fig.colorbar(im, ax=ax3, shrink=.7)
    ax4 = fig.add_subplot(3,3,6)
    sc = ax4.scatter(res.cx, res.depth_um, c=res.degree, s=20+res.n_events*2, cmap='hot')
    ax4.invert_yaxis(); ax4.set_xlabel("X (um)"); ax4.set_ylabel("depth Y (um)"); ax4.set_title("hub map (color=degree,size=events)"); fig.colorbar(sc, ax=ax4, shrink=.7)
    ax5 = fig.add_subplot(3,3,7); _s(ax5, res.depth_um, res.n_events, "depth (um)", "n events")
    ax6 = fig.add_subplot(3,3,8); _s(ax6, res.tortuosity, res.n_events, "tortuosity", "n events")
    ax7 = fig.add_subplot(3,3,9); _s(ax7, res.depth_um, res.degree, "depth (um)", "hub degree")
    fig.tight_layout(); fig.savefig(out/"event_behavior.png", dpi=150); plt.close(fig)

def _s(ax, x, y, xl, yl):
    ax.scatter(x, y, s=14, alpha=.6, color='steelblue'); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)
    m = np.isfinite(x)&np.isfinite(y)
    if m.sum()>4 and np.ptp(x[m])>0:
        r,p = stats.pearsonr(x[m],y[m]); ax.text(.05,.92,f"r={r:+.2f}\np={p:.2g}",transform=ax.transAxes,va='top',fontsize=8,bbox=dict(boxstyle='round',fc='w',alpha=.8))

if __name__ == "__main__":
    summ = []
    for d, mo, rn in RUNS:
        print(f"=== {d}/{mo}/{rn} ===")
        try:
            _, s = analyze(d, mo, rn); summ.append(s)
        except Exception as e:
            import traceback; traceback.print_exc()
    S = pd.DataFrame(summ)
    pd.set_option('display.width', 200, 'display.max_columns', 30)
    print("\n===== CROSS-RUN SUMMARY =====")
    print(S.round(3).to_string(index=False))
    S.to_csv(PR/"scape-data"/"event_behavior_summary.csv", index=False)
