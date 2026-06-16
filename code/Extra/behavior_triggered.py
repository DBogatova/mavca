#!/usr/bin/env python
"""
Behavior-triggered raw-vs-residual analysis (plan S6), pooled across runs sharing masks.
GLOBAL coupling (whole tree ~ movement/arousal) vs LOCAL residual coupling (branch-specific
after regressing out global = median across masks). Per-mask circular-shift null for residual.
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS = eb.FS; W = 15
RUNS = [("2026-05-08","rbp4_139_phpeb","run5"), ("2026-05-08","rbp4_139_phpeb","run6")]
MASK_RUN = ("2026-05-08","rbp4_139_phpeb","run5")

def onsets(sig, bounds, z=0.5, refr=10):
    s = (sig-np.nanmean(sig))/(np.nanstd(sig)+1e-9)
    rise = np.where(np.diff((s > z).astype(int)) == 1)[0]+1
    out = []
    for r in rise:
        near_b = any(abs(r-b) < W for b in bounds)
        if W <= r < len(sig)-W and not near_b and (not out or r-out[-1] >= refr): out.append(r)
    return np.array(out)

def bta(sig, ons):
    if len(ons) == 0: return np.zeros(2*W+1), np.zeros(2*W+1)
    seg = np.array([sig[o-W:o+W+1] for o in ons]); seg = seg - seg[:, :W//2].mean(1, keepdims=True)
    return seg.mean(0), seg.std(0)/np.sqrt(len(ons))

def permask_null(M, beh, nsh=300):
    T, N = M.shape; ks = np.random.randint(W, T-W, nsh); obs = np.empty(N); thr = np.empty(N)
    for i in range(N):
        obs[i] = np.corrcoef(M[:, i], beh)[0, 1]
        thr[i] = np.percentile([abs(np.corrcoef(np.roll(M[:, i], k), beh)[0, 1]) for k in ks], 95)
    return obs, thr

# pool runs (shared masks => same columns)
Xs, accs, pups, bounds, off = [], [], [], [], 0
names = None
for d, mo, rn in RUNS:
    b = PR/"scape-data"/d/mo/rn
    df = pd.read_csv(b/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
    nm = [c for c in df.columns if 'dend' in c]
    if names is None: names = nm
    if nm != names: print(f"  WARN {rn} columns differ; skipping"); continue
    X = df[names].values.astype(float); a, p = eb.load_behavior(b, rn, len(X))
    Xs.append(X); accs.append(a if a is not None else np.full(len(X), np.nan))
    pups.append(p if p is not None else np.full(len(X), np.nan))
    off += len(X); bounds.append(off)
X = np.vstack(Xs); accel = np.concatenate(accs); pupil = np.concatenate(pups)
T, Nm = X.shape; bounds = bounds[:-1]
g = np.median(X, 1); beta = (X*g[:, None]).mean(0)/(g.var()+1e-9); resid = X - np.outer(g, beta)
mean_raw = stats.zscore(X, 0).mean(1); mean_res = stats.zscore(resid, 0).mean(1)
mov = onsets(accel, bounds); pup = onsets(pupil, bounds)
def shiftp(a, b, n=500):
    o = np.corrcoef(a, b)[0,1]; null=[np.corrcoef(np.roll(a,k),b)[0,1] for k in np.random.randint(W,len(a)-W,n)]
    return o, (np.sum(np.abs(null)>=abs(o))+1)/(n+1)
ra, pa = shiftp(g, accel); rp, pp = shiftp(g, pupil)
print(f"POOLED {[r for _,_,r in RUNS]}: masks={Nm} frames={T} (n_runs={len(Xs)})  mov_onsets={len(mov)} pupil_onsets={len(pup)}")
print(f"global Ca vs accel r={ra:+.2f} (p={pa:.3f}) | vs pupil r={rp:+.2f} (p={pp:.3f})")
raw_obs, _ = permask_null(X, accel); res_obs, res_thr = permask_null(resid, accel)
frac_raw = np.mean(np.abs(raw_obs) > res_thr); frac_res = np.mean(np.abs(res_obs) > res_thr)
print(f"per-mask accel coupling: |r| median raw={np.nanmedian(np.abs(raw_obs)):.2f} residual={np.nanmedian(np.abs(res_obs)):.2f}")
print(f"  masks with significant residual coupling (>own 95% null): {frac_res*100:.0f}%  (raw: {frac_raw*100:.0f}%)")

mr = MASK_RUN; mb = PR/"scape-data"/mr[0]/mr[1]/mr[2]/"labelmaps_curated_dynamic"
mp = {p.stem.replace("_labelmap",""): p for p in mb.glob("dend_*_labelmap.tif")}
depth = np.array([eb.morphology(tifffile.imread(mp[n])>0)['depth_um'] if n in mp else np.nan for n in names])
cx = np.array([eb.morphology(tifffile.imread(mp[n])>0)['cx'] if n in mp else np.nan for n in names])

fig = plt.figure(figsize=(15, 8)); tt = (np.arange(2*W+1)-W)/FS
def panel(ax, ons, title):
    for sig, c, lab in [(g,'k','global Ca'),(mean_raw,'tab:orange','mean raw'),(mean_res,'tab:blue','mean residual')]:
        m, e = bta(sig, ons); ax.plot(tt, m, c, label=lab); ax.fill_between(tt, m-e, m+e, color=c, alpha=.2)
    ax.axvline(0, color='gray', lw=.6); ax.set_xlabel('time from onset (s)'); ax.set_title(title); ax.legend(fontsize=7)
panel(fig.add_subplot(2,3,1), mov, f"BTA @ movement onset (n={len(mov)})")
panel(fig.add_subplot(2,3,2), pup, f"BTA @ pupil dilation (n={len(pup)})")
ax3 = fig.add_subplot(2,3,3); ax3.plot(np.arange(T)/FS, stats.zscore(g),'k',lw=.5,label='global'); ax3.plot(np.arange(T)/FS, stats.zscore(np.nan_to_num(accel)),'tab:red',lw=.5,alpha=.7,label='accel')
[ax3.axvline(b/FS,color='gray',ls=':',lw=.5) for b in bounds]; ax3.legend(fontsize=7); ax3.set_title(f"global vs accel r={ra:+.2f}"); ax3.set_xlabel('s')
ax4 = fig.add_subplot(2,3,4); ax4.boxplot([np.abs(raw_obs), np.abs(res_obs)], labels=['raw','residual']); ax4.axhline(np.median(res_thr),color='r',ls='--',lw=.8,label='median null95'); ax4.legend(fontsize=7); ax4.set_ylabel('|r with accel|'); ax4.set_title("per-mask accel coupling")
ax5 = fig.add_subplot(2,3,5); s=ax5.scatter(cx,depth,c=res_obs,cmap='coolwarm',vmin=-.3,vmax=.3,s=30); ax5.invert_yaxis(); ax5.set_xlabel('X (um)'); ax5.set_ylabel('depth Y (um)'); ax5.set_title('residual-accel coupling map'); fig.colorbar(s,ax=ax5,shrink=.8)
ax6 = fig.add_subplot(2,3,6); ax6.scatter(raw_obs,res_obs,s=18,alpha=.6); ax6.plot([-.6,.6],[-.6,.6],'k:',lw=.6); ax6.axhline(0,color='gray',lw=.4); ax6.axvline(0,color='gray',lw=.4); ax6.set_xlabel('r(raw,accel)'); ax6.set_ylabel('r(residual,accel)'); ax6.set_title('raw vs residual (paired)')
out = PR/"scape-data"/mr[0]/mr[1]/mr[2]/"mask_analysis"; out.mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig(out/"behavior_triggered_pooled.png", dpi=150); plt.close(fig)
print(f"saved: {out/'behavior_triggered_pooled.png'}")
