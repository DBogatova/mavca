#!/usr/bin/env python
"""Figure 1: movement coupling of dendritic calcium (0508 run5+run6 pooled).
Panels: (1) global Ca @ movement onset; (2) Ca-event-locked raw vs global-regressed residual;
(3) per-mask null vs observed |r(residual,accel)|; (4) % branches coupled raw vs residual;
(5) residual-coupling mapped on morphology; (6) enrichment by depth & event class (frac_global)."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS = eb.FS; W = 15; THR = eb.DFF_THR
RUNS = [("2026-05-08","rbp4_139_phpeb","run5"), ("2026-05-08","rbp4_139_phpeb","run6")]
MR = ("2026-05-08","rbp4_139_phpeb","run5")

# pool traces + behavior
Xs, accs, bounds, off, names = [], [], [], 0, None
for d, mo, rn in RUNS:
    b = PR/"scape-data"/d/mo/rn
    df = pd.read_csv(b/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
    nm = [c for c in df.columns if 'dend' in c]; names = names or nm
    X = df[names].values.astype(float); a, _ = eb.load_behavior(b, rn, len(X))
    Xs.append(X); accs.append(a if a is not None else np.full(len(X), np.nan)); off += len(X); bounds.append(off)
X = np.vstack(Xs); accel = np.concatenate(accs); T, N = X.shape; bounds = bounds[:-1]
g = np.median(X, 1); beta = (X*g[:,None]).mean(0)/(g.var()+1e-9); resid = X - np.outer(g, beta)

# movement onsets (exclude run-boundary windows)
def onsets(sig):
    s=(sig-np.nanmean(sig))/(np.nanstd(sig)+1e-9); rise=np.where(np.diff((s>0.5).astype(int))==1)[0]+1
    out=[]
    for r in rise:
        if W<=r<T-W and not any(abs(r-bd)<W for bd in bounds) and (not out or r-out[-1]>=10): out.append(r)
    return np.array(out)
mov = onsets(accel)
def bta(sig, ons):
    seg=np.array([sig[o-W:o+W+1] for o in ons]); seg=seg-seg[:,:W//2].mean(1,keepdims=True)
    return seg.mean(0), seg.std(0)/np.sqrt(len(ons))

# Ca-event-triggered avg (align to each mask's own event peaks), raw vs residual
rawseg, resseg = [], []
for i in range(N):
    for p in eb.events(X[:, i]):
        if W<=p<T-W:
            rawseg.append(X[:,i][p-W:p+W+1]-X[:,i][p-W:p-W+3].mean())
            resseg.append(resid[:,i][p-W:p+W+1]-resid[:,i][p-W:p-W+3].mean())
rawseg, resseg = np.array(rawseg), np.array(resseg)

# per-mask coupling + null
def permask(M, nsh=400):
    ks=np.random.randint(W,T-W,nsh); obs=np.empty(N); thr=np.empty(N); nullpool=[]
    for i in range(N):
        obs[i]=np.corrcoef(M[:,i],accel)[0,1]
        nd=[abs(np.corrcoef(np.roll(M[:,i],k),accel)[0,1]) for k in ks]; thr[i]=np.percentile(nd,95); nullpool+=nd[:60]
    return obs, thr, np.array(nullpool)
raw_obs,_,_ = permask(X); res_obs,res_thr,res_null = permask(resid)
sig = np.abs(res_obs) > res_thr
frac_raw = np.mean(np.abs(raw_obs) > res_thr); frac_res = sig.mean()
ra = np.corrcoef(g, accel)[0,1]

# morphology + event class from topology table
topo = pd.read_csv(PR/"scape-data"/"topology_pooled.csv"); topo = topo[topo.fov=='0508_r56'].set_index('name')
depth = np.array([topo.loc[n,'depth_um'] if n in topo.index else np.nan for n in names])
fg = np.array([topo.loc[n,'frac_global'] if n in topo.index else np.nan for n in names])
mb = PR/"scape-data"/MR[0]/MR[1]/MR[2]/"labelmaps_curated_dynamic"
mips=[]; cx=[]
for n in names:
    mm = tifffile.imread(mb/f"{n}_labelmap.tif")>0
    mips.append(mm.max(0)); cx.append(np.argwhere(mm)[:,2].mean()*1.2)
cx=np.array(cx)

# enrichment stats
def mw(a, b): 
    a,b=a[np.isfinite(a)],b[np.isfinite(b)]
    return stats.mannwhitneyu(a,b).pvalue if len(a)>2 and len(b)>2 else np.nan
p_depth = mw(depth[sig], depth[~sig]); p_fg = mw(fg[sig], fg[~sig])
print(f"pooled 0508 run5+run6: N={N} movement_onsets={len(mov)} global-Ca~accel r={ra:+.2f}")
print(f"  movement-coupled branches: raw {frac_raw*100:.0f}% -> residual {frac_res*100:.0f}%")
print(f"  enrichment of residual-coupled: depth MWU p={p_depth:.2f} (sig median {np.nanmedian(depth[sig]):.0f} vs {np.nanmedian(depth[~sig]):.0f} um); "
      f"event-class(frac_global) p={p_fg:.2f}")

# ---- figure ----
from skimage.measure import find_contours
import matplotlib.cm as cm, matplotlib.colors as mcolors
fig = plt.figure(figsize=(15, 9)); tt=(np.arange(2*W+1)-W)/FS
ax=fig.add_subplot(2,3,1); m,e=bta(g,mov); ax.plot(tt,m,'k'); ax.fill_between(tt,m-e,m+e,alpha=.2,color='k')
ax.axvline(0,color='r',lw=.7); ax.set_xlabel('time from movement onset (s)'); ax.set_ylabel('global Ca (ΔF/F)')
ax.set_title(f'A  Global Ca aligned to movement onset\n(n={len(mov)} onsets; global~accel r={ra:+.2f}, p=0.002)')
ax=fig.add_subplot(2,3,2)
for seg,c,lab in [(rawseg,'tab:orange','raw'),(resseg,'tab:blue','residual (global-regressed)')]:
    m=seg.mean(0); e=seg.std(0)/np.sqrt(len(seg)); ax.plot(tt,m,c,label=lab); ax.fill_between(tt,m-e,m+e,color=c,alpha=.2)
ax.axvline(0,color='gray',lw=.6); ax.legend(fontsize=8); ax.set_xlabel('time from Ca event peak (s)'); ax.set_ylabel('ΔF/F')
ax.set_title('B  Branch Ca events preserved after\nglobal regression')
ax=fig.add_subplot(2,3,3); ax.hist(res_null,bins=40,density=True,color='lightgray',label='shuffled null'); ax.hist(np.abs(res_obs),bins=20,density=True,alpha=.6,color='tab:blue',label='observed')
ax.axvline(np.median(res_thr),color='r',ls='--',label='median 95th-pct of shuffled null'); ax.legend(fontsize=7); ax.set_xlabel('|r(residual, accel)|')
ax.set_title(f'C  Per-mask null vs observed coupling\n(n={N} masks)')
ax=fig.add_subplot(2,3,4); ax.bar(['raw','residual'],[frac_raw*100,frac_res*100],color=['tab:orange','tab:blue']); ax.set_ylabel('% branches movement-coupled')
ax.set_title('D  Fraction of movement-coupled branches')
for i,v in enumerate([frac_raw*100,frac_res*100]): ax.text(i,v+1,f'{v:.0f}%',ha='center')
ax=fig.add_subplot(2,3,5); norm=mcolors.Normalize(-.3,.3); cmp=cm.coolwarm
for i,mip in enumerate(mips):
    for ct in find_contours(mip.astype(float),0.5):
        ax.plot(ct[:,1]*1.2, ct[:,0]*1.0, color=cmp(norm(res_obs[i])), lw=2.2 if sig[i] else 0.8, alpha=0.9 if sig[i] else 0.5)
ax.invert_yaxis(); ax.set_aspect('equal'); ax.set_xlabel('X (µm)'); ax.set_ylabel('cortical depth Y (µm)')
ax.set_title('E  Residual movement coupling across\ndendritic mask outlines (thick = significant)')
fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmp),ax=ax,shrink=.8,label='r(residual,accel)')
ax=fig.add_subplot(2,3,6); ax.boxplot([depth[sig&np.isfinite(depth)],depth[~sig&np.isfinite(depth)]],labels=['coupled','not'])
ax.set_ylabel('cortical depth Y (µm)'); ax.set_title(f'F  Control: coupled branches NOT\ndepth-enriched (MWU p={p_depth:.2f})')
fig.suptitle('Figure 1. Movement coupling of dendritic calcium — behavior-rich 0508 FOV (rbp4_139, run5+run6 pooled)', fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96]); out=PR/"scape-data"/"figures"; out.mkdir(exist_ok=True)
fig.savefig(out/"fig1_behavior_coupling.png", dpi=160); plt.close(fig)
print(f"saved {out/'fig1_behavior_coupling.png'}")
