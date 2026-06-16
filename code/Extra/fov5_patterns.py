#!/usr/bin/env python
"""Patterns across the 5 single-channel 5Hz behavior FOVs (0416, 0508, 0512).
Per FOV (pooled shared-mask runs): independence (mean_r, eff_dim/N), event regime (frac_global,
Gini), movement coupling (event-locked global-Ca response at onset), depth-activity slope.
Then cross-FOV correlations."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
from sklearn.decomposition import PCA
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS=eb.FS; THR=eb.DFF_THR; CUT=0.17; W=15; VOX=np.array([3.9,1.0,1.2])
FOVS=[("0416_r123","2026-04-16","rbp4_132_phpeb","run1",["run1","run2","run3"]),
      ("0416_r567","2026-04-16","rbp4_132_phpeb","run5",["run5","run6","run7"]),
      ("0508_r56","2026-05-08","rbp4_139_phpeb","run5",["run5","run6"]),
      ("0512_r56","2026-05-12","rbp4_132_phpeb","run5",["run5","run6"]),
      ("0512_r910","2026-05-12","rbp4_132_phpeb","run9",["run9","run10"])]
def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x); return (np.sum((2*np.arange(1,n+1)-n-1)*x))/(n*x.sum()+1e-9) if x.sum()>0 else 0
rows=[]
for lab,d,mo,mr,runs in FOVS:
    base=PR/"scape-data"/d/mo; Xs=[]; accs=[]; bnd=[]; off=0; names=None
    for rn in runs:
        f=base/rn/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f); 
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        nm=[c for c in df.columns if 'dend' in c]; names=names or nm
        if not all(n in df.columns for n in names): continue
        X=df[names].values.astype(float); a,_=eb.load_behavior(base/rn,rn,len(X))
        Xs.append(X); accs.append(a if a is not None else np.full(len(X),np.nan)); off+=len(X); bnd.append(off)
    X=np.vstack(Xs); accel=np.concatenate(accs); T,N=X.shape; ac=(X>THR).sum(1); gfr=ac>=CUT*N
    cm=np.corrcoef(X.T); iu=np.triu_indices(N,1); mean_r=cm[iu].mean()
    sd=X.std(0); sd[sd==0]=1; ev=PCA().fit((X-X.mean(0))/sd).explained_variance_ratio_; eff=1/np.sum(ev**2)/N
    # events / regime
    cnt=np.zeros(N); fg=0; ne=0
    for i in range(N):
        for p in eb.events(X[:,i]):
            cnt[i]+=1; ne+=1
            if ac[max(0,p-1):p+2].max()>=CUT*N: fg+=1
    fracg=fg/ne if ne else 0
    # movement coupling: event-locked global Ca @ onset
    g=np.median(X,1); gz=(g-g.mean())/g.std()
    s=(accel-np.nanmean(accel))/(np.nanstd(accel)+1e-9); mv=s>0.5
    on=[r for r in (np.where(np.diff(mv.astype(int))==1)[0]+1) if W<=r<T-W and not any(abs(r-b)<W for b in bnd[:-1])]
    movresp=np.nan
    if len(on)>=5:
        seg=np.array([gz[o-W:o+W+1]-gz[o-W:o-W+5].mean() for o in on]); movresp=seg[:,W:W+4].mean()
    # depth-activity
    mp=base/mr/"labelmaps_curated_dynamic"
    depth=np.array([np.argwhere(tifffile.imread(mp/f"{n}_labelmap.tif")>0)[:,1].mean() for n in names])
    erate=cnt/(T/FS/60); m=np.isfinite(depth)
    dr=stats.pearsonr(depth[m],erate[m])[0] if m.sum()>4 else np.nan
    rows.append(dict(fov=lab,mouse=mo[:8],N=N,mean_r=round(mean_r,3),eff_dimN=round(eff,3),
        frac_global=round(fracg,3),gini=round(gini(cnt),3),mov_onsets=len(on),
        mov_resp_z=round(movresp,2) if np.isfinite(movresp) else np.nan,depth_evrate_r=round(dr,2)))
D=pd.DataFrame(rows); pd.set_option('display.width',200)
print(D.to_string(index=False))
print("\n--- cross-FOV relationships (n=5, descriptive) ---")
for x,y in [('frac_global','mov_resp_z'),('eff_dimN','frac_global'),('N','eff_dimN'),('frac_global','gini')]:
    a=D[[x,y]].dropna()
    if len(a)>=3 and a[x].std()>0: print(f"  {x} vs {y}: r={stats.pearsonr(a[x],a[y])[0]:+.2f} (n={len(a)})")
D.to_csv(PR/"scape-data"/"fov5_patterns.csv",index=False); print("saved fov5_patterns.csv")
