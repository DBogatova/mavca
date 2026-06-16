#!/usr/bin/env python
"""Characterize new rbp4cre Ca FOVs (green channel) in the established framework:
mean pairwise r, effective dim/N, global-event fraction (regime), event_rate vs depth(Y).
Z voxel unknown for these 69/55-plane volumes -> use only trace + depth(Y, 1.0um) metrics."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
from sklearn.decomposition import PCA
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS=eb.FS; THR=eb.DFF_THR; CUT=0.17
FOVS=[("0209_r136","2026-02-09/rbp4cre_136_phpeb/run1"),
      ("0217_r138","2026-02-17/rbp4cre_138_phpeb/run7")]
for lab,spec_ in FOVS:
    b=PR/"scape-data"/spec_
    df=pd.read_csv(b/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    names=[c for c in df.columns if 'dend' in c]; X=df[names].values.astype(float); T,N=X.shape
    cm=np.corrcoef(X.T); mean_r=cm[np.triu_indices(N,1)].mean()
    sd=X.std(0); sd[sd==0]=1; Z=(X-X.mean(0))/sd; ev=PCA().fit(Z).explained_variance_ratio_; eff=1/np.sum(ev**2)
    ac=(X>THR).sum(1); fr=[]
    for i in range(N):
        for p in eb.events(X[:,i]): fr.append(ac[max(0,p-1):p+2].max()/N)
    fg=np.mean(np.array(fr)>=CUT) if fr else np.nan; mins=T/FS/60
    depth=np.array([np.argwhere(tifffile.imread(b/"labelmaps_curated_dynamic"/f"{n}_labelmap.tif")>0)[:,1].mean() for n in names])
    erate=np.array([len(eb.events(X[:,i]))/mins for i in range(N)])
    m=np.isfinite(depth)&np.isfinite(erate); r,p=stats.pearsonr(depth[m],erate[m])
    print(f"[{lab}] N={N} T={T} mean_r={mean_r:+.2f} eff_dim/N={eff/N:.2f} frac_global={fg:.2f} | "
          f"event_rate~depth r={r:+.2f} p={p:.2g}")
print("\n(reference: independent FOVs mean_r~0.01-0.08, eff_dim/N~0.22-0.49, frac_global~0;")
print(" superficial>deep => event_rate~depth negative)")
