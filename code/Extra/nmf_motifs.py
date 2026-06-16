#!/usr/bin/env python
"""NMF spatial-motif decomposition (plan S7), 0508 run5+run6. V(masks x time)>=0 ~ W(N x K) H(K x T).
W columns = spatial motifs (mask loadings), H rows = temporal activation. Choose K by reconstruction
elbow; project motif loadings onto morphology (depth/centroid); correlate each motif's time course
with accelerometer. EXPLORATORY."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS=eb.FS
mb=PR/"scape-data"/"2026-05-08"/"rbp4_139_phpeb"/"run5"/"labelmaps_curated_dynamic"
paths=sorted(mb.glob("dend_*_labelmap.tif")); names=[p.stem.replace("_labelmap","") for p in paths]
coords=[np.argwhere(tifffile.imread(p)>0) for p in paths]
depth=np.array([c[:,1].mean() for c in coords]); cx=np.array([c[:,2].mean()*1.2 for c in coords])
Xs,accs=[],[]
for rn in ["run5","run6"]:
    b=PR/"scape-data"/"2026-05-08"/"rbp4_139_phpeb"/rn; df=pd.read_csv(b/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    X=df[names].values.astype(float); a,_=eb.load_behavior(b,rn,len(X)); Xs.append(X); accs.append(a)
X=np.vstack(Xs); accel=np.concatenate(accs); T,N=X.shape
V=np.clip(X,0,None).T   # (N, T)
errs=[NMF(k,init='nndsvda',max_iter=400,random_state=0).fit(V).reconstruction_err_ for k in range(2,9)]
d2=np.diff(errs,2); K=int(np.argmax(-d2)+3) if len(d2) else 4; K=min(max(K,3),6)
nm=NMF(K,init='nndsvda',max_iter=600,random_state=0).fit(V); W=nm.components_ if False else nm.transform(V)  # W:(N,K)
H=nm.components_   # (K,T)
dom=W.argmax(1)
print(f"0508 NMF: N={N} T={T} chosen K={K}; recon errs K2..8={np.round(errs,1).tolist()}")
for k in range(K):
    wk=W[:,k]; wd=np.average(depth,weights=wk); rk=stats.pearsonr(H[k],np.nan_to_num(accel))[0]
    print(f"  motif {k}: n_dominant={int((dom==k).sum())} wmean_depth={wd:.0f}um depth_spread={np.sqrt(np.average((depth-wd)**2,weights=wk)):.0f} accel_r={rk:+.2f}")
fig=plt.figure(figsize=(14,8)); t=np.arange(T)/FS
ax=fig.add_subplot(2,3,1); ax.plot(range(2,9),errs,'o-'); ax.axvline(K,color='r',ls='--'); ax.set_xlabel('K'); ax.set_ylabel('recon error'); ax.set_title(f'A  Reconstruction error (elbow K={K})')
ax=fig.add_subplot(2,3,2)
for k in range(K): ax.plot(t,H[k]/H[k].max()+k,lw=.7); 
ax.set_xlabel('time (s)'); ax.set_ylabel('motif (offset)'); ax.set_title('B  Motif temporal weights H')
ax=fig.add_subplot(2,3,3)
if accel is not None: ax.plot(t,(accel-np.nanmin(accel))/(np.nanmax(accel)-np.nanmin(accel)),'tab:red',lw=.6); ax.set_title('C  Accelerometer'); ax.set_xlabel('s')
ax=fig.add_subplot(2,3,4); sc=ax.scatter(cx,depth,c=dom,cmap='tab10',s=40); ax.invert_yaxis(); ax.set_xlabel('X(µm)'); ax.set_ylabel('depth Y(µm)'); ax.set_title('D  Masks colored by dominant motif'); fig.colorbar(sc,ax=ax,shrink=.8,label='motif')
ax=fig.add_subplot(2,3,5); ax.boxplot([depth[dom==k] for k in range(K)],labels=[str(k) for k in range(K)]); ax.set_xlabel('motif'); ax.set_ylabel('depth Y(µm)'); ax.set_title('E  Depth distribution per motif')
ax=fig.add_subplot(2,3,6); ar=[stats.pearsonr(H[k],np.nan_to_num(accel))[0] for k in range(K)]; ax.bar(range(K),ar,color='tab:purple'); ax.axhline(0,color='k',lw=.5); ax.set_xlabel('motif'); ax.set_ylabel('H vs accel r'); ax.set_title('F  Motif–movement coupling')
fig.suptitle('NMF spatial motifs — 0508 (exploratory)'); fig.tight_layout(rect=[0,0,1,0.96])
out=PR/"scape-data"/"figures"; fig.savefig(out/"fig5_nmf.png",dpi=150); plt.close(fig)
print(f"saved {out/'fig5_nmf.png'}")
