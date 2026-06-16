#!/usr/bin/env python
"""Validation B: does the (weak) pairwise correlation depend on space? For the 6 known-voxel FOVs
(VOX 3.9,1.0,1.2), per mask-pair compute corr r, coactivation index, 3D Euclidean distance (um),
|depth(Y) difference|, same-branch-family. Pool across FOVs; test r/coact vs distance, depth-diff,
same-family. New mice excluded (Z voxel unknown)."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
VOX=np.array([3.9,1.0,1.2]); THR=eb.DFF_THR
topo=pd.read_csv(PR/"scape-data"/"topology_pooled.csv").set_index(['fov','name'])
FOVS=[("0508_r56","2026-05-08/rbp4_139_phpeb/run5",["run5","run6"]),
      ("0331_r67","2026-03-31/rbp4_132_phpeb/run7",["run6","run7"]),
      ("0331_r8910","2026-03-31/rbp4_132_phpeb/run8",["run8","run9","run10"]),
      ("0512_r56","2026-05-12/rbp4_132_phpeb/run5",["run5","run6"]),
      ("0512_r910","2026-05-12/rbp4_132_phpeb/run9",["run9","run10"]),
      ("0416_r567","2026-04-16/rbp4_132_phpeb/run7",["run7"])]
recs=[]
for fov,mpath,truns in FOVS:
    mb=PR/"scape-data"/mpath.rsplit("/",1)[0]; date=mpath.split("/")[0]; mouse=mpath.split("/")[1]
    mdir=PR/"scape-data"/date/mouse/mpath.split("/")[2]/"labelmaps_curated_dynamic"
    paths=sorted(mdir.glob("dend_*_labelmap.tif")); names=[p.stem.replace("_labelmap","") for p in paths]
    cent=np.array([np.argwhere(tifffile.imread(p)>0).mean(0)*VOX for p in paths])
    fam=np.array([topo.loc[(fov,n),'family'] if (fov,n) in topo.index else f"{fov}_{i}" for i,n in enumerate(names)])
    Xs=[]
    for rn in truns:
        f=PR/"scape-data"/date/mouse/rn/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f); 
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        if all(n in df.columns for n in names): Xs.append(df[names].values.astype(float))
    X=np.vstack(Xs); N=X.shape[1]; cm=np.corrcoef(X.T); act=X>THR
    for i in range(N):
        for j in range(i+1,N):
            d=np.linalg.norm(cent[i]-cent[j]); dd=abs(cent[i,1]-cent[j,1])
            ai,aj=act[:,i],act[:,j]; both=(ai&aj).sum()
            coact=both/np.sqrt(ai.sum()*aj.sum()+1e-9)
            recs.append((fov,cm[i,j],coact,d,dd,int(fam[i]==fam[j])))
P=pd.DataFrame(recs,columns=['fov','r','coact','dist','ddepth','samefam'])
print(f"pairs={len(P)} across {P.fov.nunique()} FOVs")
def pr(x,y): m=np.isfinite(P[x])&np.isfinite(P[y]); rr,pp=stats.pearsonr(P[x][m],P[y][m]); return f"r={rr:+.3f} p={pp:.1g}"
print("pooled relationships:")
print("  corr r   vs distance :", pr('dist','r'))
print("  corr r   vs |Δdepth| :", pr('ddepth','r'))
print("  coact    vs distance :", pr('dist','coact'))
sf=P[P.samefam==1].r; df_=P[P.samefam==0].r
print(f"  same-family r: median={sf.median():.3f} (n={len(sf)}) vs diff-family r: median={df_.median():.3f} (n={len(df_)})  MWU p={stats.mannwhitneyu(sf,df_).pvalue:.1g}")
# binned r vs distance
bins=np.linspace(0,P.dist.quantile(.98),8); P['db']=pd.cut(P.dist,bins); g=P.groupby('db',observed=True).r.agg(['mean','sem'])
fig,ax=plt.subplots(1,3,figsize=(14,4))
ctr=[iv.mid for iv in g.index]; ax[0].errorbar(ctr,g['mean'],yerr=g['sem'],fmt='o-'); ax[0].axhline(0,color='gray',lw=.5); ax[0].set_xlabel('pair distance (µm)'); ax[0].set_ylabel('mean corr r'); ax[0].set_title(f'A  Correlation vs distance\n({pr("dist","r")})')
ax[1].boxplot([df_.dropna(),sf.dropna()],labels=['diff family','same family']); ax[1].set_ylabel('corr r'); ax[1].set_title(f'B  Correlation by branch family\n(same med {sf.median():.3f} vs diff {df_.median():.3f})')
P['ddb']=pd.cut(P.ddepth,np.linspace(0,P.ddepth.quantile(.98),8)); g2=P.groupby('ddb',observed=True).r.agg(['mean','sem']); c2=[iv.mid for iv in g2.index]
ax[2].errorbar(c2,g2['mean'],yerr=g2['sem'],fmt='o-',color='tab:green'); ax[2].axhline(0,color='gray',lw=.5); ax[2].set_xlabel('|Δ depth| (µm)'); ax[2].set_ylabel('mean corr r'); ax[2].set_title(f'C  Correlation vs depth difference\n({pr("ddepth","r")})')
fig.suptitle('Validation: spatial dependence of pairwise correlation (6 known-voxel FOVs)')
fig.tight_layout(); out=PR/"scape-data"/"figures"; fig.savefig(out/"fig7_spatial.png",dpi=150); plt.close(fig)
P.to_csv(PR/"scape-data"/"spatial_pairs.csv",index=False); print(f"saved {out/'fig7_spatial.png'}, spatial_pairs.csv")
