#!/usr/bin/env python
"""Figure 3: spatial/morphological organization of dendritic activity & movement coupling.
A masks colored by residual coupling (0508); B coupling vs depth; C coupled-vs-not depth;
D event rate vs depth by class (all FOVs); E %coupled by depth bin (0508);
F movement-triggered Ca raw vs residual, superficial vs deep (0508)."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS=eb.FS; W=15
M=pd.read_csv(PR/"scape-data"/"master_mask_table.csv")
for c in ['depth_um','rate_local','rate_subtree','rate_global']: M[c+'_z']=M.groupby('fov')[c].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
D=M[M.fov=='0508_r56'].reset_index(drop=True); sig=D.coupling_sig.values.astype(bool)
mb=PR/"scape-data"/"2026-05-08"/"rbp4_139_phpeb"/"run5"/"labelmaps_curated_dynamic"
cx=np.array([np.argwhere(tifffile.imread(mb/f"{n}_labelmap.tif")>0)[:,2].mean()*1.2 for n in D.name]); depth=D.depth_um.values

# pool 0508 for panel F
Xs,accs,bnd,off=[],[],[],0; names=None
for rn in ["run5","run6"]:
    b=PR/"scape-data"/"2026-05-08"/"rbp4_139_phpeb"/rn; df=pd.read_csv(b/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    names=names or [c for c in df.columns if 'dend' in c]; X=df[names].values.astype(float); a,_=eb.load_behavior(b,rn,len(X))
    Xs.append(X); accs.append(a); off+=len(X); bnd.append(off)
X=np.vstack(Xs); accel=np.concatenate(accs); T=len(X); bnd=bnd[:-1]
g=np.median(X,1); beta=(X*g[:,None]).mean(0)/(g.var()+1e-9); resid=X-np.outer(g,beta)
s=(accel-np.nanmean(accel))/(np.nanstd(accel)+1e-9); rise=np.where(np.diff((s>0.5).astype(int))==1)[0]+1
mov=[r for r in rise if W<=r<T-W and not any(abs(r-bd)<W for bd in bnd)]
sup=depth<np.median(depth)
def bta(Msel):
    sigz=stats.zscore(Msel,0).mean(1); seg=np.array([sigz[o-W:o+W+1] for o in mov]); seg=seg-seg[:,:W//2].mean(1,keepdims=True); return seg.mean(0)
tt=(np.arange(2*W+1)-W)/FS

fig=plt.figure(figsize=(15,9))
ax=fig.add_subplot(2,3,1); sc=ax.scatter(cx,depth,c=D.coupling_z,cmap='coolwarm',vmin=-3,vmax=3,s=45,edgecolor=np.where(sig,'k','none'),lw=1.3)
ax.invert_yaxis(); ax.set_xlabel('X (µm)'); ax.set_ylabel('depth Y (µm)'); ax.set_title('A  Residual movement coupling (0508)\n(black edge = significant)'); fig.colorbar(sc,ax=ax,shrink=.8,label='coupling z')
ax=fig.add_subplot(2,3,2); ax.scatter(depth,D.coupling_z,s=18,alpha=.6,color='steelblue'); r,p=stats.pearsonr(depth,D.coupling_z)
ax.axhline(0,color='gray',lw=.5); ax.set_xlabel('depth Y (µm)'); ax.set_ylabel('coupling z'); ax.set_title(f'B  Coupling vs depth (0508)\nr={r:+.2f} p={p:.2f} (LMM p=0.41, n.s.)')
ax=fig.add_subplot(2,3,3); ax.boxplot([depth[sig],depth[~sig]],labels=['coupled','not']); ax.set_ylabel('depth Y (µm)')
ax.set_title(f'C  Coupled vs non-coupled depth (0508)\nMWU p={stats.mannwhitneyu(depth[sig],depth[~sig]).pvalue:.2f}')
ax=fig.add_subplot(2,3,4)
for c,col,lab in [('rate_local_z','tab:green','local'),('rate_subtree_z','tab:purple','subtree'),('rate_global_z','tab:red','global')]:
    m=np.isfinite(M[c])&np.isfinite(M.depth_um_z); sl,ic,rr,pp,_=stats.linregress(M.depth_um_z[m],M[c][m])
    xs=np.linspace(-2,2,20); ax.plot(xs,ic+sl*xs,col,label=f'{lab} (slope {sl:+.2f}, p={pp:.2g})')
ax.set_xlabel('depth (within-FOV z)'); ax.set_ylabel('class event rate (z)'); ax.set_title('D  Depth gradient by event class (all 6 FOVs)'); ax.legend(fontsize=7)
ax=fig.add_subplot(2,3,5); qs=pd.qcut(depth,4,labels=False); fb=[sig[qs==q].mean()*100 for q in range(4)]; ctr=[depth[qs==q].mean() for q in range(4)]
ax.bar(range(4),fb,color='tab:blue'); ax.set_xticks(range(4)); ax.set_xticklabels([f'{c:.0f}' for c in ctr]); ax.set_xlabel('depth bin center (µm)'); ax.set_ylabel('% movement-coupled'); ax.set_title('E  %coupled by depth bin (0508)')
ax=fig.add_subplot(2,3,6)
ax.plot(tt,bta(X[:,sup]),'tab:orange',label='superficial raw'); ax.plot(tt,bta(resid[:,sup]),'tab:orange',ls='--',label='superficial resid')
ax.plot(tt,bta(X[:,~sup]),'tab:blue',label='deep raw'); ax.plot(tt,bta(resid[:,~sup]),'tab:blue',ls='--',label='deep resid')
ax.axvline(0,color='gray',lw=.6); ax.set_xlabel('time from movement onset (s)'); ax.set_ylabel('Ca (z)'); ax.set_title('F  Movement-triggered Ca: superficial vs deep (0508)'); ax.legend(fontsize=6)
fig.suptitle('Figure 3. Spatial/morphological organization: depth gradient carried by LOCAL events; '
             'movement coupling distributed (0508).', fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96]); out=PR/"scape-data"/"figures"; fig.savefig(out/"fig3_organization.png",dpi=160); plt.close(fig)
print(f"saved {out/'fig3_organization.png'}")
