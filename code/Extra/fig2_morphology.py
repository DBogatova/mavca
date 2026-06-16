#!/usr/bin/env python
"""Figure 2: morphology -> activity. (1) event_rate vs depth colored by FOV; (2) mixed-effects
fit+CI (within-FOV z); (3) per-FOV depth slopes; (4) per-FOV branch_order slopes (fragile/null);
(5) rate~branch_order vs reconstruction threshold (instability); (6) mixed-effects coef table."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats, spatial
from collections import defaultdict, deque

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
VOX = np.array([3.9,1.0,1.2])
T = pd.read_csv(PR/"scape-data"/"topology_pooled.csv")
T = T[np.isfinite(T.event_rate)].copy()
for c in ['event_rate','depth_um','branch_order','dist_trunk_um']:
    T[c+'_z'] = T.groupby('fov')[c].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
fovs = sorted(T.fov.unique()); cmap = dict(zip(fovs, plt.cm.tab10(np.linspace(0,1,len(fovs)))))

def lmm(formula):
    return smf.mixedlm(formula, T, groups=T.family).fit(reml=False)
m_d = lmm("event_rate_z ~ depth_um_z")
m_b = lmm("event_rate_z ~ depth_um_z + branch_order_z")
m_t = lmm("event_rate_z ~ depth_um_z + dist_trunk_um_z")

def slopes(xz):  # per-FOV OLS slope + se of event_rate_z ~ xz
    out={}
    for f in fovs:
        d=T[T.fov==f]; m=np.isfinite(d[xz])&np.isfinite(d.event_rate_z)
        if m.sum()>4 and np.ptp(d[xz][m])>0:
            r=stats.linregress(d[xz][m], d.event_rate_z[m]); out[f]=(r.slope,r.stderr)
    return out

# threshold instability (recompute branch_order at several adjacency thresholds)
def adjacency(coords, thr):
    N=len(coords); bb=[(c.min(0),c.max(0)) for c in coords]; g=defaultdict(set)
    for i in range(N):
        ti=spatial.cKDTree(coords[i])
        for j in range(i+1,N):
            if (bb[i][0]-thr>bb[j][1]).any() or (bb[j][0]-thr>bb[i][1]).any(): continue
            d,_=ti.query(coords[j],distance_upper_bound=thr)
            if np.isfinite(d).any(): g[i].add(j); g[j].add(i)
    return g
def border(g, depth, N):
    seen=set(); order=np.zeros(N,int); size=np.zeros(N,int)
    for s in range(N):
        if s in seen: continue
        comp=[]; dq=deque([s]); seen.add(s)
        while dq:
            u=dq.popleft(); comp.append(u)
            for v in g[u]:
                if v not in seen: seen.add(v); dq.append(v)
        root=comp[int(np.argmax(depth[comp]))]; dist={root:0}; dq=deque([root])
        while dq:
            u=dq.popleft()
            for v in g[u]:
                if v in comp and v not in dist: dist[v]=dist[u]+1; dq.append(v)
        for u in comp: order[u]=dist.get(u,0); size[u]=len(comp)
    return order, size
THR_FOVS = {"0508_r56":("2026-05-08","rbp4_139_phpeb","run5"),
            "0512_r56":("2026-05-12","rbp4_132_phpeb","run5")}
thr_curves={}
for lab,(d,mo,rn) in THR_FOVS.items():
    mb=PR/"scape-data"/d/mo/rn/"labelmaps_curated_dynamic"
    sub=T[T.fov==lab].set_index('name')
    paths=[mb/f"{n}_labelmap.tif" for n in sub.index]
    coords=[np.argwhere(tifffile.imread(p)>0)*VOX for p in paths]
    depth=np.array([c[:,1].mean() for c in coords]); rate=sub.event_rate.values; N=len(coords)
    rs=[]
    for thr in [2,3,4,6,8]:
        order,size=border(adjacency(coords,thr),depth,N); mm=size>1
        if mm.sum()>4 and np.ptp(order[mm])>0: rs.append(stats.pearsonr(order[mm],rate[mm])[0])
        else: rs.append(np.nan)
    thr_curves[lab]=rs

# ---- figure ----
fig=plt.figure(figsize=(15,9))
ax=fig.add_subplot(2,3,1)
for f in fovs: d=T[T.fov==f]; ax.scatter(d.depth_um,d.event_rate,s=14,alpha=.6,color=cmap[f],label=f)
ax.set_xlabel('cortical depth Y (µm)'); ax.set_ylabel('event rate (/min)'); ax.set_title('(1) Event rate vs depth (by FOV)'); ax.legend(fontsize=6)
ax=fig.add_subplot(2,3,2)
for f in fovs: d=T[T.fov==f]; ax.scatter(d.depth_um_z,d.event_rate_z,s=12,alpha=.5,color=cmap[f])
xs=np.linspace(T.depth_um_z.min(),T.depth_um_z.max(),50); b=m_d.params['depth_um_z']; se=m_d.bse['depth_um_z']
ax.plot(xs,b*xs,'k',lw=2,label=f'LMM slope={b:.2f}\np={m_d.pvalues["depth_um_z"]:.3f}'); ax.fill_between(xs,(b-1.96*se)*xs,(b+1.96*se)*xs,color='k',alpha=.2)
ax.set_xlabel('depth (within-FOV z)'); ax.set_ylabel('event rate (z)'); ax.set_title('(2) Mixed-effects fit (family RE)'); ax.legend(fontsize=8)
ax=fig.add_subplot(2,3,3); sl=slopes('depth_um_z'); ys=range(len(sl))
ax.errorbar([v[0] for v in sl.values()],list(ys),xerr=[1.96*v[1] for v in sl.values()],fmt='o',color='tab:blue')
ax.axvline(0,color='r',lw=.7); ax.axvline(b,color='k',ls='--',lw=.8,label='pooled'); ax.set_yticks(list(ys)); ax.set_yticklabels(list(sl.keys()),fontsize=7); ax.set_xlabel('per-FOV depth slope'); ax.set_title('(3) Depth slope by FOV (− in 5/6)'); ax.legend(fontsize=7)
ax=fig.add_subplot(2,3,4); sb=slopes('branch_order_z'); ys=range(len(sb))
ax.errorbar([v[0] for v in sb.values()],list(ys),xerr=[1.96*v[1] for v in sb.values()],fmt='o',color='tab:orange')
ax.axvline(0,color='r',lw=.7); ax.set_yticks(list(ys)); ax.set_yticklabels(list(sb.keys()),fontsize=7); ax.set_xlabel('per-FOV branch-order slope'); ax.set_title(f'(4) Branch-order slope (fragile/null)\nLMM p={m_b.pvalues["branch_order_z"]:.2f}')
ax=fig.add_subplot(2,3,5)
for lab,rs in thr_curves.items(): ax.plot([2,3,4,6,8],rs,'o-',label=lab)
ax.axhline(0,color='r',lw=.7); ax.set_xlabel('adjacency threshold (µm)'); ax.set_ylabel('r(branch_order, event_rate)'); ax.set_title('(5) Branch-order effect vs reconstruction\nthreshold (unstable, sign-flips)'); ax.legend(fontsize=7)
ax=fig.add_subplot(2,3,6); ax.axis('off')
rows=[]
for lab,m in [('depth-only',m_d),('+branch_order',m_b),('+dist_trunk',m_t)]:
    for term in m.params.index:
        if term in ('Intercept','Group Var'): continue
        ci=m.conf_int().loc[term]; rows.append([lab,term,f"{m.params[term]:+.3f}",f"{m.pvalues[term]:.3f}",f"[{ci[0]:+.2f},{ci[1]:+.2f}]"])
tab=ax.table(cellText=rows,colLabels=['model','term','coef','p','95% CI'],loc='center',cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1,1.5)
ax.set_title('(6) Mixed-effects coefficients\n(random intercept | family; within-FOV z)',fontsize=10)
fig.suptitle('Figure 2. Morphology→activity: cortical depth (robust) vs branch-order (fragile). '
             '466 masks, 6 FOVs, family random effects.', fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96]); out=PR/"scape-data"/"figures"; out.mkdir(exist_ok=True)
fig.savefig(out/"fig2_morphology.png", dpi=160); plt.close(fig)
pd.DataFrame(rows,columns=['model','term','coef','p','CI']).to_csv(out/"fig2_mixedmodel_coefs.csv",index=False)
print("depth slopes by FOV:", {k:round(v[0],2) for k,v in sl.items()})
print("branch_order slopes by FOV:", {k:round(v[0],2) for k,v in sb.items()})
print("threshold curves:", {k:[round(x,2) if np.isfinite(x) else None for x in v] for k,v in thr_curves.items()})
print(f"saved {out/'fig2_morphology.png'} + fig2_mixedmodel_coefs.csv")
