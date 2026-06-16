#!/usr/bin/env python
"""
Dendritic topology across FOVs (plan S4). Spatial-adjacency via physical min-distance
(cKDTree, threshold ADJ_UM) -> branch families; root=deepest mask; branch_order=BFS hops;
dist_trunk=centroid distance (um) to root. Pooled event rate / global participation per mask.
Writes scape-data/topology_pooled.csv for mixed-effects modeling.
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
from scipy import stats
from scipy.spatial import cKDTree
from collections import defaultdict, deque

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
VOX = np.array([3.9,1.0,1.2]); FS = eb.FS; ADJ_UM = 4.0; GLOBAL_CUT = 0.17

FOVS = [  # (label, date, mouse, mask_run, [trace_runs])
    ("0508_r56","2026-05-08","rbp4_139_phpeb","run5",["run5","run6"]),
    ("0512_r56","2026-05-12","rbp4_132_phpeb","run5",["run5","run6"]),
    ("0512_r910","2026-05-12","rbp4_132_phpeb","run9",["run9","run10"]),
    ("0416_r567","2026-04-16","rbp4_132_phpeb","run5",["run5","run6","run7"]),
    ("0331_r8910","2026-03-31","rbp4_132_phpeb","run8",["run8","run9","run10"]),
    ("0331_r67","2026-03-31","rbp4_132_phpeb","run7",["run6","run7"]),
]

def adjacency(coords):
    N = len(coords); bb = [(c.min(0), c.max(0)) for c in coords]; g = defaultdict(set)
    for i in range(N):
        ti = cKDTree(coords[i])
        for j in range(i+1, N):
            if (bb[i][0]-ADJ_UM > bb[j][1]).any() or (bb[j][0]-ADJ_UM > bb[i][1]).any(): continue
            d, _ = ti.query(coords[j], distance_upper_bound=ADJ_UM)
            if np.isfinite(d).any(): g[i].add(j); g[j].add(i)
    return g

def fam_order(g, depth, N):
    seen=set(); fam=-np.ones(N,int); order=np.zeros(N,int); root_of=np.arange(N); fid=0
    for s in range(N):
        if s in seen: continue
        comp=[]; dq=deque([s]); seen.add(s)
        while dq:
            u=dq.popleft(); comp.append(u)
            for v in g[u]:
                if v not in seen: seen.add(v); dq.append(v)
        root=comp[int(np.argmax(depth[comp]))]
        dist={root:0}; dq=deque([root])
        while dq:
            u=dq.popleft()
            for v in g[u]:
                if v in comp and v not in dist: dist[v]=dist[u]+1; dq.append(v)
        for u in comp: fam[u]=fid; root_of[u]=root; order[u]=dist.get(u,0)
        fid+=1
    return fam, order, root_of, fid

def analyze(label, date, mouse, mask_run, trace_runs):
    mb = PR/"scape-data"/date/mouse/mask_run/"labelmaps_curated_dynamic"
    paths = sorted(mb.glob("dend_*_labelmap.tif")); names=[p.stem.replace("_labelmap","") for p in paths]
    coords = [np.argwhere(tifffile.imread(p)>0)*VOX for p in paths]; N=len(coords)
    cent = np.array([c.mean(0) for c in coords]); depth = cent[:,1]
    g = adjacency(coords); fam, order, root_of, nf = fam_order(g, depth, N)
    dist = np.array([np.linalg.norm(cent[i]-cent[root_of[i]]) for i in range(N)])
    fam_size = np.array([int(np.sum(fam==fam[i])) for i in range(N)])
    ev=np.zeros(N); part=[[] for _ in range(N)]; mins=0
    for rn in trace_runs:
        f=PR/"scape-data"/date/mouse/rn/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f)
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        if [c for c in df.columns if 'dend' in c][:N]!=names[:N] or df.shape[1]<N: 
            if not all(n in df.columns for n in names): continue
        X=df[names].values.astype(float); ac=(X>eb.DFF_THR).sum(1); mins+=len(X)/FS/60
        for i in range(N):
            for p in eb.events(X[:,i]): ev[i]+=1; part[i].append(ac[max(0,p-1):p+2].max()/N)
    rate=ev/mins if mins else ev*np.nan
    fg=np.array([np.mean(np.array(p)>=GLOBAL_CUT) if p else np.nan for p in part])
    return pd.DataFrame(dict(fov=label, name=names, depth_um=depth,
        family=[f"{label}_{x}" for x in fam], fam_size=fam_size, branch_order=order,
        dist_trunk_um=dist, event_rate=rate, frac_global=fg)), nf

allr=[]
for lab,d,mo,mr,trs in FOVS:
    r,nf=analyze(lab,d,mo,mr,trs); allr.append(r)
    multi=r.fam_size>1; mm=multi & np.isfinite(r.event_rate)
    def cc(x):
        m=mm & np.isfinite(r[x])
        if m.sum()<5 or np.ptp(r[x][m])==0: return "n/a"
        rr,pp=stats.pearsonr(r[x][m],r.event_rate[m]); return f"r={rr:+.2f} p={pp:.2g}"
    print(f"[{lab}] {len(r)} masks, {nf} families ({int((r.fam_size==1).sum())} singletons, max order {r.branch_order.max()}); "
          f"multi(n={int(mm.sum())}) rate~branch_order {cc('branch_order')} | rate~dist_trunk {cc('dist_trunk_um')}")
T=pd.concat(allr,ignore_index=True); T.to_csv(PR/"scape-data"/"topology_pooled.csv",index=False)
print(f"\nsaved scape-data/topology_pooled.csv  ({len(T)} masks, {T.fov.nunique()} FOVs, {T.family.nunique()} families)")
