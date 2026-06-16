#!/usr/bin/env python
"""#7 replication: scan-corrected cross-mask depth recruitment order across global-event FOVs.
Depth (Y) scan-clean; subtract Z-acquisition offset (z_centroid * Tvol/30). Per global event,
regress per-mask true onset on depth; pooled permutation (shuffle depth within event) + per-event
sign test. +slope = surface->deep (top-down). 5 Hz => slow recruitment, not propagation."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS=eb.FS; THR=eb.DFF_THR; CUT=0.17; DZ=(1.0/FS)/30
FOVS=[("0508",  "2026-05-08","rbp4_139_phpeb","run5",["run5","run6"]),
      ("0331_r67","2026-03-31","rbp4_132_phpeb","run7",["run6","run7"]),
      ("0331_r8910","2026-03-31","rbp4_132_phpeb","run8",["run8","run9","run10"]),
      ("0512_r56","2026-05-12","rbp4_132_phpeb","run5",["run5","run6"])]

def analyze(lab,date,mouse,mr,runs):
    mb=PR/"scape-data"/date/mouse/mr/"labelmaps_curated_dynamic"
    paths=sorted(mb.glob("dend_*_labelmap.tif")); names=[p.stem.replace("_labelmap","") for p in paths]
    coords=[np.argwhere(tifffile.imread(p)>0) for p in paths]
    depth=np.array([c[:,1].mean() for c in coords]); zoff=np.array([c[:,0].mean()*DZ for c in coords])
    Xs=[]
    for rn in runs:
        f=PR/"scape-data"/date/mouse/rn/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f)
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        if not all(n in df.columns for n in names): continue
        Xs.append(df[names].values.astype(float))
    X=np.vstack(Xs); T,N=X.shape; active=X>THR; ac=active.sum(1)
    idx=np.where(ac>=CUT*N)[0]; wins=np.split(idx,np.where(np.diff(idx)>1)[0]+1) if idx.size else []
    dc,oc,evid,spreads,ne=[],[],[],[],0
    for w in wins:
        if len(w)==0: continue
        a,b=w[0]-1,w[-1]+2; part=[i for i in range(N) if active[max(0,a):b,i].any()]
        if len(part)<5: continue
        onset=np.array([max(0,a)+np.argmax(active[max(0,a):b,i]) for i in part])/FS+zoff[part]; dp=depth[part]
        if np.ptp(dp)<20: continue
        spreads.append((onset.max()-onset.min())*FS); dc+=list(dp-dp.mean()); oc+=list(onset-onset.mean()); evid+=[ne]*len(part); ne+=1
    dc,oc,evid=np.array(dc),np.array(oc),np.array(evid)
    if ne<3 or len(dc)<10 or np.ptp(dc)==0: return dict(fov=lab,n_events=ne,note="insufficient global events")
    slope=np.polyfit(dc,oc,1)[0]; null=np.empty(2000)
    for k in range(2000):
        dsh=np.empty_like(dc)
        for e in np.unique(evid): m=evid==e; v=dc[m].copy(); np.random.shuffle(v); dsh[m]=v
        null[k]=np.polyfit(dsh,oc,1)[0]
    p=(np.sum(np.abs(null)>=abs(slope))+1)/2001
    evs=np.array([np.polyfit(dc[evid==e],oc[evid==e],1)[0] for e in np.unique(evid) if np.ptp(dc[evid==e])>0])
    npos=int((evs>0).sum()); binom=stats.binomtest(npos,len(evs)).pvalue
    return dict(fov=lab,N=N,n_events=ne,slope_ms_um=round(slope*1000,2),perm_p=round(p,3),
                pos_events=f"{npos}/{len(evs)}",signtest_p=round(binom,3),med_spread_fr=round(float(np.median(spreads)),1))

rows=[analyze(*f) for f in FOVS]
print(pd.DataFrame(rows).to_string(index=False))
