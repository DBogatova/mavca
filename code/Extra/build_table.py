#!/usr/bin/env python
"""Master per-mask table. All 6 FOVs: depth, event_rate, SNR, volume, family, FOV, frac_global,
event-class rates (local/subtree/global). 0508 adds residual movement coupling z + significance.
Event class per event: global if participation>=CUT; else subtree if a co-active mask shares the
branch family; else local. Writes scape-data/master_mask_table.csv."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS=eb.FS; THR=eb.DFF_THR; W=15; CUT=0.17; VOXV=4.68
FOVS=[("0508_r56","2026-05-08","rbp4_139_phpeb","run5",["run5","run6"]),
      ("0512_r56","2026-05-12","rbp4_132_phpeb","run5",["run5","run6"]),
      ("0512_r910","2026-05-12","rbp4_132_phpeb","run9",["run9","run10"]),
      ("0416_r567","2026-04-16","rbp4_132_phpeb","run5",["run5","run6","run7"]),
      ("0331_r8910","2026-03-31","rbp4_132_phpeb","run8",["run8","run9","run10"]),
      ("0331_r67","2026-03-31","rbp4_132_phpeb","run7",["run6","run7"])]
topo=pd.read_csv(PR/"scape-data"/"topology_pooled.csv").set_index(['fov','name'])

def pool(date,mouse,runs):
    Xs,accs,off=[],[],0; names=None
    for rn in runs:
        b=PR/"scape-data"/date/mouse/rn; f=b/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f)
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        nm=[c for c in df.columns if 'dend' in c]; names=names or nm
        if not all(n in df.columns for n in names): continue
        X=df[names].values.astype(float); a,_=eb.load_behavior(b,rn,len(X)); Xs.append(X); accs.append(a if a is not None else np.full(len(X),np.nan))
    return names, np.vstack(Xs), np.concatenate(accs)

def snr(x): m=np.median(x); mad=np.median(np.abs(x-m))*1.4826+1e-9; return (np.percentile(x,95)-m)/mad
rows=[]
for lab,d,mo,mr,runs in FOVS:
    names,X,accel=pool(d,mo,runs); T,N=X.shape; mins=T/FS/60
    active=X>THR; fam=np.array([topo.loc[(lab,n),'family'] if (lab,n) in topo.index else f"{lab}_na" for n in names])
    mb=PR/"scape-data"/d/mo/mr/"labelmaps_curated_dynamic"
    vol=np.array([(tifffile.imread(mb/f"{n}_labelmap.tif")>0).sum()*VOXV for n in names])
    nl=np.zeros(N); ns=np.zeros(N); ng=np.zeros(N)
    for i in range(N):
        for p in eb.events(X[:,i]):
            coact=np.where(active[max(0,p-1):p+2].any(0))[0]; part=len(coact)/N
            if part>=CUT: ng[i]+=1
            elif any(fam[j]==fam[i] for j in coact if j!=i): ns[i]+=1
            else: nl[i]+=1
    # 0508 residual movement coupling
    cz=np.full(N,np.nan); sig=np.full(N,np.nan)
    if lab=="0508_r56":
        g=np.median(X,1); beta=(X*g[:,None]).mean(0)/(g.var()+1e-9); resid=X-np.outer(g,beta)
        ks=np.random.randint(W,T-W,400)
        for i in range(N):
            ob=abs(np.corrcoef(resid[:,i],accel)[0,1]); nd=np.array([abs(np.corrcoef(np.roll(resid[:,i],k),accel)[0,1]) for k in ks])
            cz[i]=(ob-nd.mean())/(nd.std()+1e-9); sig[i]=float(ob>np.percentile(nd,95))
    for i,n in enumerate(names):
        tot=nl[i]+ns[i]+ng[i]
        rows.append(dict(fov=lab,name=n,depth_um=topo.loc[(lab,n),'depth_um'] if (lab,n) in topo.index else np.nan,
            event_rate=(tot)/mins, SNR=snr(X[:,i]), volume_um3=vol[i], family=fam[i],
            rate_local=nl[i]/mins, rate_subtree=ns[i]/mins, rate_global=ng[i]/mins,
            frac_local=nl[i]/tot if tot else np.nan, frac_subtree=ns[i]/tot if tot else np.nan, frac_global=ng[i]/tot if tot else np.nan,
            coupling_z=cz[i], coupling_sig=sig[i]))
    print(f"[{lab}] N={N} mins={mins:.0f} events/mask~{(nl+ns+ng).mean():.0f} "
          f"class% L/S/G={nl.sum()/(nl+ns+ng).sum()*100:.0f}/{ns.sum()/(nl+ns+ng).sum()*100:.0f}/{ng.sum()/(nl+ns+ng).sum()*100:.0f}")
M=pd.DataFrame(rows); M.to_csv(PR/"scape-data"/"master_mask_table.csv",index=False)
print(f"\nsaved scape-data/master_mask_table.csv ({len(M)} masks, {M.fov.nunique()} FOVs); 0508 coupling masks={int(M.coupling_sig.notna().sum())}")
