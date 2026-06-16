#!/usr/bin/env python
"""Mask functional fingerprints + hotspot/sparsity summary (Q1+Q2 features), 8 real FOVs.
Per mask: event rate, local/global rates & preference, amplitude, duration/AUC, burstiness/IEI-CV,
morphology (n_vox, depth Y, orientation, SNR), and 0508 residual movement coupling (z, sign).
Per FOV hotspot summary: event-count Gini, top 5/10/20% event share, first/second-half &
across-run rank stability. No topology/family features."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
THR=eb.DFF_THR; FS=eb.FS; MIND=2; CUT=0.17; VOX=np.array([3.9,1.0,1.2])
FOVS=[("0508","2026-05-08","rbp4_139_phpeb","run5",["run5","run6"]),
      ("0416_r567","2026-04-16","rbp4_132_phpeb","run5",["run5","run6","run7"]),
      ("0331_r67","2026-03-31","rbp4_132_phpeb","run7",["run6","run7"]),
      ("0331_r8910","2026-03-31","rbp4_132_phpeb","run8",["run8","run9","run10"]),
      ("0512_r56","2026-05-12","rbp4_132_phpeb","run5",["run5","run6"]),
      ("0512_r910","2026-05-12","rbp4_132_phpeb","run9",["run9","run10"]),
      ("0209","2026-02-09","rbp4cre_136_phpeb","run1",["run1"]),
      ("0217","2026-02-17","rbp4cre_138_phpeb","run7",["run7"])]
def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x); 
    return (np.sum((2*np.arange(1,n+1)-n-1)*x))/(n*x.sum()+1e-9) if x.sum()>0 else 0.0
def detail(tr):
    idx=np.where(tr>THR)[0]; out=[]
    if idx.size:
        for r in np.split(idx,np.where(np.diff(idx)>1)[0]+1):
            if r.size>=MIND: out.append((r[0],float(tr[r].max()),len(r),float(tr[r].sum())))
    return out
def snr(x): m=np.median(x); mad=np.median(np.abs(x-m))*1.4826+1e-9; return (np.percentile(x,95)-m)/mad
mrows=[]; hrows=[]
for fov,d,mo,mr,runs in FOVS:
    base=PR/"scape-data"/d/mo
    runX=[]
    for rn in runs:
        f=base/rn/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f); 
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        if runX and [c for c in df.columns if 'dend' in c]!=names: continue
        names=[c for c in df.columns if 'dend' in c]; runX.append(df[names].values.astype(float))
    X=np.vstack(runX); T,N=X.shape; mins=T/FS/60; active=X>THR; ac=active.sum(1)
    mdir=base/mr/"labelmaps_curated_dynamic"
    coords=[np.argwhere(tifffile.imread(mdir/f"{n}_labelmap.tif")>0) for n in names]
    # 0508 movement coupling
    cz=np.full(N,np.nan); csign=np.array([""]*N,object)
    if fov=="0508":
        accs=[]
        for rn in runs:
            a,_=eb.load_behavior(base/rn,rn,len(pd.read_csv(base/rn/'traces/dff_traces_curated_bgsub.csv'))); accs.append(a)
        accel=np.concatenate(accs)[:T]; g=np.median(X,1); beta=(X*g[:,None]).mean(0)/(g.var()+1e-9); resid=X-np.outer(g,beta)
        ks=np.random.randint(15,T-15,400)
        for i in range(N):
            ob=np.corrcoef(resid[:,i],accel)[0,1]; nd=np.array([abs(np.corrcoef(np.roll(resid[:,i],k),accel)[0,1]) for k in ks])
            cz[i]=(abs(ob)-nd.mean())/(nd.std()+1e-9)
            csign[i]= 'pos' if (abs(ob)>np.percentile(nd,95) and ob>0) else ('neg' if (abs(ob)>np.percentile(nd,95) and ob<0) else 'unc')
    counts=np.zeros(N,int)
    for i,nm in enumerate(names):
        ev=detail(X[:,i]); counts[i]=len(ev)
        nl=ng=0; onsets=[]
        for (on,amp,dur,auc) in ev:
            onsets.append(on); co=ac[max(0,on-1):on+2].max()
            if co>=CUT*N: ng+=1
            elif co<=2: nl+=1
        iei=np.diff(onsets) if len(onsets)>1 else np.array([])
        c=coords[i]; pts=c*VOX; ctr=pts.mean(0)
        orient=np.nan
        if len(c)>=6:
            w,V=np.linalg.eigh(np.cov((pts-ctr).T)); pa=V[:,np.argmax(w)]; orient=np.degrees(np.arccos(min(abs(pa[1]),1)))
        amps=[e[1] for e in ev]; durs=[e[2] for e in ev]; aucs=[e[3] for e in ev]
        mrows.append(dict(fov=fov,mouse=mo,name=nm,n_vox=len(c),depth_um=ctr[1],orient_deg=orient,SNR=snr(X[:,i]),
            n_events=len(ev),event_rate=len(ev)/mins,rate_local=nl/mins,rate_global=ng/mins,
            frac_global=ng/len(ev) if ev else np.nan, lg_pref=(nl-ng)/(nl+ng) if (nl+ng) else np.nan,
            mean_amp=np.mean(amps) if amps else np.nan,mean_dur=np.mean(durs) if durs else np.nan,mean_auc=np.mean(aucs) if aucs else np.nan,
            iei_cv=(iei.std()/iei.mean()) if iei.size and iei.mean()>0 else np.nan,
            burstiness=((iei.std()-iei.mean())/(iei.std()+iei.mean())) if iei.size and (iei.std()+iei.mean())>0 else np.nan,
            coupling_z=cz[i],coupling_sign=csign[i]))
    # hotspot summary
    tot=counts.sum(); order=np.argsort(counts)[::-1]
    share=lambda f: counts[order[:max(1,int(np.ceil(f*N)))]].sum()/tot if tot else np.nan
    half=T//2; c1=np.array([len(detail(X[:half,i])) for i in range(N)]); c2=np.array([len(detail(X[half:,i])) for i in range(N)])
    sp_half=stats.spearmanr(c1,c2).statistic
    sp_run=np.nan
    if len(runX)>=2:
        ra=np.array([len(detail(runX[0][:,i])) for i in range(N)]); rb=np.array([len(detail(runX[1][:,i])) for i in range(N)])
        sp_run=stats.spearmanr(ra,rb).statistic
    hrows.append(dict(fov=fov,mouse=mo,N=N,total_events=int(tot),gini=round(gini(counts),3),
        top5pct=round(share(.05),3),top10pct=round(share(.10),3),top20pct=round(share(.20),3),
        halfstab_spearman=round(sp_half,3),runstab_spearman=round(sp_run,3) if not np.isnan(sp_run) else np.nan))
M=pd.DataFrame(mrows); M.to_csv(PR/"scape-data"/"mask_fingerprint_table.csv",index=False)
H=pd.DataFrame(hrows); H.to_csv(PR/"scape-data"/"hotspot_summary.csv",index=False)
pd.set_option('display.width',220,'display.max_columns',30)
print("=== HOTSPOT SUMMARY ==="); print(H.to_string(index=False))
print(f"\nmask_fingerprint_table.csv: {len(M)} masks x {M.shape[1]} features")
print("Gini range:",round(H.gini.min(),2),"-",round(H.gini.max(),2),"| top10% event share:",round(H.top10pct.min(),2),"-",round(H.top10pct.max(),2))
