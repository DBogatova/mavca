#!/usr/bin/env python
"""Figure 1B (population): movement coupling across all shared-mask-group FOVs.
Uses event-locked BTA (not full-trace Pearson, which is unreliable across frequency bands).
Panels: (A) movement-onset BTA of global Ca per FOV; (B) movement-locked global response
amplitude per FOV (signed); (C) % branches movement-coupled raw vs residual per FOV;
(D) frac_global per FOV. Reads per-mask coupling from behavior_screen.csv."""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS = eb.FS; W = 15
FOVS = [("0508_r56","2026-05-08","rbp4_139_phpeb",["run5","run6"]),
        ("0512_r56","2026-05-12","rbp4_132_phpeb",["run5","run6"]),
        ("0512_r910","2026-05-12","rbp4_132_phpeb",["run9","run10"]),
        ("0416_r567","2026-04-16","rbp4_132_phpeb",["run5","run6","run7"]),
        ("0331_r8910","2026-03-31","rbp4_132_phpeb",["run8","run9","run10"]),
        ("0331_r67","2026-03-31","rbp4_132_phpeb",["run6","run7"])]
S = pd.read_csv(PR/"scape-data"/"behavior_screen.csv").set_index('fov')

def pool(date, mouse, runs):
    Xs, accs, bnd, off = [], [], [], 0; names=None
    for rn in runs:
        b=PR/"scape-data"/date/mouse/rn; f=b/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f); 
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        nm=[c for c in df.columns if 'dend' in c]; names=names or nm
        if not all(n in df.columns for n in names): continue
        X=df[names].values.astype(float); a,_=eb.load_behavior(b,rn,len(X))
        Xs.append(X); accs.append(a if a is not None else np.full(len(X),np.nan)); off+=len(X); bnd.append(off)
    return np.vstack(Xs), np.concatenate(accs), bnd[:-1]

def onsets(sig,T,bnd):
    s=(sig-np.nanmean(sig))/(np.nanstd(sig)+1e-9); rise=np.where(np.diff((s>0.5).astype(int))==1)[0]+1
    out=[]
    for r in rise:
        if W<=r<T-W and not any(abs(r-b)<W for b in bnd) and (not out or r-out[-1]>=10): out.append(r)
    return out

tt=(np.arange(2*W+1)-W)/FS; btas={}; peakamp={}
for lab,d,mo,runs in FOVS:
    X,accel,bnd=pool(d,mo,runs); T=len(X); g=np.median(X,1)
    gz=(g-g.mean())/g.std(); ons=onsets(accel,T,bnd)
    seg=np.array([gz[o-W:o+W+1] for o in ons]); seg=seg-seg[:,:W//2].mean(1,keepdims=True)
    btas[lab]=(seg.mean(0),seg.std(0)/np.sqrt(len(ons)))
    peakamp[lab]=seg[:, W:W+4].mean()           # 0..0.6s post-onset

labs=[f[0] for f in FOVS]; col=dict(zip(labs,plt.cm.tab10(np.linspace(0,1,len(labs)))))
fig,ax=plt.subplots(2,2,figsize=(13,9))
a=ax[0,0]
for lab in labs: m,e=btas[lab]; a.plot(tt,m,color=col[lab],label=lab); a.fill_between(tt,m-e,m+e,color=col[lab],alpha=.12)
a.axvline(0,color='gray',lw=.6); a.set_xlabel('time from movement onset (s)'); a.set_ylabel('global Ca (z)'); a.set_title('(A) Movement-onset global-Ca response (event-locked)'); a.legend(fontsize=7)
a=ax[0,1]; vals=[peakamp[l] for l in labs]; a.bar(range(len(labs)),vals,color=[col[l] for l in labs])
a.axhline(0,color='k',lw=.6); a.set_xticks(range(len(labs))); a.set_xticklabels(labs,rotation=30,fontsize=7); a.set_ylabel('global Ca @0–0.6s (z)'); a.set_title('(B) Movement-locked global response (signed)\nonly 0508 clearly positive')
a=ax[1,0]; x=np.arange(len(labs)); w=.38
a.bar(x-w/2,[S.loc[l,'raw_cpl'] for l in labs],w,label='raw',color='tab:orange')
a.bar(x+w/2,[S.loc[l,'res_cpl'] for l in labs],w,label='residual',color='tab:blue')
a.set_xticks(x); a.set_xticklabels(labs,rotation=30,fontsize=7); a.set_ylabel('% branches movement-coupled'); a.legend(fontsize=8)
med=np.median([S.loc[l,'res_cpl'] for l in labs]); a.axhline(med,color='r',ls='--',lw=.8)
a.set_title(f'(C) Branch-specific (residual) coupling across FOVs\nmedian residual={med:.0f}%')
a=ax[1,1]; a.bar(range(len(labs)),[S.loc[l,'frac_global'] for l in labs],color=[col[l] for l in labs])
a.set_xticks(range(len(labs))); a.set_xticklabels(labs,rotation=30,fontsize=7); a.set_ylabel('frac global events'); a.set_title('(D) Global-event fraction (only 0508/0331_r67 > 0)')
fig.suptitle('Figure 1B. Population view: branch-specific movement coupling is widespread; '
             'global movement-activation is FOV-specific (0508).', fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96]); out=PR/"scape-data"/"figures"; out.mkdir(exist_ok=True)
fig.savefig(out/"fig1b_population_behavior.png",dpi=160); plt.close(fig)
print("movement-locked global response (z @0-0.6s):", {l:round(peakamp[l],2) for l in labs})
print("residual-coupled %:", {l:int(S.loc[l,'res_cpl']) for l in labs}, " median", med)
print(f"saved {out/'fig1b_population_behavior.png'}")
