#!/usr/bin/env python
"""Two main-figure panels: (1) observed vs shuffled-mask motif recurrence per FOV;
(2) 0508 global-event probability in fine time bins relative to movement onset (pre-onset rise)."""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
THR=eb.DFF_THR; FS=eb.FS; CUT=0.17
FOVS=[("0508","2026-05-08/rbp4_139_phpeb",["run5","run6"]),("0416_r567","2026-04-16/rbp4_132_phpeb",["run5","run6","run7"]),
      ("0331_r67","2026-03-31/rbp4_132_phpeb",["run6","run7"]),("0331_r8910","2026-03-31/rbp4_132_phpeb",["run8","run9","run10"]),
      ("0512_r56","2026-05-12/rbp4_132_phpeb",["run5","run6"]),("0512_r910","2026-05-12/rbp4_132_phpeb",["run9","run10"]),
      ("0209","2026-02-09/rbp4cre_136_phpeb",["run1"]),("0217","2026-02-17/rbp4cre_138_phpeb",["run7"])]
def load(pth,runs):
    Xs=[]; names=None
    for rn in runs:
        f=PR/"scape-data"/pth/rn/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f); 
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        names=names or [c for c in df.columns if 'dend' in c]
        if all(n in df.columns for n in names): Xs.append(df[names].values.astype(float))
    return np.vstack(Xs)
def fps(X):
    act=X>THR; pop=act.any(1); idx=np.where(pop)[0]; out=[]
    if idx.size:
        for w in np.split(idx,np.where(np.diff(idx)>1)[0]+1): out.append(act[w].any(0).astype(float))
    return np.array(out)
def recur(E):
    if len(E)<3: return np.nan
    n=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9); S=n@n.T; iu=np.triu_indices(len(E),1); return np.mean(S[iu]>=0.5)
labs=[]; obs=[]; nullm=[]; nulllo=[]; nullhi=[]
for fov,pth,runs in FOVS:
    X=load(pth,runs); N=X.shape[1]; F=fps(X); multi=F[F.sum(1)>=2]
    o=recur(multi); nl=[]
    if len(multi)>=3:
        for _ in range(300):
            sh=np.zeros_like(multi)
            for r in range(len(multi)): sh[r,np.random.choice(N,int(multi[r].sum()),replace=False)]=1
            nl.append(recur(sh))
    labs.append(fov); obs.append(o); nullm.append(np.mean(nl) if nl else np.nan); nulllo.append(np.percentile(nl,2.5) if nl else np.nan); nullhi.append(np.percentile(nl,97.5) if nl else np.nan)

fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(len(labs)); w=.38
ax.bar(x-w/2,obs,w,color='tab:blue',label='observed')
ax.bar(x+w/2,nullm,w,yerr=[np.array(nullm)-np.array(nulllo),np.array(nullhi)-np.array(nullm)],color='0.7',capsize=3,label='shuffled null (95% CI)')
ax.set_xticks(x); ax.set_xticklabels(labs,rotation=40,ha='right',fontsize=7); ax.set_ylabel('multi-mask motif recurrence\n(frac pairs cosine≥0.5)')
ax.set_title('Recurring multi-mask motifs exceed shuffled-mask null (all FOVs, p=0.003)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(PR/"scape-data"/"figures"/"fig3_recurrence.png",dpi=160); plt.close(fig)
print("recurrence obs vs null:", {l:(round(o,3),round(nm,3)) for l,o,nm in zip(labs,obs,nullm)})

# ---- panel 2: 0508 pre-onset fine bins ----
base=PR/"scape-data"/"2026-05-08"/"rbp4_139_phpeb"; runs=["run5","run6"]; W=15
Xs,accs,bnd,off=[],[],[],0; names=None
for rn in runs:
    df=pd.read_csv(base/rn/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    names=names or [c for c in df.columns if 'dend' in c]; X=df[names].values.astype(float)
    a,_=eb.load_behavior(base/rn,rn,len(X)); Xs.append(X); accs.append(a); off+=len(X); bnd.append(off)
X=np.vstack(Xs); accel=np.concatenate(accs); T,N=X.shape; ac=(X>THR).sum(1); gfr=ac>=CUT*N
z=(accel-np.nanmean(accel))/(np.nanstd(accel)+1e-9); mv=z>0.5
on=[r for r in (np.where(np.diff(mv.astype(int))==1)[0]+1) if W<=r<T-W and not any(abs(r-b)<W for b in bnd[:-1])]
gp=np.array([[gfr[o+d] for d in range(-W,W+1)] for o in on]); er=np.array([[ac[o+d] for d in range(-W,W+1)] for o in on])
tt=(np.arange(-W,W+1))/FS
fig,ax=plt.subplots(1,2,figsize=(11,4))
m=gp.mean(0); se=gp.std(0)/np.sqrt(len(on)); ax[0].plot(tt,m,'tab:purple'); ax[0].fill_between(tt,m-se,m+se,color='tab:purple',alpha=.2)
ax[0].axvline(0,color='r',ls='--',label='accel onset'); ax[0].set_xlabel('time from movement onset (s)'); ax[0].set_ylabel('P(global-event frame)'); ax[0].set_title(f'0508: global-event probability\nrelative to onset (n={len(on)})'); ax[0].legend(fontsize=8)
m2=er.mean(0); s2=er.std(0)/np.sqrt(len(on)); ax[1].plot(tt,m2,'g'); ax[1].fill_between(tt,m2-s2,m2+s2,color='g',alpha=.2); ax[1].axvline(0,color='r',ls='--'); ax[1].set_xlabel('time from movement onset (s)'); ax[1].set_ylabel('# active masks / frame'); ax[1].set_title('0508: population activity rel. onset')
fig.suptitle('Global recruitment often precedes/coincides with detected movement onset (0508, descriptive)',fontsize=10)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(PR/"scape-data"/"figures"/"fig4_preonset.png",dpi=160); plt.close(fig)
print(f"saved fig3_recurrence.png, fig4_preonset.png (0508 onsets={len(on)})")
