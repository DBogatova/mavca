#!/usr/bin/env python
"""Q3 event footprints. Population event = contiguous epoch with >=1 active mask; footprint =
binary vector of masks active during the epoch. Event-event cosine similarity; test recurring
multi-mask motifs vs a shuffled-mask null (preserve per-event size, randomize identities).
Classify events: local (<=2 masks) / multi-mask / global (>=17% masks). No family interpretation."""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
THR=eb.DFF_THR; CUT=0.17
FOVS=[("0508","2026-05-08/rbp4_139_phpeb",["run5","run6"]),
      ("0416_r567","2026-04-16/rbp4_132_phpeb",["run5","run6","run7"]),
      ("0331_r67","2026-03-31/rbp4_132_phpeb",["run6","run7"]),
      ("0331_r8910","2026-03-31/rbp4_132_phpeb",["run8","run9","run10"]),
      ("0512_r56","2026-05-12/rbp4_132_phpeb",["run5","run6"]),
      ("0512_r910","2026-05-12/rbp4_132_phpeb",["run9","run10"]),
      ("0209","2026-02-09/rbp4cre_136_phpeb",["run1"]),
      ("0217","2026-02-17/rbp4cre_138_phpeb",["run7"])]
def footprints(X):
    active=X>THR; pop=active.any(1); idx=np.where(pop)[0]; fps=[]; starts=[]
    if idx.size:
        for w in np.split(idx,np.where(np.diff(idx)>1)[0]+1):
            fp=active[w].any(0); 
            if fp.sum()>=1: fps.append(fp.astype(float)); starts.append(int(w[0]))
    return np.array(fps), starts
def recur(E):  # fraction of multi-mask event pairs with cosine>=0.5
    if len(E)<3: return np.nan
    n=E/ (np.linalg.norm(E,axis=1,keepdims=True)+1e-9); S=n@n.T; iu=np.triu_indices(len(E),1)
    return np.mean(S[iu]>=0.5)
rows=[]; mrows=[]; ex={}
for fov,pth,runs in FOVS:
    Xs=[]; names=None
    for rn in runs:
        f=PR/"scape-data"/pth/rn/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f); 
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        nm=[c for c in df.columns if 'dend' in c]; names=names or nm
        if all(n in df.columns for n in names): Xs.append(df[names].values.astype(float))
    X=np.vstack(Xs); N=X.shape[1]; fps,starts=footprints(X)
    sizes=fps.sum(1); 
    for k,(fp,st,sz) in enumerate(zip(fps,starts,sizes)):
        cls='local' if sz<=2 else ('global' if sz>=CUT*N else 'multi')
        rows.append(dict(fov=fov,event_id=k,t_start=st,size=int(sz),participation=sz/N,event_class=cls,
                         masks=";".join(np.array(names)[fp>0])))
    multi=fps[sizes>=2]
    obs=recur(multi); null=[]
    if len(multi)>=3:
        for _ in range(300):
            sh=np.zeros_like(multi)
            for r in range(len(multi)): sh[r,np.random.choice(N,int(multi[r].sum()),replace=False)]=1
            null.append(recur(sh))
        p=(np.sum(np.array(null)>=obs)+1)/(len(null)+1)
    else: p=np.nan
    fl=np.mean(sizes<=2); fg=np.mean(sizes>=CUT*N)
    mrows.append(dict(fov=fov,N=N,n_events=len(fps),frac_local=round(fl,2),frac_global=round(fg,2),
                      n_multi=int((sizes>=2).sum()),recur_obs=round(obs,3) if np.isfinite(obs) else np.nan,
                      recur_null=round(np.mean(null),3) if null else np.nan,recur_p=round(p,3) if np.isfinite(p) else np.nan))
    ex[fov]=(fps,sizes)
pd.DataFrame(rows).to_csv(PR/"scape-data"/"event_footprint_table.csv",index=False)
S=pd.DataFrame(mrows); pd.set_option('display.width',200)
print(S.to_string(index=False))
# figure
fig,ax=plt.subplots(1,3,figsize=(15,4.3))
allsz=pd.DataFrame(rows); ax[0].hist([allsz.size_ if False else allsz['size']],bins=range(1,20),color='gray'); ax[0].set_xlabel('footprint size (# masks)'); ax[0].set_ylabel('# events'); ax[0].set_title('A  Event footprint sizes (pooled)\nmost events local')
xx=np.arange(len(S)); ax[1].bar(xx-.2,S.frac_local,.4,label='local (<=2)'); ax[1].bar(xx+.2,S.frac_global,.4,label='global (>=17%)')
ax[1].set_xticks(xx); ax[1].set_xticklabels(S.fov,rotation=40,ha='right',fontsize=6); ax[1].set_ylabel('fraction of events'); ax[1].set_title('B  Event class fractions'); ax[1].legend(fontsize=7)
fp,sz=ex['0508']; mm=fp[sz>=2][:60]; ax[2].imshow(mm,aspect='auto',cmap='Greys',interpolation='nearest'); ax[2].set_xlabel('mask'); ax[2].set_ylabel('multi-mask event'); ax[2].set_title('C  0508 multi-mask event footprints')
fig.suptitle('Q3. Event footprints: mostly local; recurring multi-mask motifs tested vs shuffle null',fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(PR/"scape-data"/"figures"/"fig10_footprints.png",dpi=150); plt.close(fig)
print("saved event_footprint_table.csv, fig10_footprints.png")
