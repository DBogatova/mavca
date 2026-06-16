#!/usr/bin/env python
"""Figure: dendritic segments are largely independent functional units across mice/sessions.
A eff_dim/N per FOV (by mouse); B mean pairwise r per FOV; C pooled pairwise-r distribution vs
contaminated FOV; D cumulative PCA variance curves; E eff_dim/N grouped by mouse; F
compartmentalization (residual r after global regression). 0416 run1 = contaminated contrast."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
# (label, mouse, trace_path, contaminated?)
FOVS=[("0416 r5/6/7","rbp4_132","2026-04-16/rbp4_132_phpeb/run7",False),
      ("0331 r6/7","rbp4_132","2026-03-31/rbp4_132_phpeb/run7",False),
      ("0331 r8-10","rbp4_132","2026-03-31/rbp4_132_phpeb/run8",False),
      ("0512 r5/6","rbp4_132","2026-05-12/rbp4_132_phpeb/run5",False),
      ("0512 r9/10","rbp4_132","2026-05-12/rbp4_132_phpeb/run9",False),
      ("0508 r5/6","rbp4_139","2026-05-08/rbp4_139_phpeb/run5",False),
      ("0209","rbp4cre_136","2026-02-09/rbp4cre_136_phpeb/run1",False),
      ("0217","rbp4cre_138","2026-02-17/rbp4cre_138_phpeb/run7",False),
      ("0416 r1 (contam.)","rbp4_132","2026-04-16/rbp4_132_phpeb/run1",True)]
MICE=["rbp4_132","rbp4_139","rbp4cre_136","rbp4cre_138"]; col=dict(zip(MICE,plt.cm.tab10(range(4))))
rows=[]; cumvars={}; rdist={}
for lab,mo,pth,contam in FOVS:
    df=pd.read_csv(PR/"scape-data"/pth/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    X=df[[c for c in df.columns if 'dend' in c]].values.astype(float); T,N=X.shape
    cm=np.corrcoef(X.T); iu=np.triu_indices(N,1); mean_r=cm[iu].mean()
    sd=X.std(0); sd[sd==0]=1; Z=(X-X.mean(0))/sd; ev=PCA().fit(Z).explained_variance_ratio_
    eff=1/np.sum(ev**2)
    frac_strong=np.mean(np.abs(cm[iu])>0.3)
    rows.append(dict(lab=lab,mouse=mo,N=N,mean_r=mean_r,eff=eff/N,frac_strong=frac_strong,contam=contam))
    cumvars[lab]=(np.cumsum(ev),contam); rdist[lab]=(cm[iu],contam)
D=pd.DataFrame(rows); real=D[~D.contam]
fig=plt.figure(figsize=(15,9))
# A eff_dim/N
ax=fig.add_subplot(2,3,1); x=range(len(D))
ax.bar(x,D.eff,color=[col[m] if not c else '0.7' for m,c in zip(D.mouse,D.contam)],hatch=['' if not c else '//' for c in D.contam])
ax.set_xticks(x); ax.set_xticklabels(D.lab,rotation=40,ha='right',fontsize=6); ax.set_ylabel('effective dim / N'); ax.set_title('A  Effective dimensionality (high = independent)')
ax.axhline(real.eff.min(),color='g',ls=':',lw=.8)
# B mean r
ax=fig.add_subplot(2,3,2); ax.bar(x,D.mean_r,color=[col[m] if not c else '0.7' for m,c in zip(D.mouse,D.contam)],hatch=['' if not c else '//' for c in D.contam])
ax.set_xticks(x); ax.set_xticklabels(D.lab,rotation=40,ha='right',fontsize=6); ax.set_ylabel('mean pairwise r'); ax.set_title('B  Mean pairwise correlation (low = independent)')
# C pairwise-r distribution
ax=fig.add_subplot(2,3,3)
allr=np.concatenate([rdist[l][0] for l in D.lab if not rdist[l][1]])
ax.hist(allr,bins=60,density=True,color='steelblue',alpha=.7,label='real FOVs (all pairs)')
cl=[l for l in D.lab if rdist[l][1]][0]; ax.hist(rdist[cl][0],bins=40,density=True,color='red',alpha=.5,label='contaminated FOV')
ax.axvline(0,color='k',lw=.6); ax.set_xlabel('pairwise correlation r'); ax.set_title(f'C  Pairwise r distribution\n(real median={np.median(allr):.02f})'); ax.legend(fontsize=7)
# D cumulative variance
ax=fig.add_subplot(2,3,4)
for l in D.lab:
    cv,c=cumvars[l]; ax.plot(np.arange(1,len(cv)+1),cv,color='red' if c else '0.5',lw=1.5 if c else .8,alpha=.9 if c else .6)
ax.set_xlim(0,40); ax.set_xlabel('# principal components'); ax.set_ylabel('cumulative variance'); ax.set_title('D  PCA cumulative variance\n(red=contaminated: PC1≈90%)')
# E eff by mouse
ax=fig.add_subplot(2,3,5)
mm=real.groupby('mouse').eff; means=[mm.get_group(m).mean() if m in mm.groups else np.nan for m in MICE]
errs=[mm.get_group(m).std() if m in mm.groups and len(mm.get_group(m))>1 else 0 for m in MICE]
ax.bar(range(4),means,yerr=errs,color=[col[m] for m in MICE],capsize=4)
for i,m in enumerate(MICE):
    if m in mm.groups: ax.scatter([i]*len(mm.get_group(m)),mm.get_group(m),c='k',s=12,zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels(MICE,rotation=20,fontsize=7); ax.set_ylabel('effective dim / N'); ax.set_title('E  Independence robust across 4 mice')
# F fraction of strongly-correlated pairs
ax=fig.add_subplot(2,3,6); ax.bar(x,D.frac_strong*100,color=[col[m] if not c else '0.7' for m,c in zip(D.mouse,D.contam)],hatch=['' if not c else '//' for c in D.contam])
ax.set_xticks(x); ax.set_xticklabels(D.lab,rotation=40,ha='right',fontsize=6); ax.set_ylabel('% pairs with |r|>0.3'); ax.set_title('F  Strongly-correlated pairs are rare (real) vs ubiquitous (contam.)')
fig.suptitle('Figure. Dendritic segments are largely independent functional units across mice, sessions, acquisition modes',fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96]); out=PR/"scape-data"/"figures"; fig.savefig(out/"fig6_independence.png",dpi=160); plt.close(fig)
print(D[['lab','mouse','N','mean_r','eff','frac_strong']].round(3).to_string(index=False))
print(f"\nreal FOVs: eff_dim/N range {real.eff.min():.2f}-{real.eff.max():.2f}; mean_r {real.mean_r.min():.02f}-{real.mean_r.max():.02f}; n_FOVs={len(real)} across {real.mouse.nunique()} mice")
print(f"saved {out/'fig6_independence.png'}")
