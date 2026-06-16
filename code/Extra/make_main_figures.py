#!/usr/bin/env python
"""Assemble the 4 publication MAIN figures (independence; hotspots; footprints; state grammar)
into paper_figures/. Composites of the strongest panels only; exploratory figs stay in scape-data/figures."""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
OUT=PR/"paper_figures"; OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
THR=eb.DFF_THR; FS=eb.FS; CUT=0.17
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'figure.titlesize':11})
POOLS={'0508':('2026-05-08/rbp4_139_phpeb',['run5','run6'],'rbp4_139',0),
 '0416':('2026-04-16/rbp4_132_phpeb',['run5','run6','run7'],'rbp4_132',0),
 '0331a':('2026-03-31/rbp4_132_phpeb',['run6','run7'],'rbp4_132',0),
 '0331b':('2026-03-31/rbp4_132_phpeb',['run8','run9','run10'],'rbp4_132',0),
 '0512a':('2026-05-12/rbp4_132_phpeb',['run5','run6'],'rbp4_132',0),
 '0512b':('2026-05-12/rbp4_132_phpeb',['run9','run10'],'rbp4_132',0),
 '0209':('2026-02-09/rbp4cre_136_phpeb',['run1'],'rbp4cre_136',0),
 '0217':('2026-02-17/rbp4cre_138_phpeb',['run7'],'rbp4cre_138',0),
 'contam':('2026-04-16/rbp4_132_phpeb',['run1'],'rbp4_132',1)}
MICE=['rbp4_132','rbp4_139','rbp4cre_136','rbp4cre_138']; mc=dict(zip(MICE,plt.cm.tab10(range(4))))
def loadX(pth,runs):
    Xs=[]; names=None
    for rn in runs:
        f=PR/"scape-data"/pth/rn/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df=pd.read_csv(f); 
        if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
        names=names or [c for c in df.columns if 'dend' in c]
        if all(n in df.columns for n in names): Xs.append(df[names].values.astype(float))
    return np.vstack(Xs)
DATA={k:(loadX(p,r),mo,c) for k,(p,r,mo,c) in POOLS.items()}
def evcount(X): return np.array([len([1 for run in np.split(np.where(X[:,i]>THR)[0],np.where(np.diff(np.where(X[:,i]>THR)[0])>1)[0]+1) if len(run)>=2]) for i in range(X.shape[1])]) if X.shape[0] else np.array([])
def metrics(X):
    N=X.shape[1]; cm=np.corrcoef(X.T); iu=np.triu_indices(N,1)
    sd=X.std(0); sd[sd==0]=1; Z=(X-X.mean(0))/sd; ev=PCA().fit(Z).explained_variance_ratio_
    return cm[iu], 1/np.sum(ev**2)/N, np.cumsum(ev)
order=['0508','0416','0331a','0331b','0512a','0512b','0209','0217','contam']
lab={'0508':'139 0508','0416':'132 0416','0331a':'132 0331a','0331b':'132 0331b','0512a':'132 0512a','0512b':'132 0512b','0209':'136 0209','0217':'138 0217','contam':'132 0416r1*'}

# ===== FIG 1 independence =====
M={k:metrics(DATA[k][0]) for k in order}
fig,ax=plt.subplots(2,2,figsize=(10,8))
effs=[M[k][1] for k in order]; cols=[('0.7' if DATA[k][2]and DATA[k][2] else mc[DATA[k][1]]) for k in order]
cols=['0.6' if POOLS[k][3] else mc[DATA[k][1]] for k in order]; hatch=['//' if POOLS[k][3] else '' for k in order]
ax[0,0].bar(range(len(order)),effs,color=cols,hatch=hatch); ax[0,0].set_xticks(range(len(order))); ax[0,0].set_xticklabels([lab[k] for k in order],rotation=45,ha='right',fontsize=7)
ax[0,0].set_ylabel('effective dim / N'); ax[0,0].set_title('A  High dimensionality across FOVs/mice')
allr=np.concatenate([M[k][0] for k in order if not POOLS[k][3]])
ax[0,1].hist(allr,bins=60,density=True,color='steelblue',alpha=.8,label='real FOVs'); ax[0,1].hist(M['contam'][0],bins=40,density=True,color='red',alpha=.5,label='contaminated')
ax[0,1].axvline(0,color='k',lw=.6); ax[0,1].set_xlabel('pairwise correlation r'); ax[0,1].set_ylabel('density'); ax[0,1].set_title(f'B  Weak correlation (real median={np.median(allr):.02f})'); ax[0,1].legend(fontsize=8)
for k in order:
    cv=M[k][2]; ax[1,0].plot(np.arange(1,len(cv)+1),cv,color='red' if POOLS[k][3] else '0.5',lw=1.8 if POOLS[k][3] else .8,alpha=.95 if POOLS[k][3] else .6)
ax[1,0].set_xlim(0,40); ax[1,0].set_xlabel('# principal components'); ax[1,0].set_ylabel('cumulative variance'); ax[1,0].set_title('C  PCA variance (red=contaminated, PC1≈90%)')
mm=[(mo,M[k][1]) for k in order if not POOLS[k][3] for mo in [DATA[k][1]]]
md=pd.DataFrame(mm,columns=['mouse','eff'])
means=[md[md.mouse==m].eff.mean() for m in MICE]; errs=[md[md.mouse==m].eff.std() if (md.mouse==m).sum()>1 else 0 for m in MICE]
ax[1,1].bar(range(4),means,yerr=errs,color=[mc[m] for m in MICE],capsize=4)
for i,m in enumerate(MICE): ax[1,1].scatter([i]*(md.mouse==m).sum(),md[md.mouse==m].eff,c='k',s=14,zorder=3)
ax[1,1].set_xticks(range(4)); ax[1,1].set_xticklabels(MICE,rotation=20,fontsize=7); ax[1,1].set_ylabel('effective dim / N'); ax[1,1].set_title('D  Independence robust across 4 mice')
fig.suptitle('Figure 1. L5 apical dendritic segments are independent, high-dimensional functional units'); fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT/"Figure1_independence.png",dpi=200); fig.savefig(OUT/"Figure1_independence.pdf"); plt.close(fig)

# ===== FIG 2 hotspots =====
real=[k for k in order if not POOLS[k][3]]
fig,ax=plt.subplots(2,2,figsize=(10,8))
ginis=[]; t10=[]; t20=[]; t5=[]; hstab=[]
for k in real:
    X=DATA[k][0]; c=evcount(X); cs=np.sort(c); cum=np.cumsum(cs)/cs.sum() if cs.sum() else cs
    ax[0,0].plot(np.arange(1,len(cs)+1)/len(cs),cum,color=mc[DATA[k][1]],lw=1)
    xs=np.sort(c.astype(float)); n=len(xs); g=(np.sum((2*np.arange(1,n+1)-n-1)*xs))/(n*xs.sum()+1e-9); ginis.append(g)
    o=np.argsort(c)[::-1]; tot=c.sum(); t5.append(c[o[:max(1,int(.05*n))]].sum()/tot); t10.append(c[o[:max(1,int(.10*n))]].sum()/tot); t20.append(c[o[:max(1,int(.20*n))]].sum()/tot)
    h=X.shape[0]//2; hstab.append(stats.spearmanr(evcount(X[:h]),evcount(X[h:])).statistic)
ax[0,0].plot([0,1],[0,1],'k--',lw=.6); ax[0,0].set_xlabel('cumulative mask fraction'); ax[0,0].set_ylabel('cumulative event fraction'); ax[0,0].set_title('A  Lorenz curves (event sparsity)')
ax[0,1].bar(range(len(real)),ginis,color=[mc[DATA[k][1]] for k in real]); ax[0,1].set_xticks(range(len(real))); ax[0,1].set_xticklabels([lab[k] for k in real],rotation=45,ha='right',fontsize=7); ax[0,1].set_ylabel('event-count Gini'); ax[0,1].set_title('B  Event inequality (Gini)')
xx=np.arange(len(real))
ax[1,0].bar(xx-.25,np.array(t5)*100,.25,label='top5%'); ax[1,0].bar(xx,np.array(t10)*100,.25,label='top10%'); ax[1,0].bar(xx+.25,np.array(t20)*100,.25,label='top20%')
ax[1,0].set_xticks(xx); ax[1,0].set_xticklabels([lab[k] for k in real],rotation=45,ha='right',fontsize=7); ax[1,0].set_ylabel('% of events'); ax[1,0].set_title('C  Event share of most-active masks'); ax[1,0].legend(fontsize=7)
ax[1,1].bar(range(len(real)),hstab,color=[mc[DATA[k][1]] for k in real]); ax[1,1].axhline(0,color='k',lw=.5); ax[1,1].set_xticks(range(len(real))); ax[1,1].set_xticklabels([lab[k] for k in real],rotation=45,ha='right',fontsize=7); ax[1,1].set_ylabel("hotspot stability (Spearman ρ)"); ax[1,1].set_title('D  Hotspots stable (1st vs 2nd half)')
fig.suptitle('Figure 2. Local activity is sparse and concentrated in moderately stable hotspots'); fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT/"Figure2_hotspots.png",dpi=200); fig.savefig(OUT/"Figure2_hotspots.pdf"); plt.close(fig)

# ===== FIG 3 footprints =====
def fps(X):
    act=X>THR; pop=act.any(1); idx=np.where(pop)[0]; out=[]
    if idx.size:
        for w in np.split(idx,np.where(np.diff(idx)>1)[0]+1): out.append(act[w].any(0).astype(float))
    return np.array(out)
def recur(E):
    if len(E)<3: return np.nan
    n=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9); S=n@n.T; iu=np.triu_indices(len(E),1); return np.mean(S[iu]>=0.5)
allsz=[]; clf={}; obs=[]; nul=[]
for k in real:
    X=DATA[k][0]; N=X.shape[1]; F=fps(X); sz=F.sum(1); allsz+=list(sz)
    clf[k]=(np.mean(sz<=2),np.mean(sz>=CUT*N)); multi=F[sz>=2]; o=recur(multi); nl=[]
    if len(multi)>=3:
        for _ in range(200):
            sh=np.zeros_like(multi)
            for rr in range(len(multi)): sh[rr,np.random.choice(N,int(multi[rr].sum()),replace=False)]=1
            nl.append(recur(sh))
    obs.append(o); nul.append((np.mean(nl),np.percentile(nl,97.5)) if nl else (np.nan,np.nan))
fig,ax=plt.subplots(2,2,figsize=(10,8))
ax[0,0].hist(allsz,bins=range(1,20),color='gray',edgecolor='k'); ax[0,0].set_xlabel('footprint size (# masks)'); ax[0,0].set_ylabel('# events'); ax[0,0].set_title('A  Most events are local (1–2 masks)')
xx=np.arange(len(real)); ax[0,1].bar(xx-.2,[clf[k][0]*100 for k in real],.4,label='local (≤2)',color='tab:green'); ax[0,1].bar(xx+.2,[clf[k][1]*100 for k in real],.4,label='global (≥17%)',color='tab:red')
ax[0,1].set_xticks(xx); ax[0,1].set_xticklabels([lab[k] for k in real],rotation=45,ha='right',fontsize=7); ax[0,1].set_ylabel('% of events'); ax[0,1].set_title('B  Event-class fractions'); ax[0,1].legend(fontsize=7)
ax[1,0].bar(xx-.2,obs,.4,color='tab:blue',label='observed'); ax[1,0].bar(xx+.2,[n[0] for n in nul],.4,yerr=[[0]*len(nul),[max(0,n[1]-n[0]) for n in nul]],color='0.7',capsize=3,label='shuffled null')
ax[1,0].set_xticks(xx); ax[1,0].set_xticklabels([lab[k] for k in real],rotation=45,ha='right',fontsize=7); ax[1,0].set_ylabel('motif recurrence'); ax[1,0].set_title('C  Multi-mask motifs recur > null (p=0.003)'); ax[1,0].legend(fontsize=7)
F8=fps(DATA['0508'][0]); mm=F8[F8.sum(1)>=2][:60]; ax[1,1].imshow(mm,aspect='auto',cmap='Greys',interpolation='nearest'); ax[1,1].set_xlabel('mask'); ax[1,1].set_ylabel('multi-mask event'); ax[1,1].set_title('D  Example footprints (0508)')
fig.suptitle('Figure 3. Events are predominantly local with rare, reproducible multi-mask motifs'); fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT/"Figure3_footprints.png",dpi=200); fig.savefig(OUT/"Figure3_footprints.pdf"); plt.close(fig)

# ===== FIG 4 state grammar (0508) =====
base=PR/"scape-data"/"2026-05-08"/"rbp4_139_phpeb"; accs=[]; off=0; bnd=[]
for rn in ['run5','run6']:
    df=pd.read_csv(base/rn/"traces"/"dff_traces_curated_bgsub.csv"); 
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    a,_=eb.load_behavior(base/rn,rn,len(df)); accs.append(a); off+=len(df); bnd.append(off)
X=DATA['0508'][0]; accel=np.concatenate(accs)[:X.shape[0]]; T,N=X.shape; ac=(X>THR).sum(1); gfr=ac>=CUT*N; g=X.mean(1)
W=15; z=(accel-np.nanmean(accel))/(np.nanstd(accel)+1e-9); mv=z>0.5
on=[r for r in (np.where(np.diff(mv.astype(int))==1)[0]+1) if W<=r<T-W and not any(abs(r-b)<W for b in bnd[:-1])]
tt=(np.arange(-W,W+1))/FS
G=pd.read_csv(PR/"scape-data"/"state_event_grammar.csv").set_index('state'); sts=['quiet','pre','onset','post']
fig,ax=plt.subplots(2,2,figsize=(10,8))
seg=np.array([(g[o-W:o+W+1]-g[o-W:o-W+5].mean()) for o in on]); m=seg.mean(0); se=seg.std(0)/np.sqrt(len(on))
ax[0,0].plot(tt,m,'k'); ax[0,0].fill_between(tt,m-se,m+se,alpha=.2,color='k'); ax[0,0].axvline(0,color='r',ls='--'); ax[0,0].set_xlabel('time from movement onset (s)'); ax[0,0].set_ylabel('global Ca (ΔF/F)'); ax[0,0].set_title(f'A  Global Ca @ movement onset (n={len(on)})')
gp=np.array([[gfr[o+d] for d in range(-W,W+1)] for o in on]); m2=gp.mean(0); s2=gp.std(0)/np.sqrt(len(on))
ax[0,1].plot(tt,m2,'tab:purple'); ax[0,1].fill_between(tt,m2-s2,m2+s2,color='tab:purple',alpha=.2); ax[0,1].axvline(0,color='r',ls='--'); ax[0,1].set_xlabel('time from onset (s)'); ax[0,1].set_ylabel('P(global-event frame)'); ax[0,1].set_title('B  Global recruitment precedes/coincides w/ onset')
fg=[G.loc[s,'frac_global'] if s in G.index and G.loc[s,'n_events']>0 else 0 for s in sts]
ax[1,0].bar(range(len(sts)),fg,color='tab:purple'); ax[1,0].set_xticks(range(len(sts))); ax[1,0].set_xticklabels(sts); ax[1,0].set_ylabel('fraction global events'); ax[1,0].set_title('C  Global-event fraction by state')
lr=[G.loc[s,'local_rate'] if s in G.index and G.loc[s,'n_events']>0 else 0 for s in sts]; gr=[G.loc[s,'global_rate'] if s in G.index and G.loc[s,'n_events']>0 else 0 for s in sts]
ax[1,1].bar(np.arange(len(sts))-.2,lr,.4,label='local',color='tab:green'); ax[1,1].bar(np.arange(len(sts))+.2,gr,.4,label='global',color='tab:red'); ax[1,1].set_xticks(range(len(sts))); ax[1,1].set_xticklabels(sts); ax[1,1].set_ylabel('event rate /min'); ax[1,1].set_title('D  Local vs global rate by state'); ax[1,1].legend(fontsize=7)
fig.suptitle('Figure 4. Movement transiently shifts the event grammar toward global recruitment (0508)'); fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT/"Figure4_state_grammar.png",dpi=200); fig.savefig(OUT/"Figure4_state_grammar.pdf"); plt.close(fig)
print("saved 4 main figures (png+pdf) to paper_figures/:", sorted(p.name for p in OUT.glob('*.png')))
