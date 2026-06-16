#!/usr/bin/env python
"""Q4 behavioral-state grammar + Q5 movement-coupling sign (0508 run5+run6; behavior only here).
Q4: bin frames into quiet/pre/onset/sustained/post (from accel); per state compute local & global
event rate, participation, amplitude, footprint diversity; test movement shifts local vs global.
Q5: pos/neg/uncoupled residual-coupled masks -> compare event_rate, lg_pref, amp, depth, SNR,
spatial layout; run5-vs-run6 sign stability."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
THR=eb.DFF_THR; FS=eb.FS; CUT=0.17; WB=5
base=PR/"scape-data"/"2026-05-08"/"rbp4_139_phpeb"; runs=["run5","run6"]
Xs,accs,bnd,off=[],[],[],0; names=None
for rn in runs:
    df=pd.read_csv(base/rn/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    names=names or [c for c in df.columns if 'dend' in c]; X=df[names].values.astype(float)
    a,_=eb.load_behavior(base/rn,rn,len(X)); Xs.append(X); accs.append(a); off+=len(X); bnd.append(off)
X=np.vstack(Xs); accel=np.concatenate(accs); T,N=X.shape; active=X>THR; ac=active.sum(1)
# ---- states ----
z=(accel-np.nanmean(accel))/(np.nanstd(accel)+1e-9); mv=z>0.5
state=np.array(['quiet']*T,object); dd=np.diff(mv.astype(int))
starts=list(np.where(dd==1)[0]+1); ends=list(np.where(dd==-1)[0]+1)
if mv[0]: starts=[0]+starts
if mv[-1]: ends=ends+[T]
for s,e in zip(starts,ends):
    state[max(0,s-WB):s]='pre'; state[s:min(s+WB,e)]='onset'; state[min(s+WB,e):e]='sustained'; state[e:min(e+WB,T)]='post'
# per-mask events with onset frame
evframes=[]; evmask=[]; evamp=[]
for i in range(N):
    idx=np.where(X[:,i]>THR)[0]
    if idx.size:
        for r in np.split(idx,np.where(np.diff(idx)>1)[0]+1):
            if r.size>=2: p=r[0]; evframes.append(p); evmask.append(i); evamp.append(float(X[r,i].max()))
evframes=np.array(evframes); evmask=np.array(evmask); evamp=np.array(evamp)
co=np.array([ac[max(0,p-1):p+2].max() for p in evframes]); est=np.array([state[p] for p in evframes])
grows=[]
for s in ['quiet','pre','onset','sustained','post']:
    m=est==s; nf=(state==s).sum(); mins=nf/FS/60
    if m.sum()==0 or mins==0: grows.append(dict(state=s,n_frames=int(nf),n_events=0)); continue
    loc=(co[m]<=2).sum(); glo=(co[m]>=CUT*N).sum()
    div=len(np.unique(evmask[m]))/m.sum()   # footprint richness: unique masks / events
    grows.append(dict(state=s,n_frames=int(nf),n_events=int(m.sum()),event_rate=round(m.sum()/mins,1),
        local_rate=round(loc/mins,1),global_rate=round(glo/mins,1),frac_global=round(glo/m.sum(),3),
        mean_participation=round(np.mean(co[m]/N),3),mean_amp=round(np.mean(evamp[m]),2),footprint_diversity=round(div,2)))
G=pd.DataFrame(grows); G.to_csv(PR/"scape-data"/"state_event_grammar.csv",index=False)
print("=== Q4 STATE GRAMMAR (0508) ==="); print(G.to_string(index=False))
# test movement shifts local vs global: quiet vs sustained
q=est=='quiet'; su=est=='sustained'
ct=np.array([[(co[q]<=2).sum(),(co[q]>=CUT*N).sum()],[(co[su]<=2).sum(),(co[su]>=CUT*N).sum()]])
if ct.min()>=1: chi=stats.chi2_contingency(ct); print(f"local-vs-global balance quiet vs sustained: chi2 p={chi[1]:.3g} (frac_global quiet={ct[0,1]/ct[0].sum():.2f} sustained={ct[1,1]/ct[1].sum():.2f})")

# ---- Q5 coupling sign ----
M=pd.read_csv(PR/"scape-data"/"mask_fingerprint_table.csv"); D=M[M.fov=='0508'].reset_index(drop=True)
print("\n=== Q5 COUPLING SIGN (0508) ==="); print(D.coupling_sign.value_counts().to_dict())
for f in ['event_rate','lg_pref','mean_amp','depth_um','SNR']:
    grp=[D[D.coupling_sign==s][f].dropna() for s in ['pos','neg','unc']]; grp=[g for g in grp if len(g)>1]
    if len(grp)>=2: print(f"  {f}: medians pos/neg/unc="+"/".join(f"{D[D.coupling_sign==s][f].median():.2f}" for s in ['pos','neg','unc'])+f"  KW p={stats.kruskal(*grp).pvalue:.2g}")
# run-to-run sign stability
def signs(rn):
    df=pd.read_csv(base/rn/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    Xr=df[names].values.astype(float); a,_=eb.load_behavior(base/rn,rn,len(Xr)); Tr=len(Xr)
    g=np.median(Xr,1); beta=(Xr*g[:,None]).mean(0)/(g.var()+1e-9); res=Xr-np.outer(g,beta); ks=np.random.randint(15,Tr-15,300); out=[]
    for i in range(N):
        ob=np.corrcoef(res[:,i],a)[0,1]; nd=np.array([abs(np.corrcoef(np.roll(res[:,i],k),a)[0,1]) for k in ks]); th=np.percentile(nd,95)
        out.append('pos' if (abs(ob)>th and ob>0) else ('neg' if (abs(ob)>th and ob<0) else 'unc'))
    return np.array(out)
s5,s6=signs("run5"),signs("run6"); agree=np.mean(s5==s6); psig=(s5!='unc')|(s6!='unc')
print(f"run5-vs-run6 sign agreement: all masks {agree*100:.0f}%; among ever-coupled {np.mean(s5[psig]==s6[psig])*100:.0f}% (n={psig.sum()})")
# figure
cx=np.array([np.argwhere(tifffile.imread(base/'run5'/'labelmaps_curated_dynamic'/f'{n}_labelmap.tif')>0)[:,2].mean()*1.2 for n in names])
cmap={'pos':'tab:red','neg':'tab:blue','unc':'0.7'}
fig,ax=plt.subplots(1,3,figsize=(14,4.3))
ax[0].bar(['quiet','pre','onset','sustained','post'],[G.set_index('state').loc[s,'frac_global'] if s in G.state.values and G.set_index('state').loc[s,'n_events']>0 else 0 for s in ['quiet','pre','onset','sustained','post']],color='tab:purple')
ax[0].set_ylabel('fraction global events'); ax[0].set_title('A  Global-event fraction by state')
for s in ['quiet','onset','sustained']:
    row=G.set_index('state').loc[s]
    ax[1].bar(0,0)  # placeholder
xx=['quiet','pre','onset','sustained','post']; lr=[G.set_index('state').loc[s,'local_rate'] if s in G.state.values and 'local_rate' in G and G.set_index('state').loc[s,'n_events']>0 else 0 for s in xx]; gr=[G.set_index('state').loc[s,'global_rate'] if s in G.state.values and 'global_rate' in G and G.set_index('state').loc[s,'n_events']>0 else 0 for s in xx]
ax[1].bar(np.arange(5)-.2,lr,.4,label='local'); ax[1].bar(np.arange(5)+.2,gr,.4,label='global'); ax[1].set_xticks(range(5)); ax[1].set_xticklabels(xx,rotation=20,fontsize=7); ax[1].set_ylabel('event rate /min'); ax[1].set_title('B  Local vs global rate by state'); ax[1].legend(fontsize=7)
for s in ['pos','neg','unc']:
    mm=D.coupling_sign==s; ax[2].scatter(cx[mm.values],D.depth_um[mm],c=cmap[s],s=30,label=f'{s} (n={mm.sum()})')
ax[2].invert_yaxis(); ax[2].set_xlabel('X (µm)'); ax[2].set_ylabel('depth Y (µm)'); ax[2].set_title('C  Movement-coupling sign map'); ax[2].legend(fontsize=7)
fig.suptitle('Q4 behavioral-state grammar + Q5 movement-coupling sign (0508)',fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(PR/"scape-data"/"figures"/"fig11_grammar_sign.png",dpi=150); plt.close(fig)
print("saved state_event_grammar.csv, fig11_grammar_sign.png")
