#!/usr/bin/env python
"""Control analysis (0508 run5+run6): does residual movement coupling survive event_rate, SNR,
mask_volume, depth, with family random effect? Coupling effect size = power-adjusted z from a
per-mask circular-shift null: z=(|r(resid,accel)| - mean_null)/std_null."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS=eb.FS; W=15; VOXV=float(np.prod([3.9,1.0,1.2]))
RUNS=[("2026-05-08","rbp4_139_phpeb","run5"),("2026-05-08","rbp4_139_phpeb","run6")]
MR=("2026-05-08","rbp4_139_phpeb","run5")

Xs,accs,off=[],[],0; names=None
for d,mo,rn in RUNS:
    b=PR/"scape-data"/d/mo/rn; df=pd.read_csv(b/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    nm=[c for c in df.columns if 'dend' in c]; names=names or nm
    X=df[names].values.astype(float); a,_=eb.load_behavior(b,rn,len(X)); Xs.append(X); accs.append(a)
X=np.vstack(Xs); accel=np.concatenate(accs); T,N=X.shape
g=np.median(X,1); beta=(X*g[:,None]).mean(0)/(g.var()+1e-9); resid=X-np.outer(g,beta)

ks=np.random.randint(W,T-W,400); cz=np.empty(N); sig=np.empty(N,bool)
for i in range(N):
    ob=abs(np.corrcoef(resid[:,i],accel)[0,1]); nd=np.array([abs(np.corrcoef(np.roll(resid[:,i],k),accel)[0,1]) for k in ks])
    cz[i]=(ob-nd.mean())/(nd.std()+1e-9); sig[i]=ob>np.percentile(nd,95)

# covariates
def snr(x): m=np.median(x); mad=np.median(np.abs(x-m))*1.4826+1e-9; return (np.percentile(x,95)-m)/mad
SNR=np.array([snr(X[:,i]) for i in range(N)])
mb=PR/"scape-data"/MR[0]/MR[1]/MR[2]/"labelmaps_curated_dynamic"
vol=np.array([(tifffile.imread(mb/f"{n}_labelmap.tif")>0).sum()*VOXV for n in names])
topo=pd.read_csv(PR/"scape-data"/"topology_pooled.csv"); topo=topo[topo.fov=='0508_r56'].set_index('name')
event_rate=np.array([topo.loc[n,'event_rate'] if n in topo.index else np.nan for n in names])
depth=np.array([topo.loc[n,'depth_um'] if n in topo.index else np.nan for n in names])
family=[topo.loc[n,'family'] if n in topo.index else 'na' for n in names]

D=pd.DataFrame(dict(cz=cz,sig=sig.astype(int),event_rate=event_rate,SNR=SNR,volume=vol,depth=depth,family=family)).dropna()
for c in ['event_rate','SNR','volume','depth']: D[c+'_z']=(D[c]-D[c].mean())/D[c].std()

print(f"0508 pooled N={N}; coupling z>0 fraction={np.mean(cz>0):.2f}; mean coupling z={cz.mean():.2f} "
      f"(one-sample t p={stats.ttest_1samp(cz,0).pvalue:.1e})")
print("confound check (sig vs non-sig, Mann-Whitney p):")
for c in ['event_rate','SNR','volume','depth']:
    a=D[D.sig==1][c]; b=D[D.sig==0][c]; print(f"  {c}: sig med={a.median():.2f} vs {b.median():.2f}  p={stats.mannwhitneyu(a,b).pvalue:.3f}")
print("\nMixedLM: coupling_z ~ event_rate + SNR + volume + depth, (1|family)")
m=smf.mixedlm("cz ~ event_rate_z + SNR_z + volume_z + depth_z", D, groups=D.family).fit(reml=False)
print(m.summary().tables[1])
print(f"\nintercept (baseline coupling beyond covariates) = {m.params['Intercept']:+.2f}, p={m.pvalues['Intercept']:.1e}")

fig,ax=plt.subplots(1,2,figsize=(10,4))
for a,c in zip(ax,['event_rate','SNR']):
    a.scatter(D[c],D.cz,s=16,alpha=.6,color='steelblue'); a.axhline(0,color='gray',lw=.5)
    r,p=stats.pearsonr(D[c],D.cz); a.set_xlabel(c); a.set_ylabel('coupling z (power-adjusted)'); a.set_title(f'{c}: r={r:+.2f} p={p:.2g}'); a.grid(alpha=.3)
fig.suptitle('Control: residual movement-coupling effect size vs detectability covariates (0508)')
fig.tight_layout(); out=PR/"scape-data"/"figures"; fig.savefig(out/"fig1_controls.png",dpi=150); plt.close(fig)
print(f"saved {out/'fig1_controls.png'}")
