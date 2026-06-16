#!/usr/bin/env python
"""Models on master_mask_table.csv. Q1-4 = 0508 residual movement coupling (movement coupling is
0508-specific). Q5 = depth gradient by event class, all 6 FOVs. Mixed-effects (family RE) +
permutation. Reports ROBUST vs EXPLORATORY."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, statsmodels.formula.api as smf
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
M=pd.read_csv(PR/"scape-data"/"master_mask_table.csv")
def zc(df,c): return df.groupby('fov')[c].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
for c in ['depth_um','event_rate','SNR','volume_um3','rate_local','rate_subtree','rate_global','coupling_z']:
    M[c+'_z']=zc(M,c)
D=M[M.fov=='0508_r56'].copy(); D['sig']=D.coupling_sig.astype(int)

print("="*60,"\nQ1/Q2 — residual movement coupling vs depth (0508, family RE)")
m1=smf.mixedlm("coupling_z ~ depth_um_z", D, groups=D.family).fit(reml=False)
print(f"  Q1 coupling~depth: depth coef={m1.params['depth_um_z']:+.2f} p={m1.pvalues['depth_um_z']:.2f}")
m2=smf.mixedlm("coupling_z ~ depth_um_z + event_rate_z + SNR_z + volume_um3_z", D, groups=D.family).fit(reml=False)
print(f"  Q2 +covariates: depth coef={m2.params['depth_um_z']:+.2f} p={m2.pvalues['depth_um_z']:.2f}; "
      f"intercept={m2.params['Intercept']:+.2f} p={m2.pvalues['Intercept']:.1e}")
print("  -> ROBUST: residual movement coupling NOT organized by depth" if m2.pvalues['depth_um_z']>0.05 else "  depth predicts coupling")

print("="*60,"\nQ3 — are coupled branches enriched in event classes? (0508)")
for c in ['frac_local','frac_subtree','frac_global']:
    a=D[D.sig==1][c].dropna(); b=D[D.sig==0][c].dropna()
    print(f"  {c}: coupled med={a.median():.2f} vs {b.median():.2f}  MWU p={stats.mannwhitneyu(a,b).pvalue:.2f}")

print("="*60,"\nQ4 — spatial / family clustering of coupled branches (0508)")
mb=PR/"scape-data"/"2026-05-08"/"rbp4_139_phpeb"/"run5"/"labelmaps_curated_dynamic"
cent=np.array([np.argwhere(tifffile.imread(mb/f"{n}_labelmap.tif")>0).mean(0)*[3.9,1,1.2] for n in D.name])
sig=D.sig.values.astype(bool); k=sig.sum(); fams=D.family.values
def mpd(idx):
    c=cent[idx]; dm=np.linalg.norm(c[:,None]-c[None],axis=2); return dm[np.triu_indices(len(idx),1)].mean()
obs=mpd(np.where(sig)[0]); null=[mpd(np.random.choice(len(D),k,replace=False)) for _ in range(2000)]
ps=(np.sum(np.array(null)<=obs)+1)/2001
samefam=lambda idx: np.mean([fams[a]==fams[b] for a in idx for b in idx if a<b])
obsf=samefam(np.where(sig)[0]); nullf=[samefam(np.random.choice(len(D),k,replace=False)) for _ in range(2000)]
pf=(np.sum(np.array(nullf)>=obsf)+1)/2001
print(f"  spatial: mean pairwise dist coupled={obs:.0f}um vs null={np.mean(null):.0f} (p_clustered={ps:.2f})")
print(f"  family: same-family pair frac coupled={obsf:.2f} vs null={np.mean(nullf):.2f} (p_enriched={pf:.2f})")

print("="*60,"\nQ5 — depth gradient by event class, ALL 6 FOVs (family RE, within-FOV z)")
for c in ['event_rate','rate_local','rate_subtree','rate_global']:
    mm=smf.mixedlm(f"{c}_z ~ depth_um_z", M, groups=M.family).fit(reml=False)
    print(f"  {c:12s} ~ depth: coef={mm.params['depth_um_z']:+.2f} p={mm.pvalues['depth_um_z']:.2g}")
print("  (negative = superficial higher; identifies which class carries the depth gradient)")
