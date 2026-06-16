#!/usr/bin/env python
"""Sensitivity of the within-mask 'extended>localized' amplitude result to the localization
threshold REL_FRAC. For each event we stored a1 (strongest segment transient) and a2 (2nd).
A segment beyond the strongest is 'active' iff a2 >= REL_FRAC*a1 AND a2 >= MIN_ACT_ABS, so:
   localized  <=> a2 < REL_FRAC*a1  OR  a2 < MIN_ACT_ABS
   extended   <=> otherwise
amp compared = a1 (strongest-segment transient) in both classes. We sweep REL_FRAC and report,
per FOV and pooled, the localized count, the extended/localized amplitude ratio, and MWU p."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
E=pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_events.csv")
E=E[E.nseg>=2].copy()
MIN_ACT_ABS=1.0
RELS=[0.25,0.33,0.40,0.50,0.60,0.70,0.80]

def classify(rel):
    a2_active=(E.a2>=rel*E.a1)&(E.a2>=MIN_ACT_ABS)
    return np.where(a2_active,"extended","localized")

print(f"amp compared = a1 (strongest-segment transient). MIN_ACT_ABS={MIN_ACT_ABS}.")
print(f"{'REL':>5} {'nLoc':>5} {'nExt':>5} | pooled ext/loc(a1) ratio & p  |  per-FOV ratios")
for rel in RELS:
    kl=classify(rel); E["k"]=kl
    E["a1z"]=E.groupby("fov").a1.transform(lambda s:(s-s.mean())/(s.std()+1e-9))
    lo=E[E.k=="localized"]; ex=E[E.k=="extended"]
    if len(lo)>3 and len(ex)>3:
        _,p=stats.mannwhitneyu(ex.a1z,lo.a1z,alternative="greater")
        ratio=ex.a1.median()/max(lo.a1.median(),1e-9)
    else: p,ratio=np.nan,np.nan
    perfov=[]
    for f,g in E.groupby("fov"):
        gl=g[g.k=="localized"]; ge=g[g.k=="extended"]
        if len(gl)>=3 and len(ge)>=3:
            r=ge.a1.median()/max(gl.a1.median(),1e-9)
            _,pf=stats.mannwhitneyu(ge.a1,gl.a1,alternative="greater")
            perfov.append(f"{f.split('_')[0]}:{r:.2f}{'*' if pf<0.05 else ''}")
        else: perfov.append(f"{f.split('_')[0]}:nLoc{len(gl)}")
    print(f"{rel:>5.2f} {len(lo):>5} {len(ex):>5} | ratio={ratio:>4.2f} p={p:>7.2g} | "+"  ".join(perfov))

# reference: the OLD noise-threshold classification result (from the earlier run) for context
print("\nReference: with the per-segment NOISE-THRESHOLD classifier (earlier run), pooled ratio≈"
      "1.6x, p≈2e-19 (now known to be inflated by detection coupling).")
print("Interpretation: scan how localized-count and ratio move with REL_FRAC; a robust effect "
      "would persist across the middle range, not appear only at one extreme.")
