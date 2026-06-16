#!/usr/bin/env python
"""Why are some straight, deep masks labeled NMDA (>=1 localized event)?
Characterize NMDA-labeled vs bAP-labeled masks by depth, straightness, SNR, #events;
and inspect the localized events' within-mask amplitude structure."""
from pathlib import Path
import numpy as np, pandas as pd
PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
seg = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_segments.csv")
ev  = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_events.csv")
mm  = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_mask_metrics.csv")     # has linearity, orient_deg, length, SNR? no
FOVMAP = {"0508_r56":"0508","0416_r123":"0416_r123","0416_r567":"0416_r567",
          "0512_r56":"0512_r56","0512_r910":"0512_r910"}
mm["fov"]=mm.fov.map({v:k for k,v in FOVMAP.items()}); mm=mm.dropna(subset=["fov"])

nseg=seg.groupby(["fov","name"]).size().rename("nseg")
depth=seg.groupby(["fov","name"]).depth_corr.mean().rename("depth_corr")
nloc=ev[ev.klass=="localized"].groupby(["fov","name"]).size().rename("nloc")
next_=ev[ev.klass=="extended"].groupby(["fov","name"]).size().rename("next")
tab=pd.concat([nseg,depth,nloc,next_],axis=1).reset_index()
tab[["nloc","next"]]=tab[["nloc","next"]].fillna(0).astype(int)
tab=tab[tab.nseg>=2]
tab["ntot"]=tab.nloc+tab.next
tab["label"]=np.where(tab.nloc>=1,"NMDA",np.where(tab.next>=1,"bAP","bgcanc"))
tab=tab.merge(mm[["fov","name","linearity","orient_deg","length_um"]],on=["fov","name"],how="left")

N=tab[tab.label=="NMDA"]; B=tab[tab.label=="bAP"]
print("Per-mask comparison NMDA (>=1 localized) vs bAP (events, all extended):")
for col in ["depth_corr","linearity","orient_deg","length_um","ntot","nseg"]:
    print(f"  {col:11s}: NMDA median={N[col].median():.2f}  bAP median={B[col].median():.2f}")
print(f"\n  NMDA masks: n={len(N)}; with only 1 total event={(N.ntot==1).sum()}, "
      f"<=2 events={(N.ntot<=2).sum()}  ({100*(N.ntot<=2).mean():.0f}%)")
print(f"  NMDA masks that are 'straight' (linearity>0.8) AND deep (depth_corr>80um): "
      f"{((N.linearity>0.8)&(N.depth_corr>80)).sum()}")
print(f"    of those, fraction with <=2 events: "
      f"{100*(N[(N.linearity>0.8)&(N.depth_corr>80)].ntot<=2).mean():.0f}%")

# localized events: how 'diluted' is the whole-mask amp vs the active segment?
loc=ev[ev.klass=="localized"].copy(); ext=ev[ev.klass=="extended"].copy()
loc["whole_over_local"]=loc.amp_whole/loc.amp_local.clip(lower=1e-6)
print(f"\n  localized events: amp_whole/amp_local median={loc.whole_over_local.median():.2f} "
      f"(low => other segments truly quiet; ~1 => whole mask was up but only 1 seg passed ACT_FRAC)")
print(f"  localized events by depth: shallow(<60um) median ratio="
      f"{loc[loc.best_depth_corr<60].whole_over_local.median():.2f}  "
      f"deep(>=60um)={loc[loc.best_depth_corr>=60].whole_over_local.median():.2f}")
print(f"  localized event count by depth: shallow={int((loc.best_depth_corr<60).sum())} "
      f"deep={int((loc.best_depth_corr>=60).sum())}; extended shallow="
      f"{int((ext.best_depth_corr<60).sum())} deep={int((ext.best_depth_corr>=60).sum())}")
# list deep straight NMDA masks
ds=N[(N.linearity>0.8)&(N.depth_corr>80)].sort_values("depth_corr",ascending=False)
print("\n  deep+straight NMDA masks (linearity>0.8, depth>80um):")
print(ds[["fov","name","nseg","depth_corr","linearity","orient_deg","nloc","next"]].to_string(
      index=False,float_format=lambda x:f"{x:.1f}"))
