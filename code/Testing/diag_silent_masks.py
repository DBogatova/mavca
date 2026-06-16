#!/usr/bin/env python
"""Why are some masks 'silent' in the v2 split test? Cross-check v2 (raw re-extracted,
unsmoothed, whole-mask core-shell) silence vs the curated PIPELINE trace
(dff_traces_curated_bgsub.csv; same core-shell but smoothed + M4 F0)."""
from pathlib import Path
import numpy as np, pandas as pd
PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
seg = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_segments.csv")
ev  = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_events.csv")
mm  = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_mask_metrics.csv")     # pipeline-trace based

FOVMAP = {"0508_r56":"0508","0416_r123":"0416_r123","0416_r567":"0416_r567",
          "0512_r56":"0512_r56","0512_r910":"0512_r910"}

nseg = seg.groupby(["fov","name"]).size().rename("nseg")
nev  = ev.groupby(["fov","name"]).size().rename("nev_v2")
tab = pd.concat([nseg, nev], axis=1).reset_index()
tab["nev_v2"] = tab["nev_v2"].fillna(0).astype(int)
tab["silent_v2"] = (tab.nseg >= 2) & (tab.nev_v2 == 0)

# pipeline activity (map v2 fov label so we can merge on the same key)
mm = mm.copy(); mm["fov"] = mm.fov.map({v:k for k,v in FOVMAP.items()})
mm = mm.dropna(subset=["fov"])[["fov","name","peak_dff","n_events","mean_amp"]]
T = tab.merge(mm, on=["fov","name"], how="left")

print("Silent masks (multi-segment, 0 events in v2 re-extraction) vs PIPELINE trace:\n")
for fov in T.fov.unique():
    g = T[(T.fov==fov) & T.silent_v2].sort_values("peak_dff", ascending=False)
    if g.empty:
        print(f"[{fov}] silent=0\n"); continue
    act = int((g.n_events.fillna(0) >= 1).sum())
    print(f"[{fov}] silent_v2={len(g)}  active in pipeline (>=1 event)={act}  "
          f"median pipeline peak={g.peak_dff.median():.1f}%")
    print(g[["name","nseg","peak_dff","n_events"]].to_string(
          index=False, float_format=lambda x:f"{x:.1f}"))
    print()

sil = T[T.silent_v2]; act = T[(T.nseg>=2) & (T.nev_v2>0)]
print(f"PIPELINE peak ΔF/F%: silent_v2 median={sil.peak_dff.median():.1f} (n={sil.peak_dff.notna().sum()}) | "
      f"v2-active median={act.peak_dff.median():.1f} (n={act.peak_dff.notna().sum()})")
print(f"silent_v2 with pipeline peak<3%: {(sil.peak_dff<3).sum()}/{sil.peak_dff.notna().sum()}  "
      f"| with >=1 pipeline event: {(sil.n_events.fillna(0)>=1).sum()}/{len(sil)}")
