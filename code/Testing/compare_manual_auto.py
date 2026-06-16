#!/usr/bin/env python
"""
Compare MANUAL trunk/branch labels (from manual_trunk_branch.py) vs the AUTO classifier,
and sweep classifier parameters to see which ones are essential for matching your choices.

Metrics:
  - category accuracy: auto vs manual whole-mask class (trunk/branch/combined)
  - trunk-presence accuracy: does auto agree on 'has a trunk' (trunk|combined) vs branch
  - branch-voxel IoU: overlap of auto branch voxels vs manual branch voxels (combined masks)
Run AFTER annotating. Edit DATE/MOUSE/RUN to the annotated FOV.
"""
from pathlib import Path
import numpy as np, tifffile, pandas as pd, importlib.util, itertools
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
G=importlib.util.module_from_spec(importlib.util.spec_from_file_location("g",PR/"code/Testing/nmda_bap_gallery2.py"))
importlib.util.spec_from_file_location("g",PR/"code/Testing/nmda_bap_gallery2.py").loader.exec_module(G)
DATE,MOUSE,RUN="2026-05-08","rbp4_139_phpeb","run5"
BASE=PR/"scape-data"/DATE/MOUSE/RUN; MAN=BASE/"manual_trunk_branch"; MDIR=BASE/"labelmaps_curated_dynamic"

def load():
    if not (MAN/"manual_labels.csv").exists():
        raise SystemExit(f"No manual labels in {MAN} — run manual_trunk_branch.py first.")
    man=pd.read_csv(MAN/"manual_labels.csv").set_index("name")
    items=[]
    for p in sorted(MDIR.glob("dend_*_labelmap.tif")):
        nm=p.stem.replace("_labelmap","")
        if nm not in man.index: continue
        m=tifffile.imread(p)>0
        tb=tifffile.imread(MAN/f"{nm}_tb.tif").astype(np.uint8)
        items.append((nm,m,man.loc[nm,"category"],(tb==2)&m))
    return items

def evaluate(items):
    cat_ok=0; trunk_ok=0; ious=[]; conf={}
    for nm,m,mcat,mbranch in items:
        _,cls,acat,_=G.decompose(m)
        conf[(mcat,acat)]=conf.get((mcat,acat),0)+1
        cat_ok+= (acat==mcat)
        trunk_ok+= ((acat in('trunk','combined'))==(mcat in('trunk','combined')))
        # branch IoU (voxel)
        clab,cls2,_,_=G.decompose(m)
        abranch=np.isin(clab,[c for c,t in cls2.items() if t=='branch'])
        inter=(abranch&mbranch).sum(); uni=(abranch|mbranch).sum()
        if uni>0: ious.append(inter/uni)
    n=len(items)
    return cat_ok/n, trunk_ok/n, float(np.mean(ious)) if ious else np.nan, conf

def set_params(p):
    for k,val in p.items(): setattr(G,k,val)

DEFAULT=dict(LIN_TRUNK=G.LIN_TRUNK,COLLINEAR=G.COLLINEAR,RDP_EPS=G.RDP_EPS,TRUNK_MIN=G.TRUNK_MIN)
GRID=dict(LIN_TRUNK=[0.65,0.70,0.75,0.80,0.85],COLLINEAR=[20,30,40,50],
          RDP_EPS=[5,8,11],TRUNK_MIN=[25,35,45])

def main():
    items=load(); print(f"{len(items)} annotated masks")
    set_params(DEFAULT)
    ca,ta,iou,conf=evaluate(items)
    print(f"\nDEFAULT params {DEFAULT}\n  category acc={ca:.2f}  trunk-presence acc={ta:.2f}  branch IoU={iou:.2f}")
    print("  confusion (manual->auto):",{f"{k[0]}->{k[1]}":v for k,v in sorted(conf.items())})
    print("\n--- one-at-a-time sweeps (others at default) ---")
    sens={}
    for param,vals in GRID.items():
        accs=[]
        for val in vals:
            set_params(DEFAULT); setattr(G,param,val)
            ca,ta,iou,_=evaluate(items); accs.append(ca)
            print(f"  {param}={val}: cat_acc={ca:.2f} trunk_acc={ta:.2f} IoU={iou:.2f}")
        sens[param]=max(accs)-min(accs)
    set_params(DEFAULT)
    print("\nSENSITIVITY (max-min category acc as each param varies; higher = more essential):")
    for k,val in sorted(sens.items(),key=lambda x:-x[1]): print(f"  {k}: {val:.2f}")
    # small joint search over the 2 most sensitive params
    top2=[k for k,_ in sorted(sens.items(),key=lambda x:-x[1])[:2]]
    print(f"\n--- joint search over {top2} ---")
    best=(-1,None)
    for a in GRID[top2[0]]:
        for b in GRID[top2[1]]:
            set_params(DEFAULT); setattr(G,top2[0],a); setattr(G,top2[1],b)
            ca,ta,iou,_=evaluate(items)
            if ca>best[0]: best=(ca,{top2[0]:a,top2[1]:b,'trunk_acc':ta,'IoU':round(iou,2)})
    print(f"  best category acc={best[0]:.2f} at {best[1]}")

if __name__=="__main__": main()
