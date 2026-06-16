#!/usr/bin/env python
"""Per-mask gallery for one FOV: each curated mask drawn separately (Y x X MIP) in its own
panel, colored by class, with the REASON for the call. Combined (multi-branch) masks show the
branch-cluster decomposition (blue=vertical/Ca, red=branch/NMDA) with the cut lines.
Usage: edit FOV below."""
from pathlib import Path
import numpy as np, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
C=importlib.util.module_from_spec(importlib.util.spec_from_file_location("c",PR/"code/Testing/nmda_bap_cluster.py"))
importlib.util.spec_from_file_location("c",PR/"code/Testing/nmda_bap_cluster.py").loader.exec_module(C)
VOXEL=C.VOXEL

FOV=("0416_r567","2026-04-16","rbp4_132_phpeb","run5")
CAT_COL={'vertical':(0.20,0.45,0.95),'branch':(0.92,0.22,0.20),'combined':(1.0,0.55,0.0)}
CLU_COL={'vertical':(0.20,0.45,0.95),'branch':(0.92,0.22,0.20)}

def why(cat, sh, cls, nbp):
    a,l,ln=sh['ang'],sh['lin'],sh['length']
    if cat=='vertical': return f"Ca: vertical {a:.0f}deg, lin {l:.2f}, len {ln:.0f}um"
    if cat=='branch':
        r=[]
        if a>C.ANG_V: r.append(f"tilted {a:.0f}deg")
        if ln<C.LEN_V: r.append(f"short {ln:.0f}um")
        if l<C.LIN_V: r.append(f"low-lin {l:.2f}")
        return "NMDA: "+(", ".join(r) if r else "compact")
    nV=sum(c=='vertical' for c in cls.values()); nB=sum(c=='branch' for c in cls.values())
    return f"combined: {nbp} branch-pts -> {nV}Ca+{nB}branch clusters"

def shifts(a):
    d=np.zeros(a.shape,bool); d[1:]|=a[1:]!=a[:-1]; d[:-1]|=a[:-1]!=a[1:]
    d[:,1:]|=a[:,1:]!=a[:,:-1]; d[:,:-1]|=a[:,:-1]!=a[:,1:]; return d

def main():
    label,date,mouse,mask_run=FOV
    base=PR/"scape-data"/date/mouse
    paths=sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))
    items=[]
    for p in paths:
        nm=p.stem.replace("_labelmap",""); m=tifffile.imread(p)>0
        clab,cls,nbp,cat=C.cluster_mask(m)
        sh=C.shape_of(np.argwhere(m))
        cid_mip=clab.max(0)                              # (Y,X) cluster id per pixel
        foot=cid_mip>0
        ys,xs=np.where(foot)
        if ys.size==0: continue
        y0,y1=max(0,ys.min()-2),ys.max()+3; x0,x1=max(0,xs.min()-2),xs.max()+3
        sub=cid_mip[y0:y1,x0:x1]
        rgb=np.ones((*sub.shape,3))*0.93
        if cat in ('vertical','branch'):
            rgb[sub>0]=CAT_COL[cat]
        else:
            for cidv in np.unique(sub):
                if cidv==0: continue
                rgb[sub==cidv]=CLU_COL.get(cls.get(cidv,'branch'),(0.5,0.5,0.5))
            cut=shifts(sub)&(sub>0)
            rgb[cut]=[0,0,0]                              # cluster cut lines
        items.append((nm,cat,why(cat,sh,cls,nbp),rgb))
    n=len(items); ncol=11; nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*1.6,nrow*1.9))
    axes=np.atleast_1d(axes).ravel()
    for ax,(nm,cat,wtext,rgb) in zip(axes,items):
        ax.imshow(rgb,aspect='equal',interpolation='nearest')
        ax.set_title(f"{nm}\n{cat}",color=CAT_COL[cat],fontsize=5,pad=1)
        ax.text(0.5,-0.04,wtext,transform=ax.transAxes,fontsize=3.6,ha='center',va='top',wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(CAT_COL[cat]); s.set_linewidth(1.4)
    for ax in axes[n:]: ax.axis('off')
    nV=sum(c=='vertical' for _,c,_,_ in items); nB=sum(c=='branch' for _,c,_,_ in items); nC=sum(c=='combined' for _,c,_,_ in items)
    fig.suptitle(f"{label}: per-mask class + reason (blue=Ca/vertical, red=NMDA/branch, orange title=combined; "
                 f"black=branch cut)   vertical={nV} branch={nB} combined={nC}",fontsize=11,y=0.997)
    fig.tight_layout(rect=[0,0,1,0.99])
    out=PR/"code"/"Testing"/"figures"/f"nmda_gallery_{label}.png"
    fig.savefig(out,dpi=170); plt.close(fig); print("saved",out)

if __name__=="__main__": main()
