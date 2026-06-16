#!/usr/bin/env python
"""
Whole-mask (NO cutting) classification by orientation + verticality + depth position:
   Ca  (bAP/trunk)  : vertical (small angle to image-Y) + linear + spans a large depth range
                      (reaches from superficial well toward the bottom).
   NMDA (branch)    : superficial-confined / short / tilted.
Then test whether the two classes differ by AMPLITUDE (whole-mask peak & event amplitude),
pooled and per run. Depth is slope-corrected (pia plane); orientation vs image-Y (no slope corr).
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
A=importlib.util.module_from_spec(importlib.util.spec_from_file_location("a",PR/"code/Testing/nmda_bap_amplitude.py"))
importlib.util.spec_from_file_location("a",PR/"code/Testing/nmda_bap_amplitude.py").loader.exec_module(A)
v=importlib.util.module_from_spec(importlib.util.spec_from_file_location("v",PR/"code/Testing/nmda_bap_split_v2.py"))
importlib.util.spec_from_file_location("v",PR/"code/Testing/nmda_bap_split_v2.py").loader.exec_module(v)
VOXEL=v.VOXEL

ANG=35.0; LIN=0.60; SPAN_MIN=60.0      # Ca = angle<=ANG & linearity>=LIN & depth-span>=SPAN_MIN

def classify_masks(base, mask_run, fs, skip):
    rawp=v.raw_path(base,mask_run); tf=tifffile.TiffFile(str(rawp)); T,Z,Y,X=tf.series[0].shape
    try: arr=tifffile.memmap(str(rawp))
    except Exception: arr=tf.series[0].asarray()
    idx=np.unique(np.linspace(int(skip*fs),T-1,40).astype(int))
    sp=v.fit_surface(np.asarray(arr[idx]).astype(np.float32).mean(0)); tf.close()
    rows={}
    for p in sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif")):
        nm=p.stem.replace("_labelmap",""); m=tifffile.imread(p)>0
        coords=np.argwhere(m); pts=coords*VOXEL; c=pts.mean(0)
        if len(coords)<20:
            rows[nm]=dict(klass="NMDA",angle=90,lin=0,span=0,depth=np.nan); continue
        cov=np.cov((pts-c).T); w,Vv=np.linalg.eigh(cov); w=np.clip(w[::-1],1e-9,None); Vv=Vv[:,::-1]
        pa=Vv[:,0]; angle=float(np.degrees(np.arccos(np.clip(abs(pa[1]),0,1))))
        lin=float((w[0]-w[1])/w[0])
        dep=v.corrected_depth(coords,sp); span=float(dep.max()-dep.min())
        ca=(angle<=ANG) and (lin>=LIN) and (span>=SPAN_MIN)
        rows[nm]=dict(klass="Ca" if ca else "NMDA",angle=angle,lin=lin,span=span,
                      depth=float(dep.mean()),depth_bot=float(dep.max()),depth_top=float(dep.min()))
    return rows, sp

def amps_for_run(base, run, names, klass):
    df=A.load_traces(base,run,names)
    if df is None: return None
    ev=[]; mk=[]
    for nm in names:
        tr=df[nm].values; thr=A.thresh(tr); pk=A.event_peaks(tr,thr)
        amps=[float(tr[p]) for p in pk]
        for a in amps: ev.append((nm,klass[nm],a))
        mk.append((nm,klass[nm],float(np.max(tr)) if tr.size else 0.0,
                   float(np.median(amps)) if amps else np.nan, len(amps)))
    return (pd.DataFrame(ev,columns=["name","klass","amp"]),
            pd.DataFrame(mk,columns=["name","klass","peak","med_ev","nev"]))

def report(tag, ev, mk):
    ca=ev[ev.klass=="Ca"].amp; nm=ev[ev.klass=="NMDA"].amp
    line=f"  [{tag}] events Ca n={len(ca)} NMDA n={len(nm)}"
    if len(ca)>3 and len(nm)>3:
        u,p=stats.mannwhitneyu(ca,nm,alternative="greater")
        line+=f" | per-EVENT amp med Ca={ca.median():.2f}% NMDA={nm.median():.2f}% ({ca.median()/max(nm.median(),1e-6):.2f}x) p={p:.2g}"
    mca=mk[mk.klass=="Ca"].med_ev.dropna(); mnm=mk[mk.klass=="NMDA"].med_ev.dropna()
    if len(mca)>3 and len(mnm)>3:
        u,p=stats.mannwhitneyu(mca,mnm,alternative="greater")
        line+=f" | per-MASK med-ev amp Ca={mca.median():.2f} NMDA={mnm.median():.2f} p={p:.2g}"
    print(line)

def main():
    print(f"Ca = angle<={ANG} & linearity>={LIN} & depth-span>={SPAN_MIN}um ; else NMDA\n")
    print("==== all FOVs (pooled runs): whole-mask Ca vs NMDA amplitude ====")
    for label,d,mo,mr,runs,fs,skip in v.FOVS:
        base=PR/"scape-data"/d/mo
        klass,sp=classify_masks(base,mr,fs,skip)
        names=list(klass.keys()); nca=sum(k['klass']=="Ca" for k in klass.values())
        kl={nm:klass[nm]['klass'] for nm in names}
        evs=[]; mks=[]
        for r in runs:
            res=amps_for_run(base,r,names,kl)
            if res: evs.append(res[0]); mks.append(res[1])
        if not evs: continue
        EV=pd.concat(evs); MK=pd.concat(mks)
        print(f"[{label}] masks Ca={nca} NMDA={len(names)-nca}")
        report("pooled",EV,MK)
        if label=="0416_r567":
            for r in runs:
                res=amps_for_run(base,r,names,kl)
                if res: report(r,res[0],res[1])
            make_fig(base,mr,fs,skip,klass,runs,EV)

def make_fig(base,mr,fs,skip,klass,runs,EVpool):
    # map + per-run boxplots for 0416_r567
    rawp=v.raw_path(base,mr); tf=tifffile.TiffFile(str(rawp)); T,Z,Y,X=tf.series[0].shape
    try: arr=tifffile.memmap(str(rawp))
    except Exception: arr=tf.series[0].asarray()
    idx=np.unique(np.linspace(int(skip*fs),T-1,40).astype(int)); mip=np.asarray(arr[idx]).astype(np.float32).mean(0).max(0)
    g=np.clip((mip-mip.min())/(np.percentile(mip,99.5)-mip.min()+1e-9),0,1); rgb=np.dstack([g,g,g])*0.85
    mid=np.zeros((Y,X),np.int32); kcol=np.zeros((Y,X),np.int8)
    for mi,p in enumerate(sorted((base/mr/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif")),1):
        nm=p.stem.replace("_labelmap",""); m=tifffile.imread(p)>0
        if m.shape[1]<Y: m=np.pad(m,((0,0),(0,Y-m.shape[1]),(0,0)))
        elif m.shape[1]>Y: m=m[:,:-(m.shape[1]-Y),:]
        f=m.max(0); mid[f]=mi; kcol[f]=1 if klass[nm]['klass']=="Ca" else 2
    tf.close()
    rgb[kcol==1]=0.45*rgb[kcol==1]+0.55*np.array([0.2,0.45,0.95])
    rgb[kcol==2]=0.45*rgb[kcol==2]+0.55*np.array([0.92,0.22,0.20])
    d=np.zeros((Y,X),bool); d[1:]|=mid[1:]!=mid[:-1]; d[:-1]|=mid[:-1]!=mid[1:]; d[:,1:]|=mid[:,1:]!=mid[:,:-1]; d[:,:-1]|=mid[:,:-1]!=mid[:,1:]
    rgb[d&(mid>0)]=[1,1,1]
    fig=plt.figure(figsize=(20,4.6)); 
    axm=fig.add_subplot(1,len(runs)+2,1); axm.imshow(rgb,aspect='equal'); axm.set_ylim(Y-.5,-.5)
    axm.set_title("0416_r567 whole-mask class"); axm.set_xlabel("X"); axm.set_ylabel("Y(depth)")
    axm.legend(handles=[Patch(color=(0.2,0.45,0.95),label='Ca (vertical+deep)'),
                        Patch(color=(0.92,0.22,0.20),label='NMDA (superficial/tilted)')],fontsize=7,loc='lower right')
    for i,r in enumerate(runs):
        res=amps_for_run(base,r,list(klass.keys()),{nm:klass[nm]['klass'] for nm in klass})
        ax=fig.add_subplot(1,len(runs)+2,2+i); 
        if res:
            ev=res[0]; data=[ev[ev.klass=="NMDA"].amp.values,ev[ev.klass=="Ca"].amp.values]
            ax.boxplot(data,tick_labels=[f"NMDA\nn={len(data[0])}",f"Ca\nn={len(data[1])}"],showfliers=False)
        ax.set_title(r); ax.set_ylabel("event amp ΔF/F %"); ax.grid(alpha=.3)
    axp=fig.add_subplot(1,len(runs)+2,len(runs)+2)
    data=[EVpool[EVpool.klass=="NMDA"].amp.values,EVpool[EVpool.klass=="Ca"].amp.values]
    axp.boxplot(data,tick_labels=[f"NMDA\nn={len(data[0])}",f"Ca\nn={len(data[1])}"],showfliers=False)
    axp.set_title("pooled run5/6/7"); axp.set_ylabel("event amp ΔF/F %"); axp.grid(alpha=.3)
    fig.suptitle("0416_r567: whole-mask Ca (vertical+deep) vs NMDA (superficial/tilted) — amplitude",fontsize=12)
    fig.tight_layout(); fig.savefig(PR/"code"/"Testing"/"figures"/"nmda_wholemask_0416r567.png",dpi=150); plt.close(fig)
    print("  saved figures/nmda_wholemask_0416r567.png")

if __name__=="__main__": main()
