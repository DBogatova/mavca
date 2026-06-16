#!/usr/bin/env python
"""Inspect deep+straight NMDA-labeled masks: plot per-segment traces with the whole-mask
events marked (localized=red, extended=blue) and each segment's detection threshold.
If at a 'localized' event the non-active segments are visibly elevated (just under their
threshold), the NMDA label is a per-segment thresholding artifact, not true localization."""
from pathlib import Path
import numpy as np, tifffile, gc, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("v",PR/"code/Testing/nmda_bap_split_v2.py")
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
B1,B2,B3=v.B1,v.B2,v.B3

JOBS=[  # (fov, date, mouse, mask_run, fs, skip, [names])
 ("0512_r56","2026-05-12","rbp4_132_phpeb","run5",5.0,12,
   ["dend_044","dend_053","dend_018","dend_052","dend_033","dend_019"]),
 ("0416_r123","2026-04-16","rbp4_132_phpeb","run1",5.0,12,
   ["dend_014","dend_051","dend_001","dend_053","dend_010"]),
]

def run(fov,date,mouse,mask_run,fs,skip,names):
    base=PR/"scape-data"/date/mouse; rawp=v.raw_path(base,mask_run)
    tf=tifffile.TiffFile(str(rawp)); T,Z,Y,X=tf.series[0].shape
    try: arr=tifffile.memmap(str(rawp))
    except Exception: arr=tf.series[0].asarray()
    fidx=np.unique(np.linspace(int(skip*fs),T-1,v.NF0).astype(int))
    f0=np.percentile(np.asarray(arr[fidx]).astype(np.float32),10,axis=0).ravel(); f0e=f0+1e-6
    M={}
    for nm in names:
        m=tifffile.imread(base/mask_run/"labelmaps_curated_dynamic"/f"{nm}_labelmap.tif")>0
        if m.shape[1]<Y: m=np.pad(m,((0,0),(0,Y-m.shape[1]),(0,0)))
        elif m.shape[1]>Y: m=m[:,:-(m.shape[1]-Y),:]
        lab,nseg=v.segment_mask(m)
        inner=binary_dilation(m,B2); outer=binary_dilation(m,B3); shell=outer&~inner
        cw=binary_erosion(m,B1); cw=cw if cw.any() else m
        segs={}
        for s in range(1,nseg+1):
            sm=lab==s
            if sm.sum()<v.MIN_SEG_VOX: continue
            core=binary_erosion(sm,B1); core=core if core.any() else sm
            segs[s]=v.gidx(core)
        M[nm]=dict(whole=v.gidx(cw),shell=v.gidx(shell),segs=segs)
    W={nm:np.empty(T,np.float32) for nm in names}
    S={nm:{s:np.empty(T,np.float32) for s in M[nm]['segs']} for nm in names}
    for t in range(T):
        dff=(np.asarray(arr[t]).astype(np.float32).ravel()-f0)/f0e
        for nm in names:
            sh=dff[M[nm]['shell']].mean() if M[nm]['shell'].size else 0.0
            W[nm][t]=dff[M[nm]['whole']].mean()-sh
            for s,idx in M[nm]['segs'].items(): S[nm][s][t]=dff[idx].mean()-sh
    tf.close()
    uniq=list(dict.fromkeys(names)); n=len(uniq)
    fig,axes=plt.subplots((n+1)//2,2,figsize=(15,3*((n+1)//2)))
    axes=np.atleast_1d(axes).ravel(); tt=np.arange(T)/fs
    from scipy.ndimage import gaussian_filter1d
    for ax,nm in zip(axes,uniq):
        w=gaussian_filter1d(W[nm],v.SMOOTH_SIGMA)*100; thw=v.thresh(w)
        segtr={s:gaussian_filter1d(S[nm][s],v.SMOOTH_SIGMA)*100 for s in M[nm]['segs']}
        base={s:float(np.median(segtr[s])) for s in segtr}
        for s,tr in segtr.items(): ax.plot(tt,tr,lw=.8,label=f"seg{s}")
        ax.plot(tt,w,color='k',lw=1.3,label='whole')
        nl=ne=0
        for p in v.event_peaks(w,thw):
            lo,hi=max(0,p-v.EVTWIN),min(len(w),p+v.EVTWIN+1)
            a={s:segtr[s][lo:hi].max()-base[s] for s in segtr}; peak_amp=max(max(a.values()),1e-6)
            active=[s for s in segtr if a[s]>=v.REL_FRAC*peak_amp and a[s]>=v.MIN_ACT_ABS]
            if not active: continue
            if len(active)==1: nl+=1; col='red'
            else: ne+=1; col='blue'
            ax.axvline(tt[p],color=col,lw=.8,alpha=.5)
        ax.set_title(f"{nm} ({len(segtr)} seg)  loc={nl} ext={ne}  (red=localized)",fontsize=9)
        ax.set_xlabel("s"); ax.set_ylabel("ΔF/F %"); ax.legend(fontsize=5,ncol=2,loc='upper right')
    for ax in axes[n:]: ax.axis('off')
    fig.suptitle(f"{fov}: deep+straight NMDA-labeled masks — segment traces "
                 f"(dotted=segment threshold; vertical line=event)",fontsize=11)
    fig.tight_layout(); out=PR/"code"/"Testing"/"figures"/f"nmda_examples_{fov}.png"
    fig.savefig(out,dpi=150); plt.close(fig); print("saved",out.name)
    del W,S; gc.collect()

for j in JOBS: run(*j)
