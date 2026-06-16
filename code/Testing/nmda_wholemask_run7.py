#!/usr/bin/env python
"""0416_r567 run7 only: FOV map drawn by trunk/branch SEGMENTS (combined masks shown as their
blue-trunk + red-branch pieces with cuts), and amplitude compared per SEGMENT (all trunk
segments vs all branch segments, including the trunk/branch parts of combined masks).
Per-segment ΔF/F = mean(core of segment) - whole-mask shell; events on the smoothed trace."""
from pathlib import Path
import numpy as np, tifffile, importlib.util, gc, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter1d
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
def load(n,p): s=importlib.util.spec_from_file_location(n,PR/p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
G=load("g","code/Testing/nmda_bap_gallery2.py"); A=load("a","code/Testing/nmda_bap_amplitude.py"); v=load("v","code/Testing/nmda_bap_split_v2.py")
B1,B2,B3=v.B1,v.B2,v.B3; CLU={'trunk':(0.20,0.45,0.95),'branch':(0.92,0.22,0.20)}
FOVS=[("0416_r567","2026-04-16","rbp4_132_phpeb","run7",12),
      ("0508_run5","2026-05-08","rbp4_139_phpeb","run5",14),
      ("0512_run5","2026-05-12","rbp4_132_phpeb","run5",12),
      ("0512_run9","2026-05-12","rbp4_132_phpeb","run9",12),
      ("0320_run1","2026-03-20","rbp4cre_139_phpeb","run1",12),
      ("0320_run3","2026-03-20","rbp4cre_139_phpeb","run3",12)]
MIN_SEG=30; SM=v.SMOOTH_SIGMA

def run_fov(LABEL,DATE,MOUSE,RUN,SKIP):
    base=PR/"scape-data"/DATE/MOUSE; mdir=base/RUN/"labelmaps_curated_dynamic"
    rawp=v.raw_path(base,RUN); tf=tifffile.TiffFile(str(rawp)); T,Z,Y,X=tf.series[0].shape
    try: arr=tifffile.memmap(str(rawp))
    except Exception: arr=tf.series[0].asarray()
    sf=int(SKIP*5)
    f0=np.percentile(np.asarray(arr[np.unique(np.linspace(sf,T-1,150).astype(int))]).astype(np.float32),10,axis=0).ravel(); f0e=f0+1e-6
    mip=np.asarray(arr[np.unique(np.linspace(sf,T-1,40).astype(int))]).astype(np.float32).mean(0).max(0)
    paths=sorted(mdir.glob("dend_*_labelmap.tif"))
    masks={}; mid=np.zeros((Y,X),np.int32); ctype=np.zeros((Y,X),np.int8); cluid=np.zeros((Y,X),np.int32); gid=0
    nT=nB=nC=0
    for i,p in enumerate(paths,1):
        nm=p.stem.replace("_labelmap",""); m=tifffile.imread(p)>0
        if m.shape[1]<Y: m=np.pad(m,((0,0),(0,Y-m.shape[1]),(0,0)))
        elif m.shape[1]>Y: m=m[:,:-(m.shape[1]-Y),:]
        clab,cls,cat,info=G.decompose(m)
        nT+=cat=='trunk'; nB+=cat=='branch'; nC+=cat=='combined'
        shell=binary_dilation(m,B3)&~binary_dilation(m,B2)
        segs=[]
        for cid,typ in cls.items():
            cm=clab==cid
            if cm.sum()<MIN_SEG: continue
            core=binary_erosion(cm,B1); core=core if core.any() else cm
            segs.append((typ,v.gidx(core))); gid+=1
            f=cm.max(0); cluid[f]=gid; ctype[f]=1 if typ=='trunk' else 2
        if segs: masks[nm]=dict(shell=v.gidx(shell),segs=segs)
        mid[m.max(0)]=i
    # stream run7: per-segment trace
    segtr={nm:[ (typ,np.empty(T,np.float32)) for typ,_ in d['segs']] for nm,d in masks.items()}
    for t in range(T):
        dff=(np.asarray(arr[t]).astype(np.float32).ravel()-f0)/f0e
        for nm,d in masks.items():
            sh=dff[d['shell']].mean() if d['shell'].size else 0.0
            for k,(typ,idx) in enumerate(d['segs']): segtr[nm][k][1][t]=dff[idx].mean()-sh
    tf.close()
    rows=[]
    for nm in masks:
        for typ,tr in segtr[nm]:
            w=gaussian_filter1d(tr,SM)*100.0; thr=A.thresh(w)
            for p in A.event_peaks(w,thr): rows.append((typ,float(w[p])))
    E=pd.DataFrame(rows,columns=["typ","amp"])
    # ---- figure ----
    g=np.clip((mip-mip.min())/(np.percentile(mip,99.5)-mip.min()+1e-9),0,1); rgb=np.dstack([g,g,g])*0.85
    rgb[ctype==1]=0.45*rgb[ctype==1]+0.55*np.array(CLU['trunk']); rgb[ctype==2]=0.45*rgb[ctype==2]+0.55*np.array(CLU['branch'])
    def sh2(a):
        d=np.zeros(a.shape,bool); d[1:]|=a[1:]!=a[:-1]; d[:-1]|=a[:-1]!=a[1:]; d[:,1:]|=a[:,1:]!=a[:,:-1]; d[:,:-1]|=a[:,:-1]!=a[:,1:]; return d
    rgb[sh2(cluid)&(cluid>0)&~sh2(mid)]=[0,0,0]              # cuts between segments (same mask)
    rgb[sh2(mid)&(mid>0)]=[1,1,1]                            # mask outlines
    fig=plt.figure(figsize=(13,6.5)); axm=fig.add_axes([0.02,0.05,0.62,0.88]); axb=fig.add_axes([0.72,0.13,0.26,0.76])
    axm.imshow(rgb,aspect='equal'); axm.set_ylim(Y-.5,-.5); axm.set_xlabel("X"); axm.set_ylabel("Y (cortical depth)")
    axm.set_title(f"{LABEL} {RUN}: trunk/branch segments  (masks: trunk={nT} branch={nB} combined={nC})")
    axm.legend(handles=[Patch(color=CLU['trunk'],label='trunk segment'),Patch(color=CLU['branch'],label='branch segment')],fontsize=8,loc='lower right')
    tr=E[E.typ=='trunk'].amp; br=E[E.typ=='branch'].amp
    axb.boxplot([tr.values,br.values],tick_labels=[f"trunk\nn={len(tr)}",f"branch\nn={len(br)}"],showfliers=False)
    axb.set_ylabel("segment event amp ΔF/F %"); axb.set_title("amplitude per segment (run7)"); axb.grid(alpha=.3)
    if len(tr)>3 and len(br)>3:
        p=stats.mannwhitneyu(tr,br,alternative="greater")[1]
        axb.text(0.5,-0.13,f"trunk {tr.median():.2f}% vs branch {br.median():.2f}%  (trunk>branch p={p:.2g})",
                 transform=axb.transAxes,ha='center',va='top',fontsize=9)
    fig.savefig(PR/"code"/"Testing"/"figures"/f"nmda_wholemask_{LABEL}.png",dpi=160,bbox_inches='tight'); plt.close(fig)
    print(f"[{LABEL}] masks trunk={nT} branch={nB} combined={nC} | segment events: trunk n={len(tr)} med={tr.median():.2f}% | branch n={len(br)} med={br.median():.2f}%")
    if len(tr)>3 and len(br)>3: print(f"   trunk>branch MWU p={stats.mannwhitneyu(tr,br,alternative='greater')[1]:.3g}")

def main():
    for cfg in FOVS: run_fov(*cfg)

if __name__=="__main__": main()
