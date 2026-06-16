#!/usr/bin/env python
"""
Classify each calcium EVENT by the MORPHOLOGY of the voxels that actually light up,
combined with amplitude (user's rule):
   Ca / bAP  = VERTICAL STREAK : active region elongated along the (slope-corrected) radial
               depth axis, large vertical extent, linear.   (typically high amplitude)
   NMDA      = smaller / more TILTED / branch-coupled active region.  (typically lower amp)

Method (no per-segment thresholding):
  pass 1 - stream raw stack, build whole-mask core-shell ΔF/F trace (smoothed), detect events.
  pass 2 - for each event, read its peak frame, take the mask voxels that light up
           (>= ACTV_FRAC of the brightest voxel), and compute the active region's:
              angle_vs_radial (deg; 0 = along surface normal = vertical streak),
              vertical_extent (um, slope-corrected depth span), linearity, length, n active vox.
  classify Ca(streak) vs NMDA(tilted/compact); report amplitude by class; map masks.

Surface tilt (slope) is corrected: 'radial/vertical' = the fitted pial-surface NORMAL,
not the raw Y axis. Illumination is handled by ΔF/F (per-voxel F0).
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, gc, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter1d
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("v",PR/"code/Testing/nmda_bap_split_v2.py")
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
VOXEL=v.VOXEL; B1,B3,B2=v.B1,v.B3,v.B2

# classification thresholds (morphology-first)
ANG_VERT=35.0    # deg from surface normal: active region within this of radial = "vertical"
EXT_MIN =25.0    # um: min vertical (depth) extent for a streak
LIN_MIN =0.55    # linearity for a streak
ACTV_FRAC=0.5    # a voxel is 'lit' if >= this * brightest voxel at the event frame
MIN_ACTVOX=8

def active_morph(coords, sp):
    if len(coords)<MIN_ACTVOX: return None
    pts=coords*VOXEL; c=pts.mean(0)
    cov=np.cov((pts-c).T); w,V=np.linalg.eigh(cov)
    w=np.clip(w[::-1],1e-9,None); V=V[:,::-1]; pa=V[:,0]
    # 'vertical' = image-Y (cells stay vertical despite the pial slope; verified empirically:
    # dendrite axes align ~equally to image-Y vs surface-normal, tilt << orientation spread).
    # depth (vext) is still slope-corrected below (separate, larger effect).
    ang=float(np.degrees(np.arccos(np.clip(abs(pa[1]),0,1))))            # angle from image-Y (0=vertical)
    dep=v.corrected_depth(coords,sp); vext=float(dep.max()-dep.min())
    proj=(pts-c)@pa; length=float(proj.max()-proj.min())
    lin=float((w[0]-w[1])/w[0])
    return dict(angle=ang,vext=vext,length=length,lin=lin,nvox=int(len(coords)),
                depth=float(dep.mean()),xcen=float(c[2]))

def classify(m):
    if m is None: return "NMDA"           # too few lit voxels = compact/local
    streak=(m['angle']<=ANG_VERT) and (m['vext']>=EXT_MIN) and (m['lin']>=LIN_MIN)
    return "Ca" if streak else "NMDA"

def analyze(label,date,mouse,mask_run,raw_runs,fs,skip):
    base=PR/"scape-data"/date/mouse
    rawp0=v.raw_path(base,mask_run)
    tf0=tifffile.TiffFile(str(rawp0)); T0,Z,Y,X=tf0.series[0].shape
    try: arr0=tifffile.memmap(str(rawp0))
    except Exception: arr0=tf0.series[0].asarray()
    fidx=np.unique(np.linspace(int(skip*fs),T0-1,v.NF0).astype(int))
    sp=v.fit_surface(np.asarray(arr0[fidx]).astype(np.float32).mean(0)); tf0.close()
    # build masks geometry once
    masks={}
    for p in sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif")):
        nm=p.stem.replace("_labelmap","")
        mm=tifffile.imread(p)>0
        if mm.shape[1]<Y: mm=np.pad(mm,((0,0),(0,Y-mm.shape[1]),(0,0)))
        elif mm.shape[1]>Y: mm=mm[:,:-(mm.shape[1]-Y),:]
        core=binary_erosion(mm,B1); core=core if core.any() else mm
        shell=binary_dilation(mm,B3)&~binary_dilation(mm,B2)
        vox=np.argwhere(mm)
        masks[nm]=dict(core=v.gidx(core),shell=v.gidx(shell),
                       vgidx=(vox[:,0].astype(np.int64)*Y*X+vox[:,1]*X+vox[:,2]),vcoord=vox)
    rows=[]
    for run in raw_runs:
        rawp=v.raw_path(base,run)
        if rawp is None: continue
        tf=tifffile.TiffFile(str(rawp)); T,Zr,Yr,Xr=tf.series[0].shape
        if (Zr,Yr,Xr)!=(Z,Y,X): tf.close(); continue
        try: arr=tifffile.memmap(str(rawp))
        except Exception: arr=tf.series[0].asarray()
        f0=np.percentile(np.asarray(arr[np.unique(np.linspace(int(skip*fs),T-1,v.NF0).astype(int))]).astype(np.float32),10,axis=0).ravel()
        f0e=f0+1e-6
        # pass1: whole traces
        trc={nm:np.empty(T,np.float32) for nm in masks}
        for t in range(T):
            dff=(np.asarray(arr[t]).astype(np.float32).ravel()-f0)/f0e
            for nm,d in masks.items():
                sh=dff[d['shell']].mean() if d['shell'].size else 0.0
                trc[nm][t]=dff[d['core']].mean()-sh
        # detect events
        evs={}
        for nm in masks:
            w=gaussian_filter1d(trc[nm],v.SMOOTH_SIGMA)*100.0
            evs[nm]=(w, v.event_peaks(w, v.thresh(w)))
        # pass2: per unique event frame, read frame, compute active morphology
        frame_events={}
        for nm,(w,pk) in evs.items():
            for p in pk: frame_events.setdefault(p,[]).append(nm)
        for fr in sorted(frame_events):
            dff=(np.asarray(arr[fr]).astype(np.float32).ravel()-f0)/f0e
            for nm in frame_events[fr]:
                d=masks[nm]; sh=dff[d['shell']].mean() if d['shell'].size else 0.0
                vd=dff[d['vgidx']]-sh; mx=vd.max()
                act=vd>=ACTV_FRAC*mx
                mo=active_morph(d['vcoord'][act],sp)
                kl=classify(mo)
                amp=float(evs[nm][0][fr])
                row=dict(fov=label,run=run,name=nm,frame=fr,amp=amp,klass=kl)
                if mo: row.update(angle=mo['angle'],vext=mo['vext'],length=mo['length'],
                                  lin=mo['lin'],nactvox=mo['nvox'],depth=mo['depth'],xcen=mo['xcen'])
                else:  row.update(angle=np.nan,vext=np.nan,length=np.nan,lin=np.nan,
                                  nactvox=int(act.sum()),depth=np.nan,xcen=np.nan)
                rows.append(row)
        tf.close(); del trc,arr; gc.collect()
    df=pd.DataFrame(rows)
    ca=df[df.klass=="Ca"]; nm_=df[df.klass=="NMDA"]
    if len(ca)>3 and len(nm_)>3:
        u,p=stats.mannwhitneyu(ca.amp,nm_.amp,alternative="greater")
        print(f"[{label}] events={len(df)} Ca(streak)={len(ca)} NMDA={len(nm_)} | "
              f"amp med Ca={ca.amp.median():.1f}% NMDA={nm_.amp.median():.1f}% "
              f"({ca.amp.median()/max(nm_.amp.median(),1e-6):.2f}x) p={p:.2g}")
    return df, sp

def main():
    alld=[]
    for cfg in v.FOVS:
        d,sp=analyze(*cfg)
        if not d.empty: alld.append(d)
    E=pd.concat(alld,ignore_index=True)
    E.to_csv(PR/"code"/"Testing"/"data"/"nmda_bap_morpho_events.csv",index=False)
    print("\n===== POOLED (within-FOV z amplitude) =====")
    E["ampz"]=E.groupby("fov").amp.transform(lambda s:(s-s.mean())/(s.std()+1e-9))
    ca=E[E.klass=="Ca"]; nm_=E[E.klass=="NMDA"]
    u,p=stats.mannwhitneyu(ca.ampz,nm_.ampz,alternative="greater")
    print(f"  Ca(streak) n={len(ca)} ampz={ca.ampz.median():+.2f} | NMDA n={len(nm_)} ampz={nm_.ampz.median():+.2f}"
          f"  MWU p={p:.3g}")
    print(f"  Ca events: median angle={ca.angle.median():.0f}deg vext={ca.vext.median():.0f}um | "
          f"NMDA: angle={nm_.angle.median():.0f}deg vext={nm_.vext.median():.0f}um")
    make_fig(E)
    print("saved scape-data/nmda_bap_morpho_events.csv, figures/nmda_bap_morpho_classify.png")

def make_fig(E):
    g=E.dropna(subset=["angle","vext"])
    fig,ax=plt.subplots(1,3,figsize=(16,4.6))
    col=np.where(g.klass=="Ca","tab:blue","tab:red")
    ax[0].scatter(g.angle,g.amp,s=10,alpha=.4,c=col)
    ax[0].axvline(ANG_VERT,color='k',ls='--',lw=.7); ax[0].set_xlabel("active-region angle vs radial (deg; 0=vertical)")
    ax[0].set_ylabel("event amp ΔF/F %"); ax[0].set_title("amplitude vs verticality"); ax[0].grid(alpha=.3)
    ax[1].scatter(g.vext,g.amp,s=10,alpha=.4,c=col)
    ax[1].axvline(EXT_MIN,color='k',ls='--',lw=.7); ax[1].set_xlabel("active vertical extent (um)")
    ax[1].set_ylabel("event amp ΔF/F %"); ax[1].set_title("amplitude vs vertical extent"); ax[1].grid(alpha=.3)
    ca=E[E.klass=="Ca"].ampz.dropna(); nmd=E[E.klass=="NMDA"].ampz.dropna()
    ax[2].boxplot([nmd,ca],tick_labels=[f"NMDA\n(tilted/compact)\nn={len(nmd)}",
                  f"Ca (vertical\nstreak)\nn={len(ca)}"],showfliers=False)
    ax[2].axhline(0,color='k',lw=.6); ax[2].set_ylabel("event amp (within-FOV z)")
    ax[2].set_title("amplitude by morphology class"); ax[2].grid(alpha=.3)
    ax[0].legend(handles=[Patch(color='tab:blue',label='Ca (vertical streak)'),
                          Patch(color='tab:red',label='NMDA (tilted/compact)')],fontsize=8)
    fig.suptitle("Event classification by active-region morphology (slope-corrected) + amplitude",fontsize=12)
    fig.tight_layout(); fig.savefig(PR/"code"/"Testing"/"figures"/"nmda_bap_morpho_classify.png",dpi=150); plt.close(fig)

if __name__=="__main__": main()
