#!/usr/bin/env python
"""
NMDA vs bAP amplitude — TEST 2 v2: BRANCH/BEND segmentation + slope & illumination control.

Improvements over nmda_bap_split.py (per user feedback):
  1) Split each mask along its SKELETON at branch points AND orientation kinks
     ("changing orientation"), not at an arbitrary half. NMDA events are confined
     to a branch -> the natural sub-unit is a branch/straight-run, found from geometry.
     (Following the skeleton also makes the split robust to the surface tilt.)
  2) SLOPED SURFACE: fit the pial surface plane Y_surf(X,Z) from the time-mean
     volume; report each segment's slope-corrected depth (Y - Y_surf) and re-reference
     "verticality" to the surface normal. Raw-Y depth is confounded by X (tilt up to ~12 deg).
  3) ILLUMINATION (dim-left -> bright-right, ~1.1-1.2x): ΔF/F divides by per-voxel F0,
     so a static multiplicative gradient is normalized out; we additionally record each
     segment's X-centroid and baseline brightness (F0) and CHECK that the streak-vs-
     localized amplitude difference is not explained by X / brightness.

Classification of each whole-mask event:
  localized (NMDA-like)      : only ONE segment active
  extended/streak (bAP-like) : >=2 segments active  (signal spreads across branches)
amp_local = peak of the strongest active segment (not whole-mask mean) -> fair amplitude.

Output: scape-data/nmda_bap_split_v2_events.csv,
        scape-data/nmda_bap_split_v2_segments.csv,
        figures/nmda_bap_split_v2.png
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, gc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import binary_erosion, binary_dilation, convolve, gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, ball

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
VOXEL = np.array([3.9, 1.0, 1.2])     # Z,Y,X um ; Y = cortical depth

FOVS = [   # (label, date, mouse, mask_run, [raw_runs sharing those masks], fs, skip_s)
    ("0508_r56",  "2026-05-08", "rbp4_139_phpeb", "run5", ["run5","run6"],        5.0, 14),
    ("0416_r123", "2026-04-16", "rbp4_132_phpeb", "run1", ["run1","run2","run3"], 5.0, 12),
    ("0416_r567", "2026-04-16", "rbp4_132_phpeb", "run5", ["run5","run6","run7"], 5.0, 12),
    ("0512_r56",  "2026-05-12", "rbp4_132_phpeb", "run5", ["run5","run6"],        5.0, 12),
    ("0512_r910", "2026-05-12", "rbp4_132_phpeb", "run9", ["run9","run10"],       5.0, 12),
]
NF0=150; K_MAD=3.5; FLOOR=1.5; MIN_DUR=2
MIN_SEG_VOX=80          # min voxels per final segment (bigger -> higher SNR sub-traces)
MAX_SEG=6               # cap segments/mask (merge smallest beyond this)
PRUNE_VOX=6             # skeleton sub-paths shorter than this (voxels) are spurs -> dropped
KINK_MINLEN=30          # only consider kink-splitting skeleton branches with >= this many voxels
EPS_UM=10.0             # RDP tolerance (um): only real bends beyond this start a new segment
PATH_W=2
# --- SNR-robust event classification ---
SMOOTH_SIGMA=1.0        # gaussian smoothing (frames) before detection/scoring
REL_FRAC=0.5            # segment 'active' if its transient >= REL_FRAC * strongest segment's
MIN_ACT_ABS=1.0         # and >= this absolute % (low floor; replaces per-segment MAD threshold)
EVTWIN=1                # +/- frames windowed peak around the event
MIN_MASK_VOX=40         # need this many voxels to attempt skeleton segmentation
B1,B2,B3 = ball(1),ball(2),ball(3)
NBK = np.ones((3,3,3)); NBK[1,1,1]=0
CONN = np.ones((3,3,3))

# ---------------- surface plane ----------------
def raw_path(base, run):
    c=[p for p in sorted((base/run/"raw").glob("*reslice-bin.tif")) if "dff" not in p.name]
    return c[0] if c else None

def fit_surface(mv, frac=0.5):
    Z,Y,X = mv.shape
    base=np.percentile(mv,20); cmax=mv.max(1); mincol=np.percentile(cmax,40)
    thr=base+frac*(cmax-base); surf=np.full((Z,X),np.nan)
    for z in range(Z):
        for x in range(X):
            if cmax[z,x]<mincol: continue
            yy=np.where(mv[z,:,x]>thr[z,x])[0]
            if yy.size: surf[z,x]=yy[0]
    zz,xx=np.mgrid[0:Z,0:X]; m=np.isfinite(surf)
    A=np.column_stack([xx[m],zz[m],np.ones(m.sum())])
    (a,b,c),*_=np.linalg.lstsq(A,surf[m],rcond=None)
    # physical surface normal (z,y,x order)
    n=np.array([-b*VOXEL[1]/VOXEL[0], 1.0, -a*VOXEL[1]/VOXEL[2]]); n/=np.linalg.norm(n)
    return dict(a=a,b=b,c=c,normal=n,
                tilt_x=np.degrees(np.arctan2(a*VOXEL[1],VOXEL[2])),
                tilt_z=np.degrees(np.arctan2(b*VOXEL[1],VOXEL[0])))

def corrected_depth(coords_zyx, sp):       # coords in voxel ZYX -> depth um below surface
    z,y,x=coords_zyx[:,0],coords_zyx[:,1],coords_zyx[:,2]
    return (y-(sp['a']*x+sp['b']*z+sp['c']))*VOXEL[1]

# ---------------- skeleton branch/bend segmentation ----------------
def order_path(coords):
    """Order a set of skeleton voxel coords (cropped frame) along the path."""
    if len(coords)<=2: return list(range(len(coords)))
    tree=cKDTree(coords*VOXEL)
    # adjacency within ~sqrt(3) voxel (26-conn) in physical units
    deg=np.array([len(tree.query_ball_point(p, r=np.linalg.norm(VOXEL)+1e-3))-1 for p in coords*VOXEL])
    start=int(np.argmin(deg))
    order=[start]; seen={start}
    cur=start
    while len(order)<len(coords):
        nb=tree.query_ball_point(coords[cur]*VOXEL, r=np.linalg.norm(VOXEL)*1.01)
        nxt=[j for j in nb if j not in seen]
        if not nxt:
            rem=[j for j in range(len(coords)) if j not in seen]
            if not rem: break
            d=np.linalg.norm((coords[rem]-coords[cur])*VOXEL,axis=1); cur=rem[int(np.argmin(d))]
        else:
            d=np.linalg.norm((coords[nxt]-coords[cur])*VOXEL,axis=1); cur=nxt[int(np.argmin(d))]
        order.append(cur); seen.add(cur)
    return order

def _rdp_keep(pts, eps):
    """Ramer-Douglas-Peucker: boolean keep-mask of vertices for 3D polyline pts (um)."""
    n=len(pts); keep=np.zeros(n,bool); keep[0]=keep[-1]=True
    stack=[(0,n-1)]
    while stack:
        i,j=stack.pop()
        if j<=i+1: continue
        a=pts[i]; ab=pts[j]-a; L=np.linalg.norm(ab)
        seg=pts[i+1:j]
        if L<1e-9: d=np.linalg.norm(seg-a,axis=1)
        else:      d=np.linalg.norm(np.cross(seg-a,ab),axis=1)/L
        k=int(np.argmax(d))
        if d[k]>eps:
            idx=i+1+k; keep[idx]=True; stack.append((i,idx)); stack.append((idx,j))
    return keep

def split_kinks(path_pts):
    """Split an ordered skeleton path (n,3 um) into straight runs at orientation bends (RDP)."""
    n=len(path_pts)
    if n<6: return [np.arange(n)]
    w=5; ker=np.ones(w)/w
    sm=np.column_stack([np.convolve(path_pts[:,d],ker,mode='same') for d in range(3)])
    sm[:w//2]=path_pts[:w//2]; sm[-(w//2):]=path_pts[-(w//2):]
    kept=np.where(_rdp_keep(sm,EPS_UM))[0]
    if len(kept)<=2: return [np.arange(n)]
    segs=[np.arange(kept[i], kept[i+1]) for i in range(len(kept)-1)]
    segs[-1]=np.arange(kept[-2], n)           # last segment inclusive of final vertex
    return [s for s in segs if len(s)>0]

def segment_mask(m):
    """m: full (Z,Y,X) bool. Returns (labels_full int volume, n_segments). 0 = background.
    Segments = skeleton branches (cut at branch points, short spurs dropped), with long
    branches further split at orientation kinks. Tiny segments merged; count capped."""
    zz,yy,xx=np.where(m)
    if zz.size<MIN_MASK_VOX:
        lab=m.astype(np.int32); return lab, (1 if m.any() else 0)
    pad=3; Z,Y,X=m.shape
    z0,z1=max(0,zz.min()-pad),min(Z,zz.max()+1+pad)
    y0,y1=max(0,yy.min()-pad),min(Y,yy.max()+1+pad)
    x0,x1=max(0,xx.min()-pad),min(X,xx.max()+1+pad)
    sub=m[z0:z1,y0:y1,x0:x1]
    sk=skeletonize(sub)
    def single():
        lab=np.zeros_like(sub,np.int32); lab[sub]=1
        full=np.zeros(m.shape,np.int32); full[z0:z1,y0:y1,x0:x1]=lab; return full,1
    if sk.sum()<5: return single()
    from scipy.ndimage import label as cclabel
    nb=convolve(sk.astype(np.int32),NBK,mode='constant')
    bp=sk&(nb>=3)
    chain=sk&~bp                                  # cut at branch nodes
    cc,ncc=cclabel(chain, structure=CONN)
    comps=[np.argwhere(cc==i) for i in range(1,ncc+1)]
    branches=[c for c in comps if len(c)>=PRUNE_VOX]      # drop short spurs
    if not branches: branches=[max(comps,key=len)] if comps else []
    if not branches: return single()
    seg_skel=[]
    for c in branches:
        if len(c)>=KINK_MINLEN:                   # conservative kink split (long branches only)
            order=order_path(c); pc=c[order]
            for idxs in split_kinks(pc*VOXEL): seg_skel.append(pc[idxs])
        else:
            seg_skel.append(c)
    skel_all=np.vstack(seg_skel)
    skel_lab=np.concatenate([np.full(len(s),i+1) for i,s in enumerate(seg_skel)])
    tree=cKDTree(skel_all*VOXEL)
    mvox=np.argwhere(sub); _,nn=tree.query(mvox*VOXEL,k=1)
    lab_sub=np.zeros(sub.shape,np.int32)
    lab_sub[mvox[:,0],mvox[:,1],mvox[:,2]]=skel_lab[nn]
    lab_sub=merge_small(lab_sub, MIN_SEG_VOX, MAX_SEG)
    uniq=[u for u in np.unique(lab_sub) if u!=0]
    remap={u:i+1 for i,u in enumerate(uniq)}
    out=np.zeros_like(lab_sub)
    for u,i in remap.items(): out[lab_sub==u]=i
    full=np.zeros(m.shape,np.int32); full[z0:z1,y0:y1,x0:x1]=out
    return full, len(uniq)

def merge_small(lab, min_vox, max_seg):
    """Merge segments < min_vox into nearest neighbour; then cap to <= max_seg by merging smallest."""
    def step(force_count):
        uniq=[u for u in np.unique(lab) if u!=0]
        if len(uniq)<=1: return False
        sizes={u:int((lab==u).sum()) for u in uniq}
        if force_count and len(uniq)<=max_seg: return False
        small=[u for u in uniq if sizes[u]<min_vox]
        cand = small if (small and not force_count) else (uniq if force_count else [])
        if not cand: return False
        cents={u:np.argwhere(lab==u).mean(0) for u in uniq}
        u=min(cand,key=lambda k:sizes[k])
        others=[o for o in uniq if o!=u]
        tgt=min(others,key=lambda o:np.linalg.norm((cents[o]-cents[u])*VOXEL))
        lab[lab==u]=tgt; return True
    while step(False): pass            # merge sub-min-voxel segments
    while step(True):  pass            # cap count
    return lab

# ---------------- traces ----------------
def thresh(tr):
    m=np.median(tr); mad=np.median(np.abs(tr-m))*1.4826; return max(FLOOR,m+K_MAD*mad)
def event_peaks(tr,thr):
    idx=np.where(tr>thr)[0]; pk=[]
    if idx.size:
        for r in np.split(idx,np.where(np.diff(idx)>1)[0]+1):
            if r.size>=MIN_DUR: pk.append(int(r[np.argmax(tr[r])]))
    return pk

def gidx(boolvol):
    Z,Y,X=boolvol.shape; lz,ly,lx=np.where(boolvol)
    return (lz.astype(np.int64)*Y*X+ly*X+lx)

def load_f0_surface(rawp, skip_s, fs):
    tf=tifffile.TiffFile(str(rawp)); T,Z,Y,X=tf.series[0].shape
    try: arr=tifffile.memmap(str(rawp))
    except Exception: arr=tf.series[0].asarray()
    skip=int(skip_s*fs); fidx=np.unique(np.linspace(skip,T-1,NF0).astype(int))
    samp=np.asarray(arr[fidx]).astype(np.float32)
    mv=samp.mean(0); f0=np.percentile(samp,10,axis=0); del samp; gc.collect()
    return tf,arr,T,Z,Y,X,mv,f0

def analyze_fov(label,date,mouse,mask_run,raw_runs,fs,skip_s):
    base=PR/"scape-data"/date/mouse
    # ---- geometry/segmentation ONCE (masks shared across raw_runs) ----
    rawp0=raw_path(base,mask_run) or raw_path(base,raw_runs[0])
    tf0,arr0,T0,Z,Y,X,mv0,f00=load_f0_surface(rawp0,skip_s,fs)
    sp=fit_surface(mv0); tf0.close()
    print(f"[{label}] surface tilt_X={sp['tilt_x']:+.1f}deg tilt_Z={sp['tilt_z']:+.1f}deg | "
          f"masks={mask_run} runs={raw_runs}")
    mpaths=sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))
    masks={}; seg_rows=[]
    for p in mpaths:
        nm=p.stem.replace("_labelmap","")
        m=tifffile.imread(p)>0
        if m.shape[1]<Y: m=np.pad(m,((0,0),(0,Y-m.shape[1]),(0,0)))
        elif m.shape[1]>Y: m=m[:,:-(m.shape[1]-Y),:]
        lab,nseg=segment_mask(m)
        inner=binary_dilation(m,B2); outer=binary_dilation(m,B3); shell=outer&~inner
        segs={}
        for s in range(1,nseg+1):
            sm=lab==s
            if sm.sum()<MIN_SEG_VOX: continue
            core=binary_erosion(sm,B1); core=core if core.any() else sm
            cz=np.argwhere(sm)
            segs[s]=dict(idx=gidx(core),
                         depth_corr=float(corrected_depth(cz,sp).mean()),
                         depth_raw=float(cz[:,1].mean()*VOXEL[1]),
                         xcen=float(cz[:,2].mean()*VOXEL[2]), nvox=int(sm.sum()))
        if len(segs)<1: continue
        cw=binary_erosion(m,B1); cw=cw if cw.any() else m
        masks[nm]=dict(whole=gidx(cw), shell=gidx(shell), segs=segs, nseg=len(segs))
        for s,info in segs.items():
            seg_rows.append(dict(fov=label,name=nm,seg=s,**{k:info[k] for k in
                            ('depth_corr','depth_raw','xcen','nvox')}))
    nseg_dist=np.array([v['nseg'] for v in masks.values()])
    print(f"   masks usable={len(masks)} segments/mask: mean={nseg_dist.mean():.2f} "
          f"max={nseg_dist.max()} ; multiseg masks={(nseg_dist>=2).sum()}")

    # ---- extract + detect events per raw run, pooling ----
    rows=[]
    for run in raw_runs:
        rawp=raw_path(base,run)
        if rawp is None: print(f"   [{run}] no raw stack, skip"); continue
        tf,arr,T,Zr,Yr,Xr,mv,f0=load_f0_surface(rawp,skip_s,fs)
        if (Zr,Yr,Xr)!=(Z,Y,X): print(f"   [{run}] shape mismatch {(Zr,Yr,Xr)} vs {(Z,Y,X)}, skip"); tf.close(); continue
        f0flat=f0.ravel(); f0e=f0flat+1e-6
        seg_f0={nm:{s:float(f0flat[masks[nm]['segs'][s]['idx']].mean()) for s in masks[nm]['segs']}
                for nm in masks}
        traces={nm:{'whole':np.empty(T,np.float32),
                    'seg':{s:np.empty(T,np.float32) for s in d['segs']}} for nm,d in masks.items()}
        for t in range(T):
            dff=(np.asarray(arr[t]).astype(np.float32).ravel()-f0flat)/f0e
            for nm,d in masks.items():
                sh=dff[d['shell']].mean() if d['shell'].size else 0.0
                traces[nm]['whole'][t]=dff[d['whole']].mean()-sh
                for s,info in d['segs'].items():
                    traces[nm]['seg'][s][t]=dff[info['idx']].mean()-sh
        tf.close()
        nev=0
        for nm,d in masks.items():
            w=gaussian_filter1d(traces[nm]['whole'],SMOOTH_SIGMA)*100.0
            segtr={s:gaussian_filter1d(traces[nm]['seg'][s],SMOOTH_SIGMA)*100.0 for s in d['segs']}
            base_seg={s:float(np.median(segtr[s])) for s in d['segs']}   # baseline per segment
            thw=thresh(w); nseg=len(d['segs'])
            for p in event_peaks(w,thw):
                lo,hi=max(0,p-EVTWIN),min(len(w),p+EVTWIN+1)
                # baseline-subtracted windowed transient per segment (SNR-robust, smoothed)
                a={s:float(segtr[s][lo:hi].max()-base_seg[s]) for s in d['segs']}
                peak_amp=max(max(a.values()),1e-6)
                active=[s for s in d['segs'] if a[s]>=REL_FRAC*peak_amp and a[s]>=MIN_ACT_ABS]
                if not active: continue
                amp_local=max(a[s] for s in active)
                best=max(active,key=lambda s:a[s])
                sorted_a=sorted(a.values(),reverse=True)        # top transients (for REL_FRAC sweep)
                a1=float(sorted_a[0]); a2=float(sorted_a[1]) if len(sorted_a)>1 else 0.0
                klass="localized" if len(active)==1 else "extended"
                rows.append(dict(fov=label,run=run,name=nm,frame=p,nseg=nseg,
                                 n_active=len(active),recruit_frac=len(active)/nseg,
                                 amp_local=amp_local,amp_whole=float(w[p]),klass=klass,
                                 a1=a1,a2=a2,
                                 best_depth_corr=d['segs'][best]['depth_corr'],
                                 best_xcen=d['segs'][best]['xcen'],
                                 best_f0=seg_f0[nm][best])); nev+=1
        print(f"   [{run}] events={nev}")
        del traces; gc.collect()
    edf=pd.DataFrame(rows)
    if not edf.empty:
        ms=edf[edf.nseg>=2]
        vs=ms[ms.klass=="extended"]; lo=ms[ms.klass=="localized"]
        print(f"   POOLED events(multiseg)={len(ms)} extended={len(vs)} localized={len(lo)}")
        if len(vs)>3 and len(lo)>3:
            u,pu=stats.mannwhitneyu(vs.amp_local,lo.amp_local,alternative="greater")
            print(f"   amp_local median: extended={vs.amp_local.median():.1f}% "
                  f"localized={lo.amp_local.median():.1f}% "
                  f"({vs.amp_local.median()/max(lo.amp_local.median(),1e-6):.2f}x) MWU p={pu:.3g}")
    return edf, pd.DataFrame(seg_rows), sp

def main():
    ev=[]; seg=[]; sps={}
    for cfg in FOVS:
        e,s,sp=analyze_fov(*cfg)
        if not e.empty: ev.append(e); seg.append(s); sps[cfg[0]]=sp
    E=pd.concat(ev,ignore_index=True); S=pd.concat(seg,ignore_index=True)
    E.to_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_events.csv",index=False)
    S.to_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_segments.csv",index=False)

    M=E[E.nseg>=2].copy()
    print("\n============ POOLED (multi-segment masks, within-FOV z amp_local) ============")
    M["amp_z"]=M.groupby("fov")["amp_local"].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
    vs=M[M.klass=="extended"]; lo=M[M.klass=="localized"]
    if len(vs)>3 and len(lo)>3:
        u,pu=stats.mannwhitneyu(vs.amp_z,lo.amp_z,alternative="greater")
        print(f"  extended n={len(vs)} median_z={vs.amp_z.median():+.2f} | "
              f"localized n={len(lo)} median_z={lo.amp_z.median():+.2f}  MWU p={pu:.3g}")
    pr=[]
    for (f,nm),g in M.groupby(["fov","name"]):
        a=g[g.klass=="extended"].amp_local; b=g[g.klass=="localized"].amp_local
        if len(a)>=2 and len(b)>=2: pr.append((a.median(),b.median()))
    if len(pr)>=4:
        a=np.array([x[0] for x in pr]); b=np.array([x[1] for x in pr])
        w,pw=stats.wilcoxon(a,b,alternative="greater")
        print(f"  within-mask paired n={len(pr)}: extended>localized {(a>b).sum()}/{len(pr)} "
              f"Wilcoxon p={pw:.3g}")
    # ILLUMINATION control: does amp depend on X / brightness? does class differ in X?
    print("\n  illumination/geometry controls (multiseg events):")
    rX,pX=stats.spearmanr(M.best_xcen,M.amp_local)
    rF,pF=stats.spearmanr(M.best_f0,M.amp_local)
    print(f"    amp_local vs X-centroid:  ρ={rX:+.2f} p={pX:.2g}  (illumination gradient axis)")
    print(f"    amp_local vs baseline F0: ρ={rF:+.2f} p={pF:.2g}")
    if len(vs)>3 and len(lo)>3:
        uX,pXc=stats.mannwhitneyu(vs.best_xcen,lo.best_xcen)
        uF,pFc=stats.mannwhitneyu(vs.best_f0,lo.best_f0)
        print(f"    extended vs localized X-centroid: med {vs.best_xcen.median():.0f} vs "
              f"{lo.best_xcen.median():.0f} um (MWU p={pXc:.2g})  -> if NS, X not a confound")
        print(f"    extended vs localized baseline F0: med {vs.best_f0.median():.0f} vs "
              f"{lo.best_f0.median():.0f} (MWU p={pFc:.2g})")
    # depth (corrected) of localized events
    rD,pD=stats.spearmanr(M.best_depth_corr,M.amp_local)
    print(f"    amp_local vs corrected depth: ρ={rD:+.2f} p={pD:.2g}")
    make_fig(M)
    print("\nSaved nmda_bap_split_v2_events.csv, _segments.csv, figures/nmda_bap_split_v2.png")

def make_fig(M):
    (PR/"code"/"Testing"/"figures").mkdir(parents=True,exist_ok=True)
    fovs=list(M.fov.unique()); fig,ax=plt.subplots(1,len(fovs)+2,figsize=(4.4*(len(fovs)+2),4.2))
    for i,fov in enumerate(fovs):
        g=M[M.fov==fov]
        data=[g[g.klass=="localized"].amp_local.values,g[g.klass=="extended"].amp_local.values]
        ax[i].boxplot(data,tick_labels=[f"localized\n(NMDA)\nn={len(data[0])}",
                                        f"extended\n(bAP)\nn={len(data[1])}"],showfliers=False)
        ax[i].set_title(fov); ax[i].set_ylabel("amp_local ΔF/F (%)"); ax[i].grid(alpha=.3)
    ax[-2].scatter(M.recruit_frac,M.amp_local,s=9,alpha=.3,
                   c=np.where(M.klass=="extended","tab:blue","tab:red"))
    r,p=stats.spearmanr(M.recruit_frac,M.amp_local)
    ax[-2].set_xlabel("segment recruitment fraction"); ax[-2].set_ylabel("amp_local ΔF/F (%)")
    ax[-2].set_title(f"recruitment vs amp  ρ={r:+.2f}"); ax[-2].grid(alpha=.3)
    ax[-1].scatter(M.best_xcen,M.amp_local,s=9,alpha=.3,c='tab:purple')
    rX,_=stats.spearmanr(M.best_xcen,M.amp_local)
    ax[-1].set_xlabel("active-segment X-centroid (um)"); ax[-1].set_ylabel("amp_local ΔF/F (%)")
    ax[-1].set_title(f"illumination control  ρ={rX:+.2f}"); ax[-1].grid(alpha=.3)
    fig.suptitle("Test 2 v2: branch/bend segmentation — localized (NMDA) vs extended (bAP) amplitude",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(PR/"code"/"Testing"/"figures"/"nmda_bap_split_v2.png",dpi=150)
    plt.close(fig)

if __name__=="__main__":
    main()
