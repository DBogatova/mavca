#!/usr/bin/env python
"""
Hierarchical, morphology-only labelling of masks (no traces):

  1) Whole-mask class:
       vertical  (Ca/bAP)  : straight, vertical (~along image-Y), few/no branch points
       branch    (NMDA)     : single small/tilted piece
       combined            : >=2 skeleton branch points (multi-branch arbor)
  2) For 'combined' masks: skeletonize, label each skeleton branch vertical vs tilted
     (angle to image-Y), then CLUSTER connected same-type voxels into a few coherent
     sub-units (collinear vertical limbs vs off-axis branches), merging tiny pieces and
     capping the count. Each cluster is classified vertical (Ca) or branch (NMDA).

Verticality is measured vs image-Y (cells stay vertical despite pial slope - verified).
Output: per-FOV maps (whole-mask class | cluster decomposition) + scape-data/nmda_bap_cluster_table.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import convolve, label as cclabel, binary_erosion
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("v",PR/"code/Testing/nmda_bap_split_v2.py")
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
VOXEL=v.VOXEL
NBK=np.ones((3,3,3)); NBK[1,1,1]=0; CONN=np.ones((3,3,3))

ANG_V=30.0       # deg from image-Y: a branch/cluster within this = 'vertical'
LIN_V=0.65       # linearity for vertical
LEN_V=20.0       # um min length for vertical
PRUNE_VOX=6
MIN_CLUST=60     # min voxels per cluster (merge smaller)
MAX_CLUST=5
MIN_MASK_VOX=40

def shape_of(coords):
    pts=coords*VOXEL; c=pts.mean(0)
    if len(coords)<8: return dict(ang=90.0,lin=0.0,length=0.0)
    cov=np.cov((pts-c).T); w,V=np.linalg.eigh(cov); w=np.clip(w[::-1],1e-9,None); V=V[:,::-1]
    pa=V[:,0]; proj=(pts-c)@pa
    return dict(ang=float(np.degrees(np.arccos(np.clip(abs(pa[1]),0,1)))),
                lin=float((w[0]-w[1])/w[0]), length=float(proj.max()-proj.min()))

def is_vertical(sh): return (sh['ang']<=ANG_V) and (sh['lin']>=LIN_V) and (sh['length']>=LEN_V)

def cluster_mask(m):
    """Return (cluster_label_volume, {cid:'vertical'/'branch'}, n_branch_points, category)."""
    zz,yy,xx=np.where(m); 
    if zz.size<MIN_MASK_VOX:
        lab=m.astype(np.int32); cls={1:'branch' if not is_vertical(shape_of(np.argwhere(m))) else 'vertical'}
        return lab,cls,0,list(cls.values())[0]
    Z,Y,X=m.shape; pad=3
    z0,z1=max(0,zz.min()-pad),min(Z,zz.max()+1+pad); y0,y1=max(0,yy.min()-pad),min(Y,yy.max()+1+pad)
    x0,x1=max(0,xx.min()-pad),min(X,xx.max()+1+pad); sub=m[z0:z1,y0:y1,x0:x1]
    sk=skeletonize(sub); nbp=0
    def whole():
        lab=np.zeros(m.shape,np.int32); lab[m]=1
        sh=shape_of(np.argwhere(m)); c='vertical' if is_vertical(sh) else 'branch'
        return lab,{1:c},0,c
    if sk.sum()<5: return whole()
    nb=convolve(sk.astype(int),NBK,mode='constant'); bp=sk&(nb>=3); nbp=int(bp.sum())
    chain=sk&~bp; cc,ncc=cclabel(chain,structure=CONN)
    branches=[np.argwhere(cc==i) for i in range(1,ncc+1) if (cc==i).sum()>=PRUNE_VOX]
    if len(branches)<1: return whole()
    # label each skeleton branch vertical/tilted, assign voxels to nearest branch
    skel=np.vstack(branches)
    blab=np.concatenate([np.full(len(b),i) for i,b in enumerate(branches)])
    bvert=np.array([is_vertical(shape_of(b)) if len(b)>=8 else
                    (shape_of(b)['ang']<=ANG_V) for b in branches])
    tree=cKDTree(skel*VOXEL); mv=np.argwhere(sub); _,nn=tree.query(mv*VOXEL,k=1)
    vbvol=np.zeros(sub.shape,np.int8)  # 1=vertical-voxel, 2=tilted-voxel
    vbvol[mv[:,0],mv[:,1],mv[:,2]]=np.where(bvert[blab[nn]],1,2)
    # connected components within each type -> clusters
    clab=np.zeros(sub.shape,np.int32); nid=0; cls={}
    for typ in (1,2):
        ccx,nx=cclabel(vbvol==typ,structure=CONN)
        for i in range(1,nx+1):
            nid+=1; clab[ccx==i]=nid
    clab=merge_small(clab)
    # relabel + classify each final cluster by its own geometry
    uniq=[u for u in np.unique(clab) if u]; remap={u:i+1 for i,u in enumerate(uniq)}
    out=np.zeros(sub.shape,np.int32)
    for u,i in remap.items(): out[clab==u]=i
    full=np.zeros(m.shape,np.int32); full[z0:z1,y0:y1,x0:x1]=out
    cls={}
    for i in range(1,len(uniq)+1):
        sh=shape_of(np.argwhere(full==i)); cls[i]='vertical' if is_vertical(sh) else 'branch'
    nclust=len(uniq)
    if nclust<=1: cat=list(cls.values())[0] if cls else 'branch'
    else: cat='combined'
    return full,cls,nbp,cat

def merge_small(lab):
    def step(force):
        u=[x for x in np.unique(lab) if x]; 
        if len(u)<=1: return False
        sz={k:int((lab==k).sum()) for k in u}
        if force and len(u)<=MAX_CLUST: return False
        cand=[k for k in u if sz[k]<MIN_CLUST] if not force else u
        if not cand: return False
        cen={k:np.argwhere(lab==k).mean(0) for k in u}
        k=min(cand,key=lambda z:sz[z]); oth=[o for o in u if o!=k]
        tgt=min(oth,key=lambda o:np.linalg.norm((cen[o]-cen[k])*VOXEL)); lab[lab==k]=tgt; return True
    while step(False): pass
    while step(True): pass
    return lab

def bg_mip(rawp,fs,skip,nf=40):
    tf=tifffile.TiffFile(str(rawp)); T=tf.series[0].shape[0]
    try: arr=tifffile.memmap(str(rawp))
    except Exception: arr=tf.series[0].asarray()
    idx=np.unique(np.linspace(int(skip*fs),T-1,nf).astype(int))
    mv=np.asarray(arr[idx]).astype(np.float32).mean(0); tf.close(); return mv.max(0)

def shifts(a):
    d=np.zeros(a.shape,bool)
    d[1:]|=a[1:]!=a[:-1]; d[:-1]|=a[:-1]!=a[1:]; d[:,1:]|=a[:,1:]!=a[:,:-1]; d[:,:-1]|=a[:,:-1]!=a[:,1:]
    return d

CAT_COL={'vertical':(0.20,0.45,0.95),'branch':(0.92,0.22,0.20),'combined':(1.0,0.55,0.0)}

def main():
    rows=[]
    for label,date,mouse,mask_run,runs,fs,skip in v.FOVS:
        base=PR/"scape-data"/date/mouse; rawp=v.raw_path(base,mask_run)
        mip=bg_mip(rawp,fs,skip); Y,X=mip.shape
        cat_id=np.zeros((Y,X),np.int32); clu_id=np.zeros((Y,X),np.int32); clu_vb=np.zeros((Y,X),np.int8)
        mid=np.zeros((Y,X),np.int32); CATMAP={'vertical':1,'branch':2,'combined':3}
        gid=0; nV=nB=nC=0
        for mi,p in enumerate(sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif")),1):
            nm=p.stem.replace("_labelmap",""); m=tifffile.imread(p)>0
            if m.shape[1]<Y: m=np.pad(m,((0,0),(0,Y-m.shape[1]),(0,0)))
            elif m.shape[1]>Y: m=m[:,:-(m.shape[1]-Y),:]
            clab,cls,nbp,cat=cluster_mask(m)
            nV+=cat=='vertical'; nB+=cat=='branch'; nC+=cat=='combined'
            foot=(m).max(0); mid[foot]=mi; cat_id[foot]=CATMAP[cat]
            for cid,c in cls.items():
                cf=(clab==cid).max(0); 
                if not cf.any(): continue
                gid+=1; clu_id[cf]=gid; clu_vb[cf]=1 if c=='vertical' else 2
            rows.append(dict(fov=label,name=nm,category=cat,n_branch_pts=nbp,
                             n_clusters=len(cls),n_vert=sum(c=='vertical' for c in cls.values()),
                             n_branch=sum(c=='branch' for c in cls.values())))
        # ---- figure ----
        g=mip.astype(np.float32); g=np.clip((g-g.min())/(np.percentile(g,99.5)-g.min()+1e-9),0,1)
        base_rgb=np.dstack([g,g,g])*0.85
        figA=base_rgb.copy()
        for cat,code in CATMAP.items():
            figA[cat_id==code]=0.45*figA[cat_id==code]+0.55*np.array(CAT_COL[cat])
        figA[shifts(mid)&(mid>0)]=[1,1,1]
        figB=base_rgb.copy()
        figB[clu_vb==1]=0.45*figB[clu_vb==1]+0.55*np.array(CAT_COL['vertical'])
        figB[clu_vb==2]=0.45*figB[clu_vb==2]+0.55*np.array(CAT_COL['branch'])
        figB[shifts(clu_id)&(clu_id>0)&~shifts(mid)]=[0,0,0]   # cluster cut lines
        figB[shifts(mid)&(mid>0)]=[1,1,1]
        fig,ax=plt.subplots(1,2,figsize=(2*X/45+2,Y/45+1.3))
        ax[0].imshow(figA,aspect='equal'); ax[0].set_title(f"{label}: whole-mask class "
                     f"(vertical={nV} branch={nB} combined={nC})")
        ax[0].legend(handles=[Patch(color=CAT_COL['vertical'],label='vertical (Ca)'),
                              Patch(color=CAT_COL['branch'],label='branch (NMDA)'),
                              Patch(color=CAT_COL['combined'],label='combined (multi-branch)')],
                     fontsize=7,loc='lower right')
        ax[1].imshow(figB,aspect='equal'); ax[1].set_title(f"{label}: branch-cluster decomposition "
                     f"(blue=vertical/Ca, red=branch/NMDA)")
        for a in ax: a.set_xlabel("X"); a.set_ylabel("Y (depth)"); a.set_ylim(Y-.5,-.5); a.set_xlim(-.5,X-.5)
        fig.tight_layout(); fig.savefig(PR/"code"/"Testing"/"figures"/f"nmda_cluster_{label}.png",dpi=160); plt.close(fig)
        print(f"[{label}] vertical={nV} branch={nB} combined={nC} (n={nV+nB+nC})")
    pd.DataFrame(rows).to_csv(PR/"code"/"Testing"/"data"/"nmda_bap_cluster_table.csv",index=False)
    print("saved scape-data/nmda_bap_cluster_table.csv + figures/nmda_cluster_*.png")

if __name__=="__main__": main()
