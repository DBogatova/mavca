#!/usr/bin/env python
"""
STRAIGHTNESS-based decomposition (ORIGINAL version, reverted per request):
  - Skeletonize; build graph; LONGEST path = trunk backbone.
  - RDP-split the backbone at bends; the longest straight run (+ collinear neighbours) = TRUNK
    (any orientation). Divergent backbone runs + all off-backbone branches = BRANCH.
  - Whole-mask: 'trunk' (straight only), 'branch' (no straight trunk), 'combined' (trunk + branch).
Verticality NOT used (cells stay vertical despite the pial slope). Edit FOV below.
"""
from pathlib import Path
import numpy as np, tifffile, importlib.util, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import label as cclabel
from skimage.morphology import skeletonize
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
v=importlib.util.module_from_spec(importlib.util.spec_from_file_location("v",PR/"code/Testing/nmda_bap_split_v2.py"))
importlib.util.spec_from_file_location("v",PR/"code/Testing/nmda_bap_split_v2.py").loader.exec_module(v)
VOX=v.VOXEL

FOVS=[("0416_r567","2026-04-16","rbp4_132_phpeb","run7"),
      ("0508_run5","2026-05-08","rbp4_139_phpeb","run5"),
      ("0512_run5","2026-05-12","rbp4_132_phpeb","run5"),
      ("0512_run9","2026-05-12","rbp4_132_phpeb","run9"),
      ("0320_run1","2026-03-20","rbp4cre_139_phpeb","run1"),
      ("0320_run3","2026-03-20","rbp4cre_139_phpeb","run3")]
TRUNK_MIN=35.0; LIN_TRUNK=0.80; RDP_EPS=8.0; COLLINEAR=30.0
MIN_CLUST=50; MAX_CLUST=6; MIN_MASK_VOX=40; CONN=np.ones((3,3,3))
DEPTH_TRUNK_Y=None   # if set: any voxel with Y > this (deeper, below the line) is forced to trunk
# branch-reality + trunk-dominance post-process (0 disables each):
BRANCH_ANGLE=45.0      # a 'branch' is real only if its axis diverges from the trunk by >= this (deg)
BRANCH_MIN_FRAC=0.10   # ...and is >= this fraction of mask voxels; else it's reabsorbed into trunk
TRUNK_FRAC=0.30        # after that, trunk must be >= this fraction of voxels; else whole mask = branch

def line_axis(coords):
    pts=coords*VOX; c=pts.mean(0); cov=np.cov((pts-c).T)
    w,V=np.linalg.eigh(cov); return V[:,::-1][:,0]

def refine(full,ncls,total):
    """Reabsorb collinear/small 'branch' clusters into trunk; then enforce trunk dominance."""
    tcids=[c for c,t in ncls.items() if t=='trunk']
    tvox=np.argwhere(np.isin(full,tcids)) if tcids else np.empty((0,3),int)
    ta=line_axis(tvox) if len(tvox)>=8 else None
    for cid,t in list(ncls.items()):
        if t!='branch': continue
        bv=np.argwhere(full==cid); frac=len(bv)/max(total,1)
        coll=False
        if ta is not None and len(bv)>=8:
            ang=np.degrees(np.arccos(np.clip(abs(line_axis(bv)@ta),0,1))); coll=ang<BRANCH_ANGLE
        if frac<BRANCH_MIN_FRAC or coll: ncls[cid]='trunk'   # not a real branch -> trunk
    tvox2=np.argwhere(np.isin(full,[c for c,t in ncls.items() if t=='trunk']))
    tfrac=len(tvox2)/max(total,1)
    tlin=line_len(tvox2)[1] if len(tvox2)>=8 else 0.0        # is the (merged) trunk actually straight?
    if tfrac<TRUNK_FRAC or tlin<LIN_TRUNK:
        for cid in ncls: ncls[cid]='branch'                  # no dominant straight trunk -> all branch
    return ncls
CAT_COL={'trunk':(0.20,0.45,0.95),'branch':(0.92,0.22,0.20),'combined':(1.0,0.55,0.0)}
CLU_COL={'trunk':(0.20,0.45,0.95),'branch':(0.92,0.22,0.20)}

def line_len(coords):
    if len(coords)<2: return 0.0,0.0
    pts=coords*VOX; c=pts.mean(0); cov=np.cov((pts-c).T)
    w,V=np.linalg.eigh(cov); w=np.clip(w[::-1],1e-9,None)
    proj=(pts-c)@V[:,::-1][:,0]; return float(proj.max()-proj.min()), float((w[0]-w[1])/w[0])

def merge_small(clab,cls):
    def step(force):
        u=[x for x in np.unique(clab) if x]
        if len(u)<=1: return False
        sz={k:int((clab==k).sum()) for k in u}
        if force and len(u)<=MAX_CLUST: return False
        cand=[k for k in u if sz[k]<MIN_CLUST] if not force else u
        if not cand: return False
        cen={k:np.argwhere(clab==k).mean(0) for k in u}
        k=min(cand,key=lambda z:sz[z]); oth=[o for o in u if o!=k]
        t=min(oth,key=lambda o:np.linalg.norm((cen[o]-cen[k])*VOX)); clab[clab==k]=t; cls.pop(k,None); return True
    while step(False): pass
    while step(True): pass
    return clab,cls

def decompose(m):
    co=np.argwhere(m)
    def whole():
        L,lin=line_len(co); cat='trunk' if (L>=TRUNK_MIN and lin>=LIN_TRUNK) else 'branch'
        if DEPTH_TRUNK_Y is not None and cat=='branch' and np.median(co[:,1])>DEPTH_TRUNK_Y: cat='trunk'
        lab=np.zeros(m.shape,np.int32); lab[m]=1
        return lab,{1:cat},cat,dict(trunk_len=L,trunk_lin=lin,nbranch=0)
    if len(co)<MIN_MASK_VOX: return whole()
    Z,Y,X=m.shape; pad=2
    z0,z1=max(0,co[:,0].min()-pad),min(Z,co[:,0].max()+1+pad); y0,y1=max(0,co[:,1].min()-pad),min(Y,co[:,1].max()+1+pad)
    x0,x1=max(0,co[:,2].min()-pad),min(X,co[:,2].max()+1+pad); sub=m[z0:z1,y0:y1,x0:x1]
    sk=skeletonize(sub)
    if sk.sum()<5: return whole()
    skc=np.argwhere(sk); idx={tuple(p):i for i,p in enumerate(skc)}
    offs=[(a,b,c) for a in(-1,0,1) for b in(-1,0,1) for c in(-1,0,1) if (a,b,c)!=(0,0,0)]
    G=nx.Graph()
    for i,p in enumerate(skc):
        for o in offs:
            j=idx.get((p[0]+o[0],p[1]+o[1],p[2]+o[2]))
            if j is not None and j>i: G.add_edge(i,j,weight=float(np.linalg.norm(np.array(o)*VOX)))
    if G.number_of_nodes()<2: return whole()
    H=G.subgraph(max(nx.connected_components(G),key=len))
    deg=dict(H.degree()); ends=[n for n in H if deg[n]==1] or [next(iter(H.nodes()))]
    def far(s): d,p=nx.single_source_dijkstra(H,s,weight='weight'); t=max(d,key=d.get); return t,p[t]
    a,_=far(ends[0]); b,path=far(a); backbone=skc[path]
    keep=np.where(v._rdp_keep(backbone*VOX,RDP_EPS))[0]
    if len(keep)<2: keep=np.array([0,len(backbone)-1])
    runs=[(keep[i],keep[i+1]) for i in range(len(keep)-1)]
    def rdir(s,e):
        d=(backbone[e]-backbone[s])*VOX; n=np.linalg.norm(d); return (d/n if n>0 else d),n
    ri=[(s,e)+rdir(s,e) for s,e in runs]
    li=max(range(len(ri)),key=lambda k:ri[k][3]); tdir=ri[li][2]
    trunk={k for k in range(len(ri)) if np.degrees(np.arccos(np.clip(abs(ri[k][2]@tdir),0,1)))<=COLLINEAR}
    bblab=np.array(['branch']*len(backbone),dtype=object)
    for k,(s,e,_,_) in enumerate(ri): bblab[s:e+1]='trunk' if k in trunk else 'branch'
    sklab=np.array(['branch']*len(skc),dtype=object); sklab[path]=bblab
    tree=cKDTree(skc*VOX); mv=np.argwhere(sub); _,nn=tree.query(mv*VOX,k=1); vlab=sklab[nn].copy()
    tv=mv[vlab=='trunk']; Ltr,lintr=line_len(tv) if len(tv)>=8 else (0.0,0.0)
    if Ltr<TRUNK_MIN or lintr<LIN_TRUNK: vlab[:]='branch'; Ltr,lintr=line_len(co)
    if DEPTH_TRUNK_Y is not None:                       # deep voxels can't be branch -> trunk
        vlab[(mv[:,1]+y0)>DEPTH_TRUNK_Y]='trunk'
        tvd=mv[vlab=='trunk']
        if len(tvd)>=8: Ltr,lintr=line_len(tvd)
    clab=np.zeros(sub.shape,np.int32); cls={}; nid=0
    for c in ('trunk','branch'):
        vol=np.zeros(sub.shape,bool); sel=mv[vlab==c]
        if len(sel): vol[sel[:,0],sel[:,1],sel[:,2]]=True
        cc,nc=cclabel(vol,structure=CONN)
        for i in range(1,nc+1): nid+=1; clab[cc==i]=nid; cls[nid]=c
    clab,cls=merge_small(clab,cls)
    u=[x for x in np.unique(clab) if x]; remap={x:i+1 for i,x in enumerate(u)}
    out=np.zeros(sub.shape,np.int32); ncls={}
    for x,i in remap.items(): out[clab==x]=i; ncls[i]=cls[x]
    full=np.zeros(m.shape,np.int32); full[z0:z1,y0:y1,x0:x1]=out
    if BRANCH_ANGLE>0 or BRANCH_MIN_FRAC>0 or TRUNK_FRAC>0:
        ncls=refine(full,ncls,int(m.sum()))
    nT=sum(c=='trunk' for c in ncls.values()); nB=sum(c=='branch' for c in ncls.values())
    cat='trunk' if (nT>=1 and nB==0) else ('branch' if nT==0 else 'combined')
    return full,ncls,cat,dict(trunk_len=Ltr,trunk_lin=lintr,nbranch=nB)

def why(cat,info):
    if cat=='trunk':  return f"trunk: straight {info['trunk_len']:.0f}um lin{info['trunk_lin']:.2f}"
    if cat=='branch': return f"branch: no straight trunk (len{info['trunk_len']:.0f}um lin{info['trunk_lin']:.2f})"
    return f"combined: trunk {info['trunk_len']:.0f}um + {info['nbranch']} branch"

def shifts(a):
    d=np.zeros(a.shape,bool); d[1:]|=a[1:]!=a[:-1]; d[:-1]|=a[:-1]!=a[1:]; d[:,1:]|=a[:,1:]!=a[:,:-1]; d[:,:-1]|=a[:,:-1]!=a[:,1:]; return d

def gallery(label,date,mouse,mr):
    base=PR/"scape-data"/date/mouse
    items=[]
    for p in sorted((base/mr/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif")):
        nm=p.stem.replace("_labelmap",""); m=tifffile.imread(p)>0
        clab,cls,cat,info=decompose(m)
        cid=clab.max(0); foot=cid>0; ys,xs=np.where(foot)
        if ys.size==0: continue
        y0,y1=max(0,ys.min()-2),ys.max()+3; x0,x1=max(0,xs.min()-2),xs.max()+3; subm=cid[y0:y1,x0:x1]
        rgb=np.ones((*subm.shape,3))*0.93
        for cidv in np.unique(subm):
            if cidv==0: continue
            rgb[subm==cidv]=CLU_COL[cls[cidv]]
        if cat=='combined': rgb[shifts(subm)&(subm>0)]=[0,0,0]
        items.append((nm,cat,why(cat,info),rgb))
    n=len(items); ncol=11; nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*1.6,nrow*1.95)); axes=np.atleast_1d(axes).ravel()
    for ax,(nm,cat,wt,rgb) in zip(axes,items):
        ax.imshow(rgb,aspect='equal',interpolation='nearest')
        ax.set_title(f"{nm}\n{cat}",color=CAT_COL[cat],fontsize=5,pad=1)
        ax.text(0.5,-0.04,wt,transform=ax.transAxes,fontsize=3.6,ha='center',va='top')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(CAT_COL[cat]); s.set_linewidth(1.4)
    for ax in axes[n:]: ax.axis('off')
    nT=sum(c=='trunk' for _,c,_,_ in items); nB=sum(c=='branch' for _,c,_,_ in items); nC=sum(c=='combined' for _,c,_,_ in items)
    fig.suptitle(f"{label}: morphology (blue=trunk, red=branch, orange title=combined, black=cut)   "
                 f"trunk={nT} branch={nB} combined={nC}",fontsize=11,y=0.998)
    fig.tight_layout(rect=[0,0,1,0.99])
    out=PR/"code"/"Testing"/"figures"/f"nmda_gallery2_{label}.png"; fig.savefig(out,dpi=170); plt.close(fig)
    print("saved",out.name,"| trunk=%d branch=%d combined=%d"%(nT,nB,nC))

def main():
    for cfg in FOVS: gallery(*cfg)

if __name__=="__main__": main()
