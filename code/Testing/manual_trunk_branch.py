#!/usr/bin/env python
"""
Manual trunk/branch annotation (Napari) — ground truth for tuning the classifier.
3D viewing + 2D annotation. Branch marks are collapsed to the Y×X MIP and broadcast across Z
on save (so it doesn't matter which Z slice you mark). Every mask starts ALL TRUNK (blue);
mark BRANCH parts (red). "All trunk" = leave it and press Right.

Hotkeys:
  Left/Right : prev / next mask (autosaves)
  t          : toggle 3D / 2D  (drops to the mask's Z slice in 2D so it's visible)
  a          : whole mask = TRUNK (reset)   |   x : whole mask = BRANCH
  l          : lasso (polygon, 2D) INSIDE -> BRANCH   |   j : lasso INSIDE -> TRUNK
  b          : toggle background              |   Ctrl+S : save all
Paint: select 'label' layer, active label 2 (branch) / 1 (trunk), brush (any slice — broadcast on save).
Saves manual_trunk_branch/<name>_tb.tif (3D, 1=trunk/2=branch) + manual_labels.csv.
"""
from pathlib import Path
import numpy as np, tifffile, csv, napari
from skimage.draw import polygon2mask

DATE,MOUSE,RUN = "2026-05-08","rbp4_139_phpeb","run5"      # <- edit per FOV
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
BASE=PR/"scape-data"/DATE/MOUSE/RUN
MASKDIR=BASE/"labelmaps_curated_dynamic"
OUT=BASE/"manual_trunk_branch"; OUT.mkdir(exist_ok=True)

def raw_path():
    c=[p for p in sorted((BASE/"raw").glob("*reslice-bin.tif")) if "dff" not in p.name]
    return c[0] if c else None

def load_bg3d(Y):
    rp=raw_path()
    if rp is None: return None
    tf=tifffile.TiffFile(str(rp)); T=tf.series[0].shape[0]
    try: arr=tifffile.memmap(str(rp))
    except Exception: arr=tf.series[0].asarray()
    idx=np.unique(np.linspace(60,T-1,100).astype(int))
    bg=np.asarray(arr[idx]).astype(np.float32).max(0)        # temporal max -> (Z,Y,X)
    tf.close()
    if bg.shape[1]>Y: bg=bg[:,:Y,:]
    elif bg.shape[1]<Y: bg=np.pad(bg,((0,0),(0,Y-bg.shape[1]),(0,0)))
    return bg

def main():
    paths=sorted(MASKDIR.glob("dend_*_labelmap.tif"))
    names=[p.stem.replace("_labelmap","") for p in paths]
    masks=[tifffile.imread(p)>0 for p in paths]
    N=len(masks); Z,Y,X=masks[0].shape; print(f"{N} masks in {DATE}/{MOUSE}/{RUN}")
    bg=load_bg3d(Y)
    lab={}                                                   # 3D label per mask (1=trunk,2=branch)
    for nm,m in zip(names,masks):
        f=OUT/f"{nm}_tb.tif"
        lab[nm]=tifffile.imread(f).astype(np.uint8) if f.exists() else m.astype(np.uint8)
    idx=[0]
    v=napari.Viewer(ndisplay=3)
    if bg is not None:
        v.add_image(bg,name="bg",colormap="gray",blending="additive",
                    contrast_limits=(float(np.percentile(bg,2)),float(np.percentile(bg,99.5))))
    v.add_labels(lab[names[0]],name="label",opacity=0.8); v.layers["label"].selected_label=2
    v.add_shapes(name="lasso",ndim=3,edge_color="yellow",face_color="transparent",edge_width=2)

    def branch_mip(nm):                                      # branch = marked 2 at ANY z, within mask
        m=masks[names.index(nm)]; return ((lab[nm]==2).any(0)) & m.max(0)
    def cat(nm):
        m=masks[names.index(nm)]; br=branch_mip(nm)
        nb=int((np.broadcast_to(br,m.shape)&m).sum()); nt=int(m.sum())-nb
        return "trunk" if nb==0 else ("branch" if nt==0 else "combined")
    def store(): lab[names[idx[0]]]=v.layers["label"].data.copy()
    def jumpZ():
        if v.dims.ndisplay==2:
            zc=int(masks[idx[0]].sum((1,2)).argmax()); v.dims.set_current_step(0,zc)
    def show(i):
        v.layers["label"].data=lab[names[i]]; v.layers["lasso"].data=[]; v.layers["label"].selected_label=2
        jumpZ(); print(f"[{i+1}/{N}] {names[i]}  ({cat(names[i])})")
    def save_all():
        rows=[]
        for nm,m in zip(names,masks):
            br=np.broadcast_to(branch_mip(nm),m.shape)
            tb=m.astype(np.uint8); tb[br&m]=2
            lab[nm]=tb; tifffile.imwrite(OUT/f"{nm}_tb.tif",tb)
            rows.append(dict(name=nm,category=cat(nm),n_trunk=int(((tb==1)&m).sum()),n_branch=int(((tb==2)&m).sum())))
        with open(OUT/"manual_labels.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["name","category","n_trunk","n_branch"]); w.writeheader(); w.writerows(rows)
        c=[r['category'] for r in rows]
        print(f"💾 saved {N} (trunk={c.count('trunk')} branch={c.count('branch')} combined={c.count('combined')})")

    @v.bind_key("Right")
    def nxt(viewer):
        store(); save_all()
        if idx[0]<N-1: idx[0]+=1; show(idx[0])
    @v.bind_key("Left")
    def prv(viewer):
        store()
        if idx[0]>0: idx[0]-=1; show(idx[0])
    @v.bind_key("t")
    def toggle(viewer):
        v.dims.ndisplay=2 if v.dims.ndisplay==3 else 3; jumpZ(); print(f"display={v.dims.ndisplay}D")
    @v.bind_key("a")
    def all_t(viewer):
        lab[names[idx[0]]]=masks[idx[0]].astype(np.uint8); show(idx[0]); print("→ all TRUNK")
    @v.bind_key("x")
    def all_b(viewer):
        m=masks[idx[0]]; l=m.astype(np.uint8); l[m]=2; lab[names[idx[0]]]=l; show(idx[0]); print("→ all BRANCH")
    def inside():
        polys=v.layers["lasso"].data
        if len(polys)==0: return None
        k=np.zeros((Y,X),bool)
        for p in polys: k|=polygon2mask((Y,X),np.asarray(p)[:,-2:])
        return np.broadcast_to(k,(Z,Y,X)) & masks[idx[0]]
    @v.bind_key("l")
    def las_b(viewer):
        store(); ins=inside()
        if ins is None: print("⚠️ draw a polygon (2D mode: press t) then 'l'"); return
        l=lab[names[idx[0]]].copy(); l[ins]=2; lab[names[idx[0]]]=l; show(idx[0]); print("⭕ -> BRANCH")
    @v.bind_key("j")
    def las_t(viewer):
        store(); ins=inside()
        if ins is None: print("⚠️ draw a polygon then 'j'"); return
        l=lab[names[idx[0]]].copy(); l[ins]=1; lab[names[idx[0]]]=l; show(idx[0]); print("⭕ -> TRUNK")
    @v.bind_key("b")
    def tbg(viewer):
        if "bg" in v.layers: v.layers["bg"].visible=not v.layers["bg"].visible
    @v.bind_key("Control-S")
    def _s(viewer): store(); save_all()

    print("\nLeft/Right navigate | t=3D/2D | a=all trunk | x=all branch | l=lasso->branch | j=lasso->trunk | b=bg | Ctrl+S")
    print("Paint: 'label' layer, active label 2=branch/1=trunk, brush (any slice; broadcast over Z on save).")
    show(0); napari.run()

if __name__=="__main__": main()
