#!/usr/bin/env python
"""Do apical-dendrite masks run along IMAGE-Y (vertical) or along the SURFACE NORMAL (tilted
with the pia)? For each elongated mask compute its principal axis and its angle to (a) image-Y
and (b) the fitted surface normal. Whichever reference gives the smaller angles is the correct
'radial/vertical' axis for the morphology classifier. (Depth correction is separate & stays.)"""
from pathlib import Path
import numpy as np, tifffile, importlib.util
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("v",PR/"code/Testing/nmda_bap_split_v2.py")
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
VOXEL=v.VOXEL; YHAT=np.array([0.0,1.0,0.0])   # image-Y in ZYX

def pax(mask):
    c=np.argwhere(mask)
    if len(c)<30: return None,0
    pts=c*VOXEL; ctr=pts.mean(0); cov=np.cov((pts-ctr).T)
    w,V=np.linalg.eigh(cov); w=np.clip(w[::-1],1e-9,None); V=V[:,::-1]
    lin=(w[0]-w[1])/w[0]
    return V[:,0],lin

for label,date,mouse,mask_run,runs,fs,skip in v.FOVS:
    base=PR/"scape-data"/date/mouse; rawp=v.raw_path(base,mask_run)
    tf=tifffile.TiffFile(str(rawp)); T=tf.series[0].shape[0]
    idx=np.unique(np.linspace(int(skip*fs),T-1,40).astype(int))
    try: arr=tifffile.memmap(str(rawp))
    except Exception: arr=tf.series[0].asarray()
    sp=v.fit_surface(np.asarray(arr[idx]).astype(np.float32).mean(0)); tf.close()
    n=sp['normal']
    aY,aN=[],[]
    for p in sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif")):
        pa,lin=pax(tifffile.imread(p)>0)
        if pa is None or lin<0.6: continue
        aY.append(np.degrees(np.arccos(np.clip(abs(pa@YHAT),0,1))))
        aN.append(np.degrees(np.arccos(np.clip(abs(pa@n),0,1))))
    aY=np.array(aY); aN=np.array(aN)
    better="IMAGE-Y" if np.median(aY)<np.median(aN) else "SURFACE-NORMAL"
    print(f"[{label}] surface tilt_X={sp['tilt_x']:+.1f} tilt_Z={sp['tilt_z']:+.1f} | "
          f"n_elong={len(aY)} | median angle to image-Y={np.median(aY):.1f}deg  "
          f"to surface-normal={np.median(aN):.1f}deg  -> dendrites align better with {better}")
