#!/usr/bin/env python
"""Decisive test for 'silent' masks: re-extract their whole-mask CORE trace WITH vs WITHOUT
shell subtraction from the raw stack. If a mask is flat with shell but active without it (and
its shell is itself active), the silence is core-shell / neuropil cancellation (shared signal),
not true inactivity."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, gc, importlib.util
PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("v", PR/"code/Testing/nmda_bap_split_v2.py")
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
from scipy.ndimage import binary_erosion, binary_dilation
B1,B2,B3 = v.B1,v.B2,v.B3

seg = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_segments.csv")
ev  = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_events.csv")
nseg = seg.groupby(["fov","name"]).size().rename("nseg")
nev  = ev.groupby(["fov","name"]).size().rename("nev")
tab = pd.concat([nseg,nev],axis=1).reset_index(); tab["nev"]=tab.nev.fillna(0)
silent = {f: tab[(tab.fov==f)&(tab.nseg>=2)&(tab.nev==0)].name.tolist() for f in tab.fov.unique()}

# FOV -> (date,mouse,mask_run,fs,skip) using v.FOVS
CFG = {c[0]:(c[1],c[2],c[3],c[5],c[6]) for c in v.FOVS}
TESTFOVS = ["0416_r123","0512_r56"]   # remaining FOVs (others already verified)

for fov in TESTFOVS:
    names = silent.get(fov,[])
    if not names: continue
    date,mouse,mask_run,fs,skip = CFG[fov]
    base = PR/"scape-data"/date/mouse; rawp = v.raw_path(base,mask_run)
    tf=tifffile.TiffFile(str(rawp)); T,Z,Y,X=tf.series[0].shape
    try: arr=tifffile.memmap(str(rawp))
    except Exception: arr=tf.series[0].asarray()
    fidx=np.unique(np.linspace(int(skip*fs),T-1,v.NF0).astype(int))
    f0=np.percentile(np.asarray(arr[fidx]).astype(np.float32),10,axis=0).ravel(); f0e=f0+1e-6
    regs={}
    for nm in names:
        m=tifffile.imread(base/mask_run/"labelmaps_curated_dynamic"/f"{nm}_labelmap.tif")>0
        if m.shape[1]<Y: m=np.pad(m,((0,0),(0,Y-m.shape[1]),(0,0)))
        elif m.shape[1]>Y: m=m[:,:-(m.shape[1]-Y),:]
        core=binary_erosion(m,B1); core=core if core.any() else m
        shell=binary_dilation(m,B3)&~binary_dilation(m,B2)
        regs[nm]=(v.gidx(core), v.gidx(shell))
    core_tr={nm:np.empty(T,np.float32) for nm in names}
    shell_tr={nm:np.empty(T,np.float32) for nm in names}
    for t in range(T):
        dff=(np.asarray(arr[t]).astype(np.float32).ravel()-f0)/f0e
        for nm,(ci,si) in regs.items():
            core_tr[nm][t]=dff[ci].mean(); shell_tr[nm][t]=dff[si].mean() if si.size else 0.0
    tf.close()
    print(f"\n[{fov}] silent masks: core-only vs core-shell vs shell  (peak% / n_events)")
    rows=[]
    for nm in names:
        c=core_tr[nm]*100; s=shell_tr[nm]*100; cs=c-s
        def pk_ne(x):
            th=v.thresh(x); return float(x.max()), len(v.event_peaks(x,th))
        pc,nc=pk_ne(c); pcs,ncs=pk_ne(cs); ps,_=pk_ne(s)
        corr=float(np.corrcoef(c,s)[0,1])
        rows.append((nm,pc,nc,pcs,ncs,ps,corr))
    df=pd.DataFrame(rows,columns=["name","core_peak","core_nev","coreshell_peak","coreshell_nev","shell_peak","corr_core_shell"])
    print(df.to_string(index=False,float_format=lambda x:f"{x:.1f}"))
    rescued=int(((df.core_nev>=1)&(df.coreshell_nev==0)).sum())
    print(f"  -> {rescued}/{len(df)} silent masks show >=1 event WITHOUT shell; "
          f"median core-shell corr={df.corr_core_shell.median():.2f} "
          f"(high corr => shell cancels core = shared/neuropil signal)")
    del core_tr,shell_tr; gc.collect()
