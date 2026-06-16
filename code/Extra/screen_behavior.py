#!/usr/bin/env python
"""Screen all shared-mask-group FOVs for movement-coupling viability (pooled runs).
Reports global-Ca~accel r/p, movement onsets, frac_global, raw vs residual coupled %."""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
from scipy import stats

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS = eb.FS; W = 15; CUT = 0.17
FOVS = [("0508_r56","2026-05-08","rbp4_139_phpeb",["run5","run6"]),
        ("0512_r56","2026-05-12","rbp4_132_phpeb",["run5","run6"]),
        ("0512_r910","2026-05-12","rbp4_132_phpeb",["run9","run10"]),
        ("0416_r567","2026-04-16","rbp4_132_phpeb",["run5","run6","run7"]),
        ("0331_r8910","2026-03-31","rbp4_132_phpeb",["run8","run9","run10"]),
        ("0331_r67","2026-03-31","rbp4_132_phpeb",["run6","run7"])]

def pool(date, mouse, runs):
    Xs, accs, bnd, off, names = [], [], [], 0, None
    for rn in runs:
        b = PR/"scape-data"/date/mouse/rn
        f = b/"traces"/"dff_traces_curated_bgsub.csv"
        if not f.exists(): continue
        df = pd.read_csv(f)
        if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
        nm = [c for c in df.columns if 'dend' in c]
        if names is None: names = nm
        if not all(n in df.columns for n in names): continue
        X = df[names].values.astype(float); a,_ = eb.load_behavior(b, rn, len(X))
        Xs.append(X); accs.append(a if a is not None else np.full(len(X),np.nan)); off+=len(X); bnd.append(off)
    return np.vstack(Xs), np.concatenate(accs), bnd[:-1]

def onsets(sig, T, bnd):
    s=(sig-np.nanmean(sig))/(np.nanstd(sig)+1e-9); rise=np.where(np.diff((s>0.5).astype(int))==1)[0]+1
    out=[]
    for r in rise:
        if W<=r<T-W and not any(abs(r-b)<W for b in bnd) and (not out or r-out[-1]>=10): out.append(r)
    return out

rows=[]
for lab,d,mo,runs in FOVS:
    X, accel, bnd = pool(d, mo, runs); T,N = X.shape
    g = np.median(X,1); beta=(X*g[:,None]).mean(0)/(g.var()+1e-9); resid=X-np.outer(g,beta)
    nmov = len(onsets(accel, T, bnd))
    o = np.corrcoef(g,accel)[0,1]
    null=[np.corrcoef(np.roll(g,k),accel)[0,1] for k in np.random.randint(W,T-W,500)]
    p=(np.sum(np.abs(null)>=abs(o))+1)/501
    ac=(X>eb.DFF_THR).sum(1); fr=[]
    for i in range(N):
        for pk in eb.events(X[:,i]): fr.append(ac[max(0,pk-1):pk+2].max()/N)
    fg = np.mean(np.array(fr)>=CUT) if fr else np.nan
    def cf(M):
        ks=np.random.randint(W,T-W,300); s=0
        for i in range(N):
            ob=abs(np.corrcoef(M[:,i],accel)[0,1]); th=np.percentile([abs(np.corrcoef(np.roll(M[:,i],k),accel)[0,1]) for k in ks],95)
            s+=ob>th
        return 100*s/N
    rows.append(dict(fov=lab,N=N,T=T,mov_onsets=nmov,frac_global=round(fg,2),
                     gca_accel_r=round(o,2),p=round(p,3),raw_cpl=round(cf(X)),res_cpl=round(cf(resid))))
S=pd.DataFrame(rows); pd.set_option('display.width',200)
print(S.to_string(index=False))
S.to_csv(PR/"scape-data"/"behavior_screen.csv",index=False)
qual=S[(S.p<0.05)&(S.mov_onsets>=15)]
print("\nQUALIFYING FOVs (global~accel p<0.05 & >=15 onsets):", list(qual.fov))
