#!/usr/bin/env python
"""Validation A of independence: recompute mean_r, eff_dim/N, %|r|>0.3 under
(i) high-SNR masks, (ii) high-amplitude (event) frames; (iii) first vs second half (time
stability); (iv) per-event participation fraction. Summary by mouse/acquisition mode."""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
from sklearn.decomposition import PCA
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("eb",PR/"code/Extra/event_typing_behavior.py"); eb=importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
THR=eb.DFF_THR
FOVS=[("0416 r567","rbp4_132","30p-1ch","2026-04-16/rbp4_132_phpeb/run7",0),
      ("0331 r67","rbp4_132","30p-1ch","2026-03-31/rbp4_132_phpeb/run7",0),
      ("0331 r8910","rbp4_132","30p-1ch","2026-03-31/rbp4_132_phpeb/run8",0),
      ("0512 r56","rbp4_132","30p-1ch","2026-05-12/rbp4_132_phpeb/run5",0),
      ("0512 r910","rbp4_132","30p-1ch","2026-05-12/rbp4_132_phpeb/run9",0),
      ("0508 r56","rbp4_139","30p-1ch","2026-05-08/rbp4_139_phpeb/run5",0),
      ("0209","rbp4cre_136","deep-2ch","2026-02-09/rbp4cre_136_phpeb/run1",0),
      ("0217","rbp4cre_138","deep-2ch","2026-02-17/rbp4cre_138_phpeb/run7",0),
      ("0416 r1 contam","rbp4_132","30p-1ch","2026-04-16/rbp4_132_phpeb/run1",1)]

def metrics(X):
    T,N=X.shape
    if N<3 or T<5: return (np.nan,)*3
    cm=np.corrcoef(X.T); iu=np.triu_indices(N,1); mr=cm[iu].mean(); fs=np.mean(np.abs(cm[iu])>0.3)
    sd=X.std(0); k=sd>0; Xk=X[:,k]
    if Xk.shape[1]<3: return mr,np.nan,fs
    Z=(Xk-Xk.mean(0))/Xk.std(0); ev=PCA().fit(Z).explained_variance_ratio_
    return mr, (1/np.sum(ev**2))/Xk.shape[1], fs

def snr(x): m=np.median(x); mad=np.median(np.abs(x-m))*1.4826+1e-9; return (np.percentile(x,95)-m)/mad
rows=[]
for lab,mouse,mode,pth,contam in FOVS:
    df=pd.read_csv(PR/"scape-data"/pth/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df=df.drop(columns=['Frame'])
    X=df[[c for c in df.columns if 'dend' in c]].values.astype(float); T,N=X.shape
    base=metrics(X)
    S=np.array([snr(X[:,i]) for i in range(N)]); hi=S>=np.median(S); hisnr=metrics(X[:,hi])
    evfr=(X>THR).any(1); amp=metrics(X[evfr]) if evfr.sum()>10 else (np.nan,)*3
    h1=metrics(X[:T//2]); h2=metrics(X[T//2:])
    ac=(X>THR).sum(1); parts=[]
    for i in range(N):
        for p in eb.events(X[:,i]): parts.append(ac[max(0,p-1):p+2].max()/N)
    med_part=np.median(parts) if parts else np.nan
    rows.append(dict(fov=lab,mouse=mouse,mode=mode,contam=contam,N=N,
        r=round(base[0],3),eff=round(base[1],2),fs=round(base[2],2),
        r_hiSNR=round(hisnr[0],3),eff_hiSNR=round(hisnr[1],2),
        r_evfr=round(amp[0],3),eff_evfr=round(amp[1],2),
        eff_h1=round(h1[1],2),eff_h2=round(h2[1],2),r_h1=round(h1[0],3),r_h2=round(h2[0],3),
        med_participation=round(med_part,3)))
D=pd.DataFrame(rows); pd.set_option('display.width',260,'display.max_columns',40)
print("=== per-FOV (baseline | high-SNR masks | high-amplitude/event frames | time halves) ===")
print(D[['fov','mouse','N','r','eff','fs','r_hiSNR','eff_hiSNR','r_evfr','eff_evfr','eff_h1','eff_h2','med_participation','contam']].to_string(index=False))
real=D[D.contam==0]
print("\n=== summary by acquisition mode / mouse (real FOVs; descriptive, no inferential stats) ===")
print(real.groupby(['mode','mouse']).agg(nFOV=('fov','size'),mean_r=('r','mean'),eff=('eff','mean'),fs=('fs','mean'),part=('med_participation','mean')).round(3).to_string())
print(f"\nReal FOVs: baseline eff/N {real.eff.min():.2f}-{real.eff.max():.2f}; high-SNR eff/N {real.eff_hiSNR.min():.2f}-{real.eff_hiSNR.max():.2f}; "
      f"event-frame eff/N {real.eff_evfr.min():.2f}-{real.eff_evfr.max():.2f}; median participation {real.med_participation.min():.2f}-{real.med_participation.max():.2f}")
print(f"contaminated: eff/N {D[D.contam==1].eff.values[0]:.2f}, hiSNR {D[D.contam==1].eff_hiSNR.values[0]:.2f}, event-frame {D[D.contam==1].eff_evfr.values[0]:.2f}, participation {D[D.contam==1].med_participation.values[0]:.2f}")
D.to_csv(PR/"scape-data"/"validation_independence.csv",index=False)
print("saved scape-data/validation_independence.csv")
