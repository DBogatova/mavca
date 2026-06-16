#!/usr/bin/env python
"""
(1) Coupling control for Test 1: does the SAME mask fire bigger during whole-tree events
    than during its own isolated events? (within-mask paired -> removes per-mask brightness
    confound and the cross-mask amplitude heterogeneity).
(2) Other features that may distinguish isolated/local (NMDA-like) vs whole-tree (bAP-like)
    events, independent of the amplitude<->participation coupling: event DURATION (FWHM),
    DECAY time, AREA (integral). NMDA plateau -> expect longer; bAP -> sharper.

Uses the curated pipeline ΔF/F traces (small CSVs; high-SNR whole-mask core-shell). FAST.
"""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec=importlib.util.spec_from_file_location("amp",PR/"code/Testing/nmda_bap_amplitude.py")
A=importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
CUT=0.17  # participation >= CUT = whole-tree (per project)

def kinetics(tr,p,med,fs):
    amp=tr[p]-med
    if amp<=0: return None
    half=med+0.5*amp
    i=p
    while i>0 and tr[i]>half: i-=1
    j=p
    while j<len(tr)-1 and tr[j]>half: j+=1
    a=p
    while a>0 and tr[a]>med: a-=1
    b=p
    while b<len(tr)-1 and tr[b]>med: b+=1
    return dict(amp=float(tr[p]), fwhm_s=(j-i)/fs, decay_s=(j-p)/fs,
                rise_s=(p-i)/fs, area=float(np.sum(tr[a:b+1]-med)))

def collect():
    rows=[]
    for label,d,mo,mr,runs,fs in A.FOVS:
        base=PR/"scape-data"/d/mo
        mpaths=sorted((base/mr/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))
        names=[p.stem.replace("_labelmap","") for p in mpaths]
        dfs=[df for df in (A.load_traces(base,r,names) for r in runs) if df is not None]
        if not dfs: continue
        N=len(names)
        for df in dfs:
            thr={nm:A.thresh(df[nm].values) for nm in names}
            act=np.column_stack([df[nm].values>thr[nm] for nm in names])
            for i,nm in enumerate(names):
                tr=df[nm].values; med=np.median(tr)
                for p in A.event_peaks(tr,thr[nm]):
                    k=kinetics(tr,p,med,fs)
                    if k is None: continue
                    lo,hi=max(0,p-1),min(len(tr),p+2)
                    part=float(act[lo:hi].any(0).sum())/N
                    rows.append(dict(fov=label,name=nm,fs=fs,participation=part,
                                     n_coactive=int(round(part*N)),**k))
    return pd.DataFrame(rows)

def main():
    E=collect()
    E.to_csv(PR/"code"/"Testing"/"data"/"nmda_bap_features.csv",index=False)
    # which FOVs have whole-tree events
    fov_has=E.groupby("fov").apply(lambda g:(g.participation>=CUT).sum(),include_groups=False)
    use=[f for f in fov_has.index if fov_has[f]>=8]
    print(f"FOVs with >=8 whole-tree events: {use}\n")
    E=E[E.fov.isin(use)].copy()

    print("==== (1) COUPLING CONTROL: is a mask bigger during whole-tree vs its own isolated events? ====")
    # within-mask demeaned amplitude (subtract each mask's mean amp) -> controls mask identity, uses ALL events
    E["amp_dm"]=E.amp - E.groupby(["fov","name"]).amp.transform("mean")
    zi=E.loc[E.n_coactive<=1,"amp_dm"]; zw=E.loc[E.participation>=CUT,"amp_dm"]
    if len(zi)>3 and len(zw)>3:
        u,p=stats.mannwhitneyu(zw,zi,alternative="greater")
        print(f"  within-mask demeaned amp: whole-tree median={zw.median():+.2f}%  isolated median={zi.median():+.2f}%"
              f"  (n_iso={len(zi)} n_wt={len(zw)})  MWU p={p:.3g}")
    # paired (>=1 each) for sign count
    pairs=[]
    for (f,nm),g in E.groupby(["fov","name"]):
        iso=g[g.n_coactive<=1].amp; wt=g[g.participation>=CUT].amp
        if len(iso)>=1 and len(wt)>=1: pairs.append((f,nm,iso.median(),wt.median()))
    pr=pd.DataFrame(pairs,columns=["fov","name","iso","wt"])
    if len(pr)>=5:
        w,p=stats.wilcoxon(pr.wt,pr.iso,alternative="greater")
        print(f"  paired masks (>=1 each): n={len(pr)}; whole-tree>isolated in {(pr.wt>pr.iso).sum()}/{len(pr)}"
              f"  median(wt-iso)={np.median(pr.wt-pr.iso):+.2f}%  Wilcoxon p={p:.3g}")
        for f,g in pr.groupby("fov"):
            if len(g)>=4:
                ww,pp=stats.wilcoxon(g.wt,g.iso,alternative="greater")
                print(f"    [{f}] n={len(g)} wt>iso {(g.wt>g.iso).sum()}/{len(g)} "
                      f"med diff={np.median(g.wt-g.iso):+.2f}% p={pp:.2g}")

    print("\n==== (2) FEATURES isolated (NMDA-like) vs whole-tree (bAP-like) [within-FOV z] ====")
    iso=E[E.n_coactive<=1]; wt=E[E.participation>=CUT]
    print(f"  n isolated={len(iso)}  n whole-tree={len(wt)}")
    for feat in ["amp","fwhm_s","decay_s","rise_s","area"]:
        E["z"]=E.groupby("fov")[feat].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
        zi=E.loc[E.n_coactive<=1,"z"]; zw=E.loc[E.participation>=CUT,"z"]
        u,p=stats.mannwhitneyu(zw,zi)  # two-sided
        print(f"  {feat:8s}: whole-tree z={zw.median():+.2f}  isolated z={zi.median():+.2f}  "
              f"(raw med wt={wt[feat].median():.2f} iso={iso[feat].median():.2f})  MWU p={p:.2g}")
    # also: continuous corr of each feature with participation (within FOV, Spearman median)
    print("\n  per-FOV Spearman(feature, participation): median over FOVs")
    for feat in ["amp","fwhm_s","decay_s","area"]:
        rs=[stats.spearmanr(g.participation,g[feat])[0] for _,g in E.groupby("fov") if len(g)>10]
        rs=[r for r in rs if np.isfinite(r)]
        print(f"    {feat:8s}: median rho={np.median(rs):+.2f}  per-FOV={[round(r,2) for r in rs]}")
    make_fig(E,pr)

def make_fig(E,pr):
    fig,ax=plt.subplots(1,4,figsize=(18,4.2))
    feats=[("amp","peak ΔF/F %"),("fwhm_s","FWHM (s)"),("decay_s","decay to half (s)"),("area","area")]
    for a,(feat,lab) in zip(ax[:4],feats):
        E["z"]=E.groupby("fov")[feat].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
        zi=E.loc[E.n_coactive<=1,"z"].dropna(); zw=E.loc[E.participation>=CUT,"z"].dropna()
        a.boxplot([zi,zw],tick_labels=[f"isolated\n(NMDA)\nn={len(zi)}",f"whole-tree\n(bAP)\nn={len(zw)}"],
                  showfliers=False); a.axhline(0,color='k',lw=.6)
        a.set_title(lab); a.set_ylabel("within-FOV z"); a.grid(alpha=.3)
    fig.suptitle("Isolated (NMDA-like) vs whole-tree (bAP-like) events: amplitude + kinetics",fontsize=12)
    fig.tight_layout(); fig.savefig(PR/"code"/"Testing"/"figures"/"nmda_bap_features.png",dpi=150); plt.close(fig)
    print("\nsaved figures/nmda_bap_features.png, scape-data/nmda_bap_features.csv")

if __name__=="__main__": main()
