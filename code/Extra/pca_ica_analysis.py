#!/usr/bin/env python
"""
PCA/ICA on per-mask dF/F traces.

PCA  : effective dimensionality (participation ratio PR = 1/sum(var_ratio^2)); PR/N near 0
       = global/whole-tree (one shared mode), near 1 = independent. Plus PC1 variance frac,
       PC1 timecourse vs accel/pupil (is the dominant mode arousal-driven or contamination?),
       and PC1 loading vs cortical depth.
ICA  : FastICA assemblies — assign each mask to its strongest component; report assembly
       count/sizes and spatial compactness (centroid spread) => co-active groups / nexus.
"""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util, tifffile
from sklearn.decomposition import PCA, FastICA
from scipy import stats

PR_ROOT = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR_ROOT/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS = eb.FS

def _corr(a, b):
    if b is None: return np.nan
    m = np.isfinite(a) & np.isfinite(b)
    return round(stats.pearsonr(a[m], b[m])[0], 2) if m.sum() > 4 and np.ptp(a[m]) > 0 else np.nan

def analyze(date, mouse, run):
    base = PR_ROOT/"scape-data"/date/mouse/run
    df = pd.read_csv(base/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
    names = [c for c in df.columns if 'dend' in c]
    X = df[names].values.astype(float)          # (T,N)
    T, N = X.shape
    sd = X.std(0); sd[sd == 0] = 1
    Z = (X - X.mean(0)) / sd                     # z-score per mask
    pca = PCA().fit(Z); ev = pca.explained_variance_ratio_
    pr = 1.0 / np.sum(ev**2)                     # participation ratio (effective dims)
    scores = pca.transform(Z); pc1 = scores[:, 0]
    load1 = np.abs(pca.components_[0])
    accel, pupil = eb.load_behavior(base, run, T)
    # depth per mask
    mp = {p.stem.replace("_labelmap",""): p for p in (base/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif")}
    depth = np.array([eb.morphology(tifffile.imread(mp[n])>0)['depth_um'] if n in mp else np.nan for n in names])
    # ICA assemblies
    k = int(np.clip(np.searchsorted(np.cumsum(ev), 0.8)+1, 2, 8)); k = min(k, N-1)
    n_asm = sizes = compact = np.nan
    try:
        ica = FastICA(n_components=k, whiten='unit-variance', max_iter=1000, random_state=0).fit(Z)
        A = np.abs(ica.mixing_)                  # (N,k)
        asm = A.argmax(1)
        cents = np.array([[eb.morphology(tifffile.imread(mp[n])>0)['cx'],
                           depth[i]] for i, n in enumerate(names) if n in mp])
        asm_v = np.array([asm[i] for i, n in enumerate(names) if n in mp])
        sz = np.bincount(asm_v, minlength=k); n_asm = int((sz >= 3).sum())
        sizes = sorted(sz[sz >= 3].tolist(), reverse=True)
        # spatial compactness: mean within-assembly centroid spread (um), lower=clustered
        sp = [cents[asm_v == c].std(0).mean() for c in range(k) if (asm_v == c).sum() >= 3]
        compact = round(float(np.mean(sp)), 1) if sp else np.nan
    except Exception as e:
        print("   ICA:", e)
    return dict(run=f"{date}/{run}", N=N, PC1_frac=round(float(ev[0]), 3),
                eff_dim=round(float(pr), 1), eff_dim_frac=round(float(pr/N), 3),
                r_PC1_accel=_corr(pc1, accel), r_PC1_pupil=_corr(pc1, pupil),
                r_PC1load_depth=_corr(load1, depth), n_assemblies=n_asm,
                asm_sizes=sizes, asm_spread_um=compact)

if __name__ == "__main__":
    rows = [analyze(d, mo, rn) for d, mo, rn in eb.RUNS]
    S = pd.DataFrame(rows)
    pd.set_option('display.width', 220, 'display.max_columns', 30)
    print(S.to_string(index=False))
    S.to_csv(PR_ROOT/"scape-data"/"pca_ica_summary.csv", index=False)
