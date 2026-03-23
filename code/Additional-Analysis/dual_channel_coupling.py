#!/usr/bin/env python3
"""
Dual-Channel Ca-ACh Coupling Analysis  (with PC0 removal)

Preprocessing: ΔF/F with fast gaussian baseline, then optional low/high split.
When --remove-pc0 is set, runs all analyses twice: once on raw ΔF/F, once after
subtracting PC0 from each channel.  Separate plots/tifs saved with _nopc0 suffix.

Blocks:
  1. Correlation — Ca voxel vs global ACh + same-voxel Ca vs ACh
  2. Regression  — partial R² (ACh unique after controlling for global Ca)
  3. PCA         — Ca/ACh spatial maps + cross-correlation matrix
  + Controls    — self-corr, bleedthrough, inside/outside dendrite, permutation null

Usage:
    python dual_channel_coupling.py
    python dual_channel_coupling.py --remove-pc0
    python dual_channel_coupling.py --only corr
    python dual_channel_coupling.py --only regr
    python dual_channel_coupling.py --only pca
    python dual_channel_coupling.py --view
    python dual_channel_coupling.py --y-crop 3 --skip-first-seconds 15 --remove-pc0
"""

import argparse, gc, sys, time
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm

# ================== CONFIG ==================
DATE = "2026-02-24"
MOUSE = "rAi162_42_phpeb"
RUN = "run5"
FS_HZ = 5.0
VOXEL_SIZE = (3.9, 1.0, 1.2)  # Z, Y, X µm

BASE = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/"
            "apical-dendrites-2025/scape-data") / DATE / MOUSE / RUN
RAW_CA  = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin-green.tif"
RAW_ACH = BASE / "raw" / f"runA_{RUN}_{MOUSE}-reslice-bin-red.tif"
MASK_FOLDER = BASE / "labelmaps_curated_dynamic"

BASELINE_SIGMA = 150       # frames for gaussian baseline (~30 s)

SPATIAL_SIGMA = (0, 0.5, 1.0, 1.0)

MAX_LAG_SEC = 1.0
SLAB_Z = 4
N_PCS = 20
PCA_DS = 2



# ================== CLI ==================
def parse_args():
    p = argparse.ArgumentParser(description="Ca–ACh coupling (4-block)")
    p.add_argument("--only", choices=["corr","regr","pca"], default=None)
    p.add_argument("--view", action="store_true")
    p.add_argument("--y-crop", type=int, default=0)
    p.add_argument("--skip-first-seconds", type=float, default=7.0)
    p.add_argument("--crop-last-seconds", type=float, default=0.0,
                   help="Remove last N seconds from recording")
    p.add_argument("--spatial-smooth", action="store_true")
    p.add_argument("--remove-pc0", action="store_true",
                   help="Run analyses twice: with and without PC0")
    p.add_argument("--ica", action="store_true",
                   help="Run ICA on PCA scores (PCA → FastICA)")
    p.add_argument("--n-shuffles", type=int, default=1)
    return p.parse_args()

# ================== I/O ==================
def load_4d(path):
    print(f"  Loading {path.name} ...")
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim != 4: raise ValueError(f"Expected 4D, got {arr.shape}")
    return arr

def load_dendrite_mask(shape_zyx):
    if not MASK_FOLDER.exists(): return None
    paths = sorted(MASK_FOLDER.glob("dend_*_labelmap.tif"))
    if not paths: return None
    union = np.zeros(shape_zyx, dtype=bool)
    for p in paths:
        m = tifffile.imread(p) > 0
        if m.shape == shape_zyx: union |= m
    print(f"  Dendrite mask: {len(paths)} files, {union.sum()} voxels")
    return union

# ================== PREPROCESSING ==================
def skip_crop_align(ca, ach, skip_sec, y_crop, crop_last_sec=0.0):
    if ca.shape != ach.shape:
        T = min(ca.shape[0], ach.shape[0])
        ca, ach = ca[:T], ach[:T]
    skip = int(skip_sec * FS_HZ)
    if skip > 0:
        ca, ach = ca[skip:], ach[skip:]
        print(f"  Skipped first {skip_sec}s ({skip} frames)")
    if crop_last_sec > 0:
        drop = int(crop_last_sec * FS_HZ)
        if drop > 0 and drop < ca.shape[0]:
            ca, ach = ca[:-drop], ach[:-drop]
            print(f"  Cropped last {crop_last_sec}s ({drop} frames) → T={ca.shape[0]}")
    if y_crop > 0:
        ca, ach = ca[:,:,:-y_crop,:], ach[:,:,:-y_crop,:]
        print(f"  Y-cropped bottom {y_crop} → Y={ca.shape[2]}")
    return ca, ach

def detrend_dff(stack, label=""):
    """ΔF/F with gaussian baseline, computed slab-by-slab to save memory."""
    T, Z, Y, X = stack.shape
    out = np.empty_like(stack)
    for z0, z1 in tqdm(list(_slab_ranges(Z, SLAB_Z)), desc=f"  ΔF/F {label}"):
        slab = stack[:, z0:z1]
        baseline = gaussian_filter1d(slab, sigma=BASELINE_SIGMA, axis=0)
        out[:, z0:z1] = ((slab - baseline) / (baseline + 1e-6)).astype(np.float32)
        del baseline
    return out




def preprocess(ca_raw, ach_raw, args):
    """Returns ca_dff, ach_dff. Overwrites raw arrays in-place to save memory."""
    print("Preprocessing...")
    if args.spatial_smooth:
        print("  Spatial smooth (slab-wise)...")
        T, Z, Y, X = ca_raw.shape
        for z0, z1 in _slab_ranges(Z, SLAB_Z):
            ca_raw[:, z0:z1] = gaussian_filter(ca_raw[:, z0:z1], sigma=SPATIAL_SIGMA)
            ach_raw[:, z0:z1] = gaussian_filter(ach_raw[:, z0:z1], sigma=SPATIAL_SIGMA)
    print("  Ca ΔF/F (slab-wise)...")
    ca_dff = detrend_dff(ca_raw, "Ca")
    del ca_raw; gc.collect()
    print("  ACh ΔF/F (slab-wise)...")
    ach_dff = detrend_dff(ach_raw, "ACh")
    del ach_raw; gc.collect()
    return ca_dff, ach_dff

def remove_pc0(stack, label=""):
    """Subtract PC0 from (T,Z,Y,X) stack in-place. Returns modified stack."""
    from sklearn.decomposition import PCA
    T, Z, Y, X = stack.shape
    flat_ds = stack[:, ::2, ::2, ::2].reshape(T, -1)
    pca = PCA(n_components=1)
    tc = pca.fit_transform(flat_ds)[:, 0]
    tc_z = (tc - tc.mean()) / (tc.std() + 1e-12)
    print(f"  PC0 {label}: {pca.explained_variance_ratio_[0]:.1%} variance")
    for z0, z1 in _slab_ranges(Z, SLAB_Z):
        nz = z1 - z0
        flat = stack[:, z0:z1].reshape(T, nz * Y * X)
        flat_c = flat - flat.mean(axis=0, keepdims=True)
        loading = (tc_z[:, None] * flat_c).mean(axis=0)
        flat -= tc_z[:, None] * loading[None, :]
        stack[:, z0:z1] = flat.reshape(T, nz, Y, X)
    return stack


# ================== HELPERS ==================
def _slab_ranges(Z, sz):
    for z0 in range(0, Z, sz):
        yield z0, min(z0 + sz, Z)

def _zscore_1d(x):
    return (x - x.mean()) / (x.std() + 1e-12)

def _zscore_cols(flat):
    m = flat.mean(axis=0, keepdims=True)
    s = flat.std(axis=0, keepdims=True) + 1e-12
    return (flat - m) / s

def _corr_at_lag(ca_z, ach_z, lag):
    T = ca_z.shape[0]
    if lag < 0:   a, c = ach_z[:T+lag], ca_z[-lag:]
    elif lag > 0: a, c = ach_z[lag:],   ca_z[:T-lag]
    else:         a, c = ach_z, ca_z
    if a.ndim == 1: return (a[:, None] * c).mean(axis=0)
    return (a * c).mean(axis=0)

# ================== QC: GLOBAL TRACES ==================
def plot_global_traces(ca, ach, out):
    ca_g = ca.mean(axis=(1,2,3)); ach_g = ach.mean(axis=(1,2,3))
    ca_z = _zscore_1d(ca_g); ach_z = _zscore_1d(ach_g)
    r = np.corrcoef(ca_z, ach_z)[0,1]
    np.savez(out / "global_traces.npz", ca=ca_g, ach=ach_g, ca_z=ca_z, ach_z=ach_z)

    t = np.arange(len(ca_g)) / FS_HZ
    max_lag = int(MAX_LAG_SEC * FS_HZ)
    lags = np.arange(-max_lag, max_lag+1)
    cc = np.array([_corr_at_lag(ca_z[:,None], ach_z, l)[0] for l in lags])

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=False)
    axes[0].plot(t, ca_z, color="green", lw=0.6, label="Ca")
    axes[0].plot(t, ach_z, color="red", lw=0.6, alpha=0.7, label="ACh")
    axes[0].legend(); axes[0].set_ylabel("z-score")
    axes[0].set_title(f"Global traces (r={r:.3f}) | {DATE} {MOUSE} {RUN}")
    axes[1].plot(lags/FS_HZ, cc, color="purple", lw=1)
    axes[1].axhline(0, color="k", ls="--", lw=0.5)
    axes[1].set_xlabel("Lag (s)"); axes[1].set_ylabel("r")
    fig.tight_layout(); fig.savefig(out / "qc_global_traces.png", dpi=200); plt.close(fig)
    print(f"  Global corr: {r:.4f}")


# ================== BLOCK 1: CORRELATION ==================
def run_correlation(ca, ach, out):
    """Global-ACh + same-voxel correlation (full-band only)."""
    print("\n=== BLOCK 1: Correlation ===")
    T, Z, Y, X = ca.shape
    max_lag = int(MAX_LAG_SEC * FS_HZ)
    results = {}

    ach_g_z = _zscore_1d(ach.mean(axis=(1,2,3)))

    # Global-ACh correlation
    r0 = np.empty((Z,Y,X), np.float32)
    rmax = np.empty((Z,Y,X), np.float32)
    lag_vol = np.zeros((Z,Y,X), np.int16)
    for z0,z1 in tqdm(list(_slab_ranges(Z,SLAB_Z)), desc="  corr-global"):
        nz = z1-z0; flat = _zscore_cols(ca[:,z0:z1].reshape(T, nz*Y*X))
        r0_s = _corr_at_lag(flat, ach_g_z, 0)
        best_r, best_l = r0_s.copy(), np.zeros(flat.shape[1], np.int16)
        for lag in range(-max_lag, max_lag+1):
            if lag == 0: continue
            r = _corr_at_lag(flat, ach_g_z, lag)
            better = np.abs(r) > np.abs(best_r)
            best_r[better] = r[better]; best_l[better] = lag
        r0[z0:z1] = r0_s.reshape(nz,Y,X)
        rmax[z0:z1] = best_r.reshape(nz,Y,X)
        lag_vol[z0:z1] = best_l.reshape(nz,Y,X)
    results["r0_globalach"] = r0
    results["rmax_globalach"] = rmax
    results["lag_at_rmax_globalach"] = lag_vol

    # Same-voxel correlation
    r0v = np.empty((Z,Y,X), np.float32)
    rmaxv = np.empty((Z,Y,X), np.float32)
    lagv = np.zeros((Z,Y,X), np.int16)
    for z0,z1 in tqdm(list(_slab_ranges(Z,SLAB_Z)), desc="  corr-voxel"):
        nz = z1-z0
        cf = _zscore_cols(ca[:,z0:z1].reshape(T, nz*Y*X))
        af = _zscore_cols(ach[:,z0:z1].reshape(T, nz*Y*X))
        r0_s = _corr_at_lag(cf, af, 0)
        best_r, best_l = r0_s.copy(), np.zeros(cf.shape[1], np.int16)
        for lag in range(-max_lag, max_lag+1):
            if lag == 0: continue
            r = _corr_at_lag(cf, af, lag)
            better = np.abs(r) > np.abs(best_r)
            best_r[better] = r[better]; best_l[better] = lag
        r0v[z0:z1] = r0_s.reshape(nz,Y,X)
        rmaxv[z0:z1] = best_r.reshape(nz,Y,X)
        lagv[z0:z1] = best_l.reshape(nz,Y,X)
    results["r0_vox"] = r0v
    results["rmax_vox"] = rmaxv
    results["lag_vox"] = lagv

    print(f"  global rmax mean|r|={np.abs(rmax).mean():.4f}  "
          f"voxel rmax mean|r|={np.abs(rmaxv).mean():.4f}")

    # Save tifs
    for k, v in results.items():
        tifffile.imwrite(out / f"corr_{k}.tif", v)

    # MIP plots
    for kind, r0k, rmk, lk in [
        ("global", "r0_globalach", "rmax_globalach", "lag_at_rmax_globalach"),
        ("voxel",  "r0_vox", "rmax_vox", "lag_vox")]:
        rm = results[rmk]; lg = results[lk]
        fig, axes = plt.subplots(1,3, figsize=(15,4))
        axes[0].imshow(results[r0k].max(0), cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
        axes[0].set_title(f"r₀ {kind}")
        axes[1].imshow(rm.max(0), cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
        axes[1].set_title(f"rmax {kind}")
        bz = np.argmax(np.abs(rm), axis=0)
        lm = np.take_along_axis(lg, bz[None], 0)[0].astype(np.float32)/FS_HZ
        axes[2].imshow(lm, cmap="coolwarm", vmin=-MAX_LAG_SEC, vmax=MAX_LAG_SEC, aspect="auto")
        axes[2].set_title(f"lag {kind}")
        for ax in axes: ax.axis("off")
        fig.suptitle(f"{kind.title()} Corr | {DATE} {MOUSE} {RUN}")
        fig.tight_layout(); fig.savefig(out / f"corr_{kind}.png", dpi=200); plt.close(fig)

    # Histograms
    fig, axes = plt.subplots(1,2, figsize=(10,3.5))
    axes[0].hist(rmax.ravel(), bins=100, range=(-.8,.8), color="steelblue", alpha=.7)
    axes[0].set_title("global rmax")
    axes[1].hist(rmaxv.ravel(), bins=100, range=(-.8,.8), color="coral", alpha=.7)
    axes[1].set_title("voxel rmax")
    fig.tight_layout(); fig.savefig(out / "corr_histograms.png", dpi=200); plt.close(fig)
    return results


# ================== BLOCK 2: REGRESSION ==================
def run_regression(ca, ach, out):
    """Partial R²: ACh unique after controlling for global Ca.
    Also computes residual coupling: corr(Ca_resid|Ca_global, ACh_global)."""
    print("\n=== BLOCK 2: Regression ===")
    T, Z, Y, X = ca.shape
    ach_g_z = _zscore_1d(ach.mean(axis=(1,2,3)))
    ca_g_z  = _zscore_1d(ca.mean(axis=(1,2,3)))

    X_des = np.column_stack([ach_g_z, ca_g_z])
    XtX_inv_Xt = np.linalg.pinv(X_des.T @ X_des) @ X_des.T

    beta_ach = np.empty((Z,Y,X), np.float32)
    r2_multi = np.empty((Z,Y,X), np.float32)
    r2_caonly = np.empty((Z,Y,X), np.float32)
    r2_unique = np.empty((Z,Y,X), np.float32)
    r_resid   = np.empty((Z,Y,X), np.float32)

    for z0,z1 in tqdm(list(_slab_ranges(Z,SLAB_Z)), desc="  regression"):
        nz = z1-z0
        flat = ca[:,z0:z1].reshape(T, nz*Y*X)
        ca_c = flat - flat.mean(axis=0, keepdims=True)
        ss_tot = (ca_c**2).sum(0) + 1e-12

        betas = XtX_inv_Xt @ ca_c
        pred = X_des @ betas
        r2_m = np.clip(1 - ((ca_c-pred)**2).sum(0)/ss_tot, 0, 1)

        b_ca = (ca_g_z[:,None] * ca_c).mean(0)
        pred_ca = ca_g_z[:,None] * b_ca[None,:]
        r2_c = np.clip(1 - ((ca_c-pred_ca)**2).sum(0)/ss_tot, 0, 1)

        resid = ca_c - pred_ca
        resid_z = _zscore_cols(resid)
        r_res = (ach_g_z[:,None] * resid_z).mean(0)

        beta_ach[z0:z1] = betas[0].reshape(nz,Y,X)
        r2_multi[z0:z1] = r2_m.reshape(nz,Y,X)
        r2_caonly[z0:z1] = r2_c.reshape(nz,Y,X)
        r2_unique[z0:z1] = np.clip(r2_m - r2_c, 0, 1).reshape(nz,Y,X)
        r_resid[z0:z1] = r_res.reshape(nz,Y,X)

    for name, vol in [("beta_ach",beta_ach),("r2_multi",r2_multi),
                       ("r2_caonly",r2_caonly),("r2_ach_unique",r2_unique),
                       ("r_resid_ach",r_resid)]:
        tifffile.imwrite(out / f"regr_{name}.tif", vol)

    print(f"  R²_multi:      mean={r2_multi.mean():.4f}")
    print(f"  R²_ach_unique: mean={r2_unique.mean():.5f}  max={r2_unique.max():.4f}  "
          f">0.01:{(r2_unique>0.01).sum()}  >0.05:{(r2_unique>0.05).sum()}")
    print(f"  r_resid:       mean={r_resid.mean():.4f}")

    fig, axes = plt.subplots(1,5, figsize=(25,4))
    for ax,(vol,t,cm,lo,hi) in zip(axes, [
        (beta_ach,"β_ACh","RdBu_r",-.3,.3), (r2_multi,"R²_multi","hot",0,.3),
        (r2_caonly,"R²_caonly","hot",0,.3), (r2_unique,"R²_ACh_unique","hot",0,.15),
        (r_resid,"r(resid,ACh)","RdBu_r",-.3,.3)]):
        im = ax.imshow(vol.max(0), cmap=cm, vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(t+" Z-MIP"); ax.axis("off"); plt.colorbar(im, ax=ax)
    fig.suptitle(f"Regression | {DATE} {MOUSE} {RUN}")
    fig.tight_layout(); fig.savefig(out / "regr_mip.png", dpi=200); plt.close(fig)

    return {"beta_ach":beta_ach, "r2_multi":r2_multi, "r2_caonly":r2_caonly,
            "r2_ach_unique":r2_unique, "r_resid_ach":r_resid}


# ================== BLOCK 3: PCA ==================
def run_pca(ca, ach, out):
    from sklearn.decomposition import PCA
    print("\n=== BLOCK 3: PCA ===")
    T = ca.shape[0]; ds = PCA_DS
    ca_ds = ca[:,::ds,::ds,::ds]; ach_ds = ach[:,::ds,::ds,::ds]
    _, Zd, Yd, Xd = ca_ds.shape

    ca_flat = ca_ds.reshape(T,-1); ach_flat = ach_ds.reshape(T,-1)
    print(f"  Ca PCA ({ca_flat.shape[1]} vox, {N_PCS} PCs)...")
    pca_ca = PCA(n_components=N_PCS).fit(ca_flat)
    ca_sc = pca_ca.transform(ca_flat)
    print(f"  ACh PCA ({ach_flat.shape[1]} vox, {N_PCS} PCs)...")
    pca_ach = PCA(n_components=N_PCS).fit(ach_flat)
    ach_sc = pca_ach.transform(ach_flat)

    cc = np.array([[np.corrcoef(ca_sc[:,i], ach_sc[:,j])[0,1]
                     for j in range(N_PCS)] for i in range(N_PCS)], np.float32)

    for k in range(N_PCS):
        tifffile.imwrite(out/f"pca_ca_pc{k:02d}_spatial.tif",
                         pca_ca.components_[k].reshape(Zd,Yd,Xd).astype(np.float32))
        tifffile.imwrite(out/f"pca_ach_pc{k:02d}_spatial.tif",
                         pca_ach.components_[k].reshape(Zd,Yd,Xd).astype(np.float32))
    np.savez(out/"pca_timecourses.npz", ca_scores=ca_sc, ach_scores=ach_sc,
             ca_var=pca_ca.explained_variance_ratio_,
             ach_var=pca_ach.explained_variance_ratio_, cc=cc)

    # Spatial MIP plots
    for ch, comps, var in [("Ca",pca_ca.components_,pca_ca.explained_variance_ratio_),
                            ("ACh",pca_ach.components_,pca_ach.explained_variance_ratio_)]:
        fig, axes = plt.subplots(4,5, figsize=(20,12))
        for k in range(min(N_PCS,20)):
            ax = axes[k//5, k%5]
            mip = comps[k].reshape(Zd,Yd,Xd).max(0)
            vm = np.percentile(np.abs(mip), 99)
            ax.imshow(mip, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
            ax.set_title(f"PC{k} ({var[k]:.1%})"); ax.axis("off")
        fig.suptitle(f"{ch} PCA Spatial (Z-MIP)")
        fig.tight_layout(); fig.savefig(out/f"pca_{ch.lower()}_spatial_mip.png", dpi=200); plt.close(fig)

    # Timecourses
    t = np.arange(T)/FS_HZ
    fig, axes = plt.subplots(N_PCS, 2, figsize=(14, 2*N_PCS), sharex=True)
    for k in range(N_PCS):
        axes[k,0].plot(t, ca_sc[:,k], color="green", lw=.8); axes[k,0].set_ylabel(f"Ca PC{k}")
        axes[k,1].plot(t, ach_sc[:,k], color="red", lw=.8); axes[k,1].set_ylabel(f"ACh PC{k}")
    axes[-1,0].set_xlabel("Time (s)"); axes[-1,1].set_xlabel("Time (s)")
    fig.suptitle("PCA Timecourses")
    fig.tight_layout(); fig.savefig(out/"pca_timecourses.png", dpi=200); plt.close(fig)

    # Cross-corr matrix
    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(N_PCS)); ax.set_xticklabels([f"ACh{i}" for i in range(N_PCS)], rotation=45, ha="right")
    ax.set_yticks(range(N_PCS)); ax.set_yticklabels([f"Ca{i}" for i in range(N_PCS)])
    plt.colorbar(im, ax=ax, label="r"); ax.set_title("Ca PC × ACh PC")
    fig.tight_layout(); fig.savefig(out/"pca_cross_correlation.png", dpi=200); plt.close(fig)
    print(f"  Saved PCA to {out}")

def run_ica(ca, ach, out):
    """ICA: PCA dimensionality reduction → FastICA for independent sources."""
    from sklearn.decomposition import PCA, FastICA
    print("\n=== BLOCK 3b: ICA (PCA → FastICA) ===")
    T = ca.shape[0]; ds = PCA_DS
    ca_ds = ca[:,::ds,::ds,::ds]; ach_ds = ach[:,::ds,::ds,::ds]
    _, Zd, Yd, Xd = ca_ds.shape
    n_vox = Zd * Yd * Xd

    for ch, stack_ds in [("ca", ca_ds), ("ach", ach_ds)]:
        flat = stack_ds.reshape(T, -1)
        # PCA first for dimensionality reduction
        print(f"  {ch}: PCA ({n_vox} vox → {N_PCS} PCs)...")
        pca = PCA(n_components=N_PCS).fit(flat)
        scores = pca.transform(flat)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        print(f"  {ch}: PCA cumulative variance: {cum_var[-1]:.1%}")

        # FastICA on PCA scores
        print(f"  {ch}: FastICA ({N_PCS} ICs)...")
        ica = FastICA(n_components=N_PCS, max_iter=500, random_state=42)
        ic_scores = ica.fit_transform(scores)  # (T, N_PCS)
        # Spatial maps: project ICA unmixing back to voxel space
        # mixing_ is (N_PCS, N_PCS), components from PCA are (N_PCS, n_vox)
        ic_spatial = ica.components_ @ pca.components_  # (N_PCS, n_vox)

        # Sort ICs by kurtosis (spikier = more interesting)
        from scipy.stats import kurtosis
        kurt = np.array([kurtosis(ic_scores[:, k]) for k in range(N_PCS)])
        order = np.argsort(kurt)[::-1]
        ic_scores = ic_scores[:, order]
        ic_spatial = ic_spatial[order]
        kurt = kurt[order]

        # Save tifs
        for k in range(N_PCS):
            tifffile.imwrite(out / f"ica_{ch}_ic{k:02d}_spatial.tif",
                             ic_spatial[k].reshape(Zd, Yd, Xd).astype(np.float32))
        np.savez(out / f"ica_{ch}_timecourses.npz",
                 scores=ic_scores, kurtosis=kurt,
                 pca_var=pca.explained_variance_ratio_)

        # Spatial MIP plots
        ncols = 5; nrows = (N_PCS + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3 * nrows))
        axes = axes.ravel()
        for k in range(N_PCS):
            mip = ic_spatial[k].reshape(Zd, Yd, Xd).max(0)
            vm = np.percentile(np.abs(mip), 99)
            axes[k].imshow(mip, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
            axes[k].set_title(f"IC{k} (kurt={kurt[k]:.1f})"); axes[k].axis("off")
        for k in range(N_PCS, len(axes)):
            axes[k].axis("off")
        fig.suptitle(f"{ch.upper()} ICA Spatial (Z-MIP)")
        fig.tight_layout(); fig.savefig(out / f"ica_{ch}_spatial_mip.png", dpi=200); plt.close(fig)

        # Timecourses
        t = np.arange(T) / FS_HZ
        fig, axes_t = plt.subplots(N_PCS, 1, figsize=(14, 2 * N_PCS), sharex=True)
        color = "green" if ch == "ca" else "red"
        for k in range(N_PCS):
            axes_t[k].plot(t, ic_scores[:, k], color=color, lw=.8)
            axes_t[k].set_ylabel(f"IC{k}")
        axes_t[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{ch.upper()} ICA Timecourses (sorted by kurtosis)")
        fig.tight_layout(); fig.savefig(out / f"ica_{ch}_timecourses.png", dpi=200); plt.close(fig)

    # Cross-correlation: Ca ICs vs ACh ICs
    ca_ic = np.load(out / "ica_ca_timecourses.npz")["scores"]
    ach_ic = np.load(out / "ica_ach_timecourses.npz")["scores"]
    cc = np.array([[np.corrcoef(ca_ic[:, i], ach_ic[:, j])[0, 1]
                     for j in range(N_PCS)] for i in range(N_PCS)], np.float32)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(N_PCS)); ax.set_xticklabels([f"ACh{i}" for i in range(N_PCS)], rotation=45, ha="right")
    ax.set_yticks(range(N_PCS)); ax.set_yticklabels([f"Ca{i}" for i in range(N_PCS)])
    plt.colorbar(im, ax=ax, label="r"); ax.set_title("Ca IC × ACh IC")
    fig.tight_layout(); fig.savefig(out / "ica_cross_correlation.png", dpi=200); plt.close(fig)
    print(f"  Saved ICA to {out}")




# ================== CONTROLS ==================
def run_controls(ca, ach, regr_results, dendrite_mask, out, n_shuffles=1):
    print("\n=== CONTROLS ===")
    T, Z, Y, X = ca.shape

    for ch, stack in [("ca",ca),("ach",ach)]:
        g_z = _zscore_1d(stack.mean(axis=(1,2,3)))
        r_self = np.empty((Z,Y,X), np.float32)
        for z0,z1 in tqdm(list(_slab_ranges(Z,SLAB_Z)), desc=f"  {ch} self-corr"):
            nz=z1-z0; flat=_zscore_cols(stack[:,z0:z1].reshape(T,nz*Y*X))
            r_self[z0:z1] = (g_z[:,None]*flat).mean(0).reshape(nz,Y,X)
        tifffile.imwrite(out/f"ctrl_self_corr_{ch}.tif", r_self)
        print(f"  {ch} self-corr: mean={r_self.mean():.4f}")

    r_bleed = np.corrcoef(ca.mean(axis=(1,2,3)), ach.mean(axis=(1,2,3)))[0,1]
    print(f"  Bleedthrough: {r_bleed:.4f}")
    lines = [f"Bleedthrough: r = {r_bleed:.4f}"]

    r2u = regr_results.get("r2_ach_unique")
    if dendrite_mask is not None and r2u is not None:
        ins = r2u[dendrite_mask].ravel(); outs = r2u[~dendrite_mask].ravel()
        print(f"  R²_unique inside: {ins.mean():.5f}  outside: {outs.mean():.5f}")
        lines += [f"R2_unique inside: {ins.mean():.5f} n={len(ins)}",
                  f"R2_unique outside: {outs.mean():.5f} n={len(outs)}"]
        fig, ax = plt.subplots(figsize=(7,3.5))
        bins = np.linspace(0, .15, 80)
        ax.hist(ins, bins=bins, alpha=.6, color="green", label=f"Inside ({ins.mean():.4f})")
        ax.hist(outs, bins=bins, alpha=.6, color="gray", label=f"Outside ({outs.mean():.4f})")
        ax.set_xlabel("R²_ach_unique"); ax.legend()
        fig.tight_layout(); fig.savefig(out/"ctrl_inside_vs_outside.png", dpi=200); plt.close(fig)

    if n_shuffles > 0 and r2u is not None:
        print(f"  Permutation null ({n_shuffles} shuffles)...")
        ach_g_z = _zscore_1d(ach.mean(axis=(1,2,3)))
        ca_g_z = _zscore_1d(ca.mean(axis=(1,2,3)))
        rng = np.random.default_rng(42)
        null_signed, null_e001, null_e005 = [], [], []
        for _ in range(n_shuffles):
            shift = rng.integers(T//4, 3*T//4)
            ach_sh = np.roll(ach_g_z, shift)
            X_sh = np.column_stack([ach_sh, ca_g_z])
            XtXi_Xt = np.linalg.pinv(X_sh.T @ X_sh) @ X_sh.T
            dsum, n001, n005, nv = 0., 0, 0, 0
            for z0,z1 in _slab_ranges(Z, SLAB_Z):
                nz=z1-z0; flat=ca[:,z0:z1].reshape(T,nz*Y*X)
                cc = flat - flat.mean(0, keepdims=True)
                ss = (cc**2).sum(0)+1e-12
                r2m = np.clip(1-((cc - X_sh @ (XtXi_Xt @ cc))**2).sum(0)/ss, 0, 1)
                b = (ca_g_z[:,None]*cc).mean(0)
                r2c = np.clip(1-((cc - ca_g_z[:,None]*b[None,:])**2).sum(0)/ss, 0, 1)
                d = r2m - r2c
                dsum += d.sum(); n001 += (d>.01).sum(); n005 += (d>.05).sum(); nv += d.size
            null_signed.append(dsum/nv); null_e001.append(n001/nv); null_e005.append(n005/nv)
        ns = np.array(null_signed); ne1 = np.array(null_e001); ne5 = np.array(null_e005)
        real_e1, real_e5 = (r2u>.01).mean(), (r2u>.05).mean()
        print(f"  Null ΔR²: {ns.mean():.5f}±{ns.std():.5f}  Real: {r2u.mean():.5f}")
        print(f"  Null >0.01: {ne1.mean():.4f} (real {real_e1:.4f})  "
              f">0.05: {ne5.mean():.4f} (real {real_e5:.4f})")
        lines += [f"Null ΔR²: {ns.mean():.5f}±{ns.std():.5f}",
                  f"Null >0.01: {ne1.mean():.4f} real: {real_e1:.4f}",
                  f"Null >0.05: {ne5.mean():.4f} real: {real_e5:.4f}",
                  f"Real R2_unique: {r2u.mean():.5f}"]

    if r2u is not None:
        lines.append(f"R2_unique: mean={r2u.mean():.5f} max={r2u.max():.4f} "
                      f">0.01:{(r2u>.01).sum()} >0.05:{(r2u>.05).sum()}")
    (out/"summary.txt").write_text("\n".join(lines)+"\n")


# ================== FLY-THROUGH ==================
def export_flythrough(vol, name, out, cmap="RdBu_r", vmin=-.5, vmax=.5):
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print(f"  [skip] ffmpeg unavailable for {name}"); return
    Z = vol.shape[0]
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(vol[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax); title = ax.set_title(f"{name} Z=0"); ax.axis("off")
    def update(z):
        im.set_data(vol[z]); title.set_text(f"{name} Z={z}"); return [im, title]
    anim = FuncAnimation(fig, update, frames=Z, interval=200, blit=True)
    p = out / f"flythrough_{name}.mp4"
    anim.save(str(p), writer=FFMpegWriter(fps=5)); plt.close(fig)
    print(f"  Fly-through: {p}")

# ================== NAPARI ==================
def view_results():
    import napari
    out = BASE / "coupling_analysis"
    viewer = napari.Viewer(title="Ca–ACh Coupling"); scale = VOXEL_SIZE
    for name, cm, cl in [
        ("corr_rmax_globalach","RdBu_r",(-.5,.5)),
        ("corr_rmax_vox","RdBu_r",(-.5,.5)),
        ("regr_r2_ach_unique","hot",(0,.15)), ("regr_r2_multi","hot",(0,.3)),
        ("regr_beta_ach","RdBu_r",(-.3,.3)), ("regr_r_resid_ach","RdBu_r",(-.3,.3)),
        ("ctrl_self_corr_ca","RdBu_r",(-.5,.5))]:
        for folder in [out, out.parent/"coupling_analysis_nopc0"]:
            p = folder / f"{name}.tif"
            if p.exists():
                tag = "_nopc0" if "nopc0" in str(folder) else ""
                viewer.add_image(tifffile.imread(p), name=name+tag, colormap=cm,
                                 contrast_limits=cl, scale=scale, visible=False)
    if viewer.layers: viewer.layers[0].visible = True
    napari.run()


# ================== RUN ALL BLOCKS ==================
def run_all_blocks(ca, ach, out, only, dendrite_mask, n_shuffles, do_ica=False):
    """Run requested blocks, save to `out` folder."""
    out.mkdir(parents=True, exist_ok=True)
    plot_global_traces(ca, ach, out)

    corr_res, regr_res = {}, {}
    if only is None or only == "corr":
        corr_res = run_correlation(ca, ach, out)
    if only is None or only == "regr":
        regr_res = run_regression(ca, ach, out)
    if only is None or only == "pca":
        run_pca(ca, ach, out)
        if do_ica:
            run_ica(ca, ach, out)
    if only is None:
        run_controls(ca, ach, regr_res, dendrite_mask, out, n_shuffles)

    # Fly-throughs
    for name, cm, lo, hi in [
        ("regr_r2_ach_unique","hot",0,.1), ("corr_rmax_globalach","RdBu_r",-.5,.5),
        ("corr_rmax_vox","RdBu_r",-.5,.5),
        ("regr_r_resid_ach","RdBu_r",-.3,.3)]:
        p = out / f"{name}.tif"
        if p.exists():
            export_flythrough(tifffile.imread(p), name, out, cm, lo, hi)

# ================== MAIN ==================
def main():
    args = parse_args()
    if args.view: view_results(); return

    t0 = time.time()
    print(f"=== Ca-ACh Coupling (4-block) ===")
    print(f"    {DATE} | {MOUSE} | {RUN}\n")

    ca_raw = load_4d(RAW_CA); ach_raw = load_4d(RAW_ACH)
    ca_raw, ach_raw = skip_crop_align(ca_raw, ach_raw,
                                       args.skip_first_seconds, args.y_crop,
                                       args.crop_last_seconds)
    T, Z, Y, X = ca_raw.shape
    print(f"  Shape: T={T} Z={Z} Y={Y} X={X}\n")

    ca_dff, ach_dff = preprocess(ca_raw, ach_raw, args)
    del ca_raw, ach_raw; gc.collect()

    dendrite_mask = load_dendrite_mask((Z, Y, X))

    # --- Pass 1: standard ΔF/F ---
    out1 = BASE / "coupling_analysis"
    print(f"\n{'='*60}")
    print(f"  PASS 1: Standard ΔF/F → {out1}")
    print(f"{'='*60}")
    run_all_blocks(ca_dff, ach_dff, out1, args.only, dendrite_mask, args.n_shuffles, args.ica)

    # --- Pass 2: PC0 removed (if requested) ---
    if args.remove_pc0:
        print(f"\n{'='*60}")
        print(f"  PASS 2: PC0 removed → coupling_analysis_nopc0")
        print(f"{'='*60}")
        # Remove PC0 in-place on copies; free pass-1 data first if possible
        ca_nopc = ca_dff.copy(); del ca_dff; gc.collect()
        ca_nopc = remove_pc0(ca_nopc, "Ca")
        ach_nopc = ach_dff.copy(); del ach_dff; gc.collect()
        ach_nopc = remove_pc0(ach_nopc, "ACh")
        out2 = BASE / "coupling_analysis_nopc0"
        run_all_blocks(ca_nopc, ach_nopc, out2, args.only, dendrite_mask, args.n_shuffles, args.ica)
        del ca_nopc, ach_nopc; gc.collect()

    print(f"\n=== Done in {time.time()-t0:.0f}s ===")

if __name__ == "__main__":
    main()
