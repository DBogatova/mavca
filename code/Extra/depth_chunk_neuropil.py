#!/usr/bin/env python
"""
(a) Top-down depth-chunk event-triggered heatmap: split each mask into 5 cortical-depth
    (Y) chunks, average the chunk x time intensity around event peaks -> does signal lead
    superficial (top-down) or deep (bottom-up)?
(b) Neuropil contamination test: correlate in-mask global signal with an out-of-mask
    (background) signal from the SAME stack. High r => masks track a field-wide signal
    (neuropil/hemodynamic) = contamination.

All signals from preprocessed/stack_voxel_norm_mean_sub.tif (mask-aligned), so no CSV/stack
frame mismatch. NCHUNK=5, chunk0=superficial (low Y), chunk4=deep.
"""
from pathlib import Path
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.morphology import ball

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
FS = 5.0; NCHUNK = 5; W = 4; ZTHR = 2.0     # event z-threshold on per-mask stack trace
RUNS = [("2026-04-16","rbp4_132_phpeb","run1","outlier"),
        ("2026-05-08","rbp4_139_phpeb","run5","mixed")]

def align(m, Y):
    my = m.shape[1]
    if my > Y:   m = m[:, :-(my-Y), :]
    elif my < Y: m = np.pad(m, ((0,0),(0,Y-my),(0,0)))
    return m

def analyze(date, mouse, run, tag):
    base = PR/"scape-data"/date/mouse/run
    stack = base/"preprocessed"/"stack_voxel_norm_mean_sub.tif"
    arr = tifffile.memmap(str(stack)); T, Z, Y, X = arr.shape
    paths = sorted((base/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))
    masks = [align(tifffile.imread(p) > 0, Y) for p in paths]
    M = len(masks)
    grp = np.zeros((Z, Y, X), np.int32)          # mask*NCHUNK + chunk + 1
    union = np.zeros((Z, Y, X), bool)
    for im, m in enumerate(masks):
        union |= m
        c = np.argwhere(m); ys = c[:, 1]
        lo, hi = ys.min(), ys.max()
        ch = np.clip(((ys - lo)/(hi - lo + 1e-9)*NCHUNK).astype(int), 0, NCHUNK-1)
        grp[c[:,0], c[:,1], c[:,2]] = im*NCHUNK + ch + 1
    bg = ~binary_dilation(union, structure=ball(3))   # far-from-mask background
    NG = M*NCHUNK
    gflat = grp.ravel(); bgflat = bg.ravel()
    sums = np.zeros((NG+1, T)); inmask = np.zeros(T); neuropil = np.zeros(T)
    for t in range(T):
        fr = np.asarray(arr[t]).ravel().astype(np.float64)
        sums[:, t] = np.bincount(gflat, weights=fr, minlength=NG+1)
        inmask[t] = sums[1:, t].sum()/(gflat > 0).sum()
        neuropil[t] = fr[bgflat].mean()
    counts = np.bincount(gflat, minlength=NG+1).astype(float)
    chunk = (sums / np.maximum(counts[:, None], 1))[1:].reshape(M, NCHUNK, T)  # (M,NCHUNK,T)

    # (b) contamination
    r_np = np.corrcoef(inmask, neuropil)[0, 1]

    # (a) event-triggered chunk x time
    et = []
    peak_times = []
    for im in range(M):
        mt = chunk[im].mean(0)                    # mask trace (mean over chunks)
        z = (mt - mt.mean())/(mt.std() + 1e-9)
        idx = np.where(z > ZTHR)[0]
        if idx.size == 0: continue
        for run_ in np.split(idx, np.where(np.diff(idx) > 1)[0]+1):
            p = run_[np.argmax(mt[run_])]
            if p-W < 0 or p+W+1 > T: continue
            seg = chunk[im][:, p-W:p+W+1].copy()  # (NCHUNK, 2W+1)
            base_ = seg[:, :2].mean(1, keepdims=True)
            seg = seg - base_
            mx = seg.max()
            if mx <= 0: continue
            et.append(seg/mx)
            peak_times.append([np.argmax(seg[c]) for c in range(NCHUNK)])
    et = np.array(et); n_ev = len(et)
    eta = et.mean(0) if n_ev else np.zeros((NCHUNK, 2*W+1))      # event-triggered avg
    pt = np.array(peak_times).mean(0) if n_ev else np.full(NCHUNK, np.nan)  # mean peak frame per chunk
    slope = np.polyfit(np.arange(NCHUNK), pt, 1)[0] if n_ev else np.nan      # >0: deep later = top-down

    # plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    tt = (np.arange(2*W+1)-W)/FS
    im0 = ax[0].imshow(eta, aspect='auto', cmap='magma',
                       extent=[tt[0], tt[-1], NCHUNK-0.5, -0.5])
    ax[0].set_yticks(range(NCHUNK)); ax[0].set_yticklabels(['0 surf','1','2','3','4 deep'])
    ax[0].set_xlabel('time from peak (s)'); ax[0].set_title(f"{tag} {run}: event-avg chunk x time (n={n_ev})")
    ax[0].axvline(0, color='w', lw=.6); fig.colorbar(im0, ax=ax[0], shrink=.8)
    ax[1].plot(range(NCHUNK), pt, 'o-'); ax[1].set_xlabel('depth chunk (surf->deep)')
    ax[1].set_ylabel('mean peak frame'); ax[1].set_title(f"top-down slope={slope:+.2f} fr/chunk\n(+ = superficial leads)"); ax[1].grid(alpha=.3)
    tline = np.arange(T)/FS
    ax[2].plot(tline, (inmask-inmask.mean())/inmask.std(), 'k', lw=.7, label='in-mask global')
    ax[2].plot(tline, (neuropil-neuropil.mean())/neuropil.std(), 'tab:red', lw=.7, alpha=.7, label='neuropil(out-of-mask)')
    ax[2].set_title(f"contamination: r(in-mask, neuropil)={r_np:+.2f}"); ax[2].legend(fontsize=7); ax[2].set_xlabel('s')
    out = base/"mask_analysis"; out.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out/"depth_chunk_neuropil.png", dpi=150); plt.close(fig)
    print(f"[{tag} {run}] masks={M} events={n_ev}  topdown_slope={slope:+.2f} fr/chunk  "
          f"r(in-mask,neuropil)={r_np:+.2f}  chunk_peak_frames={np.round(pt,2).tolist()}")
    return r_np, slope

if __name__ == "__main__":
    for d, mo, rn, tag in RUNS:
        analyze(d, mo, rn, tag)
