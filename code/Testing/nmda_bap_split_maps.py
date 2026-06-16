#!/usr/bin/env python
"""
Spatial maps for the branch/bend split test (nmda_bap_split_v2).

Per FOV, two panels over the Y×X MIP (movie view; surface line drawn):
  LEFT  : every considered mask, each SEGMENT a different color -> shows WHERE THE CUTS ARE.
  RIGHT : each mask colored by its dominant EVENT class:
            bAP/Ca-dominant (mostly 'extended' events)   = blue
            NMDA-dominant   (mostly 'localized' events)   = red
            mixed                                          = purple
            single-segment (couldn't host localized-vs-extended) = gray
            no events                                      = dark outline only
          small text = (extended/localized) event counts.

Note: NMDA vs bAP is a per-EVENT label (one mask can do both); the right panel shows
the per-mask majority. Uses scape-data/nmda_bap_split_v2_events.csv for counts.
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, gc, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("v", PR/"code/Testing/nmda_bap_split_v2.py")
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
VOXEL = v.VOXEL

EV = pd.read_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_v2_events.csv")

def bg_mip(rawp, skip, fs, nf=40):
    tf = tifffile.TiffFile(str(rawp)); T = tf.series[0].shape[0]
    try: arr = tifffile.memmap(str(rawp))
    except Exception: arr = tf.series[0].asarray()
    idx = np.unique(np.linspace(int(skip*fs), T-1, nf).astype(int))
    mv = np.asarray(arr[idx]).astype(np.float32).mean(0)   # (Z,Y,X)
    tf.close()
    return mv.max(0), mv                                    # Y×X MIP, full vol

def mask_class(fov, name, nseg):
    g = EV[(EV.fov==fov) & (EV.name==name)]
    n_ext = int((g.klass=="extended").sum()); n_loc = int((g.klass=="localized").sum())
    if nseg < 2:           return "single", n_ext, n_loc
    tot = n_ext + n_loc
    if tot == 0:           return "silent", 0, 0
    if n_loc >= 2:         return "NMDA", n_ext, n_loc   # >=2 localized events (SNR-robust rule)
    return "bAP", n_ext, n_loc                            # 0-1 localized -> bAP/extended

CLASS_COLOR = {"bAP":(0.20,0.45,0.95), "NMDA":(0.92,0.22,0.20),
               "single":(0.55,0.55,0.55), "silent":(1.0,0.55,0.0)}  # silent = bg-cancelled (orange)

def shifts_differ(a):
    """boolean (Y,X): pixel differs from any 4-neighbor (ignoring equal/zero handled by caller)."""
    d = np.zeros(a.shape, bool)
    d[1:,:]  |= a[1:,:]  != a[:-1,:]
    d[:-1,:] |= a[:-1,:] != a[1:,:]
    d[:,1:]  |= a[:,1:]  != a[:,:-1]
    d[:,:-1] |= a[:,:-1] != a[:,1:]
    return d

def build_fov(label, date, mouse, mask_run, raw_run, fs, skip):
    base = PR/"scape-data"/date/mouse
    rawp = v.raw_path(base, raw_run)
    mip, mv = bg_mip(rawp, skip, fs)
    Z, Y, X = mv.shape
    sp = v.fit_surface(mv); surf_x = sp['a']*np.arange(X) + sp['c'] + sp['b']*(Z/2)
    mpaths = sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))

    seg_id = np.zeros((Y, X), np.int32)      # global segment id
    mask_id = np.zeros((Y, X), np.int32)     # mask index (1-based)
    gid = 0
    info = []                                # (mask_idx, name, klass, n_ext, n_loc, cy, cx)
    for mi, p in enumerate(mpaths, 1):
        nm = p.stem.replace("_labelmap","")
        m = tifffile.imread(p) > 0
        if m.shape[1] < Y:  m = np.pad(m, ((0,0),(0,Y-m.shape[1]),(0,0)))
        elif m.shape[1] > Y: m = m[:, :-(m.shape[1]-Y), :]
        lab, nseg = v.segment_mask(m)
        kl, ne, nl = mask_class(label, nm, nseg)
        # project each segment to Y×X
        foot_all = (lab > 0).max(0)
        for s in range(1, nseg+1):
            foot = (lab == s).max(0)
            if not foot.any(): continue
            gid += 1
            seg_id[foot] = gid
        mask_id[foot_all] = mi
        ys, xs = np.where(foot_all)
        info.append((mi, nm, kl, ne, nl, ys.mean(), xs.mean()))
    return mip, surf_x, seg_id, mask_id, info

def overlay_rgb(mip, seg_id=None, mask_id=None, class_of=None, cut=False):
    g = mip.astype(np.float32); g = (g - g.min())/(np.percentile(g,99.5)-g.min()+1e-9)
    g = np.clip(g, 0, 1)
    rgb = np.dstack([g, g, g])*0.85
    if seg_id is not None:                    # segment-colored
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20")
        ids = seg_id.copy()
        col = cmap((ids % 20)/19.0)[..., :3]
        m = ids > 0
        rgb[m] = 0.45*rgb[m] + 0.55*col[m]
        # cut lines: boundary between different segments within SAME mask
        diff = shifts_differ(seg_id) & (seg_id > 0) & ~shifts_differ(mask_id)
        rgb[diff] = np.array([0,0,0])
        # mask outlines (white)
        out = shifts_differ(mask_id) & (mask_id > 0)
        rgb[out] = np.array([1,1,1])
    elif class_of is not None:                # class-colored
        for mi, kl in class_of.items():
            foot = mask_id == mi
            c = np.array(CLASS_COLOR[kl])
            rgb[foot] = 0.45*rgb[foot] + 0.55*c
        out = shifts_differ(mask_id) & (mask_id > 0)
        rgb[out] = np.array([1,1,1])
    return rgb

def main():
    for cfg in v.FOVS:
        label, date, mouse, mask_run, raw_runs, fs, skip = cfg
        mip, surf_x, seg_id, mask_id, info = build_fov(label, date, mouse, mask_run, mask_run, fs, skip)
        class_of = {mi: kl for (mi, nm, kl, ne, nl, cy, cx) in info}
        Y, X = mip.shape
        fig, ax = plt.subplots(1, 2, figsize=(2*X/45+2, Y/45+1.5))
        ax[0].imshow(overlay_rgb(mip, seg_id=seg_id, mask_id=mask_id), aspect='equal')
        ax[0].plot(np.arange(X), surf_x, color='cyan', lw=1.0, alpha=.8)
        ax[0].set_title(f"{label}: masks + segment cuts (n={len(info)} masks)")
        ax[1].imshow(overlay_rgb(mip, mask_id=mask_id, class_of=class_of), aspect='equal')
        ax[1].plot(np.arange(X), surf_x, color='cyan', lw=1.0, alpha=.8)
        # annotate class panel with counts
        nN = sum(k=="NMDA" for k in class_of.values()); nB = sum(k=="bAP" for k in class_of.values())
        nS = sum(k=="single" for k in class_of.values()); nZ = sum(k=="silent" for k in class_of.values())
        for (mi, nm, kl, ne, nl, cy, cx) in info:
            if kl in ("bAP","NMDA") and (ne+nl) > 0:
                ax[1].text(cx, cy, f"{ne}/{nl}", color='white', fontsize=5,
                           ha='center', va='center')
        ax[1].set_title(f"{label}: NMDA = >=2 localized events  (NMDA={nN} bAP/Ca={nB} "
                        f"1seg={nS} bg-cancelled={nZ})")
        leg = [Patch(color=CLASS_COLOR['NMDA'], label='NMDA (>=2 localized events)'),
               Patch(color=CLASS_COLOR['bAP'], label='bAP/Ca (0-1 localized)'),
               Patch(color=CLASS_COLOR['single'], label='single-segment (n/a)'),
               Patch(color=CLASS_COLOR['silent'], label='bg-cancelled (core active, shell~=core)')]
        ax[1].legend(handles=leg, loc='lower right', fontsize=6, framealpha=.7)
        for a in ax:
            a.set_xlabel("X (lateral)"); a.set_ylabel("Y (cortical depth)")
            a.set_ylim(Y-0.5, -0.5); a.set_xlim(-0.5, X-0.5)
        fig.suptitle(f"{label}  ('e/l' = extended/localized event counts; cyan = pial surface)",
                     fontsize=11)
        fig.tight_layout()
        out = PR/"code"/"Testing"/"figures"/f"nmda_bap_map_{label}.png"
        fig.savefig(out, dpi=160); plt.close(fig)
        print(f"[{label}] saved {out.name}  masks={len(info)} "
              f"(NMDA={nN} bAP-only={nB} single={nS} silent={nZ})")

if __name__ == "__main__":
    main()
