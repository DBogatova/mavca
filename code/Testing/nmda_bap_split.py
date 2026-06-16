#!/usr/bin/env python
"""
NMDA vs bAP amplitude — TEST 2: within-mask TOP/BOTTOM split.

Direct, spatially-resolved version of the hypothesis. For each curated mask we
split its voxels at the median cortical depth (Y) into a SUPERFICIAL (top) half
and a DEEP (bottom) half, re-extract a ΔF/F sub-trace for each half from the raw
4D stack (M4 core-shell convention), and classify every event of that mask:

  vertical_streak (bAP-like)  : BOTH halves active  -> signal extends along depth
  localized       (NMDA-like) : ONLY ONE half active -> confined to a branch segment

Then compare event amplitude. Prediction: localized < vertical_streak.

Key methodological choices (flagged):
  * Background shell is taken around the WHOLE mask and shared by both halves.
    A per-half shell would include the other (adjacent) half and CANCEL the very
    vertical-streak signal we want to measure.
  * Event amplitude = peak of the ACTIVE half (max(top,bottom)), NOT the whole-mask
    mean, so a one-half event is not artificially diluted by the silent half.
  * Raw stack + 10th-pct F0 (M4), NOT stack_voxel_norm_mean_sub (that has the
    per-frame spatial mean removed, which would erase global/streak signal).

Output: scape-data/nmda_bap_split_events.csv + figures/nmda_bap_split.png
"""
from pathlib import Path
import numpy as np, pandas as pd, tifffile, gc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import ball

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
VOXEL = np.array([3.9, 1.0, 1.2])      # Z,Y,X um ; Y = cortical depth

# (label, date, mouse, mask_run, raw_run, fs, skip_s)
FOVS = [
    ("0508",      "2026-05-08", "rbp4_139_phpeb", "run5", "run5", 5.0, 14),  # mixed regime
    ("0416_r1",   "2026-04-16", "rbp4_132_phpeb", "run1", "run1", 5.0, 12),  # global-rich (now clean)
    ("0512_r9",   "2026-05-12", "rbp4_132_phpeb", "run9", "run9", 5.0, 12),  # local/independent
]

NF0       = 150     # frames sampled across recording for F0 (10th pct)
K_MAD     = 3.5     # event threshold = median + K*1.4826*MAD (on % trace)
FLOOR     = 1.5     # absolute floor (%)
MIN_DUR   = 2
MIN_Y_EXT = 15.0    # um : mask must span >= this in depth to be splittable
MIN_HALF  = 15      # voxels : each half must have >= this many voxels
ACT_FRAC  = 0.30    # a half counts as "active" if its amp >= ACT_FRAC * whole-peak AND > its own thr

B1, B2, B3 = ball(1), ball(2), ball(3)

def raw_path(base, raw_run, mouse):
    cands = sorted((base/raw_run/"raw").glob("*reslice-bin.tif"))
    cands = [c for c in cands if "dff" not in c.name]
    return cands[0] if cands else None

def thresh(tr):
    m = np.median(tr); mad = np.median(np.abs(tr - m)) * 1.4826
    return max(FLOOR, m + K_MAD * mad)

def event_peaks(tr, thr):
    idx = np.where(tr > thr)[0]; pk = []
    if idx.size:
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if run.size >= MIN_DUR:
                pk.append(int(run[np.argmax(tr[run])]))
    return pk

def build_regions(m, Yraw):
    """m: (Z,Ymask,X) bool. Returns dict of global flat-index arrays (in Z*Yraw*X space)
    for whole_core, top_core, bot_core, shell; or None if not usable. Splittable flag too."""
    Z, Ym, X = m.shape
    if Ym < Yraw:                       # M4: pad mask bottom (deep) to match raw Y
        m = np.pad(m, ((0,0),(0,Yraw-Ym),(0,0)))
    elif Ym > Yraw:
        m = m[:, :-(Ym-Yraw), :]
    Z, Y, X = m.shape
    zz, yy, xx = np.where(m)
    if zz.size == 0: return None
    pad = 4
    z0,z1 = max(0,zz.min()-pad), min(Z,zz.max()+1+pad)
    y0,y1 = max(0,yy.min()-pad), min(Y,yy.max()+1+pad)
    x0,x1 = max(0,xx.min()-pad), min(X,xx.max()+1+pad)
    sub = m[z0:z1, y0:y1, x0:x1]
    # whole core / shell (shared background)
    core = binary_erosion(sub, B1)
    if not core.any(): core = sub.copy()
    inner = binary_dilation(sub, B2); outer = binary_dilation(sub, B3)
    shell = outer & ~inner
    # split by median Y of the actual mask voxels (in global coords)
    ymed = np.median(yy)
    ysub = np.arange(y0, y1)[None,:,None] * np.ones_like(sub, dtype=float)
    top = sub & (ysub <  ymed)          # superficial (smaller Y)
    bot = sub & (ysub >= ymed)          # deep (larger Y)
    y_ext = (yy.max() - yy.min() + 1) * VOXEL[1]
    splittable = (y_ext >= MIN_Y_EXT) and (top.sum() >= MIN_HALF) and (bot.sum() >= MIN_HALF)
    tcore = binary_erosion(top, B1);  tcore = tcore if tcore.any() else top
    bcore = binary_erosion(bot, B1);  bcore = bcore if bcore.any() else bot

    def gidx(b):
        lz, ly, lx = np.where(b)
        return ((lz+z0).astype(np.int64)*Y*X + (ly+y0)*X + (lx+x0))
    return dict(whole=gidx(core), top=gidx(tcore), bot=gidx(bcore), shell=gidx(shell),
                splittable=bool(splittable), y_ext=float(y_ext),
                ytop_um=float((np.median(yy[yy<ymed]) if (yy<ymed).any() else ymed)*VOXEL[1]),
                ybot_um=float((np.median(yy[yy>=ymed]) if (yy>=ymed).any() else ymed)*VOXEL[1]))

def extract_traces(rawp, regions, fs, skip_s):
    tf = tifffile.TiffFile(str(rawp)); T, Z, Y, X = tf.series[0].shape
    try:    arr = tifffile.memmap(str(rawp))
    except Exception: arr = tf.series[0].asarray()
    # F0: 10th percentile over frames sampled across the recording (post-skip)
    skip = int(skip_s*fs)
    fidx = np.unique(np.linspace(skip, T-1, NF0).astype(int))
    samp = np.asarray(arr[fidx]).astype(np.float32)
    f0 = np.percentile(samp, 10, axis=0).ravel()          # (Z*Y*X,)
    del samp; gc.collect()
    f0e = f0 + 1e-6
    names = list(regions.keys())
    out = {nm: dict(whole=np.empty(T,np.float32), top=np.empty(T,np.float32),
                    bot=np.empty(T,np.float32)) for nm in names}
    for t in range(T):
        dff = (np.asarray(arr[t]).astype(np.float32).ravel() - f0) / f0e
        for nm in names:
            R = regions[nm]
            sh = dff[R['shell']].mean() if R['shell'].size else 0.0
            out[nm]['whole'][t] = dff[R['whole']].mean() - sh
            out[nm]['top'][t]   = dff[R['top']].mean()   - sh
            out[nm]['bot'][t]   = dff[R['bot']].mean()   - sh
    for nm in names:
        for k in ('whole','top','bot'):
            out[nm][k] = out[nm][k]*100.0                 # -> %
    tf.close()
    return out, T

def analyze_fov(label, date, mouse, mask_run, raw_run, fs, skip_s):
    base = PR/"scape-data"/date/mouse
    rawp = raw_path(base, raw_run, mouse)
    if rawp is None: print(f"[{label}] no raw stack"); return None
    Yraw = tifffile.TiffFile(str(rawp)).series[0].shape[2]
    mpaths = sorted((base/mask_run/"labelmaps_curated_dynamic").glob("dend_*_labelmap.tif"))
    regions = {}
    for p in mpaths:
        nm = p.stem.replace("_labelmap","")
        R = build_regions(tifffile.imread(p) > 0, Yraw)
        if R is not None and R['splittable']:
            regions[nm] = R
    print(f"[{label}] {len(regions)}/{len(mpaths)} masks splittable (y_ext>={MIN_Y_EXT}um, "
          f">={MIN_HALF} vox/half). Extracting from {rawp.name} ...")
    traces, T = extract_traces(rawp, regions, fs, skip_s)

    rows = []
    for nm, R in regions.items():
        w, top, bot = traces[nm]['whole'], traces[nm]['top'], traces[nm]['bot']
        thw = thresh(w); tht = thresh(top); thb = thresh(bot)
        for p in event_peaks(w, thw):
            ta, ba = float(top[p]), float(bot[p])
            wa = float(w[p])
            amp_local = max(ta, ba)
            peak_ref = max(wa, amp_local, 1e-6)
            top_act = (ta > tht) and (ta >= ACT_FRAC*peak_ref)
            bot_act = (ba > thb) and (ba >= ACT_FRAC*peak_ref)
            if   top_act and bot_act: klass = "vertical_streak"
            elif top_act or  bot_act: klass = "localized"
            else:                      klass = "ambiguous"
            li = abs(ta-ba)/(abs(ta)+abs(ba)+1e-9)
            rows.append(dict(fov=label, name=nm, frame=p, fs=fs,
                             amp_whole=wa, amp_local=amp_local, top_amp=ta, bot_amp=ba,
                             localization_index=li, klass=klass,
                             y_top_um=R['ytop_um'], y_bot_um=R['ybot_um']))
    df = pd.DataFrame(rows)
    if df.empty: print(f"  [{label}] no events"); return df
    vs = df[df.klass=="vertical_streak"]; lo = df[df.klass=="localized"]
    print(f"  [{label}] events={len(df)} streak={len(vs)} localized={len(lo)} "
          f"ambiguous={(df.klass=='ambiguous').sum()}")
    if len(vs)>3 and len(lo)>3:
        med_vs, med_lo = vs.amp_local.median(), lo.amp_local.median()
        u,pu = stats.mannwhitneyu(vs.amp_local, lo.amp_local, alternative="greater")
        print(f"     amp_local median: streak={med_vs:.1f}%  localized={med_lo:.1f}%  "
              f"(streak/localized={med_vs/max(med_lo,1e-6):.2f}x)  MWU p={pu:.3g}")
    return df

def main():
    alld = []
    for cfg in FOVS:
        d = analyze_fov(*cfg)
        if d is not None and not d.empty: alld.append(d)
    if not alld: print("no data"); return
    E = pd.concat(alld, ignore_index=True)
    E.to_csv(PR/"code"/"Testing"/"data"/"nmda_bap_split_events.csv", index=False)

    print("\n================ POOLED (within-FOV z-scored amp_local) ================")
    E["amp_z"] = E.groupby("fov")["amp_local"].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
    vs = E[E.klass=="vertical_streak"]; lo = E[E.klass=="localized"]
    if len(vs)>3 and len(lo)>3:
        u,pu = stats.mannwhitneyu(vs.amp_z, lo.amp_z, alternative="greater")
        print(f"  vertical_streak n={len(vs)} median_z={vs.amp_z.median():+.2f} | "
              f"localized n={len(lo)} median_z={lo.amp_z.median():+.2f}")
        print(f"  Mann-Whitney (streak>localized) p={pu:.3g}")
    # paired within-mask: masks with >=2 of each class
    pr_rows=[]
    for (fov,nm),g in E.groupby(["fov","name"]):
        a=g[g.klass=="vertical_streak"].amp_local; b=g[g.klass=="localized"].amp_local
        if len(a)>=2 and len(b)>=2: pr_rows.append((a.median(), b.median()))
    if len(pr_rows)>=4:
        a=np.array([x[0] for x in pr_rows]); b=np.array([x[1] for x in pr_rows])
        w,pw=stats.wilcoxon(a,b, alternative="greater")
        print(f"  within-mask paired (n={len(pr_rows)} masks w/ both classes): "
              f"streak>localized in {(a>b).sum()}/{len(pr_rows)}  Wilcoxon p={pw:.3g}")
    # amp vs localization index
    r,p = stats.spearmanr(E.localization_index, E.amp_local)
    print(f"  amp_local vs localization_index (1=one-half): Spearman ρ={r:+.2f} p={p:.2g} "
          f"(predict negative)")
    make_fig(E)
    print("\nSaved: scape-data/nmda_bap_split_events.csv, figures/nmda_bap_split.png")

def make_fig(E):
    (PR/"code"/"Testing"/"figures").mkdir(parents=True, exist_ok=True)
    fovs = list(E.fov.unique()); nf=len(fovs)
    fig, ax = plt.subplots(1, nf+1, figsize=(4.5*(nf+1), 4.2))
    for i,fov in enumerate(fovs):
        g=E[E.fov==fov]
        data=[g[g.klass=="localized"].amp_local.values, g[g.klass=="vertical_streak"].amp_local.values]
        ax[i].boxplot(data, labels=[f"localized\n(NMDA)\nn={len(data[0])}",
                                    f"streak\n(bAP)\nn={len(data[1])}"], showfliers=False)
        ax[i].set_title(fov); ax[i].set_ylabel("event amp_local ΔF/F (%)"); ax[i].grid(alpha=.3)
    g=E
    ax[-1].scatter(g.localization_index, g.amp_local, s=10, alpha=.3,
                   c=np.where(g.klass=="vertical_streak","tab:blue","tab:red"))
    ax[-1].set_xlabel("localization index (0=both halves, 1=one half)")
    ax[-1].set_ylabel("amp_local ΔF/F (%)")
    r,p=stats.spearmanr(g.localization_index,g.amp_local)
    ax[-1].set_title(f"pooled  ρ={r:+.2f}, p={p:.1g}"); ax[-1].grid(alpha=.3)
    fig.suptitle("Test 2: within-mask top/bottom split — localized (NMDA) vs vertical-streak (bAP) amplitude",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(PR/"code"/"Testing"/"figures"/"nmda_bap_split.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
