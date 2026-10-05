#!/usr/bin/env python
"""tune_ai162_detect.py - OFFLINE diagnosis + parameter sweep for the deep dual-channel
Ai162 SCAPE stacks (code/Auto/). NOT part of the pipeline and never imported by it.

Why: on 2026-02-09/rbp4cre_136_phpeb/run1 (69 planes) the detector finds 18 units vs 34
human masks (functional recall 0.21, voxel coverage 4%). This script asks whether the
default thresholds (FG_Z=4.5, SEED_Z=7, MIN_VOX=400, GROW_R=0.45), tuned on 30-plane
viral runs, are too strict for the lower-SNR dual-channel transgenic stacks.

How it stays faithful and cheap:
  * It calls auto_detect.segment ONCE (default params) to get the real activity map `top`
    and the default labelmap, and reuses the renormalised smoothed volume segment leaves in
    /tmp/scape_auto_scratch/<key>_zs.npy as `flat` (the exact substrate the detector sees).
  * It then replicates segment's foreground / greedy-grow / merge / clean-up steps verbatim
    (copied below) so only the cheap, parameter-dependent parts rerun for each grid point.

It writes ONLY to /tmp/tune_ai162 and PROJ/scape-auto/stats/tune_ai162. It does not modify
any pipeline file.
"""
from __future__ import annotations
import sys, time, json, itertools, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

PROJ = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
sys.path.insert(0, str(PROJ / "code"))
from Auto import auto_detect as ad                                   # noqa: E402
from Auto.scape_common import get_run, human_masks, VOXEL_ZYX        # noqa: E402

P = dict(ad.P)                       # detector defaults
OUTDIR = PROJ / "scape-auto" / "stats" / "tune_ai162"
TMP = Path("/tmp/tune_ai162")
CHUNK = 400_000


# ---------------------------------------------------------------- faithful segment pieces
def fg_substrate(flat, top, shape, fg_z):
    """Replicates segment steps 2-3: foreground, normalised trace matrix, 26-neighbour graph."""
    Z, Y, X = shape
    V = Z * Y * X
    T = flat.shape[0]
    idx = np.flatnonzero(top > fg_z)
    n = idx.size
    tr = np.empty((T, n), np.float32)
    for i in range(0, V, CHUNK):                      # chunked gather (fancy index on memmap is slow)
        sel = (idx >= i) & (idx < i + CHUNK)
        if sel.any():
            tr[:, sel] = flat[:, i:i + CHUNK][:, idx[sel] - i]
    tr = np.clip(tr, -2, None)
    tr -= tr.mean(0)
    tr /= np.linalg.norm(tr, axis=0) + 1e-6
    pos = np.full(V, -1, np.int64)
    pos[idx] = np.arange(n)
    coords = np.array(np.unravel_index(idx, (Z, Y, X))).T
    dirs = np.array([d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)])
    nbr = np.full((n, 26), -1, np.int64)
    for k, d in enumerate(dirs):
        c2 = coords + d
        ok = np.all((c2 >= 0) & (c2 < [Z, Y, X]), 1)
        nbr[ok, k] = pos[np.ravel_multi_index(c2[ok].T, (Z, Y, X))]
    return idx, tr, pos, nbr, top[idx]


def grow(tr, nbr, topv, seed_z, grow_r):
    """Replicates segment's greedy seeded region growing."""
    n = topv.size
    assigned = np.zeros(n, np.int32)
    units = []
    for s in np.argsort(-topv):
        if topv[s] < seed_z:
            break
        if assigned[s]:
            continue
        nb = nbr[s][nbr[s] >= 0]
        nb = nb[assigned[nb] == 0]
        ref = tr[:, np.r_[s, nb]].mean(1)
        ref /= np.linalg.norm(ref) + 1e-9
        region = [s]
        inreg = np.zeros(n, bool)
        inreg[s] = True
        frontier = np.array([s])
        it = 0
        while frontier.size:
            cand = nbr[frontier].ravel()
            cand = np.unique(cand[cand >= 0])
            cand = cand[(~inreg[cand]) & (assigned[cand] == 0)]
            if not cand.size:
                break
            new = cand[(ref @ tr[:, cand]) > grow_r]
            inreg[new] = True
            region.extend(new.tolist())
            frontier = new
            it += 1
            if it % 5 == 0:
                ref = tr[:, region].mean(1)
                ref /= np.linalg.norm(ref) + 1e-9
        units.append(np.asarray(region))
        assigned[units[-1]] = len(units)
    return units


def merge(units, idx, tr, shape, merge_r, merge_gap):
    """Replicates segment's touching-and-correlated merge; returns relabelled lab (Z,Y,X)."""
    Z, Y, X = shape
    V = Z * Y * X
    K = len(units)
    lab = np.zeros(V, np.int32)
    for k, u in enumerate(units, 1):
        lab[idx[u]] = k
    lab = lab.reshape(Z, Y, X)
    if K == 0:
        return lab
    U = np.stack([tr[:, u].mean(1) for u in units], 1)
    U -= U.mean(0)
    U /= np.linalg.norm(U, axis=0) + 1e-9
    C = U.T @ U
    adj = set()
    g = merge_gap
    grown = ndi.grey_dilation(lab, size=(1 + 2 * min(g, 1), 1 + 2 * g, 1 + 2 * g))
    m = (lab > 0) & (grown > 0) & (grown != lab)
    for i, j in set(zip(lab[m].tolist(), grown[m].tolist())):
        adj.add((min(i, j), max(i, j)))
    grown2 = -ndi.grey_dilation(-np.where(lab > 0, lab, K + 1), size=(1 + 2 * min(g, 1), 1 + 2 * g, 1 + 2 * g))
    m = (lab > 0) & (grown2 <= K) & (grown2 != lab)
    for i, j in set(zip(lab[m].tolist(), grown2[m].tolist())):
        adj.add((min(i, j), max(i, j)))
    parent = list(range(K + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in sorted(adj, key=lambda e: -C[e[0] - 1, e[1] - 1]):
        if C[i - 1, j - 1] > merge_r:
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a
    roots = np.array([find(i) for i in range(K + 1)])
    return roots[lab]


def finalize(lab, shape, params):
    """Replicates segment's per-unit clean-up + size/shape filters (geometry filters are off
    by default). Returns (out labelmap uint16, list of global voxel-id arrays per unit)."""
    p = dict(P, **params)
    Z, Y, X = shape
    out = np.zeros((Z, Y, X), np.uint16)
    final = []
    objs = ndi.find_objects(lab)
    nid = 0
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = lab[sl] == k
        cc, ncc = ndi.label(m, np.ones((3, 3, 3), bool))
        if ncc > 1:
            m = cc == (np.bincount(cc.ravel())[1:].argmax() + 1)
        m = ndi.binary_fill_holes(m)
        nvox = int(m.sum())
        if nvox < p["MIN_VOX"]:
            continue
        zz, yy, xx = np.nonzero(m)
        zz, yy, xx = zz + sl[0].start, yy + sl[1].start, xx + sl[2].start
        yspan = (yy.max() - yy.min() + 1) * VOXEL_ZYX[1]
        if yspan < p["MIN_YSPAN_UM"]:
            continue
        xspan = (xx.max() - xx.min() + 1) * VOXEL_ZYX[2]
        if xspan > p["MIN_SHEET_X_UM"] and xspan > p["MAX_SHEET_RATIO"] * yspan:
            continue
        # MIN_PEAK_SNR / MIN_VERTICALITY / MIN_ELONGATION are 0 by default -> never cut.
        nid += 1
        sub = out[sl]
        mm = m & (sub == 0)
        sub[mm] = nid
        zz2, yy2, xx2 = np.nonzero(mm)
        vid = np.ravel_multi_index((zz2 + sl[0].start, yy2 + sl[1].start, xx2 + sl[2].start), (Z, Y, X))
        final.append(vid)
    return out, final


# ---------------------------------------------------------------- scoring helpers
def _std_cols(M):
    """clip at -2, centre, L2-normalise each column, exactly like segment's trace prep."""
    M = np.clip(M, -2, None).astype(np.float32)
    M = M - M.mean(0)
    M /= np.linalg.norm(M, axis=0) + 1e-9
    return M


def unit_trace(tr, pos, vid):
    """Mean of the (standardised) activity over a unit's foreground voxels."""
    pv = pos[vid]
    pv = pv[pv >= 0]
    if pv.size == 0:
        return None
    return tr[:, pv].mean(1)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da < 1e-9 or db < 1e-9:
        return np.nan
    return float((a @ b) / (da * db))


def score_combo(out, final, pos, tr, shape, hvid, htr, min_vox):
    """functional recall (approx of validate_vs_human), n_units, median split-half consistency."""
    Z, Y, X = shape
    nid = len(final)
    sizes = np.array([v.size for v in final])
    atr = [unit_trace(tr, pos, v) for v in final]
    # recall: human mask recovered if some auto unit has oc>=0.2 and trace corr>=0.7
    nh = len(hvid)
    recovered = 0
    best_rs = []
    outflat = out.ravel()
    for j in range(nh):
        hv = hvid[j]
        cnt = np.bincount(outflat[hv], minlength=nid + 1)[1:]       # intersection per auto id
        best = np.nan
        for a in np.flatnonzero(cnt > 0):
            oc = cnt[a] / min(sizes[a], hv.size)
            if oc >= 0.2 and atr[a] is not None:
                r = corr(atr[a], htr[j])
                if np.isnan(best) or r > best:
                    best = r
        best_rs.append(best)
        if not np.isnan(best) and best >= 0.7:
            recovered += 1
    # split-half: mean activity of top vs bottom half of each unit by Y
    sh = []
    for v in final:
        yy = np.unravel_index(v, (Z, Y, X))[1]
        med = np.median(yy)
        a = v[yy < med]
        b = v[yy >= med]
        pa = pos[a]; pa = pa[pa >= 0]
        pb = pos[b]; pb = pb[pb >= 0]
        if pa.size >= 5 and pb.size >= 5:
            sh.append(corr(tr[:, pa].mean(1), tr[:, pb].mean(1)))
    best_rs = np.array(best_rs, float)
    return dict(
        n_units=nid,
        human_recall=round(recovered / nh, 3) if nh else None,
        n_recovered=recovered,
        median_best_r=round(float(np.nanmedian(best_rs)), 3) if np.isfinite(best_rs).any() else None,
        median_split_half=round(float(np.nanmedian(sh)), 3) if sh else None,
        median_unit_vox=int(np.median(sizes)) if nid else 0,
    )


# ---------------------------------------------------------------- diagnosis
def diagnose(top, shape, hvid, hsizes, hyspan, out0_recall_tag=""):
    V = int(np.prod(shape))
    union = np.zeros(V, bool)
    for hv in hvid:
        union[hv] = True
    tin = top[union]
    tout = top[~union]
    def frac_above(thr, arr):
        return round(float((arr > thr).mean()), 4)
    pct = {f"p{q}": round(float(np.percentile(top, q)), 3) for q in (50, 90, 95, 99, 99.5, 99.9)}
    diag = dict(
        n_human=len(hvid),
        human_union_vox=int(union.sum()),
        top_global_pct=pct,
        top_in_human=dict(mean=round(float(tin.mean()), 3), median=round(float(np.median(tin)), 3),
                          p75=round(float(np.percentile(tin, 75)), 3), p90=round(float(np.percentile(tin, 90)), 3)),
        top_outside=dict(mean=round(float(tout.mean()), 3), median=round(float(np.median(tout)), 3),
                         p90=round(float(np.percentile(tout, 90)), 3), p99=round(float(np.percentile(tout, 99)), 3)),
        human_vox_frac_above=dict(
            fg45=frac_above(4.5, tin), fg35=frac_above(3.5, tin), fg30=frac_above(3.0, tin),
            fg25=frac_above(2.5, tin), seed7=frac_above(7.0, tin), seed6=frac_above(6.0, tin),
            seed5=frac_above(5.0, tin)),
        per_mask_frac_above_fg45=round(float(np.mean([(top[hv] > 4.5).mean() for hv in hvid])), 4),
        per_mask_frac_above_seed7=round(float(np.mean([(top[hv] > 7.0).mean() for hv in hvid])), 4),
        human_mask_vox=dict(min=int(hsizes.min()), median=int(np.median(hsizes)), max=int(hsizes.max()),
                            frac_lt_400=round(float((hsizes < 400).mean()), 3),
                            frac_lt_250=round(float((hsizes < 250).mean()), 3),
                            frac_lt_150=round(float((hsizes < 150).mean()), 3)),
        human_mask_yspan_um=dict(min=round(float(hyspan.min()), 1), median=round(float(np.median(hyspan)), 1),
                                 max=round(float(hyspan.max()), 1),
                                 frac_lt_20=round(float((hyspan < 20).mean()), 3)),
    )
    return diag


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid", action="store_true", help="run the parameter grid (slow)")
    ap.add_argument("--fgz", type=float, nargs="*", default=[3.0, 3.5, 4.5])
    ap.add_argument("--seedz", type=float, nargs="*", default=[5.0, 6.0, 7.0])
    ap.add_argument("--minvox", type=int, nargs="*", default=[150, 250, 400])
    ap.add_argument("--growr", type=float, nargs="*", default=[0.4, 0.45])
    ap.add_argument("--only", default="", help="label suffix for the output files")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    r = get_run(a.key)
    z = np.load(TMP / f"z_{a.tag}.npy", mmap_mode="r")
    T, Z, Y, X = z.shape
    shape = (Z, Y, X)
    V = Z * Y * X
    print(f"[{a.key}] z {z.shape} memmap={isinstance(z, np.memmap)}", flush=True)

    # --- one real segment call: real activity map `top`, default labelmap, and the flat substrate
    t0 = time.time()
    vkey = f"tune_{a.tag}_valid"
    top_f = TMP / f"top_{a.tag}.npy"
    out0_f = TMP / f"out0_{a.tag}.npy"
    zs_f = ad.SCRATCH / f"{vkey}_zs.npy"
    if top_f.exists() and out0_f.exists() and zs_f.exists():
        top = np.load(top_f).reshape(-1)
        out0 = np.load(out0_f)
        n_default = int(out0.max())
        print(f"[{a.key}] reusing cached ad.segment result ({n_default} units)", flush=True)
    else:
        out0, top3d, info0 = ad.segment(z, verbose=True, key=vkey, params=None)
        top = np.asarray(top3d).reshape(-1)
        out0 = np.asarray(out0)
        n_default = len(info0)
        np.save(top_f, top)
        np.save(out0_f, out0)
        print(f"[{a.key}] ad.segment default -> {n_default} units ({time.time()-t0:.0f}s)", flush=True)
    flat = np.load(zs_f, mmap_mode="r").reshape(T, V)

    # --- human masks (drop any empty masks)
    hm = human_masks(r, shape)
    hvid_all = [np.flatnonzero(m.ravel()) for _, m in hm]
    n_empty = int(sum(v.size == 0 for v in hvid_all))
    hvid = [v for v in hvid_all if v.size > 0]
    hsizes = np.array([v.size for v in hvid])
    hyspan = np.array([(np.ptp(np.unravel_index(v, shape)[1]) + 1) * VOXEL_ZYX[1] for v in hvid])
    print(f"[{a.key}] human masks: {len(hm)} total, {n_empty} empty, {len(hvid)} used", flush=True)

    # human traces from the same flat substrate (chunked gather, standardised like auto)
    hud = np.unique(np.concatenate(hvid))
    Hraw = np.empty((T, hud.size), np.float32)
    for i in range(0, V, CHUNK):
        sel = (hud >= i) & (hud < i + CHUNK)
        if sel.any():
            Hraw[:, sel] = flat[:, i:i + CHUNK][:, hud[sel] - i]
    Hstd = _std_cols(Hraw)
    htr = []
    for v in hvid:
        loc = np.searchsorted(hud, v)
        htr.append(Hstd[:, loc].mean(1))

    # --- diagnosis
    diag = diagnose(top, shape, hvid, hsizes, hyspan)
    diag["n_default_units"] = n_default
    diag["n_human_total"] = len(hm)
    diag["n_human_empty"] = n_empty
    (OUTDIR / f"diagnosis_{a.tag}{a.only}.json").write_text(json.dumps(diag, indent=2))
    print("DIAGNOSIS", json.dumps(diag, indent=2), flush=True)

    # sanity: reproduce the default result with the factored pipeline, and score the REAL labelmap
    idx, tr, pos, nbr, topv = fg_substrate(flat, top, shape, P["FG_Z"])
    units = grow(tr, nbr, topv, P["SEED_Z"], P["GROW_R"])
    lab = merge(units, idx, tr, shape, P["MERGE_R"], P["MERGE_GAP"])
    out_rep, final_rep = finalize(lab, shape, {})
    print(f"[check] factored default -> {len(final_rep)} units (ad.segment={n_default})", flush=True)
    out0 = np.asarray(out0)
    final0 = [np.flatnonzero(out0.ravel() == i) for i in range(1, n_default + 1)]
    sc_real = score_combo(out0, final0, pos, tr, shape, hvid, htr, P["MIN_VOX"])
    print("[check] REAL default labelmap score (approx):", sc_real, flush=True)
    sc_fact = score_combo(out_rep, final_rep, pos, tr, shape, hvid, htr, P["MIN_VOX"])
    print("[check] factored default score (approx):", sc_fact, flush=True)
    del idx, tr, pos, nbr, topv, units, lab, out_rep, final_rep

    if not a.grid:
        (ad.SCRATCH / f"{vkey}_zs.npy").unlink(missing_ok=True)
        return

    # --- grid
    rows = []
    rows.append(dict(FG_Z="default(real)", SEED_Z=P["SEED_Z"], GROW_R=P["GROW_R"], MIN_VOX=P["MIN_VOX"],
                     source="ad.segment", **sc_real))
    for fg in a.fgz:
        nfg = int((top > fg).sum())
        if nfg * T * 4 > 7.5e9:                      # trace matrix would exceed the RAM budget
            print(f"[grid] FG_Z={fg}: {nfg} fg vox would need {nfg*T*4/1e9:.1f} GB - SKIPPED", flush=True)
            rows.append(dict(FG_Z=fg, SEED_Z=None, GROW_R=None, MIN_VOX=None, source="skipped_oom",
                             n_units=None, human_recall=None, n_recovered=None, median_best_r=None,
                             median_split_half=None, median_unit_vox=None))
            continue
        tb = time.time()
        idx, tr, pos, nbr, topv = fg_substrate(flat, top, shape, fg)
        print(f"[grid] FG_Z={fg}: {idx.size} fg vox, substrate {time.time()-tb:.0f}s", flush=True)
        for seed in a.seedz:
            for gr in a.growr:
                tg = time.time()
                units = grow(tr, nbr, topv, seed, gr)
                lab = merge(units, idx, tr, shape, P["MERGE_R"], P["MERGE_GAP"])
                mv_res = []
                for mv in a.minvox:
                    out, final = finalize(lab, shape, {"MIN_VOX": mv})
                    sc = score_combo(out, final, pos, tr, shape, hvid, htr, mv)
                    rows.append(dict(FG_Z=fg, SEED_Z=seed, GROW_R=gr, MIN_VOX=mv, source="factored", **sc))
                    mv_res.append((mv, sc["n_units"], sc["human_recall"]))
                print(f"[grid] FG={fg} SEED={seed} GROW={gr}: raw_units={len(units)} "
                      f"(mv,n,recall)={mv_res} ({time.time()-tg:.0f}s)", flush=True)
        del idx, tr, pos, nbr, topv
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / f"grid_{a.tag}{a.only}.csv", index=False)
    print("WROTE", OUTDIR / f"grid_{a.tag}{a.only}.csv", flush=True)
    print(df.to_string(), flush=True)
    # scratch zs volume is kept for reuse; cleaned up manually at end of the tuning task.


if __name__ == "__main__":
    main()
