#!/usr/bin/env python
"""
Quick-exploratory tier (analysis_plan.md roadmap items 1-3), per shared-mask-group FOV:
  - participation-fraction distribution (pooled) -> 2-comp GMM -> data-driven local/global cutoff
  - per-FOV diagnostics: mean inter-mask r (post-subtraction), effective dim/N, frac local/global,
    PC1 vs accel/pupil, compartmentalization index (1 - mean residual-after-global r),
    contamination flag (high mean_r AND PC1 decoupled from behavior).
"""
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy import stats

PR = Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
spec = importlib.util.spec_from_file_location("eb", PR/"code/Extra/event_typing_behavior.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
FS, THR = eb.FS, eb.DFF_THR

FOVS = [("2026-04-16","rbp4_132_phpeb","run1","0416_run123"),
        ("2026-04-16","rbp4_132_phpeb","run7","0416_run567"),
        ("2026-03-31","rbp4_132_phpeb","run7","0331_run67"),
        ("2026-03-31","rbp4_132_phpeb","run8","0331_run8910"),
        ("2026-05-08","rbp4_139_phpeb","run5","0508_run56"),
        ("2026-05-12","rbp4_132_phpeb","run5","0512_run56"),
        ("2026-05-12","rbp4_132_phpeb","run9","0512_run910")]

def _r(a, b):
    if b is None: return np.nan
    m = np.isfinite(a) & np.isfinite(b)
    return stats.pearsonr(a[m], b[m])[0] if m.sum() > 4 and np.ptp(a[m]) > 0 else np.nan

def load_fov(date, mouse, run):
    base = PR/"scape-data"/date/mouse/run
    df = pd.read_csv(base/"traces"/"dff_traces_curated_bgsub.csv")
    if 'Frame' in df.columns: df = df.drop(columns=['Frame'])
    X = df[[c for c in df.columns if 'dend' in c]].values.astype(float)
    accel, pupil = eb.load_behavior(base, run, len(X))
    return X, accel, pupil

def fov_metrics(X, accel, pupil):
    T, N = X.shape
    active = X > THR; acount = active.sum(1)
    cm = np.corrcoef(X.T); iu = np.triu_indices(N, 1)
    mean_r = cm[iu].mean()
    # PCA
    sd = X.std(0); sd[sd == 0] = 1; Z = (X - X.mean(0))/sd
    ev = PCA().fit(Z).explained_variance_ratio_
    eff = 1.0/np.sum(ev**2)
    pc1 = PCA(2).fit_transform(Z)[:, 0]
    # global regression -> residual correlation (compartmentalization)
    g = np.median(X, 1); beta = (X*g[:,None]).mean(0)/ (g.var()+1e-9)
    resid = X - np.outer(g, beta)
    rr = np.corrcoef(resid.T)[iu]; mean_r_resid = np.nanmean(rr)
    # participation per event
    fracs, counts = [], []
    for j in range(N):
        for p in eb.events(X[:, j]):
            c = acount[max(0,p-1):p+2].max()
            fracs.append(c/N); counts.append(c)
    return dict(N=N, mean_r=mean_r, eff_dim=eff, eff_dim_frac=eff/N,
                mean_r_resid=mean_r_resid, compart_index=1-mean_r_resid,
                r_PC1_accel=_r(pc1, accel), r_PC1_pupil=_r(pc1, pupil)), np.array(fracs), np.array(counts)

# pass 1: collect
rows, allf = [], []
for d, mo, rn, lab in FOVS:
    X, ac, pu = load_fov(d, mo, rn)
    m, fr, ct = fov_metrics(X, ac, pu)
    m.update(fov=lab, run=f"{d}/{rn}"); m['_fr'] = fr; m['_ct'] = ct
    rows.append(m); allf.append(fr)

# data-driven cutoff: 2-comp GMM on pooled participation fractions
pooled = np.concatenate(allf).reshape(-1, 1)
gm = GaussianMixture(2, random_state=0).fit(pooled)
mu = np.sort(gm.means_.ravel())
grid = np.linspace(mu[0], mu[1], 200).reshape(-1, 1)
cutoff = float(grid[np.argmin(gm.score_samples(grid))][0])   # antimode
print(f"GMM modes at participation = {mu[0]:.2f}, {mu[1]:.2f}  -> data-driven global cutoff = {cutoff:.2f}")

# pass 2: classify events per FOV with shared cutoff
for m in rows:
    fr, ct = m.pop('_fr'), m.pop('_ct')
    n = len(fr)
    m['frac_global'] = float(np.mean(fr >= cutoff)) if n else np.nan
    m['frac_local']  = float(np.mean(ct <= 2)) if n else np.nan
    m['n_events'] = n
    m['contam_flag'] = bool(m['mean_r'] > 0.3 and max(abs(np.nan_to_num(m['r_PC1_accel'])),
                                                       abs(np.nan_to_num(m['r_PC1_pupil']))) < 0.2)

S = pd.DataFrame(rows)[['fov','run','N','n_events','mean_r','eff_dim','eff_dim_frac',
                        'frac_local','frac_global','compart_index','r_PC1_accel','r_PC1_pupil','contam_flag']]
pd.set_option('display.width', 240, 'display.max_columns', 30)
print("\n===== PER-FOV DIAGNOSTIC PANEL =====")
print(S.round(3).to_string(index=False))
S.to_csv(PR/"scape-data"/"quick_explore_fov_panel.csv", index=False)

# plot pooled participation histogram + GMM + cutoff
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(pooled.ravel(), bins=40, density=True, color='lightgray', edgecolor='k', alpha=.7)
xs = np.linspace(0,1,300).reshape(-1,1)
ax.plot(xs, np.exp(gm.score_samples(xs)), 'b', label='GMM')
ax.axvline(cutoff, color='r', ls='--', label=f'global cutoff={cutoff:.2f}')
ax.set_xlabel('event participation fraction (#masks co-active / N)'); ax.set_ylabel('density')
ax.set_title('Pooled participation fractions -> data-driven event-class cutoff'); ax.legend()
fig.tight_layout(); fig.savefig(PR/"scape-data"/"quick_explore_participation.png", dpi=150)
print("\nsaved: scape-data/quick_explore_participation.png, quick_explore_fov_panel.csv")
