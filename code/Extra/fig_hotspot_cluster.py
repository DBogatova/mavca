#!/usr/bin/env python
"""Q1 hotspot/sparsity figures + Q2 functional clustering (functional features only).
Tests whether functional clusters differ by morphology and whether they just recover FOV."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
from scipy import stats
PR=Path("/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025")
M=pd.read_csv(PR/"scape-data"/"mask_fingerprint_table.csv"); H=pd.read_csv(PR/"scape-data"/"hotspot_summary.csv")
fovs=list(H.fov); col=dict(zip(fovs,plt.cm.tab10(np.linspace(0,1,len(fovs)))))
# ---- Q1 figure ----
fig,ax=plt.subplots(2,3,figsize=(15,8))
for f in fovs:
    c=np.sort(M[M.fov==f].n_events.values)[::-1]; ax[0,0].plot(np.arange(1,len(c)+1)/len(c),c/c.max(),color=col[f],lw=1,label=f)
ax[0,0].set_xlabel('mask rank fraction'); ax[0,0].set_ylabel('events (norm to max)'); ax[0,0].set_title('A  Event-count rank curves'); ax[0,0].legend(fontsize=6)
for f in fovs:
    c=np.sort(M[M.fov==f].n_events.values); cum=np.cumsum(c)/c.sum(); ax[0,1].plot(np.arange(1,len(c)+1)/len(c),cum,color=col[f],lw=1)
ax[0,1].plot([0,1],[0,1],'k--',lw=.6); ax[0,1].set_xlabel('cumulative mask fraction'); ax[0,1].set_ylabel('cumulative event fraction'); ax[0,1].set_title('B  Lorenz curves (events)')
ax[0,2].bar(range(len(H)),H.gini,color=[col[f] for f in fovs]); ax[0,2].set_xticks(range(len(H))); ax[0,2].set_xticklabels(fovs,rotation=40,ha='right',fontsize=6); ax[0,2].set_ylabel('Gini (event counts)'); ax[0,2].set_title('C  Event-count Gini per FOV')
x=np.arange(len(H)); w=.25
for i,(k,lab) in enumerate([('top5pct','top5%'),('top10pct','top10%'),('top20pct','top20%')]):
    ax[1,0].bar(x+(i-1)*w,H[k]*100,w,label=lab)
ax[1,0].set_xticks(x); ax[1,0].set_xticklabels(fovs,rotation=40,ha='right',fontsize=6); ax[1,0].set_ylabel('% of events'); ax[1,0].set_title('D  Event share by most-active masks'); ax[1,0].legend(fontsize=7)
ax[1,1].bar(x-.2,H.halfstab_spearman,.4,label='1st vs 2nd half'); ax[1,1].bar(x+.2,H.runstab_spearman,.4,label='run vs run')
ax[1,1].axhline(0,color='k',lw=.5); ax[1,1].set_xticks(x); ax[1,1].set_xticklabels(fovs,rotation=40,ha='right',fontsize=6); ax[1,1].set_ylabel("Spearman rho"); ax[1,1].set_title('E  Hotspot rank stability'); ax[1,1].legend(fontsize=7)
ax[1,2].axis('off'); ax[1,2].text(0.0,0.5,f"Events non-uniform:\nGini {H.gini.min():.2f}-{H.gini.max():.2f}\ntop10% = {H.top10pct.min()*100:.0f}-{H.top10pct.max()*100:.0f}% events\nhotspots moderately stable\n(half rho {H.halfstab_spearman.min():.2f}-{H.halfstab_spearman.max():.2f})",fontsize=11,va='center')
fig.suptitle('Q1. Event hotspots / sparsity across FOVs',fontsize=12); fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(PR/"scape-data"/"figures"/"fig8_hotspots.png",dpi=150); plt.close(fig)

# ---- Q2 functional clustering (functional features only) ----
feats=['event_rate','rate_local','rate_global','frac_global','lg_pref','mean_amp','mean_dur','mean_auc','iei_cv','burstiness']
C=M.dropna(subset=feats).copy(); Z=StandardScaler().fit_transform(C[feats].values)
ks=range(2,7); sil=[silhouette_score(Z,KMeans(k,n_init=10,random_state=0).fit_predict(Z)) for k in ks]
K=list(ks)[int(np.argmax(sil))]; lab=KMeans(K,n_init=10,random_state=0).fit_predict(Z); C['cl']=lab
P2=PCA(2).fit_transform(Z)
ami=adjusted_mutual_info_score(C.fov,C.cl)
fig,ax=plt.subplots(2,3,figsize=(15,8))
ax[0,0].plot(list(ks),sil,'o-'); ax[0,0].axvline(K,color='r',ls='--'); ax[0,0].set_xlabel('k'); ax[0,0].set_ylabel('silhouette'); ax[0,0].set_title(f'A  Cluster count (best k={K})')
sc=ax[0,1].scatter(P2[:,0],P2[:,1],c=C.cl,cmap='tab10',s=14); ax[0,1].set_xlabel('PC1'); ax[0,1].set_ylabel('PC2'); ax[0,1].set_title(f'B  Functional clusters (n={len(C)} masks)')
for j,m in enumerate(['depth_um','SNR','n_vox','orient_deg']):
    a=ax.flat[2+j]; grp=[C[C.cl==k][m].dropna() for k in range(K)]; a.boxplot(grp,labels=[str(k) for k in range(K)])
    H_=stats.kruskal(*[g for g in grp if len(g)>1]); a.set_xlabel('cluster'); a.set_ylabel(m); a.set_title(f'{"CDEF"[j]}  {m} by cluster (KW p={H_.pvalue:.2g})')
fig.suptitle(f'Q2. Functional clusters vs morphology & FOV (cluster~FOV adjusted MI={ami:.2f})',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(PR/"scape-data"/"figures"/"fig9_funcclusters.png",dpi=150); plt.close(fig)
print(f"Q2: k={K} silhouette={max(sil):.2f}; n_masks={len(C)}")
print(f"  cluster sizes: {np.bincount(lab).tolist()}")
print(f"  cluster vs FOV adjusted MI = {ami:.2f}  (0=independent of FOV, 1=clusters==FOV)")
for m in ['depth_um','SNR','n_vox','orient_deg']:
    grp=[C[C.cl==k][m].dropna() for k in range(K)]; print(f"  {m} by cluster Kruskal p={stats.kruskal(*[g for g in grp if len(g)>1]).pvalue:.3g}")
# cluster functional means
print("\ncluster functional means:"); print(C.groupby('cl')[feats].mean().round(2).to_string())
print("saved fig8_hotspots.png, fig9_funcclusters.png")
