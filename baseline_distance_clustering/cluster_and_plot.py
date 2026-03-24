#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_and_plot.py
DBSCAN distance-based clustering of particles (nucleosome / ribosome / any).
Compares predicted clusters (different ε cutoffs) against ground-truth 'class' labels.

Usage
-----
1. Add a new entry to CONFIGS below.
2. Set ACTIVE to the key of that entry.
3. Run the script.
"""

import os
import numpy as np
import pandas as pd
import starfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

# =============================================================================
#  CONFIGS  –  add new datasets here, then set ACTIVE to the desired key
# =============================================================================
CONFIGS = {
    "nucleosome_R1": dict(
        star_file    = "R1/R1_ID_Manual_1.star",
        pixel_size_a = 8.0,                         # Å/pixel
        eps_values   = [12, 15, 18, 21],             # nm, plotted panels
        save_eps     = 15,                           # nm, saved to STAR
        gt_col       = "class",                      # ground-truth column
        min_samples  = 1,
        eps_sweep    = (2.0, 50.5, 0.5),             # (start, stop, step) nm
    ),
    "nucleosome_R2": dict(
        star_file    = "R2/R2_ID_Manual_1.star",
        pixel_size_a = 8.0,
        eps_values   = [12, 15, 18, 21],
        save_eps     = 13,
        gt_col       = "class",
        min_samples  = 1,
        eps_sweep    = (2.0, 50.5, 0.5),
    ),
    "ribosome": dict(
        star_file    = "IDname_PolysomeManual_1.star",
        pixel_size_a = 1.96,
        eps_values   = [25, 29, 33],
        save_eps     = 29,
        gt_col       = "class",
        min_samples  = 1,
        eps_sweep    = (5.0, 80.0, 1.0),
    ),
}

# ← 只改这一行来切换数据集
ACTIVE = "nucleosome_R2"

# =============================================================================
#  Unpack active config
# =============================================================================
cfg          = CONFIGS[ACTIVE]
STAR_FILE    = cfg["star_file"]
PIXEL_SIZE_A = cfg["pixel_size_a"]
NM_PER_PX    = PIXEL_SIZE_A / 10.0
EPS_VALUES   = cfg["eps_values"]
SAVE_EPS     = cfg["save_eps"]
GT_COL       = cfg["gt_col"]
MIN_SAMPLES  = cfg["min_samples"]
EPS_SWEEP    = np.arange(*cfg["eps_sweep"])

# Output filenames are derived from ACTIVE name → no conflicts between datasets
_star_stem      = os.path.splitext(os.path.basename(STAR_FILE))[0]
OUTPUT_PNG      = f"clustering_comparison_{ACTIVE}.png"
OUTPUT_ARI_PNG  = f"ARI_vs_eps_{ACTIVE}.png"
OUTPUT_STAR_CLUST = f"{_star_stem}_clustered_eps{SAVE_EPS}nm.star"

# =============================================================================
#  Load data
# =============================================================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

raw = starfile.read(STAR_FILE, always_dict=True)
df  = next(iter(raw.values()))

xyz = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(dtype=float) * NM_PER_PX
gt  = df[GT_COL].to_numpy(dtype=int)

# =============================================================================
#  Run DBSCAN for each eps in EPS_VALUES
# =============================================================================
results = []
for eps in EPS_VALUES:
    labels  = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="euclidean").fit_predict(xyz)
    n_clust = len(set(labels) - {-1})
    ari     = adjusted_rand_score(gt, labels)
    results.append((eps, labels, n_clust, ari))

# =============================================================================
#  Panel plot (Ground Truth + one panel per eps)
# =============================================================================
def make_cmap(labels):
    uniq    = sorted(set(labels))
    palette = plt.cm.tab10.colors
    cmap    = {}
    col_idx = 0
    for u in uniq:
        if u == -1:
            cmap[u] = (0.6, 0.6, 0.6)
        else:
            cmap[u] = palette[col_idx % len(palette)]
            col_idx += 1
    return cmap

n_cols = 1 + len(EPS_VALUES)
fig    = plt.figure(figsize=(4 * n_cols, 5))
fig.suptitle(f"DBSCAN Clustering vs Ground Truth\n[{ACTIVE}]  pixel size = {PIXEL_SIZE_A} Å",
             fontsize=12, y=1.02)

x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

def scatter3d(ax, labels, title):
    cmap   = make_cmap(labels)
    colors = [cmap[l] for l in labels]
    ax.scatter(x, y, z, c=colors, s=35, depthshade=True, edgecolors="none")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X (nm)", fontsize=7)
    ax.set_ylabel("Y (nm)", fontsize=7)
    ax.set_zlabel("Z (nm)", fontsize=7)
    ax.tick_params(labelsize=6)

ax0 = fig.add_subplot(1, n_cols, 1, projection="3d")
scatter3d(ax0, gt, f"Ground Truth\n(n={len(set(gt))} classes)")

for k, (eps, labels, n_clust, ari) in enumerate(results):
    ax = fig.add_subplot(1, n_cols, k + 2, projection="3d")
    scatter3d(ax, labels, f"ε = {eps} nm\nn_clusters={n_clust}, ARI={ari:.2f}")

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"[OK] Saved: {OUTPUT_PNG}")

# =============================================================================
#  Save chosen eps result as annotated STAR
# =============================================================================
saved = False
for eps, labels, n_clust, ari in results:
    if abs(eps - SAVE_EPS) < 1e-6:
        df_out = df.copy()
        df_out["rlnClusterLabel"] = labels.astype(int)
        starfile.write({"particles": df_out}, OUTPUT_STAR_CLUST, overwrite=True)
        print(f"[OK] Saved clustered STAR (eps={eps}nm, n_clusters={n_clust}, ARI={ari:.2f}): {OUTPUT_STAR_CLUST}")
        saved = True
        break

if not saved:
    print(f"[WARN] save_eps={SAVE_EPS}nm not found in eps_values={EPS_VALUES}; no STAR saved.")

# =============================================================================
#  ARI vs ε sweep
# =============================================================================
ari_vals, nclust_vals = [], []
for eps_s in EPS_SWEEP:
    lbl = DBSCAN(eps=eps_s, min_samples=MIN_SAMPLES, metric="euclidean").fit_predict(xyz)
    ari_vals.append(adjusted_rand_score(gt, lbl))
    nclust_vals.append(len(set(lbl) - {-1}))

ari_arr    = np.array(ari_vals)
nclust_arr = np.array(nclust_vals)
best_idx   = int(np.argmax(ari_arr))
best_eps   = EPS_SWEEP[best_idx]
best_ari   = ari_arr[best_idx]

fig2, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()

ax1.plot(EPS_SWEEP, ari_arr,    color="steelblue", lw=2,   label="ARI")
ax2.plot(EPS_SWEEP, nclust_arr, color="tomato",    lw=1.5, ls="--", label="n_clusters")

ax1.axvline(best_eps, color="gray", lw=1, ls=":")
ax1.annotate(f"peak ε={best_eps:.1f} nm\nARI={best_ari:.3f}",
             xy=(best_eps, best_ari), xytext=(best_eps + 2, best_ari - 0.08),
             fontsize=8, color="steelblue",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

ax1.set_xlabel("ε (nm)", fontsize=11)
ax1.set_ylabel("ARI (vs ground truth)", fontsize=11, color="steelblue")
ax2.set_ylabel("n_clusters", fontsize=11, color="tomato")
ax1.set_title(f"DBSCAN: ARI and cluster count vs ε   [{ACTIVE}]\n"
              f"(min_samples={MIN_SAMPLES}, pixel size={PIXEL_SIZE_A} Å)", fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2.tick_params(axis="y", labelcolor="tomato")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

fig2.tight_layout()
fig2.savefig(OUTPUT_ARI_PNG, dpi=150, bbox_inches="tight")
print(f"[OK] Saved: {OUTPUT_ARI_PNG}  (peak ARI={best_ari:.3f} at ε={best_eps:.1f} nm)")
