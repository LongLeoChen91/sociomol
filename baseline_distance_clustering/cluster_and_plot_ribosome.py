#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_and_plot.py
DBSCAN distance-based clustering of nucleosome particles.
Compares predicted clusters (different ε cutoffs) against ground-truth _class labels.
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

# ---------- Config ----------
STAR_FILE    = "IDname_PolysomeManual_1.star"
OUTPUT_PNG   = "clustering_comparison_ribosome.png"
PIXEL_SIZE_A = 1.96                          # Å/pixel
NM_PER_PX   = PIXEL_SIZE_A / 10.0          # nm per pixel

# Cutoff distances (nm) to sweep
# EPS_VALUES_NM = [20, 24, 26, 28, 30, 32, 36, 40]
EPS_VALUES_NM = [25, 29, 33]

MIN_SAMPLES = 1     # DBSCAN min_samples; 1 = no noise points

# ---------- Load data ----------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

raw = starfile.read(STAR_FILE, always_dict=True)
df  = next(iter(raw.values()))

# Coordinates in nm
xyz = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(dtype=float) * NM_PER_PX
gt  = df["class"].to_numpy(dtype=int)   # ground-truth labels

# ---------- Run DBSCAN for each eps ----------
results = []  # list of (eps_nm, labels, n_clusters, ari)
for eps in EPS_VALUES_NM:
    model   = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="euclidean")
    labels  = model.fit_predict(xyz)
    # number of real clusters (exclude noise = -1)
    n_clust = len(set(labels) - {-1})
    ari     = adjusted_rand_score(gt, labels)
    results.append((eps, labels, n_clust, ari))

# ---------- Build colour maps ----------
def make_cmap(labels):
    """Map cluster labels to distinct colours; -1 (noise) → grey."""
    uniq = sorted(set(labels))
    palette = plt.cm.tab10.colors
    cmap = {}
    col_idx = 0
    for u in uniq:
        if u == -1:
            cmap[u] = (0.6, 0.6, 0.6)
        else:
            cmap[u] = palette[col_idx % len(palette)]
            col_idx += 1
    return cmap

# ---------- Plot ----------
n_cols  = 1 + len(EPS_VALUES_NM)
fig     = plt.figure(figsize=(4 * n_cols, 5))
fig.suptitle("DBSCAN Clustering vs Ground Truth\n(pixel size = 1.96 Å)", fontsize=13, y=1.02)

x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

def scatter3d(ax, labels, title):
    cmap    = make_cmap(labels)
    colors  = [cmap[l] for l in labels]
    ax.scatter(x, y, z, c=colors, s=35, depthshade=True, edgecolors="none")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X (nm)", fontsize=7)
    ax.set_ylabel("Y (nm)", fontsize=7)
    ax.set_zlabel("Z (nm)", fontsize=7)
    ax.tick_params(labelsize=6)

# Panel 0: ground truth
ax0 = fig.add_subplot(1, n_cols, 1, projection="3d")
scatter3d(ax0, gt, f"Ground Truth\n(n={len(set(gt))} classes)")

# Panels 1…N: DBSCAN results
for k, (eps, labels, n_clust, ari) in enumerate(results):
    ax = fig.add_subplot(1, n_cols, k + 2, projection="3d")
    scatter3d(ax, labels, f"ε = {eps} nm\nn_clusters={n_clust}, ARI={ari:.2f}")

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"[OK] Saved: {OUTPUT_PNG}")

# ---------- Save eps=15nm clustering result as new STAR ----------
SAVE_EPS_NM   = 29
OUTPUT_CLUSTERED_STAR = f"IDname_PolysomeManual_1_clustered_eps{SAVE_EPS_NM}nm.star"

# Find the matching result (eps may be int or float, compare with tolerance)
saved = False
for eps, labels, n_clust, ari in results:
    if abs(eps - SAVE_EPS_NM) < 1e-6:
        df_out = df.copy()
        df_out["rlnClusterLabel"] = labels.astype(int)
        starfile.write({"particles": df_out}, OUTPUT_CLUSTERED_STAR, overwrite=True)
        print(f"[OK] Saved clustered STAR (eps={eps}nm, n_clusters={n_clust}, ARI={ari:.2f}): {OUTPUT_CLUSTERED_STAR}")
        saved = True
        break

if not saved:
    print(f"[WARN] eps={SAVE_EPS_NM}nm not found in EPS_VALUES_NM; no STAR saved.")

# ---------- ARI vs ε sweep ----------
OUTPUT_ARI_PNG = "ARI_vs_eps_ribosome.png"
eps_sweep = np.arange(5.0, 50, 0.5)   # 2 to 50 nm in 0.5 nm steps

ari_vals    = []
nclust_vals = []
for eps_s in eps_sweep:
    lbl = DBSCAN(eps=eps_s, min_samples=MIN_SAMPLES, metric="euclidean").fit_predict(xyz)
    ari_vals.append(adjusted_rand_score(gt, lbl))
    nclust_vals.append(len(set(lbl) - {-1}))

ari_arr    = np.array(ari_vals)
nclust_arr = np.array(nclust_vals)
best_idx   = int(np.argmax(ari_arr))
best_eps   = eps_sweep[best_idx]
best_ari   = ari_arr[best_idx]

fig2, ax1 = plt.subplots(figsize=(4, 4))
ax2 = ax1.twinx()

ax1.plot(eps_sweep, ari_arr,    color="steelblue",  lw=2,   label="ARI")
ax2.plot(eps_sweep, nclust_arr, color="tomato",     lw=1.5, ls="--", label="n_clusters")

ax1.axvline(best_eps, color="gray", lw=1, ls=":")
ax1.annotate(f"peak ε={best_eps:.1f} nm\nARI={best_ari:.3f}",
             xy=(best_eps, best_ari), xytext=(best_eps + 2, best_ari - 0.08),
             fontsize=8, color="steelblue",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

ax1.set_xlabel("ε (nm)", fontsize=11)
ax1.set_ylabel("ARI (vs ground truth)", fontsize=11, color="steelblue")
ax2.set_ylabel("n_clusters", fontsize=11, color="tomato")
ax1.set_title("DBSCAN: ARI and cluster count vs ε\n(min_samples=1, pixel size=1.96 Å)", fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2.tick_params(axis="y", labelcolor="tomato")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

fig2.tight_layout()
fig2.savefig(OUTPUT_ARI_PNG, dpi=150, bbox_inches="tight")
print(f"[OK] Saved: {OUTPUT_ARI_PNG}  (peak ARI={best_ari:.3f} at ε={best_eps:.1f} nm)")

