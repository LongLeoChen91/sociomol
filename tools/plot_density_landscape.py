"""
plot_density_landscape.py
-------------------------
KDE density landscape with 5-panel (2x3) layout, matching the
signal-detection masks in estimate_effective_Lp.py for direct comparison.

For each of the 5 masks:
  - Background: KDE contourf (Blues)
  - Red dashed contour: the mask threshold boundary
  - Grey scatter: noise / excluded points
  - Red scatter:  signal / included points
  - Title:  method name + N_signal / N_total

Masks (identical to estimate_effective_Lp.py):
  1. Empirical (Top 30%)
  2. Elbow / Knee Method
  3. Otsu's Thresholding
  4. DBSCAN Clustering
  5. All Points (Baseline)

Output: density_landscape_5methods.png

Author: Long Chen
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from skimage.filters import threshold_otsu

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_SCRIPT_DIR)

from config_plot import CSV_PATH, P_THRESHOLD_MAP

# ============================================================
# 1. Configuration
# ============================================================
THETA_COL = "theta_deg"
L_COL     = "L_nm"

GRID_RES  = 150    # KDE grid resolution per axis

# ============================================================
# 2. Data Loading & Filtering
# ============================================================
df     = pd.read_csv(CSV_PATH)
df_sub = df[df["P"] > P_THRESHOLD_MAP].copy()
df_sub = df_sub.dropna(subset=[THETA_COL, L_COL])

x = df_sub[L_COL].values
y = df_sub[THETA_COL].values

print(f"[INFO] Loaded {len(x)} candidate linker points")
if len(x) < 10:
    print("[ERROR] Not enough points.")
    sys.exit(1)

# ============================================================
# 3. KDE on data points
# ============================================================
positions = np.vstack([x, y])
kde       = gaussian_kde(positions)
z_points  = kde(positions)

# KDE on a regular grid for background
x_min = max(0,   x.min() - 5)
x_max = max(x.max() + 10, 60)
y_min = max(0,   y.min() - 20)
y_max = min(180, y.max() + 200)

x_grid, y_grid = np.mgrid[x_min:x_max:GRID_RES*1j, y_min:y_max:GRID_RES*1j]
z_grid = np.reshape(kde(np.vstack([x_grid.ravel(), y_grid.ravel()])), x_grid.shape)

# ============================================================
# 4. Five Signal Masks  (identical to estimate_effective_Lp.py)
# ============================================================

# Method 1: Empirical Top 30%
DENSITY_PERCENTILE = 0.70
sorted_z         = np.sort(z_points)
idx_70           = min(int(len(sorted_z) * DENSITY_PERCENTILE), len(sorted_z) - 1)
thresh_empirical = sorted_z[idx_70]
mask_empirical   = z_points >= thresh_empirical

# Method 2: Elbow / Knee
x_kneed      = np.arange(len(sorted_z))
kneedle      = KneeLocator(x_kneed, sorted_z, S=1.0, curve="convex", direction="increasing")
knee_idx     = kneedle.knee
thresh_elbow = sorted_z[knee_idx] if knee_idx is not None else thresh_empirical
mask_elbow   = z_points >= thresh_elbow

# Method 3: Otsu
thresh_otsu_val = threshold_otsu(z_points)
mask_otsu       = z_points >= thresh_otsu_val

# Method 4: DBSCAN
x_norm  = (x - x.mean()) / x.std()
y_norm  = (y - y.mean()) / y.std()
db      = DBSCAN(eps=0.35, min_samples=5).fit(np.column_stack((x_norm, y_norm)))
mask_dbscan = db.labels_ >= 0

# Method 5: All Points
mask_all = np.ones(len(x), dtype=bool)

methods = [
    ("Empirical (Top 30%)", mask_empirical, thresh_empirical),
    ("Elbow / Knee Method", mask_elbow,     thresh_elbow),
    ("Otsu's Thresholding", mask_otsu,      thresh_otsu_val),
    ("DBSCAN Clustering",   mask_dbscan,    None),   # DBSCAN has no KDE threshold
    ("All Points (Baseline)", mask_all,     None),
]

# ============================================================
# 5. Plot 2x3 grid
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "KDE Density Landscape — Signal Detection Comparison\n"
    f"(data source: {os.path.basename(CSV_PATH)},  N_total={len(x)})",
    fontsize=13, y=1.01
)

for ax, (title, mask, thresh_kde) in zip(axes.flat, methods):
    n_signal = np.sum(mask)
    n_total  = len(x)

    # Background: KDE density field
    cf = ax.contourf(x_grid, y_grid, z_grid, levels=20, cmap="Blues", alpha=0.70)

    # Threshold contour (only for KDE-based methods)
    if thresh_kde is not None:
        cs = ax.contour(x_grid, y_grid, z_grid,
                        levels=[thresh_kde], colors="red",
                        linewidths=1.8, linestyles="--")
        ax.clabel(cs, fmt=f"thr={thresh_kde:.4f}", inline=True,
                  fontsize=7, colors="red")

    # Scatter: noise grey, signal red
    ax.scatter(x[~mask], y[~mask],
               s=8, color="grey", alpha=0.35, edgecolors="none",
               label="Noise" if np.any(~mask) else "")
    ax.scatter(x[mask], y[mask],
               s=18, color="red", alpha=0.85,
               edgecolors="black", linewidths=0.4,
               label=f"Signal (N={n_signal})")

    ax.set_title(f"{title}\nN_signal={n_signal} / {n_total}", fontsize=10)
    ax.set_xlabel("L (nm)", fontsize=11)
    ax.set_ylabel(r"$\theta$ (deg)", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print(f"[{title}]  N_signal={n_signal:4d} / {n_total}")

# Hide unused 6th panel
axes.flat[-1].set_visible(False)

plt.tight_layout()
out_png = "density_landscape_5methods.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.show()
print(f"\n[OK] Saved: {out_png}")
