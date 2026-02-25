import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from skimage.filters import threshold_otsu

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config_plot import CSV_PATH, P_THRESHOLD_MAP

# ============================================================
# 1. Configuration & Data Loading
# ============================================================
THETA_COL = "theta_deg"
L_COL = "L_nm"

df = pd.read_csv(CSV_PATH)
df_sub = df[df["P"] > P_THRESHOLD_MAP].copy()
df_sub = df_sub.dropna(subset=[THETA_COL, L_COL])

x = df_sub[L_COL].values
y = df_sub[THETA_COL].values

print(f"[INFO] Loaded {len(x)} points from CSV")
if len(x) < 2:
    print("[ERROR] Not enough points.")
    exit()

# ============================================================
# 2. Kernel Density Estimation (KDE) Computation
# ============================================================
positions = np.vstack([x, y])
kde = gaussian_kde(positions)
z_points = kde(positions)

# Generate Grid for Background
x_min, x_max = max(0, x.min() - 5), max(x.max() + 10,60)
y_min, y_max = max(0, y.min() - 20), min(180, y.max() + 200)
x_grid, y_grid = np.mgrid[x_min:x_max:150j, y_min:y_max:150j]
grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
z_grid = np.reshape(kde(grid_positions), x_grid.shape)

# ============================================================
# 3. Four Methods for Thresholding / Clustering
# ============================================================

# --- Method 1: Empirical Percentile (Top 30%) ---
DENSITY_PERCENTILE = 0.70
sorted_z = np.sort(z_points)
idx_70 = int(len(sorted_z) * DENSITY_PERCENTILE)
thresh_empirical = sorted_z[idx_70]
mask_empirical = z_points >= thresh_empirical

# --- Method 2: Elbow Method (Knee Locator) ---
# Find max curvature in the sorted density curve
x_kneed = np.arange(len(sorted_z))
kneedle = KneeLocator(x_kneed, sorted_z, S=1.0, curve="convex", direction="increasing")
knee_idx = kneedle.knee
thresh_elbow = sorted_z[knee_idx] if knee_idx is not None else thresh_empirical
mask_elbow = z_points >= thresh_elbow

# --- Method 3: Otsu's Method ---
# Bimodal histogram thresholding
thresh_otsu = threshold_otsu(z_points)
mask_otsu = z_points >= thresh_otsu

# --- Method 4: DBSCAN Clustering ---
# Directly cluster the (x, y) coordinates. We standardize first for distance metrics.
# eps represents neighborhood distance, min_samples requires X neighbors to be a core point
x_norm = (x - x.mean()) / x.std()
y_norm = (y - y.mean()) / y.std()
X_norm = np.column_stack((x_norm, y_norm))
# Parameters can be tuned; eps=0.3 is a standard start for normalized data
db = DBSCAN(eps=0.35, min_samples=5).fit(X_norm)
# mask_dbscan: True if point belongs to ANY cluster (label >= 0), False if Noise (label == -1)
mask_dbscan = db.labels_ >= 0

# ============================================================
# 4. Plotting 2x2 Subplots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
methods = [
    ("Empirical (Top 30%)", thresh_empirical, mask_empirical, axes[0, 0]),
    ("Elbow / Knee Method", thresh_elbow, mask_elbow, axes[0, 1]),
    ("Otsu's Thresholding", thresh_otsu, mask_otsu, axes[1, 0]),
    ("DBSCAN Clustering", None, mask_dbscan, axes[1, 1])
]

for title, thresh, mask, ax in methods:
    # Background Map
    cf = ax.contourf(x_grid, y_grid, z_grid, levels=20, cmap="Blues", alpha=0.6)
    
    # Contour line (Only for threshold-based methods)
    if thresh is not None:
        cs = ax.contour(x_grid, y_grid, z_grid, levels=[thresh], colors='red', linewidths=2, linestyles="--")
        ax.clabel(cs, fmt=f"{thresh:.5f}", inline=True, fontsize=10, colors='red')

    # Scatter Noise and Pattern
    df_noise = df_sub[~mask]
    df_pattern = df_sub[mask]
    
    ax.scatter(df_noise[L_COL], df_noise[THETA_COL], s=8, color='grey', alpha=0.4, label='Noise')
    ax.scatter(df_pattern[L_COL], df_pattern[THETA_COL], s=20, color='red', alpha=0.9, edgecolor='black', linewidth=0.5, label='Pattern')
    
    ax.set_title(f"{title}\nPatterns: {np.sum(mask)} | Noise: {np.sum(~mask)}", fontsize=11)
    ax.set_xlabel("Length L (nm)")
    ax.set_ylabel("Theta (deg)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Shared Legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_png = "theta_vs_L_4methods_comparison.png"
fig.savefig(out_png, dpi=300)
plt.show()

print(f"✅ 4-way comparison map saved as '{out_png}'")
