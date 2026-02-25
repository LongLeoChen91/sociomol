import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import gaussian_kde

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config_plot import CSV_PATH, P_THRESHOLD_MAP

# ============================================================
# 1. Configuration
# ============================================================

THETA_COL = "theta_deg"
L_COL = "L_nm"

# Density Threshold Percentile
# e.g., 0.70 means taking the top 30% densest points as the "pattern"
DENSITY_PERCENTILE_THRESH = 0.70  

# ============================================================
# 2. Data Loading
# ============================================================
df = pd.read_csv(CSV_PATH)

if THETA_COL not in df.columns or L_COL not in df.columns or "P" not in df.columns:
    raise KeyError(f"Missing required columns in CSV. Available: {list(df.columns)}")

# Filter based on global threshold
df_sub = df[df["P"] > P_THRESHOLD_MAP].copy()

# Drop rows with NaNs
df_sub = df_sub.dropna(subset=[THETA_COL, L_COL])

x = df_sub[L_COL].values
y = df_sub[THETA_COL].values

print(f"[INFO] Loaded {len(x)} points from CSV")

if len(x) < 2:
    print("[ERROR] Not enough points to calculate density.")
    exit()

# ============================================================
# 3. Kernel Density Estimation (KDE) Computation
# ============================================================
# Combine x and y into a 2xN array
positions = np.vstack([x, y])

# Train the Gaussian KDE model
kde = gaussian_kde(positions)

# Calculate the density (Z) for every actual data point
z_points = kde(positions)

# ============================================================
# 4. Pattern vs Noise Classification 
# ============================================================
# Sort densities to find the threshold
sorted_z = np.sort(z_points)
threshold_idx = int(len(sorted_z) * DENSITY_PERCENTILE_THRESH)

# Make sure we don't go out of bounds
if threshold_idx >= len(sorted_z):
    threshold_idx = len(sorted_z) - 1

density_threshold = sorted_z[threshold_idx]
print(f"[INFO] Top {int((1-DENSITY_PERCENTILE_THRESH)*100)}% Density Threshold: {density_threshold:.6f}")

# Mask to classify points
pattern_mask = z_points >= density_threshold
df_pattern = df_sub[pattern_mask]
df_noise = df_sub[~pattern_mask]

print(f"[INFO] Classified {len(df_pattern)} Pattern points, {len(df_noise)} Noise points.")

# ============================================================
# 5. Plotting
# ============================================================
# Generate a regular grid to plot the density background and contours
x_min, x_max = max(0, x.min() - 5), x.max() + 10
y_min, y_max = max(0, y.min() - 20), min(180, y.max() + 20)

x_grid, y_grid = np.mgrid[x_min:x_max:150j, y_min:y_max:150j]
grid_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

# Evaluate KDE on the grid
z_grid = np.reshape(kde(grid_positions), x_grid.shape)

fig, ax = plt.subplots(figsize=(8.5, 6))

# a) Background heatmap of density
cf = ax.contourf(
    x_grid, y_grid, z_grid, 
    levels=20, 
    cmap="Blues", 
    alpha=0.7
)
cbar = fig.colorbar(cf, ax=ax, label="Estimated Point Density")

# b) Draw the specific contour line for our threshold
# Use red to highlight the boundary of the "Pattern"
cs = ax.contour(
    x_grid, y_grid, z_grid,
    levels=[density_threshold],
    colors='red',
    linewidths=2,
    linestyles="--"
)
ax.clabel(cs, fmt=f"Thr={density_threshold:.4f}", inline=True, fontsize=10, colors='red')

# c) Scatter plot the data points, coloring them by classification
ax.scatter(
    df_noise[L_COL], df_noise[THETA_COL],
    s=10, color='grey', alpha=0.5, edgecolor='none',
    label='Noise (Low Density)'
)

ax.scatter(
    df_pattern[L_COL], df_pattern[THETA_COL],
    s=25, color='red', alpha=0.9, edgecolor='black', linewidth=0.5,
    label='Pattern (High Density)'
)

# Aesthetics
ax.set_xlabel("Linker length L (nm)", fontsize=13)
ax.set_ylabel("Bending angle θ (degrees)", fontsize=13)
ax.set_title(
    "Data-Driven Pattern Density Landscape\n"
    f"KDE Threshold (Top {int((1-DENSITY_PERCENTILE_THRESH)*100)}%): {density_threshold:.6f}",
    fontsize=12,
    pad=12
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right")

fig.tight_layout()

out_png = "theta_vs_L_data_density_landscape.png"
fig.savefig(out_png, dpi=300)
plt.show()

print(f"✅ Density Landscape Figure saved as '{out_png}'")
