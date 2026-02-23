import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config_plot import CSV_PATH, LP, L0, THETA0_DEG, W_WLC, W_L, W_TH, CONTOUR_THRESHOLDS

# ============================================================
# 1. Configuration
# ============================================================

THETA_COL = "theta_deg"
L_COL = "L_nm"

# ============================================================
# 2. Data Loading
# ============================================================
df = pd.read_csv(CSV_PATH)

if THETA_COL not in df.columns or L_COL not in df.columns:
    raise KeyError(f"Missing required columns in CSV. Available: {list(df.columns)}")

theta_data = pd.to_numeric(df[THETA_COL], errors="coerce")
L_data = pd.to_numeric(df[L_COL], errors="coerce")

mask = theta_data.notna() & L_data.notna()
theta_data = theta_data[mask]
L_data = L_data[mask]

print(f"[INFO] Loaded {len(theta_data)} points from CSV")

# ============================================================
# 3. Model Computation (Energy / Probability Map)
# ============================================================
L_min, L_max, nL = 1.0, 100.0, 400
t_min, t_max, nT = 0.0, np.pi, 400

L_vals = np.linspace(L_min, L_max, nL)
theta_vals_rad = np.linspace(t_min, t_max, nT)
theta_vals_deg = np.degrees(theta_vals_rad)

L_grid, theta_grid_rad = np.meshgrid(L_vals, theta_vals_rad)
_, theta_grid_deg = np.meshgrid(L_vals, theta_vals_deg)

theta0 = np.deg2rad(THETA0_DEG)
if theta0 <= 0:
    raise ValueError("theta0_deg must be > 0")

E_wlc = (2.0 * LP / L_grid) * (0.5 * theta_grid_rad) ** 2
E_len = (L_grid / L0)
E_ang = (0.5 * theta_grid_rad) / theta0

E_total = W_WLC * E_wlc + W_L * E_len + W_TH * E_ang
P = np.exp(-E_total)

# ============================================================
# 4. Plotting
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 6))

# a) Background probability map
cf = ax.contourf(
    L_grid, theta_grid_deg, P, 
    levels=np.linspace(0, 1, 101), 
    cmap="YlGnBu", 
    alpha=0.8
)
fig.colorbar(cf, ax=ax, label="P(L, θ) (unnormalized score)", ticks=np.linspace(0, 1, 11))

# b) Contour lines for thresholds
colors = ["red", "orange", "magenta", "black", "grey"]
linestyles = ["-", "--", "-.", ":", "-"]
linewidths = [2.5, 2.5, 2.0, 2.0, 2.0]

for i, thr in enumerate(CONTOUR_THRESHOLDS):
    c = colors[i % len(colors)]
    ls = linestyles[i % len(linestyles)]
    lw = linewidths[i % len(linewidths)]
    
    cs = ax.contour(
        L_grid, theta_grid_deg, P,
        levels=[thr],
        colors=c,
        linestyles=ls,
        linewidths=lw
    )
    
    # Optional annotation per contour line
    fmt = {thr: f"P={thr}"}
    ax.clabel(cs, fmt=fmt, inline=True, fontsize=11)

# c) Scatter specific predicted points
ax.scatter(
    L_data,
    theta_data,
    s=25, 
    color='red',
    edgecolor='black',
    linewidth=0.5,
    alpha=0.9,
    label='Predicted Edges',
    zorder=5  # Put it on top of contours
)

# Aesthetics
ax.set_xlabel("Linker length L (nm)", fontsize=13)
ax.set_ylabel("Bending angle θ (degrees)", fontsize=13)
ax.set_title(
    "Prediction Verification vs Linker Probability Map\n"
    f"lp={LP:g} nm, L0={L0:g} nm, θ0={THETA0_DEG:g}° | "
    f"w_wlc={W_WLC:g}, w_L={W_L:g}, w_th={W_TH:g}",
    fontsize=12,
    pad=12
)

# Set limits similar to scatter plot, or bounded by data + contour
max_L_plot = max(60, L_data.max() * 1.1 if len(L_data) > 0 else 60)
ax.set_xlim(0, max_L_plot)
ax.set_ylim(0, 180)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right")

fig.tight_layout()

out_png = "theta_vs_L_energy_landscape.png"
fig.savefig(out_png, dpi=300)
plt.show()

print(f"✅ Energy Landscape Figure saved as '{out_png}'")
