"""
Side-by-side comparison: theta0=45 vs theta0=20 contour lines.
This script proves whether changing theta0_deg affects the contour shape.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

res = 400
D_vals = np.linspace(1.0, 40.0, res)
t_vals_rad = np.linspace(0.0, np.pi, res)
t_vals_deg = np.degrees(t_vals_rad)

D_grid, theta_grid_rad = np.meshgrid(D_vals, t_vals_rad)
_, theta_grid_deg = np.meshgrid(D_vals, t_vals_deg)

# Convert D to arc length L
half_t = 0.5 * theta_grid_rad
sin_half = np.sin(half_t)
denom = 2.0 * sin_half
L_grid = np.where(np.abs(denom) < 1e-8, D_grid, (theta_grid_rad * D_grid) / np.maximum(denom, 1e-12))

L0 = 20.0

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, theta0_deg in zip(axes, [45.0, 20.0]):
    theta0_rad = np.radians(theta0_deg)
    
    E_len = L_grid / L0
    E_ang = (0.5 * theta_grid_rad) / theta0_rad
    E_total = E_len + E_ang
    P_grid = np.exp(-E_total)
    
    cf = ax.contourf(D_grid, theta_grid_deg, P_grid,
                     levels=np.linspace(0, 1, 101), cmap="YlGnBu", alpha=0.6)
    
    cs = ax.contour(D_grid, theta_grid_deg, P_grid,
                    levels=[0.01, 0.05, 0.2, 0.5],
                    colors=["red", "orange", "magenta", "black"],
                    linewidths=[2.5, 2.0, 1.5, 1.5],
                    linestyles=["-", "--", "-.", ":"])
    ax.clabel(cs, fmt={thr: f"P={thr}" for thr in [0.01, 0.05, 0.2, 0.5]},
              inline=True, fontsize=11)
    
    ax.set_title(f"theta0_deg = {theta0_deg}", fontsize=14, fontweight='bold')
    ax.set_xlabel("D (nm)", fontsize=12)
    ax.set_ylabel("theta (degrees)", fontsize=12)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 180)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.colorbar(cf, ax=axes, label="P(D, theta)")
plt.suptitle("Contour Comparison: theta0=45 vs theta0=20\n(w_wlc=0, w_L=1, w_th=1, L0=20)", 
             fontsize=13, y=1.02)
plt.tight_layout()

out_path = os.path.join(_SCRIPT_DIR, "outputs", "debug_contour_compare.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"[OK] Comparison saved to {out_path}")
