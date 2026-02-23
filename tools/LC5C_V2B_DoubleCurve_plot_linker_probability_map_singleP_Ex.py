import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Target model:
# P(L,θ) ∝ exp[
#   - w_wlc * (2 lp / L) * (θ/2)^2
#   - w_L   * (L / L0)
#   - w_th  * ( (θ/2) / θ0 )
# ]
# ============================================================

# -------------------------
# User-tunable parameters
# -------------------------
lp = 1.5
L0 = 20
theta0_deg = 45

P_thresholds = [0.05, 0.4]

w_wlc = 1.0
w_L   = 1.0
w_th  = 1.0

# -------------------------
# Ranges / grids
# -------------------------
L_min, L_max, nL = 1.0, 100.0, 400
t_min, t_max, nT = 0.0, np.pi, 400

L_vals = np.linspace(L_min, L_max, nL)
theta_vals_rad = np.linspace(t_min, t_max, nT)
theta_vals_deg = np.degrees(theta_vals_rad)

L_grid, theta_grid_rad = np.meshgrid(L_vals, theta_vals_rad)
_, theta_grid_deg = np.meshgrid(L_vals, theta_vals_deg)

# -------------------------
# Model terms (ENERGY)
# -------------------------
theta0 = np.deg2rad(theta0_deg)
if theta0 <= 0:
    raise ValueError("theta0_deg must be > 0")

E_wlc = (2.0 * lp / L_grid) * (0.5 * theta_grid_rad) ** 2
E_len = (L_grid / L0)
E_ang = (0.5 * theta_grid_rad) / theta0

E_total = w_wlc * E_wlc + w_L * E_len + w_th * E_ang
P = np.exp(-E_total)

# -------------------------
# Helper
# -------------------------
def max_theta_and_L_for_threshold(P, theta_grid_deg, L_grid, thr):
    mask = P > thr
    if not np.any(mask):
        return float("nan"), float("nan")
    allowed_theta_deg = theta_grid_deg[mask]
    max_theta_deg = float(np.max(allowed_theta_deg))
    idx = int(np.argmax(allowed_theta_deg))
    L_at_max_theta = float(L_grid[mask][idx])
    return max_theta_deg, L_at_max_theta

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(6.8, 5.6))

cf = ax.contourf(L_grid, theta_grid_deg, P, levels=100, cmap="YlGnBu")
fig.colorbar(cf, ax=ax, label="P(L, θ) (unnormalized score)")

colors = ["red", "orange"]
linestyles = ["-", "--"]
linewidths = [2.5, 2.5]

# Move these to reposition the annotation boxes (axes coords: 0..1)
box_positions = [(0.5, 0.93), (0.5, 0.70)]

# Move these to reposition the labels on the contour lines (data coords: nm, deg)
# Tip: choose points near the segment you want to label.
label_positions = [(25, 90), (12, 55)]

for thr, c, ls, lw, (bx, by), (lx, ly) in zip(
    P_thresholds, colors, linestyles, linewidths, box_positions, label_positions
):
    cs = ax.contour(
        L_grid, theta_grid_deg, P,
        levels=[thr],
        colors=c,
        linestyles=ls,
        linewidths=lw
    )

    # Label on the contour line
    texts = ax.clabel(
        cs,
        fmt={thr: f"P={thr}"},
        inline=True,
        fontsize=11,
        manual=[(lx, ly)]
    )

    # Force horizontal labels (optional but often nicer)
    for t in texts:
        t.set_rotation(0)
        t.set_rotation_mode("anchor")

    max_theta_deg, L_at_max_theta = max_theta_and_L_for_threshold(P, theta_grid_deg, L_grid, thr)

    annot = (
        f"Contour: P = {thr}\n"
        f"Max θ (exists some L): {max_theta_deg:.1f}°\n"
        f"Example L at max θ: {L_at_max_theta:.1f} nm"
    )
    ax.text(
        bx, by, annot,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10, color=c,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=c)
    )

ax.set_xlabel("Linker length L (nm)", fontsize=13)
ax.set_ylabel("Bending angle θ (degrees)", fontsize=13)
ax.set_title(
    "Linker probability map (angular tolerance) with two thresholds\n"
    f"lp={lp:g} nm, L0={L0:g} nm, θ0={theta0_deg:g}° | "
    f"w_wlc={w_wlc:g}, w_L={w_L:g}, w_th={w_th:g}",
    fontsize=12,
    pad=12
)

fig.tight_layout()

out_png = "WLC_linker_probability_map_angular_tolerance_two_thresholds_labeled.png"
fig.savefig(out_png, dpi=300)
plt.show()

print(f"✅ Figure saved as '{out_png}'")
