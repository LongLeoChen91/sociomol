"""
visualize_Lp_mapping.py
-----------------------
Publication-quality figure: 4x4 synthetic cluster grid on the (L, θ) manifold,
demonstrating how data position encodes effective persistence length Lp*.

Layout
------
Single axes with:
  1. Background heatmap  f(L, θ) = θ_rad² / (4L)  [MLE sufficient statistic]
     cmap='coolwarm'  — blue = rigid region, red = flexible region
  2. Iso-Lp* contour lines  (white dashed) showing where Lp* is constant
  3. 16 synthetic Gaussian clusters on a 4×4 (L, θ) grid
     - Scatter points: light-grey, translucent (position only)
     - Centroid diamonds: colored by log10(Lp*) via shared colorbar
     - Compact label: "Xnm" printed beside each diamond
  4. Two separate colorbars: f(L,θ) background + Lp* diamond

Physics
-------
  Simplified WLC MLE:  Lp* ≈ 2L / θ_rad²   (analytic, single-cluster approximation)
  Full MLE:            Lp* = 2N / Σ(θ_i² / L_i)  (used for sampled points)
  Relative Cramér-Rao: σ/Lp* ≈ √(2/N)
  Note: Gaussian WLC approximation is valid for θ ≲ 90°; shown up to 180° for illustration.

Author: Long Chen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)

# ============================================================
# 1. Grid for background heatmap
# ============================================================
L_min, L_max = 2.0,  62.0     # nm
T_min, T_max = 1.0, 180.0     # degrees

nL, nT = 500, 500
L_vals    = np.linspace(L_min, L_max, nL)
T_deg     = np.linspace(T_min, T_max, nT)
T_rad     = np.radians(T_deg)

L_grid, T_grid_deg = np.meshgrid(L_vals, T_deg)
_,      T_grid_rad = np.meshgrid(L_vals, T_rad)

f_grid = T_grid_rad**2 / (4.0 * L_grid)    # MLE sufficient stat; physics in radians

# ============================================================
# 2. 4×4 Cluster Grid Definition
# ============================================================
L_levels  = [8, 20, 30, 40, 52]          # nm  (Short → Long)
T_levels  = [15, 50, 70, 100, 150]        # degrees  (Rigid → Flexible)

N_PER    = 120     # points per cluster
L_STD    = 1.8     # nm   spread of each cluster
T_STD_DEG = 3.5    # degrees spread

# ============================================================
# 3. MLE Helper
# ============================================================
def analytic_lp_star(L_arr, theta_rad_arr):
    denom = np.sum(theta_rad_arr**2 / L_arr)
    return (2.0 * len(L_arr) / denom) if denom > 0 else np.nan

# ============================================================
# 4. Pre-compute Lp* for all 16 clusters
# ============================================================
clusters = []

for Lmu in L_levels:
    for Tmu_deg in T_levels:
        L_s = np.random.normal(Lmu, L_STD, N_PER)
        T_s_deg = np.abs(np.random.normal(Tmu_deg, T_STD_DEG, N_PER))
        T_s_rad = np.radians(T_s_deg)

        valid = (L_s > L_min) & (L_s < L_max) & (T_s_deg > T_min) & (T_s_deg < T_max)
        L_v, T_v_deg, T_v_rad = L_s[valid], T_s_deg[valid], T_s_rad[valid]

        lp = analytic_lp_star(L_v, T_v_rad)
        rel_err = np.sqrt(2.0 / len(L_v)) if len(L_v) > 0 else np.nan
        clusters.append((Lmu, Tmu_deg, L_v, T_v_deg, T_v_rad, lp, rel_err))

lp_arr = np.array([c[5] for c in clusters])
lp_min, lp_max = np.nanmin(lp_arr), np.nanmax(lp_arr)

# ============================================================
# 5. Figure
# ============================================================
fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)

# --- 5a. Background heatmap ---
hm = ax.pcolormesh(
    L_vals, T_deg, f_grid,
    cmap="coolwarm",
    norm=LogNorm(vmin=f_grid.min() + 1e-8, vmax=f_grid.max()),
    shading="auto", alpha=0.65, zorder=0,
)
# Two colorbars: f(L,θ) on the right, Lp* horizontal at the bottom
cb1 = fig.colorbar(hm, ax=ax, pad=0.02, fraction=0.035, location="right")
cb1.set_label(r"$f(L,\theta)=\theta_{\rm rad}^2/(4L)$" + "   [rad\u00b2 nm\u207b\u00b9]",
              fontsize=9)
cb1.ax.tick_params(labelsize=8)

# --- 5b. Iso-Lp* contour lines ---
# Lp*(L,θ) = 2L / θ_rad²  → θ_rad = sqrt(2L / Lp)
iso_lp_vals = [5, 15, 30, 50, 150, 500]      # nm
for lp_iso in iso_lp_vals:
    # f = θ²/(4L) = 1/(2*Lp)  → constant on iso-Lp contours
    f_iso = 1.0 / (2.0 * lp_iso)
    cs = ax.contour(L_grid, T_grid_deg, f_grid,
                    levels=[f_iso], colors="#FFD700",
                    linestyles="--", linewidths=1.5, alpha=0.95, zorder=1)
    fmt = {f_iso: f"$L_p$={lp_iso} nm"}
    lbls = ax.clabel(cs, fmt=fmt, inline=True, fontsize=7.5,
                     colors="#1a1a1a", use_clabeltext=True,
                     rightside_up=True)
    for lbl in lbls:
        lbl.set_bbox(dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

# --- 5c. Scatter + diamonds ---
# coolwarm_r: low Lp* (flexible) = red, high Lp* (rigid) = blue — matches background
lp_norm = mcolors.LogNorm(vmin=2, vmax=200)      # cap: colour detail in 2–200 nm range
cmap_lp  = plt.colormaps["coolwarm_r"]

for (Lmu, Tmu_deg, L_v, T_v_deg, T_v_rad, lp, rel_err) in clusters:
    # Light grey scatter (more visible than white)
    ax.scatter(L_v, T_v_deg, s=6, color="#cccccc", alpha=0.45,
               edgecolors="none", zorder=2)

    if not np.isnan(lp):
        rgba = cmap_lp(lp_norm(lp))
        # Diamond centroid — white edge for pop-out contrast
        ax.plot(Lmu, Tmu_deg, marker="D", ms=12,
                color=rgba, markeredgecolor="white",
                markeredgewidth=1.5, zorder=5)
        # Compact Lp* label
        lp_txt = f"{lp:.0f}" if lp < 1000 else f"{lp/1000:.1f}k"
        # Offset alternating rows to reduce overlap
        dy = 6 if Tmu_deg < 90 else -8
        dx = 1.5
        ax.text(Lmu + dx, Tmu_deg + dy, f"{lp_txt} nm",
                fontsize=7, ha="left", va="center",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc=rgba, ec="none", alpha=0.80),
                zorder=6)

    print(f"L={Lmu:3d} nm  th={Tmu_deg:3d}deg  N_valid={len(L_v):3d}  "
          f"Lp* = {lp:7.1f} nm  +/-{rel_err*100:.0f}%")

# --- 5d. Lp* colorbar — horizontal strip below the axes ---
sm = cm.ScalarMappable(cmap=cmap_lp, norm=lp_norm)
sm.set_array([])
cb2 = fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.04,
                   location="bottom", orientation="horizontal",
                   extend="both")    # arrows show out-of-range values
cb2.set_label("$L_p^*$  (nm) ", fontsize=9)
cb2.ax.tick_params(labelsize=8)

# ============================================================
# 6. Aesthetics
# ============================================================
ax.set_xlim(L_min, L_max)
ax.set_ylim(T_min, T_max)
ax.set_xlabel("Linker arc length  $L$  (nm)", fontsize=13)
ax.set_ylabel("Bending angle  $\\theta$  (degrees)", fontsize=13)
ax.set_title(
    "$(L,\\,\\theta)$ Manifold \u2192 Effective Persistence Length  $L_p^*$\n"
    r"Background: $f(L,\theta)=\theta^2/(4L)$  |  "
    r"Gold dashed: iso-$L_p^*$ lines  |  "
    r"Diamonds: MLE $L_p^*=2N/\!\sum(\theta_i^2/L_i)$",
    fontsize=10, pad=10,
)

# Region labels
ax.text(3,  10, "Low $f$\n(Rigid)", fontsize=8, color="white", style="italic",
        bbox=dict(boxstyle="round,pad=0.2", fc="steelblue", alpha=0.5, lw=0))
ax.text(38, 165, "High $f$\n(Flexible)", fontsize=8, color="white", style="italic",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="firebrick", alpha=0.5, lw=0))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

out_png = "visualize_Lp_mapping.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"\n[OK] Saved: {out_png}")
