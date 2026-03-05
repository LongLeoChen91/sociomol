"""
lp_heatmap_2d.py
----------------
2D spatial heatmap of the effective persistence length Lp*(x,y) across the
XY plane, computed from DoubleLinker_edges.csv + the companion STAR file.

Algorithm
---------
1. Load STAR  → particle XYZ coordinates (pixel units)
2. Load CSV   → linker edges (i_idx, j_idx, theta_deg, L_nm, P)
3. Join       → linker midpoint XY = (X[i] + X[j]) / 2  (converted to nm)
4. P filter   → keep only rows with P > P_THRESHOLD
5. Sliding window:
     for each grid point (xi, yi) with stride STRIDE_NM:
         collect all linkers within WINDOW_RADIUS_NM in XY
         if N_valid >= N_MIN:
             Lp*(xi,yi) = 2*N / Σ(θ²/L)   [WLC-MLE analytic]
             CR(xi,yi)  = sqrt(2/N) * 100  [Cramér-Rao relative error, %]
6. Plot 3-panel figure:
     Panel 1: Lp* heatmap (log scale, fixed colorbar ticks)
     Panel 2: N linkers per window cell
     Panel 3: Cramér-Rao relative error map (reliability indicator)

Author: Long Chen
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import pandas as pd
import starfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_SCRIPT_DIR)

from config_plot import CSV_PATH, P_THRESHOLD_MAP, R_OFFSET_NM, L_MIN_NM

# ============================================================
# 1. Configuration
# ============================================================
# STAR file: annotated output lives in the same directory as the CSV
# and already contains all particle coordinates (superset of input STAR)
_EXP_DIR  = os.path.dirname(CSV_PATH)

# Sliding window parameters
WINDOW_RADIUS_NM = 50.0   # half-diameter of the circular window (nm)
STRIDE_NM        = 15.0   # grid evaluation spacing (nm)
N_MIN            = 5      # minimum linkers per window for reliable Lp*

# Column names
THETA_COL = "theta_deg"
L_COL     = "L_nm"
I_IDX     = "i_idx"
J_IDX     = "j_idx"

# STAR coordinate column (pixel units)
# Note: starfile library strips the leading '_' from RELION column names
STAR_X_COL = "rlnLC_CoordinateX0"
STAR_Y_COL = "rlnLC_CoordinateY0"

# Pixel size from experiment config (Å/px → nm/px = Å/px / 10)
PIXEL_SIZE_NM = 1.513 / 10.0   # nm per pixel for FV3h24_2005010012

# Colourmap limits (nm) — log scale
LP_VMIN = 5.0
LP_VMAX = 50.0

# ============================================================
# 2. Locate annotated STAR file (same dir as CSV)
# ============================================================
_ann_stars = [f for f in os.listdir(_EXP_DIR) if f.endswith("_annotated.star")]
if not _ann_stars:
    raise FileNotFoundError(f"No *_annotated.star found in {_EXP_DIR}")
STAR_PATH = os.path.join(_EXP_DIR, sorted(_ann_stars)[0])
print(f"[INFO] STAR: {os.path.basename(STAR_PATH)}")
print(f"[INFO] CSV : {os.path.basename(CSV_PATH)}")

# ============================================================
# 3. Load annotated STAR → particle XY positions (nm)
# ============================================================
particles = starfile.read(STAR_PATH)   # returns flat DataFrame
print(f"[INFO] Particles: {len(particles)}")

nm_x = particles[STAR_X_COL].values.astype(float) * PIXEL_SIZE_NM
nm_y = particles[STAR_Y_COL].values.astype(float) * PIXEL_SIZE_NM

# ============================================================
# 4. Load CSV → join midpoint XY
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[THETA_COL, L_COL, "P", I_IDX, J_IDX])
df[I_IDX] = df[I_IDX].astype(int)
df[J_IDX] = df[J_IDX].astype(int)

# Validate index range
assert df[I_IDX].max() < len(particles), "i_idx out of range vs STAR rows"
assert df[J_IDX].max() < len(particles), "j_idx out of range vs STAR rows"

# Midpoint XY in nm
df["mid_x_nm"] = (nm_x[df[I_IDX].values] + nm_x[df[J_IDX].values]) / 2.0
df["mid_y_nm"] = (nm_y[df[I_IDX].values] + nm_y[df[J_IDX].values]) / 2.0

print(f"[INFO] Total edges before P filter: {len(df)}")

# ============================================================
# 5. P threshold filter + L geometric correction
# ============================================================
P_THR = P_THRESHOLD_MAP if P_THRESHOLD_MAP > 0 else 0.05  # sensible default
df_f  = df[df["P"] > P_THR].copy()
L_true = df_f[L_COL].values - 2.0 * R_OFFSET_NM
valid  = L_true > L_MIN_NM
df_f   = df_f[valid].copy()
L_v    = (df_f[L_COL].values - 2.0 * R_OFFSET_NM)
T_v    = np.radians(df_f[THETA_COL].values)
X_v    = df_f["mid_x_nm"].values
Y_v    = df_f["mid_y_nm"].values

print(f"[INFO] Edges after P>{P_THR} + L_MIN>{L_MIN_NM} nm: {len(df_f)}")
print(f"[INFO] XY extent: x=[{X_v.min():.0f}, {X_v.max():.0f}] nm  "
      f"y=[{Y_v.min():.0f}, {Y_v.max():.0f}] nm")

# ============================================================
# 6. Build XY evaluation grid
# ============================================================
xi_vals = np.arange(X_v.min(), X_v.max(), STRIDE_NM)
yi_vals = np.arange(Y_v.min(), Y_v.max(), STRIDE_NM)
Lp_map  = np.full((len(yi_vals), len(xi_vals)), np.nan)
N_map   = np.zeros((len(yi_vals), len(xi_vals)), dtype=int)
CR_map  = np.full((len(yi_vals), len(xi_vals)), np.nan)   # Cramér-Rao rel. error (%)

print(f"[INFO] Grid: {len(xi_vals)} × {len(yi_vals)}  "
      f"(stride={STRIDE_NM} nm, radius={WINDOW_RADIUS_NM} nm)")

R2 = WINDOW_RADIUS_NM ** 2
for j, yi in enumerate(yi_vals):
    dy2 = (Y_v - yi) ** 2
    for i, xi in enumerate(xi_vals):
        dist2 = (X_v - xi) ** 2 + dy2
        mask  = dist2 <= R2
        N     = mask.sum()
        N_map[j, i] = N
        if N < N_MIN:
            continue
        denom = np.sum(T_v[mask] ** 2 / L_v[mask])
        if denom > 0:
            Lp_map[j, i] = 2.0 * N / denom
        CR_map[j, i] = np.sqrt(2.0 / N) * 100.0   # Cramér-Rao bound (%)

# stats
valid_cells = ~np.isnan(Lp_map)
print(f"[INFO] Valid grid cells: {valid_cells.sum()} / {Lp_map.size}")
print(f"[INFO] Lp* range: {np.nanmin(Lp_map):.1f} – {np.nanmax(Lp_map):.1f} nm")

# ============================================================
# 7. Plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                         gridspec_kw={"width_ratios": [1.0, 0.85, 0.85]})

# --- Panel 1: Lp* heatmap (log scale, fixed colorbar ticks) ---
ax = axes[0]
norm = mcolors.LogNorm(vmin=LP_VMIN, vmax=LP_VMAX)
im = ax.pcolormesh(xi_vals, yi_vals, Lp_map,
                   cmap="coolwarm_r", norm=norm,
                   shading="auto")

# Overlay raw linker midpoints (tiny grey dots)
ax.scatter(X_v, Y_v, s=1, c="dimgray", alpha=0.3, linewidths=0, zorder=2,
           label="Linker midpoints")

ax.set_aspect("equal")
ax.set_xlabel("X  (nm)", fontsize=11)
ax.set_ylabel("Y  (nm)", fontsize=11)
ax.set_title(
    f"$L_p^*$ heatmap  —  {os.path.basename(CSV_PATH)}\n"
    f"Window radius={WINDOW_RADIUS_NM:.0f} nm  |  P>{P_THR}  |  N_min={N_MIN}",
    fontsize=10
)
ax.legend(fontsize=8, markerscale=5, loc="upper right")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label(r"$L_p^*$  (nm)", fontsize=10)
# Use explicit tick values so colorbar shows plain numbers, not scientific notation
_lp_ticks = [t for t in [5, 6, 8, 10, 15, 20, 30, 50]
              if LP_VMIN <= t <= LP_VMAX]
cb.set_ticks(_lp_ticks)
cb.set_ticklabels([str(t) for t in _lp_ticks])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel 2: N linkers per window cell ---
ax2 = axes[1]
im2 = ax2.pcolormesh(xi_vals, yi_vals, N_map,
                     cmap="viridis", shading="auto")
ax2.set_aspect("equal")
ax2.set_xlabel("X  (nm)", fontsize=11)
ax2.set_title(f"N linkers per window cell\n(white = < {N_MIN} → Lp* = NaN)",
              fontsize=10)
cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
cb2.set_label("N linkers", fontsize=10)
cb2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# --- Panel 3: Cramér-Rao relative error (reliability indicator) ---
# CR(%) = sqrt(2/N) * 100.  Lower = more reliable estimate.
ax3 = axes[2]
im3 = ax3.pcolormesh(xi_vals, yi_vals, CR_map,
                     cmap="YlOrRd", shading="auto",
                     vmin=0, vmax=100)
ax3.set_aspect("equal")
ax3.set_xlabel("X  (nm)", fontsize=11)
ax3.set_title(
    r"Cramér-Rao relative error  $\sqrt{2/N}$"
    f"\n(NaN where N < {N_MIN}; lower = more reliable)",
    fontsize=10
)
cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)
cb3.set_label(r"Rel. error  (%)", fontsize=10)
cb3.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

print(f"[INFO] CR range: {np.nanmin(CR_map):.1f}% – {np.nanmax(CR_map):.1f}%")

plt.tight_layout()
out_png = "lp_heatmap_2d.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.show()
print(f"\n[OK] Saved: {out_png}")
