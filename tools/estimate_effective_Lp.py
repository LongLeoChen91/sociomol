"""
estimate_effective_Lp.py
------------------------
Estimates the Effective Persistence Length (Lp) from linker scatter data
using Maximum Likelihood Estimation (MLE) under the WLC model.

For each of the 4 signal masks from plot_density_landscape_4way.py,
this script:
  1. Extracts the (L, theta) points belonging to the signal region.
  2. Sweeps Lp over a range and computes NLL(Lp):
       NLL(Lp) = Lp * sum(theta^2 / (4*L)) - N/2 * log(Lp) + const
  3. Marks the analytic minimum Lp* = 2N / sum(theta^2/L).
  4. Plots a 2x2 panel of NLL curves with the optimal Lp annotated.

Physics note:
  The WLC bending distribution used is:
      P(theta | L, Lp) ~ sqrt(Lp/L) * exp(-Lp * theta^2 / (4*L))
  which is a Gaussian approximation valid for small-to-moderate bending angles.
  The full bending energy consistent with linker_prediction/probability.py is:
      E_wlc = (2*Lp/L) * (theta/2)^2 = Lp * theta^2 / (2*L)
  The factor 4L (not 2L) in the NLL denominator comes from the variance of
  the Gaussian distribution sigma^2 = 2*L/Lp, giving exponent -theta^2/(2*sigma^2)
  = -Lp*theta^2/(4*L).

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

# ---------------------------------------------------------------------------
# Setup paths so we can import linker_prediction from repo root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_SCRIPT_DIR)

from linker_prediction.probability import bending_energy_lp  # for consistency check
from config_plot import CSV_PATH, P_THRESHOLD_MAP, R_OFFSET_NM, L_MIN_NM

# ============================================================
# 1. Configuration
# ============================================================
THETA_COL = "theta_deg"
L_COL     = "L_nm"

LP_MIN    = 0.5       # nm - lower bound of sweep
LP_MAX    = 300.0     # nm - upper bound of sweep
LP_STEPS  = 600       # resolution of parameter sweep

# R_OFFSET_NM and L_MIN_NM are imported from config_plot.py

# ============================================================
# 2. Data Loading & Filtering
# ============================================================
df     = pd.read_csv(CSV_PATH)
df_sub = df[df["P"] > P_THRESHOLD_MAP].copy()
df_sub = df_sub.dropna(subset=[THETA_COL, L_COL])

x = df_sub[L_COL].values          # arc length   [nm]
y = df_sub[THETA_COL].values      # bending angle [deg]
y_rad = np.radians(y)              # convert to radians for physics

print(f"[INFO] Loaded {len(x)} candidate linker points from CSV")
if len(x) < 10:
    print("[ERROR] Not enough points for reliable estimation.")
    sys.exit(1)

# ============================================================
# 3. KDE + Four Signal Masks  (identical logic to plot_density_landscape_4way.py)
# ============================================================

# --- KDE ---
positions = np.vstack([x, y])
kde       = gaussian_kde(positions)
z_points  = kde(positions)

# --- Method 1: Empirical (Top 30%) ---
DENSITY_PERCENTILE = 0.70
sorted_z           = np.sort(z_points)
idx_70             = min(int(len(sorted_z) * DENSITY_PERCENTILE), len(sorted_z) - 1)
thresh_empirical   = sorted_z[idx_70]
mask_empirical     = z_points >= thresh_empirical

# --- Method 2: Elbow / Knee ---
x_kneed        = np.arange(len(sorted_z))
kneedle        = KneeLocator(x_kneed, sorted_z, S=1.0, curve="convex", direction="increasing")
knee_idx       = kneedle.knee
thresh_elbow   = sorted_z[knee_idx] if knee_idx is not None else thresh_empirical
mask_elbow     = z_points >= thresh_elbow

# --- Method 3: Otsu ---
thresh_otsu_val = threshold_otsu(z_points)
mask_otsu       = z_points >= thresh_otsu_val

# --- Method 4: DBSCAN ---
x_norm   = (x - x.mean()) / x.std()
y_norm   = (y - y.mean()) / y.std()
X_norm   = np.column_stack((x_norm, y_norm))
db       = DBSCAN(eps=0.35, min_samples=5).fit(X_norm)
mask_dbscan = db.labels_ >= 0

# --- All Points (baseline, no density filtering) ---
mask_all = np.ones(len(x), dtype=bool)

methods = [
    ("Empirical (Top 30%)", mask_empirical),
    ("Elbow / Knee Method", mask_elbow),
    ("Otsu's Thresholding", mask_otsu),
    ("DBSCAN Clustering",   mask_dbscan),
    ("All Points (Baseline)", mask_all),
]

# ============================================================
# 4. MLE Lp Estimation Function
# ============================================================

def compute_nll_curve(L_arr: np.ndarray, theta_rad_arr: np.ndarray,
                      lp_range: np.ndarray) -> np.ndarray:
    """
    Negative log-likelihood of WLC model over a sweep of Lp values.

    NLL(Lp) = Lp * S - N/2 * log(Lp)   where S = sum(theta^2 / (4*L))

    Parameters
    ----------
    L_arr        : arc lengths in nm, shape (N,)
    theta_rad_arr: bending angles in radians, shape (N,)
    lp_range     : array of Lp candidates in nm

    Returns
    -------
    nll : shape (len(lp_range),)
    """
    N = len(L_arr)
    # Sufficient statistic: S = sum(theta^2 / (4*L))
    S = np.sum(theta_rad_arr ** 2 / (4.0 * L_arr))
    nll = lp_range * S - (N / 2.0) * np.log(lp_range)
    return nll


def analytic_lp_star(L_arr: np.ndarray, theta_rad_arr: np.ndarray) -> float:
    """
    Analytic MLE solution: Lp* = 2N / sum(theta^2 / L).

    Derived by setting dNLL/dLp = 0.
    """
    N = len(L_arr)
    denom = np.sum(theta_rad_arr ** 2 / L_arr)
    if denom <= 0:
        return float("nan")
    return 2.0 * N / denom


# ============================================================
# 5. Sweep & Estimate for Each Method
# ============================================================
lp_range = np.linspace(LP_MIN, LP_MAX, LP_STEPS)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
_corr_label = f"$L_{{true}} = L_{{meas}} - 2\\times{R_OFFSET_NM:.0f}$ nm" if R_OFFSET_NM > 0 else "No length correction"
fig.suptitle(
    f"Effective $L_p$ Estimation via WLC-MLE\n"
    f"(NLL curve per signal detection method  |  {_corr_label},  outlier floor {L_MIN_NM:.0f} nm)",
    fontsize=12, y=1.01
)

for ax, (title, mask) in zip(axes.flat, methods):
    N_mask = np.sum(mask)

    if N_mask < 5:
        ax.set_title(f"{title}\nInsufficient points ({N_mask})")
        ax.text(0.5, 0.5, "Not enough signal points", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        continue

    L_m_raw = x[mask]
    th_m_raw = y_rad[mask]

    # --- Geometric correction: subtract rigid-body offset ---
    L_m_corr = L_m_raw - 2.0 * R_OFFSET_NM

    # --- Outlier filtering: remove physically impossible points ---
    valid = L_m_corr > L_MIN_NM
    n_outliers = np.sum(~valid)
    L_m   = L_m_corr[valid]
    th_m  = th_m_raw[valid]

    if len(L_m) < 5:
        ax.set_title(f"{title}\nToo few valid points after correction ({len(L_m)})")
        ax.text(0.5, 0.5, "Increase R_OFFSET_NM or check data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="firebrick")
        print(f"[{title}]  N_raw={np.sum(mask):4d}  outliers={n_outliers}  → too few valid points")
        continue

    # NLL curve
    nll     = compute_nll_curve(L_m, th_m, lp_range)

    # Analytic optimum
    lp_star = analytic_lp_star(L_m, th_m)
    nll_star = float(compute_nll_curve(L_m, th_m, np.array([lp_star]))[0]) if not np.isnan(lp_star) else np.nan

    # --- Plot ---
    ax.plot(lp_range, nll, color="steelblue", linewidth=2, label="NLL$(L_p)$")

    if not np.isnan(lp_star):
        ax.axvline(lp_star, color="red", linestyle="--", linewidth=1.5,
                   label=f"$L_p^*$ = {lp_star:.1f} nm")
        ax.plot(lp_star, nll_star, "ro", markersize=9, zorder=5)
        ax.annotate(f"$L_p^*$ = {lp_star:.1f} nm",
                    xy=(lp_star, nll_star),
                    xytext=(lp_star + (LP_MAX - LP_MIN) * 0.05, nll_star),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

    ax.set_title(f"{title}  (N_raw={N_mask}, N_valid={len(L_m)}, outliers={n_outliers})", fontsize=9)
    ax.set_xlabel("$L_p$ (nm)")
    ax.set_ylabel("NLL (relative)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print(f"[{title}]  N_raw={N_mask:4d}  outliers={n_outliers:3d}  N_valid={len(L_m):4d}  Lp* = {lp_star:.2f} nm")

plt.tight_layout()
# Hide unused 6th panel (2x3 grid has 6 slots, we use 5)
axes.flat[-1].set_visible(False)

out_png = "estimate_Lp_5methods.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.show()
print(f"\n[OK] Saved: {out_png}")
