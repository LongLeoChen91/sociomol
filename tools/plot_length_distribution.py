import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. Configuration
# ============================================================

CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\Ribosome_tomo0017\Linker_edges.csv"
P_THRESHOLD = 0.5            # Min probability for histogram plotting

FIT_MODE = "single"             # Choose from: "single" (Standard Gaussian) or "gmm" (Gaussian Mixture)

# GMM parameters (only used if FIT_MODE == "gmm")
N_COMPONENTS = 2
RANDOM_STATE = 0
N_INIT = 20

# Histogram parameters
BINS_MIN = 10
BINS_MAX = 36
BINS_STEP = 2

# Derived parameter grids
bins = np.arange(BINS_MIN, BINS_MAX, BINS_STEP)
x_grid = np.linspace(bins.min(), bins.max(), 800)

out_png = f"polysome_repeat_length_Pgt{str(P_THRESHOLD).replace('.', 'p')}_{FIT_MODE}.png"

# ============================================================
# 2. Helper: Standard Normal PDF
# ============================================================
def normal_pdf(x_vals, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    z = (x_vals - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))

# ============================================================
# 3. Load Data
# ============================================================
df = pd.read_csv(CSV_PATH)
for col in ["P", "L_nm"]:
    if col not in df.columns:
        raise ValueError(f"CSV must contain column '{col}'. Available: {list(df.columns)}")

df_sub = df[df["P"] > P_THRESHOLD]
if df_sub.empty:
    raise ValueError(f"No data points left after filtering for P > {P_THRESHOLD}.")

x = df_sub["L_nm"].astype(float).to_numpy()
x = x[np.isfinite(x)]

if len(x) < 5:
    raise ValueError(f"Not enough data points ({len(x)}) found for fitting.")

X = x.reshape(-1, 1)

# ============================================================
# 4. Fitting Logic
# ============================================================
component_pdfs = []
mixture_pdf = np.zeros_like(x_grid)
labels = []

if FIT_MODE == "single":
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    if sigma <= 0:
        raise ValueError("Standard deviation is zero, cannot fit Gaussian.")
    
    pdf = normal_pdf(x_grid, mu, sigma)
    mixture_pdf = pdf
    labels.append(f"$\\mu$={mu:.1f}nm\n$\\sigma$={sigma:.1f}nm")
    
elif FIT_MODE == "gmm":
    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=500
    )
    gmm.fit(X)
    
    weights = gmm.weights_.copy()
    means = gmm.means_.flatten().copy()
    stds = np.sqrt(gmm.covariances_.reshape(-1)).copy()
    
    # Sort components by mean
    order = np.argsort(means)
    weights, means, stds = weights[order], means[order], stds[order]

    for k, (w, mu, sd) in enumerate(zip(weights, means, stds)):
        pdf = w * normal_pdf(x_grid, mu, sd)
        component_pdfs.append(pdf)
        mixture_pdf += pdf
        labels.append(f"Comp {k+1}: μ={mu:.1f}nm, σ={sd:.1f}nm, w={w:.2f}")

else:
    raise ValueError(f"Unknown FIT_MODE: '{FIT_MODE}'. Use 'single' or 'gmm'.")

# ============================================================
# 5. Plotting
# ============================================================
fig, ax = plt.subplots(figsize=(3.4, 3.2), dpi=180)

# Histogram
ax.hist(
    x, bins=bins, density=True,
    color="#F6B26B", edgecolor="0.35", linewidth=0.7, alpha=0.85
)

# Overall fit curve
ax.plot(x_grid, mixture_pdf, color="0.15", linewidth=2.4, zorder=3)

# Add annotations/components based on mode
if FIT_MODE == "single":
    ax.axvline(mu, linestyle="--", linewidth=1.2, color="0.2", zorder=4)
    ax.text(
        0.58, 0.95, "\n".join(labels),
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", linewidth=0.8)
    )
elif FIT_MODE == "gmm":
    for k, pdf in enumerate(component_pdfs):
        ax.plot(x_grid, pdf, linestyle="--", linewidth=1.5, label=labels[k])
    
    leg = ax.legend(fontsize=7, frameon=True, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=1)
    leg.get_frame().set_edgecolor("0.3")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.3)

# Aesthetics
ax.set_xlabel("Linker length (nm)")
ax.set_ylabel("Probability density")
ax.set_title(f"Polysome repeat length (P > {P_THRESHOLD})", fontsize=10, pad=12)

ax.set_xlim(bins.min(), bins.max())
ymax = max(ax.get_ylim()[1], mixture_pdf.max() * 1.15)
ax.set_ylim(0, ymax)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(direction="out", length=3, width=1)

fig.tight_layout()
fig.savefig(out_png, dpi=300)
plt.show()

print(f"✅ Saved distribution plot ({FIT_MODE} mode): {out_png}")
