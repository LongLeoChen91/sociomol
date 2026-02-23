import os

# ============================================================
# Global Plotting Configuration for linker predictions
# ============================================================

# --- Data Source ---
CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\Ribosome_tomo0017\Linker_edges.csv"

# --- Probability Filtering ---
P_THRESHOLD = 0.5            # Min probability for plots to consider an edge valid

# --- Energy Model Physics Constants ---
LP = 1.5                   # (nm) Persistence length (e.g., 1.5 for mRNA, 50 for DNA)
L0 = 20                    # (nm) Reference length (ideally connection length)
THETA0_DEG = 45            # (deg) Angular tolerance
W_WLC = 1.0                # Weight for worm-like chain (bending) energy
W_L = 1.0                  # Weight for length penalty
W_TH = 1.0                 # Weight for angular penalty

# --- Plotting Visuals : 2D Overlay Map ---
CONTOUR_THRESHOLDS = [0.05]

# --- Plotting Visuals : 1D Length Distribution ---
FIT_MODE = "single"          # "single" for normal Gaussian, "gmm" for Gaussian Mixture
N_COMPONENTS = 2             # Number of peaks if FIT_MODE is "gmm"
BINS_MIN = 10                # Histogram x-axis start (nm)
BINS_MAX = 36                # Histogram x-axis end (nm)
BINS_STEP = 2                # Histogram bin width (nm)
