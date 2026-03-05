import os

# ============================================================
# Global Plotting Configuration for linker predictions
# ============================================================

# --- Data Source ---
# CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\MR_arm2_0_di30_139_BaseOnEndDensity\GroundTruth_edges_s139.csv"
# CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\MR_arm2_0_di30_139_BaseOnEndDensity\DoubleLinker_edges.csv"


# --- Data Source ---
# CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\MR_arm2_0_di25_72_BaseOnEndDensity\GroundTruth_edges_s72.csv"
# CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\MR_arm2_0_di25_72_BaseOnEndDensity\DoubleLinker_edges.csv"


# --- Data Source ---

# CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\Nucleosome_Ben_tomo_2173\DoubleLinker_edges.csv"


# --- Data Source ---
# CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\Ribosome_tomo0017\Linker_edges.csv"

# --- Data Source ---
# CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\Single_tomo3\DoubleLinker_edges.csv"

# --- Data Source ---
CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\Single_tomo_FV3h24_2005010012\DoubleLinker_edges.csv"


# --- Probability Filtering ---
P_THRESHOLD_MAP = 0       # Applied to the 2D energy landscape overlay
P_THRESHOLD_DIST = 0       # Applied to the 1D length distribution histogram

# --- Geometric L correction (for estimate_effective_Lp.py) ---
# L_true = L_measured - 2 * R_OFFSET_NM
# Ribosome (A-site/E-site to surface): 10–15 nm
# Nucleosome (DNA exit at surface):      0 nm
R_OFFSET_NM = 0         # (nm) rigid-body offset per arm; set 0 to disable
L_MIN_NM    = 0         # (nm) outlier floor after correction

# --- Energy Model Physics Constants (For 2D Energy Landscape) ---
LP = 50                     # (nm) Persistence length (e.g., 1.5 for mRNA, 50 for DNA)
L0 = 20                    # (nm) Reference length (ideally connection length)
THETA0_DEG = 45            # (deg) Angular tolerance
W_WLC = 0                # Weight for worm-like chain (bending) energy
W_L = 1                 # Weight for length penalty
W_TH = 1                # Weight for angular penalty

# --- Sub-Gaussian Penalties (Squared bounds) ---
W_L_SQ = 0            # Weight for squared distance penalty
W_TH_SQ = 0           # Weight for squared angle penalty
L_IDEAL = 0          # (nm) Ideal distance (center of the Gaussian)
L_STD = 20            # (nm) Distance tolerance (standard deviation)
THETA_STD_DEG = 90    # (deg) Angular tolerance (standard deviation)

# --- Plotting Visuals : 2D Overlay Map ---
CONTOUR_THRESHOLDS = [0.03,0.37]

# --- Plotting Visuals : 1D Length Distribution ---
FIT_MODE = "single"          # "single" for normal Gaussian, "gmm" for Gaussian Mixture
N_COMPONENTS = 2             # Number of peaks if FIT_MODE is "gmm"
BINS_MIN = 10                # Histogram x-axis start (nm)
BINS_MAX = 50                # Histogram x-axis end (nm)
BINS_STEP = 2                # Histogram bin width (nm)
