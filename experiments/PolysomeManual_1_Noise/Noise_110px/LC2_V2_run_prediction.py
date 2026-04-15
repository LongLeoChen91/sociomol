import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from linker_prediction import run_prediction_pipeline

# ==========================================
# 1. Configuration
# ==========================================
INPUT_STAR  = "Avg_Linkers.star"
OUTPUT_STAR = "Avg_Linkers_annotated.star"
EDGES_CSV   = "Linker_edges.csv"

PIXEL_SIZE_A  = 1.96                # Å/pixel
DIST_CUTOFF_NM = 30                 # [nm] Arm–arm distance cutoff
P_THRESHOLD = 0                  # 0.05; Probability threshold for assignment

# ==========================================
# 2. Five-term Energy Model Configuration
# P(L, θ) ∝ exp( -W_WLC*E_wlc - W_L*E_len - W_TH*E_ang )
# ==========================================

# -- A. Physics Base Parameters --
LP_NM = 1.5                         # [nm] Persistence length (bending stiffness)
L0_NM = 20                          # [nm] Reference length (ideal connection distance)
THETA0_DEG = 45.0                   # [deg] Reference angle for angle penalty

# -- B. Formula Component Weights --
W_WLC = 0                         # Weight for WLC bending energy
W_L = 1                           # Weight for linear distance penalty
W_TH = 1                            # Weight for relative angle tolerance

# -- Sub-Gaussian Penalties (Squared bounds) --
W_L_SQ = 0                        # Weight for squared distance penalty
W_TH_SQ = 0                       # Weight for squared angle penalty
L_IDEAL_NM = 0                   # [nm] Ideal distance for squared penalty
L_STD_NM = 20.0                     # [nm] Distance tolerance (std dev)
THETA_STD_DEG = 90.0                # [deg] Angular tolerance (std dev)

# -- C. Geometry & Structural Constraints --
PORT_PAIRING = "complement"         # "any" (all pairs) or "complement" (forbid 0->0, 1->1)

# Angle calculation mode for geometric bending (theta)
# "alpha_sum": (default) Physically realistic; evaluates sum of deflection angles from the straight connection line.
# "tangent_tangent": (legacy) Naive 3D angle between the two arm direction vectors (ignores translation offsets).
THETA_MODE = "alpha_sum"            

MAX_HALF_BENDING_DEG = 30.0          # Angle with line threshold (0.0 for <90°)

if __name__ == "__main__":
    run_prediction_pipeline(
        input_star=INPUT_STAR,
        output_star=OUTPUT_STAR,
        edges_csv=EDGES_CSV,
        pixel_size_a=PIXEL_SIZE_A,
        dist_cutoff_nm=DIST_CUTOFF_NM,
        lp_nm=LP_NM,
        l0_nm=L0_NM,
        p_threshold=P_THRESHOLD,
        w_wlc=W_WLC,
        w_L=W_L,
        w_th=W_TH,
        w_L_sq=W_L_SQ,
        w_th_sq=W_TH_SQ,
        l_ideal_nm=L_IDEAL_NM,
        l_std_nm=L_STD_NM,
        theta_std_deg=THETA_STD_DEG,
        theta0_deg=THETA0_DEG,
        port_pairing=PORT_PAIRING,
        theta_mode=THETA_MODE,        max_half_bending_deg=MAX_HALF_BENDING_DEG
    )
