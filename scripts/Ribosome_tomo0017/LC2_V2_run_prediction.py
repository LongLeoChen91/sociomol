import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from linker_prediction import run_prediction_pipeline

# ==========================================
# 1. Configuration
# ==========================================
INPUT_STAR  = "Avg_Linkers_ID_Reset_A_ribosome80s_T0017_with_origin.star"
OUTPUT_STAR = "Avg_Linkers_annotated.star"
EDGES_CSV   = "Linker_edges.csv"

PIXEL_SIZE_A  = 1.96                # Å/pixel
DIST_CUTOFF_NM = 40                 # [nm] Arm–arm distance cutoff
LP_NM = 1.5                         # [nm] Persistence length
L0_NM = 40                          # [nm] Reference length
P_THRESHOLD = 0.3                   # Probability threshold for assignment

# Triple-term Energy Model Weights
W_WLC = 1.0                         # Weight for WLC bending energy
W_L = 1.0                           # Weight for linear distance penalty
W_TH = 1.0                          # Weight for relative angle tolerance
THETA0_DEG = 45.0                   # Reference angle (degrees) for angle penalty

PORT_PAIRING = "complement"         # "any" (all pairs) or "complement" (forbid 0->0, 1->1)
THETA_MODE = "alpha_sum"            # Angle calculation mode
REQUIRE_TOWARD_LINE = True          # Require arms pointing toward connection line
TOWARD_COS_THRESHOLD = 0.0          # Angle with line threshold (0.0 for <90°)

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
        theta0_deg=THETA0_DEG,
        port_pairing=PORT_PAIRING,
        theta_mode=THETA_MODE,
        require_toward_line=REQUIRE_TOWARD_LINE,
        toward_cos_threshold=TOWARD_COS_THRESHOLD
    )
