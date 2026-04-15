import os

# Repository root discovery
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

CASES = {
    "PolysomeManual_1": {
        "label": "Polysome (Manual 1)",
        "source_star": os.path.join(_REPO_ROOT, "experiments", "PolysomeManual_1_Noise", "Avg_Linkers.star"),
        "gt_csv": os.path.join(_REPO_ROOT, "experiments", "PolysomeManual_1", "GroundTruth_edges_PoM1.csv"),
        "pixel_size_a": 1.96,
        "dist_cutoff_nm": 30,
        "p_threshold": 0.0,
        "lp_nm": 1.5,
        "l0_nm": 20.0,
        "theta0_deg": 45.0,
        "port_pairing": "complement",
        "theta_mode": "alpha_sum",
        "max_half_bending_deg": 90.0,
        "w_wlc": 0.0,
        "w_L": 1.0,
        "w_th": 1.0,
        "w_L_sq": 0.0,
        "w_th_sq": 0.0,
        "l_ideal_nm": 0.0,
        "l_std_nm": 20.0,
        "theta_std_deg": 90.0,
    },
    "Manual_1": {
        "label": "Nucleosome (Manual 1)",
        "source_star": os.path.join(_REPO_ROOT, "experiments", "Manual_1_Noise", "H1_DoubleLinker.star"),
        "gt_csv": os.path.join(_REPO_ROOT, "experiments", "Manual_1", "GroundTruth_edges_M1.csv"),
        "pixel_size_a": 8.0,
        "dist_cutoff_nm": 30.0,
        "p_threshold": 0.0,
        "lp_nm": 50.0,
        "l0_nm": 20.0,
        "theta0_deg": 45.0,
        "port_pairing": "any",
        "theta_mode": "alpha_sum",
        "max_half_bending_deg": 90.0,
        "w_wlc": 0.0,
        "w_L": 1.0,
        "w_th": 1.0,
        "w_L_sq": 0.0,
        "w_th_sq": 0.0,
        "l_ideal_nm": 20.0,
        "l_std_nm": 10.0,
        "theta_std_deg": 30.0,
    }
}

def get_case(case_name):
    if case_name not in CASES:
        raise ValueError(f"Unknown case: {case_name}. Available: {list(CASES.keys())}")
    return CASES[case_name]
