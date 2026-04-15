import os

# =============================================================================
#  CONFIGS – Centralized configuration for sweep scripts
# =============================================================================
CONFIGS = {
    "s139": dict(
        exp_name   = "MR_arm2_0_di30_139_BaseOnEndDensity",
        truth_file = "GroundTruth_edges_s139.csv",
    ),
    "s72": dict(
        exp_name   = "MR_arm2_0_di25_72_BaseOnEndDensity",
        truth_file = "GroundTruth_edges_s72.csv",
    ),
    "manual1": dict(
        exp_name   = "Manual_1",
        truth_file = "GroundTruth_edges_M1.csv",
    ),
    "manual1_Noise30": dict(
        exp_name   = "Manual_1_Noise30",
        truth_file = "GroundTruth_edges_M1.csv",
    ),  
    "PolysomeManual_1": dict(
        exp_name   = "PolysomeManual_1",
        truth_file = "GroundTruth_edges_PoM1.csv",
        pred_file  = "Linker_edges.csv",  # Polysome uses different output name
    ),
    "PolysomeManual_1_Noise20": dict(
        exp_name   = "PolysomeManual_1_Noise20",
        truth_file = "GroundTruth_edges_PoM1.csv",
        pred_file  = "Linker_edges.csv",  # Polysome uses different output name
    ),  
}

# Change this line to switch between datasets for all sweep scripts
ACTIVE = "manual1"

def get_sweep_paths(base_dir):
    """
    Returns the mapped paths for the currently active dataset.
    Returns: (exp_dir, pred_script, eval_script, truth_csv, pred_csv)
    """
    cfg = CONFIGS[ACTIVE]
    exp_dir = os.path.join(base_dir, "experiments", cfg["exp_name"])
    pred_script = os.path.join(exp_dir, "LC2_V2_run_prediction.py")
    eval_script = os.path.join(base_dir, "tools", "evaluate_predictions.py")
    truth_csv = os.path.join(exp_dir, cfg["truth_file"])
    
    # Default to DoubleLinker_edges.csv if not specified explicitly
    pred_filename = cfg.get("pred_file", "DoubleLinker_edges.csv")
    pred_csv = os.path.join(exp_dir, pred_filename)
    
    return exp_dir, pred_script, eval_script, truth_csv, pred_csv
