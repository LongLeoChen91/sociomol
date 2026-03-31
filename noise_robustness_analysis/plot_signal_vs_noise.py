import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Connect to the core directories
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_SCRIPT_DIR)

# Configurations for physics (using PolysomeManual_1 physics setup)
LP = 1.5
L0 = 20.0
THETA0_DEG = 45.0
W_WLC = 0.0
W_L = 1.0
W_TH = 1.0
W_L_SQ = 0.0
W_TH_SQ = 0.0
L_IDEAL = 0.0
L_STD = 20.0
THETA_STD_DEG = 90.0

# Paths
PRED_CSV = os.path.join(_REPO_ROOT, "experiments", "PolysomeManual_1_Noise", "Linker_edges.csv")
GT_CSV = os.path.join(_REPO_ROOT, "experiments", "PolysomeManual_1", "GroundTruth_edges_PoM1.csv")

def extract_edge_key(row):
    """Uniformly sort IDs and Arms to create a directionless unique edge key for strict matching."""
    id1, id2 = int(row['i_id']), int(row['j_id'])
    # Handle string cases safely just in case
    arm1, arm2 = int(float(row['arm_i'])), int(float(row['arm_j']))
    
    if id1 < id2:
        return f"{id1}_{arm1}-{id2}_{arm2}"
    elif id1 > id2:
        return f"{id2}_{arm2}-{id1}_{arm1}"
    else:
        if arm1 <= arm2: return f"{id1}_{arm1}-{id2}_{arm2}"
        else: return f"{id2}_{arm2}-{id1}_{arm1}"

def main():
    if not os.path.exists(PRED_CSV) or not os.path.exists(GT_CSV):
        print(f"Error: Required CSV files not found!")
        print(f"Pred: {PRED_CSV}")
        print(f"GT: {GT_CSV}")
        return

    print("Loading datasets...")
    df_pred = pd.read_csv(PRED_CSV)
    df_gt = pd.read_csv(GT_CSV)

    # 1. Build Ground Truth Set (Arm-level strict adherence)
    gt_keys = set()
    for _, row in df_gt.iterrows():
        gt_keys.add(extract_edge_key(row))

    # 2. Separate Predictions and Ground Truth into TP, FP, and FN
    pred_keys = {}
    for _, row in df_pred.iterrows():
        if pd.isna(row['D_nm']) or pd.isna(row['theta_deg']): continue
        key = extract_edge_key(row)
        pred_keys[key] = row

    gt_dict = {}
    for _, row in df_gt.iterrows():
        gt_dict[extract_edge_key(row)] = row

    tp_rows = []
    fp_rows = []
    fn_rows = []

    # True Positives & False Positives from Prediction perspective
    for key, row in pred_keys.items():
        if key in gt_dict:
            tp_rows.append(row)
        else:
            fp_rows.append(row)

    # False Negatives from GT perspective (Missed edges)
    for key, row in gt_dict.items():
        if key not in pred_keys:
            fn_rows.append(row)

    df_tp = pd.DataFrame(tp_rows)
    df_fp = pd.DataFrame(fp_rows)
    df_fn = pd.DataFrame(fn_rows)

    print("-" * 50)
    print(f"Total Predicted Valid Edges: {len(df_tp) + len(df_fp)}")
    print(f"True Positives (Signal TP):  {len(df_tp)}")
    print(f"False Positives (Noise FP):  {len(df_fp)}")
    print(f"False Negatives (Missed GT): {len(df_fn)}")
    print("-" * 50)

    # 3. Prepare the Physics Energy Background (using D = straight-line distance)
    max_D = max(float(df_pred['D_nm'].max()), 35.0)
    D_vals = np.linspace(1.0, max_D * 1.05, 400)
    t_vals_rad = np.linspace(0.0, np.pi, 400)
    t_vals_deg = np.degrees(t_vals_rad)

    D_grid, theta_grid_rad = np.meshgrid(D_vals, t_vals_rad)
    _, theta_grid_deg = np.meshgrid(D_vals, t_vals_deg)

    theta0 = np.deg2rad(THETA0_DEG)
    theta_std = np.deg2rad(THETA_STD_DEG)

    # Convert D to arc length L for the energy model: L = theta * D / (2*sin(theta/2))
    half_t = 0.5 * theta_grid_rad
    sin_half = np.sin(half_t)
    L_grid = np.where(np.abs(sin_half) < 1e-8, D_grid, theta_grid_rad * D_grid / (2.0 * sin_half))

    E_wlc = (2.0 * LP / L_grid) * (0.5 * theta_grid_rad) ** 2
    E_len = (L_grid / L0)
    E_ang = (0.5 * theta_grid_rad) / theta0
    E_len_sq = ((L_grid - L_IDEAL) / L_STD)**2
    E_ang_sq = (theta_grid_rad / theta_std)**2

    E_total = (W_WLC * E_wlc) + (W_L * E_len) + (W_TH * E_ang) + (W_L_SQ * E_len_sq) + (W_TH_SQ * E_ang_sq)
    P_grid = np.exp(-E_total)

    # 4. Plotting
    print("Generating Visualized Energy Landscape...")
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Contour Maps Overlay
    cf = ax.contourf(
        D_grid, theta_grid_deg, P_grid, 
        levels=np.linspace(0, 1, 101), 
        cmap="YlGnBu", 
        alpha=0.6
    )
    fig.colorbar(cf, ax=ax, label="P(D, θ) (Probability Score)", ticks=np.linspace(0, 1, 11))

    # Threshold contour lines
    cs = ax.contour(
        D_grid, theta_grid_deg, P_grid,
        levels=[0.01, 0.05, 0.2, 0.5],
        colors=["red", "orange", "magenta", "black"],
        linewidths=[2.5, 2.0, 1.5, 1.5],
        linestyles=["-", "--", "-.", ":"]
    )
    ax.clabel(cs, fmt={thr: f"P={thr}" for thr in [0.01, 0.05, 0.2, 0.5]}, inline=True, fontsize=11, use_clabeltext=True)

    # Scatter Plots
    # Noise (False Positives)
    if not df_fp.empty:
        ax.scatter(df_fp['D_nm'], df_fp['theta_deg'], 
                   s=15, color='dodgerblue', alpha=0.35, 
                   linewidth=0, label=f'Noise FP (n={len(df_fp)})')
    
    # Ground Truths (True Positives - Correctly found)
    if not df_tp.empty:
        ax.scatter(df_tp['D_nm'], df_tp['theta_deg'], 
                   s=45, color='crimson', edgecolor='black', 
                   linewidth=1.0, alpha=0.95, label=f'Signal TP (n={len(df_tp)})')

    # False Negatives (Missed Ground Truths)
    if not df_fn.empty:
        ax.scatter(df_fn['D_nm'], df_fn['theta_deg'], 
                   s=80, facecolors='none', edgecolors='red', 
                   linewidth=1.5, alpha=1.0, label=f'Signal FN (Missed, n={len(df_fn)})')

    ax.set_xlabel("Arm–Arm Distance $D$ (nm)", fontsize=13)
    ax.set_ylabel("Bending Angle $\\theta$ (degrees)", fontsize=13)
    ax.set_xlim(0, max_D * 1.05)
    ax.set_ylim(0, 180)
    
    ax.set_title(
        f"Ground Truth Signal vs. Robustness Noise Distribution\n"
        f"10x Contamination Noise Pressure on Polysome Manual 1",
        fontsize=12, pad=15
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Make legend markers solid regardless of alpha
    legend = ax.legend(loc="upper right", framealpha=0.95, edgecolor='black')
    for lh in legend.legend_handles: 
        lh.set_alpha(1) 
        
    plt.tight_layout()
    out_png = "signal_vs_noise_landscape.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show() # Attempt to display the GUI pane as well
    print(f"Plot saved successfully to: {os.path.abspath(out_png)}")

if __name__ == "__main__":
    main()
