import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

# Link project modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_SCRIPT_DIR)

from noise_robustness_analysis.cases_config import get_case

def extract_edge_key(row):
    id1, id2 = int(row['i_id']), int(row['j_id'])
    arm1, arm2 = int(float(row['arm_i'])), int(float(row['arm_j']))
    if id1 < id2: return f"{id1}_{arm1}-{id2}_{arm2}"
    elif id1 > id2: return f"{id2}_{arm2}-{id1}_{arm1}"
    else:
        if arm1 <= arm2: return f"{id1}_{arm1}-{id2}_{arm2}"
        else: return f"{id2}_{arm2}-{id1}_{arm1}"

def main():
    parser = argparse.ArgumentParser(description="Plot Signal vs Noise over Physics Energy Landscape.")
    parser.add_argument("--case", type=str, default="PolysomeManual_1", help="Case name from cases_config.py")
    parser.add_argument("--pred-csv", type=str, help="Override path to prediction CSV (Linker_edges.csv)")
    args = parser.parse_args()

    config = get_case(args.case)
    
    # 默认寻找该 Case 对应的典型结果
    if args.pred_csv:
        PRED_CSV = args.pred_csv
    else:
        # 推测默认预测文件位置 (实验目录下)
        exp_dir = os.path.dirname(config['source_star'])
        # 尝试寻找通用的 Linker_edges.csv 或 DoubleLinker_edges.csv
        possible_csvs = ["Linker_edges.csv", "DoubleLinker_edges.csv"]
        PRED_CSV = None
        for name in possible_csvs:
            path = os.path.join(exp_dir, name)
            if os.path.exists(path):
                PRED_CSV = path
                break
        
        if not PRED_CSV:
            raise FileNotFoundError(f"Could not find default Linker_edges.csv in {exp_dir}. Please provide --pred-csv.")

    GT_CSV = config['gt_csv']

    print(f"\n[INFO] Plotting Signal vs Noise for: {config['label']}")
    print(f"Prediction: {PRED_CSV}")
    print(f"Ground Truth: {GT_CSV}")

    # ==========================================
    # 1. Load Data
    # ==========================================
    df_pred = pd.read_csv(PRED_CSV)
    df_gt = pd.read_csv(GT_CSV)

    gt_dict = {extract_edge_key(row): row for _, row in df_gt.iterrows()}

    # Separate Predictions and Ground Truth into TP, FP, and FN
    pred_keys = {}
    for _, row in df_pred.iterrows():
        if pd.isna(row['D_nm']) or pd.isna(row['theta_deg']): continue
        key = extract_edge_key(row)
        pred_keys[key] = row

    tp_rows, fp_rows, fn_rows = [], [], []

    for key, row in pred_keys.items():
        if key in gt_dict: tp_rows.append(row)
        else: fp_rows.append(row)

    for key, row in gt_dict.items():
        if key not in pred_keys: fn_rows.append(row)

    df_tp, df_fp, df_fn = pd.DataFrame(tp_rows), pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)

    print("-" * 50)
    print(f"TP={len(df_tp)}, FP={len(df_fp)}, FN={len(df_fn)}")
    print("-" * 50)

    # ==========================================
    # 2. Prepare Probability Landscape (a4e9046 Style)
    # ==========================================
    max_D = max(float(df_pred['D_nm'].max()), 35.0)
    res = 400

    D_vals = np.linspace(1.0, max_D * 1.05, res)
    t_vals_rad = np.linspace(0.0, np.pi, res)
    t_vals_deg = np.degrees(t_vals_rad)

    D_grid, theta_grid_rad = np.meshgrid(D_vals, t_vals_rad)
    _, theta_grid_deg = np.meshgrid(D_vals, t_vals_deg)

    # Physics parameters from config
    lp = config['lp_nm']
    L0 = config['l0_nm']
    theta0_deg = config['theta0_deg'] # Keep for printing
    theta0_rad = np.radians(theta0_deg)
    w_wlc, w_L, w_th = config['w_wlc'], config['w_L'], config['w_th']
    w_L_sq, w_th_sq = config['w_L_sq'], config['w_th_sq']
    L_ideal, L_std = config['l_ideal_nm'], config['l_std_nm']
    theta_std_rad = np.radians(config['theta_std_deg'])

    print("\n" + "#"*40)
    print(f"DEBUG: Using Case Config -> {args.case}")
    print(f"DEBUG: THETA0_DEG used for Plot = {config['theta0_deg']}")
    print(f"DEBUG: L0 used for Plot = {config['l0_nm']}")
    print("#"*40 + "\n")

    # Convert D to arc length L: L = theta * D / (2*sin(theta/2))
    half_t = 0.5 * theta_grid_rad
    sin_half = np.sin(half_t)
    denom = 2.0 * sin_half
    L_grid = np.where(np.abs(denom) < 1e-8, D_grid, (theta_grid_rad * D_grid) / np.maximum(denom, 1e-12))

    E_wlc = (2.0 * lp / np.maximum(L_grid, 1e-6)) * (0.5 * theta_grid_rad) ** 2
    E_len = L_grid / L0
    E_ang = (0.5 * theta_grid_rad) / theta0_rad
    E_len_sq = ((L_grid - L_ideal) / L_std) ** 2
    E_ang_sq = (theta_grid_rad / theta_std_rad) ** 2

    E_total = (w_wlc * E_wlc) + (w_L * E_len) + (w_th * E_ang) + \
              (w_L_sq * E_len_sq) + (w_th_sq * E_ang_sq)
    P_grid = np.exp(-E_total)

    # ==========================================
    # 3. Plotting (a4e9046 Original Style)
    # ==========================================
    case_label = config.get('label', args.case)
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Probability Contour Fill (YlGnBu colormap)
    cf = ax.contourf(
        D_grid, theta_grid_deg, P_grid,
        levels=np.linspace(0, 1, 101),
        cmap="YlGnBu",
        alpha=0.6
    )
    fig.colorbar(cf, ax=ax, label="P(D, $\\theta$) (Probability Score)", ticks=np.linspace(0, 1, 11))

    # Threshold contour lines with labels
    cs = ax.contour(
        D_grid, theta_grid_deg, P_grid,
        levels=[0.01, 0.05, 0.2, 0.5],
        colors=["red", "orange", "magenta", "black"],
        linewidths=[2.5, 2.0, 1.5, 1.5],
        linestyles=["-", "--", "-.", ":"]
    )
    ax.clabel(cs, fmt={thr: f"P={thr}" for thr in [0.01, 0.05, 0.2, 0.5]},
              inline=True, fontsize=11, use_clabeltext=True)

    # Scatter: Noise FP
    if not df_fp.empty:
        ax.scatter(df_fp['D_nm'], df_fp['theta_deg'],
                   s=15, color='dodgerblue', alpha=0.35,
                   linewidth=0, label=f'Noise FP (n={len(df_fp)})')

    # Scatter: Signal TP
    if not df_tp.empty:
        ax.scatter(df_tp['D_nm'], df_tp['theta_deg'],
                   s=45, color='crimson', edgecolor='black',
                   linewidth=1.0, alpha=0.95, label=f'Signal TP (n={len(df_tp)})')

    # Scatter: Signal FN (Missed)
    if not df_fn.empty:
        ax.scatter(df_fn['D_nm'], df_fn['theta_deg'],
                   s=80, facecolors='none', edgecolors='red',
                   linewidth=1.5, alpha=1.0, label=f'Signal FN (Missed, n={len(df_fn)})')

    ax.set_xlabel("Arm-Arm Distance $D$ (nm)", fontsize=13)
    ax.set_ylabel("Bending Angle $\\theta$ (degrees)", fontsize=13)
    ax.set_xlim(0, max_D * 1.05)
    ax.set_ylim(0, 180)

    ax.set_title(
        f"Ground Truth Signal vs. Robustness Noise Distribution\n{case_label}",
        fontsize=12, pad=15
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend with solid markers
    legend = ax.legend(loc="upper right", framealpha=0.95, edgecolor='black')
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()

    OUT_DIR = os.path.join(_SCRIPT_DIR, "outputs", args.case)
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_out = os.path.join(OUT_DIR, "signal_vs_noise_landscape.png")
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to {plot_out}")

if __name__ == "__main__":
    main()
