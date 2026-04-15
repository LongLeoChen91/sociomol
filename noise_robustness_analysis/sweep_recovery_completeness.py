import os
import sys
import numpy as np
import pandas as pd
import starfile
import matplotlib.pyplot as plt
import argparse

# Link project modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_SCRIPT_DIR)

from linker_prediction.pipeline import run_prediction_pipeline
from noise_robustness_analysis.cases_config import get_case

def extract_edge_key(row):
    id1, id2 = int(row['i_id']), int(row['j_id'])
    arm1, arm2 = int(float(row['arm_i'])), int(float(row['arm_j']))
    if id1 < id2:
        return f"{id1}_{arm1}-{id2}_{arm2}"
    elif id1 > id2:
        return f"{id2}_{arm2}-{id1}_{arm1}"
    else:
        if arm1 <= arm2: return f"{id1}_{arm1}-{id2}_{arm2}"
        else: return f"{id2}_{arm2}-{id1}_{arm1}"

def evaluate_predictions(pred_csv, gt_csv):
    df_pred = pd.read_csv(pred_csv)
    df_gt = pd.read_csv(gt_csv)

    gt_keys = set([extract_edge_key(row) for _, row in df_gt.iterrows()])
    pred_keys = set([extract_edge_key(row) for _, row in df_pred.iterrows() if not pd.isna(row['L_nm'])])

    tp = len(pred_keys.intersection(gt_keys))
    fp = len(pred_keys - gt_keys)
    fn = len(gt_keys - pred_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Simulate Incomplete Recovery across different cases.")
    parser.add_argument("--case", type=str, default="PolysomeManual_1", help="Case name from cases_config.py")
    parser.add_argument("--removal-ratios", type=float, nargs="+", default=list(np.arange(0.0, 0.51, 0.05)), help="List of removal ratios (0.0 to 0.5)")
    args = parser.parse_args()

    config = get_case(args.case)
    print(f"\n[INFO] Starting Recovery Sweep for Case: {config['label']}")
    print(f"[INFO] Source STAR: {config['source_star']}")

    OUT_DIR = os.path.join(_SCRIPT_DIR, "recovery_outputs", args.case)
    os.makedirs(OUT_DIR, exist_ok=True)

    df_all = starfile.read(config['source_star'])
    df_signal = df_all[df_all['class'] != 99].copy()
    print(f"[INFO] Clean Signal Pool: {len(df_signal)} particles")

    results = []

    for ratio in args.removal_ratios:
        print(f"\n==============================================")
        print(f"--- Simulating Incomplete Recovery: [{ratio*100:3.0f}% Particles Removed] ---")
        
        # 1. 随机移除颗粒
        num_to_keep = int(len(df_signal) * (1.0 - ratio))
        df_sampled = df_signal.sample(n=num_to_keep, random_state=42)
        
        # 2. 写入临时 STAR 并进行预测
        temp_star = os.path.join(OUT_DIR, f"recovery_input_r{ratio:.2f}.star")
        temp_annotated = os.path.join(OUT_DIR, f"recovery_annotated_r{ratio:.2f}.star")
        temp_edges = os.path.join(OUT_DIR, f"recovery_edges_r{ratio:.2f}.csv")
        
        starfile.write(df_sampled, temp_star, overwrite=True)
        
        run_prediction_pipeline(
            input_star=temp_star,
            output_star=temp_annotated,
            edges_csv=temp_edges,
            pixel_size_a=config['pixel_size_a'],
            dist_cutoff_nm=config['dist_cutoff_nm'],
            lp_nm=config['lp_nm'],
            l0_nm=config['l0_nm'],
            p_threshold=config['p_threshold'],
            w_wlc=config['w_wlc'],
            w_L=config['w_L'],
            w_th=config['w_th'],
            w_L_sq=config['w_L_sq'],
            w_th_sq=config['w_th_sq'],
            l_ideal_nm=config['l_ideal_nm'],
            l_std_nm=config['l_std_nm'],
            theta_std_deg=config['theta_std_deg'],
            theta0_deg=config['theta0_deg'],
            port_pairing=config['port_pairing'],
            theta_mode=config['theta_mode'],
            max_half_bending_deg=config['max_half_bending_deg']
        )
        
        # 3. 评估指标
        tp, fp, fn, precision, recall, f1 = evaluate_predictions(temp_edges, config['gt_csv'])
        print(f" -> Result: Particles={len(df_sampled)}, TP={tp}, FN={fn}, P={precision:.3f}, R={recall:.3f}")

        results.append({
            "Removal_Ratio": ratio,
            "Kept_Ratio": 1.0 - ratio,
            "Total_Particles": len(df_sampled),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        })
        
        # 清理临时文件
        if os.path.exists(temp_star): os.remove(temp_star)
        if os.path.exists(temp_annotated): os.remove(temp_annotated)

    # ==========================================
    # 4. 保存报表与绘图
    # ==========================================
    df_results = pd.DataFrame(results)
    csv_out = os.path.join(OUT_DIR, "recovery_sweep_metrics.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n[OK] Metrics saved to {csv_out}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_results["Removal_Ratio"] * 100, df_results["Recall"], marker='o', color='forestgreen', linewidth=2, label="Recall (Recovery)")
    ax.plot(df_results["Removal_Ratio"] * 100, df_results["Precision"], marker='s', color='crimson', linewidth=2, label="Precision")
    ax.plot(df_results["Removal_Ratio"] * 100, df_results["F1_Score"], marker='D', color='black', linestyle='--', linewidth=2, label="F1 Score")
    
    ax.set_title(f"Incomplete Recovery Robustness ({args.case})", fontsize=13)
    ax.set_xlabel("Percentage of Particles Removed (%)", fontsize=12)
    ax.set_ylabel("Global Metric Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2, 55)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    p = df_results["Removal_Ratio"]
    theoretical_recall = (1.0 - p)**2
    ax.plot(df_results["Removal_Ratio"] * 100, theoretical_recall, color='gray', linestyle=':', alpha=0.5, label="Theoretical Limit $(1-p)^2$")

    ax.legend(fontsize=10)
    plt.tight_layout()
    plot_out = os.path.join(OUT_DIR, "recovery_sweep_curves.png")
    fig.savefig(plot_out, dpi=300)
    print(f"[OK] Plot saved to {plot_out}")

if __name__ == "__main__":
    main()
