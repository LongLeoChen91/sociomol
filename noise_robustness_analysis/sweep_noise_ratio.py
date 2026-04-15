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
    """Uniformly sort IDs and Arms to create a directionless unique edge key for strict matching."""
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
    """Returns TP, FP, FN, Precision, Recall, F1"""
    df_pred = pd.read_csv(pred_csv)
    df_gt = pd.read_csv(gt_csv)

    gt_keys = set([extract_edge_key(row) for _, row in df_gt.iterrows()])
    pred_keys = set()
    for _, row in df_pred.iterrows():
        if pd.isna(row['L_nm']): continue
        pred_keys.add(extract_edge_key(row))

    tp = len(pred_keys.intersection(gt_keys))
    fp = len(pred_keys - gt_keys)
    fn = len(gt_keys - pred_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Sweep Noise Ratio across different cases.")
    parser.add_argument("--case", type=str, default="PolysomeManual_1", help="Case name from cases_config.py")
    parser.add_argument("--noise-ratios", type=float, nargs="+", default=list(np.arange(0.0, 1.01, 0.05)), help="List of ratios to sweep (0.0 to 1.0)")
    parser.add_argument("--save-stars", action="store_true", help="Save the sampled STAR files for each ratio in a subfolder")
    args = parser.parse_args()

    config = get_case(args.case)
    print(f"\n[INFO] Starting Noise Sweep for Case: {config['label']}")
    print(f"[INFO] Source STAR: {config['source_star']}")

    OUT_DIR = os.path.join(_SCRIPT_DIR, "sweep_outputs", args.case)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if args.save_stars:
        sampled_stars_dir = os.path.join(OUT_DIR, "sampled_stars")
        os.makedirs(sampled_stars_dir, exist_ok=True)

    df_all = starfile.read(config['source_star'])
    
    # 根据 'class' 切分信号与噪声（噪声颗粒被标记为 class 99）
    # 注意：starfile 库会自动剥离 '_class' 的下划线
    df_signal = df_all[df_all['class'] != 99].copy()
    df_noise_pool = df_all[df_all['class'] == 99].copy()

    print(f"[INFO] Baseline Signal Particles: {len(df_signal)}")
    print(f"[INFO] Noise Pool Particles: {len(df_noise_pool)}")

    results = []

    for ratio in args.noise_ratios:
        print(f"\n==============================================")
        print(f"--- Running Sweep Node: [{ratio*100:3.0f}% Noise Component] ---")
        
        # 1. 采样噪声并重组
        num_noise = int(len(df_noise_pool) * ratio)
        df_noise_sampled = df_noise_pool.sample(n=num_noise, random_state=42)
        df_merged = pd.concat([df_signal, df_noise_sampled], ignore_index=True)
        
        # 2. 写入临时 STAR 并进行预测
        if args.save_stars:
            temp_star = os.path.join(sampled_stars_dir, f"sampled_noise_{ratio*100:03.0f}pct.star")
        else:
            temp_star = os.path.join(OUT_DIR, f"temp_input_r{ratio:.1f}.star")
            
        temp_annotated = os.path.join(OUT_DIR, f"temp_annotated_r{ratio:.1f}.star")
        temp_edges = os.path.join(OUT_DIR, f"temp_edges_r{ratio:.1f}.csv")
        
        starfile.write(df_merged, temp_star, overwrite=True)
        
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
        print(f" -> Result: TP={tp}, FP={fp}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        results.append({
            "Noise_Ratio": ratio,
            "Total_Particles": len(df_merged),
            "Noise_Particles": num_noise,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        })
        
        # 清理临时文件
        if not args.save_stars and os.path.exists(temp_star): 
            os.remove(temp_star)
        if os.path.exists(temp_annotated): os.remove(temp_annotated)

    # ==========================================
    # 4. 保存报表与绘图
    # ==========================================
    df_results = pd.DataFrame(results)
    csv_out = os.path.join(OUT_DIR, "noise_sweep_metrics.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n[OK] Metrics saved to {csv_out}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_results["Noise_Ratio"] * 100, df_results["Recall"], marker='o', color='forestgreen', linewidth=2, label="Recall")
    ax.plot(df_results["Noise_Ratio"] * 100, df_results["Precision"], marker='s', color='crimson', linewidth=2, label="Precision")
    ax.plot(df_results["Noise_Ratio"] * 100, df_results["F1_Score"], marker='D', color='black', linestyle='--', linewidth=2, label="F1 Score")
    
    ax.set_title(f"Performance vs. Noise Ratio ({args.case})", fontsize=14)
    ax.set_xlabel("Noise Particles Ratio ($N_{noise} / MaxNoise$)", fontsize=12)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-5, 105)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=11)
    
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(df_results["Noise_Ratio"] * 100)
    ax2.set_xticklabels(df_results["Total_Particles"].astype(str), rotation=45, fontsize=8)
    ax2.set_xlabel("Total Particles (Signal + Noise)", fontsize=10)

    plt.tight_layout()
    plot_out = os.path.join(OUT_DIR, "noise_sweep_curves.png")
    fig.savefig(plot_out, dpi=300)
    print(f"[OK] Plot saved to {plot_out}")

if __name__ == "__main__":
    main()
