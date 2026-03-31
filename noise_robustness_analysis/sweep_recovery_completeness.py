import os
import sys
import numpy as np
import pandas as pd
import starfile
import matplotlib.pyplot as plt

# Link project modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_SCRIPT_DIR)

from linker_prediction.pipeline import run_prediction_pipeline

# ==========================================
# 1. Configs & Paths
# ==========================================
SOURCE_STAR = os.path.join(_REPO_ROOT, "experiments", "PolysomeManual_1_Noise", "Avg_Linkers.star")
GROUND_TRUTH_CSV = os.path.join(_REPO_ROOT, "experiments", "PolysomeManual_1", "GroundTruth_edges_PoM1.csv")

OUT_DIR = os.path.join(_SCRIPT_DIR, "recovery_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 物理模型参数 (使用最新的 25nm Cutoff 以匹配实验)
PIXEL_SIZE_A  = 1.96
DIST_CUTOFF_NM = 25
P_THRESHOLD = 0.0
LP_NM = 1.5
L0_NM = 20.0
THETA0_DEG = 45.0
W_WLC = 0.0
W_L = 1.0
W_TH = 1.0
W_L_SQ = 0.0
W_TH_SQ = 0.0
L_IDEAL_NM = 0.0
L_STD_NM = 20.0
THETA_STD_DEG = 90.0
PORT_PAIRING = "complement"
THETA_MODE = "alpha_sum"
MAX_HALF_BENDING_DEG = 90.0

# 移除比例：从 0% 到 50%，步长 5%
REMOVAL_RATIOS = np.arange(0.0, 0.51, 0.05)

# ==========================================
# 2. 评估体系函数 (基于 Particle ID 的 Keys)
# ==========================================
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

    # 这里的 Recall 是 Global Recall，即相对于“完整数据集”的漏失程度
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1

# ==========================================
# 3. 核心循环：模拟不完全恢复 (Simulate Incomplete Recovery)
# ==========================================
def main():
    print(f"[INFO] Source STAR: {SOURCE_STAR}")
    df_all = starfile.read(SOURCE_STAR)
    
    # 彻底去掉所有 noise class 99，只留 signal
    df_signal = df_all[df_all['class'] != 99].copy()
    print(f"[INFO] Clean Signal Pool: {len(df_signal)} particles")

    results = []

    for ratio in REMOVAL_RATIOS:
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
            theta_mode=THETA_MODE,
            max_half_bending_deg=MAX_HALF_BENDING_DEG
        )
        
        # 3. 评估指标 (Global Metrics)
        tp, fp, fn, precision, recall, f1 = evaluate_predictions(temp_edges, GROUND_TRUTH_CSV)
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
    csv_out = os.path.join(_SCRIPT_DIR, "recovery_sweep_metrics.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n[OK] Metrics saved to {csv_out}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_results["Removal_Ratio"] * 100, df_results["Recall"], marker='o', color='forestgreen', linewidth=2, label="Recall (Recovery)")
    ax.plot(df_results["Removal_Ratio"] * 100, df_results["Precision"], marker='s', color='crimson', linewidth=2, label="Precision")
    ax.plot(df_results["Removal_Ratio"] * 100, df_results["F1_Score"], marker='D', color='black', linestyle='--', linewidth=2, label="F1 Score")
    
    ax.set_title("Robustness to Incomplete Particle Recovery (In-silico Deletion)", fontsize=13)
    ax.set_xlabel("Percentage of Particles Removed (%)", fontsize=12)
    ax.set_ylabel("Global Metric Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2, 55)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # 标注理论存活曲线作为参考：Theoretical Edge Recovery ~ (1-p)^2
    p = df_results["Removal_Ratio"]
    theoretical_recall = (1.0 - p)**2
    ax.plot(df_results["Removal_Ratio"] * 100, theoretical_recall, color='gray', linestyle=':', alpha=0.5, label="Theoretical Limit $(1-p)^2$")

    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plot_out = os.path.join(_SCRIPT_DIR, "recovery_sweep_curves.png")
    fig.savefig(plot_out, dpi=300)
    print(f"[OK] Plot saved to {plot_out}")
    plt.show()

if __name__ == "__main__":
    main()
