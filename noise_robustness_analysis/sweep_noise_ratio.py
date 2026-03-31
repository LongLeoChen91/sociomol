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

OUT_DIR = os.path.join(_SCRIPT_DIR, "sweep_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 物理模型参数 (保持与 PolysomeManual_1_Noise/LC2 相同)
PIXEL_SIZE_A  = 1.96
DIST_CUTOFF_NM = 30
P_THRESHOLD = 0.0  # 沿用用户之前的0截断配置，如需更高精度可自行改为 0.05
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

NOISE_RATIOS = np.arange(0.0, 0.21, 0.02)  # 0.0 到 1.0，步长 0.1

# ==========================================
# 2. 评估体系函数
# ==========================================
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

# ==========================================
# 3. 核心循环：动态注入并预测
# ==========================================
def main():
    print(f"[INFO] Source STAR: {SOURCE_STAR}")
    df_all = starfile.read(SOURCE_STAR)
    
    # 根据 'class' 切分信号与噪声（Polysome 的噪声颗粒被标记为 class 99）
    # 注意：starfile 库会自动剥离 '_class' 的下划线
    df_signal = df_all[df_all['class'] != 99].copy()
    df_noise_pool = df_all[df_all['class'] == 99].copy()

    print(f"[INFO] Baseline Signal Particles: {len(df_signal)}")
    print(f"[INFO] Noise Pool Particles: {len(df_noise_pool)}")

    results = []

    for ratio in NOISE_RATIOS:
        print(f"\n==============================================")
        print(f"--- Running Sweep Node: [{ratio*100:3.0f}% Noise Component] ---")
        
        # 1. 采样噪声并重组
        num_noise = int(len(df_noise_pool) * ratio)
        df_noise_sampled = df_noise_pool.sample(n=num_noise, random_state=42)
        df_merged = pd.concat([df_signal, df_noise_sampled], ignore_index=True)
        
        # 2. 写入临时 STAR 并进行预测
        temp_star = os.path.join(OUT_DIR, f"temp_input_r{ratio:.1f}.star")
        temp_annotated = os.path.join(OUT_DIR, f"temp_annotated_r{ratio:.1f}.star")
        temp_edges = os.path.join(OUT_DIR, f"temp_edges_r{ratio:.1f}.csv")
        
        starfile.write(df_merged, temp_star, overwrite=True)
        
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
        
        # 3. 评估指标
        tp, fp, fn, precision, recall, f1 = evaluate_predictions(temp_edges, GROUND_TRUTH_CSV)
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
        
        # 清理过大的星点文件以节省空间（可选），保留 edges csv 用于回溯
        if os.path.exists(temp_star): os.remove(temp_star)
        if os.path.exists(temp_annotated): os.remove(temp_annotated)

    # ==========================================
    # 4. 保存报表与绘图
    # ==========================================
    df_results = pd.DataFrame(results)
    csv_out = os.path.join(_SCRIPT_DIR, "noise_sweep_metrics.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n[OK] Metrics saved to {csv_out}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_results["Noise_Ratio"] * 100, df_results["Recall"], marker='o', color='forestgreen', linewidth=2, label="Recall")
    ax.plot(df_results["Noise_Ratio"] * 100, df_results["Precision"], marker='s', color='crimson', linewidth=2, label="Precision")
    ax.plot(df_results["Noise_Ratio"] * 100, df_results["F1_Score"], marker='D', color='black', linestyle='--', linewidth=2, label="F1 Score")
    
    ax.set_title("Performance vs. Noise Injection Ratio (PoM1 Dataset)", fontsize=14)
    ax.set_xlabel("Noise Particles Ratio ($N_{noise} / MaxNoise$)", fontsize=12)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2, 22)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=11)
    
    # Annotate X-axis with total particles count on secondary axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(df_results["Noise_Ratio"] * 100)
    ax2.set_xticklabels(df_results["Total_Particles"].astype(str), rotation=45, fontsize=8)
    ax2.set_xlabel("Total Particles (Signal + Noise)", fontsize=10)

    plt.tight_layout()
    plot_out = os.path.join(_SCRIPT_DIR, "noise_sweep_curves.png")
    fig.savefig(plot_out, dpi=300)
    print(f"[OK] Plot saved to {plot_out}")
    plt.show()

if __name__ == "__main__":
    main()
