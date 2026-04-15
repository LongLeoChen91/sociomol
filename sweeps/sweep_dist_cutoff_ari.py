import sys
import os
import re
import subprocess
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import starfile
from sklearn.metrics import adjusted_rand_score

from sweep_config import get_sweep_paths

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir, pred_script, eval_script, truth_csv, pred_csv = get_sweep_paths(base_dir)

dist_values = list(np.arange(1, 70.5, 0.5))
ari_scores = []

print(f"Sweeping DIST_CUTOFF_NM over: {dist_values}")

for dist in dist_values:
    print(f"\n--- Testing DIST_CUTOFF_NM = {dist} ---")
    
    # 1. Modify the prediction script
    with open(pred_script, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace DIST_CUTOFF_NM value
    # Regex looks for DIST_CUTOFF_NM = <number>
    new_content = re.sub(r'DIST_CUTOFF_NM\s*=\s*[\d.]+', f'DIST_CUTOFF_NM = {dist}', content)
    
    with open(pred_script, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    # 2. Run the prediction script
    print("Running prediction...")
    subprocess.run([sys.executable, pred_script], cwd=exp_dir, check=True, stdout=subprocess.DEVNULL)
    
    # 3. Read annotated STAR file and calculate ARI
    print("Extracting classes and computing ARI...")
    # Find the newly generated annotated star file
    annotated_files = glob.glob(os.path.join(exp_dir, "*_annotated.star"))
    if not annotated_files:
        print("[ERROR] Annotated STAR file not found in experiment directory!")
        sys.exit(1)
        
    # Pick the most recently modified one just to be safe
    annotated_files.sort(key=os.path.getmtime, reverse=True)
    annotated_star_path = annotated_files[0]
    
    try:
        raw = starfile.read(annotated_star_path, always_dict=True)
        df = next(iter(raw.values()))
        
        gt_labels = df['class'].to_numpy(dtype=int)
        # Using fillna(-1) to handle perfectly unassigned/isolated points if they happen to be NaNs
        pred_labels = df['rlnLC_ChainComponent'].fillna(-1).to_numpy(dtype=int)
        
        ari = adjusted_rand_score(gt_labels, pred_labels)
        print(f"-> ARI: {ari:.4f}")
        ari_scores.append(ari)
        
    except Exception as e:
        print(f"[ERROR] Failed to calculate ARI: {e}")
        sys.exit(1)

# Restore the original config to 30
with open(pred_script, "r", encoding="utf-8") as f:
    content = f.read()
new_content = re.sub(r'DIST_CUTOFF_NM\s*=\s*[\d.]+', f'DIST_CUTOFF_NM = 30', content)
with open(pred_script, "w", encoding="utf-8") as f:
    f.write(new_content)

# --- Save data to CSV ---
print("\nSaving sweep metrics to CSV...")
valid_len = len(ari_scores)
df_out = pd.DataFrame({
    "DIST_CUTOFF_NM": dist_values[:valid_len],
    "ARI_Score": ari_scores
})

csv_out = os.path.join(exp_dir, "DIST_CUTOFF_sweep_ARI.csv")
df_out.to_csv(csv_out, index=False)
print(f"[OK] CSV saved to {csv_out}\n")

# 4. Plotting
plt.figure(figsize=(8, 6))
plt.plot(dist_values, ari_scores, marker='o', label='ARI', color='purple', linewidth=2.5)

plt.title('Topology Clustering Performance vs DIST_CUTOFF_NM', fontsize=14, pad=15)
plt.xlabel('Distance Cutoff (DIST_CUTOFF_NM) [nm]', fontsize=12)
plt.ylabel('Adjusted Rand Index (ARI)', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)

# Mark the peak of the curve
if len(ari_scores) > 0:
    ari_arr = np.array(ari_scores)
    best_idx = int(np.argmax(ari_arr))
    best_dist = dist_values[best_idx]
    best_ari = ari_scores[best_idx]

    plt.scatter([best_dist], [best_ari], color='purple', s=60, zorder=5)
    plt.annotate(f"Peak ARI = {best_ari:.3f}\n(dist = {best_dist} nm)",
                 xy=(best_dist, best_ari), xytext=(0, -35),
                 textcoords="offset points", ha='center', va='top',
                 fontsize=10, fontweight='bold', color='purple',
                 arrowprops=dict(arrowstyle="->", color="purple", lw=1))

plt.legend(fontsize=11)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

out_plot = os.path.join(exp_dir, "DIST_CUTOFF_sweep_ARI.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
print(f"\nSweep complete! Plot saved to {out_plot}")
