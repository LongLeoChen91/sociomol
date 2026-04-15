import sys
import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from sweep_config import get_sweep_paths

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir, pred_script, eval_script, truth_csv, pred_csv = get_sweep_paths(base_dir)

# Sweeping from 30 degrees to 180 degrees
deg_values = list(range(10, 185, 10))
precisions = []
recalls = []
f1_scores = []

print(f"Sweeping MAX_HALF_BENDING_DEG over: {deg_values}")

for deg in deg_values:
    print(f"\n--- Testing MAX_HALF_BENDING_DEG = {deg}.0 ---")
    
    # 1. Modify the prediction script
    with open(pred_script, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace MAX_HALF_BENDING_DEG value
    new_content = re.sub(r'MAX_HALF_BENDING_DEG\s*=\s*[\d.]+', f'MAX_HALF_BENDING_DEG = {float(deg)}', content)
    
    with open(pred_script, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    # 2. Run the prediction script
    print("Running prediction...")
    subprocess.run([sys.executable, pred_script], cwd=exp_dir, check=True, stdout=subprocess.DEVNULL)
    
    # 3. Run evaluation
    print("Running evaluation...")
    eval_cmd = [
        sys.executable, eval_script,
        "--truth", truth_csv,
        "--pre", pred_csv
    ]
    result = subprocess.run(eval_cmd, cwd=base_dir, capture_output=True, text=True, check=True)
    output = result.stdout
    
    # 4. Parse metrics
    precision = None
    recall = None
    f1 = None
    
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("Precision"):
            precision = float(line.split(":")[1].strip())
        elif line.startswith("Recall"):
            recall = float(line.split(":")[1].strip())
        elif line.startswith("F1 Score"):
            f1 = float(line.split(":")[1].strip())
            
    if precision is None or recall is None or f1 is None:
        print("Failed to parse metrics! Empty prediction.")
        precision, recall, f1 = 0.0, 0.0, 0.0
        
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Restore the original config to 90.0
with open(pred_script, "r", encoding="utf-8") as f:
    content = f.read()
new_content = re.sub(r'MAX_HALF_BENDING_DEG\s*=\s*[\d.]+', f'MAX_HALF_BENDING_DEG = 90.0', content)
with open(pred_script, "w", encoding="utf-8") as f:
    f.write(new_content)

# --- Save data to CSV ---
print("\nSaving sweep metrics to CSV...")
df_out = pd.DataFrame({
    "MAX_HALF_BENDING_DEG": deg_values,
    "Precision": precisions,
    "Recall": recalls,
    "F1_Score": f1_scores
})

csv_out = os.path.join(exp_dir, "MAX_HALF_BENDING_DEG_sweep_F1.csv")
df_out.to_csv(csv_out, index=False)
print(f"[OK] CSV saved to {csv_out}\n")

# 5. Plotting
plt.figure(figsize=(8, 6))
plt.plot(deg_values, precisions, marker='o', label='Precision', color='blue')
plt.plot(deg_values, recalls, marker='s', label='Recall', color='green')
plt.plot(deg_values, f1_scores, marker='^', label='F1 Score', color='red')

plt.title('Performance Metrics vs MAX_HALF_BENDING_DEG')
plt.xlabel('Maximum Allowed Half-Bending Angle (deg)')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

out_plot = os.path.join(exp_dir, "MAX_HALF_BENDING_DEG_sweep.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
print(f"\nSweep complete! Plot saved to {out_plot}")
