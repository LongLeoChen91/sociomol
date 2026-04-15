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

theta_values = list(range(10, 95, 5))
precisions = []
recalls = []
f1_scores = []

print(f"Sweeping THETA0_DEG over: {theta_values}")

for theta in theta_values:
    print(f"\n--- Testing THETA0_DEG = {theta}.0 ---")
    
    # 1. Modify the prediction script
    with open(pred_script, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace THETA0_DEG value
    # Regex looks for THETA0_DEG = <number>
    new_content = re.sub(r'THETA0_DEG\s*=\s*[\d.]+', f'THETA0_DEG = {float(theta)}', content)
    
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
    # Format typically:
    # Precision : 0.9231
    # Recall    : 0.9057
    # F1 Score  : 0.9143
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
        print("Failed to parse metrics!")
        print(output)
        sys.exit(1)
        
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Restore the original config to 90.0
with open(pred_script, "r", encoding="utf-8") as f:
    content = f.read()
new_content = re.sub(r'THETA0_DEG\s*=\s*[\d.]+', f'THETA0_DEG = 45.0', content)
with open(pred_script, "w", encoding="utf-8") as f:
    f.write(new_content)

# --- Save data to CSV ---
print("\nSaving sweep metrics to CSV...")
df_out = pd.DataFrame({
    "THETA0_DEG": theta_values,
    "Precision": precisions,
    "Recall": recalls,
    "F1_Score": f1_scores
})

csv_out = os.path.join(exp_dir, "THETA0_sweep_F1.csv")
df_out.to_csv(csv_out, index=False)
print(f"[OK] CSV saved to {csv_out}\n")

# 5. Plotting
plt.figure(figsize=(8, 6))
plt.plot(theta_values, precisions, marker='o', label='Precision', color='blue')
plt.plot(theta_values, recalls, marker='s', label='Recall', color='green')
plt.plot(theta_values, f1_scores, marker='^', label='F1 Score', color='red')

plt.title('Performance Metrics vs THETA0_DEG')
plt.xlabel('Reference Angle Tolerance (THETA0_DEG)')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

out_plot = os.path.join(exp_dir, "THETA0_sweep.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
print(f"\nSweep complete! Plot saved to {out_plot}")
