import sys
import os
import re
import subprocess
import matplotlib.pyplot as plt

from sweep_config import get_sweep_paths

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir, pred_script, eval_script, truth_csv, pred_csv = get_sweep_paths(base_dir)

lp_values = list(range(5, 55, 5))
precisions = []
recalls = []
f1_scores = []

print(f"Sweeping LP_NM over: {lp_values}")

for lp in lp_values:
    print(f"\n--- Testing LP_NM = {lp} ---")
    
    # 1. Modify the prediction script
    with open(pred_script, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace LP_NM value
    # Regex looks for LP_NM = <number>
    new_content = re.sub(r'LP_NM\s*=\s*[\d.]+', f'LP_NM = {lp}', content)
    
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

# Restore the original config to 50
with open(pred_script, "r", encoding="utf-8") as f:
    content = f.read()
new_content = re.sub(r'LP_NM\s*=\s*[\d.]+', f'LP_NM = 50', content)
with open(pred_script, "w", encoding="utf-8") as f:
    f.write(new_content)

# 5. Plotting
plt.figure(figsize=(8, 6))
plt.plot(lp_values, precisions, marker='o', label='Precision', color='blue')
plt.plot(lp_values, recalls, marker='s', label='Recall', color='green')
plt.plot(lp_values, f1_scores, marker='^', label='F1 Score', color='red')

plt.title('Performance Metrics vs LP_NM')
plt.xlabel('Persistence Length (LP_NM)')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

out_plot = os.path.join(exp_dir, "LP_NM_sweep.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
print(f"\nSweep complete! Plot saved to {out_plot}")
