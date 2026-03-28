import sys
import os
import re
import subprocess
import matplotlib.pyplot as plt

from sweep_config import get_sweep_paths

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir, pred_script, eval_script, truth_csv, pred_csv = get_sweep_paths(base_dir)

l0_values = list(range(5, 55, 5))
precisions = []
recalls = []
f1_scores = []

print(f"Sweeping L0_NM over: {l0_values}")

for l0 in l0_values:
    print(f"\n--- Testing L0_NM = {l0} ---")
    
    # 1. Modify the prediction script
    with open(pred_script, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace L0_NM value
    # Regex looks for L0_NM = <number>
    new_content = re.sub(r'L0_NM\s*=\s*[\d.]+', f'L0_NM = {l0}', content)
    
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

# Restore the original config to 20
with open(pred_script, "r", encoding="utf-8") as f:
    content = f.read()
new_content = re.sub(r'L0_NM\s*=\s*[\d.]+', f'L0_NM = 20', content)
with open(pred_script, "w", encoding="utf-8") as f:
    f.write(new_content)

# 5. Plotting
plt.figure(figsize=(8, 6))
plt.plot(l0_values, precisions, marker='o', label='Precision', color='blue')
plt.plot(l0_values, recalls, marker='s', label='Recall', color='green')
plt.plot(l0_values, f1_scores, marker='^', label='F1 Score', color='red')

plt.title('Performance Metrics vs L0_NM')
plt.xlabel('Ideal Linker Distance (L0_NM)')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

out_plot = os.path.join(exp_dir, "L0_NM_sweep.png")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
print(f"\nSweep complete! Plot saved to {out_plot}")
