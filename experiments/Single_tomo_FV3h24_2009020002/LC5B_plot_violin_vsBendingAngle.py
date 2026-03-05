import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# === Load CSV file ===
csv_path = "DoubleLinker_edges.csv"
df = pd.read_csv(csv_path)

# Read CSV
df = pd.read_csv(csv_path)

# Filter by P > 0.1 before any plotting or binning
if "P" not in df.columns:
    raise ValueError("CSV must contain column 'P' for filtering.")
df = df[df["P"] > 0.01]

# Check if data is empty after filtering
if df.empty:
    raise ValueError("No rows remain after filtering with P > 0.1.")

# Define thresholds for P
thresholds = [20, 40, 60, 80, 100]
labels = [f"theta_deg<{t}" for t in thresholds]

# Extract L_nm values for each threshold group
data_groups = []
for t in thresholds:
    if t == 0.0:
        data_groups.append(df["L_nm"].values)
    else:
        data_groups.append(df[df["theta_deg"] < t]["L_nm"].values)

# === Adjustable parameters ===
area_scale = 3   # control violin width

# Common vertical scale for all violins
global_min = min([data.min() for data in data_groups])
global_max = max([data.max() for data in data_groups])
x_eval = np.linspace(global_min, global_max, 300)

# === Plot equal-area normalized violin plots ===
plt.figure(figsize=(8,6))
positions = range(1, len(data_groups)+1)
colors = plt.cm.Blues(np.linspace(0.4, 1, len(data_groups)))

for pos, data, color, label in zip(positions, data_groups, colors, labels):
    # KDE density estimation on common vertical scale
    kde = gaussian_kde(data)
    density = kde(x_eval)
    
    # Equal-area normalization: each violin integrates to same area
    area = np.trapz(density, x_eval)
    density /= area
    density *= area_scale  # control total violin width
    
    # Draw violin
    plt.fill_betweenx(x_eval, pos - density, pos + density, facecolor=color, alpha=0.6, label=label)
    
    # Mean and median
    mean_val = np.mean(data)
    median_val = np.median(data)
    count = len(data)
    
    # Plot mean and median
    plt.scatter([pos], [mean_val], color="red", marker="o", label="Mean" if pos==1 else "")
    plt.scatter([pos], [median_val], color="blue", marker="s", label="Median" if pos==1 else "")
    
    # Annotate numeric values
    plt.text(pos+0.1, mean_val, f"{mean_val:.2f}", color="red", va="center")
    plt.text(pos+0.1, median_val, f"{median_val:.2f}", color="blue", va="center")
    
    # Print sample size below x-axis
    plt.text(pos, global_min - (global_max-global_min)*0.05, f"n={count}", 
             ha='center', va='top', fontsize=9, color="black")

# Final plot settings
plt.xticks(positions, labels, rotation=20)
plt.ylim(global_min - (global_max-global_min)*0.1, global_max)  # add margin for counts
plt.ylabel("L_nm (nm)")
plt.title("Equal-Area Normalized Violin Plots with Counts")
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

out_file = "violin_plot_equal_area.png"
plt.savefig(out_file, dpi=300)
print(f"Figure saved to {out_file}")

plt.show()
