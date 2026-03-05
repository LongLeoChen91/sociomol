import pandas as pd
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

# === Step 1: Load .star file and find the data section ===
file_path = "run_data.star"  # Change to your .star file path

# Read the whole file
with open(file_path, "r") as f:
    lines = f.readlines()

# Find the start of data_particles section
data_start = 0
for i, line in enumerate(lines):
    if line.strip().startswith("data_particles"):
        # find the first line of actual data after loop_ and column names
        for j in range(i + 1, len(lines)):
            if lines[j].strip().startswith("loop_"):
                continue
            if lines[j].strip().startswith("_"):
                continue
            if lines[j].strip() == "":
                continue
            data_start = j
            break
        break

# === Step 2: Read data into DataFrame ===
# split lines by whitespace, skip empty and comment lines
data = [l.strip().split() for l in lines[data_start:] if l.strip() and not l.startswith("#")]
df = pd.DataFrame(data)

# === Step 3: Extract the _rlnLogLikeliContribution column ===
# It's the 21st column in your file (index starts from 0)
col_idx = 21 - 1
loglik = df[col_idx].astype(float)

# Compute basic statistics
min_val, max_val = loglik.min(), loglik.max()
mean_val, std_val = loglik.mean(), loglik.std()
print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")

# === Step 4: Normalize to [0, 1] ===
L = loglik.values

# Method A: Min-max normalization
w_minmax = (L - L.min()) / (L.max() - L.min())

# Method B: Z-score + sigmoid mapping
z = (L - L.mean()) / L.std()
beta = 2.0  # control sharpness
w_sigmoid = 1 / (1 + np.exp(-beta * z))

# Method C: Quantile normalization (rank-based)
w_rank = rankdata(L) / len(L)

# === Step 5: Plot the weight distributions ===
plt.figure(figsize=(8,5))
plt.hist(w_minmax, bins=50, alpha=0.5, label='Min-max')
plt.hist(w_sigmoid, bins=50, alpha=0.5, label='Z-score+Sigmoid')
plt.hist(w_rank, bins=50, alpha=0.5, label='Quantile')
plt.legend()
plt.xlabel("Normalized weight")
plt.ylabel("Frequency")
plt.title("Comparison of normalization methods")
plt.tight_layout()
plt.show()

# === Step 6: Side-by-side comparison of original loglik vs Min-max normalized ===
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# # Left: original loglik distribution
# axes[0].hist(L, bins=50, alpha=0.7, color='steelblue')
# axes[0].set_title("Original log-likelihood")
# axes[0].set_xlabel("log-likelihood")
# axes[0].set_ylabel("Frequency")

# # Right: Min-max normalized distribution
# axes[1].hist(w_minmax, bins=50, alpha=0.7, color='darkorange')
# axes[1].set_title("Min-max normalized")
# axes[1].set_xlabel("Normalized value [0-1]")
# axes[1].set_ylabel("Frequency")

# plt.tight_layout()
# plt.savefig("normalized_weights_comparison.png", dpi=300)  # High-resolution PNG
# plt.show()
