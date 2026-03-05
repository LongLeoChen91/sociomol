import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# ==== Input/Output settings ====
in_path = "Nucleosome_coords_from_clustered_Reset_Z48_deduplicated_subtomo_coords_C2cc5.star"   # Input STAR file
out_path = "Nucleosome_coords_with_CCCweights.star"  # Output STAR file with weights
target_col_name = "_rlnLC_CCC"  # Column to normalize

# ==== Step 1: Read the STAR file ====
with open(in_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# ==== Step 2: Find the data_particles block ====
data_start_idx = None
header_start_idx = None
header_end_idx = None

for i, line in enumerate(lines):
    if line.strip().startswith("data_particles"):
        header_start_idx = i + 1
        j = header_start_idx
        # skip blank lines
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        # skip optional 'loop_'
        if j < len(lines) and lines[j].strip().startswith("loop_"):
            j += 1
        # collect header column lines (start with '_')
        header_cols = []
        while j < len(lines) and lines[j].strip().startswith("_"):
            header_cols.append(lines[j])
            j += 1
        header_end_idx = j
        # first non-empty line after header is the first data row
        while header_end_idx < len(lines) and lines[header_end_idx].strip() == "":
            header_end_idx += 1
        data_start_idx = header_end_idx
        break

assert data_start_idx is not None, "Could not find data_particles block."

# ==== Step 3: Parse header columns ====
col_names = [col_line.strip().split()[0] for col_line in header_cols]

# Find the index of the target column
if target_col_name in col_names:
    target_idx = col_names.index(target_col_name)
else:
    raise RuntimeError(f"Column {target_col_name} not found in STAR file. Available: {col_names}")

# ==== Step 4: Load data rows ====
data_rows = []
k = data_start_idx
while k < len(lines):
    line = lines[k]
    # stop when a new data_ block starts
    if line.strip().startswith("data_") and k != data_start_idx:
        break
    if line.strip() and not line.strip().startswith("#"):
        toks = line.strip().split()
        # accept rows matching header length
        if len(toks) == len(col_names):
            data_rows.append(toks)
    k += 1

df = pd.DataFrame(data_rows, columns=col_names)

# ==== Step 5: Extract and normalize the target column ====
# Extract CCC as float array
CCC = df[target_col_name].astype(float).values

# Basic statistics (for logging/QA)
min_val, max_val, mean_val, std_val = CCC.min(), CCC.max(), CCC.mean(), CCC.std()
print(f"[CCC] min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")

# Method A: Min-max normalization
if max_val > min_val:
    w_minmax = (CCC - min_val) / (max_val - min_val)
else:
    # Degenerate case: all CCC are equal
    w_minmax = np.zeros_like(CCC)

# Method B: Z-score + sigmoid mapping (kept for downstream use)
z = (CCC - mean_val) / (std_val if std_val > 0 else 1.0)
beta = 2.0  # Controls logistic steepness
w_sigmoid = 1 / (1 + np.exp(-beta * z))

# Method C: Quantile (rank-based) (kept for downstream use)
w_rank = rankdata(CCC, method="average") / len(CCC)

# ==== Step 6: Side-by-side plots: Original CCC vs Min-max normalized ====
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: original CCC distribution
axes[0].hist(CCC, bins=50, alpha=0.7)
axes[0].set_title("Original CCC")
axes[0].set_xlabel("CCC")
axes[0].set_ylabel("Frequency")

# Right: Min-max normalized distribution
axes[1].hist(w_minmax, bins=50, alpha=0.7)
axes[1].set_title("Min-max normalized CCC")
axes[1].set_xlabel("Normalized value [0-1]")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# ==== Step 7: Append new weight columns to the STAR file ====
new_cols = ["_rlnLC_CCC_MinMax", "_rlnLC_CCC_SigmoidBeta2", "_rlnLC_CCC_Quantile"]

# Helper: extract trailing '#N' index from a header line
def next_hash_index(col_line: str):
    m = re.search(r"#\s*(\d+)\s*$", col_line)
    return int(m.group(1)) if m else None

current_indices = [next_hash_index(c) for c in header_cols]
start_idx_num = max([idx for idx in current_indices if idx is not None]) + 1

# Extend header with new columns and running indices
extended_header_cols = header_cols.copy()
for offset, name in enumerate(new_cols):
    extended_header_cols.append(f"{name} #{start_idx_num + offset}\n")

# Add the new weight columns to the dataframe (string conversion not required here)
df[new_cols[0]] = w_minmax
df[new_cols[1]] = w_sigmoid
df[new_cols[2]] = w_rank

# Convert a row to STAR-format line (whitespace-separated)
def row_to_line(row_vals):
    return " ".join(str(v) for v in row_vals) + "\n"

# Rebuild STAR text
out_lines = []
# Keep original content up to header_start_idx
out_lines.extend(lines[:header_start_idx])
# Re-emit a clean loop_ and the extended headers
out_lines.append("loop_\n")
out_lines.extend(extended_header_cols)
# Emit data rows with the appended weight columns
for _, r in df.iterrows():
    vals = [r[c] for c in col_names + new_cols]
    out_lines.append(row_to_line(vals))
# Append any remaining content after the consumed data block
out_lines.extend(lines[k:])

# ==== Step 8: Save new STAR file ====
with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(out_lines)

print(f"Done! New STAR file with weights saved to: {out_path}")
