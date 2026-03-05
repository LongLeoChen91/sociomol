import starfile
import pandas as pd
import numpy as np
import os

# Input STAR file
input_star = "negComp_gt2_Linker_Sticks.star" # negComp_gt2_Linker_Sticks.star; negComp_gt2_Nucleosome_by_LinkerSticks.star
output_dir = "split_by_length"
os.makedirs(output_dir, exist_ok=True)

# Read STAR file
star_data = starfile.read(input_star)
if isinstance(star_data, dict):
    df = star_data.get("particles", None)
else:
    df = star_data

if df is None or "rlnStickLength" not in df.columns:
    raise ValueError("STAR file does not contain 'rlnStickLength' column.")

# Define length bins
bins = [0, 5, 10, 15, 20, 25, 30, np.inf]
labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "gt30"]

# Split data by bins
lengths = df["rlnStickLength"].values
bin_indices = np.digitize(lengths, bins) - 1  # bins index for each row

for idx, label in enumerate(labels):
    subset = df[bin_indices == idx]
    if len(subset) > 0:
        base = os.path.splitext(os.path.basename(input_star))[0]
        out_path = os.path.join(output_dir, f"{base}_{label}.star")
        starfile.write({"particles": subset}, out_path, overwrite=True)
        print(f"Saved {len(subset)} rows to {out_path}")
    else:
        print(f"No rows for bin {label}, skipping.")

print("Splitting done.")
