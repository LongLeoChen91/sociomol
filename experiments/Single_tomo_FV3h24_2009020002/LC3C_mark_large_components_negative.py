#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import starfile
import os

# ====== Paths ======
csv_path   = "chain_component_distribution.csv"
input_star = "Nucleosome_by_LinkerSticks.star" # Linker_Sticks.star; Nucleosome_by_LinkerSticks.star 
output_star = os.path.join(os.path.dirname(input_star),"negComp_gt2_" + os.path.basename(input_star))

# ====== Load CSV of component counts ======
# Expect a CSV where index = component_id, column 'count' = occurrences
# (created previously via: comp_sizes.to_csv("chain_component_distribution.csv", header=["count"]))
comp_df = pd.read_csv(csv_path, index_col=0)

if "count" not in comp_df.columns:
    raise ValueError("CSV must contain a 'count' column.")

# Components with count > 2
targets_idx = comp_df.index[comp_df["count"] > 2]
targets_str = set(targets_idx.astype(str))  # match on string form for robustness

print(f"[INFO] {len(targets_str)} components have count > 2.")

# ====== Load STAR (particles table) ======
data = starfile.read(input_star, always_dict=True)
# pick particles-like block
if "particles" in data:
    df = data["particles"]
else:
    # fallback to first DataFrame block
    df = next((v for v in data.values() if isinstance(v, pd.DataFrame)), None)

if df is None:
    raise ValueError("Could not find a particles table in the STAR file.")

col = "rlnLC_ChainComponent"
if col not in df.columns:
    raise ValueError(f"Column '{col}' not found in the STAR file.")

# ====== Build mask for rows to negate ======
comp_as_str = df[col].astype(str).str.strip()
mask = comp_as_str.isin(targets_str)

# We'll try to negate numeric values; if non-numeric, we leave them unchanged and warn.
comp_as_num = pd.to_numeric(df[col], errors="coerce")

n_to_change = int(mask.sum())
n_numeric   = int((mask & comp_as_num.notna()).sum())
n_nonnum    = n_to_change - n_numeric

print(f"[INFO] Rows to negate: {n_to_change} (numeric: {n_numeric}, non-numeric: {n_nonnum})")

# Negate numeric component IDs for selected rows (set to -abs(value))
df.loc[mask & comp_as_num.notna(), col] = -comp_as_num[mask & comp_as_num.notna()].abs().astype(int)

# (Optional) warn if some targeted rows were non-numeric
if n_nonnum > 0:
    print("[WARN] Some targeted rlnLC_ChainComponent values are non-numeric; left unchanged.")

# ====== Write out new STAR ======
# Preserve the same block name if it existed; otherwise write a simple dict with 'particles'
if "particles" in data:
    data["particles"] = df
    starfile.write(data, output_star, overwrite=True)
else:
    starfile.write({"particles": df}, output_star, overwrite=True)

print(f"[DONE] Wrote: {output_star}")
