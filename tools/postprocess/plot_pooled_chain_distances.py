#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pooled_chain_distances.py — Pool and plot all inter_chain_distances.csv files.

Finds all inter_chain_distances.csv files in the subdirectories of a given input directory,
concatenates them, and plots a global histogram of distances between chains.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Pool and plot all chain distances.")
    parser.add_argument("--csv-dir", required=True, help="Directory containing tomo subdirectories with inter_chain_distances.csv files.")
    parser.add_argument("--out-csv", required=True, help="Output global CSV file for all pooled pairwise distances.")
    parser.add_argument("--out-plot", required=True, help="Output global histogram plot (.png).")
    parser.add_argument("--cutoff", type=float, default=30.0, help="Maximum distance cutoff in nm for the plot (default: 30 nm).")
    
    args = parser.parse_args()

    base_dir = Path(args.csv_dir)
    if not base_dir.is_dir():
        print(f"[ERROR] Directory not found: {base_dir}")
        sys.exit(1)

    print(f"Searching for inter_chain_distances.csv in {base_dir}...")
    csv_files = list(base_dir.rglob("inter_chain_distances.csv"))
    
    if not csv_files:
        print("[WARN] No inter_chain_distances.csv files found. Exiting gracefully.")
        pd.DataFrame(columns=["Tomo", "ChainA", "ChainA_Size", "ChainB", "ChainB_Size", "ChainDistance_nm"]).to_csv(args.out_csv, index=False)
        plt.figure()
        plt.title("No distance data found")
        plt.savefig(args.out_plot)
        sys.exit(0)

    # Read and concatenate all CSVs
    all_dfs = []
    total_pairs = 0
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # Add tomo name as a column for traceability (parent dir name)
            tomo_name = file_path.parent.name
            df.insert(0, "Tomo", tomo_name)
            all_dfs.append(df)
            total_pairs += len(df)
            print(f"  Loaded {len(df)} pairs from {tomo_name}")
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")

    if not all_dfs:
        print("[WARN] All distance CSVs are empty. Exiting gracefully.")
        pd.DataFrame(columns=["Tomo", "ChainA", "ChainA_Size", "ChainB", "ChainB_Size", "ChainDistance_nm"]).to_csv(args.out_csv, index=False)
        plt.figure()
        plt.title("No chains found across all tomograms")
        plt.savefig(args.out_plot)
        sys.exit(0)

    pooled_df = pd.concat(all_dfs, ignore_index=True)
    pooled_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved pooled {len(pooled_df)} pairwise distances to: {args.out_csv}")
    
    # Plotting
    cutoff_nm = args.cutoff
    filtered_distances = pooled_df[pooled_df["ChainDistance_nm"] <= cutoff_nm]["ChainDistance_nm"]
    
    plt.figure(figsize=(8, 6))
    if not filtered_distances.empty:
        # choose a reasonable bin size, e.g., 2 nm
        bins = np.arange(0, cutoff_nm + 2, 2)
        # Using a nice distinct color for the global plot
        plt.hist(filtered_distances, bins=bins, color="#228b22", edgecolor="black", alpha=0.8)
    
    plt.title(f"Global Distance Between Polysome Chains\n(Pooled from {len(all_dfs)} tomograms | Cutoff: {cutoff_nm} nm)")
    plt.xlabel("Distance (nm)")
    plt.ylabel("Frequency (Pairs of Chains)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=300)
    print(f"Saved pooled histogram plot to: {args.out_plot}")
    
if __name__ == "__main__":
    main()
