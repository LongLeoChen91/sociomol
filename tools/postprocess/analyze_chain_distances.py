#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_chain_distances.py — Measure and plot the minimum distance between chains.

Reads a STAR file (with rlnLC_ChainComponent and rlnLC_ComponentSize).
Only considers chains with Size >= min_size (default 2).
For every pair of valid chains in a tomogram, calculates the minimum
Euclidean distance between their ribosomes.
Outputs a CSV of pairwise distances and a histogram plot, with an optional cutoff.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import starfile

def main():
    parser = argparse.ArgumentParser(description="Analyze distances between polysome chains.")
    parser.add_argument("--star", required=True, help="Input STAR file (e.g. ranked_particles.star).")
    parser.add_argument("--out-csv", required=True, help="Output CSV file for pairwise distances.")
    parser.add_argument("--out-plot", required=True, help="Output histogram plot (.png).")
    parser.add_argument("--pixel-size", type=float, required=True, help="Pixel size in Angstroms/pixel.")
    parser.add_argument("--min-size", type=int, default=2, help="Minimum chain size to be considered (default: 2).")
    parser.add_argument("--cutoff", type=float, default=30.0, help="Maximum distance cutoff in nm for the plot (default: 30 nm).")
    
    args = parser.parse_args()

    # ---- Read Data ----
    print(f"Reading {args.star}...")
    try:
        data = starfile.read(args.star, always_dict=True)
        if "particles" in data:
            df = data["particles"]
        else:
            df = list(data.values())[0]
    except Exception as e:
        print(f"[ERROR] Failed to read STAR file: {e}")
        sys.exit(1)
        
    required_cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnLC_ChainComponent", "rlnLC_ComponentSize"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Missing required column: {col}")
            sys.exit(1)

    # Convert columns to correct types
    for col in ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnLC_ComponentSize"]:
        df[col] = pd.to_numeric(df[col])
        
    df["rlnLC_ChainComponent"] = df["rlnLC_ChainComponent"].astype(str)

    # Filter for real chains
    chains_df = df[df["rlnLC_ComponentSize"] >= args.min_size]
    
    if chains_df.empty:
        print(f"[WARN] No chains found with size >= {args.min_size}. Exiting gracefully.")
        pd.DataFrame(columns=["ChainA", "ChainA_Size", "ChainB", "ChainB_Size", "ChainDistance_nm"]).to_csv(args.out_csv, index=False)
        plt.figure()
        plt.title("No chains found")
        plt.savefig(args.out_plot)
        sys.exit(0)

    # Group by chain component
    grouped = chains_df.groupby("rlnLC_ChainComponent")
    chain_ids = list(grouped.groups.keys())
    
    n_chains = len(chain_ids)
    print(f"Found {n_chains} chains with size >= {args.min_size}.")
    
    if n_chains < 2:
        print(f"[WARN] Need at least 2 chains to compute pairwise distances. Found {n_chains}. Exiting.")
        pd.DataFrame(columns=["ChainA", "ChainA_Size", "ChainB", "ChainB_Size", "ChainDistance_nm"]).to_csv(args.out_csv, index=False)
        plt.figure()
        plt.title("Not enough chains")
        plt.savefig(args.out_plot)
        sys.exit(0)

    # Compute distances
    print("Computing pairwise chain distances...")
    results = []
    
    for i in range(n_chains):
        idx_i = grouped.groups[chain_ids[i]]
        coords_i = chains_df.loc[idx_i, ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].values
        size_i = chains_df.loc[idx_i[0], "rlnLC_ComponentSize"]
        for j in range(i + 1, n_chains):
            idx_j = grouped.groups[chain_ids[j]]
            coords_j = chains_df.loc[idx_j, ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].values
            size_j = chains_df.loc[idx_j[0], "rlnLC_ComponentSize"]
            
            # calculate matrix of pairwise distances
            dist_matrix = cdist(coords_i, coords_j) * args.pixel_size
            chain_dist_nm = np.min(dist_matrix) / 10.0
            
            results.append({
                "ChainA": chain_ids[i],
                "ChainA_Size": size_i,
                "ChainB": chain_ids[j],
                "ChainB_Size": size_j,
                "ChainDistance_nm": chain_dist_nm
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out_csv, index=False)
    print(f"Saved distances to: {args.out_csv}")
    
    # Plotting
    cutoff_nm = args.cutoff
    filtered_distances = res_df[res_df["ChainDistance_nm"] <= cutoff_nm]["ChainDistance_nm"]
    
    plt.figure(figsize=(8, 6))
    if not filtered_distances.empty:
        # choose a reasonable bin size, e.g., 2 nm
        bins = np.arange(0, cutoff_nm + 2, 2)
        plt.hist(filtered_distances, bins=bins, color="#8b0000", edgecolor="black", alpha=0.8)
    
    plt.title(f"Distance Between Polysome Chains (Size $\\geq$ {args.min_size})\nCutoff: {cutoff_nm} nm")
    plt.xlabel("Distance (nm)")
    plt.ylabel("Frequency (Pairs of Chains)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=300)
    print(f"Saved histogram plot to: {args.out_plot}")
    
if __name__ == "__main__":
    main()
