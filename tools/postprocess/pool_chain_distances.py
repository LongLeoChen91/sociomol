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
    parser.add_argument("--csv-dir", required=True, help="Directory containing tomo subdirectories with distance CSV files.")
    parser.add_argument("--out-csv", required=True, help="Output global CSV file for all pooled pairwise distances.")
    parser.add_argument("--out-plot", required=True, help="Output global histogram plot (.png).")
    parser.add_argument("--out-nn-csv", help="Optional output global CSV for pooled nearest chain distances.")
    parser.add_argument("--out-nn-plot", help="Optional output global histogram plot for nearest chain distances.")
    parser.add_argument("--cutoff", type=float, default=30.0, help="Maximum distance cutoff in nm for the plot (default: 30 nm).")
    
    args = parser.parse_args()

    base_dir = Path(args.csv_dir)
    if not base_dir.is_dir():
        print(f"[ERROR] Directory not found: {base_dir}")
        sys.exit(1)

    print(f"Searching for distance CSVs in {base_dir}...")
    
    # ---- 1. Pool Pairwise Distances ----
    csv_files = list(base_dir.rglob("inter_chain_distances.csv"))
    if not csv_files:
        print("[WARN] No inter_chain_distances.csv files found. Exiting gracefully.")
        pd.DataFrame(columns=["TomoName", "ChainA", "ChainA_Size", "ChainB", "ChainB_Size", "ChainDistance_nm"]).to_csv(args.out_csv, index=False)
        plt.figure()
        plt.title("No distance data found")
        plt.savefig(args.out_plot)
    else:
        all_dfs = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    continue
                tomo_name = file_path.parent.name
                df.insert(0, "TomoName", tomo_name)
                all_dfs.append(df)
            except Exception as e:
                print(f"[ERROR] Failed to read {file_path}: {e}")

        if not all_dfs:
            pd.DataFrame(columns=["TomoName", "ChainA", "ChainA_Size", "ChainB", "ChainB_Size", "ChainDistance_nm"]).to_csv(args.out_csv, index=False)
            plt.figure()
            plt.title("No chains found across all tomograms")
            plt.savefig(args.out_plot)
            plt.close()
        else:
            pooled_df = pd.concat(all_dfs, ignore_index=True)
            pooled_df.to_csv(args.out_csv, index=False)
            print(f"Saved {len(pooled_df)} pairwise distances to: {args.out_csv}")
            
            cutoff_nm = args.cutoff
            filtered_distances = pooled_df[pooled_df["ChainDistance_nm"] <= cutoff_nm]["ChainDistance_nm"]
            
            plt.figure(figsize=(8, 6))
            if not filtered_distances.empty:
                bins = np.arange(0, cutoff_nm + 2, 2)
                plt.hist(filtered_distances, bins=bins, color="#228b22", edgecolor="black", alpha=0.8)
            plt.title(f"Global Pairwise Distances Between Chains\n(Pooled from {len(all_dfs)} tomograms | Cutoff: {cutoff_nm} nm)")
            plt.xlabel("Pairwise Distance (nm)")
            plt.ylabel("Number of Unique Chain Pairs")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.out_plot, dpi=300)
            print(f"Saved pooled pairwise histogram to: {args.out_plot}")
            plt.close()

    # ---- 2. Pool Nearest Neighbor Distances ----
    if args.out_nn_csv or args.out_nn_plot:
        nn_files = list(base_dir.rglob("nearest_chain_distances.csv"))
        if not nn_files:
            print("[WARN] No nearest_chain_distances.csv files found.")
            if args.out_nn_csv:
                pd.DataFrame(columns=["TomoName", "ChainID", "ChainSize", "NearestChainID", "NNDistance_nm"]).to_csv(args.out_nn_csv, index=False)
            if args.out_nn_plot:
                plt.figure()
                plt.title("No NN distance data found")
                plt.savefig(args.out_nn_plot)
                plt.close()
        else:
            nn_dfs = []
            for file_path in nn_files:
                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        continue
                    tomo_name = file_path.parent.name
                    df.insert(0, "TomoName", tomo_name)
                    nn_dfs.append(df)
                except Exception as e:
                    print(f"[ERROR] Failed to read {file_path}: {e}")
            
            if not nn_dfs:
                if args.out_nn_csv:
                    pd.DataFrame(columns=["TomoName", "ChainID", "ChainSize", "NearestChainID", "NNDistance_nm"]).to_csv(args.out_nn_csv, index=False)
                if args.out_nn_plot:
                    plt.figure()
                    plt.title("No NN chains found across all tomograms")
                    plt.savefig(args.out_nn_plot)
                    plt.close()
            else:
                pooled_nn = pd.concat(nn_dfs, ignore_index=True)
                if args.out_nn_csv:
                    pooled_nn.to_csv(args.out_nn_csv, index=False)
                    print(f"Saved {len(pooled_nn)} nearest chain distances to: {args.out_nn_csv}")
                
                if args.out_nn_plot:
                    cutoff_nm = args.cutoff
                    nn_distances = pooled_nn[pooled_nn["NNDistance_nm"] <= cutoff_nm]["NNDistance_nm"]
                    plt.figure(figsize=(8, 6))
                    if not nn_distances.empty:
                        bins = np.arange(0, cutoff_nm + 2, 2)
                        plt.hist(nn_distances, bins=bins, color="#ff8c00", edgecolor="black", alpha=0.8)
                    plt.title(f"Global Nearest Neighbor Distance Per Chain\n(Pooled from {len(nn_dfs)} tomograms | Cutoff: {cutoff_nm} nm)")
                    plt.xlabel("Nearest Neighbor Distance (nm)")
                    plt.ylabel("Number of Individual Chains")
                    plt.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(args.out_nn_plot, dpi=300)
                    print(f"Saved pooled NN histogram to: {args.out_nn_plot}")
                    plt.close()

if __name__ == "__main__":
    main()
