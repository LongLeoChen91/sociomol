#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_cluster_sizes.py — Analyze sizes of multi-chain clusters.

Reads the global clustered CSV, extracts unique clusters (groups by TomoName and ClusterID),
filters out singleton clusters (ClusterChainCount <= 1), and generates histograms
for both ClusterChainCount and ClusterParticleCount.

Requires: pandas, matplotlib, seaborn
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot cluster sizes from the global clustered CSV.")
    parser.add_argument("--csv", required=True, help="Input CSV (e.g., global_chains_clustered.csv)")
    parser.add_argument("--out-csv", required=True, help="Output CSV for unique valid clusters")
    parser.add_argument("--out-plot", required=True, help="Output plot containing histograms (.png)")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"[ERROR] Failed to read {args.csv}: {e}")
        sys.exit(1)

    req_cols = ["TomoName", "ClusterID", "ClusterChainCount", "ClusterParticleCount"]
    for col in req_cols:
        if col not in df.columns:
            print(f"[ERROR] Missing required column: {col}")
            sys.exit(1)

    # 1. Group to unique clusters (one row per cluster)
    unique_clusters = df.drop_duplicates(subset=["TomoName", "ClusterID"]).copy()

    # 2. Filter out unclustered items (ClusterID == -1) and singleton clusters (ClusterChainCount == 1)
    # A true "cluster" must contain at least 2 chains.
    valid_clusters = unique_clusters[
        (unique_clusters["ClusterID"] != -1) & 
        (unique_clusters["ClusterChainCount"] > 1)
    ]

    print(f"[INFO] Found {len(valid_clusters)} valid multi-chain clusters.")

    # 3. Save the reduced dataset
    valid_clusters.to_csv(args.out_csv, index=False)
    print(f"[DONE] Saved unique cluster data to: {args.out_csv}")

    if valid_clusters.empty:
        print("[WARN] No valid multi-chain clusters found to plot. Exiting gracefully.")
        sys.exit(0)

    # 4. Generate Plot (Two subplots: one for Chains, one for Particles)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Chains per Cluster
    sns.histplot(data=valid_clusters, x="ClusterChainCount", discrete=True, color="#3498db", ax=axes[0])
    axes[0].set_title("Distribution of Chains per Cluster")
    axes[0].set_xlabel("Number of Chains in Cluster")
    axes[0].set_ylabel("Count of Clusters")

    # Plot 2: Particles per Cluster
    sns.histplot(data=valid_clusters, x="ClusterParticleCount", discrete=True, color="#e74c3c", ax=axes[1])
    axes[1].set_title("Distribution of Particles per Cluster")
    axes[1].set_xlabel("Total Number of Particles in Cluster")
    axes[1].set_ylabel("Count of Clusters")

    # Explicitly set x-ticks to actual data values to avoid empty integer ticks when data is sparse
    chain_counts = sorted(valid_clusters["ClusterChainCount"].unique())
    axes[0].set_xticks(chain_counts)
    
    particle_counts = sorted(valid_clusters["ClusterParticleCount"].unique())
    axes[1].set_xticks(particle_counts)

    # Force integer ticks on y-axes (counts are always integers)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved cluster size histograms to: {args.out_plot}")


if __name__ == "__main__":
    main()
