#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_tomos.py - Generate a per-tomogram summary table for quick quality assessment.

Reads the global_chains_clustered.csv and produces a compact summary CSV
with one row per tomogram, including particle counts, chain statistics,
and spatial clustering metrics.

Output is sorted by LinkedRatio (descending) so the "best" tomograms
appear at the top.

Requires: pandas
"""

import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Generate a per-tomogram summary table for quick quality assessment.",
    )
    parser.add_argument(
        "--clustered-csv", required=True,
        help="Global chains_clustered.csv (output of find_clusters_of_chains.py merge-global).",
    )
    parser.add_argument(
        "--out-csv", required=True,
        help="Output summary CSV (one row per tomogram).",
    )
    args = parser.parse_args()

    # ---- Read Data ----
    print(f"Reading {args.clustered_csv}...")
    df = pd.read_csv(args.clustered_csv)

    required_cols = ["TomoName", "ChainID", "ChainSize", "ClusterID", "ChainNND_nm"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Missing required column: {col}")
            sys.exit(1)

    # ---- Compute per-tomo summary ----
    rows = []
    for tomo, grp in df.groupby("TomoName"):
        total_particles = int(grp["ChainSize"].sum())
        num_chains = len(grp)

        # Linked chains = chains with size > 1 (multi-particle chains)
        linked = grp[grp["ChainSize"] > 1]
        num_linked_chains = len(linked)
        linked_particles = int(linked["ChainSize"].sum())

        # Exclude single-chain "clusters" (ChainSize = 1) from the total cluster count.
        # Now NumClusters perfectly represents the number of true polysome communities.
        num_clusters = int(linked["ClusterID"].nunique())

        # LinkedRatio: fraction of particles that belong to multi-particle chains
        linked_ratio = (
            (linked_particles / total_particles * 100)
            if total_particles > 0 else 0.0
        )

        # MeanLinkedChainSize: average size of multi-particle chains only
        mean_linked_chain_size = (
            linked["ChainSize"].mean() if len(linked) > 0 else 0.0
        )

        # MaxChainSize
        max_chain_size = int(grp["ChainSize"].max())

        # MedianChainNND_nm: median nearest-neighbor distance, excluding
        # the -9999.0 sentinel value assigned to chains without neighbours
        valid_nnd = grp[
            (grp["ChainNND_nm"] != -9999.0) & grp["ChainNND_nm"].notna()
        ]
        median_nnd = (
            valid_nnd["ChainNND_nm"].median()
            if len(valid_nnd) > 0 else -9999.0
        )

        # MultiChainClusterRatio: among ALL clusters in this tomo, what
        # fraction contain >= 2 linked chains (ChainSize > 1)?
        if len(linked) > 0:
            linked_per_cluster = linked.groupby("ClusterID")["ChainID"].count()
            multi_chain_clusters = int((linked_per_cluster >= 2).sum())
            multi_chain_cluster_ratio = (
                (multi_chain_clusters / num_clusters * 100)
                if num_clusters > 0 else 0.0
            )
        else:
            multi_chain_clusters = 0
            multi_chain_cluster_ratio = 0.0

        # ChainsInMultiClusterRatio: what fraction of TRUE polysomes belong to a cluster with > 1 chain?
        if "ClusterChainCount" in grp.columns:
            chains_in_multi = int((grp["ClusterChainCount"] > 1).sum())
            chains_in_multi_ratio = (
                (chains_in_multi / num_linked_chains * 100)
                if num_linked_chains > 0 else 0.0
            )
        else:
            chains_in_multi = 0
            chains_in_multi_ratio = 0.0

        mean_chains_per_multi_cluster = (
            chains_in_multi / multi_chain_clusters
            if multi_chain_clusters > 0 else 0.0
        )

        rows.append({
            "TomoName": tomo,
            # Particle metrics
            "TotalParticles": total_particles,
            "NumLinkedParticles": linked_particles,
            "LinkedRatio": round(linked_ratio, 2),
            # Chain metrics
            "NumChains": num_chains,
            "MaxChainSize": max_chain_size,
            "NumLinkedChains": num_linked_chains,
            "MeanLinkedChainSize": round(mean_linked_chain_size, 2),
            "MedianChainNND_nm": round(median_nnd, 2),
            # Cluster metrics (Group 1: the clusters themselves)
            "NumClusters": num_clusters,
            "NumMultiChainClusters": multi_chain_clusters,
            "MultiChainClusterRatio": round(multi_chain_cluster_ratio, 2),
            # Cluster metrics (Group 2: the chains inside those clusters)
            "NumChainsInMultiClusters": chains_in_multi,
            "ChainsInMultiClusterRatio": round(chains_in_multi_ratio, 2),
            "MeanChainsPerMultiCluster": round(mean_chains_per_multi_cluster, 2),
        })

    summary = pd.DataFrame(rows)

    # Sort by LinkedRatio descending - best tomo first
    summary = summary.sort_values("LinkedRatio", ascending=False).reset_index(drop=True)

    summary.to_csv(args.out_csv, index=False)

    print(f"\n{'=' * 60}")
    print("Per-Tomogram Summary")
    print(f"{'=' * 60}")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {args.out_csv}")


if __name__ == "__main__":
    main()
