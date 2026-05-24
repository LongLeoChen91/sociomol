#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_chains.py — Cluster nearby chains based on spatial proximity.

Uses inter-chain distances to group chains into spatial clusters via
single-linkage clustering (connected components with a distance threshold).
Adds ClusterID, ClusterChainCount, and ClusterParticleCount columns
to the chain sizes CSV.

Two modes:
  per-tomo    : Cluster chains within a single tomogram.
  merge-global: Merge per-tomo cluster info into global chain sizes CSV.

Requires: pandas (no additional dependencies)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set) — zero external dependencies
# ---------------------------------------------------------------------------
class _UnionFind:
    """Minimal union-find for string keys with path compression."""

    def __init__(self):
        self._parent = {}

    def find(self, x):
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[rx] = ry


# ---------------------------------------------------------------------------
# Per-tomo clustering
# ---------------------------------------------------------------------------
def _cluster_per_tomo(args):
    """Cluster chains within a single tomogram."""

    dist_df = pd.read_csv(args.distance_csv)
    sizes_df = pd.read_csv(args.chain_sizes_csv)

    # Clean IDs to prevent float vs int string mismatch (e.g. '1.0' vs '1')
    def clean_id(x):
        s = str(x)
        return s[:-2] if s.endswith('.0') else s

    sizes_df["ChainID"] = sizes_df["ChainID"].apply(clean_id)
    if not dist_df.empty and "ChainDistance_nm" in dist_df.columns:
        dist_df["ChainA"] = dist_df["ChainA"].apply(clean_id)
        dist_df["ChainB"] = dist_df["ChainB"].apply(clean_id)

    uf = _UnionFind()

    # Register every chain as a singleton
    for cid in sizes_df["ChainID"]:
        uf.find(cid)

    # Union chain pairs whose distance is within the threshold
    if not dist_df.empty and "ChainDistance_nm" in dist_df.columns:
        close = dist_df[dist_df["ChainDistance_nm"] <= args.threshold]
        for _, row in close.iterrows():
            uf.union(row["ChainA"], row["ChainB"])

    # Assign sequential ClusterIDs (1, 2, 3, …)
    root_to_id = {}
    next_id = 1
    cluster_ids = []
    for cid in sizes_df["ChainID"]:
        root = uf.find(cid)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        cluster_ids.append(root_to_id[root])

    sizes_df["ClusterID"] = cluster_ids

    # Per-cluster statistics
    cluster_stats = sizes_df.groupby("ClusterID").agg(
        ClusterChainCount=("ChainID", "count"),
        ClusterParticleCount=("ChainSize", "sum"),
    ).reset_index()

    # Drop old stats if re-running, then merge fresh ones
    for col in ["ClusterChainCount", "ClusterParticleCount"]:
        if col in sizes_df.columns:
            sizes_df = sizes_df.drop(columns=[col])

    sizes_df = sizes_df.merge(cluster_stats, on="ClusterID", how="left")

    # Ensure cluster columns are at the end
    base_cols = [c for c in sizes_df.columns
                 if c not in ("ClusterID", "ClusterChainCount", "ClusterParticleCount")]
    sizes_df = sizes_df[base_cols + ["ClusterID", "ClusterChainCount", "ClusterParticleCount"]]

    sizes_df.to_csv(args.out_csv, index=False)

    n_clusters = len(root_to_id)
    n_multi = int((cluster_stats["ClusterChainCount"] > 1).sum())
    print(f"[INFO] Threshold: {args.threshold} nm → "
          f"{n_clusters} clusters ({n_multi} multi-chain, "
          f"{n_clusters - n_multi} singleton).")
    print(f"Saved to: {args.out_csv}")


# ---------------------------------------------------------------------------
# Global merge
# ---------------------------------------------------------------------------
def _merge_global(args):
    """Merge per-tomo cluster info into the global chain sizes CSV."""

    global_df = pd.read_csv(args.global_csv)
    tomo_dir = Path(args.tomo_dir)

    # Collect per-tomo cluster info: ChainID → (globally-unique ClusterID, counts)
    chain_info = {}
    for tomo_folder in sorted(tomo_dir.iterdir()):
        if not tomo_folder.is_dir():
            continue
        csv_path = tomo_folder / "chains_clustered.csv"
        if not csv_path.exists():
            continue
        tomo_name = tomo_folder.name
        df = pd.read_csv(csv_path)
        if "ClusterID" not in df.columns:
            continue
        # Clean IDs to prevent float vs int string mismatch
        def clean_id(x):
            s = str(x)
            return s[:-2] if s.endswith('.0') else s

        df["ChainID"] = df["ChainID"].apply(clean_id)
        for _, row in df.iterrows():
            cid = row["ChainID"]
            # Prefix with tomo name to guarantee global uniqueness
            global_cluster_id = f"{tomo_name}_C{int(row['ClusterID'])}"
            # Compound key to avoid cross-tomo overwriting of ChainIDs
            compound_key = f"{tomo_name}_{cid}"
            chain_info[compound_key] = {
                "ClusterID": global_cluster_id,
                "ClusterChainCount": int(row["ClusterChainCount"]),
                "ClusterParticleCount": int(row["ClusterParticleCount"]),
            }

    # Map into global dataframe
    def clean_id(x):
        s = str(x)
        return s[:-2] if s.endswith('.0') else s
        
    global_df["_key"] = global_df["TomoName"].astype(str) + "_" + global_df["ChainID"].apply(clean_id)
    for col in ["ClusterID", "ClusterChainCount", "ClusterParticleCount"]:
        global_df[col] = global_df["_key"].map(
            lambda x, c=col: chain_info.get(x, {}).get(c)
        )
    global_df = global_df.drop(columns=["_key"])

    global_df.to_csv(args.out_csv, index=False)

    n_mapped = global_df["ClusterID"].notna().sum()
    n_total = len(global_df)
    print(f"[INFO] Merged cluster info for {n_mapped}/{n_total} chains.")
    print(f"Saved to: {args.out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cluster nearby chains based on spatial proximity.",
    )
    sub = parser.add_subparsers(dest="mode")

    # -- per-tomo --
    p_tomo = sub.add_parser(
        "per-tomo",
        help="Cluster chains within a single tomogram.",
    )
    p_tomo.add_argument("--distance-csv", required=True,
                        help="Per-tomo inter_chain_distances.csv.")
    p_tomo.add_argument("--chain-sizes-csv", required=True,
                        help="Per-tomo chain_sizes.csv.")
    p_tomo.add_argument("--threshold", type=float, default=40.0,
                        help="Distance threshold in nm (default: 40).")
    p_tomo.add_argument("--out-csv", required=True,
                        help="Output CSV (can overwrite --chain-sizes-csv).")

    # -- merge-global --
    p_global = sub.add_parser(
        "merge-global",
        help="Merge per-tomo cluster info into global chain sizes CSV.",
    )
    p_global.add_argument("--global-csv", required=True,
                          help="Global chain_sizes.csv to update.")
    p_global.add_argument("--tomo-dir", required=True,
                          help="Directory containing per-tomo subdirectories.")
    p_global.add_argument("--out-csv", required=True,
                          help="Output CSV (can overwrite --global-csv).")

    args = parser.parse_args()

    if args.mode == "per-tomo":
        _cluster_per_tomo(args)
    elif args.mode == "merge-global":
        _merge_global(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
