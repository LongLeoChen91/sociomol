#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_chain_sizes.py — Add a size-rank column to a STAR file.

Reads a ranked chain-size CSV (from ``analyze_chain_sizes.py``)
and a STAR file containing ``rlnLC_ChainComponent``.  Merges a
``rlnLC_ComponentSizeRank`` column into the output so that downstream
tools can filter or colour particles by chain rank.

The original ``rlnLC_ChainComponent`` values are preserved unchanged.

Requires: pandas, starfile
"""

import argparse

import numpy as np
import pandas as pd
import starfile


def main():
    parser = argparse.ArgumentParser(
        description="Merge chain-size rank into a STAR file.",
    )
    parser.add_argument("--csv", required=True,
                        help="Input ranked chain-size CSV (from analyze_chain_sizes).")
    parser.add_argument("--input-star", required=True,
                        help="Input STAR file to be ranked.")
    parser.add_argument("--ref-star", required=False,
                        help="Optional reference STAR (e.g. annotated.star) to pull rlnLC_ChainComponent from if it's missing in input-star.")
    parser.add_argument("--output-star", required=True,
                        help="Output STAR file with added rlnLC_ComponentSizeRank column.")
    args = parser.parse_args()

    # ---- Load ranked CSV ----
    comp_df = pd.read_csv(args.csv)
    if "ChainID" not in comp_df.columns or "ChainSize" not in comp_df.columns:
        raise ValueError("CSV must contain 'ChainID' and 'ChainSize' columns.")
    comp_df = comp_df.set_index("ChainID")

    # ---- Load STAR ----
    data = starfile.read(args.input_star, always_dict=True)
    if "particles" in data:
        df = data["particles"]
    else:
        df = next((v for v in data.values() if isinstance(v, pd.DataFrame)), None)
    if df is None:
        raise ValueError("Could not find a particles table in the STAR file.")

    if df.empty:
        print("[WARN] Input STAR file is empty (0 rows). Writing empty output.")
        if "particles" in data:
            starfile.write(data, args.output_star, overwrite=True)
        else:
            starfile.write({"particles": df}, args.output_star, overwrite=True)
        return

    col = "rlnLC_ChainComponent"
    if col not in df.columns:
        if args.ref_star:
            ref_data = starfile.read(args.ref_star)
            ref_df = next(iter(ref_data.values())) if isinstance(ref_data, dict) else ref_data
            if col in ref_df.columns and "rlnTomoParticleId" in ref_df.columns and "rlnTomoParticleId" in df.columns:
                mapping = dict(zip(ref_df["rlnTomoParticleId"], ref_df[col]))
                df[col] = df["rlnTomoParticleId"].map(mapping)
                print(f"[INFO] Pulled '{col}' from reference STAR based on particle ID.")
            else:
                raise ValueError(f"Reference STAR must contain '{col}' and 'rlnTomoParticleId'.")
        else:
            raise ValueError(f"Column '{col}' not found in the STAR file and no --ref-star provided.")

    # ---- Merge columns from CSV ----
    cols_to_merge = {
        "ChainSize": "rlnLC_ComponentSize",
        "ChainSizeRank": "rlnLC_ComponentSizeRank",
        "ClusterID": "rlnLC_ClusterID",
        "ClusterParticleCount": "rlnLC_ClusterParticleCount",
        "ClusterChainCount": "rlnLC_ClusterChainCount",
        "ChainNND_nm": "rlnLC_ChainNND_nm",
        "ClusterSizeRank": "rlnLC_ClusterSizeRank"
    }
    for csv_col, star_col in cols_to_merge.items():
        if csv_col in comp_df.columns:
            merge_series = comp_df[csv_col].copy()
            merge_series.index = merge_series.index.astype(str)
            df[star_col] = df[col].astype(str).map(merge_series.to_dict())
            print(f"[INFO] Added column '{star_col}' to STAR.")
        else:
            print(f"[WARN] Column '{csv_col}' not found in CSV, skipping.")

    # ---- Post-process: mark singletons and create randomized rank ----
    size_col = "rlnLC_ComponentSize"
    rank_col = "rlnLC_ComponentSizeRank"
    if size_col in df.columns and rank_col in df.columns:
        # Singletons (size < 2) OR unmapped particles (NaN) have no meaningful rank
        singleton_mask = df[size_col].isna() | (df[size_col] < 2)
        df.loc[singleton_mask, rank_col] = -9999
        print(f"[INFO] Set {rank_col} to -9999 for {singleton_mask.sum()} singleton particles.")

        # Create a randomized rank column for better ChimeraX color separation.
        # Shuffles at the unique-rank level so all particles in the same chain
        # still share the same random rank.  Seed=42 for reproducibility.
        random_col = "rlnLC_ComponentSizeRankRandom"
        valid_ranks = np.sort(df.loc[~singleton_mask, rank_col].dropna().unique())
        shuffled = np.random.default_rng(42).permutation(valid_ranks)
        rank_map = dict(zip(valid_ranks, shuffled))
        df[random_col] = df[rank_col].map(rank_map).fillna(-9999).astype(int)
        df[rank_col] = df[rank_col].fillna(-9999).astype(int)
        print(f"[INFO] Added column '{random_col}' ({len(valid_ranks)} unique ranks shuffled).")

    # For clusters
    cluster_size_col = "rlnLC_ClusterChainCount"
    cluster_rank_col = "rlnLC_ClusterSizeRank"
    if cluster_size_col in df.columns and cluster_rank_col in df.columns:
        # Singleton clusters (count < 2) OR unmapped (NaN) have no meaningful rank
        singleton_cluster_mask = df[cluster_size_col].isna() | (df[cluster_size_col] < 2)
        df.loc[singleton_cluster_mask, cluster_rank_col] = -9999
        print(f"[INFO] Set {cluster_rank_col} to -9999 for {singleton_cluster_mask.sum()} particles in singleton clusters.")

        cluster_random_col = "rlnLC_ClusterSizeRankRandom"
        valid_cluster_ranks = np.sort(df.loc[~singleton_cluster_mask, cluster_rank_col].dropna().unique())
        shuffled_cluster = np.random.default_rng(43).permutation(valid_cluster_ranks)
        cluster_rank_map = dict(zip(valid_cluster_ranks, shuffled_cluster))
        df[cluster_random_col] = df[cluster_rank_col].map(cluster_rank_map).fillna(-9999).astype(int)
        df[cluster_rank_col] = df[cluster_rank_col].fillna(-9999).astype(int)
        print(f"[INFO] Added column '{cluster_random_col}' ({len(valid_cluster_ranks)} unique ranks shuffled).")

    # ---- Write ----
    if "particles" in data:
        data["particles"] = df
        starfile.write(data, args.output_star, overwrite=True)
    else:
        starfile.write({"particles": df}, args.output_star, overwrite=True)

    print(f"[DONE] Wrote: {args.output_star}")


if __name__ == "__main__":
    main()
