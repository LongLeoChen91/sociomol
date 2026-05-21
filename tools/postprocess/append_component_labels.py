#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_component_rank.py — Add a size-rank column to a STAR file.

Reads a ranked component-size CSV (from ``plot_chain_distribution.py``)
and a STAR file containing ``rlnLC_ChainComponent``.  Merges a
``rlnLC_ComponentSizeRank`` column into the output so that downstream
tools can filter or colour particles by chain-component rank.

The original ``rlnLC_ChainComponent`` values are preserved unchanged.

Requires: pandas, starfile
"""

import argparse

import pandas as pd
import starfile


def main():
    parser = argparse.ArgumentParser(
        description="Merge component-size rank into a STAR file.",
    )
    parser.add_argument("--csv", required=True,
                        help="Input ranked component-size CSV (from plot_chain_distribution).")
    parser.add_argument("--input-star", required=True,
                        help="Input STAR file to be ranked.")
    parser.add_argument("--ref-star", required=False,
                        help="Optional reference STAR (e.g. annotated.star) to pull rlnLC_ChainComponent from if it's missing in input-star.")
    parser.add_argument("--output-star", required=True,
                        help="Output STAR file with added rlnLC_ComponentSizeRank column.")
    args = parser.parse_args()

    # ---- Load ranked CSV ----
    comp_df = pd.read_csv(args.csv, index_col=0)
    if "count" not in comp_df.columns:
        raise ValueError("CSV must contain a 'count' column.")

    # ---- Load STAR ----
    data = starfile.read(args.input_star, always_dict=True)
    if "particles" in data:
        df = data["particles"]
    else:
        df = next((v for v in data.values() if isinstance(v, pd.DataFrame)), None)
    if df is None:
        raise ValueError("Could not find a particles table in the STAR file.")

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

    # ---- Merge rank ----
    rank_col = "rlnLC_ComponentSizeRank"
    if rank_col in comp_df.columns:
        rank_series = comp_df[rank_col].copy()
        rank_series.index = rank_series.index.astype(str)
        df[rank_col] = df[col].astype(str).map(rank_series.to_dict())
        print(f"[INFO] Added column '{rank_col}' to STAR.")
    else:
        raise ValueError(f"Column '{rank_col}' not found in CSV.")

    # ---- Write ----
    if "particles" in data:
        data["particles"] = df
        starfile.write(data, args.output_star, overwrite=True)
    else:
        starfile.write({"particles": df}, args.output_star, overwrite=True)

    print(f"[DONE] Wrote: {args.output_star}")


if __name__ == "__main__":
    main()
