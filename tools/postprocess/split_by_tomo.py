#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_by_tomo.py -- Split annotated STAR + edges CSV + raw STAR by rlnTomoName.

Creates one sub-directory per tomogram inside --out-dir, each containing:
  annotated.star   (rows from the annotated STAR for this tomo)
  edges.csv        (edges from the CSV for this tomo)
  raw.star         (rows from the raw STAR for this tomo, if provided)

Requires: pandas, starfile
"""

import argparse
import os

import pandas as pd
import starfile


def _read_star(path: str) -> pd.DataFrame:
    """Read a STAR file and return the particles DataFrame."""
    data = starfile.read(path, always_dict=True)
    if "particles" in data:
        return data["particles"]
    if "data_particles" in data:
        return data["data_particles"]
    return next(iter(data.values()))


def main():
    parser = argparse.ArgumentParser(
        description="Split annotated STAR + edges CSV (+ optional raw STAR) by tomogram.",
    )
    parser.add_argument("--annotated", required=True,
                        help="Annotated STAR file (output of sociomol predict).")
    parser.add_argument("--edges", required=True,
                        help="Edges CSV file (output of sociomol predict).")
    parser.add_argument("--raw", required=False, default=None,
                        help="Raw STAR file (before annotation). Optional.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory; per-tomo sub-dirs will be created here.")
    args = parser.parse_args()

    # ---- Load annotated STAR ----
    ann_df = _read_star(args.annotated)
    if "rlnTomoName" not in ann_df.columns:
        raise KeyError("Annotated STAR must contain 'rlnTomoName'.")
    tomos = sorted(ann_df["rlnTomoName"].unique())
    print(f"Found {len(tomos)} tomogram(s): {', '.join(tomos)}")

    # ---- Load edges CSV ----
    edges_df = pd.read_csv(args.edges)

    # ---- Optionally load raw STAR ----
    raw_df = _read_star(args.raw) if args.raw else None

    # ---- Split per tomo ----
    for tomo in tomos:
        tomo_dir = os.path.join(args.out_dir, tomo)
        os.makedirs(tomo_dir, exist_ok=True)

        # Annotated subset
        ann_sub = ann_df[ann_df["rlnTomoName"] == tomo].copy()
        starfile.write({"particles": ann_sub},
                       os.path.join(tomo_dir, "annotated.star"), overwrite=True)

        # Edges subset
        if "tomo_name" in edges_df.columns:
            edge_sub = edges_df[edges_df["tomo_name"] == tomo].copy()
        else:
            # Fallback: keep all edges whose i_id matches a particle in this tomo
            if "i_id" in edges_df.columns and "rlnTomoParticleId" in ann_sub.columns:
                valid_ids = set(ann_sub["rlnTomoParticleId"].astype(int))
                edge_sub = edges_df[edges_df["i_id"].astype(int).isin(valid_ids)].copy()
            else:
                edge_sub = pd.DataFrame()
        edge_sub.to_csv(os.path.join(tomo_dir, "edges.csv"), index=False)

        # Raw subset
        if raw_df is not None:
            raw_sub = raw_df[raw_df["rlnTomoName"] == tomo].copy()
            starfile.write({"particles": raw_sub},
                           os.path.join(tomo_dir, "raw.star"), overwrite=True)

        n_ann = len(ann_sub)
        n_edge = len(edge_sub)
        n_raw = len(raw_sub) if raw_df is not None else "-"
        print(f"  {tomo}: {n_ann} particles, {n_edge} edges, {n_raw} raw rows")

    print("Split complete.")


if __name__ == "__main__":
    main()
