#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_component_sizes.py — Visualise chain-component size distribution.

Reads an annotated STAR file (output of ``sociomol predict``), counts how
many particles belong to each connected component, and produces:

1. A ranked CSV summarising component sizes (sorted descending).
2. A histogram PNG showing the distribution of component sizes.

Requires: pandas, starfile, matplotlib
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # headless-safe; overridden by --show
import matplotlib.pyplot as plt
import pandas as pd
import starfile


def main():
    parser = argparse.ArgumentParser(
        description="Plot chain-component size distribution from an annotated STAR file.",
    )
    parser.add_argument("--annotated", required=True,
                        help="Input annotated STAR file (from sociomol predict).")
    parser.add_argument("--out-csv", required=True,
                        help="Output CSV with ranked component sizes.")
    parser.add_argument("--out-plot", required=True,
                        help="Output PNG histogram.")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive plot window after saving.")
    parser.add_argument("--min-plot-size", type=int, default=2,
                        help="Minimum component size to include in the histogram plot (default: 2).")
    args = parser.parse_args()

    # ---- Load ----
    df = starfile.read(args.annotated)

    # ---- Count components ----
    comp_sizes = df["rlnLC_ChainComponent"].value_counts().sort_index()

    # ---- Ranked CSV ----
    sorted_df = comp_sizes.sort_values(ascending=False).to_frame("rlnLC_ComponentSize")
    sorted_df["rlnLC_ComponentSizeRank"] = range(1, len(sorted_df) + 1)
    sorted_df.to_csv(args.out_csv)
    print(f"Sorted CSV with rank saved to {args.out_csv}")

    # ---- Histogram ----
    plot_sizes = comp_sizes[comp_sizes >= args.min_plot_size]
    if len(plot_sizes) == 0:
        print(f"[WARN] No components with size >= {args.min_plot_size} to plot. Exiting.")
        return

    size_counts = plot_sizes.value_counts().sort_index()
    avg_size = plot_sizes.mean()
    max_size = plot_sizes.max()

    plt.figure(figsize=(4, 4))
    plt.bar(size_counts.index, size_counts.values,
            color="#e0f3f8", edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Number of connected particles", fontsize=13)
    plt.ylabel("Number of occurrences", fontsize=13)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.text(0.95, 0.95, f"Average size: {avg_size:.2f}\nMax size: {max_size}",
             transform=plt.gca().transAxes,
             ha="right", va="top", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=300)
    print(f"Figure saved to {args.out_plot}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()
