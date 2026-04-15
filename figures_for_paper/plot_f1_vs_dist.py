#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_f1_vs_dist.py
------------------
Compile and visualize edge-level F1 scores across different distance cutoffs
for SocioMol on nucleosome and ribosome datasets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Set up paths relative to the project root
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

csv_nuc = os.path.join(
    project_root,
    "experiments",
    "Manual_1",
    "DIST_CUTOFF_sweep_F1.csv"
)

csv_ribo = os.path.join(
    project_root,
    "experiments",
    "PolysomeManual_1",
    "DIST_CUTOFF_sweep_F1.csv"
)

output_png = os.path.join(script_dir, "f1_vs_dist_comparison.png")

# ------------------------------------------------------------
# Color settings
# ------------------------------------------------------------
NUC_COLOR = "#084594"     # deep blue
RIBO_COLOR = "#8073ac"    # muted lavender
PEAK_EDGE_COLOR = "0.35"  # dark gray edge for peak highlights


def main():
    print(f"Loading Nucleosome data from: {csv_nuc}")
    print(f"Loading Ribosome data from: {csv_ribo}")

    # Load data
    df_nuc = pd.read_csv(csv_nuc) if os.path.exists(csv_nuc) else pd.DataFrame()
    df_ribo = pd.read_csv(csv_ribo) if os.path.exists(csv_ribo) else pd.DataFrame()

    if df_nuc.empty and df_ribo.empty:
        print("[ERROR] Both CSV files are missing. Please generate them first.")
        return

    # Sort arrays just in case
    if not df_nuc.empty:
        df_nuc = df_nuc.sort_values("DIST_CUTOFF_NM")
    if not df_ribo.empty:
        df_ribo = df_ribo.sort_values("DIST_CUTOFF_NM")

    # ------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.8, 3.1), dpi=180)

    # ------------------------------------------------------------
    # Nucleosome
    # ------------------------------------------------------------
    if not df_nuc.empty and "F1_Score" in df_nuc.columns:
        x_nuc = df_nuc["DIST_CUTOFF_NM"].to_numpy(dtype=float)
        y_nuc = df_nuc["F1_Score"].to_numpy(dtype=float)
        idx_nuc = int(np.argmax(y_nuc))

        ax.plot(
            x_nuc,
            y_nuc,
            color=NUC_COLOR,
            linestyle="-",
            linewidth=2.3,
            alpha=0.95,
            label=f"Nucleosome (peak {y_nuc[idx_nuc]:.2f})"
        )

        ax.scatter(
            [x_nuc[idx_nuc]],
            [y_nuc[idx_nuc]],
            facecolor=NUC_COLOR,
            edgecolor=PEAK_EDGE_COLOR,
            linewidth=1.2,
            s=45,
            zorder=5
        )

    # ------------------------------------------------------------
    # Ribosome
    # ------------------------------------------------------------
    if not df_ribo.empty and "F1_Score" in df_ribo.columns:
        x_ribo = df_ribo["DIST_CUTOFF_NM"].to_numpy(dtype=float)
        y_ribo = df_ribo["F1_Score"].to_numpy(dtype=float)
        idx_ribo = int(np.argmax(y_ribo))

        ax.plot(
            x_ribo,
            y_ribo,
            color=RIBO_COLOR,
            linestyle="-",
            linewidth=2.3,
            alpha=0.95,
            label=f"Ribosome (peak {y_ribo[idx_ribo]:.2f})"
        )

        ax.scatter(
            [x_ribo[idx_ribo]],
            [y_ribo[idx_ribo]],
            facecolor=RIBO_COLOR,
            edgecolor=PEAK_EDGE_COLOR,
            linewidth=1.2,
            s=45,
            zorder=5
        )

    # ------------------------------------------------------------
    # Labels and title
    # ------------------------------------------------------------
    ax.set_title("SocioMol sensitivity to distance cutoff", fontsize=9.5, pad=8)
    ax.set_xlabel("Distance cutoff (nm)", fontsize=9)
    ax.set_ylabel("Edge-level F1 score", fontsize=9)

    # ------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------
    x_candidates = []
    if not df_nuc.empty:
        x_candidates.append(float(df_nuc["DIST_CUTOFF_NM"].max()))
    if not df_ribo.empty:
        x_candidates.append(float(df_ribo["DIST_CUTOFF_NM"].max()))

    if x_candidates:
        ax.set_xlim(0, max(x_candidates) + 5)

    ax.set_ylim(-0.03, 1.05)

    # ------------------------------------------------------------
    # Clean visual style
    # ------------------------------------------------------------
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=1, labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.25)

    # ------------------------------------------------------------
    # Legend setup
    # ------------------------------------------------------------
    legend = ax.legend(
        loc="lower right",
        fontsize=6.8,
        frameon=True,
        borderpad=0.25,
        handlelength=1.6,
        labelspacing=0.3,
        borderaxespad=0.3
    )
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.95)

    # ------------------------------------------------------------
    # Save Image
    # ------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", transparent=False)
    print(f"\n[OK] F1 vs Distance comparison generated at:\n     {output_png}")


if __name__ == "__main__":
    main()