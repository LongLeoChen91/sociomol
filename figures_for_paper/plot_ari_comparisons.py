#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_ari_comparisons.py
-----------------------
Compile and visualize ARI curves for particle-based DBSCAN baselines
and the arm-based DBSCAN model using a compact publication-style layout.
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

baseline_csv = os.path.join(
    project_root,
    "baseline_distance_clustering",
    "combined_ARI_curves.csv"
)
model_csv = os.path.join(
    project_root,
    "experiments",
    "Manual_1",
    "DIST_CUTOFF_sweep_ARI.csv"
)
output_png = os.path.join(script_dir, "baseline_vs_model_ARI.png")

# ------------------------------------------------------------
# Color settings
# ------------------------------------------------------------
# Dataset colors (for ARI panel)
PARTICLE_NUC_COLOR = "#fed98e"
PARTICLE_RIBO_COLOR = "#66c2a4"

# Shared semantic colors used across panels
DIST_ONLY_HINT = "#43a2ca"   # distance-only hint / chosen cutoff
FULL_COLOR = "#969696"          # arm-based curve / strongest emphasis


def main():
    print(f"Loading baseline data from: {baseline_csv}")
    print(f"Loading model data from: {model_csv}")

    if not os.path.exists(baseline_csv) or not os.path.exists(model_csv):
        print("[ERROR] Required CSV files are missing. Please generate them first.")
        return

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    df_base = pd.read_csv(baseline_csv)
    df_model = pd.read_csv(model_csv)

    required_base_cols = {"Dataset", "DBSCAN_eps_nm", "ARI_Score"}
    required_model_cols = {"DIST_CUTOFF_NM", "ARI_Score"}

    if not required_base_cols.issubset(df_base.columns):
        missing = required_base_cols - set(df_base.columns)
        raise ValueError(f"Baseline CSV is missing columns: {sorted(missing)}")

    if not required_model_cols.issubset(df_model.columns):
        missing = required_model_cols - set(df_model.columns)
        raise ValueError(f"Model CSV is missing columns: {sorted(missing)}")

    df_nuc = df_base[df_base["Dataset"] == "nucleosome"].copy()
    df_ribo = df_base[df_base["Dataset"] == "ribosome"].copy()

    # Sort by x for cleaner plotting
    if not df_nuc.empty:
        df_nuc = df_nuc.sort_values("DBSCAN_eps_nm")
    if not df_ribo.empty:
        df_ribo = df_ribo.sort_values("DBSCAN_eps_nm")
    if not df_model.empty:
        df_model = df_model.sort_values("DIST_CUTOFF_NM")

    # ------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.8, 3.1), dpi=180)

    # ------------------------------------------------------------
    # Particle-based DBSCAN: nucleosome
    # ------------------------------------------------------------
    if not df_nuc.empty:
        x_nuc = df_nuc["DBSCAN_eps_nm"].to_numpy(dtype=float)
        y_nuc = df_nuc["ARI_Score"].to_numpy(dtype=float)
        idx_nuc = int(np.argmax(y_nuc))

        ax.plot(
            x_nuc,
            y_nuc,
            color=PARTICLE_NUC_COLOR,
            linestyle="--",
            linewidth=1.6,
            alpha=0.95,
            label=f"Particle (nuc, peak {y_nuc[idx_nuc]:.2f})"
        )
        ax.scatter(
            [x_nuc[idx_nuc]],
            [y_nuc[idx_nuc]],
            color=PARTICLE_NUC_COLOR,
            edgecolor="0.35",
            linewidth=0.5,
            s=26,
            zorder=4
        )

    # ------------------------------------------------------------
    # Particle-based DBSCAN: ribosome
    # ------------------------------------------------------------
    if not df_ribo.empty:
        x_ribo = df_ribo["DBSCAN_eps_nm"].to_numpy(dtype=float)
        y_ribo = df_ribo["ARI_Score"].to_numpy(dtype=float)
        idx_ribo = int(np.argmax(y_ribo))

        ax.plot(
            x_ribo,
            y_ribo,
            color=PARTICLE_RIBO_COLOR,
            linestyle="--",
            linewidth=1.6,
            alpha=0.95,
            label=f"Particle (ribo, peak {y_ribo[idx_ribo]:.2f})"
        )
        ax.scatter(
            [x_ribo[idx_ribo]],
            [y_ribo[idx_ribo]],
            color=PARTICLE_RIBO_COLOR,
            edgecolor="0.35",
            linewidth=0.5,
            s=26,
            zorder=4
        )

    # ------------------------------------------------------------
    # Arm-based DBSCAN: nucleosome
    # ------------------------------------------------------------
    if not df_model.empty:
        x_mod = df_model["DIST_CUTOFF_NM"].to_numpy(dtype=float)
        y_mod = df_model["ARI_Score"].to_numpy(dtype=float)
        idx_mod = int(np.argmax(y_mod))
        chosen_cutoff = x_mod[idx_mod]

        ax.plot(
            x_mod,
            y_mod,
            color=FULL_COLOR,
            linestyle="-",
            linewidth=2.3,
            label=f"Arm (nuc, peak {y_mod[idx_mod]:.2f})"
        )

        # Peak marker:
        # dark center = arm-based ARI peak
        # light-blue edge = downstream distance cutoff hint
        ax.scatter(
            [chosen_cutoff],
            [y_mod[idx_mod]],
            s=62,
            facecolor=FULL_COLOR,
            edgecolor=DIST_ONLY_HINT,
            linewidth=1.8,
            zorder=6
        )

        # Chosen cutoff line, styled to echo distance-only ablation
        ax.axvline(
            chosen_cutoff,
            linestyle="--",
            linewidth=1.1,
            color=DIST_ONLY_HINT,
            alpha=0.95,
            zorder=1
        )

        # ax.text(
        #     chosen_cutoff + 0.9,
        #     0.06,
        #     f"chosen cutoff\n{chosen_cutoff:.0f} nm",
        #     ha="left",
        #     va="bottom",
        #     fontsize=7.0,
        #     color="0.35"
        # )

    # ------------------------------------------------------------
    # Labels and title
    # ------------------------------------------------------------
    ax.set_title("Particle- vs arm-based DBSCAN", fontsize=9.5, pad=8)
    ax.set_xlabel("Distance cutoff / ε (nm)", fontsize=9)
    ax.set_ylabel("Adjusted Rand index", fontsize=9)

    # ------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------
    x_candidates = []
    if not df_nuc.empty:
        x_candidates.append(float(df_nuc["DBSCAN_eps_nm"].max()))
    if not df_ribo.empty:
        x_candidates.append(float(df_ribo["DBSCAN_eps_nm"].max()))
    if not df_model.empty:
        x_candidates.append(float(df_model["DIST_CUTOFF_NM"].max()))

    if x_candidates:
        ax.set_xlim(0, max(x_candidates) + 2)

    ax.set_ylim(-0.03, 1.03)

    # ------------------------------------------------------------
    # Clean visual style
    # ------------------------------------------------------------
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=1, labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.25)

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=6.4,
        frameon=True,
        borderpad=0.22,
        handlelength=1.6,
        labelspacing=0.28,
        borderaxespad=0.25
    )
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.95)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", transparent=False)
    print(f"\n[OK] Combined plot successfully generated at:\n     {output_png}")


if __name__ == "__main__":
    main()