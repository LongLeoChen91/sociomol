#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_combined_ari.py
--------------------
A standalone script to sweep DBSCAN distance cutoffs (eps) for multiple
molecular systems (e.g., nucleosome, ribosome) and plot their Adjusted
Rand Index (ARI) performance curves on a single unified plot.
"""

import os
import numpy as np
import pandas as pd
import starfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

# =============================================================================
#  Configuration
# =============================================================================
CONFIGS = {
    "nucleosome": dict(
        star_file    = "R3_ID_Manual_1.star",
        pixel_size_a = 8.0,
        gt_col       = "class",
        min_samples  = 1,
        eps_sweep    = (1, 70.5, 0.5),
        color        = "steelblue",
    ),
    "ribosome": dict(
        star_file    = "IDname_PolysomeManual_1.star",
        pixel_size_a = 1.96,
        gt_col       = "class",
        min_samples  = 1,
        eps_sweep    = (1, 70.5, 0.5),
        color        = "tomato",
    ),
}

# Output file path
OUTPUT_PNG = "combined_ARI_curves.png"

def main():
    # Set working directory to the script's directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"[INFO] Working directory set to: {script_dir}")

    # Prepare plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    all_results = []

    for dataset_name, cfg in CONFIGS.items():
        print(f"\n[INFO] Processing dataset: {dataset_name}")
        star_file = cfg["star_file"]
        if not os.path.exists(star_file):
            print(f"[ERROR] Could not find {star_file}. Skipping {dataset_name}.")
            continue
        
        # Load data
        print(f"       -> Loading {star_file}...")
        raw = starfile.read(star_file, always_dict=True)
        df = next(iter(raw.values()))

        # Convert to physical distance (nm)
        nm_per_px = cfg["pixel_size_a"] / 10.0
        xyz = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(dtype=float) * nm_per_px
        gt = df[cfg["gt_col"]].to_numpy(dtype=int)

        # Generate eps array for the sweep
        eps_sweep_arr = np.arange(*cfg["eps_sweep"])
        min_samples = cfg["min_samples"]
        
        ari_vals = []
        
        print(f"       -> Sweeping DBSCAN eps from {eps_sweep_arr[0]:.1f} to {eps_sweep_arr[-1]:.1f} nm...")
        for eps_val in eps_sweep_arr:
            labels = DBSCAN(eps=eps_val, min_samples=min_samples, metric="euclidean").fit_predict(xyz)
            ari_score = adjusted_rand_score(gt, labels)
            ari_vals.append(ari_score)
            
        ari_arr = np.array(ari_vals)
        
        # Identify the peak ARI
        best_idx = int(np.argmax(ari_arr))
        best_eps = eps_sweep_arr[best_idx]
        best_ari = ari_arr[best_idx]
        print(f"       -> Peak ARI: {best_ari:.3f} at eps = {best_eps:.1f} nm")

        # Collect for CSV
        for e, a in zip(eps_sweep_arr, ari_arr):
            all_results.append({
                "Dataset": dataset_name,
                "DBSCAN_eps_nm": e,
                "ARI_Score": a
            })

        # Plot curve
        color = cfg.get("color", "gray")
        ax.plot(eps_sweep_arr, ari_arr, color=color, lw=2.5, 
                label=f"{dataset_name.capitalize()} (Peak ARI: {best_ari:.2f})")
        
        # Mark the peak
        ax.scatter([best_eps], [best_ari], color=color, s=50, zorder=5)
        ax.annotate(f"ε={best_eps:.1f}nm",
                    xy=(best_eps, best_ari),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color)

    # Finalize plot formatting
    ax.set_xlabel("DBSCAN Distance Threshold (ε) [nm]", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=12)
    ax.set_title("Baseline Clustering Performance: ARI vs. Distance", fontsize=14, pad=15)
    
    # Configure axes limits and ticks
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, 40)
    
    # Clean up aesthetics
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Customize legend
    legend = ax.legend(loc="lower right", fontsize=11, frameon=True)
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    
    # Save data to CSV
    csv_out = os.path.join(script_dir, "combined_ARI_curves.csv")
    pd.DataFrame(all_results).to_csv(csv_out, index=False)
    
    # Save image
    out_path = os.path.join(script_dir, OUTPUT_PNG)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n[OK] Combined plot saved successfully to:\n     {out_path}")
    print(f"[OK] Combined CSV saved successfully to:\n     {csv_out}")

if __name__ == "__main__":
    main()
