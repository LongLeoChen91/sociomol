#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
distance_angle_ablation_F1.py
-----------------------------
Bar plot for the contribution of arm distance and bending angle
to strict arm-level edge recovery.

Emphasis:
- distance-only and angle-only are shown as distinct, complementary cues
- the combined condition is highlighted as the strongest integrated model
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Output path
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
output_png = os.path.join(script_dir, "ablation_f1_bar.png")

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
labels = ["Distance only", "Angle only", "Distance + angle"]
f1_scores = [0.6195, 0.8246, 1.0000]

# Shared cutoff selected from the ARI peak
chosen_cutoff_nm = 30

# ------------------------------------------------------------
# Color settings
# ------------------------------------------------------------
# Complementary single-cue colors
DIST_ONLY_COLOR = "#43a2ca"   # light blue
ANGLE_ONLY_COLOR = "#8073ac"  # light lavender

# Strong integrated model color
FULL_COLOR = "#084594"        # deep blue

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=180)

x = np.arange(len(labels))
bars = ax.bar(
    x,
    f1_scores,
    width=0.62,
    color=[DIST_ONLY_COLOR, ANGLE_ONLY_COLOR, FULL_COLOR],
    edgecolor="0.35",
    linewidth=0.7
)

# Value labels
for bar, val in zip(bars, f1_scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.025,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

# Note: the same cutoff is used across all ablation conditions
# ax.text(
#     0.03,
#     0.97,
#     f"All conditions: cutoff = {chosen_cutoff_nm} nm",
#     transform=ax.transAxes,
#     ha="left",
#     va="top",
#     fontsize=7.2,
#     bbox=dict(
#         boxstyle="round,pad=0.22",
#         facecolor="white",
#         edgecolor="0.35",
#         linewidth=0.8
#     )
# )

# ------------------------------------------------------------
# Labels and title
# ------------------------------------------------------------
ax.set_title("Ablation of distance and bending angle", fontsize=10, pad=10)
ax.set_ylabel("Edge-level F1 score", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)

# ------------------------------------------------------------
# Axis limits and style
# ------------------------------------------------------------
ax.set_ylim(0, 1.08)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(direction="out", length=3, width=1, labelsize=8)
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.25)

fig.tight_layout()
fig.savefig(output_png, dpi=300, bbox_inches="tight", transparent=False)
print(f"Saved: {output_png}")