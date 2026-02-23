#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------
# Input CSV
# ----------------------------
# CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\Nucleosome_Ben_tomo_2173\DoubleLinker_edges.csv"

CSV_PATH = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\Ribosome_tomo0017\Linker_edges.csv"

# CSV_PATH = r"C:\Users\LongChen\OneDrive - Nexus365\Laptop\Research202408\Research_Dave\Dev\SocioMol\dev2\V1_noUI\NucC2Align\manual_linker\scripts\MR_arm2_0_di25_72_BaseOnEndDensity\DoubleLinker_edges.csv"


# CSV_PATH = r"C:\Users\LongChen\OneDrive - Nexus365\Laptop\Research202408\Research_Dave\Dev\SocioMol\dev2\V1_noUI\NucC2Align\manual_linker\scripts\Single_tomo3\DoubleLinker_edges.csv"


THETA_COL = "theta_deg"
L_COL = "L_nm"

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(CSV_PATH)

if THETA_COL not in df.columns or L_COL not in df.columns:
    raise KeyError(f"Missing required columns. Available: {list(df.columns)}")

theta = pd.to_numeric(df[THETA_COL], errors="coerce")
L = pd.to_numeric(df[L_COL], errors="coerce")

mask = theta.notna() & L.notna()
theta = theta[mask]
L = L[mask]

print(f"[INFO] n_points = {len(theta)}")

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(4.5, 4), dpi=180)

plt.scatter(
    L,
    theta,
    s=5,
    alpha=0.6
)

plt.xlabel("Linker length L (nm)")
plt.ylabel("Theta (degrees)")
plt.title("Theta vs Linker Length")

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xlim(0, 60)
plt.ylim(0, 180)

plt.tight_layout()
plt.savefig("theta_vs_L_scatter.png", dpi=300)
plt.show()
