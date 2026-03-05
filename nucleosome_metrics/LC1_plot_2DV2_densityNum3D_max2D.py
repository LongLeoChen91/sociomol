"""
Overlay scatter coordinates on top of density heatmap (Fig 2a style)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import starfile
from pathlib import Path

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ===== User parameters =====
# STAR_PATH = Path("Man_Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T2009020002_C2th4p5_man.star")

# STAR_PATH = Path("Man_Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T2009010021_C2th5_man.star")

# STAR_PATH = Path("Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T1001010006_C3_man.star")

STAR_PATH = Path("Nucleosome_coords_from_clustered_Reset_Z48_deduplicated_subtomo_coords_C2cc5.star")

PIXEL_SIZE_A = 1.513  # Å/px
TOMO_SIZE_PX = (4096, 4096, 3000)  # (X, Y, Z) bin1
BIN_SIZE_NM = 50   # cubic window size in nm
STEP_NM = 10       # sliding window step in nm
NUCLEOSOME_MASS_G = 4.018507e-19  # g per nucleosome
OUT_PNG = "fig2a_overlay.png"

def nm_scale():
    return PIXEL_SIZE_A / 10.0  # Å → nm

def load_coords_nm(star_path):
    df = starfile.read(star_path)
    if isinstance(df, dict):
        df = df["particles"]
    coords_nm = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy() * nm_scale()
    return coords_nm

def calc_density_mgml(coords_nm, bin_size_nm, step_nm):
    min_coords = coords_nm.min(axis=0)
    max_coords = coords_nm.max(axis=0)
    x_range = np.arange(min_coords[0], max_coords[0], step_nm)
    y_range = np.arange(min_coords[1], max_coords[1], step_nm)
    z_range = np.arange(min_coords[2], max_coords[2], step_nm)

    density_grid = np.zeros((len(x_range), len(y_range), len(z_range)))
    half_bin = bin_size_nm / 2.0
    cube_volume_ml = (bin_size_nm * 1e-9) ** 3 * 1e6  # m³ → mL

    for ix, x in enumerate(x_range):
        for iy, y in enumerate(y_range):
            for iz, z in enumerate(z_range):
                mask = (
                    (coords_nm[:, 0] >= x - half_bin) & (coords_nm[:, 0] < x + half_bin) &
                    (coords_nm[:, 1] >= y - half_bin) & (coords_nm[:, 1] < y + half_bin) &
                    (coords_nm[:, 2] >= z - half_bin) & (coords_nm[:, 2] < z + half_bin)
                )
                count = np.sum(mask)
                density_grid[ix, iy, iz] = (count * NUCLEOSOME_MASS_G / cube_volume_ml) * 1000  # mg/mL

    density_2d = density_grid.max(axis=2)  # Max projection along Z
    extent = [min_coords[0], max_coords[0], min_coords[1], max_coords[1]]
    return density_2d, extent

if __name__ == "__main__":
    print(f"[INFO] Reading STAR: {STAR_PATH}")
    coords_nm = load_coords_nm(STAR_PATH)
    print(f"[INFO] Loaded {coords_nm.shape[0]} particles")

    # Calculate density map
    print("[INFO] Calculating local density (mg/mL)...")
    density_2d, extent = calc_density_mgml(coords_nm, BIN_SIZE_NM, STEP_NM)

    # --- Make zeros white ---
    density_masked = np.ma.masked_where(density_2d <= 0, density_2d)  # mask zeros
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('white')  # masked (0) values → white

    # Plot overlay
    fig, ax = plt.subplots(figsize=(6, 6))

    # Background density heatmap (with zero as white)
    im = ax.imshow(
        density_masked.T, origin="lower", cmap=cmap, extent=extent, vmin=0,vmax=160
    )

    # Overlay particle coordinates
    ax.scatter(coords_nm[:, 0], coords_nm[:, 1], s=2, color="black", alpha=0.5)

    # Axis settings
    ax.set_xlim(0, TOMO_SIZE_PX[0] * nm_scale())
    ax.set_ylim(0, TOMO_SIZE_PX[1] * nm_scale())
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Overlay: Density ({BIN_SIZE_NM} nm cube, mg/mL) + Particles")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Local mass density (mg/mL)")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()

    print(f"[OK] Saved overlay figure: {OUT_PNG}")
