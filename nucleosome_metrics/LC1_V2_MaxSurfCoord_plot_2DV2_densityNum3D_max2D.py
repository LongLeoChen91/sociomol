# compare_density_violin_equalarea_sig.py
# Bug fix: `from __future__ import annotations` must be the first statement
# in the file (after module docstring only). Moved here from line 21.
from __future__ import annotations
"""
Area-normalized violin + box for local nucleosome mass density (mg/mL)
with statistical testing and significance annotations.

Pipeline:
- For each STAR: load coords(px)->nm, build 3D sliding-window density (mg/mL)
- Take NON-ZERO voxels as distribution
- Plot equal-area violins + slim box overlay
- Kruskal–Wallis (global) + Dunn's pairwise test (Bonferroni)
- Annotate significant pairs on the plot
- Save PNG + per-group stats CSV + pairwise p-values CSV
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import starfile
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.stats import gaussian_kde, kruskal

# Requires: pip install scikit-posthocs
try:
    import scikit_posthocs as sp
except ImportError as e:
    raise SystemExit(
        "scikit-posthocs is required.\n"
        "Install via: pip install scikit-posthocs"
    ) from e

# ============== User params ==============
STAR_FILES: List[str] = [
    "Man_Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T2009010021_C2th5_man.star",
    "Man_Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T2009020002_C2th4p5_man.star",
    "Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T1001010006_C3_man.star",
    "Nucleosome_coords_from_clustered_Reset_Z48_deduplicated_subtomo_coords_C2cc5.star",
]
# X-axis labels (short names or descriptive)
LABEL_MAP: Dict[str, str] = {
    "Man_Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T2009010021_C2th5_man": "Assembly\nintermediate",
    "Man_Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T2009020002_C2th4p5_man": "Paracrystalline\narray",
    "Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T1001010006_C3_man": "Fully packaged\nvirion",
    "Nucleosome_coords_from_clustered_Reset_Z48_deduplicated_subtomo_coords_C2cc5": "No virus\nnearby",
}

PIXEL_SIZE_A = 1.513
BIN_SIZE_NM = 50.0
STEP_NM = 10.0
NUCLEOSOME_MASS_G = 4.018507e-19  # g per nucleosome

OUT_PNG = "density_violin_equalarea_sig.png"
OUT_CSV = "density_violin_equalarea_stats.csv"
OUT_DUNN_CSV = "pairwise_dunn_pvalues.csv"

SHOW_FLIERS_IN_BOX = False
VIOLIN_TARGET_AREA = 20.0       # ∫ 2*halfwidth(y) dy
VIOLIN_MAX_HALFWIDTH = 0.45
VIOLIN_ALPHA = 0.6
BOX_ALPHA = 0.25
MEDIAN_LW = 1.8

USE_GLOBAL_BOUNDS = False
DOWNSAMPLE_NONZERO: int | None = None
YLIM = (0, 160)                  # y-axis (mg/mL) for consistency; set to None to auto
ALPHA_SIG = 0.05                 # significance threshold for stars

# ============== Helpers ==============
def nm_scale() -> float:
    """Å -> nm."""
    return PIXEL_SIZE_A / 10.0

def load_coords_nm(star_path: str) -> np.ndarray:
    """Read STAR and return Nx3 coords in nm."""
    df = starfile.read(star_path)
    if isinstance(df, dict): df = df["particles"]
    cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    if not set(cols).issubset(df.columns):
        raise ValueError(f"Missing columns in {star_path}")
    return df[cols].to_numpy(float) * nm_scale()

def compute_bounds(coord_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Global min/max across coord arrays."""
    mins = np.vstack([c.min(axis=0) for c in coord_list]).min(axis=0)
    maxs = np.vstack([c.max(axis=0) for c in coord_list]).max(axis=0)
    return mins, maxs

def build_density_grid_mgml(coords_nm: np.ndarray,
                            bin_size_nm: float,
                            step_nm: float,
                            mass_g: float,
                            bounds: Tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """3D sliding-window local mass density (mg/mL)."""
    if coords_nm.size == 0: return np.zeros((0,0,0), np.float32)
    mn, mx = (coords_nm.min(axis=0), coords_nm.max(axis=0)) if bounds is None else bounds
    xr, yr, zr = (np.arange(mn[i], mx[i], step_nm) for i in range(3))
    grid = np.zeros((len(xr), len(yr), len(zr)), np.float32)
    half, vol_ml = bin_size_nm/2.0, (bin_size_nm*1e-9)**3 * 1e6
    for ix, x in enumerate(xr):
        mxmask = (coords_nm[:,0] >= x-half) & (coords_nm[:,0] < x+half)
        if not np.any(mxmask): continue
        for iy, y in enumerate(yr):
            mxy = mxmask & (coords_nm[:,1] >= y-half) & (coords_nm[:,1] < y+half)
            if not np.any(mxy): continue
            zvals = coords_nm[mxy, 2]
            for iz, z in enumerate(zr):
                count = np.count_nonzero((zvals >= z-half) & (zvals < z+half))
                grid[ix,iy,iz] = (count * mass_g / vol_ml) * 1000.0
    return grid

def nonzero_voxels(grid: np.ndarray, cap: int | None = None) -> np.ndarray:
    """Flatten to non-zero values; optional random down-sample to cap."""
    vals = grid[grid > 0].astype(float)
    if cap and vals.size > cap:
        idx = np.random.default_rng().choice(vals.size, size=cap, replace=False)
        vals = vals[idx]
    return vals

def kde_pdf(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1D KDE normalized to unit area."""
    # Bug fix: guard against empty array before any operation
    if values.size == 0:
        return np.zeros_like(y)
    if values.size < 2:
        mu = values.mean()          # size == 1, always safe
        sigma = np.std(values) + 1e-6  # std of single value is 0; +eps avoids /0
        pdf = np.exp(-0.5*((y-mu)/sigma)**2)
    else:
        pdf = gaussian_kde(values).evaluate(y)
    area = np.trapz(pdf, y)
    return pdf/area if area > 0 else pdf

def draw_equal_area_violins(ax: plt.Axes,
                            groups: List[np.ndarray],
                            labels: List[str],
                            target_area: float,
                            max_halfwidth: float,
                            violin_alpha: float,
                            box_alpha: float,
                            median_lw: float,
                            show_fliers: bool):
    """Draw equal-area violins + slim box overlay."""
    nonempty = [g for g in groups if g.size > 0]
    if not nonempty:
        return
    ymin, ymax = min(g.min() for g in nonempty), max(g.max() for g in nonempty)
    margin = 0.02*(ymax - ymin + 1e-9)
    y = np.linspace(ymin - margin, ymax + margin, 400)

    pdfs = [kde_pdf(g, y) if g.size > 0 else np.zeros_like(y) for g in groups]
    halfwidths = [ (target_area/2.0) * p for p in pdfs ]   # ∫2*hw*pdf dy = target_area
    cur_max = max((hw.max() for hw in halfwidths), default=0)
    if cur_max > max_halfwidth > 0:
        scale = max_halfwidth/cur_max
        halfwidths = [hw*scale for hw in halfwidths]

    pos = np.arange(1, len(groups)+1)
    for p, hw in zip(pos, halfwidths):
        ax.fill_betweenx(y, p-hw, p+hw, alpha=violin_alpha, color="#FC8338", linewidth=0)

    b = ax.boxplot(
        groups, positions=pos, widths=0.12, showfliers=show_fliers, patch_artist=True,
        boxprops=dict(facecolor="none", alpha=1, edgecolor="black", linewidth=1.4),
        medianprops=dict(color="black", linewidth=median_lw),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2)
    )
    for patch in b["boxes"]:
        patch.set_alpha(1)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=0, ha="center")

def p_to_stars(p: float) -> str:
    """Convert p-value to star annotation."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def add_sig_bars(ax: plt.Axes,
                 pairs: List[Tuple[int, int]],
                 pvals: List[float],
                 y_start: float,
                 y_step: float,
                 line_width: float = 1.2,
                 line_color: str = "black",
                 text_offset: float = 0.01):
    """
    Draw significance bars for given index pairs (1-based positions).
    y_start: starting y in data coords; y_step: vertical increment per bar.
    """
    for i, ((a, b), p) in enumerate(zip(pairs, pvals)):
        y = y_start + i * y_step
        ax.plot([a, a, b, b], [y, y + y_step*0.3, y + y_step*0.3, y],
                lw=line_width, c=line_color)
        ax.text((a + b)/2., y + y_step*0.35, p_to_stars(p),
                ha='center', va='bottom', fontsize=11)

# ============== Main ==============
if __name__ == "__main__":
    # Load all coords (for optional global bounds)
    all_coords = [load_coords_nm(p) for p in STAR_FILES]
    gb = compute_bounds(all_coords) if USE_GLOBAL_BOUNDS else None

    # Build distributions and per-group stats
    dists, rows, labels = [], [], []
    for star, coords in zip(STAR_FILES, all_coords):
        print(f"[INFO] Processing: {star}")
        grid = build_density_grid_mgml(coords, BIN_SIZE_NM, STEP_NM, NUCLEOSOME_MASS_G, bounds=gb)
        vals = nonzero_voxels(grid, cap=DOWNSAMPLE_NONZERO)
        dists.append(vals)
        label = LABEL_MAP.get(Path(star).stem, Path(star).stem)
        labels.append(label)
        rows.append(dict(
            star=label,
            n_voxels_nonzero=int(vals.size),
            min_mgml=float(vals.min()) if vals.size else np.nan,
            mean_mgml=float(vals.mean()) if vals.size else np.nan,
            median_mgml=float(np.median(vals)) if vals.size else np.nan,
            max_mgml=float(vals.max()) if vals.size else np.nan,
        ))

    # ---------- Global test: Kruskal–Wallis ----------
    kw_stat, kw_p = kruskal(*[g for g in dists if g.size > 0])
    print(f"[STATS] Kruskal–Wallis: H={kw_stat:.3f}, p={kw_p:.3e}")

    # ---------- Post-hoc: Dunn's test (Bonferroni) ----------
    # Bug fix: posthoc_dunn crashes/gives wrong results when passed empty arrays.
    # Filter to non-empty groups before calling, then build a full NaN matrix
    # and fill in the computed p-values so that indexing by label still works.
    nonempty_mask = [g.size > 0 for g in dists]
    nonempty_dists = [g for g, ok in zip(dists, nonempty_mask) if ok]
    nonempty_labels = [lb for lb, ok in zip(labels, nonempty_mask) if ok]

    import itertools
    dunn_full = pd.DataFrame(np.nan, index=labels, columns=labels)
    if len(nonempty_dists) >= 2:
        dunn_sub = sp.posthoc_dunn(nonempty_dists, p_adjust='bonferroni')
        dunn_sub.index = nonempty_labels
        dunn_sub.columns = nonempty_labels
        for r, c in itertools.product(nonempty_labels, nonempty_labels):
            dunn_full.loc[r, c] = dunn_sub.loc[r, c]
    else:
        print("[WARN] Fewer than 2 non-empty groups; skipping Dunn's test.")
    dunn = dunn_full
    dunn.to_csv(OUT_DUNN_CSV)
    print(f"[OK] Saved Dunn pairwise p-values: {OUT_DUNN_CSV}")

    # ---------- Plot violin+box ----------
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    draw_equal_area_violins(ax, dists, labels,
                            target_area=VIOLIN_TARGET_AREA,
                            max_halfwidth=VIOLIN_MAX_HALFWIDTH,
                            violin_alpha=VIOLIN_ALPHA,
                            box_alpha=BOX_ALPHA,
                            median_lw=MEDIAN_LW,
                            show_fliers=SHOW_FLIERS_IN_BOX)
    ax.set_ylabel("Local nucleosome mass density (mg/mL)", fontsize=13)
    ax.set_title(f"Local density distributions (cube={BIN_SIZE_NM} nm, step={STEP_NM} nm)\n"
                 f"Kruskal–Wallis p={kw_p:.2e}", fontsize=12, pad=10)

    if YLIM is not None:
        ax.set_ylim(*YLIM)
    y_min, y_max = ax.get_ylim()

    # ---------- Add significance annotations (pairs with p < alpha) ----------
    # Build all pair indices (1-based positions)
    n = len(labels)
    pairs_idx = []
    pvals = []
    for i in range(n):
        for j in range(i+1, n):
            p = dunn.iloc[i, j]
            if p < ALPHA_SIG:
                pairs_idx.append((i+1, j+1))  # positions are 1..n
                pvals.append(p)

    if pairs_idx:
        # Start a bit above top whisker
        y_start = y_max * 1.02 if YLIM is None else (YLIM[1] * 1.02)
        # Step: fraction of the axis range
        y_step = (y_max - y_min) * 0.05
        add_sig_bars(ax, pairs_idx, pvals, y_start=y_start, y_step=y_step)
        # Extend ylim to fit bars
        ax.set_ylim(y_min, y_start + y_step * (len(pairs_idx) + 1))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()
    print(f"[OK] Saved: {OUT_PNG}")

    # Save per-group summary stats
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved: {OUT_CSV}")
