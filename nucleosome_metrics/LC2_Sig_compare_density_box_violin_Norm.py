# SigV4_compare_density_box_violin_effect_colored.py
"""
Equal-area violin + box for local nucleosome mass density (mg/mL),
with Kruskal–Wallis (global), Dunn's post-hoc (Bonferroni),
and effect sizes (eta-squared for KW, Cliff's delta for pairs).

NEW:
- Significance bars colored by effect-size class (Cliff's delta).
- Text on bars shows stars + delta value and class (e.g., '**  Δ=0.08 (small)').
- Use numpy.trapezoid (no deprecation warning).

Outputs:
- density_violin_equalarea_sig.png
- density_violin_equalarea_stats.csv
- pairwise_dunn_pvalues.csv
- pairwise_effect_sizes.csv
"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import starfile
from scipy.stats import gaussian_kde, kruskal

try:
    import scikit_posthocs as sp
except ImportError as e:
    raise SystemExit("Please install scikit-posthocs: pip install scikit-posthocs") from e

# ---------------- User params ----------------
STAR_FILES: List[str] = [
    "Man_Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T2009010021_C2th5_man.star",
    "Man_Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T2009020002_C2th4p5_man.star",
    "Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T1001010006_C3_man.star",
    "Nucleosome_coords_from_clustered_Reset_Z48_deduplicated_subtomo_coords_C2cc5.star",
]
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
OUT_STAT_CSV = "density_violin_equalarea_stats.csv"
OUT_DUNN_CSV = "pairwise_dunn_pvalues.csv"
OUT_EFFECT_CSV = "pairwise_effect_sizes.csv"

SHOW_FLIERS_IN_BOX = False
VIOLIN_TARGET_AREA = 20.0       # constant area for violins: ∫ 2*halfwidth(y) dy
VIOLIN_MAX_HALFWIDTH = 0.45
VIOLIN_ALPHA = 0.6
BOX_ALPHA = 0.25
MEDIAN_LW = 1.8

USE_GLOBAL_BOUNDS = False       # if True, all groups use same 3D grid bounds
DOWNSAMPLE_NONZERO: int | None = None
YLIM = (0, 160)
ALPHA_SIG = 0.05

# Colors by effect-size class (Cliff's delta absolute value)
EFFECT_COLORS = {
    "negligible": "#9E9E9E",   # grey
    "small":      "#4CAF50",   # green
    "medium":     "#FF9800",   # orange
    "large":      "#E53935",   # red
}

# ---------------- Helpers ----------------
def nm_scale() -> float:
    """Å -> nm."""
    return PIXEL_SIZE_A / 10.0

def load_coords_nm(star_path: str) -> np.ndarray:
    """Read STAR and return Nx3 coords in nm."""
    df = starfile.read(star_path)
    if isinstance(df, dict):
        df = df["particles"]
    cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    if not set(cols).issubset(df.columns):
        raise ValueError(f"Missing {cols} in {star_path}")
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
    if coords_nm.size == 0:
        return np.zeros((0, 0, 0), np.float32)
    mn, mx = (coords_nm.min(axis=0), coords_nm.max(axis=0)) if bounds is None else bounds
    xr, yr, zr = (np.arange(mn[i], mx[i], step_nm) for i in range(3))
    grid = np.zeros((len(xr), len(yr), len(zr)), np.float32)
    half, vol_ml = bin_size_nm/2.0, (bin_size_nm*1e-9)**3 * 1e6
    for ix, x in enumerate(xr):
        mxmask = (coords_nm[:, 0] >= x-half) & (coords_nm[:, 0] < x+half)
        if not np.any(mxmask):
            continue
        for iy, y in enumerate(yr):
            mxy = mxmask & (coords_nm[:, 1] >= y-half) & (coords_nm[:, 1] < y+half)
            if not np.any(mxy):
                continue
            zvals = coords_nm[mxy, 2]
            for iz, z in enumerate(zr):
                count = np.count_nonzero((zvals >= z-half) & (zvals < z+half))
                grid[ix, iy, iz] = (count * mass_g / vol_ml) * 1000.0
    return grid

def nonzero_voxels(grid: np.ndarray, cap: int | None = None) -> np.ndarray:
    """Flatten to non-zero values; optional random down-sample to cap."""
    vals = grid[grid > 0].astype(float)
    if cap and vals.size > cap:
        idx = np.random.default_rng().choice(vals.size, size=cap, replace=False)
        vals = vals[idx]
    return vals

def kde_pdf(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1D KDE normalized to unit area; uses numpy.trapezoid (no deprecation)."""
    if values.size < 2:
        mu = values.mean() if values.size == 1 else y.mean()
        sigma = np.std(values) + 1e-6
        pdf = np.exp(-0.5 * ((y - mu) / sigma) ** 2)
    else:
        pdf = gaussian_kde(values).evaluate(y)
    area = np.trapezoid(pdf, y)
    return pdf / area if area > 0 else pdf

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
    margin = 0.02 * (ymax - ymin + 1e-9)
    y = np.linspace(ymin - margin, ymax + margin, 400)

    pdfs = [kde_pdf(g, y) if g.size > 0 else np.zeros_like(y) for g in groups]
    halfwidths = [(target_area / 2.0) * p for p in pdfs]  # ∫2*hw*pdf dy = target_area
    cur_max = max((hw.max() for hw in halfwidths), default=0)
    if cur_max > max_halfwidth > 0:
        scale = max_halfwidth / cur_max
        halfwidths = [hw * scale for hw in halfwidths]

    pos = np.arange(1, len(groups) + 1)
    for p, hw in zip(pos, halfwidths):
        ax.fill_betweenx(y, p - hw, p + hw, alpha=violin_alpha, color="#FC8338", linewidth=0)

    b = ax.boxplot(
        groups, positions=pos, widths=0.12, showfliers=show_fliers, patch_artist=True,
        boxprops=dict(facecolor="none", alpha=1, edgecolor="black", linewidth=1.4),
        medianprops=dict(color="black", linewidth=median_lw),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
    )
    for patch in b["boxes"]:
        patch.set_alpha(1)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=0, ha="center")

def p_to_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def effect_class(delta_abs: float) -> str:
    """Classify effect size by |Cliff's delta|."""
    if delta_abs < 0.147:
        return "negligible"
    if delta_abs < 0.33:
        return "small"
    if delta_abs < 0.474:
        return "medium"
    return "large"

def add_sig_bars_with_effect(ax: plt.Axes,
                             pairs: List[Tuple[int, int]],
                             pvals: List[float],
                             deltas: List[float],
                             y_start: float,
                             y_step: float):
    """
    Draw significance bars with effect-size coloring and text:
    e.g., '**  Δ=0.08 (small)'
    """
    for i, ((a, b), p, d) in enumerate(zip(pairs, pvals, deltas)):
        y = y_start + i * y_step
        d_abs = abs(d)
        cls = effect_class(d_abs)
        color = EFFECT_COLORS.get(cls, "#000000")
        # Bar
        ax.plot([a, a, b, b], [y, y + y_step * 0.3, y + y_step * 0.3, y],
                lw=1.4, c=color)
        # Label
        ax.text((a + b) / 2., y + y_step * 0.35,
                f"{p_to_stars(p)}  Δ={d_abs:.2f} ({cls})",
                ha='center', va='bottom', fontsize=10, color=color)

# ---- Effect sizes ----
def eta_squared_kruskal(H: float, groups: List[np.ndarray]) -> float:
    """Eta-squared for Kruskal–Wallis: η² = (H - k + 1)/(N - k)."""
    k = len(groups)
    N = sum(len(g) for g in groups)
    return max(0.0, (H - k + 1) / max(1, N - k))

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: P(X > Y) - P(X < Y).
    Range [-1,1]; |delta|: 0.147 small, 0.33 medium, 0.474 large.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    # Efficient rank-based computation (via Mann–Whitney U)
    xy = np.concatenate([x, y])
    ranks = pd.Series(xy).rank(method='average').to_numpy()
    rx = ranks[:x.size].sum()
    ry = ranks[x.size:].sum()
    Ux = rx - x.size * (x.size + 1) / 2.0
    Uy = ry - y.size * (y.size + 1) / 2.0
    delta = (Ux - Uy) / (x.size * y.size)
    return float(delta)

# ---------------- Main ----------------
if __name__ == "__main__":
    # 1) Load coords & optionally compute global bounds
    all_coords = [load_coords_nm(p) for p in STAR_FILES]
    gb = compute_bounds(all_coords) if USE_GLOBAL_BOUNDS else None

    # 2) Build distributions (non-zero voxels) & per-group stats
    dists, rows, labels = [], [], []
    for star, coords in zip(STAR_FILES, all_coords):
        print(f"[INFO] Processing: {star}")
        grid = build_density_grid_mgml(coords, BIN_SIZE_NM, STEP_NM, NUCLEOSOME_MASS_G, bounds=gb)
        vals = nonzero_voxels(grid, cap=DOWNSAMPLE_NONZERO)
        dists.append(vals)
        label = LABEL_MAP.get(Path(star).stem, Path(star).stem)
        labels.append(label)
        rows.append(dict(
            group=label,
            n_voxels_nonzero=int(vals.size),
            min_mgml=float(vals.min()) if vals.size else np.nan,
            mean_mgml=float(vals.mean()) if vals.size else np.nan,
            median_mgml=float(np.median(vals)) if vals.size else np.nan,
            max_mgml=float(vals.max()) if vals.size else np.nan,
        ))

    # 3) Global test: Kruskal–Wallis + eta-squared
    kw_stat, kw_p = kruskal(*[g for g in dists if g.size > 0])
    eta2 = eta_squared_kruskal(kw_stat, dists)
    print(f"[STATS] Kruskal–Wallis: H={kw_stat:.3f}, p={kw_p:.3e}, eta^2={eta2:.3f}")

    # 4) Post-hoc: Dunn (Bonferroni) + pairwise effect sizes (Cliff's delta & median diffs)
    dunn = sp.posthoc_dunn(dists, p_adjust='bonferroni')
    dunn.index = labels
    dunn.columns = labels
    dunn.to_csv(OUT_DUNN_CSV)
    print(f"[OK] Saved Dunn p-values (Bonferroni): {OUT_DUNN_CSV}")

    # Pairwise effect sizes table (+ class)
    pairs_rows = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a, b = labels[i], labels[j]
            delta = cliffs_delta(dists[i], dists[j])
            med_diff = float(np.median(dists[i]) - np.median(dists[j]))
            cls = effect_class(abs(delta))
            pairs_rows.append(dict(
                group_A=a, group_B=b,
                cliffs_delta=delta,
                effect_class=cls,
                median_diff_mgml=med_diff,
                dunn_p_bonf=float(dunn.loc[a, b])
            ))
    pd.DataFrame(pairs_rows).to_csv(OUT_EFFECT_CSV, index=False)
    print(f"[OK] Saved effect sizes: {OUT_EFFECT_CSV}")

    # 5) Plot violin + box + significance bars with effect-size coloring
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    draw_equal_area_violins(ax, dists, labels,
                            target_area=VIOLIN_TARGET_AREA,
                            max_halfwidth=VIOLIN_MAX_HALFWIDTH,
                            violin_alpha=VIOLIN_ALPHA,
                            box_alpha=BOX_ALPHA,
                            median_lw=MEDIAN_LW,
                            show_fliers=SHOW_FLIERS_IN_BOX)
    ax.set_ylabel("Local nucleosome mass density (mg/mL)", fontsize=13)
    ax.set_title(f"Local density distributions (cube={BIN_SIZE_NM} nm, step={STEP_NM} nm)\n"
                 f"Kruskal–Wallis p={kw_p:.2e}, η²={eta2:.3f}", fontsize=12, pad=10)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    y_min, y_max = ax.get_ylim()
    

    # Collect significant pairs (p < alpha)
    pairs_idx, pvals, deltas = [], [], []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            p = float(dunn.iloc[i, j])
            if p < ALPHA_SIG:
                pairs_idx.append((i+1, j+1))  # 1-based plotting positions
                pvals.append(p)
                # Get corresponding delta from the table we saved
                d = [r["cliffs_delta"] for r in pairs_rows if r["group_A"] == labels[i] and r["group_B"] == labels[j]][0]
                deltas.append(d)

    # Draw bars above the plot; spacing based on number of pairs
    if pairs_idx:
        y_start = y_max * 0.8
        y_step = (y_max - y_min) * 0.05
        add_sig_bars_with_effect(ax, pairs_idx, pvals, deltas, y_start=y_start, y_step=y_step)
        ax.set_ylim(y_min, y_start + y_step * (len(pairs_idx) + 1))
    # ax.set_ylim(0, 160)

    # Legend for effect-size colors
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0],[0], color=c, lw=2, label=k) for k,c in EFFECT_COLORS.items()]
    ax.legend(handles=legend_elems, title="Effect size (|Cliff's Δ|)", loc="upper left", frameon=False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()
    print(f"[OK] Saved plot: {OUT_PNG}")

    # 6) Save per-group summary
    pd.DataFrame(rows).to_csv(OUT_STAT_CSV, index=False)
    print(f"[OK] Saved summary stats: {OUT_STAT_CSV}")
