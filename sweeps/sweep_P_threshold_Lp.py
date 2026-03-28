"""
sweep_P_threshold_Lp.py
-----------------------
Sweeps the probability threshold P and computes the effective persistence
length Lp* at each threshold level using the WLC-MLE analytic formula.

This script answers: "How does our Lp* estimate change as we raise the
quality bar on which linker predictions we include?"

Output (2-panel figure):
  Panel 1:  Lp* (nm) vs P_threshold
            - Blue line = MLE estimate
            - Shaded band = ±Lp* × sqrt(2/N)  (Cramér-Rao bound)
            - Vertical dashed line = P_THRESHOLD_MAP from config
  Panel 2:  N_valid vs P_threshold
            - Orange bars / line showing how many points survive each cut
            - Guides intuition: high P = fewer but cleaner points

Physics note:
  Lp* = 2N / Σ(θ_rad² / L_true)   (analytic WLC-MLE)
  Relative error ≈ sqrt(2/N)       (Cramér-Rao)
  L_true = L_meas - 2 * R_OFFSET_NM (geometric correction)

Author: Long Chen
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
os.chdir(_SCRIPT_DIR)

from tools.config_plot import CSV_PATH, P_THRESHOLD_MAP, R_OFFSET_NM, L_MIN_NM

# ============================================================
# 1. Configuration
# ============================================================
THETA_COL = "theta_deg"
L_COL     = "L_nm"

P_START  = 0.03
P_STOP   = 0.40
P_STEP   = 0.02
P_THRESHOLDS = np.arange(P_START, P_STOP + P_STEP * 0.5, P_STEP)

# ============================================================
# 2. Load Full Dataset (no P filter yet)
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[THETA_COL, L_COL, "P"])

print(f"[INFO] CSV: {os.path.basename(CSV_PATH)}")
print(f"[INFO] Total rows: {len(df)}  |  R_OFFSET={R_OFFSET_NM} nm  |  L_MIN={L_MIN_NM} nm")
print(f"[INFO] Sweeping P_threshold: {P_START:.2f} → {P_STOP:.2f}  step={P_STEP:.2f}\n")

# ============================================================
# 3. Sweep
# ============================================================
results = []   # list of (thresh, N_raw, N_valid, lp_star, rel_err)

for thresh in P_THRESHOLDS:
    # 3a. P filter
    df_sub = df[df["P"] > thresh].copy()
    N_raw = len(df_sub)

    if N_raw == 0:
        results.append((thresh, 0, 0, np.nan, np.nan))
        print(f"  P>{thresh:.2f}:  N_raw=0  — skipped")
        continue

    # 3b. L geometric correction
    L_meas = df_sub[L_COL].values
    theta_deg = df_sub[THETA_COL].values
    L_true = L_meas - 2.0 * R_OFFSET_NM
    valid  = L_true > L_MIN_NM
    L_v    = L_true[valid]
    T_v_rad = np.radians(theta_deg[valid])
    N_valid = len(L_v)

    if N_valid < 3:
        results.append((thresh, N_raw, N_valid, np.nan, np.nan))
        print(f"  P>{thresh:.2f}:  N_raw={N_raw:4d}  N_valid={N_valid:4d}  — too few for MLE")
        continue

    # 3c. Analytic WLC-MLE
    denom  = np.sum(T_v_rad**2 / L_v)
    lp_star = 2.0 * N_valid / denom if denom > 0 else np.nan
    rel_err = np.sqrt(2.0 / N_valid)

    results.append((thresh, N_raw, N_valid, lp_star, rel_err))
    print(f"  P>{thresh:.2f}:  N_raw={N_raw:4d}  N_valid={N_valid:4d}  "
          f"Lp* = {lp_star:7.2f} nm  +/-{rel_err*100:.0f}%")

# ============================================================
# 4. Unpack & Plot
# ============================================================
P_arr      = np.array([r[0] for r in results])
N_raw_arr  = np.array([r[1] for r in results])
N_val_arr  = np.array([r[2] for r in results])
lp_arr     = np.array([r[3] for r in results])
err_arr    = np.array([r[4] for r in results])

valid_plot = ~np.isnan(lp_arr)

# ============================================================
# 4b. Stability Detection: rolling window CV < 5%
# ============================================================
STABLE_CV_THRESH = 0.05    # coefficient of variation threshold
WINDOW           = 3       # consecutive points to check

stable_windows = []
for i in range(len(lp_arr) - WINDOW + 1):
    seg = lp_arr[i : i + WINDOW]
    if np.any(np.isnan(seg)):
        continue
    cv = np.std(seg) / np.mean(seg)
    if cv < STABLE_CV_THRESH:
        stable_windows.append((i, i + WINDOW - 1, np.mean(seg), cv))

if stable_windows:
    # merge overlapping windows into contiguous spans
    merged = [stable_windows[0]]
    for sw in stable_windows[1:]:
        if sw[0] <= merged[-1][1] + 1:    # overlapping or adjacent
            merged[-1] = (merged[-1][0], sw[1],
                          np.nanmean(lp_arr[merged[-1][0]:sw[1]+1]),
                          merged[-1][3])
        else:
            merged.append(sw)
    print("\n[Stability Analysis]  (rolling window CV < 5%)")
    for start_i, end_i, lp_mean, cv in merged:
        P_lo = P_arr[start_i]
        P_hi = P_arr[end_i]
        print(f"  Stable window:  P=[{P_lo:.2f}, {P_hi:.2f}]  "
              f"Lp* = {lp_mean:.2f} nm  (CV={cv*100:.1f}%)")
    best = max(merged, key=lambda s: s[1] - s[0])   # longest stable span
    print(f"\n  ==> Recommended Lp* = {best[2]:.2f} nm  "
          f"(P=[{P_arr[best[0]]:.2f}, {P_arr[best[1]]:.2f}])")
else:
    print("\n[Stability Analysis]  No stable window found (CV < 5%).")
    best = None

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                gridspec_kw={"height_ratios": [3, 1.5]},
                                sharex=True)

# --- Panel 1: Lp* with CR error band ---
lp_v   = lp_arr[valid_plot]
err_v  = err_arr[valid_plot]
P_v    = P_arr[valid_plot]

ax1.fill_between(P_v,
                 lp_v * (1 - err_v),
                 lp_v * (1 + err_v),
                 alpha=0.20, color="steelblue",
                 label=r"$\pm\sqrt{2/N}\cdot L_p^*$ (Cramér-Rao)")
ax1.plot(P_v, lp_v, "o-", color="steelblue", ms=5, lw=1.8,
         label=r"$L_p^*$ (WLC-MLE analytic)")

# Mark current config threshold
ax1.axvline(P_THRESHOLD_MAP, color="crimson", linestyle="--", lw=1.2,
            label=f"config P_THRESHOLD_MAP = {P_THRESHOLD_MAP}")

ax1.set_ylabel(r"Effective $L_p^*$  (nm)", fontsize=12)
ax1.set_title(
    f"Lp* Convergence vs Probability Threshold\n"
    f"{os.path.basename(CSV_PATH)}  |  R_OFFSET={R_OFFSET_NM} nm",
    fontsize=11
)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

# Highlight stable window(s) on panel 1
if stable_windows:
    for start_i, end_i, lp_mean, cv in merged:
        ax1.axvspan(P_arr[start_i] - P_STEP / 2,
                    P_arr[end_i]   + P_STEP / 2,
                    color="limegreen", alpha=0.15, zorder=0)
    if best:
        ax1.axhline(best[2], color="green", linestyle=":", lw=1.2,
                    label=f"Recommended Lp* = {best[2]:.1f} nm")
        ax1.legend(fontsize=9, loc="upper right")

# --- Panel 2: N_valid bar chart ---
ax2.bar(P_arr, N_val_arr, width=P_STEP * 0.8,
        color="darkorange", alpha=0.75, label="N_valid (after L correction)")
ax2.bar(P_arr, N_raw_arr - N_val_arr, width=P_STEP * 0.8,
        bottom=N_val_arr, color="salmon", alpha=0.5, label="Outliers removed")
ax2.axvline(P_THRESHOLD_MAP, color="crimson", linestyle="--", lw=1.2)
ax2.set_xlabel("Probability threshold  P", fontsize=12)
ax2.set_ylabel("N points", fontsize=11)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3, axis="y")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
out_png = "sweep_P_threshold_Lp.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"\n[OK] Saved: {out_png}")
