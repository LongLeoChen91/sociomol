# LC1B_subset_star_by_ids_from_csv_sorted_dedup.py
# Read IDs from edges CSV (i_id, j_id), concatenate in order WITHOUT de-duplication,
# extract matching rows from a big STAR table (allow duplicates during collection),
# THEN drop duplicates by _rlnTomoParticleId and sort ascending before writing.

import pandas as pd
import numpy as np
import starfile
from pathlib import Path

# ==== File paths (edit as needed) ====
BIG_STAR   = "subtomo_coords__tomo-HIV_Nuc_Lena_grid_1072.star"
EDGES_CSV  = "DoubleLinker_edges.csv"
OUT_STAR   = "Nucleosome_byCVS_from_subtomo_coords__tomo-HIV_Nuc_Lena_grid_1072.star"

# User option: whether to deduplicate by ID before saving
DEDUPLICATE = True   # True； set False to keep duplicates
# Define P threshold (set to 0.0 to disable filtering)
P_THRESHOLD = 0.1

# ---------- helpers ----------
def load_particles_tbl(star_path: str) -> pd.DataFrame:
    """Load the particles table from a STAR file; be robust to block names."""
    data = starfile.read(star_path, always_dict=True)
    for k in ("data_particles", "particles", "data_"):
        if k in data:
            return data[k]
    # Fallback to the first block if no common key matches
    return next(iter(data.values()))

def detect_id_col(df: pd.DataFrame) -> str:
    """Detect the tomo/particle ID column; prefer rlnTomoParticleId-like names."""
    candidates = [c for c in df.columns if "TomoParticleId" in c or c.lower().endswith("particleid")]
    if not candidates:
        # broader search
        candidates = [c for c in df.columns if "Id" in c or "ID" in c]
    if not candidates:
        raise KeyError("Could not find an ID column in the STAR table.")
    return candidates[0]

def to_int_like_str(x) -> str:
    """Normalize IDs to an int-like string (e.g., '1046.0' -> '1046', '  27 ' -> '27')."""
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s  # last resort

def find_col_case_insensitive(df: pd.DataFrame, *names: str) -> str:
    """Return the first matching column name in df among given names (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"Missing required column; tried: {names}")

# ---------- main ----------
def main():
    # 1) Load the big STAR table
    df_big = load_particles_tbl(BIG_STAR)
    col_id_big = detect_id_col(df_big)

    # Normalize IDs from the big table to int-like strings (for robust matching)
    big_ids_norm = df_big[col_id_big].map(to_int_like_str)

    # Build a lookup dict: normalized ID -> list of row indices in df_big
    # (If the same ID appears multiple times in the big STAR, keep all indices.)
    id_to_rows = {}
    for idx, nid in enumerate(big_ids_norm.values):
        id_to_rows.setdefault(nid, []).append(idx)

    # 2) Load edges CSV and get i_id / j_id columns (case-insensitive)
    edges = pd.read_csv(EDGES_CSV)

    if P_THRESHOLD > 0:
        edges = edges[edges["P"] > P_THRESHOLD]
        print(f"Applied P > {P_THRESHOLD} filter, remaining edges: {len(edges)}")
    else:
        print("No P filtering applied.")
    col_i = find_col_case_insensitive(edges, "i_id", "I_ID", "iId", "iidx", "i_idx")
    col_j = find_col_case_insensitive(edges, "j_id", "J_ID", "jId", "jidx", "j_idx")

    # Concatenate i_id and j_id in the original order WITHOUT de-duplication
    ids_concat = pd.concat([edges[col_i], edges[col_j]], ignore_index=True)
    ids_norm = ids_concat.map(to_int_like_str)

    # 3) Collect rows from df_big in the same order as ids_norm (allow duplicates here)
    out_rows = []
    missing_ids = []
    for nid in ids_norm:
        rows = id_to_rows.get(nid, [])
        if rows:
            # Append all matching rows if the ID appears multiple times in df_big
            for r in rows:
                out_rows.append(df_big.iloc[r])
        else:
            missing_ids.append(nid)

    if not out_rows:
        raise RuntimeError("No rows matched any IDs from the CSV.")

    df_out = pd.DataFrame(out_rows).reset_index(drop=True)

    # 4) Sort ascending by _rlnTomoParticleId (or detected ID column) and drop duplicates
    col_id_out = detect_id_col(df_out)

    # Convert ID column to numeric for stable sorting; keep non-convertible as NaN
    df_out[col_id_out] = pd.to_numeric(df_out[col_id_out], errors="coerce")

    if DEDUPLICATE:
        # Drop duplicate IDs, keep the first occurrence
        df_out = df_out.drop_duplicates(subset=[col_id_out], keep="first")
        print(f"De-duplicated by column {col_id_out}, rows left: {len(df_out)}")
    else:
        print("Skipping de-duplication, keeping all duplicates.")

    # Sort ascending by ID
    df_out = df_out.sort_values(by=col_id_out, ascending=True).reset_index(drop=True)

    # 5) Write out to STAR
    starfile.write({"particles": df_out}, OUT_STAR, overwrite=True)

    # 6) Report
    print(f"[OK] Wrote sorted & deduplicated STAR: {OUT_STAR}")
    print(f" - Input STAR rows: {len(df_big)}")
    print(f" - Requested IDs (with duplicates): {len(ids_norm)}")
    print(f" - Output rows (unique by {col_id_out}): {len(df_out)}")
    if missing_ids:
        uniq_missing = sorted(set(missing_ids))
        show = uniq_missing[:20]
        print(f" - Missing IDs (not found in STAR): {len(uniq_missing)} unique "
              f"({len(show)} shown) -> {show}")

if __name__ == "__main__":
    main()
