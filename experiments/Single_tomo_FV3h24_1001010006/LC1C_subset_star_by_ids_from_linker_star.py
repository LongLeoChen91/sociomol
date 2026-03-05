#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LC1C_subset_star_by_ids_from_linker_star (ordered by Linker + keep selected linker columns)

Purpose:
  For each stick in Linker_Sticks.star, extract rlnStickSourceId1 and rlnStickSourceId2
  from BIG_STAR (subtomo_coords...), following the exact order of sticks.
  Keep duplicates; do not sort.
  Additionally, copy the following linker columns to each matched particle row:
      _rlnStickLength
      _rlnStickUnits
      _rlnLC_L_nm
      _rlnLC_ThetaDeg
      _rlnEdgeProbability
      _rlnLC_ChainComponent

Dependencies: pandas, starfile
"""

import pandas as pd
import starfile

# ==== File paths (edit as needed) ====
BIG_STAR    = "Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T1001010006_C3_man.star"
STICKS_STAR = "Linker_Sticks.star"
OUT_STAR    = "Nucleosome_by_LinkerSticks.star"

# ==== Helpers ====
def load_particles(path: str) -> pd.DataFrame:
    """Load the particles table from a STAR file (robust to common block names)."""
    data = starfile.read(path, always_dict=True)
    for k in ("data_particles", "particles", "data_"):
        if k in data:
            return data[k]
    return next(iter(data.values()))

def find_big_id_col(df_big: pd.DataFrame) -> str:
    """Detect the particle ID column in BIG_STAR."""
    if "rlnTomoParticleId" in df_big.columns:
        return "rlnTomoParticleId"
    cands = [c for c in df_big.columns if "TomoParticleId" in c]
    if len(cands) == 1:
        return cands[0]
    raise ValueError("Cannot find a valid ID column in BIG_STAR.")

# ==== Load data ====
df_big    = load_particles(BIG_STAR)
df_sticks = load_particles(STICKS_STAR)

ID1_COL, ID2_COL = "rlnStickSourceId1", "rlnStickSourceId2"
for c in (ID1_COL, ID2_COL):
    if c not in df_sticks.columns:
        raise ValueError(f"Missing column '{c}' in {STICKS_STAR}")

BIG_ID = find_big_id_col(df_big)

# Columns to copy from Linker to output
linker_cols = [
    "rlnStickLength",
    "rlnStickUnits",
    "rlnLC_L_nm",
    "rlnLC_ThetaDeg",
    "rlnEdgeProbability",
    "rlnLC_ChainComponent",
]
# Filter to keep only existing ones
linker_cols = [c for c in linker_cols if c in df_sticks.columns]

# Build map: ID -> list of BIG row indices (keep duplicates, preserve order)
id_to_rows = {}
for i, v in df_big[BIG_ID].items():
    key = str(v).strip()
    id_to_rows.setdefault(key, []).append(i)

# Collect rows following Linker order + attach linker info
records = []
missing = []

for _, stick in df_sticks.iterrows():
    stick_info = {col: stick[col] for col in linker_cols}
    for col in (ID1_COL, ID2_COL):
        pid = str(stick[col]).strip()
        rows = id_to_rows.get(pid)
        if rows:
            for idx in rows:
                rec = df_big.iloc[idx].to_dict()
                rec.update(stick_info)
                records.append(rec)
        else:
            missing.append(pid)

# Combine into DataFrame
df_out = pd.DataFrame(records, columns=list(df_big.columns) + linker_cols)

# Write output
starfile.write(df_out, OUT_STAR)

# ---- Summary ----
print(f"[INFO] BIG rows={len(df_big)}, ID_col={BIG_ID}")
print(f"[INFO] STICKS rows={len(df_sticks)} ({ID1_COL}, {ID2_COL})")
print(f"[INFO] OUTPUT rows={len(df_out)} → {OUT_STAR}")
if missing:
    print(f"[WARN] {len(set(missing))} stick IDs not found in BIG (first 20 shown): {list(dict.fromkeys(missing))[:20]}")
