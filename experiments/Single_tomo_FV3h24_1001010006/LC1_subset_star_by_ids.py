# subset_star_by_ids_simple.py
# Subset H1_DoubleLinker_deduplicated_subtomo_coords_C2cc5.star
# by rlnTomoParticleId from Nucleosome_coords_from_Best_path_particles.star

import pandas as pd
import starfile

# ==== File paths (edit as needed) ====
BIG_STAR = "H1_DoubleLinker_deduplicated_subtomo_coords_T1001010006_C3_man.star"
IDS_STAR = "Nucleosome_coords_from_clustered_N_Reset_Z48_deduplicated_subtomo_coords_T1001010006_C3_man.star"
OUT_STAR = "subset_H1_DoubleLinker_T1001010006.star"

def load_particles(path):
    """Load particles table from STAR (robust to different keys)."""
    data = starfile.read(path, always_dict=True)
    for k in ("data_particles", "particles", "data_"):
        if k in data:
            return data[k]
    return next(iter(data.values()))

def norm_ids(series):
    """Convert IDs to integer-like strings (e.g. 1046.0 -> '1046')."""
    def conv(x):
        try:
            return str(int(float(str(x).strip())))
        except Exception:
            return str(x).strip()
    return series.astype(str).map(conv)

# Load both STARs
df_big = load_particles(BIG_STAR)
df_ids = load_particles(IDS_STAR)

# Find ID columns
col_big = [c for c in df_big.columns if "TomoParticleId" in c][0]
col_ids = [c for c in df_ids.columns if "TomoParticleId" in c][0]

# Normalize IDs
big_ids = norm_ids(df_big[col_big])
ids_keep = set(norm_ids(df_ids[col_ids]))

# Filter
df_out = df_big[big_ids.isin(ids_keep)].copy()

# Write out
starfile.write({"data_particles": df_out}, OUT_STAR, overwrite=True)

print(f"Subset saved to {OUT_STAR}, rows kept = {len(df_out)}")
