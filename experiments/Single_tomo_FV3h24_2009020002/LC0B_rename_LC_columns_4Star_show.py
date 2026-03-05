import pandas as pd
import starfile

# ====== paths ======
INPUT_STAR  = "H1_Linkers__tomo-HIV_Nuc_Lena_grid_1072.star"
OUT_ARM1    = "arm1_as_rln.star"
OUT_ARM2    = "arm2_as_rln.star"
OUT_ARM0    = "arm0_as_rln.star"

# ====== ID column preference (optional) ======
ID_COL_EXPLICIT = "_rlnTomoParticleId"   # or None

ID_CANDIDATES = [
    "_rlnTomoParticleId", "rlnTomoParticleId",
    "_rlnParticleId", "rlnParticleId",
    "_rlnOriginalParticleId", "rlnOriginalParticleId",
]

TARGETS = [
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
]

COMMON_BLOCKS = ("data_particles","particles","data_")

def read_first_block(path):
    obj = starfile.read(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    for k in COMMON_BLOCKS:
        if k in obj:
            return obj[k]
    return next(iter(obj.values()))

def pick_id_column(df: pd.DataFrame):
    if ID_COL_EXPLICIT and ID_COL_EXPLICIT in df.columns:
        return ID_COL_EXPLICIT
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower().endswith("id"):
            return c
    return None

def first_existing(df: pd.DataFrame, candidates):
    """Return the first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def resolve_src_cols(df: pd.DataFrame, source: str):
    """
    Accept both naming variants:
      with underscore:  _rlnLC_CoordinateX{S}, ...
      without:          rlnLC_CoordinateX{S}, ...
    Return the actual column names present in df (6 of them).
    """
    base_pairs = [
        (f"_rlnLC_CoordinateX{source}", f"rlnLC_CoordinateX{source}"),
        (f"_rlnLC_CoordinateY{source}", f"rlnLC_CoordinateY{source}"),
        (f"_rlnLC_CoordinateZ{source}", f"rlnLC_CoordinateZ{source}"),
        (f"_rlnLC_AngleRot{source}",     f"rlnLC_AngleRot{source}"),
        (f"_rlnLC_AngleTilt{source}",    f"rlnLC_AngleTilt{source}"),
        (f"_rlnLC_AnglePsi{source}",     f"rlnLC_AnglePsi{source}"),
    ]
    chosen = []
    missing = []
    for a, b in base_pairs:
        found = first_existing(df, (a, b))
        if found is None:
            missing.append((a, b))
        else:
            chosen.append(found)
    if missing:
        miss_str = ", ".join([f"{a} | {b}" for a,b in missing])
        raise KeyError(f"[LC{source}] Missing columns (checked both variants): {miss_str}")
    return chosen  # length 6 in order: X,Y,Z,Rot,Tilt,Psi

def remap_and_reorder(df_in: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df_in.copy()
    src_cols = resolve_src_cols(df, source)       # 6 columns from LC{source}
    # copy values into TARGETS
    for s, t in zip(src_cols, TARGETS):
        df[t] = df[s]

    id_col = pick_id_column(df)
    targets_present = [c for c in TARGETS if c in df.columns]
    cols_all = list(df.columns)

    if id_col:
        first = [id_col] + targets_present
    else:
        first = targets_present

    rest = [c for c in cols_all if c not in first]
    return df.loc[:, first + rest]

def main():
    df = read_first_block(INPUT_STAR)

    out1 = remap_and_reorder(df, "1")
    starfile.write({"particles": out1}, OUT_ARM1, overwrite=True)
    print(f"[OK] wrote {OUT_ARM1}")

    out2 = remap_and_reorder(df, "2")
    starfile.write({"particles": out2}, OUT_ARM2, overwrite=True)
    print(f"[OK] wrote {OUT_ARM2}")

    out0 = remap_and_reorder(df, "0")
    starfile.write({"particles": out0}, OUT_ARM0, overwrite=True)
    print(f"[OK] wrote {OUT_ARM0}")

if __name__ == "__main__":
    main()
