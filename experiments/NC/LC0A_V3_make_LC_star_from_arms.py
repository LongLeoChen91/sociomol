#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build LC linker coordinates/angles from 4 reset-ARM STARs.

Inputs (same particle order or share an ID column):
  - Reset_arm1_0_subtomo_coords__tomo-*.star
  - Reset_arm1_1_subtomo_coords__tomo-*.star
  - Reset_arm2_0_subtomo_coords__tomo-*.star
  - Reset_arm2_1_subtomo_coords__tomo-*.star

Outputs:
  - LC_Linkers__tomo-*.star with:
    rlnLC_CoordinateX1/Y1/Z1, rlnLC_AngleRot1/Tilt1/Psi1
    rlnLC_CoordinateX2/Y2/Z2, rlnLC_AngleRot2/Tilt2/Psi2
    rlnLC_CoordinateX0/Y0/Z0, rlnLC_AngleRot0/Tilt0/Psi0
    (and it will keep rlnTomoParticleId if present)

Angle definitions (matches current code):
  - Arm1:   Euler from arm1_1 → arm1_0
  - Arm2:   Euler from arm2_1 → arm2_0
  - Center: Euler from midpoint(arm1_1, arm2_1) → midpoint(arm1_0, arm2_0)
"""

import sys, os
import numpy as np
import pandas as pd
import starfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# Angle/orientation calculation uses nucleosome_linker_prediction functions
from linker_prediction import (
    euler_zyz_from_two_points,
    midpoint_from_two_points,
)

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))



# You can also make these as command line arguments; here we hardcode for simplicity
# ===== Base directory (edit here only if path changes) =====
BASE_DIR = r"C:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\NC\arm_star_file"

# ===== STAR files =====
ARM1_0 = rf"{BASE_DIR}\Reset_arm1_0_IDandName_lamellae_014.star"
ARM1_1 = rf"{BASE_DIR}\Reset_arm1_1_IDandName_lamellae_014.star"
ARM2_0 = rf"{BASE_DIR}\Reset_arm2_0_IDandName_lamellae_014.star"
ARM2_1 = rf"{BASE_DIR}\Reset_arm2_1_IDandName_lamellae_014.star"


OUT_STAR = "H1_Linkers_IDandName_lamellae_014.star"

# --------- helpers ---------
COORD_COLS = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")

def read_star_first_block(path):
    obj = starfile.read(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    # Check common block names; otherwise return the first block
    for k in ("data_particles", "particles", "data_"):
        if k in obj:
            return obj[k]
    return next(iter(obj.values()))

def get_id_col(df):
    for c in ("rlnTomoParticleId", "_rlnTomoParticleId", "rlnParticleId", "_rlnParticleId"):
        if c in df.columns:
            return c
    return None  # If no ID column, align by row index

def ensure_cols(df, need=COORD_COLS, label=""):
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")

def to_xyz(df):
    return df.loc[:, COORD_COLS].to_numpy(dtype=float)

def add_cols(dst, prefix, coord, euler=None):
    dst[f"{prefix}CoordinateX"] = coord[:, 0]
    dst[f"{prefix}CoordinateY"] = coord[:, 1]
    dst[f"{prefix}CoordinateZ"] = coord[:, 2]
    if euler is not None:
        rot, tilt, psi = euler
        dst[f"{prefix}AngleRot"]  = rot
        dst[f"{prefix}AngleTilt"] = tilt
        dst[f"{prefix}AnglePsi"]  = psi

def euler_from_two_point_arrays(p0, p1):
    """Compute Euler angles row by row using euler_zyz_from_two_points; returns (rot, tilt, psi) arrays."""
    n = p0.shape[0]
    rot = np.empty(n, dtype=float)
    tilt = np.empty(n, dtype=float)
    psi = np.empty(n, dtype=float)
    for i in range(n):
        R, T, P = euler_zyz_from_two_points(tuple(p0[i]), tuple(p1[i]))
        rot[i], tilt[i], psi[i] = float(R), float(T), float(P)
    return rot, tilt, psi

# --------- main ---------
def main():
    df_a10 = read_star_first_block(ARM1_0)
    df_a11 = read_star_first_block(ARM1_1)
    df_a20 = read_star_first_block(ARM2_0)
    df_a21 = read_star_first_block(ARM2_1)

    for df, tag in [(df_a10, "arm1_0"), (df_a11, "arm1_1"), (df_a20, "arm2_0"), (df_a21, "arm2_1")]:
        ensure_cols(df, COORD_COLS, tag)

    # Align by rlnTomoParticleId if present; otherwise align by row index
    id_col = get_id_col(df_a10)
    if id_col and all(id_col in d.columns for d in (df_a11, df_a20, df_a21)):
        # Inner join on the ID column
        key = id_col
        a10 = df_a10[[key, *COORD_COLS]].rename(columns={c: f"{c}__a10" for c in COORD_COLS})
        a11 = df_a11[[key, *COORD_COLS]].rename(columns={c: f"{c}__a11" for c in COORD_COLS})
        a20 = df_a20[[key, *COORD_COLS]].rename(columns={c: f"{c}__a20" for c in COORD_COLS})
        a21 = df_a21[[key, *COORD_COLS]].rename(columns={c: f"{c}__a21" for c in COORD_COLS})

        m = a10.merge(a11, on=key, how="inner").merge(a20, on=key, how="inner").merge(a21, on=key, how="inner")
        if len(m) == 0:
            raise RuntimeError("No rows after merging by ID — check ID columns.")
    else:
        # Row alignment: require all 4 files to have same number of rows
        if not (len(df_a10) == len(df_a11) == len(df_a20) == len(df_a21)):
            raise RuntimeError("No ID column and row counts differ — cannot align by index.")
        m = pd.DataFrame(index=np.arange(len(df_a10)))
        for df, suf in [(df_a10, "a10"), (df_a11, "a11"), (df_a20, "a20"), (df_a21, "a21")]:
            for c in COORD_COLS:
                m[f"{c}__{suf}"] = df[c].to_numpy()
        # If available, keep one ID column as reference
        for c in ("rlnTomoParticleId", "_rlnTomoParticleId", "rlnParticleId", "_rlnParticleId"):
            if c in df_a10.columns:
                m[c] = df_a10[c].to_numpy()
                id_col = c
                break

    # Extract 4 sets of coordinates
    a10_xyz = m[[f"{c}__a10" for c in COORD_COLS]].to_numpy()
    a11_xyz = m[[f"{c}__a11" for c in COORD_COLS]].to_numpy()
    a20_xyz = m[[f"{c}__a20" for c in COORD_COLS]].to_numpy()
    a21_xyz = m[[f"{c}__a21" for c in COORD_COLS]].to_numpy()

    # 1) Arm1 coordinates = arm1_0; angles = arm1_1 → arm1_0
    coord1 = a10_xyz
    euler1 = euler_from_two_point_arrays(a11_xyz, a10_xyz)

    # 2) Arm2 coordinates = arm2_0; angles = arm2_1 → arm2_0
    coord2 = a20_xyz
    euler2 = euler_from_two_point_arrays(a21_xyz, a20_xyz)

    # 3) Center coordinates = midpoint(arm1_0, arm2_0);
    #    angles = midpoint(arm1_1, arm2_1) → midpoint(arm1_0, arm2_0)
    coord0 = np.empty_like(coord1)
    for i in range(len(coord0)):
        x0, y0, z0 = midpoint_from_two_points(tuple(coord1[i]), tuple(coord2[i]))
        coord0[i] = (x0, y0, z0)

    mid_11_21 = np.empty_like(coord1)
    for i in range(len(mid_11_21)):
        X01, Y01, Z01 = midpoint_from_two_points(tuple(a11_xyz[i]), tuple(a21_xyz[i]))
        mid_11_21[i] = (X01, Y01, Z01)

    euler0 = euler_from_two_point_arrays(mid_11_21, coord0)

    # ---------- assemble output: keep all non-coordinate columns from ARM1_0 ----------
    base_cols = [c for c in df_a10.columns if c not in COORD_COLS]

    if id_col and id_col in df_a10.columns and id_col in m.columns:
        # Build base table from ARM1_0, then align its row order to m via ID
        base_keep = df_a10[base_cols].copy()
        if id_col not in base_keep.columns:
            base_keep[id_col] = df_a10[id_col].to_numpy()
        other_cols = [c for c in base_keep.columns if c != id_col]
        base_keep = base_keep[[id_col] + other_cols]
        out = m[[id_col]].merge(base_keep, on=id_col, how="left")
    else:
        # No ID alignment: assume same order/length already checked above
        out = df_a10[base_cols].reset_index(drop=True)

    # ---------- append LC columns ----------
    add_cols(out, "rlnLC_", coord1, euler1)
    out.rename(columns={
        "rlnLC_CoordinateX": "rlnLC_CoordinateX1",
        "rlnLC_CoordinateY": "rlnLC_CoordinateY1",
        "rlnLC_CoordinateZ": "rlnLC_CoordinateZ1",
        "rlnLC_AngleRot":    "rlnLC_AngleRot1",
        "rlnLC_AngleTilt":   "rlnLC_AngleTilt1",
        "rlnLC_AnglePsi":    "rlnLC_AnglePsi1",
    }, inplace=True)

    add_cols(out, "rlnLC_", coord2, euler2)
    out.rename(columns={
        "rlnLC_CoordinateX": "rlnLC_CoordinateX2",
        "rlnLC_CoordinateY": "rlnLC_CoordinateY2",
        "rlnLC_CoordinateZ": "rlnLC_CoordinateZ2",
        "rlnLC_AngleRot":    "rlnLC_AngleRot2",
        "rlnLC_AngleTilt":   "rlnLC_AngleTilt2",
        "rlnLC_AnglePsi":    "rlnLC_AnglePsi2",
    }, inplace=True)

    add_cols(out, "rlnLC_", coord0, euler0)
    out.rename(columns={
        "rlnLC_CoordinateX": "rlnLC_CoordinateX0",
        "rlnLC_CoordinateY": "rlnLC_CoordinateY0",
        "rlnLC_CoordinateZ": "rlnLC_CoordinateZ0",
        "rlnLC_AngleRot":    "rlnLC_AngleRot0",
        "rlnLC_AngleTilt":   "rlnLC_AngleTilt0",
        "rlnLC_AnglePsi":    "rlnLC_AnglePsi0",
    }, inplace=True)

    # Write STAR
    starfile.write({"particles": out}, OUT_STAR, overwrite=True)
    print(f"[OK] wrote: {OUT_STAR}  n={len(out)}")

if __name__ == "__main__":
    main()
