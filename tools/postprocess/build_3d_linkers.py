#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sticks.py — Build 3D linker-stick particles from a predicted edge list.

Reads the annotated STAR file (output of ``sociomol predict``) and the
corresponding edges CSV, then creates a new STAR file where each row
represents a "stick" connecting two particle arms.  Each stick carries
midpoint coordinates, Euler angles aligned to the connection axis,
geometric length, and metadata copied from the edge list.

The output STAR file can be loaded directly in ChimeraX / ArtiaX for
3D visualisation of predicted linker connections.

Requires: numpy, pandas, starfile, linker_prediction (``pip install .``)
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
import starfile

try:
    from linker_prediction import euler_zyz_from_two_points, midpoint_from_two_points
except ImportError:
    sys.exit(
        "Error: the 'linker_prediction' package is not installed.\n"
        "Install SocioMol first:  pip install ."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *candidates: str) -> str:
    """Return the first matching column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"Missing required column. Tried: {candidates}")


def _norm_arm(val) -> str:
    """Map 0 → 'arm1', 1 → 'arm2'."""
    try:
        n = int(float(val))
    except Exception:
        raise ValueError(f"Invalid arm label '{val}' (expected 0 or 1).")
    if n == 0:
        return "arm1"
    elif n == 1:
        return "arm2"
    else:
        raise ValueError(f"Invalid arm label '{val}' (expected 0 or 1).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build 3D linker-stick particles from predicted edges.",
    )
    parser.add_argument("--particles", required=True,
                        help="Input annotated STAR file (from sociomol predict).")
    parser.add_argument("--edges", required=True,
                        help="Input edges CSV file (from sociomol predict).")
    parser.add_argument("--output", required=True,
                        help="Output STAR file for linker sticks.")
    parser.add_argument("--pixel-size", type=float, required=True,
                        help="Pixel size in Angstroms per pixel.")
    parser.add_argument("--p-threshold", type=float, default=0.0,
                        help="Minimum probability to keep an edge (default: 0.0 = keep all).")
    args = parser.parse_args()

    pixel_size_a = args.pixel_size
    scale_to_nm = pixel_size_a / 10.0

    # ---- Load particles ----
    star = starfile.read(args.particles)
    df = next(iter(star.values())) if isinstance(star, dict) else star

    col_id = "rlnTomoParticleId"
    x1, y1, z1 = "rlnLC_CoordinateX1", "rlnLC_CoordinateY1", "rlnLC_CoordinateZ1"
    x2, y2, z2 = "rlnLC_CoordinateX2", "rlnLC_CoordinateY2", "rlnLC_CoordinateZ2"
    missing = [c for c in [col_id, x1, y1, z1, x2, y2, z2] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required STAR columns: {missing}")

    pid_arr = df[col_id].astype(int).values
    idx_to_id = {int(i): int(pid) for i, pid in enumerate(pid_arr)}

    # Vectorised endpoint lookup
    id_to_endpoints = {
        int(pid): {
            "arm1": np.array([px1, py1, pz1], dtype=float),
            "arm2": np.array([px2, py2, pz2], dtype=float),
        }
        for pid, px1, py1, pz1, px2, py2, pz2 in zip(
            pid_arr,
            df[x1].values, df[y1].values, df[z1].values,
            df[x2].values, df[y2].values, df[z2].values,
        )
    }

    # Chain-component mapping (if available)
    if "rlnLC_ChainComponent" in df.columns:
        comp_map = dict(zip(df[col_id], df["rlnLC_ChainComponent"]))
    else:
        print("[WARN] Column 'rlnLC_ChainComponent' not found; will write NaN.")
        comp_map = {}

    # ---- Load edges ----
    edges = pd.read_csv(args.edges, dtype={"arm_i": "string", "arm_j": "string"})

    if args.p_threshold > 0:
        if "P" in edges.columns:
            edges = edges[edges["P"] > args.p_threshold]
            print(f"Applied P > {args.p_threshold} filter, remaining edges: {len(edges)}")
        else:
            print("[WARN] No column 'P' found in edges; skipping filter.")
    else:
        print("No P filtering applied.")

    # Detect ID columns
    has_id = all(c.lower() in {x.lower() for x in edges.columns} for c in ["i_id", "j_id"])
    has_idx = all(c.lower() in {x.lower() for x in edges.columns} for c in ["i_idx", "j_idx"])
    if not (has_id or has_idx):
        raise ValueError("edges CSV must contain (i_id, j_id) or (i_idx, j_idx).")

    col_arm_i = _find_col(edges, "arm_i")
    col_arm_j = _find_col(edges, "arm_j")
    col_i = _find_col(edges, "i_id") if has_id else _find_col(edges, "i_idx")
    col_j = _find_col(edges, "j_id") if has_id else _find_col(edges, "j_idx")

    # Optional metric columns
    col_p = col_theta = col_L = None
    try:
        col_p = _find_col(edges, "P", "prob", "probability")
    except KeyError:
        pass
    try:
        col_theta = _find_col(edges, "theta_deg", "theta")
    except KeyError:
        pass
    try:
        col_L = _find_col(edges, "L_nm", "L")
    except KeyError:
        pass

    # ---- Build sticks ----
    sticks = []
    for _, erow in edges.iterrows():
        arm_i = _norm_arm(erow[col_arm_i])
        arm_j = _norm_arm(erow[col_arm_j])

        if has_id:
            i_id, j_id = int(erow[col_i]), int(erow[col_j])
        else:
            i_id = idx_to_id.get(int(erow[col_i]))
            j_id = idx_to_id.get(int(erow[col_j]))

        if i_id not in id_to_endpoints or j_id not in id_to_endpoints:
            continue

        p_i = id_to_endpoints[i_id][arm_i] * scale_to_nm
        p_j = id_to_endpoints[j_id][arm_j] * scale_to_nm

        mid_nm = midpoint_from_two_points(p_i, p_j)
        mid_px = mid_nm * 10 / pixel_size_a
        dist = float(np.linalg.norm(p_j - p_i))
        # Compute Euler angles directly from direction vector.
        # Bypasses eulerangles.matrix2euler which has a decomposition bug
        # for certain orientations (e.g. near-axis-aligned directions).
        # For ZYZ intrinsic R = Rz(rot) @ Ry(tilt) @ Rz(psi):
        #   R[2,:] = [-sin(tilt)*cos(psi), sin(tilt)*sin(psi), cos(tilt)]
        # Setting rot=0 (free for symmetric cylinders), solve for tilt & psi.
        _v = p_j - p_i
        _d = float(np.linalg.norm(_v))
        if _d < 1e-12:
            rot, tilt, psi = 0.0, 0.0, 0.0
        else:
            _vh = _v / _d
            tilt = math.degrees(math.acos(float(np.clip(_vh[2], -1.0, 1.0))))
            _st = math.sin(math.radians(tilt))
            if _st > 1e-12:
                psi = math.degrees(math.atan2(float(_vh[1]), float(-_vh[0])))
            else:
                psi = 0.0
            rot = 0.0

        p_val = float(erow[col_p]) if col_p and pd.notnull(erow[col_p]) else np.nan
        theta_val = float(erow[col_theta]) if col_theta and pd.notnull(erow[col_theta]) else np.nan
        L_val = float(erow[col_L]) if col_L and pd.notnull(erow[col_L]) else np.nan
        comp_val = comp_map.get(i_id, np.nan) if comp_map else np.nan

        sticks.append({
            "rlnCoordinateX": mid_px[0],
            "rlnCoordinateY": mid_px[1],
            "rlnCoordinateZ": mid_px[2],
            "rlnAngleRot": rot,
            "rlnAngleTilt": tilt,
            "rlnAnglePsi": psi,
            "rlnStickSourceId1": int(i_id),
            "rlnStickSourceId2": int(j_id),
            "rlnStickEndpoint1": arm_i,
            "rlnStickEndpoint2": arm_j,
            "rlnStickLength": dist,
            "rlnStickUnits": "nm",
            "rlnLC_L_nm": L_val,
            "rlnLC_ThetaDeg": theta_val,
            "rlnEdgeProbability": p_val,
            "rlnLC_ChainComponent": comp_val,
        })

    sticks_df = pd.DataFrame(sticks)
    starfile.write({"particles": sticks_df}, args.output, overwrite=True)
    print(f"[OK] Wrote {len(sticks_df)} sticks -> {args.output}")


if __name__ == "__main__":
    main()
