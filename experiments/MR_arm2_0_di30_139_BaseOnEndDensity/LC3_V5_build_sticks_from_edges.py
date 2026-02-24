import numpy as np
import pandas as pd
import starfile

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nucleosome_linker_prediction import (
    euler_zyz_from_two_points,
    midpoint_from_two_points,
)

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- config ----------
PARTICLES_STAR = "H1_DoubleLinker_annotated.star"
EDGES_CSV     = "DoubleLinker_edges.csv"
OUTPUT_STAR   = "Linker_Sticks.star"

# Conversion factor: 1 unit = 1.513 Å = 0.1513 nm
PIXEL_SIZE_A  = 8               # Å/pixel
SCALE_TO_NM = PIXEL_SIZE_A / 10.0 # nm/pixel

# Define P threshold (set to 0.0 to disable filtering)
P_THRESHOLD = 0.0

# ---------- helpers ----------
def find_col(df: pd.DataFrame, *cands: str) -> str:
    """Return the first matching column name in df among candidates (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in cands:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(f"Missing required column. Tried: {cands}")

def norm_arm(val) -> str:
    """
    Map 0 -> 'arm1', 1 -> 'arm2'.
    Accepts 0/1 as int, float, or string ('0','1','0.0','1.0').
    """
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

def main():
    # Read STAR (either dict-of-blocks or single DF)
    star = starfile.read(PARTICLES_STAR)
    df = next(iter(star.values())) if isinstance(star, dict) else star

    # Required STAR columns (assumes starfile normalized labels to rln*)
    col_id = "rlnTomoParticleId"
    x1, y1, z1 = "rlnLC_CoordinateX1", "rlnLC_CoordinateY1", "rlnLC_CoordinateZ1"
    x2, y2, z2 = "rlnLC_CoordinateX2", "rlnLC_CoordinateY2", "rlnLC_CoordinateZ2"
    missing = [c for c in [col_id, x1, y1, z1, x2, y2, z2] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required STAR columns: {missing}")

    # Row index (0-based) -> particle ID (for i_idx/j_idx)
    id_series = df[col_id].astype(int).reset_index(drop=True)
    idx_to_id = {int(i): int(pid) for i, pid in enumerate(id_series.values)}

    # Build particleID -> endpoints dict
    id_to_endpoints = {}
    for _, row in df.iterrows():
        pid = int(row[col_id])
        arm1 = np.array([row[x1], row[y1], row[z1]], dtype=float)
        arm2 = np.array([row[x2], row[y2], row[z2]], dtype=float)
        id_to_endpoints[pid] = {"arm1": arm1, "arm2": arm2}

    # --- NEW: build particleID -> chain component mapping ---
    # We read the component label per nucleosome from the annotated STAR.
    # Later, each linker stick (edge) will inherit the component of its source nucleosome.
    if "rlnLC_ChainComponent" in df.columns:
        comp_map = dict(zip(df[col_id], df["rlnLC_ChainComponent"]))
    else:
        print("[WARN] Column 'rlnLC_ChainComponent' not found in STAR; will write NaN.")
        comp_map = {}

    # Read edges.csv and detect schema
    edges = pd.read_csv(EDGES_CSV, dtype={"arm_i": "string", "arm_j": "string"})

     # Apply P threshold filter if requested
    if P_THRESHOLD > 0:
        if "P" in edges.columns:
            edges = edges[edges["P"] > P_THRESHOLD]
            print(f"Applied P > {P_THRESHOLD} filter, remaining edges: {len(edges)}")
        else:
            print("[WARN] No column 'P' found in edges; skipping filter.")
    else:
        print("No P filtering applied.")
           
    print("Unique arm_i:", sorted(edges["arm_i"].dropna().astype(str).unique().tolist()))
    print("Unique arm_j:", sorted(edges["arm_j"].dropna().astype(str).unique().tolist()))

    # Columns (case-insensitive)
    has_id  = all(c.lower() in {x.lower() for x in edges.columns} for c in ["i_id", "j_id"])
    has_idx = all(c.lower() in {x.lower() for x in edges.columns} for c in ["i_idx", "j_idx"])
    if not (has_id or has_idx):
        raise ValueError("edges.csv must contain either (i_id, j_id, arm_i, arm_j) or (i_idx, j_idx, arm_i, arm_j).")

    col_arm_i = find_col(edges, "arm_i")
    col_arm_j = find_col(edges, "arm_j")
    col_i = find_col(edges, "i_id")  if has_id  else find_col(edges, "i_idx")
    col_j = find_col(edges, "j_id")  if has_id  else find_col(edges, "j_idx")

    # NEW: find probability column (tolerate common aliases)
    col_p = None
    try:
        col_p = find_col(edges, "P", "prob", "probability", "edge_prob", "linker_prob")
    except KeyError:
        print("[WARN] No probability column found in CSV; will write NaN for probabilities.")

    # NEW: find theta and linker length columns from edges.csv
    col_theta_deg = None
    col_L = None
    try:
        col_theta_deg = find_col(edges, "theta_deg", "theta", "Theta_deg", "Theta")
    except KeyError:
        print("[WARN] No 'theta_deg' column found in edges; will write NaN.")
    try:
        col_L = find_col(edges, "L_nm", "L", "linker_length_nm", "LinkerLength_nm")
    except KeyError:
        print("[WARN] No 'L_nm' column found in edges; will write NaN.")

    # Build stick particles
    sticks = []
    for _, erow in edges.iterrows():
        arm_i = norm_arm(erow[col_arm_i])
        arm_j = norm_arm(erow[col_arm_j])

        if has_id:
            i_id = int(erow[col_i]); j_id = int(erow[col_j])
        else:
            i_idx = int(erow[col_i]); j_idx = int(erow[col_j])
            i_id = idx_to_id.get(i_idx); j_id = idx_to_id.get(j_idx)

        # Validate IDs
        if i_id not in id_to_endpoints or j_id not in id_to_endpoints:
            # skip edges that don't map to STAR particles
            continue

        # Fetch endpoints strictly as specified by CSV, convert to nm
        p_i = id_to_endpoints[i_id][arm_i] * SCALE_TO_NM
        p_j = id_to_endpoints[j_id][arm_j] * SCALE_TO_NM

        # Midpoint, length, Euler (Z aligned to p_j - p_i)
        mid_nm  = midpoint_from_two_points(p_i, p_j)
        mid = mid_nm*10/PIXEL_SIZE_A # back to original pixel units for STAR
        
        dist = float(np.linalg.norm(p_j - p_i))   # nm
        rot, tilt, psi = euler_zyz_from_two_points(p_i, p_j)

        # Get probability value if present
        p_val = float(erow[col_p]) if col_p is not None and pd.notnull(erow[col_p]) else np.nan

        # NEW: pull theta_deg and L_nm from edges.csv (if present)
        theta_deg_val = float(erow[col_theta_deg]) if col_theta_deg is not None and pd.notnull(erow[col_theta_deg]) else np.nan
        L_csv_val = float(erow[col_L]) if col_L is not None and pd.notnull(erow[col_L]) else np.nan        

        # --- NEW: get chain component for this edge ---
        # i_id and j_id should share the same component; take from i_id.
        comp_val = np.nan
        if comp_map:
            comp_val = comp_map.get(i_id, np.nan)

        sticks.append({
            "rlnCoordinateX": mid[0],
            "rlnCoordinateY": mid[1],
            "rlnCoordinateZ": mid[2],
            "rlnAngleRot": rot,
            "rlnAngleTilt": tilt,
            "rlnAnglePsi": psi,
            "rlnStickSourceId1": int(i_id),
            "rlnStickSourceId2": int(j_id),
            "rlnStickEndpoint1": arm_i,     # as specified by CSV
            "rlnStickEndpoint2": arm_j,     # as specified by CSV
            "rlnStickLength": dist,         # computed geometric length (nm)
            "rlnStickUnits": "nm",
            # NEW: values copied from edges.csv
            "rlnLC_L_nm": L_csv_val,      # L_nm from CSV
            "rlnLC_ThetaDeg": theta_deg_val,  # theta_deg from CSV
            "rlnEdgeProbability": p_val, # probability
            "rlnLC_ChainComponent": comp_val,  # NEW: component label
        })

    sticks_df = pd.DataFrame(sticks)
    starfile.write({"particles": sticks_df}, OUTPUT_STAR, overwrite=True)
    print(f"[OK] Wrote sticks: {OUTPUT_STAR}, n={len(sticks_df)}")

if __name__ == "__main__":
    main()
