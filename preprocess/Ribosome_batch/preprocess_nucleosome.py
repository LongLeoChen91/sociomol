#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nucleosome Preprocessing Script (Declarative Version)
Replaces the old 4-step bash/jupyter process with a single clean pass.
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import starfile

# Attempt to import functions from linker_prediction
try:
    # Assuming running from repository root or it's in PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from linker_prediction import euler_zyz_from_two_points, midpoint_from_two_points
except ImportError:
    print("Warning: Could not import linker_prediction. Please run from repository root.")
    sys.exit(1)

# ==========================================
# Default declarative geometry config 
# (Equivalent to a future nucleosome_model.json)
# ==========================================
DEFAULT_GEOMETRY_CONFIG = {
    # 1. Base reference points (local coordinates relative to nucleosome center)
    "points": {
        "arm1_inner": [ -31.36,-29.40,-5.88],  # Originally arm1_0
        "arm1_outer": [ -25.48,-37.24,7.84],  # Originally arm1_1
        "arm2_inner": [ -25.48,-37.24,7.84],  # Originally arm2_0
        "arm2_outer": [ -31.36,-29.40,-5.88]   # Originally arm2_1
    },
    
    # 2. Virtual computational points (e.g., midpoints)
    "virtual_points": {
        "center_inner": {"type": "midpoint", "p1": "arm1_inner", "p2": "arm2_inner"},
        "center_outer": {"type": "midpoint", "p1": "arm1_outer", "p2": "arm2_outer"}
    },
    
    # 3. Output features definition
    "features": {
        "Arm1": {
            "output_suffix": "1",               
            "coordinate_anchor": "arm1_inner",  
            "direction_vector": ["arm1_outer", "arm1_inner"] 
        },
        "Arm2": {
            "output_suffix": "2",               
            "coordinate_anchor": "arm2_inner",
            "direction_vector": ["arm2_outer", "arm2_inner"] 
        },
        "Center": {
            "output_suffix": "0",               
            "coordinate_anchor": "center_inner", 
            "direction_vector": ["center_outer", "center_inner"]
        }
    }
}

# ==========================================
# Core computation functions
# ==========================================

def get_relion_local_to_global_matrix(rot_deg, tilt_deg, psi_deg):
    """
    Builds a rotation matrix from local vector to global space based on RELION Euler angles.
    (Equivalent to the inverse of the original relion_center rotation matrix)
    """
    rot, tilt, psi = np.deg2rad([rot_deg, tilt_deg, psi_deg])
    
    c_rot, s_rot = np.cos(rot), np.sin(rot)
    c_tilt, s_tilt = np.cos(tilt), np.sin(tilt)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    r11 = c_rot * c_tilt * c_psi - s_rot * s_psi
    r12 = -c_rot * c_tilt * s_psi - s_rot * c_psi
    r13 = c_rot * s_tilt

    r21 = s_rot * c_tilt * c_psi + c_rot * s_psi
    r22 = -s_rot * c_tilt * s_psi + c_rot * c_psi
    r23 = s_rot * s_tilt

    r31 = -s_tilt * c_psi
    r32 = s_tilt * s_psi
    r33 = c_tilt

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

def process_particles(df, config, pixel_size):
    """
    Process geometric features for each particle in the DataFrame
    """
    # Check for required columns
    required_cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
                     "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column in input STAR: {col}")
            
    out_df = df.copy()
    n_particles = len(df)
    
    # Prepare empty columns for the features
    for feature_name, feature_def in config["features"].items():
        suf = feature_def["output_suffix"]
        for prefix in ["CoordinateX", "CoordinateY", "CoordinateZ", "AngleRot", "AngleTilt", "AnglePsi"]:
            out_df[f"rlnLC_{prefix}{suf}"] = 0.0

    print(f"Processing {n_particles} particles...")
    
    # Pre-load local points into NumPy arrays
    local_points = {name: np.array(coord, dtype=float) for name, coord in config["points"].items()}
    
    # Process particles row by row
    for i in range(n_particles):
        row = df.iloc[i]
        
        # 1. Get current particle's global center coordinates and rotation matrix
        global_center = np.array([row["rlnCoordinateX"], row["rlnCoordinateY"], row["rlnCoordinateZ"]])
        rot_mat = get_relion_local_to_global_matrix(row["rlnAngleRot"], row["rlnAngleTilt"], row["rlnAnglePsi"])
        
        # 2. Dynamically calculate the global coordinates of base points
        # Formula: global_coord = center_coord + (R_mat @ local_vec) / pixel_size
        current_points = {}
        for p_name, local_vec in local_points.items():
            global_vec = rot_mat.T @ local_vec
            current_points[p_name] = global_center + (global_vec / pixel_size)
            
        # 3. Calculate virtual midpoints
        if "virtual_points" in config:
            for v_name, v_def in config["virtual_points"].items():
                if v_def["type"] == "midpoint":
                    p1 = current_points[v_def["p1"]]
                    p2 = current_points[v_def["p2"]]
                    current_points[v_name] = np.array(midpoint_from_two_points(tuple(p1), tuple(p2)))
                    
        # 4. Extract coordinates and calculate Euler angles based on Features definition
        for feature_name, feature_def in config["features"].items():
            suf = feature_def["output_suffix"]
            anchor = current_points[feature_def["coordinate_anchor"]]
            p_from = current_points[feature_def["direction_vector"][0]]
            p_to   = current_points[feature_def["direction_vector"][1]]
            
            # Use original functions to calculate Euler angles
            r, t, p = euler_zyz_from_two_points(tuple(p_from), tuple(p_to))
            
            # Write back to dataframe
            out_df.at[i, f"rlnLC_CoordinateX{suf}"] = anchor[0]
            out_df.at[i, f"rlnLC_CoordinateY{suf}"] = anchor[1]
            out_df.at[i, f"rlnLC_CoordinateZ{suf}"] = anchor[2]
            out_df.at[i, f"rlnLC_AngleRot{suf}"]  = r
            out_df.at[i, f"rlnLC_AngleTilt{suf}"] = t
            out_df.at[i, f"rlnLC_AnglePsi{suf}"]  = p

    return out_df

def main():
    parser = argparse.ArgumentParser(description="One-step Nucleosome Preprocessing")
    parser.add_argument("--input", required=True, help="Input STAR file (e.g., R3_ID_Manual_1.star)")
    parser.add_argument("--output", required=True, help="Output STAR file (e.g., H1_DoubleLinker.star)")
    parser.add_argument("--pixel_size", type=float, default=8.0, help="Pixel size in Angstroms (default: 8.0)")
    parser.add_argument("--model_json", type=str, default=None, help="Optional geometry definition JSON")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.model_json and os.path.exists(args.model_json):
        with open(args.model_json, "r") as f:
            config = json.load(f)
        print(f"Loaded geometry from {args.model_json}")
    else:
        config = DEFAULT_GEOMETRY_CONFIG
        print("Using default internal geometry configuration.")

    # Extract and process Star file
    print(f"Reading {args.input}...")
    star_data = starfile.read(args.input)
    
    # Handle single Dataframe or Dict structure
    if isinstance(star_data, pd.DataFrame):
        block_name = "particles"
        df = star_data
    else:
        # Search for a block containing coordinates
        for k, v in star_data.items():
            if isinstance(v, pd.DataFrame) and "rlnCoordinateX" in v.columns:
                block_name = k
                df = v
                break
                
    processed_df = process_particles(df, config, args.pixel_size)
    
    # Write results
    if isinstance(star_data, pd.DataFrame):
        output_data = {block_name: processed_df}
    else:
        output_data = star_data.copy()
        output_data[block_name] = processed_df
        
    starfile.write(output_data, args.output, overwrite=True)
    print(f"Success! Output written to {args.output}")

if __name__ == "__main__":
    main()
