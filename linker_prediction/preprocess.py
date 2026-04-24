#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized preprocessing for SocioMol.

Converts a standard RELION STAR file (with center coordinates and Euler angles)
into an arm-annotated STAR file with ``rlnLC_*`` columns, using a particle
geometry model provided as a JSON configuration.

Public API
----------
load_geometry(model_name_or_path)
    Load a geometry config by built-in name or JSON file path.

list_models()
    List available built-in geometry model names.

preprocess_star(input_star, output_star, geometry, pixel_size)
    One-pass preprocessing: raw RELION STAR -> arm-annotated STAR.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import starfile

from .linker_geometry import euler_zyz_from_two_points

# ---------------------------------------------------------------------------
# Geometry model loading
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).parent / "arm_geometry"

_VALID_TANGENT_DIRECTIONS = frozenset({
    "direction_point_to_anchor",
    "anchor_to_direction_point",
})


def list_models() -> List[str]:
    """Return names of all built-in geometry models."""
    return sorted(f.stem for f in _MODELS_DIR.glob("*.json"))


def load_geometry(model_name_or_path: str) -> dict:
    """Load a geometry configuration by built-in name or file path.

    Parameters
    ----------
    model_name_or_path : str
        Either a built-in model name (e.g. ``"nucleosome_modelA_8A"``)
        or a path to a JSON file (must end in ``.json``).

    Returns
    -------
    dict
        Validated geometry configuration.

    Raises
    ------
    ValueError
        If the model is not found or the config is invalid.
    """
    p = Path(model_name_or_path)

    # Try as file path first
    if p.suffix == ".json":
        if not p.exists():
            raise FileNotFoundError(f"Geometry file not found: {p}")
        with open(p, "r") as f:
            geometry = json.load(f)
    else:
        # Try as built-in model name
        builtin = _MODELS_DIR / f"{model_name_or_path}.json"
        if not builtin.exists():
            available = list_models()
            raise ValueError(
                f"Unknown geometry model: '{model_name_or_path}'. "
                f"Available built-in models: {available}"
            )
        with open(builtin, "r") as f:
            geometry = json.load(f)

    _validate_geometry(geometry)
    return geometry


def _validate_geometry(geometry: dict) -> None:
    """Validate geometry config structure."""
    if "arms" not in geometry:
        raise ValueError("Geometry config must contain an 'arms' key.")

    arms = geometry["arms"]
    if not isinstance(arms, list) or len(arms) < 2:
        raise ValueError("Geometry config must define at least 2 arms.")

    for i, arm in enumerate(arms):
        for key in ("anchor", "direction_point", "tangent"):
            if key not in arm:
                raise ValueError(f"Arm {i} missing required key: '{key}'")

        if arm["tangent"] not in _VALID_TANGENT_DIRECTIONS:
            raise ValueError(
                f"Arm {i} tangent must be one of {sorted(_VALID_TANGENT_DIRECTIONS)}, "
                f"got '{arm['tangent']}'"
            )

        for key in ("anchor", "direction_point"):
            val = arm[key]
            if not isinstance(val, (list, tuple)) or len(val) != 3:
                raise ValueError(f"Arm {i} '{key}' must be a list of 3 floats")


# ---------------------------------------------------------------------------
# RELION rotation matrix
# ---------------------------------------------------------------------------

def _relion_rotation_matrix(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    """Build the RELION ZYZ rotation matrix from Euler angles (degrees).

    This is the standard RELION convention. The transpose of the returned
    matrix transforms vectors from the local (particle) frame to the global
    (tomogram) frame.
    """
    rot, tilt, psi = np.deg2rad([rot_deg, tilt_deg, psi_deg])

    c_rot, s_rot = np.cos(rot), np.sin(rot)
    c_tilt, s_tilt = np.cos(tilt), np.sin(tilt)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    return np.array([
        [c_rot * c_tilt * c_psi - s_rot * s_psi,
         -c_rot * c_tilt * s_psi - s_rot * c_psi,
         c_rot * s_tilt],
        [s_rot * c_tilt * c_psi + c_rot * s_psi,
         -s_rot * c_tilt * s_psi + c_rot * c_psi,
         s_rot * s_tilt],
        [-s_tilt * c_psi,
         s_tilt * s_psi,
         c_tilt],
    ])


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def _process_particles(df: pd.DataFrame, geometry: dict, pixel_size: float) -> pd.DataFrame:
    """Compute rlnLC_* arm columns for every particle row.

    For each arm defined in *geometry*, the arm's local anchor and
    direction_point are rotated into the global frame and the tangent
    Euler angles are computed.  A centre feature (suffix ``0``) is
    auto-derived as the mean of all arm anchors / direction_points.
    """
    required_cols = [
        "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
        "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column in input STAR: {col}")

    arms = geometry["arms"]
    n_arms = len(arms)
    out_df = df.copy()

    # Create output columns: suffix 1..N for arms, 0 for centre
    suffixes = [str(i + 1) for i in range(n_arms)] + ["0"]
    for suf in suffixes:
        for prefix in ("CoordinateX", "CoordinateY", "CoordinateZ",
                        "AngleRot", "AngleTilt", "AnglePsi"):
            out_df[f"rlnLC_{prefix}{suf}"] = 0.0

    # Pre-parse local arm coordinates
    arm_anchors_local = [np.array(a["anchor"], dtype=float) for a in arms]
    arm_dp_local = [np.array(a["direction_point"], dtype=float) for a in arms]
    arm_tangent_dirs = [a["tangent"] for a in arms]

    n_particles = len(df)
    print(f"Processing {n_particles} particles ({n_arms} arms per particle)...")

    for row_idx in range(n_particles):
        row = df.iloc[row_idx]

        # Global centre and rotation matrix for this particle
        global_centre = np.array([
            row["rlnCoordinateX"], row["rlnCoordinateY"], row["rlnCoordinateZ"]
        ])
        rot_mat = _relion_rotation_matrix(
            row["rlnAngleRot"], row["rlnAngleTilt"], row["rlnAnglePsi"]
        )

        global_anchors = []
        global_dps = []

        # --- Per-arm features (suffix 1, 2, ...) ---
        for arm_idx in range(n_arms):
            suf = str(arm_idx + 1)

            g_anchor = global_centre + (rot_mat.T @ arm_anchors_local[arm_idx]) / pixel_size
            g_dp = global_centre + (rot_mat.T @ arm_dp_local[arm_idx]) / pixel_size

            global_anchors.append(g_anchor)
            global_dps.append(g_dp)

            # Tangent direction
            if arm_tangent_dirs[arm_idx] == "direction_point_to_anchor":
                p_from, p_to = g_dp, g_anchor
            else:
                p_from, p_to = g_anchor, g_dp

            r, t, p = euler_zyz_from_two_points(p_from, p_to)

            out_df.at[row_idx, f"rlnLC_CoordinateX{suf}"] = g_anchor[0]
            out_df.at[row_idx, f"rlnLC_CoordinateY{suf}"] = g_anchor[1]
            out_df.at[row_idx, f"rlnLC_CoordinateZ{suf}"] = g_anchor[2]
            out_df.at[row_idx, f"rlnLC_AngleRot{suf}"] = r
            out_df.at[row_idx, f"rlnLC_AngleTilt{suf}"] = t
            out_df.at[row_idx, f"rlnLC_AnglePsi{suf}"] = p

        # --- Centre feature (suffix 0) ---
        centre_anchor = np.mean(global_anchors, axis=0)
        centre_dp = np.mean(global_dps, axis=0)

        r0, t0, p0 = euler_zyz_from_two_points(centre_dp, centre_anchor)

        out_df.at[row_idx, "rlnLC_CoordinateX0"] = centre_anchor[0]
        out_df.at[row_idx, "rlnLC_CoordinateY0"] = centre_anchor[1]
        out_df.at[row_idx, "rlnLC_CoordinateZ0"] = centre_anchor[2]
        out_df.at[row_idx, "rlnLC_AngleRot0"] = r0
        out_df.at[row_idx, "rlnLC_AngleTilt0"] = t0
        out_df.at[row_idx, "rlnLC_AnglePsi0"] = p0

    return out_df


def preprocess_star(
    input_star: str,
    output_star: str,
    geometry: dict,
    pixel_size: float,
) -> None:
    """One-pass preprocessing: raw RELION STAR -> arm-annotated STAR.

    Parameters
    ----------
    input_star : str
        Path to a RELION-style STAR file containing at minimum
        ``rlnCoordinateX/Y/Z`` and ``rlnAngleRot/Tilt/Psi``.
    output_star : str
        Path for the output STAR file with added ``rlnLC_*`` columns.
    geometry : dict
        Geometry configuration (from :func:`load_geometry`).
    pixel_size : float
        Pixel size in Angstroms.
    """
    _validate_geometry(geometry)

    print(f"Reading input STAR file: {input_star}")
    star_data = starfile.read(input_star)

    # Handle single DataFrame or dict-of-blocks
    if isinstance(star_data, pd.DataFrame):
        block_name = "particles"
        df = star_data
    else:
        block_name = None
        for k, v in star_data.items():
            if isinstance(v, pd.DataFrame) and "rlnCoordinateX" in v.columns:
                block_name = k
                df = v
                break
        if block_name is None:
            raise ValueError("No block with rlnCoordinateX found in the input STAR file.")

    processed_df = _process_particles(df, geometry, pixel_size)

    # Write output
    if isinstance(star_data, pd.DataFrame):
        output_data = {block_name: processed_df}
    else:
        output_data = star_data.copy()
        output_data[block_name] = processed_df

    starfile.write(output_data, output_star, overwrite=True)
    print(f"Success! Output written to {output_star}")
