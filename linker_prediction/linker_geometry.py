#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linker_geometry.py
Helper functions to build "stick" midpoints and Euler angles (Relion intrinsic ZYZ).

Conventions
-----------
- Coordinates: 3D numpy arrays shaped (3,), unit-free but assumed in nanometers for consistency
  with cryo-ET geometry.
- Frames/Rotations: Right-handed; rotation matrix R maps from local -> lab.
- Euler angles: Relion "intrinsic ZYZ" convention; return order is (Rot, Tilt, Psi) in DEGREES.
- Vectors: "Z" axis of the local frame is aligned to the direction p_i -> p_j.

Functions
---------
- unit(v): Normalize a 3D vector robustly (safe for near-zero norms).
- build_frame_from_z_and_trialx(z_dir, x_trial): Construct an orthonormal frame with local Z along z_dir,
  and local X chosen close to x_trial projected onto the Z-plane; returns 3x3 rotation matrix whose rows
  are [X; Y; Z] in the LAB frame, or None if degenerate.
- euler_zyz_from_two_points(p_i, p_j, x_ref=None): Compute Relion intrinsic ZYZ angles (deg) whose local
  Z aligns to vector (p_j - p_i). If x_ref is given, try to keep X near x_ref (projected on Z-plane).
- midpoint_from_two_points(p_i, p_j): Return midpoint of two 3D points.

Notes
-----
- If p_i == p_j or z_dir is degenerate/collinear with x_trial, fallbacks are used to keep the function
  stable and predictable.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
from eulerangles import matrix2euler  # requires: pip install eulerangles


def unit(v: np.ndarray) -> np.ndarray:
    """Normalize a vector robustly; returns zero-vector unchanged if norm is ~0."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        # Return as-is to make caller decide fallback behavior.
        return v
    return v / n

def safe_clip_dot(u: np.ndarray, v: np.ndarray) -> float:
    """Clip dot product to [-1, 1] for robust arccos."""
    x = float(np.dot(u, v))
    return max(-1.0, min(1.0, x))

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two unit vectors (radians)."""
    c = safe_clip_dot(u, v)
    return math.acos(c)

def angle_from_dot(dotval: float) -> float:
    """Clamp dot to [-1,1] then arccos, return radians."""
    return math.acos(max(-1.0, min(1.0, float(dotval))))

def arc_length_from_endpoints_and_angle(D: float, theta: float) -> Optional[float]:
    """
    L = theta * r, where r = D / (2*sin(theta/2)).
    Handles small-angle and degenerate cases robustly.
    Returns None if invalid geometry (theta ~ 0 and D ~ 0).
    """
    eps = 1e-8
    half = 0.5 * theta
    s = math.sin(half)
    if abs(s) < eps:
        # For very small theta, the circular arc radius -> large; use straight-line approximation L≈D
        return D
    r = D / (2.0 * s)
    if r <= 0:
        return None
    return theta * abs(r)

def theta_alpha_sum(center_i: np.ndarray, ai: np.ndarray, ti: np.ndarray,
                    center_j: np.ndarray, aj: np.ndarray, tj: np.ndarray,
                    max_half_bending_deg: float = 90.0) -> Tuple[float, bool]:
    """
    Bending angle using arm-line (no centers).

    a_ij = unit(aj - ai)
    alpha_i = angle(ti, +a_ij)
    alpha_j = angle(tj, -a_ij)
    theta   = alpha_i + alpha_j

    Both tangents must point toward ±a_ij within the max allowed half-bending cone. 
    If ai≈aj, fall back to angle(ti,tj).
    """
    aij = unit(aj - ai)
    if np.linalg.norm(aij) == 0:
        # Degenerate case: arm endpoints coincide;, fall back to ti·-tj method
        return angle_from_dot(np.dot(ti, -tj)), True

    dot_i = float(np.dot(ti, aij))    # ti toward +a_ij
    dot_j = float(np.dot(tj, -aij))   # tj toward -a_ij

    # Orientation constraint: require arms pointing toward the bridge within a cone
    toward_cos_threshold = math.cos(math.radians(max_half_bending_deg))
    if dot_i < toward_cos_threshold or dot_j < toward_cos_threshold:
        return float("nan"), False

    alpha_i = angle_from_dot(dot_i)
    alpha_j = angle_from_dot(dot_j)
    theta = alpha_i + alpha_j
    return theta, True



def build_frame_from_z_and_trialx(z_dir: np.ndarray, x_trial: np.ndarray) -> Optional[np.ndarray]:
    """
    Build a right-handed orthonormal frame from a target Z direction and a trial X.
    The returned rotation matrix R has rows [X; Y; Z] expressed in the LAB frame.

    Parameters
    ----------
    z_dir : (3,) array-like
        Desired local Z axis in LAB coordinates (will be normalized).
    x_trial : (3,) array-like
        A "hint" direction to define local X; will be projected onto the plane orthogonal to Z.

    Returns
    -------
    R : (3,3) np.ndarray or None
        Orthonormal right-handed frame with rows [X; Y; Z], or None if construction fails
        (e.g., z_dir ~ 0 or x_trial parallel to z_dir).
    """
    z = unit(z_dir)
    if np.linalg.norm(z) < 1e-12:
        return None

    # Project x_trial onto the plane perpendicular to z
    x_proj = np.asarray(x_trial, dtype=float) - np.dot(x_trial, z) * z
    if np.linalg.norm(x_proj) < 1e-12:
        return None

    x = unit(x_proj)
    y = np.cross(z, x)
    if np.linalg.norm(y) < 1e-12:
        return None
    y = unit(y)

    # Rows are [X; Y; Z] in LAB frame
    R = np.vstack([x, y, z])
    return R


def euler_zyz_from_two_points(p_i: np.ndarray,
                              p_j: np.ndarray,
                              x_ref: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    Given two 3D points p_i -> p_j, compute (Rot, Tilt, Psi) in degrees for Relion intrinsic ZYZ.
    The local Z axis is aligned to the vector v = (p_j - p_i). If x_ref is provided, we try to keep
    the local X axis close to x_ref projected in the Z-plane, improving orientation consistency.

    Parameters
    ----------
    p_i, p_j : (3,) array-like
        Endpoints defining the oriented "stick". Local Z is along v = p_j - p_i.
    x_ref : (3,) array-like or None
        Optional preferred X direction (in LAB frame). If given, it is prepended to the trial list.

    Returns
    -------
    rot, tilt, psi : float
        Relion intrinsic ZYZ angles in degrees.

    Robustness & Fallback
    ---------------------
    - If |v| ~ 0, we return (0, 0, 0).
    - If no valid frame can be built (e.g., x_trial parallel to v), we fall back to spherical
      parametrization with psi = 0.
    """
    p_i = np.asarray(p_i, float)
    p_j = np.asarray(p_j, float)

    v = p_j - p_i
    if np.linalg.norm(v) < 1e-12:
        return 0.0, 0.0, 0.0

    # Trial X candidates; prepend x_ref if provided
    trials = [
        np.array([ 1, 0, 0], dtype=float), np.array([-1, 0, 0], dtype=float),
        np.array([ 0, 1, 0], dtype=float), np.array([ 0,-1, 0], dtype=float),
        np.array([ 1, 1, 0], dtype=float), np.array([ 1,-1, 0], dtype=float),
        np.array([ 1, 0, 1], dtype=float), np.array([ 1, 0,-1], dtype=float),
        np.array([ 0, 1, 1], dtype=float), np.array([ 0, 1,-1], dtype=float),
        np.array([ 0, 0, 1], dtype=float), np.array([ 0, 0,-1], dtype=float),
    ]
    if x_ref is not None:
        trials = [np.asarray(x_ref, float)] + trials

    R = None
    for tx in trials:
        R = build_frame_from_z_and_trialx(v, tx)
        if R is not None:
            break

    if R is None:
        # Fallback: use spherical angles for v; set psi=0
        v_hat = unit(v)
        # rot = atan2(y, x), tilt = arccos(z)
        rot  = math.degrees(math.atan2(v_hat[1], v_hat[0]))
        tilt = math.degrees(math.acos(float(np.clip(v_hat[2], -1.0, 1.0))))
        psi  = 0.0
        return rot, tilt, psi

    # Convert rotation matrix to Relion intrinsic ZYZ (degrees)
    rot, tilt, psi = matrix2euler(
        R, axes="zyz", intrinsic=True, right_handed_rotation=True
    )
    return float(rot), float(tilt), float(psi)


def midpoint_from_two_points(p_i: np.ndarray, p_j: np.ndarray) -> np.ndarray:
    """
    Return the midpoint between p_i and p_j.

    Parameters
    ----------
    p_i, p_j : (3,) array-like

    Returns
    -------
    mid : (3,) np.ndarray
    """
    p_i = np.asarray(p_i, float)
    p_j = np.asarray(p_j, float)
    return 0.5 * (p_i + p_j)
