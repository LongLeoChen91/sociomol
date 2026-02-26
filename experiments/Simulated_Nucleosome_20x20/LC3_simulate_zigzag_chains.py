"""
Simulate 20 nucleosome chains x 20 nucleosomes per chain,
arranged as a 3D zig-zag fiber with each chain randomly placed in space.

Local geometry (per chain, in chain-local frame):
  - Chain advances along local Z axis.
  - Nucleosomes alternate between two lateral stacks at ±ZZ_AMP_PX in local X.
  - Dyad (Z-axis) of even stack → local +X;  odd stack → local -X.
    → adjacent nucleosomes are face-to-face across the fiber.

Spatial randomization (per chain):
  - Each chain is given a random 3D rotation (uniformly on SO(3)) and a random
    3D translation within a bounding box.
  - Nucleosome positions and dyad orientations are both transformed accordingly.

Output: simulated_zigzag_chains_20x20.star
"""

import os
import math
import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation as ScipyRot

# ==========================================
# Parameters
# ==========================================
N_CHAINS          = 20
N_NUC_PER_CHAIN   = 20
PIXEL_A           = 8.0          # Å/pixel
PIXEL_NM          = PIXEL_A / 10.0   # 0.8 nm/pixel

SPACING_NM        = 30.0         # [nm] center-to-center between adjacent nucleosomes
ZZ_AMP_NM         = 8.0          # [nm] zig-zag lateral amplitude (half inter-stack distance)

CHAIN_SEP_NM      = 100.0        # [nm] chain-to-chain separation (Y direction)

# ==========================================
# Derived pixel distances
# ==========================================
SPACING_PX   = SPACING_NM  / PIXEL_NM   # 37.5 px
ZZ_AMP_PX    = ZZ_AMP_NM   / PIXEL_NM   # 10.0 px
CHAIN_SEP_PX = CHAIN_SEP_NM / PIXEL_NM  # 125.0 px

# Vertical (Z) step between adjacent nucleosomes
# From: SPACING_PX^2 = (2 * ZZ_AMP_PX)^2 + DZ_PX^2
DZ_PX = math.sqrt(SPACING_PX**2 - (2 * ZZ_AMP_PX)**2)

# Bounding box for chain *origins* (in pixels).
# All 20 nucleosomes in a chain span ~634px along the chain axis, which after
# random rotation can project onto any axis. To keep everything comfortably
# inside the box we shrink the origin sampling range by CHAIN_LENGTH_PX.
CHAIN_LENGTH_PX = (N_NUC_PER_CHAIN - 1) * DZ_PX   # ~603 px for 20 nucleosomes

# ---- EDIT THESE to set the visible volume ----
BOX_X = 800    # [px] total X range: particles will be in [0, BOX_X]
BOX_Y = 800    # [px] total Y range
BOX_Z = 800    # [px] total Z range
# -------------------------------------------------

RANDOM_SEED = 42

# ==========================================
# Helper: convert rotation matrix → ZYZ Euler angles (degrees)
# ==========================================
def rotmat_to_zyz_deg(R):
    """Convert 3x3 rotation matrix to Relion ZYZ Euler angles (rot, tilt, psi) in degrees.
    Uses scipy, which solves R = Rz(alpha) * Ry(beta) * Rz(gamma).
    """
    r = ScipyRot.from_matrix(R)
    # scipy 'ZYZ' intrinsic = Rz(a)*Ry(b)*Rz(c), matching Relion's convention
    alpha, beta, gamma = r.as_euler('ZYZ', degrees=True)
    return float(alpha), float(beta), float(gamma)

# ==========================================
# Main
# ==========================================
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Pixel size     : {PIXEL_A} Å/px = {PIXEL_NM} nm/px")
    print(f"Neighbor step  : {SPACING_NM} nm = {SPACING_PX:.3f} px")
    print(f"Zigzag amp (a) : {ZZ_AMP_NM} nm = {ZZ_AMP_PX:.3f} px")
    print(f"Z-step per nuc : {DZ_PX*PIXEL_NM:.3f} nm = {DZ_PX:.3f} px")
    print()

    rows = []

    # Pre-generate random rigid body transforms for each chain
    # Random rotation: uniform on SO(3) via scipy
    chain_rotations = ScipyRot.random(N_CHAINS, random_state=rng)
    # Random translation: origins sampled within [margin, BOX-margin] so the
    # chain (length ~CHAIN_LENGTH_PX) stays mostly inside the bounding box.
    margin = CHAIN_LENGTH_PX / 2.0
    chain_origins = np.column_stack([
        rng.uniform(margin, max(margin + 1, BOX_X - margin), size=N_CHAINS),
        rng.uniform(margin, max(margin + 1, BOX_Y - margin), size=N_CHAINS),
        rng.uniform(margin, max(margin + 1, BOX_Z - margin), size=N_CHAINS),
    ])
    print(f"Bounding box   : X[0,{BOX_X}]  Y[0,{BOX_Y}]  Z[0,{BOX_Z}] px")
    print(f"Chain length   : {CHAIN_LENGTH_PX:.1f} px;  origin margin: ±{margin:.1f} px")

    for c_idx in range(N_CHAINS):
        tomo_name  = f"simulated_chain_{c_idx:02d}"
        R_chain    = chain_rotations[c_idx].as_matrix()   # 3x3 rotation matrix
        origin     = chain_origins[c_idx]                 # (3,) translation

        for n_idx in range(N_NUC_PER_CHAIN):
            # ------ Local-frame position ------
            # Even (stack A): local_x = +ZZ_AMP_PX
            # Odd  (stack B): local_x = -ZZ_AMP_PX
            local_pos = np.array([
                ZZ_AMP_PX * (1 if n_idx % 2 == 0 else -1),
                0.0,
                n_idx * DZ_PX
            ])

            # ------ Local-frame dyad orientation ------
            # Stack A (even): dyad → local +X  →  R_local rotates Z to +X
            # Stack B (odd):  dyad → local -X  →  R_local rotates Z to -X
            if n_idx % 2 == 0:
                # Z → +X: rotate around Y by -90° (i.e., Ry(-90))
                R_local = ScipyRot.from_euler('y', -90, degrees=True).as_matrix()
            else:
                # Z → -X: rotate around Y by +90° (i.e., Ry(+90))
                R_local = ScipyRot.from_euler('y', 90, degrees=True).as_matrix()

            # ------ Apply chain random rigid body transform ------
            # World position: rotate then translate
            world_pos = R_chain @ local_pos + origin

            # World orientation: compose chain rotation with local dyad rotation
            # R_world = R_chain * R_local  (rotate local frame into world frame)
            R_world = R_chain @ R_local

            # Convert R_world back to ZYZ Euler angles
            rot, tilt, psi = rotmat_to_zyz_deg(R_world)

            rows.append({
                "rlnTomoName"      : tomo_name,
                "rlnTomoParticleId": n_idx + 1,
                "rlnCoordinateX"   : round(float(world_pos[0]), 4),
                "rlnCoordinateY"   : round(float(world_pos[1]), 4),
                "rlnCoordinateZ"   : round(float(world_pos[2]), 4),
                "rlnAngleRot"      : round(rot,  4),
                "rlnAngleTilt"     : round(tilt, 4),
                "rlnAnglePsi"      : round(psi,  4),
            })

    df = pd.DataFrame(rows)

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_star = os.path.join(out_dir, "simulated_zigzag_chains_20x20.star")
    starfile.write({"particles": df}, out_star, overwrite=True)

    print(df.head(10).to_string(index=False))
    print(f"\nWrote: {out_star}")
    print(f"Total particles: {len(df)}")

if __name__ == "__main__":
    main()
