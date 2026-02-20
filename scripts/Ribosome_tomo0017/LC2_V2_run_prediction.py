# Build nucleosome linker connectivity with coordinates converted to nm
# Input STAR is in bin1 pixel units; pixel_size = 1.513 Å -> 0.1513 nm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation as R

from linker_prediction import Nucleosome, LinkerAssigner
  # <- from your existing file

# ---------- User params ----------
INPUT_STAR  = "Avg_Linkers_ID_Reset_A_ribosome80s_T0017_with_origin.star"
OUTPUT_STAR = "Avg_Linkers_annotated.star"
EDGES_CSV   = "Linker_edges.csv"

PIXEL_SIZE_A  = 1.96               # Å/pixel
PIXEL_SIZE_NM = PIXEL_SIZE_A / 10.0 # nm/pixel

# cutoff & model params (nm)
DIST_CUTOFF_NM = 40 # [nm] Arm–arm distance cutoff; (i,j) kept if min(|ai-aj|) < cutoff
LP_NM = 1.5     # persistence length
L0_NM = 40       # reference length
P_THRESHOLD = 0.3  
IGNORE_MEASURED_L = False      # False or True: ignore measured L_nm, use constant 15 nm in bending energy, only θ matters

# ---------- Helpers ----------
def pick_cols(df, names):
    """Return first existing columns matching names (accepts with/without leading underscore)."""
    out = []
    for n in names:
        if n in df.columns:
            out.append(n)
        else:
            alt = "_" + n if not n.startswith("_") else n[1:]
            if alt in df.columns:
                out.append(alt)
            else:
                raise KeyError(f"Column not found: {n} or {alt}")
    return out

def pick_col(df, name):
    return pick_cols(df, [name])[0]

def colvec(df, cols):
    return df[cols].to_numpy(dtype=float)

import numpy as np
import eulerangles

def euler_zyz_to_Zaxis(rot_tilt_psi_deg: np.ndarray) -> np.ndarray:
    """
    Intrinsic zyz → rotate local Z=[0,0,1] to global.
    Return (N,3) array, each row is a unit vector.
    """
    rot_mats = eulerangles.euler2matrix(
        rot_tilt_psi_deg, axes="zyz", intrinsic=True, right_handed_rotation=True
    )   # (N,3,3)

    # Take the third row of each matrix
    Zs = np.array([r[2, :] for r in rot_mats])  # shape (N,3)

    # Normalize
    Zs = Zs / np.linalg.norm(Zs, axis=1, keepdims=True)
    return Zs


# ---------- Main ----------
if __name__ == "__main__":
    print(f"Reading input STAR file: {INPUT_STAR}")
    data = starfile.read(INPUT_STAR, always_dict=True)
    # pick particles table
    if "data_particles" in data:
        df = data["data_particles"]
    elif "particles" in data:
        df = data["particles"]
    else:
        # fallback: first table
        df = next(iter(data.values()))

    # strip accidental spaces in column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # print("Columns in STAR:", list(df.columns))

    # ---- pick columns (accept rln* or _rln*) ----
    IDCOL = pick_col(df, "rlnTomoParticleId")
    TOMO_COL = pick_col(df, "rlnTomoName")

    C0 = pick_cols(df, ["rlnLC_CoordinateX0","rlnLC_CoordinateY0","rlnLC_CoordinateZ0"]) # nucleosome dyad  (H1 binding site) coordinate; not used in linker assignment
    # A0 = pick_cols(df, ["rlnLC_AngleRot0","rlnLC_AngleTilt0","rlnLC_AnglePsi0"]) # nucleosome dyad orientation (H1 binding site); not used in linker assignment

    C1 = pick_cols(df, ["rlnLC_CoordinateX1","rlnLC_CoordinateY1","rlnLC_CoordinateZ1"])
    A1 = pick_cols(df, ["rlnLC_AngleRot1","rlnLC_AngleTilt1","rlnLC_AnglePsi1"])

    C2 = pick_cols(df, ["rlnLC_CoordinateX2","rlnLC_CoordinateY2","rlnLC_CoordinateZ2"])
    A2 = pick_cols(df, ["rlnLC_AngleRot2","rlnLC_AngleTilt2","rlnLC_AnglePsi2"])

    # ---- load arrays (pixels for coords; degrees for angles) ----
    ids = df[IDCOL].to_numpy()
    cen_px = colvec(df, C0)
    a1_px  = colvec(df, C1)
    a2_px  = colvec(df, C2)

    ang1_deg = colvec(df, A1)
    ang2_deg = colvec(df, A2)

    # ---- convert coordinates to nm ----
    cen = cen_px * PIXEL_SIZE_NM
    a1  = a1_px  * PIXEL_SIZE_NM
    a2  = a2_px  * PIXEL_SIZE_NM

    # ---- unit tangents from intrinsic zyz Euler ----
    t1 = euler_zyz_to_Zaxis(ang1_deg)
    t2 = euler_zyz_to_Zaxis(ang2_deg)

    # ---- build nucleosomes list ----
    nucs = [Nucleosome(center=cen[i], a1=a1[i], a2=a2[i], t1=t1[i], t2=t2[i]) for i in range(len(df))]

    # ---- run linker assignment (nm domain) ----
    assigner = LinkerAssigner(
        nucs,
        lp=LP_NM,
        L0=L0_NM,
        dist_cutoff_nm=DIST_CUTOFF_NM,   # coordinates already in nm
        p_threshold=P_THRESHOLD,
        theta_mode="alpha_sum",          # Improved θ = α_i + α_j; theta_mode="tangent_tangent" (old definition, for comparison)
        require_toward_line=True,        # Require arms pointing toward connection line
        toward_cos_threshold=0.0,         # angle with line < 90° (half bending angle/each arm-line angle); can change to 0.5≈60° stricter; (1=0°,0=90°,-1=180°)
        # === NEW: single switch down to the assigner ===
        ignore_measured_L=IGNORE_MEASURED_L,  # if True: ignore measured L in Ubend, use L_fixed internally
        port_pairing="complement"
    )
    assignments, adj = assigner.run()


    # ---- write edges CSV ----
    idmap = {i: int(float(ids[i])) if str(ids[i]).replace('.', '', 1).isdigit() else ids[i] for i in range(len(ids))}
    rows = []
    for a in assignments:
        rows.append({
            "i_idx": a.i,
            "j_idx": a.j,
            "i_id": idmap[a.i],
            "j_id": idmap[a.j],
            "arm_i": a.arm_i,
            "arm_j": a.arm_j,
            "theta_rad": a.theta,
            "theta_deg": np.degrees(a.theta),
            "L_nm": a.L,
            "D_nm": a.D,                 # NEW
            "P": a.prob,
            "Psecond": a.psecond,        # NEW
            "Pmax_over_Psecond": a.pmax_over_psecond,
        })
    pd.DataFrame(rows).to_csv(EDGES_CSV, index=False)

    # ---- annotate STAR ----
    partner_arm0 = np.full(len(df), -1, dtype=int)
    partner_arm1 = np.full(len(df), -1, dtype=int)
    for a in assignments:
        j_id = idmap[a.j]
        i_id = idmap[a.i]
        if a.arm_i == 0:
            partner_arm0[a.i] = j_id
        else:
            partner_arm1[a.i] = j_id
        if a.arm_j == 0:
            partner_arm0[a.j] = i_id
        else:
            partner_arm1[a.j] = i_id

    df["rlnLC_LinkPartnerArm0"] = partner_arm0
    df["rlnLC_LinkPartnerArm1"] = partner_arm1

    # connected components
    visited = np.zeros(len(df), dtype=bool)
    comp_id = np.full(len(df), -1, dtype=int)
    comp = 0
    for s in range(len(df)):
        if visited[s]:
            continue
        queue = [s]
        touched = False
        while queue:
            u = queue.pop(0)
            if visited[u]:
                continue
            visited[u] = True
            comp_id[u] = comp
            for v in adj.get(u, []):
                if not visited[v]:
                    queue.append(v)
            touched = True
        if touched:
            comp += 1
    df["rlnLC_ChainComponent"] = comp_id

    # keep original tables structure; write as data_particles
    starfile.write({"particles": df}, OUTPUT_STAR, overwrite=True)

    # ---- console summary ----
    n_particles = len(df)
    n_edges = len(assignments)
    n_components = comp
    n_tomos = df[TOMO_COL].nunique() if TOMO_COL in df.columns else "NA"
    print(f"Particles (rows): {n_particles}; tomos: {n_tomos}")
    print(f"Assigned edges: {n_edges}; Components: {n_components}")
    print(f"Wrote {EDGES_CSV} and {OUTPUT_STAR}")
