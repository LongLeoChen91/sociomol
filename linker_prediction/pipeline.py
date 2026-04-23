import os
import numpy as np
import pandas as pd
import starfile

from .models import Particle
from .assigner import LinkerAssigner
from .star_utils import pick_col, pick_cols, colvec, euler_zyz_to_Zaxis

def run_prediction_pipeline(
    input_star: str,
    output_star: str,
    edges_csv: str,
    pixel_size_a: float,
    dist_cutoff_nm: float,
    lp_nm: float,
    l0_nm: float,
    p_threshold: float,
    w_wlc: float = 1.0,
    w_L: float = 1.0,
    w_th: float = 1.0,
    w_L_sq: float = 0.0,
    w_th_sq: float = 0.0,
    theta0_deg: float = 45.0,
    l_ideal_nm: float = 20.0,
    l_std_nm: float = 10.0,
    theta_std_deg: float = 45.0,
    port_pairing: str = "any",
    theta_mode: str = "alpha_sum",
    max_half_bending_deg: float = 90.0,
):
    """
    Runs the full DSU Linker assignment pipeline covering:
      - Reading the Relion STAR file
      - Parsing pixels to physical nanoscale arrays
      - Deriving tangents from Euler coordinates
      - Running prediction logic via `LinkerAssigner`
      - Updating connected component IDs via BFS 
      - Outputting annotated edges CSV and the finished STAR block
    
    Paths must be relative to the caller (or absolute).
    """
    
    pixel_size_nm = pixel_size_a / 10.0
    
    print(f"Reading input STAR file: {input_star}")
    data = starfile.read(input_star, always_dict=True)
    
    if "data_particles" in data:
        df = data["data_particles"]
    elif "particles" in data:
        df = data["particles"]
    else:
        df = next(iter(data.values()))

    df = df.rename(columns={c: c.strip() for c in df.columns})

    IDCOL = pick_col(df, "rlnTomoParticleId")
    TOMO_COL = pick_col(df, "rlnTomoName") if "rlnTomoName" in df.columns or "_rlnTomoName" in df.columns else None

    # We do not use C0/A0 but maintain exact structure handling original files:
    C0 = pick_cols(df, ["rlnLC_CoordinateX0", "rlnLC_CoordinateY0", "rlnLC_CoordinateZ0"])
    
    C1 = pick_cols(df, ["rlnLC_CoordinateX1", "rlnLC_CoordinateY1", "rlnLC_CoordinateZ1"])
    A1 = pick_cols(df, ["rlnLC_AngleRot1", "rlnLC_AngleTilt1", "rlnLC_AnglePsi1"])

    C2 = pick_cols(df, ["rlnLC_CoordinateX2", "rlnLC_CoordinateY2", "rlnLC_CoordinateZ2"])
    A2 = pick_cols(df, ["rlnLC_AngleRot2", "rlnLC_AngleTilt2", "rlnLC_AnglePsi2"])

    ids = df[IDCOL].to_numpy()
    tomo_names = df[TOMO_COL].to_numpy() if TOMO_COL else None
    cen_px = colvec(df, C0)
    a1_px  = colvec(df, C1)
    a2_px  = colvec(df, C2)
    ang1_deg = colvec(df, A1)
    ang2_deg = colvec(df, A2)

    # Conversion to physical metric structure
    cen = cen_px * pixel_size_nm
    a1  = a1_px  * pixel_size_nm
    a2  = a2_px  * pixel_size_nm

    t1 = euler_zyz_to_Zaxis(ang1_deg)
    t2 = euler_zyz_to_Zaxis(ang2_deg)

    tomo_names_for_grouping = df[TOMO_COL].to_numpy() if TOMO_COL else np.array(["unknown"] * len(df))
    unique_tomos = np.unique(tomo_names_for_grouping)
    
    all_assignments = []
    
    for tomo in unique_tomos:
        # Get global indices for this tomogram
        global_indices = np.where(tomo_names_for_grouping == tomo)[0]
        
        if len(global_indices) == 0:
            continue
            
        # Build objects for this tomogram
        tomo_nucs = [Particle(center=cen[i], a1=a1[i], a2=a2[i], t1=t1[i], t2=t2[i]) for i in global_indices]
        
        # Evaluation for this tomogram
        assigner = LinkerAssigner(
            tomo_nucs,
            lp=lp_nm,
            L0=l0_nm,
            dist_cutoff_nm=dist_cutoff_nm,
            p_threshold=p_threshold,
            w_wlc=w_wlc,
            w_L=w_L,
            w_th=w_th,
            w_L_sq=w_L_sq,
            w_th_sq=w_th_sq,
            theta0_deg=theta0_deg,
            l_ideal_nm=l_ideal_nm,
            l_std_nm=l_std_nm,
            theta_std_deg=theta_std_deg,
            theta_mode=theta_mode,
            max_half_bending_deg=max_half_bending_deg,
            port_pairing=port_pairing
        )
        tomo_assignments, _ = assigner.run()
        
        # Map local indices back to global indices
        for a in tomo_assignments:
            a.i = global_indices[a.i]
            a.j = global_indices[a.j]
            all_assignments.append(a)
            
    assignments = all_assignments

    # CSV saving formatting
    idmap = {i: int(float(ids[i])) if str(ids[i]).replace('.', '', 1).isdigit() else ids[i] for i in range(len(ids))}
    rows = []
    for a in assignments:
        rows.append({
            "tomo_name": tomo_names[a.i] if tomo_names is not None else "unknown",
            "i_idx": a.i,
            "j_idx": a.j,
            "i_id": idmap[a.i],
            "j_id": idmap[a.j],
            "arm_i": a.arm_i,
            "arm_j": a.arm_j,
            "theta_rad": a.theta,
            "theta_deg": np.degrees(a.theta),
            "L_nm": a.L,
            "D_nm": a.D,
            "P": a.prob,
            "Psecond": a.psecond,
            "Pmax_over_Psecond": a.pmax_over_psecond,
        })
    pd.DataFrame(rows).to_csv(edges_csv, index=False)

    # STAR file mutation
    partner_arm0 = np.full(len(df), -1, dtype=int)
    partner_arm1 = np.full(len(df), -1, dtype=int)
    for a in assignments:
        j_id = idmap[a.j]
        i_id = idmap[a.i]
        
        if a.arm_i == 0: partner_arm0[a.i] = j_id
        else:            partner_arm1[a.i] = j_id
        
        if a.arm_j == 0: partner_arm0[a.j] = i_id
        else:            partner_arm1[a.j] = i_id

    df["rlnLC_LinkPartnerArm0"] = partner_arm0
    df["rlnLC_LinkPartnerArm1"] = partner_arm1

    # Traverse undirected graph mapping for clusters / chains utilizing BFS
    adj = {}
    for a in assignments:
        adj.setdefault(a.i, []).append(a.j)
        adj.setdefault(a.j, []).append(a.i)

    visited = np.zeros(len(df), dtype=bool)
    comp_id = np.full(len(df), -1, dtype=int)
    comp = 0
    for s in range(len(df)):
        if visited[s]: continue
        
        queue = [s]
        touched = False
        while queue:
            u = queue.pop(0)
            if visited[u]: continue
            visited[u] = True
            comp_id[u] = comp
            for v in adj.get(u, []):
                if not visited[v]:
                    queue.append(v)
            touched = True
            
        if touched: comp += 1
            
    df["rlnLC_ChainComponent"] = comp_id
    starfile.write({"particles": df}, output_star, overwrite=True)

    # Console readout status outputs
    n_particles = len(df)
    n_edges = len(assignments)
    n_components = comp
    n_tomos = df[TOMO_COL].nunique() if TOMO_COL and TOMO_COL in df.columns else "NA"
    
    print(f"Particles (rows): {n_particles}; tomos: {n_tomos}")
    print(f"Assigned edges: {n_edges}; Components: {n_components}")
    print(f"Wrote {edges_csv} and {output_star}")
