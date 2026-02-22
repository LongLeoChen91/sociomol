import numpy as np
import eulerangles

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
