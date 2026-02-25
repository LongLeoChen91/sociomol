import math
import numpy as np

def bending_energy_lp(L: float, theta: float, lp: float = 50.0) -> float:
    eps = 1e-12
    L = max(L, eps)
    return (2.0 * lp / L) * (0.5 * theta) ** 2

def connection_probability(L: float, theta: float, L0: float = 20.0, lp: float = 1.5,
                           w_wlc: float = 1.0, w_L: float = 1.0, w_th: float = 1.0, 
                           w_L_sq: float = 0.0, w_th_sq: float = 0.0,
                           theta0_rad: float = math.pi/4.0,
                           L_ideal: float = 20.0, L_std: float = 10.0,
                           theta_std_rad: float = math.pi/4.0) -> float:
    """
    P(L, theta) ∝ exp[
      - w_wlc * (2 lp / L) * (θ/2)^2
      - w_L   * (L / L0)
      - w_th  * ( (θ/2) / θ0 )
      - w_L_sq * ((L - L_ideal) / L_std)^2
      - w_th_sq * (θ / theta_std)^2
    ]
    """
    if not np.isfinite(theta):
        return 0.0

    if L is None or L <= 0 or not np.isfinite(L):
        return 0.0
        
    E_wlc = bending_energy_lp(L, theta, lp=lp)
    E_len = L / L0
    E_ang = (0.5 * theta) / theta0_rad
    
    # Sub-Gaussian / Squared terms
    E_len_sq = ((L - L_ideal) / L_std) ** 2
    E_ang_sq = (theta / theta_std_rad) ** 2
    
    E_total = (w_wlc * E_wlc) + (w_L * E_len) + (w_th * E_ang) + (w_L_sq * E_len_sq) + (w_th_sq * E_ang_sq)
    return math.exp(-E_total)
