# -*- coding: utf-8 -*-
"""
Linker assignment engine for generalized two-armed particles
Implements the physics-based probability, greedy assignment, and constraints:
- One connection per arm/port
- No cycles across particles/assemblies
- Probability threshold stopping rule

Coordinates in nanometers. Angles in radians.
Author: Long Chen
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# -------------------------
# Geometry & probability
# -------------------------

def safe_clip_dot(u: np.ndarray, v: np.ndarray) -> float:
    """Clip dot product to [-1, 1] for robust arccos."""
    x = float(np.dot(u, v))
    return max(-1.0, min(1.0, x))

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two unit vectors (radians)."""
    c = safe_clip_dot(u, v)
    return math.acos(c)

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

def bending_energy_lp(L: float, theta: float, lp: float = 50.0) -> float:
    eps = 1e-12
    L = max(L, eps)
    return (2.0 * lp / L) * (0.5 * theta) ** 2

def connection_probability(L: float, theta: float, L0: float = 15.0, lp: float = 50.0, ignore_measured_L: bool = False) -> float:
    """
    P(L, theta) ∝ exp(-L/L0) * exp( - Ubend/(kBT) ), with Ubend/(kBT) in reduced units.
    
    If ignore_measured_L = True:
        - Use a fixed contour length (L_fixed = 15 nm) inside Ubend
        - Drop the exp(-L/L0) length prior
        → Probability depends only on θ (smaller θ = higher P)    
    """
    if not np.isfinite(theta):
        return 0.0
    if ignore_measured_L:
        # Angle-only WLC model: fixed arc length, no length prior
        L_fixed = 15.0
        Ub_over_kT = bending_energy_lp(L_fixed, theta, lp=lp)
        return math.exp(-Ub_over_kT)
    else:
        # Original WLC model: use measured L with length prior
        if L is None or L <= 0 or not np.isfinite(L):
            return 0.0
        Ub_over_kT = bending_energy_lp(L, theta, lp=lp)
        return math.exp(-L / L0) * math.exp(-Ub_over_kT)
    
# -------------------------
# Data structures
# -------------------------

@dataclass
class Particle:
    """A generalized two-armed particle with a center, two arm exit points a1,a2 and unit tangents t1,t2."""
    center: np.ndarray      # shape (3,)
    a1: np.ndarray          # exit point for arm 1
    a2: np.ndarray          # exit point for arm 2
    t1: np.ndarray          # unit tangent at a1
    t2: np.ndarray          # unit tangent at a2

    def arm_point(self, arm: int) -> np.ndarray:
        return self.a1 if arm == 0 else self.a2

    def arm_tangent(self, arm: int) -> np.ndarray:
        return self.t1 if arm == 0 else self.t2

# -------------------------
# Graph utilities
# -------------------------

class DSU:
    """Disjoint Set Union (Union-Find) to prevent cycles."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

# -------------------------
# Core algorithm
# -------------------------

@dataclass
class LinkerAssignment:
    i: int
    j: int
    arm_i: int  # 0 or 1
    arm_j: int  # 0 or 1
    theta: float
    L: float
    D: float                       # NEW: straight-line distance between arm endpoints (nm)
    prob: float
    psecond: Optional[float] = None  # NEW: second-best probability for (i,j) over remaining combos
    pmax_over_psecond: Optional[float] = None

class LinkerAssigner:
    def __init__(self, particles: List[Particle],
                 lp: float = 50.0, L0: float = 15.0,
                 dist_cutoff_nm: float = 20.0, p_threshold: float = 0.1,
                 theta_mode: str = "alpha_sum",            # mode for angle calculation
                 require_toward_line: bool = True,         # require tangents pointing toward the arm-line
                 toward_cos_threshold: float = 0.0,       # cosine threshold for orientation constraint
                 ignore_measured_L: bool = False,                 # <<< NEW: single switch (True = only theta)
                 port_pairing: str = "any"):              # "any" or "complement"
        """
        require_toward_line : bool, default True
            If True, enforce that both arm tangents point sufficiently toward
            the arm-line direction (a_ij = unit(aj - ai)):
                dot(t_i, +a_ij) >= toward_cos_threshold  and
                dot(t_j, -a_ij) >= toward_cos_threshold.
            Set False to skip this feasibility check.

        toward_cos_threshold : float, default 0.0
            Cosine threshold for the orientation constraint relative to the
            arm-line. Typical values:
                0.0  ~ angle < 90° (lenient)
                0.2  ~ angle < ~78° (moderate)
                0.5  ~ angle < 60° (strict)
                
        port_pairing : str, default "any"
            If "any", allows any port combinations (0->0, 1->1, 0->1, 1->0).
            If "complement", only allows complementary pairing (0->1 or 1->0), 
            and forbids (0->0) and (1->1).
        """
        self.particles = particles
        self.N = len(particles)
        self.lp = lp
        self.L0 = L0
        self.dist_cutoff = dist_cutoff_nm
        self.p_threshold = p_threshold

        # New configuration options
        self.theta_mode = theta_mode
        self.require_toward_line = require_toward_line
        self.toward_cos_threshold = toward_cos_threshold
        # <<< NEW: store the single switch
        self.ignore_measured_L = ignore_measured_L
        self.port_pairing = port_pairing

        self.arm_used = np.zeros((self.N, 2), dtype=bool)
        self.adj: Dict[int, List[int]] = {i: [] for i in range(self.N)}
        self.assignments: List[LinkerAssignment] = []

    def _pairwise_candidates(self) -> List[Tuple[int,int]]:
        """Find candidate node pairs if any arm-arm distance < cutoff."""
        cands = []
        for i in range(self.N):
            ai1, ai2 = self.particles[i].a1, self.particles[i].a2
            for j in range(i+1, self.N):
                nj = self.particles[j]
                dlist = [
                    np.linalg.norm(ai1 - nj.a1),
                    np.linalg.norm(ai1 - nj.a2),
                    np.linalg.norm(ai2 - nj.a1),
                    np.linalg.norm(ai2 - nj.a2),
                ]
                if min(dlist) < self.dist_cutoff:
                    cands.append((i, j))
        return cands

    def _compute_four_probs(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For arms (i:0/1) x (j:0/1), compute theta, L, P.
        Supports two theta modes: 
          - "alpha_sum": arm-line definition:a_ij = unit(aj - ai); theta = angle(t_i, +a_ij) + angle(t_j, -a_ij), with optional orientation checks against the arm-line.
          - "tangent_tangent": theta = arccos(clip(t_i · t_j))  (legacy)
        Returns arrays of shape (2,2).
        """
        theta = np.zeros((2,2), dtype=float)
        L = np.zeros((2,2), dtype=float)
        P = np.zeros((2,2), dtype=float)

        ci = self.particles[i].center
        cj = self.particles[j].center

        for arm_i in (0, 1):
            for arm_j in (0, 1):
                # Apply port pairing rules
                if self.port_pairing == "complement" and arm_i == arm_j:
                    theta[arm_i, arm_j] = float("nan")
                    L[arm_i, arm_j] = float("nan")
                    P[arm_i, arm_j] = 0.0
                    continue
                
                ai = self.particles[i].arm_point(arm_i)
                aj = self.particles[j].arm_point(arm_j)
                ti = self.particles[i].arm_tangent(arm_i)
                tj = self.particles[j].arm_tangent(arm_j)

                # ---- theta computation modes ----
                if getattr(self, "theta_mode", "alpha_sum") == "alpha_sum":
                    # --- compute bending angle θ and feasibility ---
                    # NOTE:
                    #   - If feasible == False, this pair is discarded (probability forced to 0).
                    #   - In that case, θ is only a placeholder (0.0 or NaN) and must NOT be used for scoring or analysis.
                    th, feasible = theta_alpha_sum(
                        ci, ai, ti, cj, aj, tj,
                        toward_cos_threshold=getattr(self, "toward_cos_threshold", 0.0),
                        require_toward_line=getattr(self, "require_toward_line", True),
                    )
                    if not feasible:
                        theta[arm_i, arm_j] = 0.0
                        L[arm_i, arm_j] = 0.0
                        P[arm_i, arm_j] = 0.0
                        continue
                else:
                    # legacy: theta = arccos( ti · tj )
                    th = angle_between(ti, tj)

                # D = |aj - ai|
                D = float(np.linalg.norm(aj - ai))

                # L = theta * r, r = D / (2 sin(theta/2)); robust fallback handles tiny theta
                Lij = arc_length_from_endpoints_and_angle(D, th)

                # P(L, θ)
                Pij = connection_probability(Lij, th, L0=self.L0, lp=self.lp, ignore_measured_L=self.ignore_measured_L)
                theta[arm_i, arm_j] = th
                L[arm_i, arm_j] = (Lij if Lij is not None else 0.0)
                P[arm_i, arm_j] = Pij

        return theta, L, P


    def run(self):
        # Build DSU for no-cycles constraint
        dsu = DSU(self.N)

        # Precompute candidates and probabilities
        candidates = self._pairwise_candidates()

        # Storage for 3D probabilities: dict keyed by (i,j) -> (theta[2,2], L[2,2], P[2,2])
        prob_cache: Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for (i, j) in candidates:
            prob_cache[(i, j)] = self._compute_four_probs(i, j)

        # Greedy loop
        while True:
            # Find global max P over all feasible entries
            best = None  # (Pval, i, j, arm_i, arm_j)
            second_best_same_pair: Dict[Tuple[int,int], float] = {}

            for (i, j), (theta_ij, L_ij, P_ij) in prob_cache.items():
                # Skip if already connected i-j
                if j in self.adj[i]:
                    continue
                # Loop over four arm-arm combos
                for arm_i in (0,1):
                    for arm_j in (0,1):
                        Pval = P_ij[arm_i, arm_j]
                        if Pval <= 0.0:
                            continue
                        # respect one-connection-per-arm
                        if self.arm_used[i, arm_i] or self.arm_used[j, arm_j]:
                            continue
                        # candidate for max
                        if best is None or Pval > best[0]:
                            best = (Pval, i, j, arm_i, arm_j)

            # Stopping condition: no candidate
            if best is None or best[0] < self.p_threshold:
                break

            Pbest, i, j, arm_i, arm_j = best

            # Check cycle constraint via DSU on node-level
            if not dsu.union(i, j):
                # would make a cycle; forbid this combination
                # zero it out and continue searching again
                theta_ij, L_ij, P_ij = prob_cache[(i, j)]
                P_ij[arm_i, arm_j] = 0.0
                prob_cache[(i, j)] = (theta_ij, L_ij, P_ij)
                continue

            # Compute Psecond for the same row (pair i,j) excluding the chosen arms
            theta_ij, L_ij, P_ij = prob_cache[(i, j)]
            chosen = P_ij[arm_i, arm_j]
            # second best across remaining combos of this pair (excluding chosen cell)
            mask = np.ones_like(P_ij, dtype=bool)
            mask[arm_i, arm_j] = False
            Psecond = float(P_ij[mask].max()) if np.any(mask) else 0.0
            ratio = (chosen / Psecond) if Psecond > 0 else float('inf')

            # Accept the edge
            self.adj[i].append(j)
            self.adj[j].append(i)
            self.arm_used[i, arm_i] = True
            self.arm_used[j, arm_j] = True

            # Compute D on-the-fly for the chosen arms to avoid changing caches:
            ai_sel = self.particles[i].arm_point(arm_i)   # endpoint on i for the chosen arm
            aj_sel = self.particles[j].arm_point(arm_j)   # endpoint on j for the chosen arm
            D_sel = float(np.linalg.norm(aj_sel - ai_sel))

            # Save assignment (now includes D and psecond)
            la = LinkerAssignment(
                i=i, j=j,
                arm_i=arm_i, arm_j=arm_j,
                theta=float(theta_ij[arm_i, arm_j]),
                L=float(L_ij[arm_i, arm_j]),
                D=D_sel,                                # NEW
                prob=float(Pbest),
                psecond=float(Psecond),                 # NEW
                pmax_over_psecond=ratio
            )
            self.assignments.append(la)

            # Zero-out (i, j, all arms) to avoid duplicate i-j connections
            P_ij[:, :] = 0.0
            prob_cache[(i, j)] = (theta_ij, L_ij, P_ij)

            # Also zero-out all entries that reuse the used arms on i or j with any other partner
            for (u, v), (th_uv, L_uv, P_uv) in prob_cache.items():
                if u == i or v == i:
                    # zero any cell that uses i's chosen arm
                    if u == i:
                        P_uv[arm_i, :] = 0.0
                    if v == i:
                        P_uv[:, arm_i] = 0.0
                    prob_cache[(u, v)] = (th_uv, L_uv, P_uv)
                if u == j or v == j:
                    if u == j:
                        P_uv[arm_j, :] = 0.0
                    if v == j:
                        P_uv[:, arm_j] = 0.0
                    prob_cache[(u, v)] = (th_uv, L_uv, P_uv)

        return self.assignments, self.adj

# ==== Utility helpers ====

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def angle_from_dot(dotval: float) -> float:
    """Clamp dot to [-1,1] then arccos, return radians."""
    return math.acos(max(-1.0, min(1.0, float(dotval))))

def theta_alpha_sum(center_i: np.ndarray, ai: np.ndarray, ti: np.ndarray,
                    center_j: np.ndarray, aj: np.ndarray, tj: np.ndarray,
                    toward_cos_threshold: float = 0.0,
                    require_toward_line: bool = True) -> Tuple[float, bool]:
    """
    Bending angle using arm-line (no centers).

    a_ij = unit(aj - ai)
    alpha_i = angle(ti, +a_ij)
    alpha_j = angle(tj, -a_ij)
    theta   = alpha_i + alpha_j

    If require_toward_line=True, both tangents must point toward ±a_ij
    above the given cosine threshold. If ai≈aj, fall back to angle(ti,tj).
    """
    aij = unit(aj - ai)
    if np.linalg.norm(aij) == 0:
        # Degenerate case: arm endpoints coincide;, fall back to ti·-tj method
        return angle_from_dot(np.dot(ti, -tj)), True

    dot_i = float(np.dot(ti, aij))    # ti toward +a_ij
    dot_j = float(np.dot(tj, -aij))   # tj toward -a_ij

    # Orientation constraint: require ti pointing toward j and tj pointing toward i
    if require_toward_line and (dot_i < toward_cos_threshold or dot_j < toward_cos_threshold):
        return float("nan"), False

    alpha_i = angle_from_dot(dot_i)
    alpha_j = angle_from_dot(dot_j)
    theta = alpha_i + alpha_j
    return theta, True

# -------------------------
# Example usage
# -------------------------

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def demo():
    """
    Minimal synthetic example with 5 particles laid out in space.
    In practice, provide real experimental coordinates and orientations for your particles.
    """
    np.random.seed(0)

    particles: List[Particle] = []
    # Create a small chain roughly along x-axis
    for k in range(5):
        center = np.array([k*12.0, 0.0 + (np.random.rand()-0.5)*2.0, 0.0])
        # mock arm exits ~ 5 nm from center along ±x
        a1 = center + np.array([+5.0, 0.0, 0.0])
        a2 = center + np.array([-5.0, 0.0, 0.0])
        # tangents roughly ±x with small noise
        t1 = _unit(np.array([+1.0, 0.1*(np.random.rand()-0.5), 0.1*(np.random.rand()-0.5)]))
        t2 = _unit(np.array([-1.0, 0.1*(np.random.rand()-0.5), 0.1*(np.random.rand()-0.5)]))
        particles.append(Particle(center, a1, a2, t1, t2))

    assigner = LinkerAssigner(particles,
                              lp=50.0,   # Polymer persistence length (nm)
                              L0=15.0,   # reference length (nm)
                              dist_cutoff_nm=20.0,
                              p_threshold=0.1,
                              port_pairing="any")
    assignments, adj = assigner.run()

    print("Assigned linkers:")
    for a in assignments:
        print(f"(i={a.i}, j={a.j}) arms=({a.arm_i}->{a.arm_j}) "
              f"theta={a.theta:.3f} rad, L={a.L:.2f} nm, P={a.prob:.3f}, "
              f"Pmax/Psecond={a.pmax_over_psecond:.2f}")

    print("\nAdjacency:")
    for i in sorted(adj.keys()):
        print(i, ":", sorted(adj[i]))

if __name__ == "__main__":
    demo()
