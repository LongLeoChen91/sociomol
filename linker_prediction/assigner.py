# -*- coding: utf-8 -*-
"""
Linker assignment engine for generalized two-armed particles
Implements the greedy assignment and constraints:
- One connection per arm/port
- No cycles across particles/assemblies
- Probability threshold stopping rule

Author: Long Chen
"""

import math
import numpy as np
from typing import List, Tuple, Dict

from .models import Particle, LinkerAssignment
from .graph import DSU
from .probability import connection_probability
from .linker_geometry import angle_between, arc_length_from_endpoints_and_angle, theta_alpha_sum

class LinkerAssigner:
    def __init__(self, particles: List[Particle],
                 lp: float = 1.5, L0: float = 20.0,
                 dist_cutoff_nm: float = 20.0, p_threshold: float = 0.1,
                 w_wlc: float = 1.0, w_L: float = 1.0, w_th: float = 1.0,
                 theta0_deg: float = 45.0,
                 theta_mode: str = "alpha_sum",            # mode for angle calculation
                 require_toward_line: bool = True,         # require tangents pointing toward the arm-line
                 toward_cos_threshold: float = 0.0,       # cosine threshold for orientation constraint
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
        
        # New energy weights
        self.w_wlc = w_wlc
        self.w_L = w_L
        self.w_th = w_th
        self.theta0_rad = math.radians(theta0_deg)

        # New configuration options
        self.theta_mode = theta_mode
        self.require_toward_line = require_toward_line
        self.toward_cos_threshold = toward_cos_threshold
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
                Pij = connection_probability(
                    Lij, th, L0=self.L0, lp=self.lp,
                    w_wlc=self.w_wlc, w_L=self.w_L, w_th=self.w_th, theta0_rad=self.theta0_rad
                )
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
                D=D_sel,
                prob=float(Pbest),
                psecond=float(Psecond),
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
