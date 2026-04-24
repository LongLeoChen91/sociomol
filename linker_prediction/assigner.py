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
from scipy.spatial import cKDTree

from .models import Particle, LinkerAssignment
from .graph import DSU
from .probability import connection_probability
from .linker_geometry import angle_between, arc_length_from_endpoints_and_angle, theta_alpha_sum

class LinkerAssigner:
    def __init__(self, particles: List[Particle],
                 lp: float = 1.5, L0: float = 20.0,
                 dist_cutoff_nm: float = 20.0, p_threshold: float = 0.1,
                 w_wlc: float = 1.0, w_L: float = 1.0, w_th: float = 1.0,
                 w_L_sq: float = 0.0, w_th_sq: float = 0.0,
                 theta0_deg: float = 90.0,
                 l_ideal_nm: float = 20.0,
                 l_std_nm: float = 10.0,
                 theta_std_deg: float = 45.0,
                 theta_mode: str = "alpha_sum",            # mode for angle calculation
                 max_bending_deg: float = 180.0,      # max allowed total bending angle
                 port_pairing: str = "any"):              # "any" or "complement"
        """
        require_toward_line : bool, default True
            If True, enforce that both arm tangents point sufficiently toward
            the arm-line direction (a_ij = unit(aj - ai)):
                dot(t_i, +a_ij) >= toward_cos_threshold  and
                dot(t_j, -a_ij) >= toward_cos_threshold.
            Set False to skip this feasibility check.

        max_bending_deg : float, default 180.0
            Maximum allowed total bending angle (alpha_i + alpha_j). 
            Typical values:
                180.0 ~ arms must point generally forward
                120.0 ~ strict forward cone
                360.0 ~ lenient, no orientation checking
                
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
        self.w_L_sq = w_L_sq
        self.w_th_sq = w_th_sq
        self.theta0_rad = math.radians(theta0_deg)
        self.L_ideal = l_ideal_nm
        self.L_std = l_std_nm
        self.theta_std_rad = math.radians(theta_std_deg)

        # New configuration options
        self.theta_mode = theta_mode
        self.max_bending_deg = max_bending_deg
        self.port_pairing = port_pairing

        self.arm_used = np.zeros((self.N, 2), dtype=bool)
        self.adj: Dict[int, List[int]] = {i: [] for i in range(self.N)}
        self.assignments: List[LinkerAssignment] = []

    def _pairwise_candidates(self) -> List[Tuple[int,int]]:
        """Find candidate node pairs if any arm-arm distance < cutoff using KDTree for O(N log N) performance."""
        if self.N == 0:
            return []
            
        # Build an array of all arm endpoints: shape (2N, 3)
        # Even indices (2i) correspond to arm 0, odd indices (2i+1) to arm 1
        pts = np.empty((2 * self.N, 3), dtype=float)
        for i in range(self.N):
            pts[2*i] = self.particles[i].a1
            pts[2*i+1] = self.particles[i].a2
            
        tree = cKDTree(pts)
        
        # Query all pairs within dist_cutoff
        # raw_pairs contains tuples of indices within the `pts` array (u, v)
        raw_pairs = tree.query_pairs(r=self.dist_cutoff)
        
        cands = set()
        for u, v in raw_pairs:
            i, j = u // 2, v // 2
            if i != j:
                # Ensure ordered tuples to match legacy (i < j)
                cands.add(tuple(sorted((i, j))))

        # Post-filter: when port_pairing is "complement", ensure at least one
        # allowed arm combination (0↔1 or 1↔0) has distance < cutoff.
        # Without this, pairs where only a forbidden combo (0↔0 or 1↔1) is
        # close would leak into the candidate set, inflating false positives.
        if self.port_pairing == "complement":
            cutoff_sq = self.dist_cutoff ** 2
            filtered = []
            for (i, j) in sorted(cands):
                d01_sq = np.sum((self.particles[i].a1 - self.particles[j].a2) ** 2)
                d10_sq = np.sum((self.particles[i].a2 - self.particles[j].a1) ** 2)
                if d01_sq < cutoff_sq or d10_sq < cutoff_sq:
                    filtered.append((i, j))
            return filtered

        return sorted(list(cands))

    def _compute_all_probs_batch(self, candidates: List[Tuple[int,int]]) -> Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Vectorized computation of theta, L, P for all candidate pairs in a single Numpy operation burst."""
        if not candidates:
            return {}
            
        E = len(candidates)
        # Gather data, shape (E, 2, 3) 
        A_i = np.empty((E, 2, 3), dtype=np.float64)
        T_i = np.empty((E, 2, 3), dtype=np.float64)
        A_j = np.empty((E, 2, 3), dtype=np.float64)
        T_j = np.empty((E, 2, 3), dtype=np.float64)
        
        for idx, (i, j) in enumerate(candidates):
            pi, pj = self.particles[i], self.particles[j]
            A_i[idx, 0] = pi.a1; A_i[idx, 1] = pi.a2
            T_i[idx, 0] = pi.t1; T_i[idx, 1] = pi.t2
            A_j[idx, 0] = pj.a1; A_j[idx, 1] = pj.a2
            T_j[idx, 0] = pj.t1; T_j[idx, 1] = pj.t2

        # Expand shapes to (E, 2, 2, 3) where dim 1 is arm_i, dim 2 is arm_j
        A_i = A_i[:, :, None, :]
        A_j = A_j[:, None, :, :]
        T_i = T_i[:, :, None, :]
        T_j = T_j[:, None, :, :]
        
        # Compute D = |A_j - A_i|
        diff = A_j - A_i
        D = np.linalg.norm(diff, axis=-1)
        
        # Compute angles robustly
        n = np.where(D < 1e-12, 1.0, D)[..., None]
        aij = diff / n
        
        dot_i = np.sum(T_i * aij, axis=-1)
        dot_j = np.sum(T_j * (-aij), axis=-1)
        
        theta = np.zeros((E, 2, 2), dtype=np.float64)
        feasible = np.ones((E, 2, 2), dtype=bool)
        
        if getattr(self, "theta_mode", "alpha_sum") == "alpha_sum":
            thresh = math.cos(math.radians(getattr(self, "max_bending_deg", 180.0) / 2.0))
            feasible = (dot_i >= thresh) & (dot_j >= thresh)
            
            alpha_i = np.arccos(np.clip(dot_i, -1.0, 1.0))
            alpha_j = np.arccos(np.clip(dot_j, -1.0, 1.0))
            theta = alpha_i + alpha_j
            
            # fallback for degenerate D
            degen = D < 1e-12
            if np.any(degen):
                dot_fallback = np.sum(T_i * (-T_j), axis=-1)
                theta[degen] = np.arccos(np.clip(dot_fallback[degen], -1.0, 1.0))
                feasible[degen] = True
        else:
            # legacy tangent_tangent
            dot_fallback = np.sum(T_i * T_j, axis=-1)
            theta = np.arccos(np.clip(dot_fallback, -1.0, 1.0))

        # Port pairing constraint
        if self.port_pairing == "complement":
            feasible[:, 0, 0] = False
            feasible[:, 1, 1] = False
            
        theta[~feasible] = 0.0
        
        # Compute arc length L
        eps = 1e-8
        half = 0.5 * theta
        s = np.sin(half)
        
        L = np.zeros_like(D)
        small = np.abs(s) < eps
        L[small] = D[small]
        
        normal = ~small
        if np.any(normal):
            r = D[normal] / (2.0 * s[normal])
            invalid_r = r <= 0
            
            # map valid/invalid to the overall shape safely
            full_invalid = np.zeros_like(feasible)
            full_invalid[normal] = invalid_r
            feasible[full_invalid] = False
            
            r[invalid_r] = 1.0 # purely to avoid nan in next step
            L[normal] = theta[normal] * np.abs(r)
            
        L[~feasible] = 0.0
        
        # Build probability cache using exact legacy python float models
        prob_cache = {}
        for idx, (i, j) in enumerate(candidates):
            t_ij = theta[idx]
            l_ij = L[idx]
            p_ij = np.zeros((2, 2), dtype=np.float64)
            for arm_i in (0, 1):
                for arm_j in (0, 1):
                    if feasible[idx, arm_i, arm_j]:
                        l_val = l_ij[arm_i, arm_j]
                        t_val = t_ij[arm_i, arm_j]
                        p_ij[arm_i, arm_j] = connection_probability(
                            float(l_val), float(t_val), L0=self.L0, lp=self.lp,
                            w_wlc=self.w_wlc, w_L=self.w_L, w_th=self.w_th, 
                            w_L_sq=self.w_L_sq, w_th_sq=self.w_th_sq,
                            theta0_rad=self.theta0_rad,
                            L_ideal=self.L_ideal, L_std=self.L_std,
                            theta_std_rad=self.theta_std_rad
                        )
            prob_cache[(i, j)] = (t_ij, l_ij, p_ij)
            
        return prob_cache


    def run(self):
        # Build DSU for no-cycles constraint
        dsu = DSU(self.N)

        # Precompute candidates and probabilities
        candidates = self._pairwise_candidates()

        # Batch-compute all math in vector matrices
        prob_cache = self._compute_all_probs_batch(candidates)
        
        # Max-Heap for greedy selection
        # Format: (-Pval, tiebreaker, i, j, arm_i, arm_j)
        import heapq
        heap = []
        counter = 0

        for (i, j) in candidates:
            th_ij, L_ij, P_ij = prob_cache[(i, j)]
            for arm_i in (0, 1):
                for arm_j in (0, 1):
                    Pval = P_ij[arm_i, arm_j]
                    if Pval > 0.0:
                        heapq.heappush(heap, (-Pval, counter, i, j, arm_i, arm_j))
                        counter += 1

        # Greedy loop
        while heap:
            neg_Pval, _, i, j, arm_i, arm_j = heapq.heappop(heap)
            Pbest = -neg_Pval

            # Stopping condition: top of max-heap is below threshold
            if Pbest < self.p_threshold:
                break

            # Check if either arm is already used
            if self.arm_used[i, arm_i] or self.arm_used[j, arm_j]:
                continue
                
            # Skip if already connected i-j
            if j in self.adj[i]:
                continue

            # Check cycle constraint via DSU on node-level
            if not dsu.union(i, j):
                # would make a cycle; forbid this combination
                continue

            # Compute Psecond iteratively without mutating cache
            # Psecond is the max probability among remaining unused arm combinations for (i,j)
            theta_ij, L_ij, P_ij = prob_cache[(i, j)]
            Psecond = 0.0
            
            for ai in (0, 1):
                for aj in (0, 1):
                    if ai == arm_i and aj == arm_j:
                        continue
                        
                    # Skip if the alternative arm is already occupied by a prior edge
                    if self.arm_used[i, ai] or self.arm_used[j, aj]:
                        continue
                        
                    Pval2 = P_ij[ai, aj]
                    if Pval2 > Psecond:
                        Psecond = float(Pval2)

            ratio = (Pbest / Psecond) if Psecond > 0 else float('inf')

            # Accept the edge tracking states
            self.adj[i].append(j)
            self.adj[j].append(i)
            self.arm_used[i, arm_i] = True
            self.arm_used[j, arm_j] = True

            # Compute D on-the-fly for the chosen arms:
            ai_sel = self.particles[i].arm_point(arm_i)   # endpoint on i for the chosen arm
            aj_sel = self.particles[j].arm_point(arm_j)   # endpoint on j for the chosen arm
            D_sel = float(np.linalg.norm(aj_sel - ai_sel))

            # Save assignment
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

        return self.assignments, self.adj
