from dataclasses import dataclass
from typing import Optional
import numpy as np

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

@dataclass
class LinkerAssignment:
    i: int
    j: int
    arm_i: int  # 0 or 1
    arm_j: int  # 0 or 1
    theta: float
    L: float
    D: float                       # straight-line distance between arm endpoints (nm)
    prob: float
    psecond: Optional[float] = None  # second-best probability for (i,j) over remaining combos
    pmax_over_psecond: Optional[float] = None
