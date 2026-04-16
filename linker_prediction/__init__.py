# linker_prediction/__init__.py

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("sociomol")
except PackageNotFoundError:
    __version__ = "unknown"

from .models import Particle, LinkerAssignment
from .assigner import LinkerAssigner
from .linker_geometry import (
    unit,
    build_frame_from_z_and_trialx,
    euler_zyz_from_two_points,
    midpoint_from_two_points,
)
from .pipeline import run_prediction_pipeline

__all__ = [
    "__version__",
    "run_prediction_pipeline",
    "LinkerAssigner",
    "Particle",
    "LinkerAssignment",
    "unit",
    "build_frame_from_z_and_trialx",
    "euler_zyz_from_two_points",
    "midpoint_from_two_points",
]
