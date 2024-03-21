from abc import ABC
from dataclasses import dataclass


@dataclass
class ParamsSlopeSolverAbstract(ABC):
    """Abstract SlopeSolver input parameters class. Contains parameters
    common to all surface types.
    """

    robust_least_squares: bool
    downsample: int
