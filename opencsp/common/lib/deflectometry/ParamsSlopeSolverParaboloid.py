from dataclasses import dataclass


@dataclass
class ParamsSlopeSolverParaboloid:
    """SlopeSolver input parameters class for parabolic surface type
    """
    initial_focal_lengths_xy: tuple[float, float]
    robust_least_squares: bool
    downsample: int
