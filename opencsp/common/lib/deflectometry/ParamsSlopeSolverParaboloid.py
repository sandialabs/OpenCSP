from dataclasses import dataclass

from opencsp.common.lib.deflectometry.ParamsSlopeSolverAbstract import ParamsSlopeSolverAbstract


@dataclass
class ParamsSlopeSolverParaboloid(ParamsSlopeSolverAbstract):
    """SlopeSolver input parameters class for parabolic surface type
    """
    initial_focal_lengths_xy: tuple[float, float]
