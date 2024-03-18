from dataclasses import dataclass

from opencsp.common.lib.deflectometry.ParamsSlopeSolver import ParamsSlopeSolver


@dataclass
class ParamsSlopeSolverPlano(ParamsSlopeSolver):
    """SlopeSolver input parameters class for plano (perfectly flat) surface type
    """
    robust_least_squares: bool
    downsample: int
