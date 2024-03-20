from dataclasses import dataclass

from opencsp.common.lib.deflectometry.ParamsSlopeSolverAbstract import ParamsSlopeSolverAbstract


@dataclass
class ParamsSlopeSolverPlano(ParamsSlopeSolverAbstract):
    """SlopeSolver input parameters class for plano (perfectly flat) surface type
    """
