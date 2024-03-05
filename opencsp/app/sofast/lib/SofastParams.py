"""Parameter dataclass for SOFAST
"""
from dataclasses import dataclass, field


from opencsp.common.lib.deflectometry.GeometryProcessingParams import GeometryProcessingParams
from opencsp.common.lib.deflectometry.GeometryDataDebug import GeometryDataDebug
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug

@dataclass
class SofastParams:
    """Parameters for SOFAST processing calculation"""
    mask_hist_thresh: float = 0.5
    mask_filt_width: int = 9
    mask_filt_thresh: int = 4
    mask_thresh_active_pixels: float = 0.05
    mask_keep_largest_area: bool = False

    slope_solver_data_debug: SlopeSolverDataDebug = field(default_factory=SlopeSolverDataDebug)
    geometry_data_debug: GeometryDataDebug = field(default_factory=GeometryDataDebug)
    geometry_params: GeometryProcessingParams = field(default_factory=GeometryProcessingParams)
