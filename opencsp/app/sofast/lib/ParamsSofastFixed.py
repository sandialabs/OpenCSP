"""Parameters class for FixedPatternScreen class"""

from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug


class ParamsSofastFixed:
    """Parameters for FixedPatternScreenParams"""

    def __init__(self) -> 'ParamsSofastFixed':
        """Instantiates class

        Attributes
        ----------
        blob_search_thresh : float
            Pixels, search radius when finding next dot
        search_perp_axis_ratio : float
            Defines search region when searching for next dot. Ratio of length along search direction
            to perpendicular distance. Larger value equals narrower search region.
        mask_* : mask finding parameters
        *debug : debug objects
        geometry_params : ParamsOpticGeometry
            Parameters to use when processing geometry of facet
        """
        self.blob_search_thresh: float = 5.0
        self.search_perp_axis_ratio: float = 3.0
        self.mask_hist_thresh: float = 0.5
        self.mask_filt_width: int = 9
        self.mask_filt_thresh: int = 4
        self.mask_thresh_active_pixels: float = 0.05
        self.mask_keep_largest_area: bool = False

        self.slope_solver_data_debug: SlopeSolverDataDebug = SlopeSolverDataDebug()
        self.geometry_data_debug: DebugOpticsGeometry = DebugOpticsGeometry()
        self.geometry_params: ParamsOpticGeometry = ParamsOpticGeometry()
