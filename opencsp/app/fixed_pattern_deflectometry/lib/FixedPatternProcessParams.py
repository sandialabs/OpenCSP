"""Parameters class for FixedPatternScreen class"""
from opencsp.common.lib.deflectometry.GeometryProcessingParams import GeometryProcessingParams
from opencsp.common.lib.deflectometry.GeometryDataDebug import GeometryDataDebug
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug


class FixedPatternProcessParams:
    """Parameters for FixedPatternScreenParams"""

    def __init__(self) -> 'FixedPatternProcessParams':
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
        geometry_params : GeometryProcessingParams
            Parameters to use when processing geometry of facet
        """
        self.blob_search_thresh: float = 5.
        self.search_perp_axis_ratio: float = 3.
        self.mask_hist_thresh: float = 0.5
        self.mask_filt_width: int = 9
        self.mask_filt_thresh: int = 4
        self.mask_thresh_active_pixels: float = 0.05
        self.mask_keep_largest_area: bool = False

        self.slope_solver_data_debug: SlopeSolverDataDebug = SlopeSolverDataDebug()
        self.geometry_data_debug: GeometryDataDebug = GeometryDataDebug()
        self.geometry_params: GeometryProcessingParams = GeometryProcessingParams()
