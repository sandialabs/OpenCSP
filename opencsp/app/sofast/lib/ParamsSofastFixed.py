"""Parameters class for FixedPatternScreen class"""

from dataclasses import dataclass, field

from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
import opencsp.common.lib.tool.hdf5_tools as ht


@dataclass
class ParamsSofastFixed(ht.HDF5_IO_Abstract):
    """Parameters for FixedPatternScreenParams"""

    blob_search_thresh: float = 5.0
    """Pixels, search radius when finding next dot"""
    search_perp_axis_ratio: float = 3.0
    """Defines search region when searching for next dot. Ratio of length along search direction
    to perpendicular distance. Larger value equals narrower search region."""
    mask_hist_thresh: float = 0.5
    """Defines threshold to use when calculating optic mask. Uses a histogram of pixel values
    of the mask difference image (light image - dark image). This is the fraction of the way
    from the first histogram peak (most common dark pixel value) to the the last histogram peak
    (most common light pixel value)."""
    mask_filt_width: int = 9
    """Side length of square kernel used to filter mask image"""
    mask_filt_thresh: int = 4
    """Threshold (minimum number of active pixels) to use when removing small active mask areas."""
    mask_thresh_active_pixels: float = 0.05
    """If number of active mask pixels is below this fraction of total image pixels, thow error."""
    mask_keep_largest_area: bool = False
    """Flag to apply processing step that keeps only the largest mask area"""
    geometry_params: ParamsOpticGeometry = field(default_factory=ParamsOpticGeometry)
    """Parameters to use when processing geometry of facet"""

    # Debug objects
    slope_solver_data_debug: SlopeSolverDataDebug = field(default_factory=SlopeSolverDataDebug)
    """Debug options for slope solving"""
    geometry_data_debug: DebugOpticsGeometry = field(default_factory=DebugOpticsGeometry)
    """Debug options for geometry processing"""

    def save_to_hdf(self, file: str, prefix: str = ''):
        """Saves data to given HDF5 file. Data is stored in PREFIX + ParamsSofastFixed/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.blob_search_thresh,
            self.search_perp_axis_ratio,
            self.mask_hist_thresh,
            self.mask_filt_width,
            self.mask_filt_thresh,
            self.mask_thresh_active_pixels,
            self.mask_keep_largest_area,
        ]
        datasets = [
            prefix + 'ParamsSofastFixed/blob_search_thresh',
            prefix + 'ParamsSofastFixed/search_perp_axis_ratio',
            prefix + 'ParamsSofastFixed/mask_hist_thresh',
            prefix + 'ParamsSofastFixed/mask_filt_width',
            prefix + 'ParamsSofastFixed/mask_filt_thresh',
            prefix + 'ParamsSofastFixed/mask_thresh_active_pixels',
            prefix + 'ParamsSofastFixed/mask_keep_largest_area',
        ]
        ht.save_hdf5_datasets(data, datasets, file)

        self.geometry_params.save_to_hdf(file, prefix + 'ParamsSofastFixed/')

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ''):
        """Loads data from given file. Assumes data is stored as: PREFIX + ParamsSofastFixed/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        datasets = [
            prefix + 'ParamsSofastFixed/blob_search_thresh',
            prefix + 'ParamsSofastFixed/search_perp_axis_ratio',
            prefix + 'ParamsSofastFixed/mask_hist_thresh',
            prefix + 'ParamsSofastFixed/mask_filt_width',
            prefix + 'ParamsSofastFixed/mask_filt_thresh',
            prefix + 'ParamsSofastFixed/mask_thresh_active_pixels',
            prefix + 'ParamsSofastFixed/mask_keep_largest_area',
        ]
        data = ht.load_hdf5_datasets(datasets, file)

        # TODO: should load from HDF file
        geometry_params = ParamsOpticGeometry()

        data['geometry_params'] = geometry_params

        return cls(**data)
