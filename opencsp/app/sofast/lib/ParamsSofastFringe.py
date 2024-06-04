"""Parameter dataclass for SofastFringe
"""

from dataclasses import dataclass, field

from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
from opencsp.common.lib.tool import hdf5_tools


@dataclass
class ParamsSofastFringe(hdf5_tools.HDF5_IO_Abstract):
    """Parameters for SofastFringe processing calculation"""

    mask_hist_thresh: float = 0.5
    mask_filt_width: int = 9
    mask_filt_thresh: int = 4
    mask_thresh_active_pixels: float = 0.05
    mask_keep_largest_area: bool = False
    geometry_params: ParamsOpticGeometry = field(default_factory=ParamsOpticGeometry)

    # Debug objects
    slope_solver_data_debug: SlopeSolverDataDebug = field(default_factory=SlopeSolverDataDebug)
    geometry_data_debug: DebugOpticsGeometry = field(default_factory=DebugOpticsGeometry)

    def save_to_hdf(self, file: str, prefix: str = ''):
        """Saves data to given HDF5 file. Data is stored in PREFIX + ParamsSofastFringe/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.mask_hist_thresh,
            self.mask_filt_width,
            self.mask_filt_thresh,
            self.mask_thresh_active_pixels,
            self.mask_keep_largest_area,
        ]
        datasets = [
            prefix + 'ParamsSofastFringe/mask_hist_thresh',
            prefix + 'ParamsSofastFringe/mask_filt_width',
            prefix + 'ParamsSofastFringe/mask_filt_thresh',
            prefix + 'ParamsSofastFringe/mask_thresh_active_pixels',
            prefix + 'ParamsSofastFringe/mask_keep_largest_area',
        ]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)

        self.geometry_params.save_to_hdf(file, prefix + 'ParamsSofastFringe/')

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ''):
        """Loads data from given file. Assumes data is stored as: PREFIX + ParamsSofastFringe/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        # Load geometry parameters
        params_geometry = ParamsOpticGeometry.load_from_hdf(file, prefix + '/ParamsSofastFringe/')

        # Load sofast parameters
        datasets = [
            prefix + 'ParamsSofastFringe/mask_hist_thresh',
            prefix + 'ParamsSofastFringe/mask_filt_width',
            prefix + 'ParamsSofastFringe/mask_filt_thresh',
            prefix + 'ParamsSofastFringe/mask_thresh_active_pixels',
            prefix + 'ParamsSofastFringe/mask_keep_largest_area',
        ]
        data = hdf5_tools.load_hdf5_datasets(datasets, file)
        data['geometry_params'] = params_geometry

        return cls(**data)
