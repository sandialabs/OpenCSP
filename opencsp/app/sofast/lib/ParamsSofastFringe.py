"""Parameter dataclass for SofastFringe
"""

from dataclasses import dataclass, field


from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


@dataclass
class ParamsSofastFringe:
    """Parameters for SofastFringe processing calculation"""

    mask_hist_thresh: float = 0.5
    mask_filt_width: int = 9
    mask_filt_thresh: int = 4
    mask_thresh_active_pixels: float = 0.05
    mask_keep_largest_area: bool = False
    geometry_params: ParamsOpticGeometry = field(default_factory=ParamsOpticGeometry)

    # Debug objects
    slope_solver_data_debug: SlopeSolverDataDebug = field(
        default_factory=SlopeSolverDataDebug
    )
    geometry_data_debug: DebugOpticsGeometry = field(
        default_factory=DebugOpticsGeometry
    )

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
