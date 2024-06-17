"""Parameters class for SofastFixed"""

from dataclasses import dataclass, field

from opencsp.app.sofast.lib.ParamsMaskCalculation import ParamsMaskCalculation
from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
import opencsp.common.lib.tool.hdf5_tools as ht


@dataclass
class ParamsSofastFixed(ht.HDF5_IO_Abstract):
    """Parameters for SofastFixed processing calculation"""

    blob_search_thresh: float = 5.0
    """Pixels, search radius when finding next dot"""
    search_perp_axis_ratio: float = 3.0
    """Defines search region when searching for next dot. Ratio of length along search direction
    to perpendicular distance. Larger value equals narrower search region."""
    mask_params: ParamsMaskCalculation = field(default_factory=ParamsMaskCalculation)
    """Parameters for calculating optic mask"""
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
        data = [self.blob_search_thresh, self.search_perp_axis_ratio]
        datasets = [
            prefix + 'ParamsSofastFixed/blob_search_thresh',
            prefix + 'ParamsSofastFixed/search_perp_axis_ratio',
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
        ]
        data = ht.load_hdf5_datasets(datasets, file)

        geometry_params = ParamsOpticGeometry.load_from_hdf(file, prefix)
        mask_params = ParamsMaskCalculation.load_from_hdf(file, prefix)

        data['geometry_params'] = geometry_params
        data['msak_params'] = mask_params

        return cls(**data)
