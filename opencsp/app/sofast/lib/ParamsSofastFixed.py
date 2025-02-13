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
    mask: ParamsMaskCalculation = field(default_factory=ParamsMaskCalculation)
    """Parameters for calculating optic mask"""
    geometry: ParamsOpticGeometry = field(default_factory=ParamsOpticGeometry)
    """Parameters to use when processing geometry of facet"""

    # Debug objects
    debug_slope_solver: SlopeSolverDataDebug = field(default_factory=SlopeSolverDataDebug)
    """Debug options for slope solving"""
    debug_geometry: DebugOpticsGeometry = field(default_factory=DebugOpticsGeometry)
    """Debug options for geometry processing"""

    def save_to_hdf(self, file: str, prefix: str = ""):
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
            prefix + "ParamsSofastFixed/blob_search_thresh",
            prefix + "ParamsSofastFixed/search_perp_axis_ratio",
        ]
        ht.save_hdf5_datasets(data, datasets, file)

        self.geometry.save_to_hdf(file, prefix + "ParamsSofastFixed/")
        self.mask.save_to_hdf(file, prefix + "ParamsSofastFixed/")

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ""):
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
            prefix + "ParamsSofastFixed/blob_search_thresh",
            prefix + "ParamsSofastFixed/search_perp_axis_ratio",
        ]
        data = ht.load_hdf5_datasets(datasets, file)

        # Load geometry parameters
        params_mask = ParamsMaskCalculation.load_from_hdf(file, prefix + "ParamsSofastFixed/")
        params_geometry = ParamsOpticGeometry.load_from_hdf(file, prefix + "ParamsSofastFixed/")

        # Load sofast parameters
        data["geometry"] = params_geometry
        data["mask"] = params_mask

        return cls(**data)
