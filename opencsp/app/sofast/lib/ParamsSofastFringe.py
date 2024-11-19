"""Parameter dataclass for SofastFringe"""

from dataclasses import dataclass, field

from opencsp.app.sofast.lib.DebugOpticsGeometry import DebugOpticsGeometry
from opencsp.app.sofast.lib.ParamsMaskCalculation import ParamsMaskCalculation
from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
from opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
from opencsp.common.lib.tool import hdf5_tools


@dataclass
class ParamsSofastFringe(hdf5_tools.HDF5_IO_Abstract):
    """Parameters for SofastFringe processing calculation"""

    # Parameters
    mask: ParamsMaskCalculation = field(default_factory=ParamsMaskCalculation)
    """Parameters for calculating optic mask"""
    geometry: ParamsOpticGeometry = field(default_factory=ParamsOpticGeometry)
    """Parameters for calculating optic geometry"""

    # Debug objects
    debug_slope_solver: SlopeSolverDataDebug = field(default_factory=SlopeSolverDataDebug)
    """Debug options for slope solving"""
    debug_geometry: DebugOpticsGeometry = field(default_factory=DebugOpticsGeometry)
    """Debug options for geometry processing"""

    def save_to_hdf(self, file: str, prefix: str = ''):
        """Saves data to given HDF5 file. Data is stored in PREFIX + ParamsSofastFringe/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        self.mask.save_to_hdf(file, prefix + 'ParamsSofastFringe/')
        self.geometry.save_to_hdf(file, prefix + 'ParamsSofastFringe/')

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
        params_mask = ParamsMaskCalculation.load_from_hdf(file, prefix + 'ParamsSofastFringe/')
        params_geometry = ParamsOpticGeometry.load_from_hdf(file, prefix + 'ParamsSofastFringe/')

        # Load sofast parameters
        data = {'geometry': params_geometry, 'mask': params_mask}

        return cls(**data)
