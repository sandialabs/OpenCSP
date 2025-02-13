"""Data class for holding output of a SlopeSolver calculation"""

from dataclasses import dataclass

from numpy import ndarray

from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


@dataclass
class SlopeSolverData:
    """Contains output data of a SlopeSolver calculation"""

    surf_coefs_facet: ndarray = None
    slope_coefs_facet: ndarray = None
    trans_alignment: TransformXYZ = None
    v_surf_points_facet: Vxyz = None
    slopes_facet_xy: ndarray = None

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given HDF5 file. Data is stored in PREFIX + SlopeSolverData/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.surf_coefs_facet,
            self.slope_coefs_facet,
            self.trans_alignment.matrix,
            self.v_surf_points_facet.data,
            self.slopes_facet_xy,
        ]
        datasets = [
            prefix + "SlopeSolverData/surf_coefs_facet",
            prefix + "SlopeSolverData/slope_coefs_facet",
            prefix + "SlopeSolverData/trans_alignment",
            prefix + "SlopeSolverData/v_surf_points_facet",
            prefix + "SlopeSolverData/slopes_facet_xy",
        ]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
