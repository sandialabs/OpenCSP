import numpy as np
from numpy import ndarray

from opencsp.app.sofast.lib.PatternSofastFixed import PatternSofastFixed
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


class DotLocationsFixedPattern:
    """Class that holds locations of dots for fixed pattern deflectometry."""

    def __init__(self, x_dot_index: ndarray, y_dot_index: ndarray, xyz_dot_loc: ndarray) -> "DotLocationsFixedPattern":
        """Instantiates class with xy indices and xyz points.

        Parameters
        ----------
        x_dot_index : ndarray
            Shape (M,) array holding x dot index values, must be in increasing order
        y_dot_index : ndarray
            Shape (N,) array holding y dot index values, must be in increasing order
        xyz_dot_loc : ndarray
            Shape (N, M, 3) array holding xyz locations of dots in screen coordinates
        """
        if x_dot_index.size != xyz_dot_loc.shape[1]:
            raise ValueError(f"X dimensions do not match: {x_dot_index.size} and {xyz_dot_loc.shape}")
        if y_dot_index.size != xyz_dot_loc.shape[0]:
            raise ValueError(f"Y dimensions do not match: {y_dot_index.size} and {xyz_dot_loc.shape}")

        # Store data
        self.x_dot_index = x_dot_index
        self.y_dot_index = y_dot_index

        self.xyz_dot_loc = xyz_dot_loc

        # Calculate extents
        self.nx = x_dot_index.size
        self.ny = y_dot_index.size
        self.dot_extent = (x_dot_index.min(), x_dot_index.max(), y_dot_index.min(), y_dot_index.max())

        self.x_min = self.x_dot_index.min()
        self.x_offset = -self.x_min
        self.x_max = self.x_dot_index.max()

        self.y_min = self.y_dot_index.min()
        self.y_offset = -self.y_min
        self.y_max = self.y_dot_index.max()

    @classmethod
    def from_projection_and_display(
        cls, fixed_pattern_projection: PatternSofastFixed, display: Display
    ) -> "DotLocationsFixedPattern":
        """Instantiates a DotLocationsFixedPattern from a PatternSofastFixed object
        and a display object. This is used as a convenience if a Display calibration has
        already been done for a screen setup."""
        # Calculate xy points in screen fractions
        x_frac = fixed_pattern_projection.x_locs_frac
        y_frac = fixed_pattern_projection.y_locs_frac
        x_frac_mat, y_frac_mat = np.meshgrid(x_frac, y_frac)
        xy_pts_frac = Vxy((x_frac_mat, y_frac_mat))

        # Calculate xyz locations
        xyz_dot_loc = display.interp_func(xy_pts_frac)
        x = xyz_dot_loc.x.reshape((fixed_pattern_projection.ny, fixed_pattern_projection.nx, 1))
        y = xyz_dot_loc.y.reshape((fixed_pattern_projection.ny, fixed_pattern_projection.nx, 1))
        z = xyz_dot_loc.z.reshape((fixed_pattern_projection.ny, fixed_pattern_projection.nx, 1))
        xyz_dot_loc_mat = np.concatenate((x, y, z), axis=2)

        return cls(fixed_pattern_projection.x_indices, fixed_pattern_projection.y_indices, xyz_dot_loc_mat)

    def xy_indices_to_screen_coordinates(self, pts_idxs: Vxy) -> Vxyz:
        """Convertes xy point indices to xyz screen coordinates.

        Parameters
        ----------
        pts_idxs : Vxy
            Input xy point index

        Returns
        -------
        Vxyz
            Output xyz screen coordinate
        """
        pts_idxs_x = pts_idxs.x + self.x_offset
        pts_idxs_y = pts_idxs.y + self.y_offset

        pts_x = self.xyz_dot_loc[pts_idxs_y, pts_idxs_x, 0]
        pts_y = self.xyz_dot_loc[pts_idxs_y, pts_idxs_x, 1]
        pts_z = self.xyz_dot_loc[pts_idxs_y, pts_idxs_x, 2]

        return Vxyz((pts_x, pts_y, pts_z))

    def save_to_hdf(self, file: str) -> None:
        """
        Saves to HDF file

        Parameters
        ----------
        file : string
            HDF5 file to save

        """
        data = [self.x_dot_index, self.y_dot_index, self.xyz_dot_loc]
        datasets = [
            "DotLocationsFixedPattern/x_dot_index",
            "DotLocationsFixedPattern/y_dot_index",
            "DotLocationsFixedPattern/xyz_dot_loc",
        ]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str) -> "DotLocationsFixedPattern":
        """
        Loads from HDF file

        Parameters
        ----------
        file : string
            HDF5 file to load

        """
        datasets = [
            "DotLocationsFixedPattern/x_dot_index",
            "DotLocationsFixedPattern/y_dot_index",
            "DotLocationsFixedPattern/xyz_dot_loc",
        ]
        data = hdf5_tools.load_hdf5_datasets(datasets, file)
        return cls(**data)
