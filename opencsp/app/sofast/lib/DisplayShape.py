import numpy as np
from scipy.interpolate import LinearNDInterpolator

from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as h5


class DisplayShape(h5.HDF5_IO_Abstract):
    """Representation of a screen/projector for deflectometry."""

    def __init__(self, grid_data: dict, name: str = "") -> "DisplayShape":
        """
        Instantiates deflectometry display representation.

        Parameters
        ----------
        grid_data : dict
            Contains different data depending on the model being used.

            - Rectangular 2D
                - Description: Model with no distortion (useful for LCD
                screens, etc.).

                - Needs the following fields.
                    - "screen_model" : str
                        - 'rectangular2D'
                    - "screen_x" : float
                        - Screen dimension in x
                    - "screen_y" : float
                        - Screen dimension in y

            - Distorted 2D
                - Description: Model that assumes screen ia perfectly flat
                2D surface (useful for projector system with very flat
                wall).

                - Needs the following fields.
                    - "screen_model" : str
                        - 'distorted2D'
                    - "xy_screen_fraction" : Vxy
                        - XY screen points in fractional screens.
                    - "xy_screen_coords" : Vxy
                        - XY screen points in meters (screen coordinates).

            - Distorted 3D
                - Description: Model that can completely define the 3D
                shape of a distorted screen in 3D.

                - Needs the following fields.
                    - "screen_model" : str
                        - 'distorted3D'
                    - "xy_screen_fraction" : Vxy
                        - XY screen points in fractional screens.
                    - "xyz_screen_coords" : Vxyz
                        - XYZ screen points in meters (screen coordinates).

        name : str, optional
            The name of the calibrated display.
        """
        # Save display model name
        self.name = name

        # Instantiate fractional screen to screen coordinate function
        self.grid_data = grid_data
        self._init_interp_func()

    def __repr__(self):
        return "DisplayShape: { " + self.name + " }"

    def _init_interp_func(self):
        # Rectangular (undistorted) screen model
        if self.grid_data["screen_model"] == "rectangular2D":
            self.interp_func = self._interp_func_rectangular2D

        # Distorted screen model
        elif self.grid_data["screen_model"] == "distorted2D":
            # Create X/Y interpolation function
            points = self.grid_data["xy_screen_fraction"]  # Vxy, fractional screens
            values = self.grid_data["xy_screen_coords"]  # Vxy, screen coordinates

            # Check input types
            if not isinstance(values, Vxy):
                raise ValueError("Values must be type Vxy for 2D distorted model.")
            if len(points) != len(values):
                raise ValueError("Input points and values must be same length.")

            func_xy = LinearNDInterpolator(points.data.T, values.data.T)

            self.interp_func = lambda Vuv: self._interp_func_2D(Vuv, func_xy)

        elif self.grid_data["screen_model"] == "distorted3D":
            # Create X/Y/Z interpolation function
            points = self.grid_data["xy_screen_fraction"]  # Vxy, fractional screens
            values = self.grid_data["xyz_screen_coords"]  # Vxyz, screen coordinates

            # Check input types
            if not isinstance(values, Vxyz):
                raise ValueError("Values must be type Vxyz for 3D distorted model.")
            if len(points) != len(values):
                raise ValueError("Input points and values must be same length.")

            func_xyz = LinearNDInterpolator(points.data.T, values.data.T)

            self.interp_func = lambda Vuv: self._interp_func_3D(Vuv, func_xyz)

        else:
            raise ValueError(f'Unknown screen model: {self.grid_data["screen_model"]}')

    def _interp_func_rectangular2D(self, uv_display_pts: Vxy) -> Vxyz:
        """
        Rectangular (undistorted) screen model

        Parameters
        ----------
        uv_display_pts : Vxy
            Input XY points in fractional screens.

        Returns
        -------
        Vxyz
            XYZ points in display coordinates.
        """
        xm = (uv_display_pts.x - 0.5) * self.grid_data["screen_x"]  # meters
        ym = (uv_display_pts.y - 0.5) * self.grid_data["screen_y"]  # meters
        zm = np.zeros(xm.shape)  # meters
        return Vxyz((xm, ym, zm))  # meters, display coordinates

    def _interp_func_2D(self, uv_display_pts: Vxy, func_xy) -> Vxyz:
        """
        Distorted screen model

        Parameters
        ----------
        uv_display_pts : Vxy
            Length N input XY screen points, fractional screens.
        func_xy : LinearNDInterpolator object.
            Interpolant function from screen widths to display coordinates.

        Returns
        -------
        Vxyz
            XYZ points in display coordinates.
        """
        xy = func_xy(uv_display_pts.x, uv_display_pts.y).T  # (2, N) ndarray meters
        zm = np.zeros((1, len(uv_display_pts)))  # (1, N) ndarray meters
        return Vxyz(np.concatenate((xy, zm)))  # meters, display coordinates

    def _interp_func_3D(self, uv_display_pts: Vxy, func_xyz) -> Vxyz:
        """
        Distorted screen model

        Parameters
        ----------
        uv_display_pts : Vxy
            Length N input XY screen points, fractional screens.
        func_xyz : Scattered interpolant
            Interpolant function from screen widths to display coordinates.

        Returns
        -------
        Vxyz
            XYZ points in display coordinates.
        """
        xyz = func_xyz(uv_display_pts.x, uv_display_pts.y).T  # (3, N) ndarray meters
        return Vxyz(xyz)  # meters, display coordinates

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = "") -> "DisplayShape":
        """Loads data from given file. Assumes data is stored as: PREFIX + DisplayShape/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        # Load grid data
        datasets = [prefix + "DisplayShape/screen_model", prefix + "DisplayShape/name"]
        data = h5.load_hdf5_datasets(datasets, file)

        # Rectangular
        if data["screen_model"] == "rectangular2D":
            datasets = [prefix + "DisplayShape/screen_x", prefix + "DisplayShape/screen_y"]
            grid_data = h5.load_hdf5_datasets(datasets, file)

        # Distorted 2D
        elif data["screen_model"] == "distorted2D":
            datasets = [prefix + "DisplayShape/xy_screen_fraction", prefix + "DisplayShape/xy_screen_coords"]
            grid_data = h5.load_hdf5_datasets(datasets, file)
            grid_data["xy_screen_fraction"] = Vxy(grid_data["xy_screen_fraction"])
            grid_data["xy_screen_coords"] = Vxy(grid_data["xy_screen_coords"])

        # Distorted 3D
        elif data["screen_model"] == "distorted3D":
            datasets = [prefix + "DisplayShape/xy_screen_fraction", prefix + "DisplayShape/xyz_screen_coords"]
            grid_data = h5.load_hdf5_datasets(datasets, file)
            grid_data["xy_screen_fraction"] = Vxy(grid_data["xy_screen_fraction"])
            grid_data["xyz_screen_coords"] = Vxyz(grid_data["xyz_screen_coords"])

        else:
            raise ValueError(f'Model, {data["screen_model"]}, not supported.')

        grid_data.update({"screen_model": data["screen_model"]})
        # Return display object
        kwargs = {"name": data["name"], "grid_data": grid_data}
        return cls(**kwargs)

    def save_to_hdf(self, file: str, prefix: str = "") -> None:
        """Saves data to given file. Data is stored as: PREFIX + DisplayShape/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        # Get "grid data" datset names and data
        datasets = []
        data = []
        for dataset in self.grid_data.keys():
            datasets.append(prefix + "DisplayShape/" + dataset)
            if isinstance(self.grid_data[dataset], (Vxy, Vxyz)):
                data.append(self.grid_data[dataset].data)
            else:
                data.append(self.grid_data[dataset])

        # Add name
        datasets.append(prefix + "DisplayShape/name")
        data.append(self.name)

        # Save data
        h5.save_hdf5_datasets(data, datasets, file)
