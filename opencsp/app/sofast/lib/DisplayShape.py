import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


class DisplayShape:
    """Representation of a screen/projector for deflectometry."""

    def __init__(
        self,
        v_cam_screen_screen: Vxyz,
        r_screen_cam: Rotation,
        grid_data: dict,
        name: str = '',
    ) -> 'DisplayShape':
        """
        Instantiates deflectometry display representation.

        Parameters
        ----------
        v_cam_screen_screen : Vxyz
            Translation vector from camera to screen in screen coordinates.
        r_screen_cam : Rotation
            Rotation vector from camera to screen coordinates.
        grid_data : dict
            DisplayShape distortion data. Must contain the field "screen_model"
            that defines distortion model and necessary input data.

            1) Rectangular 2D
                - Description: Model with no distortion (useful for LCD
                screens, etc.).

                - Needs the following fields.
                    "screen_model" : str
                        'rectangular2D'

            2) Distorted 2D
                - Description: Model that assumes screen ia perfectly flat
                2D surface (useful for projector system with very flat
                wall).

                - Needs the following fields.
                    "screen_model" : str
                        'distorted2D'
                    "xy_screen_fraction" : Vxy
                        XY screen points in fractional screens.
                    "xy_screen_coords" : Vxy
                        XY screen points in meters (screen coordinates).

            3) Distorted 3D
                - Description: Model that can completely define the 3D
                shape of a distorted screen in 3D.

                - Needs the following fields.
                    "screen_model" : str
                        'distorted3D'
                    "xy_screen_fraction" : Vxy
                        XY screen points in fractional screens.
                    "xyz_screen_coords" : Vxyz
                        XYZ screen points in meters (screen coordinates).

        name : str, optional
            The name of the calibrated display.

        """

        # Rotation matrices
        self.r_screen_cam = r_screen_cam
        self.r_cam_screen = self.r_screen_cam.inv()

        # Translation vectors
        self.v_cam_screen_screen = v_cam_screen_screen
        self.v_cam_screen_cam = self.v_cam_screen_screen.rotate(self.r_screen_cam)

        # Save display model name
        self.name = name

        # Instantiate fractional screen to screen coordinate function
        self.grid_data = grid_data
        self._init_interp_func()

    def __repr__(self):
        return 'DisplayShape: { ' + self.name + ' }'

    def _init_interp_func(self):
        # Rectangular (undistorted) screen model
        if self.grid_data['screen_model'] == 'rectangular2D':
            self.interp_func = self._interp_func_rectangular2D

        # Distorted screen model
        elif self.grid_data['screen_model'] == 'distorted2D':
            # Create X/Y interpolation function
            points = self.grid_data['xy_screen_fraction']  # Vxy, fractional screens
            values = self.grid_data['xy_screen_coords']  # Vxy, screen coordinates

            # Check input types
            if not isinstance(values, Vxy):
                raise ValueError('Values must be type Vxy for 2D distorted model.')
            if len(points) != len(values):
                raise ValueError('Input points and values must be same length.')

            func_xy = LinearNDInterpolator(points.data.T, values.data.T)

            self.interp_func = lambda Vuv: self._interp_func_2D(Vuv, func_xy)

        elif self.grid_data['screen_model'] == 'distorted3D':
            # Create X/Y/Z interpolation function
            points = self.grid_data['xy_screen_fraction']  # Vxy, fractional screens
            values = self.grid_data['xyz_screen_coords']  # Vxyz, screen coordinates

            # Check input types
            if not isinstance(values, Vxyz):
                raise ValueError('Values must be type Vxyz for 3D distorted model.')
            if len(points) != len(values):
                raise ValueError('Input points and values must be same length.')

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
        xm = (uv_display_pts.x - 0.5) * self.grid_data['screen_x']  # meters
        ym = (uv_display_pts.y - 0.5) * self.grid_data['screen_y']  # meters
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
    def load_from_hdf(cls, file: str):
        """
        Loads from HDF file

        Parameters
        ----------
        file : string
            HDF5 file to load

        """
        # Load grid data
        grid_data = hdf5_tools.load_hdf5_datasets(['DisplayShape/screen_model'], file)

        # Rectangular
        if grid_data['screen_model'] == 'rectangular2D':
            datasets = ['DisplayShape/screen_x', 'DisplayShape/screen_y']
            grid_data.update(hdf5_tools.load_hdf5_datasets(datasets, file))

        # Distorted 2D
        elif grid_data['screen_model'] == 'distorted2D':
            datasets = [
                'DisplayShape/xy_screen_fraction',
                'DisplayShape/xy_screen_coords',
            ]
            grid_data.update(hdf5_tools.load_hdf5_datasets(datasets, file))
            grid_data['xy_screen_fraction'] = Vxy(grid_data['xy_screen_fraction'])
            grid_data['xy_screen_coords'] = Vxy(grid_data['xy_screen_coords'])

        # Distorted 3D
        elif grid_data['screen_model'] == 'distorted3D':
            datasets = [
                'DisplayShape/xy_screen_fraction',
                'DisplayShape/xyz_screen_coords',
            ]
            grid_data.update(hdf5_tools.load_hdf5_datasets(datasets, file))
            grid_data['xy_screen_fraction'] = Vxy(grid_data['xy_screen_fraction'])
            grid_data['xyz_screen_coords'] = Vxyz(grid_data['xyz_screen_coords'])

        else:
            raise ValueError(f'Model, {grid_data["screen_model"]}, not supported.')

        # Load display parameters
        datasets = [
            'DisplayShape/rvec_screen_cam',
            'DisplayShape/tvec_cam_screen_screen',
            'DisplayShape/name',
        ]
        data = hdf5_tools.load_hdf5_datasets(datasets, file)

        # Return display object
        kwargs = {
            'r_screen_cam': Rotation.from_rotvec(data['rvec_screen_cam']),
            'v_cam_screen_screen': Vxyz(data['tvec_cam_screen_screen']),
            'name': data['name'],
            'grid_data': grid_data,
        }
        return cls(**kwargs)

    def save_to_hdf(self, file: str):
        """
        Saves to HDF file

        Parameters
        ----------
        file : string
            HDF5 file to save

        """

        # Get "grid data" datset names and data
        datasets = []
        data = []
        for dataset in self.grid_data.keys():
            datasets.append('DisplayShape/' + dataset)
            if isinstance(self.grid_data[dataset], Vxy) or isinstance(
                self.grid_data[dataset], Vxyz
            ):
                data.append(self.grid_data[dataset].data)
            else:
                data.append(self.grid_data[dataset])

        # Screen data
        datasets += [
            'DisplayShape/rvec_screen_cam',
            'DisplayShape/tvec_cam_screen_screen',
            'DisplayShape/name',
        ]
        data += [
            self.r_screen_cam.as_rotvec(),
            self.v_cam_screen_screen.data,
            self.name,
        ]

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
