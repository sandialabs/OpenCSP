import numpy as np
from scipy.spatial.transform import Rotation

import opencsp.common.lib.deflectometry.slope_fitting_2d as sf2
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import save_hdf5_datasets, load_hdf5_datasets


class Surface2DParabolic(Surface2DAbstract):
    """Representation of 2D fit parabolic surface."""

    def __init__(self, initial_focal_lengths_xy: tuple[float, float], robust_least_squares: bool, downsample: int):
        """
        Representation of 2D fit parabolic surface.

        Parameters
        ----------
        initial_focal_lengths_xy : tuple[float, float]
            Focal lengths used as starting point for optimization algorithm.
        robust_least_squares : bool
            To use robust least squares solver or not.
        downsample : int
            Amount to downsample the data by in X and Y when performing the
            alignment initialization.

        """
        super().__init__()

        # Save initial parabola shape data
        self.slope_fit_poly_order = 1
        self.initial_focal_lengths_xy = initial_focal_lengths_xy
        self.surf_coefs = np.array(
            [0, 0, 1 / 4 / initial_focal_lengths_xy[0], 0, 0, 1 / 4 / initial_focal_lengths_xy[1]], dtype=float
        )
        self.slope_coefs = np.zeros((2, 3))

        # Save fitting data
        self.robust_least_squares = robust_least_squares
        self.downsample = downsample

    def set_spatial_data(
        self,
        u_active_pixel_pointing_optic: Uxyz,
        v_screen_points_optic: Vxyz,
        v_optic_cam_optic: Vxyz,
        u_measure_pixel_pointing_optic: Uxyz,
        v_align_point_optic: Vxyz,
        v_optic_screen_optic: Vxyz,
    ) -> None:
        """
        Saves all spatial orientation information in object.

        Parameters
        ----------
        u_active_pixel_pointing_optic : Uxyz
            Active pixel pointing directions in optic coordinates.
        v_screen_points_optic : Vxyz
            Positions of screen intersection points in optic coordinates.
        v_optic_cam_optic : Vxyz
            Optic to camera vector in optic coordinates.
        u_measure_pixel_pointing_optic : Uxyz
            Measure pixel pointing direction in optic cooridinates.
        v_align_point_optic : Vxyz
            Position of align point in optic coordinates.
        v_optic_screen_optic : Vxyz
            Optic to screen vector in optic coordinates.

        """
        # Downsample and save measurement data
        self.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic[:: self.downsample]
        self.v_screen_points_optic = v_screen_points_optic[:: self.downsample]

        # Save position data
        self.v_optic_cam_optic = v_optic_cam_optic
        self.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic
        self.v_align_point_optic = v_align_point_optic
        self.v_optic_screen_optic = v_optic_screen_optic

        if self.robust_least_squares:
            self.num_pts = len(self.u_active_pixel_pointing_optic)
            self.weights = np.ones(self.num_pts)

    def intersect(self, u_pixel_pointing: Uxyz, v_origin: Vxyz) -> Vxyz:
        """
        Intersects incoming rays with parabolic surface.

        Parameters
        ----------
        u_pixel_pointing : Uxyz
            Unit vector, pixel pointing directions of camera pixels in optic
            coordinates.
        v_origin : Vxyz
            Location of origin point of rays in optic coordinates.

        Returns
        -------
        Vxyz
            Camera ray intersection points with optic surface in optic coordinates.

        """
        q = u_pixel_pointing.x
        r = u_pixel_pointing.y
        s = u_pixel_pointing.z

        Xc = v_origin.x[0]
        Yc = v_origin.y[0]
        Zc = v_origin.z[0]

        F = self.surf_coefs[0]  # constant
        D = self.surf_coefs[1]  # X
        A = self.surf_coefs[2]  # X^2
        E = self.surf_coefs[3]  # Y
        C = self.surf_coefs[4]  # XY
        B = self.surf_coefs[5]  # Y^2

        # Mask unit vectors pointing straight down (handled below)
        mask = np.logical_and(np.abs(q) < 1e-6, np.abs(r) < 1e-6)

        # Solve quadratic formula for ray intersections with parabola
        a = (A * q**2) + (B * r**2) + (C * q * r)
        b = (2 * A * Xc * q) + (2 * B * Yc * r) + (C * Xc * r) + (C * Yc * q) + (D * q) + (E * r) - s
        c = (A * Xc**2) + (B * Yc**2) + (C * Xc * Yc) + (D * Xc) + (E * Yc) + F - Zc

        a[mask] = np.nan
        b[mask] = np.nan

        scale_1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        scale_2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        mean_1 = scale_1.mean()
        mean_2 = scale_2.mean()

        if mean_1 < 0 and mean_2 < 0:
            raise ValueError('Camera ray intersection points with parabolic surface are found to be behind camera.')
        elif mean_1 > 0 and mean_2 > 0:  # Default to use scale 1
            scale = scale_1
        elif mean_2 > 0:
            scale = scale_2
        else:
            scale = scale_1

        # Calculate unit vectors pointing straight down
        if mask.sum() > 0:
            # Calculate z intersection point
            z_pt_norm = A * Xc**2 + B * Yc**2 + C * Xc * Yc + D * Xc + E * Yc + F
            # Calculate distance from origin to intersection point
            z_scale = Zc - z_pt_norm
            # Update scales
            scale[mask] = z_scale

        # Calculate intersection points
        int_pts = v_origin + u_pixel_pointing.as_Vxyz() * scale[np.newaxis, :]  # optic coordinates
        return int_pts

    def normal_design_at_align_point(self) -> Vxyz:
        """
        Returns the surface normal of the design surface at the align point.

        Returns
        -------
        Vxyz
            Surface normal vector.

        """
        dzdx_design = -self.v_align_point_optic.x[0] / 2 / self.initial_focal_lengths_xy[0]
        dzdy_design = -self.v_align_point_optic.y[0] / 2 / self.initial_focal_lengths_xy[1]
        return Uxyz([dzdx_design, dzdy_design, 1])

    def normal_fit_at_align_point(self) -> Vxyz:
        """
        Returns the surface normal of the fit surface at the align point.

        Returns
        -------
        Vxyz
            Surface normal vector.

        """
        dzdx_meas = -(
            self.slope_coefs[0, 1] * self.v_align_point_optic.x[0]
            + self.slope_coefs[0, 0]
            + self.slope_coefs[1, 2] * self.v_align_point_optic.x[0]
        )
        dzdy_meas = -(
            self.slope_coefs[1, 2] * self.v_align_point_optic.y[0]
            + self.slope_coefs[1, 0]
            + self.slope_coefs[1, 1] * self.v_align_point_optic.y[0]
        )
        return Uxyz((dzdx_meas, dzdy_meas, 1))

    def calculate_surface_intersect_points(self) -> None:
        """
        Calculates pixel ray intersection points with surface.

        """
        # Calculate pixel intersection points with existing fitting function
        self.v_surf_int_pts_optic = self.intersect(self.u_active_pixel_pointing_optic, self.v_optic_cam_optic)

    def calculate_slopes(self) -> tuple[Vxyz, np.ndarray]:
        """
        Calculate slopes of each measurement point.

        """
        self.slopes = sf2.calc_slopes(self.v_surf_int_pts_optic, self.v_optic_cam_optic, self.v_screen_points_optic)

    def fit_slopes(self) -> None:
        """
        Fits slopes to surface.

        """
        # Fit Nth order surfaces to slope distributions in X and Y
        if self.robust_least_squares:
            slope_coefs_x, weights_x = sf2.fit_slope_robust_ls(
                self.slope_fit_poly_order, self.slopes[0], self.weights.copy(), self.v_surf_int_pts_optic
            )
            slope_coefs_y, weights_y = sf2.fit_slope_robust_ls(
                self.slope_fit_poly_order, self.slopes[1], self.weights.copy(), self.v_surf_int_pts_optic
            )
            self.weights = np.array((weights_x, weights_y)).min(0)
        else:
            slope_coefs_x = sf2.fit_slope_ls(self.slope_fit_poly_order, self.slopes[0], self.v_surf_int_pts_optic)
            slope_coefs_y = sf2.fit_slope_ls(self.slope_fit_poly_order, self.slopes[1], self.v_surf_int_pts_optic)

        # Save slope coefficients
        self.slope_coefs = np.array((slope_coefs_x, slope_coefs_y))

        # Average to create surface shape coefficients
        self.surf_coefs = np.array(
            [
                0,
                slope_coefs_x[0],
                slope_coefs_x[1] / 2,
                slope_coefs_y[0],
                (slope_coefs_x[2] + slope_coefs_y[1]) / 2,
                slope_coefs_y[2] / 2,
            ]
        )

        # Calculate z coordinate
        z_pt = self.v_align_point_optic.z[0] - sf2.coef_to_points(self.v_align_point_optic, self.surf_coefs, 2)
        self.surf_coefs[0] = z_pt

    def rotate_all(self, r_align_step: Rotation) -> None:
        """
        Rotates all spatial vectors about align point by given rotation.

        Parameters
        ----------
        r_align_step : Rotation
            Rotation object to rotate all vectors by.

        """
        self.v_optic_cam_optic = self.v_optic_cam_optic.rotate_about(r_align_step, self.v_align_point_optic)
        self.v_screen_points_optic = self.v_screen_points_optic.rotate_about(r_align_step, self.v_align_point_optic)
        self.u_active_pixel_pointing_optic = self.u_active_pixel_pointing_optic.rotate(r_align_step)
        self.u_measure_pixel_pointing_optic = self.u_measure_pixel_pointing_optic.rotate(r_align_step)
        self.v_optic_screen_optic = self.v_optic_screen_optic.rotate_about(r_align_step, self.v_align_point_optic)

    def shift_all(self, v_align_optic_step: Vxyz) -> None:
        """
        Shifts all spatial vectors by given step size.

        Parameters
        ----------
        v_align_optic_step : Vxyz
            Shift vector to apply to all data.

        """
        self.v_optic_cam_optic += v_align_optic_step
        self.v_screen_points_optic += v_align_optic_step
        self.v_optic_screen_optic += v_align_optic_step

    def save_to_hdf(self, file: str, prefix: str = ''):
        """Saves data to given file. Data is stored as: PREFIX + ParamsSurface/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        data = [self.initial_focal_lengths_xy, self.robust_least_squares, self.downsample, 'parabolic']
        datasets = [
            prefix + 'ParamsSurface/initial_focal_lengths_xy',
            prefix + 'ParamsSurface/robust_least_squares',
            prefix + 'ParamsSurface/downsample',
            prefix + 'ParamsSurface/surface_type',
        ]
        save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ''):
        """Loads data from given file. Assumes data is stored as: PREFIX + ParamsSurface/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        # Check surface type
        data = load_hdf5_datasets([prefix + 'ParamsSurface/surface_type'], file)
        if data['surface_type'] != 'parabolic':
            raise ValueError(f'Surface2DParabolic cannot load surface type, {data["surface_type"]:s}')

        # Load
        datasets = [
            prefix + 'ParamsSurface/initial_focal_lengths_xy',
            prefix + 'ParamsSurface/robust_least_squares',
            prefix + 'ParamsSurface/downsample',
        ]
        data = load_hdf5_datasets(datasets, file)
        return cls(**data)
