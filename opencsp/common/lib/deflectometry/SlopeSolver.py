import warnings

import matplotlib.pyplot as plt
import numpy as np
from   scipy.optimize import minimize

from   opencsp.common.lib.deflectometry.SlopeSolverDataDebug import SlopeSolverDataDebug
from   opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
import opencsp.common.lib.deflectometry.slope_fitting_2d as sf2
from   opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from   opencsp.common.lib.deflectometry.Surface2DPlano import Surface2DPlano
from   opencsp.common.lib.geometry.Uxyz import Uxyz
from   opencsp.common.lib.geometry.Vxyz import Vxyz
from   opencsp.common.lib.geometry.TransformXYZ import TransformXYZ


class SlopeSolver:
    def __init__(self,
                 v_optic_cam_optic: Vxyz,
                 u_active_pixel_pointing_optic: Uxyz,
                 u_measure_pixel_pointing_optic: Uxyz,
                 v_screen_points_facet: Vxyz,
                 v_optic_screen_optic: Vxyz,
                 v_align_point_optic: Vxyz,
                 dist_optic_screen: float,
                 surface_data: dict,
                 debug: SlopeSolverDataDebug = SlopeSolverDataDebug()):
        """
        Initializes the slope solving object.

        Parameters
        ----------
        v_optic_cam_optic : Vxyz
            Optic to camera vector in optic coordinates.
        u_active_pixel_pointing_optic : Uxyz
            Active pixel pointing directions in optic coordinates.
        u_measure_pixel_pointing_optic : Uxyz
            Measure pixel pointing direction in optic cooridinates.
        v_screen_points_facet : Vxyz
            Positions of screen points in optic coordinates.
        v_optic_screen_optic : Vxyz
            Optic to screen vector in optic coordinates.
        v_align_point_optic : Vxyz
            Position of align point in optic coordinates.
        dist_optic_screen : float
            Measured optic to screen distance.
        surface_data : dict
            Dictionary containing surface data information to use when
            solving slopes. The data fields depend on the surface fit being
            performed. The following options are supported:

            1) Parabolic fit
                surface_type
                initial_focal_lengths_xy
                robust_least_squares
                downsample
            2) Plano fit
                surface_type
                robust_least_squares
                downsample
            3) Spherical fit (will raise NotImplementedError)
                surface_type
                radius
                robust_least_squares
                downsample

            Data field descriptions
            -----------------------
                surface_type : str {'parabolic', 'plano', 'spherical'}
                    The type of surface being characterized
                initial_focal_lengths_xy : list[tuple(float, float)]
                    The focal lengths to use as the starting point for the
                    fitting algorithm.
                robust_least_squares : bool
                    To use robust least squares fitting, or just least squares
                    fitting.
                downsample : int
                    The amount to downsample data for surface fitting
                radius : float
                    The initial radius to use as the starting point for the
                    fitting algorithm.
        debug: SlopeSolverDataDebug
            SlopeSolverDataDebug object for debugging.

        """
        # Instantiate surface fit object depending on fit type
        surface_data_copy = surface_data.copy()
        self.surface_type = surface_data_copy.pop('surface_type')
        if self.surface_type == 'parabolic':
            self.surface = Surface2DParabolic(**surface_data_copy)
        elif self.surface_type == 'plano':
            self.surface = Surface2DPlano(**surface_data_copy)
        elif self.surface_type == 'spherical':
            raise NotImplementedError('Currently, "spherical" surface type is not implemented.')
        else:
            raise ValueError(f'Given surface_type "{self.surface_type:s}" not supported.')

        # Store inputs in class
        self.v_optic_cam_optic = v_optic_cam_optic
        self.u_active_pixel_pointing_optic = u_active_pixel_pointing_optic
        self.u_measure_pixel_pointing_optic = u_measure_pixel_pointing_optic
        self.v_screen_points_facet = v_screen_points_facet
        self.v_optic_screen_optic = v_optic_screen_optic
        self.v_align_point_optic = v_align_point_optic
        self.dist_optic_screen = dist_optic_screen
        self.debug = debug

        # Load initialization data in surface fit object
        self.surface.set_spatial_data(u_active_pixel_pointing_optic,
                                      v_screen_points_facet,
                                      v_optic_cam_optic,
                                      u_measure_pixel_pointing_optic,
                                      v_align_point_optic,
                                      v_optic_screen_optic)

        self._data = SlopeSolverData()

    def get_data(self) -> SlopeSolverData:
        """Returns data output object"""
        return self._data

    def fit_surface(self) -> None:
        """
        Performs the initial fine-tuning alignment of the facet, screen, and
        camera. Fits a surface to the calculated slope data.

        """
        # Gather inputs
        v_optic_cam_optic = self.v_optic_cam_optic
        u_measure_pixel_pointing_optic = self.u_measure_pixel_pointing_optic
        v_optic_screen_optic = self.v_optic_screen_optic
        v_align_point_optic = self.v_align_point_optic
        dist_optic_screen = self.dist_optic_screen

        # Instantiate alignment transform
        trans_align = TransformXYZ.from_zero_zero()

        for idx1 in range(4):
            for idx2 in range(3):
                # Calculate surface intersection points
                self.surface.calculate_surface_intersect_points()

                # Check for invalid points
                num_nans = np.isnan(self.surface.v_surf_int_pts_optic.data)
                if np.any(num_nans):
                    warnings.warn(f'{num_nans.sum():d} / {num_nans.size:d} values are NANs in surface intersection points in iteration: ({idx1:d}, {idx2:d}).', stacklevel=2)

                # Calculate measurement point slopes
                self.surface.calculate_slopes()

                # Check for invalid points
                num_nans = np.isnan(self.surface.slopes)
                if np.any(num_nans):
                    warnings.warn(f'{num_nans.sum():d} / {num_nans.size:d} values are NANs in slope data in iteration: ({idx1:d}, {idx2:d}).', stacklevel=2)

                # Update slope fit
                self.surface.fit_slopes()

                # Plot debug plot
                if self.debug.debug_active:
                    self._plot_debug_plots(idx1, idx2)

            # Calculate measure point intersection point with existing fitting function
            v_meas_pts_surf_int_optic = self.surface.intersect(u_measure_pixel_pointing_optic, v_optic_cam_optic)

            # Calculate design normal at alignment point
            n_design = self.surface.normal_design_at_align_point()

            # Calculate measured normal at alignment point
            n_meas = self.surface.normal_fit_at_align_point()

            # Calculate the rotation needed to align the normal vectors
            r_align_step = n_meas.align_to(n_design)

            # Rotate all points about alignment point
            self.surface.rotate_all(r_align_step)

            # Calculate scale so that align-point to screen matches measurement
            args = (dist_optic_screen, v_align_point_optic, v_optic_cam_optic, v_optic_screen_optic, v_meas_pts_surf_int_optic)
            out = minimize(sf2.optic_screen_dist_error, np.array([1.]), args=args)
            scale = out.x[0]
            v_align_optic_step = (v_optic_cam_optic - v_align_point_optic) * (scale - 1)

            # Shift all points along align-point to camera axis
            self.surface.shift_all(v_align_optic_step)

            # Calculate alignment transform
            trans_step = TransformXYZ.from_R_V(r_align_step, v_align_optic_step)
            trans_align = trans_step * trans_align

        # Store alignment parameters
        self._data.surf_coefs_facet = self.surface.surf_coefs
        self._data.slope_coefs_facet = self.surface.slope_coefs
        self._data.trans_alignment = trans_align

    def solve_slopes(self) -> None:
        """
        Solves the surface slopes of the optic using camera position
        and alignment transform from self.fit_surface.

        Raises
        ------
        Exception
            Raises ValueError if initial alignment has not been performed
            prior to calling this function (self.fit_surface()).

        """
        # Check alignment has been completed
        if self._data.trans_alignment is None:
            raise ValueError('Initial alignment needs to be completed before final slope fitting (self.fit_surface).')

        # Apply alignment transforms about alignment point
        trans_shift_1 = TransformXYZ.from_V(-self.surface.v_align_point_optic)
        trans_shift_2 = TransformXYZ.from_V(self.surface.v_align_point_optic)
        trans: TransformXYZ = trans_shift_2 * self._data.trans_alignment * trans_shift_1

        u_active_pixel_pointing_optic = self.u_active_pixel_pointing_optic.rotate(trans.R)
        v_screen_points_facet = trans.apply(self.v_screen_points_facet)

        # Calculate intersection points on optic surface
        v_surf_points_facet = self.surface.intersect(u_active_pixel_pointing_optic, self.surface.v_optic_cam_optic)

        # Calculate pixel slopes (assuming parabolic surface intersection)
        slopes_facet_xy = sf2.calc_slopes(v_surf_points_facet,
                                          self.surface.v_optic_cam_optic,
                                          v_screen_points_facet)

        self._data.v_surf_points_facet = v_surf_points_facet
        self._data.slopes_facet_xy = slopes_facet_xy

    def _plot_debug_plots(self, idx1: int, idx2: int):
        # Create figure and axes
        if self.debug.slope_solver_single_plot and isinstance(self.debug.slope_solver_figures, list):
            # Create first figure if needed
            fig = plt.figure()
            axes = fig.add_subplot(projection='3d')
            self.debug.slope_solver_figures = fig
            # Plot facet corners
            facet_outline = self.debug.optic_data.v_facet_corners.data
            axes.scatter(*facet_outline, color='k')
            # Format
            axes.set_title('Slope Solver')
        elif self.debug.slope_solver_single_plot:
            # Get axes for single plot
            axes = self.debug.slope_solver_figures.gca()
        else:
            # Create a new figure
            fig = plt.figure()
            axes = fig.add_subplot(projection='3d')
            self.debug.slope_solver_figures.append(fig)
            # Plot facet corners
            facet_outline = self.debug.optic_data.v_facet_corners.data
            axes.scatter(*facet_outline, color='k')
            # Format
            axes.set_title(f'Slope Solver ({idx1:d}, {idx2:d})')

        # Plot intersection points
        self.surface.plot_intersection_points(axes,
                                              self.debug.slope_solver_point_downsample,
                                              self.debug.slope_solver_camera_rays_length,
                                              self.debug.slope_solver_plot_camera_screen_points)
