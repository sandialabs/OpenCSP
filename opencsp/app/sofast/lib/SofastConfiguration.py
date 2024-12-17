"""Class that analyzes the configuration of a Sofast run. This includes:
- The physical system layout
- Statistics from a Sofast measurement
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.log_tools as lt


class SofastConfiguration:
    """Class for analyzing the configuration of a Sofast setup/measurement."""

    def __init__(self) -> 'SofastConfiguration':
        self.data_sofast_object: ProcessSofastFringe | ProcessSofastFixed = None
        self._is_fringe = None
        self._is_fixed = None

    def load_sofast_object(self, process_sofast: ProcessSofastFringe | ProcessSofastFixed):
        """Loads ProcessSofast object (Fixed or Fringe) for further analysis

        Parameters
        ----------
        process_sofast: ProcessSofastFringe | ProcessSofastFixed
            ProcessSofast object
        """
        self.data_sofast_object = process_sofast
        if isinstance(self.data_sofast_object, ProcessSofastFringe):
            self._is_fringe = True
            self._is_fixed = False
        elif isinstance(self.data_sofast_object, ProcessSofastFixed):
            self._is_fringe = False
            self._is_fixed = True

    def get_measurement_stats(self) -> list[dict]:
        """Returns measurement statistics dictionary for each facet in Sofast calculation

        Returns
        -------
        list of dictionaries for each facet in the ProcessSofast object with following fields:
        - delta_x_sample_points_average
        - delta_y_sample_average
        - number_samples
        - focal_lengths_parabolic_xy
        """
        self._check_sofast_object_loaded()
        num_facets = self.data_sofast_object.num_facets

        stats = []
        for idx_facet in range(num_facets):
            if self._is_fringe:
                # Get surface data
                data_calc = self.data_sofast_object.data_calculation_facet[idx_facet]
                data_im_proc = self.data_sofast_object.data_image_processing_facet[idx_facet]
                data_surf = self.data_sofast_object.data_surfaces[idx_facet]

                # Assemble surface points in 2d arrays
                mask = data_im_proc.mask_processed
                im_x = np.zeros(mask.shape) * np.nan
                im_y = np.zeros(mask.shape) * np.nan
                im_x[mask] = data_calc.v_surf_points_facet.x
                im_y[mask] = data_calc.v_surf_points_facet.y

                # Number of points
                num_samps = len(data_calc.v_surf_points_facet)
            else:
                # Get surface data
                data_surf = self.data_sofast_object.slope_solvers[idx_facet].surface
                data_calc = self.data_sofast_object.data_calculation_facet[idx_facet]

                # Assemble surface points in 2d arrays
                surf_points = self.data_sofast_object.data_calculation_facet[idx_facet].v_surf_points_facet
                mask = self.data_sofast_object.data_calculation_blob_assignment[idx_facet].active_point_mask
                im_x = mask.astype(float) * np.nan
                im_y = mask.astype(float) * np.nan
                im_x[mask] = surf_points.x
                im_y[mask] = surf_points.y

                # Number of points
                num_samps = len(surf_points)

            # Calculate average sample resolution
            dx = np.diff(im_x, axis=1)  # meters
            dy = np.diff(im_y, axis=0)  # meters
            dx_avg = abs(np.nanmean(dx))  # meters
            dy_avg = abs(np.nanmean(dy))  # meters

            # Parabolic focal length in x and y
            if isinstance(data_surf, Surface2DParabolic):
                surf_coefs = data_calc.surf_coefs_facet
                focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
            else:
                focal_lengths_xy = [np.nan, np.nan]

            stats.append(
                {
                    'delta_x_sample_points_average': dx_avg,
                    'delta_y_sample_points_average': dy_avg,
                    'number_samples': num_samps,
                    'focal_lengths_parabolic_xy': focal_lengths_xy,
                }
            )

        return stats

    def visualize_setup(
        self,
        length_z_axis_cam: float = 8,
        axes_length: float = 2,
        min_axis_length_screen: float = 2,
        ax: plt.Axes | None = None,
        v_screen_object_screen: Vxyz = None,
        r_object_screen: Rotation = None,
    ) -> None:
        """Draws the given SOFAST setup components on a 3d axis.

        Parameters
        ----------
        length_z_axis_cam : float, optional
            Length of camera z axis to draw (m), by default 8
        axes_length : float, optional
            Length of all other axes to draw (m), by default 2
        min_axis_length_screen : float, optional
            Minimum length of axes to draw (m), by default 2
        ax : plt.Axes | None, optional
            3d matplotlib axes, if None, creates new axes, by default None
        v_screen_object_screen : Vxyz, optional
            Vector (m), screen to object in screen reference frame, by default None.
            If None, the object reference frame is not plotted.
        r_object_screen : Rotation, optional
            Rotation, object to screen reference frames, by default None.
            Only used if v_screen_object_screen is not None
        """
        orientation = self.data_sofast_object.orientation
        camera = self.data_sofast_object.camera

        # Get axes
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        ax.view_init(-15, 135, roll=180, vertical_axis='y')

        # Calculate camera position
        v_screen_cam_screen = -orientation.v_cam_screen_screen

        # Calculate camera FOV
        x = camera.image_shape_xy[0]
        y = camera.image_shape_xy[1]
        v_cam_fov_screen = (
            camera.vector_from_pixel(Vxy(([0, 0, x, x, 0], [0, y, y, 0, 0]))).as_Vxyz() * length_z_axis_cam
        )
        v_cam_fov_screen.rotate_in_place(orientation.r_cam_screen)
        v_cam_fov_screen += v_screen_cam_screen

        # Calculate camera X/Y axes
        v_cam_x_screen = Vxyz(([0, axes_length], [0, 0], [0, 0])).rotate(orientation.r_cam_screen) + v_screen_cam_screen
        v_cam_y_screen = Vxyz(([0, 0], [0, axes_length], [0, 0])).rotate(orientation.r_cam_screen) + v_screen_cam_screen
        v_cam_z_screen = (
            Vxyz(([0, 0], [0, 0], [0, length_z_axis_cam])).rotate(orientation.r_cam_screen) + v_screen_cam_screen
        )

        # Calculate object axes
        if v_screen_object_screen is not None:
            v_obj_x_screen = Vxyz(([0, axes_length], [0, 0], [0, 0])).rotate(r_object_screen) + v_screen_object_screen
            v_obj_y_screen = Vxyz(([0, 0], [0, axes_length], [0, 0])).rotate(r_object_screen) + v_screen_object_screen
            v_obj_z_screen = Vxyz(([0, 0], [0, 0], [0, axes_length])).rotate(r_object_screen) + v_screen_object_screen

        # Calculate screen outline and center
        if self._is_fringe:
            display = self.data_sofast_object.display
            p_screen_outline = display.interp_func(Vxy(([0, 0.95, 0.95, 0, 0], [0, 0, 0.95, 0.95, 0])))
            p_screen_cent = display.interp_func(Vxy((0.5, 0.5)))
        elif self._is_fixed:
            locs = self.data_sofast_object.fixed_pattern_dot_locs.xyz_dot_loc
            p_screen_outline = Vxyz((locs[..., 0], locs[..., 1], locs[..., 2]))
            p_screen_cent = self.data_sofast_object.fixed_pattern_dot_locs.xy_indices_to_screen_coordinates(
                Vxy([0, 0], dtype=int)
            )

        # Define positive xyz screen axes extent
        if v_screen_object_screen is None:
            obj_x = [np.nan]
            obj_y = [np.nan]
            obj_z = [np.nan]
        else:
            obj_x = v_screen_object_screen.x
            obj_y = v_screen_object_screen.y
            obj_z = v_screen_object_screen.z
        lx1 = max(
            np.nanmax(np.concatenate((v_screen_cam_screen.x, v_cam_fov_screen.x, p_screen_outline.x, obj_x))),
            min_axis_length_screen,
        )
        ly1 = max(
            np.nanmax(np.concatenate((v_screen_cam_screen.y, v_cam_fov_screen.y, p_screen_outline.y, obj_y))),
            min_axis_length_screen,
        )
        lz1 = max(
            np.nanmax(np.concatenate((v_screen_cam_screen.z, v_cam_fov_screen.z, p_screen_outline.z, obj_z))),
            min_axis_length_screen,
        )
        # Define negative xyz screen axes extent
        lx2 = min(
            np.nanmin(np.concatenate((v_screen_cam_screen.x, v_cam_fov_screen.x, p_screen_outline.x, obj_x))),
            -min_axis_length_screen,
        )
        ly2 = min(
            np.nanmin(np.concatenate((v_screen_cam_screen.y, v_cam_fov_screen.y, p_screen_outline.y, obj_y))),
            -min_axis_length_screen,
        )
        lz2 = min(
            np.nanmin(np.concatenate((v_screen_cam_screen.z, v_cam_fov_screen.z, p_screen_outline.z, obj_z))),
            -min_axis_length_screen,
        )
        # Add screen axes
        x = p_screen_cent.x[0]
        y = p_screen_cent.y[0]
        z = p_screen_cent.z[0]
        # Screen X axis
        ax.plot([x, x + lx1], [y, y], [z, z], color='red')
        ax.plot([x, x + lx2], [y, y], [z, z], color='black')
        ax.text(x + lx1, y, z, 'x')
        # Screen Y axis
        ax.plot([x, x], [y, y + ly1], [z, z], color='green')
        ax.plot([x, x], [y, y + ly2], [z, z], color='black')
        ax.text(x, y + ly1, z, 'y')
        # Screen Z axis
        ax.plot([x, x], [y, y], [z, z + lz1], color='blue')
        ax.plot([x, x], [y, y], [z, z + lz2], color='black')
        ax.text(x, y, z + lz1, 'z')

        if self._is_fixed:
            # Add screen points
            ax.scatter(*p_screen_outline.data, marker='.', alpha=0.5, color='blue', label='Screen Points')
        else:
            # Add screen outline
            ax.plot(*p_screen_outline.data, color='red', label='Screen Outline')

        # Add camera position origin
        ax.scatter(*v_screen_cam_screen.data, color='black')
        ax.text(*v_screen_cam_screen.data.squeeze(), 'camera')

        # Add camera XYZ axes
        ax.plot(*v_cam_x_screen.data, color='red')
        ax.text(*v_cam_x_screen[1].data.squeeze(), 'x', color='blue')
        ax.plot(*v_cam_y_screen.data, color='green')
        ax.text(*v_cam_y_screen[1].data.squeeze(), 'y', color='blue')
        ax.plot(*v_cam_z_screen.data, color='blue')
        ax.text(*v_cam_z_screen[1].data.squeeze(), 'z', color='blue')

        # Add camera FOV bounding box
        ax.plot(*v_cam_fov_screen.data)

        if v_screen_object_screen is not None:
            # Add object position origin
            ax.scatter(*v_screen_object_screen.data, color='black')
            ax.text(*v_screen_object_screen.data.squeeze(), 'object')

            # Add object XYZ axes
            ax.plot(*v_obj_x_screen.data, color='red')
            ax.text(*v_obj_x_screen[1].data.squeeze(), 'x', color='blue')
            ax.plot(*v_obj_y_screen.data, color='green')
            ax.text(*v_obj_y_screen[1].data.squeeze(), 'y', color='blue')
            ax.plot(*v_obj_z_screen.data, color='blue')
            ax.text(*v_obj_z_screen[1].data.squeeze(), 'z', color='blue')

        # Format and show
        plt.title('SOFAST Physical Setup\n(Screen Coordinates)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        plt.axis('equal')

    def _check_sofast_object_loaded(self) -> bool:
        if self.data_sofast_object is None:
            lt.error_and_raise(ValueError, 'ProcessSofast object not loaded. Use self.load_sofast_object() first.')
