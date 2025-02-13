from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import HDF5_IO_Abstract


class Surface2DAbstract(HDF5_IO_Abstract):
    """Representation of 2d surface for SOFAST processing"""

    def __init__(self):
        # Instantiate variables
        self.v_surf_int_pts_optic: Vxyz
        self.slopes: np.ndarray
        self.u_active_pixel_pointing_optic: Vxyz
        self.v_screen_points_optic: Vxyz
        self.v_optic_cam_optic: Vxyz
        self.u_measure_pixel_pointing_optic: Uxyz
        self.v_optic_screen_optic: Vxyz
        self.v_align_point_optic: Vxyz
        self.weights: np.ndarray
        self.num_pts: int
        self.surf_coefs: np.ndarray
        self.slope_coefs: np.ndarray

    @abstractmethod
    def set_spatial_data(
        self,
        u_active_pixel_pointing_optic: Uxyz,
        v_screen_points_optic: Vxyz,
        v_optic_cam_optic: Vxyz,
        u_measure_pixel_pointing_optic: Uxyz,
        v_align_point_optic: Vxyz,
        v_optic_screen_optic: Vxyz,
    ) -> None:
        """Saves spatial data in class"""

    @abstractmethod
    def intersect(self, u_pixel_pointing: Uxyz, v_origin: Vxyz) -> Vxyz:
        """Intersects rays with the surface"""

    @abstractmethod
    def normal_design_at_align_point(self) -> Vxyz:
        """Normal vector of design surface at point"""

    @abstractmethod
    def normal_fit_at_align_point(self) -> Vxyz:
        """Normal vector of fit surface at point"""

    @abstractmethod
    def calculate_surface_intersect_points(self) -> Vxyz:
        """Calculates surface intersection points"""

    @abstractmethod
    def calculate_slopes(self) -> tuple[Vxyz, np.ndarray]:
        """Calculate slopes for all measurement points"""

    @abstractmethod
    def fit_slopes(self) -> None:
        """Fits slopes to using coefficients"""

    @abstractmethod
    def rotate_all(self, r_align_step: Rotation) -> None:
        """Rotates all data vectors"""

    @abstractmethod
    def shift_all(self, v_align_optic_step: Vxyz) -> None:
        """Shifts all data vectors"""

    def plot_intersection_points(
        self,
        axes: plt.Axes,
        downsample: int = 50,
        camera_ray_length: float = 0.0,
        plot_camera_screen_points: bool = False,
    ) -> None:
        """Plots calculated intersection points with surface and align point. Optionally
        plots camera rays and screen/camera locations.

        Parameters
        ----------
        axes : plt.Axes
            Matplotlib axes, None to make new figure.
        downsample : int, optional
            Ray downsample factor, by default 50
        camera_ray_length : float
            Ray lengths, meters. Set to 0 to turn off.
        plot_camera_screen_points : bool
            To plot camera and screen locations, by default False.

        Returns
        -------
        Matplotlib axes
        """
        # Plot intersection points surface
        axes.plot_trisurf(
            *self.v_surf_int_pts_optic[::downsample].data, edgecolor="none", alpha=0.5, linewidth=0, antialiased=False
        )

        # Plot camera rays
        if camera_ray_length != 0:
            for ray in self.u_active_pixel_pointing_optic[::downsample]:
                x = [self.v_optic_cam_optic.x, self.v_optic_cam_optic.x + ray.x * camera_ray_length]
                y = [self.v_optic_cam_optic.y, self.v_optic_cam_optic.y + ray.y * camera_ray_length]
                z = [self.v_optic_cam_optic.z, self.v_optic_cam_optic.z + ray.z * camera_ray_length]
                axes.plot(x, y, z, color="gray", alpha=0.3)

        # Plot fit normal at align point
        v_fit = self.normal_fit_at_align_point()
        pt1 = self.v_align_point_optic
        pt2 = self.v_align_point_optic + v_fit
        axes.plot([pt1.x, pt2.x], [pt1.y, pt2.y], [pt1.z, pt2.z], color="k", linestyle="-")
        # Plot design normal at align point
        v_des = self.normal_design_at_align_point()
        pt1 = self.v_align_point_optic
        pt2 = self.v_align_point_optic + v_des
        axes.plot([pt1.x, pt2.x], [pt1.y, pt2.y], [pt1.z, pt2.z], color="k", linestyle="--")

        # Plot other points
        axes.scatter(*self.v_align_point_optic.data, marker="o", color="r", label="Align Point")
        if plot_camera_screen_points:
            axes.scatter(*self.v_optic_cam_optic.data, marker="*", color="k", label="Camera")
            axes.scatter(*self.v_optic_screen_optic.data, marker="+", color="b", label="Screen Center")

        # Format
        axes.axis("equal")
        axes.set_xlabel("x (meter)")
        axes.set_ylabel("y (meter)")
        axes.set_zlabel("z (meter)")
