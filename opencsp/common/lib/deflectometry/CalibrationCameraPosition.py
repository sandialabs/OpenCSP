import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation

from opencsp.common.lib.photogrammetry.photogrammetry import find_aruco_marker
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


class CalibrationCameraPosition:
    """Class that calculates the relative pose of a camera viewing several Aruco markers.
    Must have a list of xyz marker corner points and corner IDs. There can be more IDs in
    this list than is viewed by the camera.

    Calculates
    ----------
        - rvec : rotation vector, screen to camera rotation vector
        - tvec : translation vector, camera to screen (in screen coordinates) translation vector

    Verbose Settings (self.verbose)
    -------------------------------
        - 0 no output
        - 1 print only output
        - 2 print and plot output
        - 3 plot only output
    """

    def __init__(
        self,
        camera: Camera,
        pts_xyz_corners: Vxyz,
        ids_corners: ndarray,
        cal_image: ndarray,
    ) -> 'CalibrationCameraPosition':
        """Instantiates class

        Parameters
        ----------
        camera : Camera
            Camera object
        pts_xyz_corners : Vxyz
            Aruco corner xyz positions
        ids_corners : ndarray
            Corner IDs corresponding to pts_xyz_corners
        cal_image : ndarray
            Calibration image captured by camera
        """
        # Initialize attributes data types
        self.camera = camera
        self.pts_xyz_corners = pts_xyz_corners
        self.ids_corners = ids_corners
        self.image = cal_image

        self.ids_corners: ndarray
        self.image: ndarray
        self.ids_marker: ndarray[int]
        self.pts_xy_marker_corners_list: list[ndarray]
        self.pts_xyz_active_corner_locations: Vxyz
        self.rot_screen_cam: Rotation
        self.v_cam_screen_cam: Vxyz
        self.v_cam_screen_screen: Vxyz
        self.errors_reprojection_xy: Vxy
        self.pts_xy_marker_corners_reprojected: Vxy

        # Save figures
        self.verbose: int = 0
        self.figures: list[plt.Figure] = []

    @property
    def _to_print(self) -> bool:
        """If verbose printing is turned on"""
        return self.verbose in [1, 2]

    @property
    def _to_plot(self) -> bool:
        """If verbose plotting is turned on"""
        return self.verbose in [2, 3]

    def find_markers(self) -> None:
        """Finds marker corner locations in image"""
        self.ids_marker, self.pts_xy_marker_corners_list = find_aruco_marker(self.image)

    def collect_corner_xyz_locations(self) -> None:
        """Collects corner locations of viewed markers"""
        # Extract object points
        self.pts_xyz_active_corner_locations = Vxyz.empty()
        ids_corners_list = self.ids_corners.tolist()
        for marker_id in self.ids_marker:
            # Get index of current marker
            index = ids_corners_list.index(marker_id * 4)
            # Extract calibrated corner locations (4 corners per marker)
            self.pts_xyz_active_corner_locations = (
                self.pts_xyz_active_corner_locations.concatenate(
                    self.pts_xyz_corners[index:index + 4]
                )
            )

    def calculate_camera_pose(self) -> None:
        """Calculates the camera pose"""
        # Concatenate image points
        pts_img = np.vstack(self.pts_xy_marker_corners_list)

        # Calculate rvec/tvec
        ret, rvec, tvec = cv.solvePnP(
            self.pts_xyz_active_corner_locations.data.T,
            pts_img,
            self.camera.intrinsic_mat,
            self.camera.distortion_coef,
        )
        if not ret:
            raise ValueError('Camera calibration was not successful.')

        rvec: ndarray = rvec.squeeze()
        tvec: ndarray = tvec.squeeze()

        self.rot_screen_cam = Rotation.from_rotvec(rvec)
        self.v_cam_screen_cam = Vxyz(tvec)
        self.v_cam_screen_screen = self.v_cam_screen_cam.rotate(
            self.rot_screen_cam.inv()
        )

        if self._to_print:
            print('Camera pose calculated:')
            print(f'   rvec: {self.rot_screen_cam.as_rotvec()}')
            print(f'   tvec: {self.v_cam_screen_screen.data.squeeze()}')

    def get_data(self) -> tuple[ndarray, ndarray]:
        """Returns rvec and tvec orienting camera to screen coordinates

        Returns
        -------
        rot_screen_cam : ndarray
            (3,) vector, screen to camera rotation
        v_cam_screen_screen : ndarray
            (3,) vector, camera to screen translation in screen coordinates
        """
        rvec = self.rot_screen_cam.as_rotvec()
        tvec = self.v_cam_screen_screen.data.squeeze()
        return rvec, tvec

    def save_data_as_csv(self, file: str) -> None:
        """Saves rvec/tvec as csv file"""
        rvec, tvec = self.get_data()
        data = np.vstack((rvec, tvec))
        np.savetxt(file, data, delimiter=',', fmt='%.8f')

        print(f'Saved camera rvec and tvec to: {os.path.abspath(file):s}')

    def calculate_reprojection_error(self) -> None:
        """Calculates reprojection error"""
        # Project points
        self.pts_xy_marker_corners_reprojected = self.camera.project(
            self.pts_xyz_active_corner_locations,
            self.rot_screen_cam,
            self.v_cam_screen_cam,
        )

        # Calculate errors
        self.errors_reprojection_xy = self.pts_xy_marker_corners_reprojected - Vxy(
            np.vstack(self.pts_xy_marker_corners_list).T
        )

        if self._to_print:
            errors_mag: ndarray = np.sqrt(
                (self.errors_reprojection_xy.data.T**2).sum(axis=1)
            )
            print('Camera pose reprojection error:')
            print(f'   Mean error: {errors_mag.mean():.3f} pixels')
            print(
                f'   STDEV of errors (N={errors_mag.size}): {errors_mag.std():.4f} pixels'
            )

    def plot_found_corners(self) -> None:
        """Plots camera image and found corners"""
        fig = plt.figure('CalibrationCameraPosition_Found_Markers')
        self.figures.append(fig)
        ax = fig.gca()

        ax.imshow(self.image, cmap='gray')
        for id_, pts in zip(self.ids_marker, self.pts_xy_marker_corners_list):
            plt.scatter(*pts.T, marker='+')
            plt.text(*pts.mean(0).T + np.array([60, 0]), id_, backgroundcolor='white')

    def plot_reprojection_error(self) -> None:
        """Plots reprojection error"""
        fig = plt.figure('CalibrationCameraPosition_Reprojection_Error')
        self.figures.append(fig)
        ax = fig.gca()

        pts_img = np.vstack(self.pts_xy_marker_corners_list)

        ax.imshow(self.image, cmap='gray')
        ax.scatter(
            *pts_img.T, edgecolor='green', facecolor='none', label='Image Points'
        )
        ax.scatter(
            *self.pts_xy_marker_corners_reprojected.data,
            marker='.',
            color='blue',
            label='Reprojected',
        )
        dx = self.errors_reprojection_xy.x
        dy = -self.errors_reprojection_xy.y
        ax.quiver(
            *self.pts_xy_marker_corners_reprojected.data,
            dx,
            dy,
            label='Error',
            color='red',
        )
        ax.legend()
        ax.axis('off')

    def run_calibration(self) -> None:
        """Runs calibration sequence"""

        # Run calibration
        self.find_markers()
        self.collect_corner_xyz_locations()
        self.calculate_camera_pose()
        self.calculate_reprojection_error()

        # Plot figures
        if self._to_plot:
            self.plot_found_corners()
            self.plot_reprojection_error()
