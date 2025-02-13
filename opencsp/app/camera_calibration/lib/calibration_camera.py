"""Library of functions used to calibation a machine vision camera.
Functions are based off OpenCV library.
"""

from typing import Iterable

import cv2 as cv
from matplotlib.axes import Axes
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


def calibrate_camera(
    p_object: Iterable[Vxyz], p_image: Iterable[Vxy], img_shape_xy: tuple[int, int], name: str
) -> tuple[Camera, Iterable[Rotation], Iterable[Vxyz], float]:
    """
    Performs 4 term camera calibration for non-fisheye lens.
    Calculates only distortion coefficients, [K1, K2, P1, P2] (K3 = 0). Higher
    order fits are generally used for fisheye and other exotic lenses and are
    not supported here.

    Parameters
    ----------
    p_object : list[Vxyz, ...]
        List of object points (grid coordinates).
    p_image : list[Vxy, ...]
        List of image points (pixels).
    img_shape_xy : tuple[int, int]
        Size of image in pixels.
    name : str
        Name of camera.

    Returns
    -------
    Camera : opencsp.common.lib.camera.Camera.Camera
        Camera class.
    r_cam_object : list[Rotation, ...]
        Camera-object rotation vector
    v_cam_object_cam : list[Vxyz, ...]
        Camera location vector
    Error : float
        Average reprojection error (pixels).

    """
    # Perform calibraiton with: K3 = 0
    obj_pts_list = [v.data.T for v in p_object]
    img_pts_list = [v.data.T for v in p_image]
    dist_input = np.zeros(4, dtype=np.float32)
    error, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        obj_pts_list, img_pts_list, img_shape_xy, None, dist_input, flags=cv.CALIB_FIX_K3
    )
    # Keep only first four distortion coefficients
    dist = dist[:4].squeeze()

    # Process rvecs
    r_cam_object = [Rotation.from_rotvec(r.squeeze()) for r in rvecs]

    # Process tvecs
    v_cam_object_cam = [Vxyz(t) for t in tvecs]

    # Save in Camera object
    camera = Camera(mtx, dist, img_shape_xy, name)

    return camera, r_cam_object, v_cam_object_cam, error


def view_distortion(camera: Camera, ax1: Axes, ax2: Axes, ax3: Axes, num_samps: int = 12):
    """
    Plots the radial/tangential distortion of a camera object.

    Parameters
    ----------
    camera : opencsp.common.lib.camera.Camera.Camera
        Camera to visualize.
    ax1 : Axes
        Axis to plot radial distortion.
    ax2 : Axes
        Axis to plot tangential distortion.
    ax3 : Axes
        Axis to plot total distortion.
    num_samps : int, optional
        Number of samples across short side of image. The default is 12.

    """
    # Get distortion coefficients for tangential only
    dist_coef_tan = np.zeros(4)
    dist_coef_tan[2:4] = camera.distortion_coef.copy()[2:4]

    # Get distortion coefficients for radial only
    dist_coef_rad = camera.distortion_coef.copy()
    dist_coef_rad[2] = 0
    dist_coef_rad[3] = 0

    img_shape = camera.image_shape_xy

    # Calculate ideal x and y pixel maps (pinhole)
    mx, my = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))

    mx = mx.astype(np.float32)
    my = my.astype(np.float32)
    mx_ph = mx - np.float32(camera.intrinsic_mat[0, 2])
    my_ph = my - np.float32(camera.intrinsic_mat[1, 2])

    def calc_dx_dy(dist_coef):
        """Calculate distorted x and y pixel maps"""
        mx_cal, my_cal = cv.initUndistortRectifyMap(
            camera.intrinsic_mat, dist_coef, np.eye(3).astype(np.float32), camera.intrinsic_mat, img_shape, cv.CV_32FC1
        )

        mx_cal -= np.float32(camera.intrinsic_mat[0, 2])
        my_cal -= np.float32(camera.intrinsic_mat[1, 2])
        dx = mx_cal - mx_ph
        dy = my_cal - my_ph

        return dx, dy

    # Define pixel spacing between samples
    N = int(np.min(img_shape) / num_samps)  # pixels

    # Create pixel map relative to center
    x1 = y1 = int(N / 2)

    # Calculate distortions
    dx_rad, dy_rad = calc_dx_dy(dist_coef_rad)
    dx_tan, dy_tan = calc_dx_dy(dist_coef_tan)
    dx_tot, dy_tot = calc_dx_dy(camera.distortion_coef)

    # Plot radial distortion
    ax1.quiver(mx[y1::N, x1::N], my[y1::N, x1::N], dx_rad[y1::N, x1::N], dy_rad[y1::N, x1::N])
    ax1.set_ylim(0, img_shape[1])
    ax1.set_xlim(0, img_shape[0])
    ax1.set_xlabel("X (pixel)")
    ax1.set_ylabel("Y (pixel)")
    ax1.set_title("Radial Distortion")
    ax1.axis("image")
    ax1.grid()

    # Plot tangential distortion
    ax2.quiver(mx[y1::N, x1::N], my[y1::N, x1::N], dx_tan[y1::N, x1::N], dy_tan[y1::N, x1::N])
    ax2.set_ylim(0, img_shape[1])
    ax2.set_xlim(0, img_shape[0])
    ax2.set_xlabel("X (pixel)")
    ax2.set_ylabel("Y (pixel)")
    ax2.set_title("Tangential Distortion")
    ax2.axis("image")
    ax2.grid()

    # Plot total distortion
    ax3.quiver(mx[y1::N, x1::N], my[y1::N, x1::N], dx_tot[y1::N, x1::N], dy_tot[y1::N, x1::N])
    ax3.set_ylim(0, img_shape[1])
    ax3.set_xlim(0, img_shape[0])
    ax3.set_xlabel("X (pixel)")
    ax3.set_ylabel("Y (pixel)")
    ax3.set_title("Total Distortion")
    ax3.axis("image")
    ax3.grid()
