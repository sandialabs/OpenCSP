"""
Bundle adjustment algorithm based on example at:
https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

"""

from typing import Literal

import cv2 as cv
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

import opencsp.common.lib.tool.log_tools as lt


def bundle_adjust(
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    pts_obj: np.ndarray,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    pts_img: np.ndarray,
    intrinsic_mat: np.ndarray,
    dist_coefs: np.ndarray,
    opt_type: Literal["camera", "points", "both"],
    verbose: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform bundel adjustment algorithm on object points and camera poses.
    Nim = number of images
    Npts = number of points
    Nobs = number of observations

    Parameters
    ----------
    rvecs : np.ndarray
        (Nim,3) rvecs.
    tvecs : np.ndarray
        (Nim,3) tvecs.
    pts_obj : np.ndarray
        (Npts,3) 3D object point array.
    camera_indices : np.ndarray
        (Nobs,) array.
    point_indices : np.ndarray
        (Nobs,) array.
    pts_img : np.ndarray
        (Nobs,2) image point array.
    intrinsic_mat : np.ndarray
        (3,3) intrinsic camera matrix.
    dist_coefs : np.ndarray
        (n,) distortion coefficients array.
    opt_type : str
        What to optimize: {'camera', 'points', 'both'}
    verbose : int
        Level of verbosity of least squares solver [0, 1, 2]

    Returns
    -------
    rvecs_opt : np.ndarray
        (Nim,3) optimized rvecs.
    tvecs_opt : np.ndarray
        (Nim,3) optimized tvecs.
    pts_obj_opt : np.ndarray
        (Nimx,3) optimized object points.

    """
    # Check inputs
    if opt_type not in ["camera", "points", "both"]:
        raise ValueError(f'Given opt_type must be one of ("camera", "points", "both"), not "{opt_type:s}"')

    # Calculate number of cameras and points
    n_cameras = rvecs.shape[0]
    n_points = pts_obj.shape[0]

    # Define input parameters initial state
    params = np.hstack((rvecs, tvecs))
    x0 = np.hstack((params.ravel(), pts_obj.ravel()))

    # Create Jacobian sparsity structure
    jac_sparsity = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, opt_type)

    # Optimize
    res = least_squares(
        fun,
        x0,
        jac_sparsity=jac_sparsity,
        verbose=verbose,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        args=(n_cameras, n_points, camera_indices, point_indices, pts_img, intrinsic_mat, dist_coefs),
    )
    lt.debug("Bundle adjustment finished: " + res.message)

    # Return data
    data = res.x[: n_cameras * 6].reshape((n_cameras, 6))
    rvecs_opt = data[:, :3]
    tvecs_opt = data[:, 3:]
    pts_obj_opt = res.x[n_cameras * 6 :].reshape((n_points, 3))

    return rvecs_opt, tvecs_opt, pts_obj_opt


def rotate(points: np.ndarray, rot_vecs: np.ndarray):
    """
    Rotate points by given rotation vectors.

    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points: np.ndarray, camera_params: np.ndarray, intrinsic_mat: np.ndarray, dist_coefs: np.ndarray):
    """
    Convert 3-D points to 2-D by projecting onto camera sensor. A ray with
    normal incidence has a (0, 0) coordinate.

    """
    # Convert object points to local camera coordinates
    points_cam = rotate(points, camera_params[:, :3])
    points_cam += camera_params[:, 3:6]

    # Project
    rvec = np.array([0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 0.0])
    points_proj = cv.projectPoints(points_cam, rvec, tvec, intrinsic_mat, dist_coefs)[0][:, 0, :]

    return points_proj


def fun(
    params: np.ndarray,
    n_cameras: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    intrinsic_mat: np.ndarray,
    dist_coefs: np.ndarray,
) -> np.ndarray:
    """
    Compute reprojection errors in x and y.

    """
    camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], intrinsic_mat, dist_coefs)
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(
    n_cameras: int, n_points: int, camera_indices: np.ndarray, point_indices: np.ndarray, opt_type: str
):
    """
    Returns Jacobian sparsity structure.

    """
    m = camera_indices.size * 2  # num observed points (x and y)
    n = n_cameras * 6 + n_points * 3  # Nvars
    jac_sparsity = lil_matrix((m, n), dtype=int)  # Num xy points x num vars

    i = np.arange(camera_indices.size)

    # Fill in diagonals of Jacobian sparsity matrix
    if opt_type in ["camera", "both"]:
        for s in range(6):
            jac_sparsity[2 * i, camera_indices * 6 + s] = 1  # rotation
            jac_sparsity[2 * i + 1, camera_indices * 6 + s] = 1  # translation

    if opt_type in ["points", "both"]:
        for s in range(3):
            jac_sparsity[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            jac_sparsity[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return jac_sparsity
