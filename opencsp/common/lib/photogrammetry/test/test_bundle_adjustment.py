import numpy as np
from numpy import array

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.photogrammetry import bundle_adjustment as ba


def get_test_camera() -> Camera:
    """Creates a test camera with focal length of 1 pixel"""
    mat = np.array([[1000, 0, 1000], [0, 1000, 1000], [0, 1, 1]], dtype=float)
    return Camera(mat, np.zeros(4), (100, 100), "Test Camera")


def test_bundle_adjust_points():
    camera = get_test_camera()
    rvecs = np.zeros((2, 3))
    rvecs[1, 1] = -np.pi / 4
    tvecs = array([[0, 0, 0], [10, 0, 0]], dtype=float)
    pts_obj = array([[0, 0, 10], [1, 0, 10], [-1, 0, 10]], dtype=float)
    cam_indices = array([0, 0, 0, 1, 1, 1])
    point_indices = array([0, 1, 2, 0, 1, 2])
    int_mat = camera.intrinsic_mat
    dist_coefs = camera.distortion_coef

    # Calculate point locations
    pts_img_0 = camera.project_mat(pts_obj, rvecs[0], tvecs[0])
    pts_img_1 = camera.project_mat(pts_obj, rvecs[1], tvecs[1])
    pts_img_ideal = np.concatenate((pts_img_0, pts_img_1), axis=0)
    # Add errors
    np.random.seed(0)
    pts_img = pts_img_ideal + np.random.randn(*pts_img_ideal.shape) * 0.1
    rvecs_in = rvecs.copy()
    tvecs_in = tvecs.copy()

    rvecs_out, tvecs_out, pts_obj_out = ba.bundle_adjust(
        rvecs_in, tvecs_in, pts_obj, cam_indices, point_indices, pts_img, int_mat, dist_coefs, "points", verbose=True
    )

    np.testing.assert_allclose(rvecs_out, rvecs, atol=1e-6, rtol=0)
    np.testing.assert_allclose(tvecs_out, tvecs, atol=1e-6, rtol=0)
    np.testing.assert_allclose(pts_obj_out, pts_obj, atol=1e-2, rtol=0)


def test_bundle_adjust_camera():
    camera = get_test_camera()
    rvecs = np.zeros((2, 3))
    rvecs[1, 1] = -np.pi / 4
    tvecs = array([[0, 0, 0], [10, 0, 0]], dtype=float)
    pts_obj = array([[0, 0, 10], [1, 0, 10], [-1, 0, 10]], dtype=float)
    cam_indices = array([0, 0, 0, 1, 1, 1])
    point_indices = array([0, 1, 2, 0, 1, 2])
    int_mat = camera.intrinsic_mat
    dist_coefs = camera.distortion_coef

    # Calculate point locations
    pts_img_0 = camera.project_mat(pts_obj, rvecs[0], tvecs[0])
    pts_img_1 = camera.project_mat(pts_obj, rvecs[1], tvecs[1])
    pts_img_ideal = np.concatenate((pts_img_0, pts_img_1), axis=0)
    # Add errors
    np.random.seed(0)
    pts_img = pts_img_ideal
    rvecs_in = rvecs.copy() + np.random.randn(*rvecs.shape) * 0.001
    tvecs_in = tvecs.copy() + np.random.randn(*tvecs.shape) * 0.01

    rvecs_out, tvecs_out, pts_obj_out = ba.bundle_adjust(
        rvecs_in, tvecs_in, pts_obj, cam_indices, point_indices, pts_img, int_mat, dist_coefs, "camera", verbose=True
    )

    np.testing.assert_allclose(rvecs_out, rvecs, atol=1e-3, rtol=0)
    np.testing.assert_allclose(tvecs_out, tvecs, atol=1e-2, rtol=0)
    np.testing.assert_allclose(pts_obj_out, pts_obj, atol=1e-6, rtol=0)
