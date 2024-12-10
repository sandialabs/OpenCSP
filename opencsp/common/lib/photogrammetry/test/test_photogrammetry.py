"""Tests functions in the phogrammetry library"""

import os
from os.path import join

import numpy as np
from numpy import nan
from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.photogrammetry import photogrammetry as ph


def get_test_camera() -> Camera:
    """Creates a test camera with focal length of 1 pixel"""
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=float)
    return Camera(mat, np.zeros(4), (100, 100), 'Test Camera')


def test_load_grayscale():
    img = ph.load_image_grayscale(join(os.path.dirname(__file__), 'data/image.png'))
    np.testing.assert_equal(np.ndim(img), 2)


def test_find_aruco_marker():
    img = ph.load_image_grayscale(join(os.path.dirname(__file__), 'data/image.png'))
    ids, corners = ph.find_aruco_marker(img, 7, 0.01)

    # Test IDs
    np.testing.assert_equal(np.array([20, 19, 18, 16, 15, 14, 17]), ids)
    # Test corners
    corns_exp = np.array(
        [
            [[330.0, 846.0], [470.0, 845.0], [474.0, 984.0], [334.0, 987.0]],
            [[759.0, 1049.0], [897.0, 1050.0], [898.0, 1190.0], [760.0, 1189.0]],
            [[757.0, 601.0], [892.0, 601.0], [896.0, 737.0], [761.0, 737.0]],
            [[1421.0, 891.0], [1502.0, 889.0], [1505.0, 969.0], [1424.0, 972.0]],
            [[1305.0, 546.0], [1386.0, 546.0], [1386.0, 627.0], [1305.0, 626.0]],
            [[1116.0, 543.0], [1197.0, 549.0], [1192.0, 629.0], [1111.0, 623.0]],
            [[1015.0, 872.0], [1095.0, 876.0], [1091.0, 957.0], [1011.0, 953.0]],
        ]
    )
    np.testing.assert_equal(corns_exp, np.array(corners))


def test_valid_camera_pose():
    camera = get_test_camera()
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    pts_image = np.zeros((1, 2), dtype=float)

    # Correct case
    pts_object = np.array([[0, 0, 1]], dtype=float)
    assert ph.valid_camera_pose(camera, rvec, tvec, pts_image, pts_object)

    # Too high reproj error
    pts_object = np.array([[150, 0, 1]], dtype=float)
    assert not ph.valid_camera_pose(camera, rvec, tvec, pts_image, pts_object)

    # Behind camera
    pts_object = np.array([[0, 0, -1]], dtype=float)
    assert not ph.valid_camera_pose(camera, rvec, tvec, pts_image, pts_object)


def test_reprojection_errors():
    camera = get_test_camera()
    rvecs = np.zeros((2, 3))
    tvecs = np.zeros((2, 3))
    pts_obj = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    camera_indices = np.array([0, 0, 1, 1])
    point_indices = np.array([0, 1, 0, 1])

    # Perfect case
    points_2d = np.zeros((4, 2))
    errors = ph.reprojection_errors(rvecs, tvecs, pts_obj, camera, camera_indices, point_indices, points_2d)
    np.testing.assert_equal(errors, np.zeros((4, 2)))

    # 1 pixel off in x on camera 0
    points_2d = np.zeros((4, 2))
    points_2d[:2, 0] = 1
    errors = ph.reprojection_errors(rvecs, tvecs, pts_obj, camera, camera_indices, point_indices, points_2d)
    errors_exp = np.zeros((4, 2))
    errors_exp[:2, 0] = -1
    np.testing.assert_equal(errors, errors_exp)


def test_align_points_no_scale():
    # Three point, no scale
    pts_obj_aligned = Vxyz(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
    rot = Rotation.from_rotvec(np.array([0.1, 0.05, 0.1]))
    pts_obj = pts_obj_aligned.rotate(rot) + Vxyz((0.1, 0.2, 0.3))
    vals = Vxyz(np.array(([0, nan, 0], [0, 0, nan], [0, 0, 0])))

    trans, scale, error = ph.align_points(pts_obj, vals)
    pts_obj_optimized = trans.apply(pts_obj)

    np.testing.assert_allclose(error, np.zeros(3), atol=1e-6, rtol=0)
    np.testing.assert_equal(scale, 1.0)
    np.testing.assert_allclose(pts_obj_aligned.data, pts_obj_optimized.data, atol=1e-6, rtol=0)


def test_align_points_with_scale():
    # Three point, with scale
    pts_obj_aligned = Vxyz(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
    rot = Rotation.from_rotvec(np.array([0.1, 0.05, 0.1]))
    pts_obj = pts_obj_aligned.rotate(rot) + Vxyz((0.1, 0.2, 0.3))
    vals = Vxyz(np.array(([0, 2, 0], [0, 0, 2], [0, 0, 0])), dtype=float)

    trans, scale, error = ph.align_points(pts_obj, vals, True)
    pts_obj_optimized = trans.apply(pts_obj * scale)

    np.testing.assert_allclose(error, np.zeros(3), atol=1e-6, rtol=0)
    np.testing.assert_almost_equal(scale, 2.0, 6)
    np.testing.assert_allclose(pts_obj_aligned.data * 2, pts_obj_optimized.data, atol=1e-6, rtol=0)


def test_scale_points():
    pts_obj = Vxyz(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
    point_ids = np.array([0, 1, 2])
    point_pairs = np.array([[0, 1], [0, 2], [1, 2]])
    dists = np.array([2, 2, np.sqrt(2) * 2])

    scales = ph.scale_points(pts_obj, point_ids, point_pairs, dists)
    np.testing.assert_allclose(scales, np.ones(3) * 2)


def test_dist_from_ray():
    v_pt = Vxyz([1.5, 0, 1])
    u_ray_dir = Vxyz([[0, 0], [0, 0], [1, 1]])
    v_ray_origin = Vxyz([[0, 1], [0, 0], [0, 0]])

    dists = ph.dist_from_rays(v_pt, u_ray_dir, v_ray_origin)
    np.testing.assert_allclose(dists, np.array([1.5, 0.5]), atol=1e-6, rtol=0)


def test_triangulate():
    cameras = [get_test_camera()] * 2
    rots = [Rotation.identity(), Rotation.identity()]
    tvecs = Vxyz([[0, 1], [0, 1], [0, 0]])
    pts_img = Vxy([[0, 1], [0, 1]])
    pt, dists = ph.triangulate(cameras, rots, tvecs, pts_img)

    np.testing.assert_allclose(pt.data.squeeze(), np.array([0, 0, 1]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(dists, np.array([0.0, 0.0]), rtol=0, atol=1e-6)


def test_nearest_ray_intersection():
    p_origins = Vxyz([[5, 0], [0, 1], [0, 0]])
    u_dirs = Uxyz([[-5, 0], [0, -1], [1, 1]])

    pt, dists = ph.nearest_ray_intersection(p_origins, u_dirs)

    np.testing.assert_allclose(pt.data.squeeze(), np.array([0, 0, 1]), atol=1e-6, rtol=0)
    np.testing.assert_allclose(dists, np.array([0, 0]), atol=1e-6, rtol=0)
