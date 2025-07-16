"""Library of photogrammetry-related functions and algorithms"""

import cv2 as cv
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import opencsp.common.lib.photogrammetry.bundle_adjustment as ba
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
import opencsp.common.lib.tool.log_tools as lt


def load_image_grayscale(file: str) -> ndarray:
    """
    Loads image. Converts to grayscale if needed
    """
    # Load image
    img = cv.imread(file)

    # Convert to gray
    if np.ndim(img) == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        return img


def find_aruco_marker(
    image: ndarray, adaptiveThreshConstant: float = 10, minMarkerPerimeterRate: float = 0.01
) -> tuple[ndarray[int], list[ndarray]]:  # ,
    """
    Finds aruco marker corners in given image to the nearest pixel.

    Parameters
    ----------
    image : ndarray
        2D grayscale image.
    adaptiveThreshConstant : float, optional
        aruco parameter. The default is 10.
    minMarkerPerimeterRate : float, optional
        aruco parameter. The default is 0.01.

    Returns
    -------
    ids : ndarray[int, ...]
        1D array of IDs of aruco markers seen in each image.
    pts : list[ndarray, ...]
        List of 4x2 image point arrays of all four corners of aruco markers
        seen by each camera.

    """
    # Setup detection parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    aruco_detect_params = aruco.DetectorParameters()
    aruco_detect_params.adaptiveThreshConstant = adaptiveThreshConstant
    aruco_detect_params.minMarkerPerimeterRate = minMarkerPerimeterRate

    # Find targets
    arcuoDetector = aruco.ArucoDetector(aruco_dict, detectorParams=aruco_detect_params)
    (corners, ids, _) = arcuoDetector.detectMarkers(image)

    # Refine corner locations (inaccurate using cv.cornerSubPix)
    #     criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, max_iterations, precision)
    #     corners_orig = []
    #     for corner in corners:
    #         corners_orig.append(corner.copy())
    #         cv.cornerSubPix(image, corner, window_size, (-1, -1), criteria)

    # Reshape and resize
    pts = []
    for corner in corners:
        pts.append(corner[0, :, :])

    return ids.squeeze().astype(int), pts


def valid_camera_pose(
    camera: Camera, rvec: ndarray, tvec: ndarray, pts_image: ndarray, pts_object: ndarray, reproj_thresh: float = 100.0
) -> bool:
    """
    Returns image IDs that have points behind the camera or have high
    reprojection error.

    Parameters
    ----------
    camera : opencsp.common.lib.camera.Camera.Camera
        Camera object.
    rvecs : ndarray
        Shape (3,) rvec array.
    tvecs : ndarray
        Shape (3,) tvec array.
    pts_image : ndarray
        Nx2 array of 2d image points
    pts_object : ndarray
        Nx3 array of 3d points to project.
    reproj_thresh : float, optional
        Threshold reprojection error. The default is 100.

    Returns
    -------
    bool
        True if camera pose is valid

    """
    # Calculate reprojection errors
    rot = Rotation.from_rotvec(rvec)
    pts_uv = camera.project_mat(pts_object, rot.as_rotvec(), tvec)
    error = np.sqrt(np.sum((pts_uv - pts_image) ** 2, 1))

    # Calculate mean point distances in front of camera
    pts_cam = tvec[None, :] + rot.apply(pts_object)
    pts_cam_z = pts_cam[:, 2]

    # Check if z < 0 or reprojection error is large
    valid = True
    if pts_cam_z.min() < 0:
        lt.debug("Object points located behind camera during camera pose calculation.")
        valid = False
    if error.max() > reproj_thresh:
        lt.debug(f"Reprojection error above {reproj_thresh:.2f} during camera pose calculation")
        valid = False
    return valid


def reprojection_errors(
    rvecs: ndarray,
    tvecs: ndarray,
    pts_obj: ndarray,
    camera: Camera,
    camera_indices: ndarray,
    point_indices: ndarray,
    points_2d: ndarray,
) -> ndarray:
    """
    Calculates the xy reprojection error for each point.
    See bundle_adjustment.bundle_adjust() for more information.

    Returns
    -------
    ndarray
        Num observations x 2 ndarray.

    """
    # Calculate errors
    a = np.hstack((rvecs, tvecs))
    x = np.hstack((a.ravel(), pts_obj.ravel()))
    pts = ba.fun(
        x,
        rvecs.shape[0],
        pts_obj.shape[0],
        camera_indices,
        point_indices,
        points_2d,
        camera.intrinsic_mat,
        camera.distortion_coef,
    )

    return pts.reshape((-1, 2))


def plot_pts_3d(ax: plt.Axes, pts_obj: ndarray, rots: list[Rotation], tvecs: Vxyz, needle_length: float = 1) -> None:
    """
    Plots 3D points and camera poses (points with needles defined by rvec/tvec).

    Parameters
    ----------
    ax : plt.Axes
        3D axes to plot on.
    pts_obj : ndarray
        Nx3 object points.
    rots : list[Rotation]
        N rotation objects.
    tvecs : Vxyz
        N tvecs.
    needle_length : float, optional
        Length of camera pointing needles, meters. The default is 1.

    """
    # Plot points
    ax.scatter3D(*(pts_obj).T)
    for i in range(pts_obj.shape[0]):
        ax.text3D(*(pts_obj[i]), i)
    # Camera locations
    for idx, (v_cam, rot_obj_cam) in enumerate(zip(tvecs, rots, strict=True)):
        rot_cam_obj = rot_obj_cam.inv()
        # Point
        vec_obj = -v_cam.rotate(rot_cam_obj)
        ax.scatter3D(*vec_obj.data.squeeze(), color="k")
        # Pose arrow
        vec_obj_pose = Pxyz((0, 0, needle_length)).rotate(rot_cam_obj)
        xs = [vec_obj.x[0], vec_obj.x[0] + vec_obj_pose.x[0]]
        ys = [vec_obj.y[0], vec_obj.y[0] + vec_obj_pose.y[0]]
        zs = [vec_obj.z[0], vec_obj.z[0] + vec_obj_pose.z[0]]
        ax.plot(xs, ys, zs, color="red")
        # Text
        ax.text3D(*vec_obj.data.squeeze(), idx)
    # Axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def align_points(pts_obj: Vxyz, vals: Vxyz, scale: bool = False) -> tuple[TransformXYZ, float, ndarray[float]]:
    """
    Returns 2D homogeneous transform to apply to input data according to
    alignment criteria. Values are scaled (if applicable) FIRST, then
    spatially transformed.

    Parameters
    ----------
    pts_obj : Vxyz
        Object points, meters.
    vals : Vxyz
        ``[(x1, y1, z1), (x2, y2, z2), ...]`` The expected coordinate values
        of each coordinate index that correspond to points in pts_obj.
        If a coordinate is to be ignored, set it to np.nan.

        For example: ``[(np.nan, 0, np.nan), (np.nan, 0, np.nan), (0, 0, np.nan)]``
    scale : bool
        To apply a scaling factor to points (default False)

    Returns
    -------
    TransformXYZ, float, ndarray[float]
        Point cloud Transform object
        Point cloud scale factor
        Point alignment error, meters
    """

    def calc_point_errors(vec: ndarray):
        """Calculate alignment errors for each point

        vec[0:3] - rotation vector
        vec[3:6] - translation vector
        vec[6]   - scale

        """
        # Rotate/shift the points
        rov = Rotation.from_rotvec(vec[0:3])
        trans = Vxyz(vec[3:6])
        # Apply scale factor
        if scale:
            v_proc = pts_obj * vec[6]
        else:
            v_proc = pts_obj
        # Apply transformation
        v_proc = v_proc.rotate(rov) + trans
        # Calculate the coordinate error
        return _ref_coord_error(v_proc, vals)

    def align_merit_fcn(vec: ndarray):
        """Merit function for alignment optimization function"""
        # Calcule alinment error for each point
        e = calc_point_errors(vec)

        # Calculate RMS error
        return np.sqrt(np.mean(e**2))

    # Optimize points
    if scale:
        vec = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)
    else:
        vec = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    x = minimize(align_merit_fcn, vec, method="Powell")

    # Calculate final alignment error
    e_final = calc_point_errors(x.x)

    # Return transform and scale
    rot_out = Rotation.from_rotvec(x.x[:3])
    trans_out = Vxyz(x.x[3:6])
    scale_out = 1.0 if not scale else x.x[6]
    return TransformXYZ.from_R_V(rot_out, trans_out), scale_out, e_final


def _ref_coord_error(pts_obj: Vxyz, pts_exp: Vxyz) -> np.ndarray:
    """
    Calculates errors in expected coordinate values of a set of indices
    and expected coordinate values. Returns the RSS of errors in x, y, and z
    for each index given ignoring NaNs.

    Parameters
    ----------
    pts_obj : Vxyz
        Length N object points vector, meters
    pts_exp : Vxyz
        Length N expected coordinate vector, meters or nan for unspecified.

    Returns
    -------
    ndarray
        Errors in coordinate values for every index, meters

    """
    error = []
    for pt_obj, coord in zip(pts_obj.data.T, pts_exp.data.T):
        e: ndarray = pt_obj - coord
        e = np.nan_to_num(e, nan=0.0)
        error.append(np.sqrt(np.sum(e**2)))
    return np.array(error)


def scale_points(pts_obj: Vxyz, point_ids: ndarray, point_pairs: ndarray, dists: ndarray) -> ndarray[float]:
    """
    Scales object points and tvecs. A list of point pairs is given, and the
    corresponding expected distance between them. The object points are scaled
    to the average of the calculated scales.

    Parameters
    ----------
    pts_obj : Vxyz
        Object xyz points, meters.
    point_ids : ndarray
        1d array, point IDs corresponding to pts_obj.
    point_pairs : ndarray
        2d array of form: [[a1, a2], [b1, b2], ...], point pairs.
    dists : ndarray
        1d array of expected distances between point pairs, meters.

    Returns
    -------
    scales : ndarray
        Calculated scale factor for each point pair.

    """
    # Calculate distances between all scale index pairs
    dists_cur = []
    for i1, i2 in point_pairs:
        mask_1 = point_ids == i1
        mask_2 = point_ids == i2
        d = pts_obj[mask_1] - pts_obj[mask_2]  # xyz delta, meters
        dists_cur.append(d.magnitude()[0])  # meters

    # Calculate ideal scales for each index pair
    return np.array(dists) / np.array(dists_cur)


def dist_from_rays(v_pt: Vxyz, u_ray_dir: Vxyz | Uxyz, v_ray_ori: Vxyz) -> ndarray:
    """
    Calculates perpendicular distances from a point to N rays.

    Parameters
    ----------
    v_pt : Vxyz
        Length 1, Point to calculate distance to.
    u_ray_dir : Vxyz | Uxyz
        Length N, Ray pointing directions. Must be a UNIT VECTOR.
    v_ray_ori : Vxyz
        Length N, Ray origin points.

    Returns
    -------
    np.ndarray
        Length N array of distances, meters.

    """
    # Calculate vector from camera to intersection point
    v_cam_int_pt_screen = v_pt - v_ray_ori

    # Calculate point along unit vector that is closest to given intersection point
    if isinstance(u_ray_dir, Uxyz):
        u_ray_dir = u_ray_dir.as_Vxyz()
    scales = u_ray_dir.dot(v_cam_int_pt_screen)
    v_cam_pt_screen = u_ray_dir * scales[None, :]

    # Calculate perpendicular distance from intersection point to rays
    return (v_cam_int_pt_screen - v_cam_pt_screen).magnitude()


def triangulate(
    cameras: list[Camera], rots: list[Rotation], tvecs: Vxyz | list[Vxyz], pts_img: Vxy | list[Vxy]
) -> tuple[Vxyz, ndarray]:
    """Triangulates position of unknown marker.

    Parameters
    ----------
    cameras : list[opencsp.common.lib.camera.Camera.Camera]
        N Camera objects used to capture images
    rots : list[Rotation]
        N world to camera Rotations
    tvecs : Vxyz | list[Vxyz]
        N camera to world translation vectors (camera coordinates)
    pts_img : Vxy | list[Vxy]
        N Vxy image points

    Returns
    -------
    tuple[Vxyz, ndarray]
        Intersection point, perpendicular distances from point to rays
    """
    # Create data containers
    pts_origins = np.zeros((3, len(cameras)))  # location of cameras
    u_rays = np.zeros((3, len(cameras)))  # direction of rays

    # Collect rays/origins and convert to lab reference frame
    for idx, (camera, rvec, tvec, pt_img) in enumerate(zip(cameras, rots, tvecs, pts_img)):
        # Calculate camera position and rays in object reference frame
        r_cam_world = rvec.inv()
        v_world_cam_world = -tvec.rotate(r_cam_world)
        u_cam = camera.vector_from_pixel(pt_img)
        u_world = u_cam.rotate(r_cam_world)

        # Store data
        pts_origins[:, idx] = v_world_cam_world.data.squeeze()
        u_rays[:, idx] = u_world.data.squeeze()

    # Convert to Vxyz
    pts_origins = Vxyz(pts_origins)
    u_rays = Vxyz(u_rays)

    # Intersect rays for current point from each camera to find point location
    pt_int, dists = nearest_ray_intersection(pts_origins, u_rays)

    return pt_int, dists


def nearest_ray_intersection(p_origins: Vxyz, u_dirs: Vxyz | Uxyz) -> tuple[Vxyz, ndarray]:
    """
    Finds the least squares point of intersection between N skew rays. And
    calculates residuals.
    Source: https://en.wikipedia.org/wiki/Line-line_intersection

    Parameters
    ----------
    p_origins : Vxyz
        Vector of XYZ origin of each ray.
    u_dirs : Vxyz | Uxyz
        Vector of unit pointing direction vectors. Must be UNIT VECTORS.

    Returns
    -------
    Vxyz, ndarray
        Least squares XYZ intersection point, perpendicular distances from point to rays

    """
    # Format pointing directions: dir(3x1) * dir(3x1)^T = (3x3) array
    # (N, 3, 3)  =  (N, 3, 1)             x (N, 1, 3)
    u_dirs_mat = u_dirs.data.T[:, :, np.newaxis] @ u_dirs.data.T[:, np.newaxis, :]

    # Format ray points
    # (N, 3)      = (N, 3, 1)
    p_origins_mat = p_origins.data.T[:, :, np.newaxis]

    # Find least squares solution to: Ax = b
    #    A: sum(I - v*v_T)     [summed over N rays] -> (3, 3) array
    #    b: sum((I - v*v_T)*p) [summed over N rays] -> (3, 1) array
    #    x: 3d point                                -> (3 ,1) array
    i_mat = np.eye(3)
    p_int = np.linalg.lstsq(
        a=(i_mat - u_dirs_mat).sum(axis=0), b=((i_mat - u_dirs_mat) @ p_origins_mat).sum(axis=0), rcond=None
    )[0]

    # Calculate intersection errors (perpendicular distances to rays)
    p_int = Vxyz(p_int.squeeze())
    dists = dist_from_rays(p_int, u_dirs, p_origins)

    return p_int, dists
