import cv2 as cv
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.log_tools as lt


def t_from_distance(Puv_cam: Vxy, dist: float, camera: Camera, v_cam_screen_cam: Vxyz) -> Vxyz:
    """
    Calculates the 3D point given a 2D camera pixel location and a distance
    from the center of the screen.

    Parameters
    ----------
    Puv_cam : Vxy
        The 2D point on the camera.
    dist : float
        Distance from the screen center, meters.
    camera : Camera
        Camera object.
    v_cam_screen_cam : Vxyz
        Camera to screen vector in camera coordinates.

    Returns
    -------
    v_cam_optic_cam : Vxyz
        Location of point in camera coordinates.

    """
    # Calculate pointing direction of centroid pixel
    u_cam = camera.vector_from_pixel(Puv_cam).as_Vxyz()

    # Calculate location of point relative to camera
    a = np.sqrt(u_cam.dot(v_cam_screen_cam) ** 2 - v_cam_screen_cam.dot(v_cam_screen_cam) + dist**2)
    cam_facet_dist = u_cam.dot(v_cam_screen_cam) + a

    # Calculate position of point relative to camera
    v_cam_optic_cam = u_cam * cam_facet_dist

    return v_cam_optic_cam


def r_from_position(v_cam_optic_cam: Vxyz, v_cam_screen_cam: Vxyz) -> Rotation:
    """
    Calculates the 3D rotation of a mirror given the relative locations of
    the mirror and screen from the camera.

    Parameters
    ----------
    v_cam_optic_cam : Vxyz
        Vector, camera to optic in camera coordinates.
    v_cam_screen_cam : Vxyz
        Vector, camera to screen in camera coordinates.

    Returns
    -------
    r_cam_optic : Rotation
        Rotation from camera to optic coordinates.

    """
    # Calculate screen to optic vector in camera coordinates
    v_screen_optic_cam = v_cam_optic_cam - v_cam_screen_cam

    # Calculate the optic normal (assuming crosshairs in the reflection)
    u_optic_screen_cam = -v_screen_optic_cam.normalize()
    u_optic_cam_cam = -v_cam_optic_cam.normalize()
    u_optic_norm = (u_optic_screen_cam + u_optic_cam_cam).normalize()

    # Calculate rotation from normal to camera
    r_align = u_optic_cam_cam.align_to(u_optic_norm)

    # Rotate points about approximate center (optic coordinates are flipped 180 about x axis)
    Rx = Rotation.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    r_cam_optic = r_align * Rx

    return r_cam_optic


def refine_v_distance(
    v_cam_optic_cam: Vxyz, dist_optic_screen: float, v_cam_screen_cam: Vxyz, v_meas_pt_optic_cam: Vxyz
) -> Vxyz:
    """
    Refines the camera to optic translation vector so that measured optic
    screen distance exactly matches calculated value.

    Parameters
    ----------
    v_cam_optic_cam : Vxyz
        Camera to optic vector in camera coordinates.
    dist_optic_screen : float
        Measured optic to screen distance.
    v_cam_screen_cam : Vxyz
        Camera to screen vector in camera coordinates.
    v_meas_pt_optic_cam : Vxyz
        Optic origin to measure point vector in camera coordintes.

    Returns
    -------
    Vxyz
        Scaled v_cam_optic_cam.

    """

    def error_func(scale):
        # Calculate the distance error
        v_cam_meas_pt_cam = (v_cam_optic_cam * scale) + v_meas_pt_optic_cam
        error = distance_error(v_cam_screen_cam, v_cam_meas_pt_cam, dist_optic_screen)
        return np.abs(error)

    # Perform optimization
    scale_0 = 1.0
    out = minimize(error_func, scale_0, method="Powell")

    # Return refined tvec
    return v_cam_optic_cam * out.x


def calc_rt_from_img_pts(pts_image: Vxy, pts_object: Vxyz, camera: Camera) -> tuple[Rotation, Vxyz]:
    """
    Calculates Translation and Rotation given object and image points.

    Parameters
    ----------
    pts_image : Vxy
        Points in image.
    pts_object : Vxyz
        Points in object coordinates.
    camera : Camera
        Camera object.

    Returns
    -------
    r_object_cam : Rotation
        Object to camera rotation.
    v_cam_object_cam : Vxyz
        Camera-to-object cector in camera coordinates.

    """
    ret, rvec, tvec = cv.solvePnP(pts_object.data.T, pts_image.data.T, camera.intrinsic_mat, camera.distortion_coef)

    if not ret:
        lt.error_and_raise(ValueError, "Could not find position of optic relative to camera.")

    return Rotation.from_rotvec(rvec.squeeze()), Vxyz(tvec.squeeze())


def calc_r_from_img_pts(
    Puv_image: Vxy, P_object: Vxyz, r_object_cam_0: Rotation, v_cam_object_cam: Vxyz, camera: Camera
) -> Rotation:
    """
    Calculates Rotation from points in image.

    Parameters
    ----------
    Puv_image : Vxy
        Points in mask image.
    P_object : Vxyz
        Points in "mirror coordinates".
    r_object_cam_0 : Rotation
        Initial guess of object_to_camera rotation.
    v_cam_object_cam : Vxyz
        Camera-to-object Vector in camera coordinates.
    camera : Camera
        Camera object.

    Returns
    -------
    Rotation
        Object-to-camera rotation.

    """

    def error_func(rvec):
        # Calculate reprojection error
        r_object_cam = Rotation.from_rotvec(rvec)
        reproj_error = reprojection_error(camera, P_object, Puv_image, r_object_cam, v_cam_object_cam)  # RSS pixels

        return reproj_error

    # Perform optimization
    out = minimize(error_func, r_object_cam_0.as_rotvec(), method="Powell")

    return Rotation.from_rotvec(out.x)


def distance_error(v_cam_screen_cam: Vxyz, v_cam_meas_pt_cam: Vxyz, dist: float) -> float:
    """
    Calculates optic to screen distance error as
    Error = MeasuredDistance - CalculatedDistance

    Parameters
    ----------
    v_cam_screen_cam : Vxyz
        Camera to screen vector in camera coordinates.
    v_cam_meas_pt_cam : Vxyz
        Camera to optic vector in camera coordinates.
    dist : float
        Measured camera-display distance.

    Returns
    -------
    float
        Distance error.

    """
    # Calculated distance to display
    v_optic_screen = v_cam_screen_cam - v_cam_meas_pt_cam
    dist_calc = v_optic_screen.magnitude()[0]  # meters

    # Calculate magnitude
    return dist - dist_calc  # meters


def reprojection_error(
    camera: Camera, P_object: Vxyz, Puv_image: Vxy, r_object_cam: Rotation, v_cam_object_cam: Vxyz
) -> float:
    """
    Calculates reprojection error as RMS pixels.

    Parameters
    ----------
    camera : Camera
        Camera object.
    P_object : Vxyz
        XYZ points in object coordinates.
    Puv_image : Vxy
        UV image points.
    r_object_cam : Rotation
        Object to camera rotation vector.
    v_cam_object_cam : Vxyz
        Camera to object translation vector.

    Returns
    -------
    float
        Reprojection error, RMS pixels.

    """
    # Project points to camera
    Puv_calc = camera.project(P_object, r_object_cam, v_cam_object_cam)

    # Calculate distance between points
    Puv_delta = Puv_calc - Puv_image
    return np.sqrt(np.mean(Puv_delta.magnitude() ** 2))  # pixels RMS
