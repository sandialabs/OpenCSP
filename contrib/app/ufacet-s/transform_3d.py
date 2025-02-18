"""
Transforms, Included Rotation and Translation
"""

import math
import numpy as np
from warnings import warn


def axisrotation(unit_vector, angle):  # ?? SCAFFOLDING RCB -- ADD UNDERSCORE BETWEEN "AXIS" AND "ROTATION"
    """
    DEPRECATED: Calculates the rotation matrix for a given angle around a specified axis.

    This function uses the right-hand rule to compute the rotation matrix based on
    the provided unit vector and angle. The angle is expected to be in radians.

    Parameters
    ----------
    unit_vector : np.ndarray
        A 3D unit vector representing the axis of rotation.
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix corresponding to the rotation around the specified axis.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be replaced with scipy.spatial.transform.Rotation.
    AssertionError
        If the input unit_vector is not of unit length.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "transform_3d.axisrotation is deprecated. Replace with scipy.spatial.transform.Rotation",
        DeprecationWarning,
        stacklevel=2,
    )
    # The original version of the routine used the left-hand rule.
    # This is reflected in the detailed code below.  Here we negate
    # the angle so that we use the right-hand rule.
    #
    # lhr_angle = "left-hand rule angle"
    #
    lhr_angle = -angle

    ux = unit_vector[0]
    uy = unit_vector[1]
    uz = unit_vector[2]

    # The input vector must be a unit vector.
    norm = np.sqrt(ux**2 + uy**2 + uz**2)
    if abs(norm - 1.0) > 1e-9:  # tolerance
        print("ERROR: In axisrotation(), input unit_vector =", unit_vector, " is not of unit length.  Length =", norm)

    c = np.cos(lhr_angle)
    s = np.sin(lhr_angle)
    t = 1 - c

    R1 = [t * ux * ux + c, t * ux * uy + uz * s, t * ux * uz - uy * s]
    R2 = [t * ux * uy - uz * s, t * uy * uy + c, t * uy * uz + ux * s]
    R3 = [t * ux * uz + uy * s, t * uy * uz - ux * s, t * uz * uz + c]

    R = [R1, R2, R3]
    return np.array(R)


def rotation_matrix_to_euler_angles(R):
    """
    DEPRECATED: Converts a rotation matrix to Euler angles.

    This function computes the Euler angles corresponding to the given rotation matrix.
    The order of the angles is such that the x and z angles are swapped compared to MATLAB.

    Parameters
    ----------
    R : np.ndarray
        A 3x3 rotation matrix.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the Euler angles (rot_x, rot_y, rot_z) in radians.

    Raises
    ------
    AssertionError
        If the input matrix is not a valid rotation matrix.
    DeprecationWarning
        Indicates that this function is deprecated and should be replaced with scipy.spatial.transform.Rotation.

    Notes
    -----
    See https://learnopencv.com/rotation-matrix-to-euler-angles/
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "transform_3d.rotation_matrix_to_euler_angles is deprecated. Replace with scipy.spatial.transform.Rotation.",
        DeprecationWarning,
        stacklevel=2,
    )
    assert is_rotation_matrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        rot_x = math.atan2(R[2, 1], R[2, 2])
        rot_y = math.atan2(-R[2, 0], sy)
        rot_z = math.atan2(R[1, 0], R[0, 0])
    else:
        rot_x = math.atan2(-R[1, 2], R[1, 1])
        rot_y = math.atan2(-R[2, 0], sy)
        rot_z = 0
    # return np.array([x, y, z])  # Original code.
    return rot_x, rot_y, rot_z


def is_rotation_matrix(R):
    """
    DEPRECATED: Checks if a matrix is a valid rotation matrix.

    A valid rotation matrix must be orthogonal and have a determinant of 1.

    Parameters
    ----------
    R : np.ndarray
        A 3x3 matrix to check.

    Returns
    -------
    bool
        True if the matrix is a valid rotation matrix, False otherwise.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be replaced with scipy.spatial.transform.Rotation.

    Notes
    -----
    See https://learnopencv.com/rotation-matrix-to-euler-angles/
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "transform_3d.is_rotation_matrix is deprecated. Replace with scipy.spatial.transform.Rotation.",
        DeprecationWarning,
        stacklevel=2,
    )
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
