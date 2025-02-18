"""
3-d Geometry Utiltiies



"""

import math
import numpy as np
import scipy.linalg as sla
from warnings import warn


def direction_uxyz_given_azimuth_elevation(azimuth: float, elevation: float):  # Both radians.
    """
    DEPRECATED: Calculates the direction vector in 3D space given azimuth and elevation angles.

    This function converts azimuth and elevation angles (in radians) into a unit vector
    in 3D space.

    Parameters
    ----------
    azimuth : float
        The azimuth angle in radians, measured from the positive x-axis.
    elevation : float
        The elevation angle in radians, measured from the xy-plane.

    Returns
    -------
    np.ndarray
        A 3D unit vector representing the direction corresponding to the given azimuth and elevation.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    DEPRECATED: Calculates the direction vector in 3D space given azimuth and elevation angles.

    This function converts azimuth and elevation angles (in radians) into a unit vector
    in 3D space.

    Parameters
    ----------
    azimuth : float
        The azimuth angle in radians, measured from the positive x-axis.
    elevation : float
        The elevation angle in radians, measured from the xy-plane.

    Returns
    -------
    np.ndarray
        A 3D unit vector representing the direction corresponding to the given azimuth and elevation.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.direction_uxyz_given_azimuth_elevation is deprecated. This function should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Convert to degrees ccw from East.
    nu = (np.pi / 2.0) - azimuth
    # Construct the vector.
    z: float = np.sin(elevation)
    r: float = np.cos(elevation)
    x: float = r * np.cos(nu)
    y: float = r * np.sin(nu)
    uxyz = np.array([x, y, z])
    # Return.
    return uxyz


def distance_between_xyz_points(xyz_1, xyz_2):
    """
    DEPRECATED: Calculates the Euclidean distance between two points in 3D space.

    Parameters
    ----------
    xyz_1 : np.ndarray
        The coordinates of the first point (x1, y1, z1).
    xyz_2 : np.ndarray
        The coordinates of the second point (x2, y2, z2).

    Returns
    -------
    float
        The Euclidean distance between the two points.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should use Vxyz subtraction/magnitude instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    DEPRECATED: Calculates the Euclidean distance between two points in 3D space.

    Parameters
    ----------
    xyz_1 : np.ndarray
        The coordinates of the first point (x1, y1, z1).
    xyz_2 : np.ndarray
        The coordinates of the second point (x2, y2, z2).

    Returns
    -------
    float
        The Euclidean distance between the two points.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should use Vxyz subtraction/magnitude instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.distance_between_xyz_points is deprecated. Use Vxyz subtraction/magnitude instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    x1 = xyz_1[0]
    y1 = xyz_1[1]
    z1 = xyz_1[2]
    x2 = xyz_2[0]
    y2 = xyz_2[1]
    z2 = xyz_2[2]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# -------------------------------------------------------------------------------------------------------
# 3-D VECTOR CALCULATIONS
#


def vector_3d_cross_product(vxyz_1, vxyz_2):
    """
    DEPRECATED: Calculates the cross product of two 3D vectors.

    Parameters
    ----------
    vxyz_1 : np.ndarray
        The first vector represented as an array-like object [x1, y1, z1].
    vxyz_2 : np.ndarray
        The second vector represented as an array-like object [x2, y2, z2].

    Returns
    -------
    list[float]
        The resulting vector from the cross product, represented as [x, y, z].

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should use Vxyz.cross() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    DEPRECATED: Calculates the cross product of two 3D vectors.

    Parameters
    ----------
    vxyz_1 : np.ndarray
        The first vector represented as an array-like object [x1, y1, z1].
    vxyz_2 : np.ndarray
        The second vector represented as an array-like object [x2, y2, z2].

    Returns
    -------
    list[float]
        The resulting vector from the cross product, represented as [x, y, z].

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should use Vxyz.cross() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.vector_3d_cross_product is deprecated. Use Vxyz.cross() instead.", DeprecationWarning, stacklevel=2
    )
    return list(
        np.cross(np.array(vxyz_1), np.array(vxyz_2))
    )  # ?? SCAFFOLDING RCB -- THESE ARRAY-LIST CONVERSIONS ARE YET ANOTHER ARGUMENT FOR USING NUMPY ARRAYS INSTEAD OF LISTS.  OR BETTER TO MAKE A VECTOR CLASS?


def vector_3d_norm(vxyz):
    """
    DEPRECATED: Calculates the norm (magnitude) of a 3D vector.

    Parameters
    ----------
    vxyz : np.ndarray
        The vector represented as an array-like object [x, y, z].

    Returns
    -------
    float
        The magnitude of the vector.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should use Vxyz.magnitude() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    DEPRECATED: Calculates the norm (magnitude) of a 3D vector.

    Parameters
    ----------
    vxyz : np.ndarray
        The vector represented as an array-like object [x, y, z].

    Returns
    -------
    float
        The magnitude of the vector.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should use Vxyz.magnitude() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn("geometry_3d.vector_3d_norm is deprecated. Use Vxyz.magnitude() instead.", DeprecationWarning, stacklevel=2)
    return np.sqrt(vxyz[0] ** 2 + vxyz[1] ** 2 + vxyz[2] ** 2)


def normalize_vector_3d(vxyz):
    """
    DEPRECATED: Normalizes a 3D vector to unit length.

    Parameters
    ----------
    vxyz : np.ndarray
        The vector represented as an array-like object [x, y, z].

    Returns
    -------
    list[float]
        The normalized vector represented as [x', y', z'].

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should use Vxyz.normalize() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    DEPRECATED: Normalizes a 3D vector to unit length.

    Parameters
    ----------
    vxyz : np.ndarray
        The vector represented as an array-like object [x, y, z].

    Returns
    -------
    list[float]
        The normalized vector represented as [x', y', z'].

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should use Vxyz.normalize() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.normalize_vector_3d is deprecated. Use Vxyz.normalize() instead.", DeprecationWarning, stacklevel=2
    )
    norm = vector_3d_norm(vxyz)
    return [vxyz[0] / norm, vxyz[1] / norm, vxyz[2] / norm]


# -------------------------------------------------------------------------------------------------------
# BEST-FIT PLANE TO 3-D POINTS
#


# Routines that compute the best-fit plane of a set of (x,y,z) points.
# This code taken from:
# <Experiments_dir>\2020-08-01_0415_OpenApertureStudy\image_analysis\2020-08-03_0617\image_from_file_modified.py
#
# Later on the code uses version B, which indicates that at the time,
# I decided that B was a better choice for that data.  This data is similar,
# so we recommend version B here.  More evaluation is needed.
#
# def best_fit_plane_A(xs, ys, zs):
#     # do fit
#     tmp_A = []
#     tmp_b = []
#     for i in range(len(xs)):
#         tmp_A.append([xs[i], ys[i], 1])
#         tmp_b.append(zs[i])
#     b = np.matrix(tmp_b).T
#     A = np.matrix(tmp_A)
#     print('A:')
#     print(A)
#     print('b:')
#     print(b)
#     fit = (A.T * A).I * A.T * b
#     errors = b - A * fit
#     residual = np.linalg.norm(errors)
#
#     print("solution:")
#     print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
#     print("errors:")
#     print(errors)
#     print("residual:")
#     print(residual)
#
#     a = fit[0,0]
#     b = fit[1,0]
#     c = fit[2,0]
#     return a, b, c
#
def best_fit_plane_B(xs, ys, zs):
    """
    DEPRECATED: Calculates the coefficients of the best-fit plane for a set of 3D points.

    This function computes the best-fit plane that minimizes the distance to a set of points in 3D space.

    Parameters
    ----------
    xs : list[float]
        A list of x-coordinates of the points.
    ys : list[float]
        A list of y-coordinates of the points.
    zs : list[float]
        A list of z-coordinates of the points.

    Returns
    -------
    list[float]
        A list containing the coefficients [A, B, C] of the plane equation Ax + By + Cz + D = 0.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    DEPRECATED: Calculates the coefficients of the best-fit plane for a set of 3D points.

    This function computes the best-fit plane that minimizes the distance to a set of points in 3D space.

    Parameters
    ----------
    xs : list[float]
        A list of x-coordinates of the points.
    ys : list[float]
        A list of y-coordinates of the points.
    zs : list[float]
        A list of z-coordinates of the points.

    Returns
    -------
    list[float]
        A list containing the coefficients [A, B, C] of the plane equation Ax + By + Cz + D = 0.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.best_fit_plane_B is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ?? SCAFFOLDING RCB -- IS THIS A BUG, SINCE IT DOESN'T RETURN D COEFFICIENT?
    # ?? SCAFFOLDING RCB -- SHOULD CALLERS BE UPDATED?
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    # TODO RCB: Delete the following deprecated code using np.matrix() once it's clear that it is working properly.
    # Message:
    # PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linears.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
    # b = np.matrix(tmp_b).T
    # A = np.matrix(tmp_A)
    b = np.array(tmp_b).T
    A = np.array(tmp_A)
    fit, residual, rnk, s = sla.lstsq(A, b)

    #     print("solution:")
    #     print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    #     print("residual:")
    #     print(residual)

    # TODO RCB: Delete the following deprecated code using np.matrix() once it's clear that it is working properly.
    # A = fit[0,0]
    # B = fit[1,0]
    # C = fit[2,0]
    A = fit[0]
    B = fit[1]
    C = fit[2]
    return [A, B, C]


def best_fit_plane(xyz_list):
    """
    DEPRECATED: Calculates the coefficients of the best-fit plane for a sequence of 3D points.

    This function converts the list of 3D points into separate coordinate lists and
    computes the best-fit plane coefficients.

    Parameters
    ----------
    xyz_list : list[list[float]]
        A list of points where each point is represented as [x, y, z].

    Returns
    -------
    list[float]
        A list containing the coefficients [A, B, C] of the plane equation Ax + By + Cz + D = 0.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    DEPRECATED: Calculates the coefficients of the best-fit plane for a sequence of 3D points.

    This function converts the list of 3D points into separate coordinate lists and
    computes the best-fit plane coefficients.

    Parameters
    ----------
    xyz_list : list[list[float]]
        A list of points where each point is represented as [x, y, z].

    Returns
    -------
    list[float]
        A list containing the coefficients [A, B, C] of the plane equation Ax + By + Cz + D = 0.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.best_fit_plane is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Convert to individual coordinate lists.
    xs = [p[0] for p in xyz_list]
    ys = [p[1] for p in xyz_list]
    zs = [p[2] for p in xyz_list]
    # We prefer version B. See above.
    return best_fit_plane_B(xs, ys, zs)


# -------------------------------------------------------------------------------------------------------
# HOMOGEONEOUS PLANE CALCULATIONS
#


def flip_homogeneous_plane(plane):
    """
    Reverses the sense of the homogeneous plane.

    Parameters
    ----------
    plane : list[float]
        The coefficients of the plane in homogeneous coordinates [A, B, C, D].

    Returns
    -------
    list[float]
        The coefficients of the flipped plane.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    Reverses the sense of the homogeneous plane.

    Parameters
    ----------
    plane : list[float]
        The coefficients of the plane in homogeneous coordinates [A, B, C, D].

    Returns
    -------
    list[float]
        The coefficients of the flipped plane.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.flip_homogeneous_plane is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Reverse the sense of the homogeneous plane.
    return [-x for x in plane]


def homogeneous_plane_signed_distance_to_xyz(xyz, plane):
    """
    DEPRECATED: Calculates the signed distance from a point to a homogeneous plane.

    Parameters
    ----------
    xyz : list[float]
        The coordinates of the point (x, y, z).
    plane : list[float]
        The coefficients of the plane in homogeneous coordinates [A, B, C, D].

    Returns
    -------
    float
        The signed distance from the point to the plane.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    """
    DEPRECATED: Calculates the signed distance from a point to a homogeneous plane.

    Parameters
    ----------
    xyz : list[float]
        The coordinates of the point (x, y, z).
    plane : list[float]
        The coefficients of the plane in homogeneous coordinates [A, B, C, D].

    Returns
    -------
    float
        The signed distance from the point to the plane.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.homogeneous_plane_signed_distance_to_xyz is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = plane[3]
    return (A * x) + (B * y) + (C * z) + D


# -------------------------------------------------------------------------------------------------------
# 3-D LINE CALCULATIONS
#


def construct_line_3d_given_two_points(
    xyz_1, xyz_2, tolerance=0.0
):  # ?? SCAFFOLDING RCB -- THE 3-D LINE SHOULD BE A CLASS.
    """
    DEPRECATED: Constructs a 3D line from two points.

    This function creates a representation of an infinite line defined by two points in 3D space.

    Parameters
    ----------
    xyz_1 : list[float]
        The coordinates of the first point (x1, y1, z1).
    xyz_2 : list[float]
        The coordinates of the second point (x2, y2, z2).
    tolerance : float, optional
        A tolerance value to determine if the two points are considered the same. Defaults to 0.0.

    Returns
    -------
    dict
        A dictionary containing attributes of the line, including its length, midpoint, and direction.

    Raises
    ------
    AssertionError
        If the two points are the same or too close to each other.

    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    DEPRECATED: Constructs a 3D line from two points.

    This function creates a representation of an infinite line defined by two points in 3D space.

    Parameters
    ----------
    xyz_1 : list[float]
        The coordinates of the first point (x1, y1, z1).
    xyz_2 : list[float]
        The coordinates of the second point (x2, y2, z2).
    tolerance : float, optional
        A tolerance value to determine if the two points are considered the same. Defaults to 0.0.

    Returns
    -------
    dict
        A dictionary containing attributes of the line, including its length, midpoint, and direction.

    Raises
    ------
    AssertionError
        If the two points are the same or too close to each other.

    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.construct_line_3d_given_two_points is deprecated.  Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Fetch individual coordinates, for clarity.
    x1 = xyz_1[0]
    y1 = xyz_1[1]
    z1 = xyz_1[2]
    x2 = xyz_2[0]
    y2 = xyz_2[1]
    z2 = xyz_2[2]
    # Check input.
    if (abs(x1 - x2) <= tolerance) and (abs(y1 - y2) <= tolerance) and (abs(z1 - z2) <= tolerance):
        print("ERROR: In construct_line_3d_given_two_points(), degenerate point pair encountered.")
        assert False
    # Compute attributes.
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    length_xy = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    mid_xyz = [(x2 + x1) / 2, (y2 + y1) / 2, (z2 + z1) / 2]
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1
    vxyz = [vx, vy, vz]
    uxyz = [vx / length, vy / length, vz / length]
    theta = math.atan2(vy, vx)
    eta = math.atan2(vz, length_xy)
    # Setup line_3d object.
    line_3d = {}
    line_3d["xyz_1"] = xyz_1  # First example point.
    line_3d["xyz_2"] = xyz_2  # Second example point.
    line_3d["length"] = length  # Euclidean distance between example points.
    line_3d["length_xy"] = length_xy  # Euclidean distance between example points, projected onto the xy plane.
    line_3d["mid_xyz"] = mid_xyz  # Point midway between the two example points.
    line_3d["vxyz"] = vxyz  # Vector pointing from first example point to second example point.
    line_3d["uxyz"] = uxyz  # Unit vector pointing from first example point to second example point.
    line_3d["theta"] = (
        theta  # Angle the line points, after projecting onto the xy plane, measured ccw about the z axis.
    )
    line_3d["eta"] = eta  # Angle the line points above the xy plane (negative values indicate below the xy plane).
    # Return.
    return line_3d


def closest_point_on_line_3d(xyz, line_3d):  # ?? SCAFFOLDING RCB -- THE 3-D LINE SHOULD BE A CLASS.
    """
    DEPRECATED: Finds the closest point on a 3D line to a given point.

    This function calculates the point on the infinite line that is closest to the specified point in 3D space.

    Parameters
    ----------
    xyz : list[float]
        The coordinates of the point (x, y, z).
    line_3d : dict
        A dictionary representing the line, containing attributes such as 'mid_xyz' and 'uxyz'.

    Returns
    -------
    list[float]
        The coordinates of the closest point on the line to the given point.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.closest_point_on_line_3d is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Fetch coordinates.
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    mid_x = line_3d["mid_xyz"][0]
    mid_y = line_3d["mid_xyz"][1]
    mid_z = line_3d["mid_xyz"][2]
    ux = line_3d["uxyz"][0]
    uy = line_3d["uxyz"][1]
    uz = line_3d["uxyz"][2]
    # Compute vector from mid point to given point.
    mid_to_xyz_x = x - mid_x
    mid_to_xyz_y = y - mid_y
    mid_to_xyz_z = z - mid_z
    # Compute the dot product of the mid_to_xyz vector and the line direction unit vector.
    # Because (v1 dot v2) = |v1|*|v2|*cos(theta), where theta is the angle betwe the vectors,
    # and because uxyz is a unit vector, the dot product is the signed distance along the line,
    # measured from the mid point.
    signed_distance_along_line = (ux * mid_to_xyz_x) + (uy * mid_to_xyz_y) + (uz * mid_to_xyz_z)
    # Construct the closest point.
    closest_x = mid_x + (ux * signed_distance_along_line)
    closest_y = mid_y + (uy * signed_distance_along_line)
    closest_z = mid_z + (uz * signed_distance_along_line)
    closest_xyz = [closest_x, closest_y, closest_z]
    # Return.
    return closest_xyz


def distance_to_line_3d(xyz, line_3d):  # ?? SCAFFOLDING RCB -- THE 3-D LINE SHOULD BE A CLASS.
    """
    DEPRECATED: Calculates the shortest distance from a point to an infinite 3D line.

    Parameters
    ----------
    xyz : list[float]
        The coordinates of the point (x, y, z).
    line_3d : dict
        A dictionary representing the line, containing attributes such as 'mid_xyz' and 'uxyz'.

    Returns
    -------
    float
        The shortest distance from the point to the line.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    DEPRECATED: Calculates the shortest distance from a point to an infinite 3D line.

    Parameters
    ----------
    xyz : list[float]
        The coordinates of the point (x, y, z).
    line_3d : dict
        A dictionary representing the line, containing attributes such as 'mid_xyz' and 'uxyz'.

    Returns
    -------
    float
        The shortest distance from the point to the line.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_3d.distance_to_line_3d is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Fetch coordinates.
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    # Find the closest point on the line.
    closest_xyz = closest_point_on_line_3d(xyz, line_3d)
    closest_x = closest_xyz[0]
    closest_y = closest_xyz[1]
    closest_z = closest_xyz[2]
    # Compute the distance.
    d = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2 + (z - closest_z) ** 2)
    # Return.
    return d
