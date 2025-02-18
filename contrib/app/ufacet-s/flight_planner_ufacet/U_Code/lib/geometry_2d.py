"""
2-d Geometry Utiltiies
"""

import math
import numpy as np
from warnings import warn

import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


# -------------------------------------------------------------------------------------------------------
# HOMOGEONEOUS LINE CALCULATIONS
#


def homogeneous_line(xy1, xy2):
    """
    DEPRECATED: Calculates the homogeneous line coefficients from two points.

    This function returns the coefficients of the line in homogeneous coordinates,
    normalized to ensure the coefficients are in a standard form.

    Parameters
    ----------
    xy1 : np.ndarray
        The coordinates of the first point (x1, y1).
    xy2 : np.ndarray
        The coordinates of the second point (x2, y2).

    Returns
    -------
    list[float]
        A list containing the normalized coefficients [A, B, C] of the line equation Ax + By + C = 0.

    Raises
    ------
    AssertionError
        If the two points are the same, resulting in a degenerate case.
    DeprecationWarning
        geometry_2d.homogeneous_line is deprecated. Use LineXY instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn("geometry_2d.homogeneous_line is deprecated. Use LineXY instead.", DeprecationWarning, stacklevel=2)
    # Returns homogeneous line coeffcients, in normalized form.
    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]
    A = y2 - y1
    B = x1 - x2
    C = (x2 * y1) - (x1 * y2)
    n = np.sqrt((A * A) + (B * B))
    if n == 0:
        print("\nERROR: In homogeneous_line, degenerate case encountered.", DeprecationWarning, stacklevel=2)
        print("   xy1 =", xy1)
        print("   xy2 =", xy2)
        print("\n")
        assert False
    A = A / n
    B = B / n
    C = C / n
    return [A, B, C]


def flip_homogeneous_line(line):
    """
    DEPRECATED: Reverses the sense of the homogeneous line.

    Parameters
    ----------
    line : list[float]
        The coefficients of the line in homogeneous coordinates [A, B, C].

    Returns
    -------
    list[float]
        The coefficients of the flipped line.

    Raises
    ------
    DeprecationWarning
        geometry_2d.flip_homogeneous_line is deprecated. Use LineXY.flip() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.flip_homogeneous_line is deprecated. Use LineXY.flip() instead.", DeprecationWarning, stacklevel=2
    )
    # Reverse the sense of the homogeneous line.
    return [-x for x in line]


def homogeneous_line_signed_distance_to_xy(xy, line):
    """
    DEPRECATED: Calculates the signed distance from a point to a homogeneous line.

    Parameters
    ----------
    xy : np.ndarray
        The coordinates of the point (x, y).
    line : list[float]
        The coefficients of the line in homogeneous coordinates [A, B, C].

    Returns
    -------
    float
        The signed distance from the point to the line.

    Raises
    ------
    DeprecationWarning
        geometry_2d.homogeneous_line_signed_distance_to_xy is deprecated. Use LineXY.dist_from_line_signed() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.homogeneous_line_signed_distance_to_xy is deprecated. Use LineXY.dist_from_line_signed() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    x = xy[0]
    y = xy[1]
    A = line[0]
    B = line[1]
    C = line[2]
    return (A * x) + (B * y) + C


def homogeneous_line_y_given_x(x, line):
    """
    DEPRECATED: Calculates the y-coordinate on a homogeneous line given an x-coordinate.

    Parameters
    ----------
    x : float
        The x-coordinate for which to find the corresponding y-coordinate on the line.
    line : list[float]
        The coefficients of the line in homogeneous coordinates [A, B, C].

    Returns
    -------
    float
        The y-coordinate corresponding to the given x-coordinate on the line.
        Returns NaN if the line is vertical (B = 0).

    Raises
    ------
    DeprecationWarning
        geometry_2d.homogeneous_line_y_given_x is deprecated. Use LineXY.y_from_x() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.homogeneous_line_y_given_x is deprecated. Use LineXY.y_from_x() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    A = line[0]
    B = line[1]
    C = line[2]
    if B == 0:
        return np.nan
    else:
        return (-A * x - C) / B


def homogeneous_line_x_given_y(y, line):
    """
    DEPRECATED: Calculates the x-coordinate on a homogeneous line given a y-coordinate.

    Parameters
    ----------
    y : float
        The y-coordinate for which to find the corresponding x-coordinate on the line.
    line : list[float]
        The coefficients of the line in homogeneous coordinates [A, B, C].

    Returns
    -------
    float
        The x-coordinate corresponding to the given y-coordinate on the line.
        Returns NaN if the line is horizontal (A = 0).

    Raises
    ------
    DeprecationWarning
        geometry_2d.homogeneous_line_x_given_y is deprecated. Use LineXY.x_from_y() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.homogeneous_line_x_given_y is deprecated. Use LineXY.x_from_y() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    A = line[0]
    B = line[1]
    C = line[2]
    if A == 0:
        return np.nan
    else:
        return (-B * y - C) / A


def intersect_lines(line1, line2):
    """
    DEPRECATED: Calculates the intersection point of two homogeneous lines.

    Parameters
    ----------
    line1 : list[float]
        The coefficients of the first line in homogeneous coordinates [A1, B1, C1].
    line2 : list[float]
        The coefficients of the second line in homogeneous coordinates [A2, B2, C2].

    Returns
    -------
    list[float]
        The intersection point (x, y) of the two lines. Returns [NaN, NaN] if the lines are parallel.

    Raises
    ------
    ValueError
        If the lines are parallel and do not intersect.
    DeprecationWarning
        geometry_2d.intersect_lines is deprecated. Use LineXY.intersect_with() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.intersect_lines is deprecated. Use LineXY.intersect_with() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Line 1 coefficients
    A1 = line1[0]
    B1 = line1[1]
    C1 = line1[2]
    # Line 2 coefficients
    A2 = line2[0]
    B2 = line2[1]
    C2 = line2[2]
    # Solve.
    denom = (A1 * B2) - (A2 * B1)
    if denom == 0:
        # Lines are parallel.
        return [np.nan, np.nan]
    elif abs(A1) > abs(B1):
        y = ((A2 * C1) - (A1 * C2)) / denom
        x = homogeneous_line_x_given_y(y, line1)
    else:
        x = ((B1 * C2) - (B2 * C1)) / denom
        y = homogeneous_line_y_given_x(x, line1)
    # Return.
    return [x, y]


def shift_x(ray, dx):
    """
    DEPRECATED: Shifts a ray in the x-direction by a specified amount.

    Parameters
    ----------
    ray : list[list[float]]
        A list containing two points that define the ray, each represented as [x, y].
    dx : float
        The amount to shift the ray in the x-direction.

    Returns
    -------
    list[list[float]]
        A new ray represented by two points shifted in the x-direction.

    Raises
    ------
    DeprecationWarning
        geometry_2d.shift_x is deprecated. Use Vxy.__add__() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn("geometry_2d.shift_x is deprecated. Use Vxy.__add__() instead.", DeprecationWarning, stacklevel=2)
    x0 = ray[0][0]
    y0 = ray[0][1]
    x1 = ray[1][0]
    y1 = ray[1][1]
    return [[x0 + dx, y0], [x1 + dx, y1]]


def intersect_rays(ray1, ray2):
    """
    DEPRECATED: Calculates the intersection point of two rays.

    Parameters
    ----------
    ray1 : list[list[float]]
        A list containing two points that define the first ray, each represented as [x, y].
    ray2 : list[list[float]]
        A list containing two points that define the second ray, each represented as [x, y].

    Returns
    -------
    list[float]
        The intersection point (x, y) of the two rays. Returns [NaN, NaN] if the rays do not intersect.

    Raises
    ------
    DeprecationWarning
        geometry_2d.intersect_rays is deprecated. Use LineXY.intersect_with() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.intersect_rays is deprecated. Use LineXY.intersect_with() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    line1 = homogeneous_line(ray1[0], ray1[1])
    line2 = homogeneous_line(ray2[0], ray2[1])
    return intersect_lines(line1, line2)


# -------------------------------------------------------------------------------------------------------
# CLIPPING IN 2-D
#


def draw_clip_xy_box(view, clip_xy_box):
    """
    DEPRECATED: Draws a clipping box in the XY plane.

    Parameters
    ----------
    view : object
        The view object in which to draw the clipping box.
    clip_xy_box : tuple[tuple[float, float], tuple[float, float]]
        A tuple defining the clipping box as ((xmin, xmax), (ymin, ymax)).

    Raises
    ------
    DeprecationWarning
        geometry_2d.draw_clip_xy_box is deprecated. Should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.draw_clip_xy_box is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    xy_min = clip_xy_box[0]
    xy_max = clip_xy_box[1]
    p_min = xy_min[0]
    y_min = xy_min[1]
    p_max = xy_max[0]
    y_max = xy_max[1]
    xy_list = [[p_min, y_min], [p_max, y_min], [p_max, y_max], [p_min, y_max]]
    view.draw_pq_list(xy_list, close=True, style=rcps.outline(color="r"), label="Clip Box")


def clip_line_to_xy_box(line, clip_xy_box):
    """
    DEPRECATED: Clips a line to the specified XY clipping box.

    Parameters
    ----------
    line : list[float]
        The coefficients of the line in homogeneous coordinates [A, B, C].
    clip_xy_box : tuple[tuple[float, float], tuple[float, float]]
        A tuple defining the clipping box as ((xmin, xmax), (ymin, ymax)).

    Returns
    -------
    list[list[float]]
        A list of points where the line intersects the clipping box edges.
        Returns an empty list if the line is completely outside the clipping box.

    Raises
    ------
        geometry_2d.clip_line_to_xy_box is deprecated. Should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.clip_line_to_xy_box is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    xy_min = clip_xy_box[0]
    xy_max = clip_xy_box[1]
    x_min = xy_min[0]
    y_min = xy_min[1]
    x_max = xy_max[0]
    y_max = xy_max[1]
    # Find intersection points.
    y_given_x_min = homogeneous_line_y_given_x(x_min, line)
    y_given_x_max = homogeneous_line_y_given_x(x_max, line)
    x_given_y_min = homogeneous_line_x_given_y(y_min, line)
    x_given_y_max = homogeneous_line_x_given_y(y_max, line)
    # Find points on box edge.
    clip_points = []
    # Check y bounds, including equality.
    if not np.isnan(y_given_x_min):
        if (y_min <= y_given_x_min) and (y_given_x_min <= y_max):
            clip_points.append([x_min, y_given_x_min])
    if not np.isnan(y_given_x_max):
        if (y_min <= y_given_x_max) and (y_given_x_max <= y_max):
            clip_points.append([x_max, y_given_x_max])
    # Check p bounds, excluding eyuailty to avoid duplicates.
    if not np.isnan(x_given_y_min):
        if (x_min < x_given_y_min) and (x_given_y_min < x_max):
            clip_points.append([x_given_y_min, y_min])
    if not np.isnan(x_given_y_max):
        if (x_min < x_given_y_max) and (x_given_y_max < x_max):
            clip_points.append([x_given_y_max, y_max])
    # Check result.
    if len(clip_points) != 2:
        print(
            "WARNING: In clip_line_to_xy_box(), unexpected result with ", len(clip_points), " clip points encountered."
        )
        print("          line        = ", line)
        print("          clip_xy_box = ", clip_xy_box)
    # Return.
    return clip_points


def extend_ray(ray, clip_xy_box, fail_if_null_result=True):
    """
    DEPRECATED: Extends a ray to intersect with the specified clipping box.

    Parameters
    ----------
    ray : list[list[float]]
        A list containing two points that define the ray, each represented as [x, y].
    clip_xy_box : tuple[tuple[float, float], tuple[float, float]]
        A tuple defining the clipping box as ((xmin, xmax), (ymin, ymax)).
    fail_if_null_result : bool, optional
        If True, raises an error if the ray does not intersect the clipping box. Defaults to True.

    Returns
    -------
    list[list[float]]
        A new ray represented by two points that extend to the clipping box.
        Returns None if the ray is completely outside the clipping box and fail_if_null_result is False.

    Raises
    ------
    AssertionError
        If the ray is completely outside the clipping box and fail_if_null_result is True.
    DeprecationWarning
        geometry_2d.extend_ray is deprecated. Should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.extend_ray is deprecated. Should be migrated to another library.", DeprecationWarning, stacklevel=2
    )
    xy0 = ray[0]
    xy1 = ray[1]

    line = homogeneous_line(xy0, xy1)

    clip_points = clip_line_to_xy_box(line, clip_xy_box)
    if len(clip_points) == 0:
        print("ERROR: In extend_ray(), ray is outside clip box.")
        print("          ray = ", ray)
        print("          clip_xy_box = ", clip_xy_box)
    clip_xy0 = clip_points[0]
    clip_xy1 = clip_points[1]

    # Determine which clip points are along the ray direction from the tail.
    v01 = np.array(xy1) - np.array(xy0)
    v0c0 = np.array(clip_xy0) - np.array(xy0)
    v0c1 = np.array(clip_xy1) - np.array(xy0)
    c0dot = v01.dot(v0c0)
    c1dot = v01.dot(v0c1)
    if (c0dot <= 0) and (c1dot <= 0):
        # The whole ray is outside the clip box, and it points away from the box.
        if fail_if_null_result:
            print("ERROR: In extend_ray(), Unexpected situation encountered.")
            print("          ray = ", ray)
            print("          clip_xy_box = ", clip_xy_box)
            assert False
        else:
            print("WARNING: In extend_ray(), Unexpected situation encountered.")
            print("            ray = ", ray)
            print("            clip_xy_box = ", clip_xy_box)
            print("         Proceeding....")
            return None
    elif c0dot <= 0:
        return [xy0, clip_xy1]
    elif c1dot <= 0:
        return [xy0, clip_xy0]
    else:
        # The whole ray is outside the clip box, and it points toward the box.
        if c0dot == c1dot:
            # Degenerate case.
            if fail_if_null_result:
                print("ERROR: In extend_ray(), degenerate case encountered.")
                print("          ray = ", ray)
                print("          clip_xy_box = ", clip_xy_box)
                assert False
            else:
                print("WARNING: In extend_ray(), degenerate case encountered.")
                print("            ray = ", ray)
                print("            clip_xy_box = ", clip_xy_box)
                print("         Proceeding....")
                return None
        elif c0dot > c1dot:
            return [clip_xy1, clip_xy0]
        else:
            return [clip_xy0, clip_xy1]


# -------------------------------------------------------------------------------------------------------
# BEST-FIT LINE TO 2-D POINTS
#


# Derived from:
#    https://scipython.com/book/chapter-6-numpy/examples/finding-a-best-fit-straight-line/
# The above web site assumes the data describe y as a function of x.
# This routine allows x as a function of y.
# Decision is made based on which variable has the largest domain;
# this allows code to work if data are a vertical or horizontal line.
#
def best_fit_line_segment_A(xy_seq):
    """
    DEPRECATED: Calculates the best-fit line segment for a sequence of 2D points.

    This function determines the best-fit line segment that minimizes the distance
    to a set of points in 2D space.

    Parameters
    ----------
    xy_seq : list[list[float]]
        A list of points where each point is represented as [x, y].

    Returns
    -------
    list[list[float]]
        A list containing two points that define the best-fit line segment.

    Raises
    ------
    AssertionError
        If the input points are ill-conditioned (all points are the same).
    DeprecationWarning
        geometry_2d.best_fit_line_segment_A is deprecated. Use LineXY.fit_from_points() instead.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.best_fit_line_segment_A is deprecated. Use LineXY.fit_from_points() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Fetch x and y sequences.
    x_seq = [xy[0] for xy in xy_seq]
    y_seq = [xy[1] for xy in xy_seq]
    # Fetch bounds.
    x_min = min(x_seq)
    x_max = max(x_seq)
    y_min = min(y_seq)
    y_max = max(y_seq)
    # Compute range.
    dx = x_max - x_min
    dy = y_max - y_min

    # Compute best fit.
    degree = 1
    # Select x vs. y sense for better numerical conditioning.
    if (dx == 0) and (dy == 0):
        print("ERROR: In best_fit_line(), ill-conditioned input:")
        print("             xy_seq:", xy_seq)
        print("       Ill-conditioned because dx = dy = 0")
        print(" ")
        assert False
    if dx > dy:
        pfit, stats = np.polynomial.Polynomial.fit(
            x_seq, y_seq, degree, full=True, window=(x_min, x_max), domain=(x_min, x_max)  # linear
        )
        b, m = pfit
        x0 = x_min
        y0 = m * x0 + b
        x1 = x_max
        y1 = m * x1 + b
        # Compute residual.
        resid, rank, sing_val, rcond = stats
        rms = np.sqrt(resid[0] / len(x_seq))
        # print('Fit: y = {:.3f}x + {:.3f}'.format(m, b), ' (rms residual = {:.4f})'.format(rms))
    else:
        pfit, stats = np.polynomial.Polynomial.fit(
            y_seq,  # Transpose x and y
            x_seq,  #
            degree,  # linear
            full=True,
            window=(y_min, y_max),
            domain=(y_min, y_max),
        )
        b, m = pfit
        y0 = y_min  # Recall x and y are transposed.
        x0 = m * y0 + b  #
        y1 = y_max  #
        x1 = m * y1 + b  #
        # Compute residual.
        resid, rank, sing_val, rcond = stats
        rms = np.sqrt(resid[0] / len(y_seq))
        # print('Fit: x = {:.3f}y + {:.3f}'.format(m, b), ' (rms residual = {:.4f})'.format(rms))

    # Assemble line segment to return.
    xy0 = [x0, y0]
    xy1 = [x1, y1]
    return [xy0, xy1]


def best_fit_line_segment(xy_list):
    """
    DEPRECATED: Calculates the best-fit line segment for a sequence of 2D points.

    This function determines the best-fit line segment that minimizes the distance
    to a set of points in 2D space. This is a deprecated function; use LineXY.fit_from_points() instead.

    Parameters
    ----------
    xy_list : list[list[float]]
        A list of points where each point is represented as [x, y].

    Returns
    -------
    list[list[float]]
        A list containing two points that define the best-fit line segment.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be replaced with LineXY.fit_from_points().
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.best_fit_line_segment is deprecated. Use LineXY.fit_from_points() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return best_fit_line_segment_A(xy_list)


# -------------------------------------------------------------------------------------------------------
# TRANSFORMATIONS
#


def rotate_about_origin(xy, theta):
    """
    DEPRECATED: Rotates a point around the origin by a specified angle.

    This function rotates the point (x, y) by the angle theta (in radians) around the origin (0, 0).
    This is a deprecated function; use Vxy.rotate() or TransformXY instead.

    Parameters
    ----------
    xy : list[float]
        The coordinates of the point to rotate, represented as [x, y].
    theta : float
        The angle of rotation in radians.

    Returns
    -------
    list[float]
        The new coordinates of the point after rotation, represented as [x', y'].

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be replaced with Vxy.rotate() or TransformXY.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.rotate_about_origin is deprecated. Use Vxy.rotate() or TransformXY instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    x = xy[0]
    y = xy[1]
    x2 = x * c_theta - y * s_theta
    y2 = x * s_theta + y * c_theta
    return [x2, y2]


def rotate_about_center(xy, theta, center_xy):
    """
    DEPRECATED: Rotates a point around a specified center by a specified angle.

    This function rotates the point (x, y) around the point (cx, cy) by the angle theta (in radians).
    This is a deprecated function; use TransformXY instead.

    Parameters
    ----------
    xy : list[float]
        The coordinates of the point to rotate, represented as [x, y].
    theta : float
        The angle of rotation in radians.
    center_xy : list[float]
        The coordinates of the center point around which to rotate, represented as [cx, cy].

    Returns
    -------
    list[float]
        The new coordinates of the point after rotation, represented as [x', y'].

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be replaced with TransformXY.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn("geometry_2d.rotate_about_center is deprecated. Use TransformXY instead.", DeprecationWarning, stacklevel=2)
    x = xy[0]
    y = xy[1]
    cx = center_xy[0]
    cy = center_xy[1]
    x2 = x - cx
    y2 = y - cy
    xy3 = rotate_about_origin([x2, y2], theta)
    x3 = xy3[0]
    y3 = xy3[1]
    x4 = x3 + cx
    y4 = y3 + cy
    return [x4, y4]


def rotate_xyz_about_center_xy(xyz, theta, center_xy):
    """
    DEPRECATED: Rotates a 3D point around a specified center in the XY plane.

    This function performs a planar rotation of the point (x, y, z) around the center (cx, cy)
    by the angle theta (in radians). The z-coordinate remains unchanged.

    Parameters
    ----------
    xyz : list[float]
        The coordinates of the point to rotate, represented as [x, y, z].
    theta : float
        The angle of rotation in radians.
    center_xy : list[float]
        The coordinates of the center point around which to rotate, represented as [cx, cy].

    Returns
    -------
    list[float]
        The new coordinates of the point after rotation, represented as [x', y', z].

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be replaced with TransformXYZ, TransformXY, Vxyz.rotate(), or Vxy.rotate().
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.rotate_xyz_about_center_xy is deprecated. Use TransformXYZ, TransformXY, Vxyz.rotate() oro Vxy.rotate() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Fetch coordinates.
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    # Rotate the (x,y) component.
    xy = [x, y]
    xy2 = rotate_about_center(xy, theta, center_xy)
    # Convert to 3-d point.
    return [xy2[0], xy2[1], z]


# -------------------------------------------------------------------------------------------------------
# POLYGONS
#
# A polygon is a list of [x,y] vertices, with line segments connecting subsequent vertices.
# All polygons are closed, meaning thqt thet is a line segment connecting the last and first vertices.
# The first vertex does not need to be repeated.
#


def label_point(xy_list):
    """
    DEPRECATED: Calculates a central label point for a list of 2D points.

    This function computes a reasonable central label point for a given list of points.
    It calculates the mean of the x and y coordinates of the points.

    Parameters
    ----------
    xy_list : list[list[float]]
        A list of points where each point is represented as [x, y].

    Returns
    -------
    list[float]
        The coordinates of the central label point, represented as [x_mean, y_mean].

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    warn(
        "geometry_2d.label_point is deprecated. Should be migrated to another library.",
        DeprecationWarning,
        stacklevel=2,
    )
    """
    A reasonable place to put a central label.
    This is not the centroid of a polygon, but a quick-and-dirty first choice.
    This also works for xy lists that are not polygons -- even singleton points.
    """
    x_array = np.array([xy[0] for xy in xy_list])
    y_array = np.array([xy[1] for xy in xy_list])
    x_mean = x_array.mean()
    y_mean = y_array.mean()
    return [x_mean, y_mean]
