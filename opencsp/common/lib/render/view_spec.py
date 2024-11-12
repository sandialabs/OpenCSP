"""
Specifying a view of a three-dimensional space.

Options include general 3-d (allowing interactive rotation), xy, xz, yz, general section plane.

In all of the 2-d cases, there is an embedded (p,q) parameter space, which corresponds to 
different projections of the 3-d coordinates.
"""

import numpy as np

import opencsp.common.lib.geometry.geometry_2d as g2d


# COMMON VIEWS


def view_spec_3d() -> dict:
    """Returns a specification dictionary for a 3D view.

    Returns
    -------
    dict
        A dictionary containing the view type set to '3d'.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    spec = {}
    spec['type'] = '3d'
    return spec


def view_spec_xy() -> dict:
    """Returns a specification dictionary for an XY view.

    Returns
    -------
    dict
        A dictionary containing the view type set to 'xy'.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    spec = {}
    spec['type'] = 'xy'
    return spec


def view_spec_xz() -> dict:
    """Returns a specification dictionary for an XZ view.

    Returns
    -------
    dict
        A dictionary containing the view type set to 'xz'.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    spec = {}
    spec['type'] = 'xz'
    return spec


def view_spec_yz() -> dict:
    """Returns a specification dictionary for a YZ view.

    Returns
    -------
    dict
        A dictionary containing the view type set to 'yz'.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    spec = {}
    spec['type'] = 'yz'
    return spec


def view_spec_im() -> dict:
    """Returns a specification dictionary for an image view.

    Returns
    -------
    dict
        A dictionary containing the view type set to 'image'.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    spec = {}
    spec['type'] = 'image'
    return spec


def view_spec_vplane(segment_xy) -> dict:  # A vertical plane containing the segment.
    """Returns a specification dictionary for a vertical plane defined by a segment.

    This function constructs a vertical plane based on the provided segment defined by two points.

    Parameters
    ----------
    segment_xy : list[list[float]]
        A list containing two points that define the segment in 2D space, where each point is represented as [x, y].

    Returns
    -------
    dict
        A dictionary containing the view type set to 'vplane' and additional information about the plane and segment.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Construct embedding line.
    line = g2d.homogeneous_line(segment_xy[0], segment_xy[1])

    # Find origin of (p,q) section.
    # This is the point on the line closest to the (x,y,z) origin.
    # This is also the point where the line intersects the ray from the origin perpendicular to the line.
    # The vector [A B]^T is perpendicular to the line.
    line_A = line[0]
    line_B = line[1]
    line_C = line[2]
    perpendicular_ray = [[0, 0], [line_A, line_B]]
    origin_xy = g2d.intersect_rays(segment_xy, perpendicular_ray)
    origin_xyz = [origin_xy[0], origin_xy[1], 0]

    # Construct the p,q,w axes.
    # The p axis points in the direction of the segment, following its orientation.
    segment_xy0 = segment_xy[0]
    segment_x0 = segment_xy0[0]
    segment_y0 = segment_xy0[1]
    segment_xy1 = segment_xy[1]
    segment_x1 = segment_xy1[0]
    segment_y1 = segment_xy1[1]
    segment_dx = segment_x1 - segment_x0
    segment_dy = segment_y1 - segment_y0
    segment_d = np.sqrt((segment_dx * segment_dx) + (segment_dy * segment_dy))
    p_uxyz = [segment_dx / segment_d, segment_dy / segment_d, 0]
    # The q axis points straight up.
    q_uxyz = [0, 0, 1]
    # The w axis points perpendicular to the p and q axes, following the right-hand rule.
    w_uxyz = np.cross(p_uxyz, q_uxyz)

    # Construct the section 3-d plane.
    plane_A = line_A  # x component of surface normal.
    plane_B = line_B  # y component of surface normal.
    plane_C = 0  # z component of surface normal.
    plane_D = line_C  # Distance to origin.
    plane = [plane_A, plane_B, plane_C, plane_D]

    # Store results.
    spec = {}
    spec['type'] = 'vplane'
    spec['defining_segment_xy'] = segment_xy
    spec['line_intersecting_xy_plane'] = line
    spec['section_plane'] = plane
    spec['origin_xyz'] = origin_xyz
    spec['p_uxyz'] = p_uxyz
    spec['q_uxyz'] = q_uxyz
    spec['w_uxyz'] = w_uxyz
    # Return.
    return spec


def view_spec_camera(camera, camera_xform) -> dict:
    """Returns a specification dictionary for a camera view.

    Parameters
    ----------
    camera : object
        The camera object representing the camera's properties.
    camera_xform : object
        The transformation object for the camera.

    Returns
    -------
    dict
        A dictionary containing the view type set to 'camera' and the camera properties.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    spec = {}
    spec['type'] = 'camera'
    spec['camera'] = camera
    spec['camera_xform'] = camera_xform
    return spec


# XYZ <---> PQ CONVERSION


def xyz2pqw(xyz, view_spec):
    """Converts 3D coordinates to (p, q, w) coordinates based on the view specification.

    Parameters
    ----------
    xyz : np.ndarray
        The 3D coordinates to convert.
    view_spec : dict
        The view specification that determines the conversion method.

    Returns
    -------
    list[float] | None
        The converted (p, q, w) coordinates, or None if the point is behind the camera in a camera view.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Enforces the right-hand rule in all conversions.
    # That is, [p,q,w] is a right-handed coordinate system.
    if view_spec['type'] == '3d':
        return xyz
    elif view_spec['type'] == 'xy':
        return [xyz[0], xyz[1], xyz[2]]
    elif view_spec['type'] == 'xz':
        return [xyz[0], xyz[2], -xyz[1]]
    elif view_spec['type'] == 'yz':
        return [xyz[1], xyz[2], xyz[0]]
    elif view_spec['type'] == 'vplane':
        # Fetch section coordinate system.
        origin_xyz = np.array(view_spec['origin_xyz'])  # Make arrays so we can do simple vactor math.
        p_uxyz = np.array(view_spec['p_uxyz'])  #
        q_uxyz = np.array(view_spec['q_uxyz'])  #
        w_uxyz = np.array(view_spec['w_uxyz'])  #
        # Construct vector from origin to xyz.
        vxyz = np.array(xyz) - origin_xyz
        # Construct (p,q,w) components.
        p = vxyz.dot(p_uxyz)
        q = vxyz.dot(q_uxyz)
        w = vxyz.dot(w_uxyz)
        return [p, q, w]
    elif view_spec['type'] == 'camera':
        camera = view_spec['camera']
        camera_xform = view_spec['camera_xform']
        pq = camera_xform.pq_or_none(camera, xyz)
        if pq == None:
            # (x,y,z) point is behind camera and thus should not be drawn.
            return None
        else:
            p = pq[0]
            q = pq[1]
            w = 0
            # Return.
            return [p, q, w]
    else:
        print("ERROR: In xyz2pqw(), unrecognized view_spec['type'] = '" + str(view_spec['type']) + "' encountered.")
        assert False


def xyz2pq(xyz, view_spec):
    """Converts 3D coordinates to (p, q) coordinates based on the view specification.

    Parameters
    ----------
    xyz : np.ndarray
        The 3D coordinates to convert.
    view_spec : dict
        The view specification that determines the conversion method.

    Returns
    -------
    list[float] | None
        The converted (p, q) coordinates, or None if the point is behind the camera in a camera view.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    pqw = xyz2pqw(xyz, view_spec)
    if pqw == None:
        return None
    else:
        return pqw[0:2]


def pqw2xyz(pqw, view_spec):
    """Converts (p, q, w) coordinates back to 3D coordinates based on the view specification.

    Parameters
    ----------
    pqw : list[float]
        The (p, q, w) coordinates to convert.
    view_spec : dict
        The view specification that determines the conversion method.

    Returns
    -------
    list[float]
        The converted 3D coordinates.

    Raises
    ------
    ValueError
        If the view specification type is unrecognized.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Assumes the right-hand rule for all conversions.
    # That is, [p,q,w] is viewed as a right-handed coordinate system.
    if view_spec['type'] == '3d':
        return pqw
    elif view_spec['type'] == 'xy':
        return [pqw[0], pqw[1], pqw[2]]
    elif view_spec['type'] == 'xz':
        return [pqw[0], -pqw[2], pqw[1]]
    elif view_spec['type'] == 'yz':
        return [pqw[2], pqw[0], pqw[1]]
    elif view_spec['type'] == 'vplane':
        # Fetch section coordinate system.
        origin_xyz = np.array(view_spec['origin_xyz'])  # Make arrays so we can do simple vactor math.
        p_uxyz = np.array(view_spec['p_uxyz'])  #
        q_uxyz = np.array(view_spec['q_uxyz'])  #
        w_uxyz = np.array(view_spec['w_uxyz'])  #
        # Extract (p,q,w) components.
        p = pqw[0]
        q = pqw[1]
        w = pqw[2]
        # Construct (x,y,z) components.
        xyz = origin_xyz + (p * p_uxyz) + (q * q_uxyz) + (w * w_uxyz)
        return [xyz[0], xyz[1], xyz[2]]
    else:
        print("ERROR: In pqw2xyz(), unrecognized view_spec['type'] = '" + str(view_spec['type']) + "' encountered.")
        assert False


def pq2xyz(pq, view_spec):
    """Converts (p, q) coordinates to 3D coordinates based on the view specification.

    Parameters
    ----------
    pq : list[float]
        The (p, q) coordinates to convert.
    view_spec : dict
        The view specification that determines the conversion method.

    Returns
    -------
    list[float]
        The converted 3D coordinates.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    pqw = pq.copy()
    pqw.append(0)
    return pqw2xyz(pqw, view_spec)
