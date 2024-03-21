# -*- coding: utf-8 -*-
"""
Sun Position Calculation and Tracking



"""

import math
import numpy as np

import opencsp.common.lib.csp.sun_position as sp
from opencsp.common.lib.tool.typing_tools import strict_types
from opencsp.common.lib.geometry.Vxyz import Vxyz

# def tracking_surface_normal_xyz(heliostat_xyz,    # (x,y,z) in m.     Heliostat origin.
#                                 aimpoint_xyz,     # (x,y,z) in m.     Reflection aim point.
#                                 location_lon_lat, # (lon,lat) in rad. Solar field origin.
#                                 when_ymdhmsz):    # (year, month, day, hour, minute, second, timezone) tuple.
#                                                   #  Example: (2022, 7, 4, 11, 20, 0, -6)
#                                                   #              => July 4, 2022 at 11:20 am MDT (-6 hours)
#     """
#     Computes heliostat surface normal which tracks the sun to the aimpoint.
#     """
#     # Sun position.
#     sun_xyz = sp.sun_position(location_lon_lat, when_ymdhmsz)

#     # Construct points as arrays.
#     sun = np.array(sun_xyz)
#     h   = np.array(heliostat_xyz)
#     aim = np.array(aimpoint_xyz)

#     # Reflected ray to aimpoint.
#     ha = aim - h
#     ha = ha / np.linalg.norm(ha)

#     # Surface normal
#     h_sn = sun + ha
#     h_sn = h_sn / np.linalg.norm(h_sn)
#     n_x  = h_sn[0]
#     n_y  = h_sn[1]
#     n_z  = h_sn[2]

# # aim_u = aim / np.linalg.norm(aim)
# # print('\nIn tracking_surface_normal_xyz()...')
# # print('In tracking_surface_normal_xyz(), aimpoint_xyz = ', aimpoint_xyz)
# # print('In tracking_surface_normal_xyz(), sun_xyz = ', sun_xyz)
# # print('In tracking_surface_normal_xyz(), aim = ', aim)
# # print('In tracking_surface_normal_xyz(), sun = ', sun)
# # print('In tracking_surface_normal_xyz(), [n_x, n_y, n_z] = ', np.linalg.norm([n_x, n_y, n_z]))
# # print('In tracking_surface_normal_xyz(), norm aim = ', np.linalg.norm(aim))
# # print('In tracking_surface_normal_xyz(), norm aim_u = ', np.linalg.norm(aim_u))
# # print('In tracking_surface_normal_xyz(), norm sun = ', np.linalg.norm(sun))
# # print('In tracking_surface_normal_xyz(), norm [n_x, n_y, n_z] = ', np.linalg.norm([n_x, n_y, n_z]))
# # print('In tracking_surface_normal_xyz(), sun dot n = ', np.array(sun_xyz).dot(np.array([n_x, n_y, n_z])))
# # print('In tracking_surface_normal_xyz(), aim_u dot n = ', np.array(aim_u).dot(np.array([n_x, n_y, n_z])))
# # print()

#     # Return.
#     return [n_x, n_y, n_z]


# def tracking_surface_normal_xy(heliostat_xyz,    # (x,y,z) in m.     Heliostat origin.
#                                aimpoint_xyz,     # (x,y,z) in m.     Reflection aim point.
#                                location_lon_lat, # (lon,lat) in rad. Solar field origin.
#                                when_ymdhmsz):    # (year, month, day, hour, minute, second, timezone) tuple.
#                                                  #  Example: (2022, 7, 4, 11, 20, 0, -6)
#                                                  #              => July 4, 2022 at 11:20 am MDT (-6 hours)
#     """
#     Computes heliostat surface normal which tracks the sun to the aimpoint.
#     Returns only (x,y) components of the surface normal.
#     """
#     normal_xyz = tracking_surface_normal_xyz(heliostat_xyz, aimpoint_xyz, location_lon_lat, when_ymdhmsz)
#     return [normal_xyz[0], normal_xyz[1]]


# def tracking_nu(heliostat_xyz,    # (x,y,z) in m.     Heliostat origin.
#                 aimpoint_xyz,     # (x,y,z) in m.     Reflection aim point.
#                 location_lon_lat, # (lon,lat) in rad. Solar field origin.
#                 when_ymdhmsz):    # (year, month, day, hour, minute, second, timezone) tuple.
#                                   #  Example: (2022, 7, 4, 11, 20, 0, -6)
#                                   #              => July 4, 2022 at 11:20 am MDT (-6 hours)
#     """
#     Computes nu angle of the heliostat surface normal which tracks the sun to the aimpoint.

#     nu is the angle to the projection of the surface normal onto the (x,y) plane, measured ccw from the x axis.
#     """
#     # Compute heliostat surface normal which tracks the sun to the aimpoint.
#     n_xy = tracking_surface_normal_xy(heliostat_xyz, aimpoint_xyz, location_lon_lat, when_ymdhmsz)

#     # Extract surface normal coordinates.
#     n_x = n_xy[0]
#     n_y = n_xy[1]

#     # Compute nu.
#     nu = math.atan2(n_y, n_x)

#     # Return.
#     return nu


@strict_types
def tracking_surface_normal_xyz(
    heliostat_xyz: list | np.ndarray | tuple,  # (x,y,z) in m.     Heliostat origin.
    aimpoint_xyz: list | np.ndarray | tuple,  # (x,y,z) in m.     Reflection aim point.
    location_lon_lat: list | np.ndarray | tuple,  # (lon,lat) in rad. Solar field origin.
    when_ymdhmsz: list | np.ndarray | tuple,
):  # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6)
    #              => July 4, 2022 at 11:20 am MDT (-6 hours)
    """
    Computes heliostat surface normal which tracks the sun to the aimpoint.
    """
    # Sun position.
    sun_xyz = sp.sun_position(location_lon_lat, when_ymdhmsz)

    # Construct points as arrays.
    sun = np.array(sun_xyz)
    h = np.array(heliostat_xyz)
    aim = np.array(aimpoint_xyz)

    # Reflected ray to aimpoint.
    ha = aim - h
    ha = ha / np.linalg.norm(ha)

    # Surface normal
    h_sn = sun + ha
    h_sn = h_sn / np.linalg.norm(h_sn)
    n_x = h_sn[0]
    n_y = h_sn[1]
    n_z = h_sn[2]

    # aim_u = aim / np.linalg.norm(aim)
    # print('\nIn tracking_surface_normal_xyz()...')
    # print('In tracking_surface_normal_xyz(), aimpoint_xyz = ', aimpoint_xyz)
    # print('In tracking_surface_normal_xyz(), sun_xyz = ', sun_xyz)
    # print('In tracking_surface_normal_xyz(), aim = ', aim)
    # print('In tracking_surface_normal_xyz(), sun = ', sun)
    # print('In tracking_surface_normal_xyz(), [n_x, n_y, n_z] = ', np.linalg.norm([n_x, n_y, n_z]))
    # print('In tracking_surface_normal_xyz(), norm aim = ', np.linalg.norm(aim))
    # print('In tracking_surface_normal_xyz(), norm aim_u = ', np.linalg.norm(aim_u))
    # print('In tracking_surface_normal_xyz(), norm sun = ', np.linalg.norm(sun))
    # print('In tracking_surface_normal_xyz(), norm [n_x, n_y, n_z] = ', np.linalg.norm([n_x, n_y, n_z]))
    # print('In tracking_surface_normal_xyz(), sun dot n = ', np.array(sun_xyz).dot(np.array([n_x, n_y, n_z])))
    # print('In tracking_surface_normal_xyz(), aim_u dot n = ', np.array(aim_u).dot(np.array([n_x, n_y, n_z])))
    # print()

    # Return.
    return [n_x, n_y, n_z]


@strict_types
def tracking_surface_normal_xyz_given_sun_vector(
    heliostat_xyz: list | np.ndarray | tuple,  # (x,y,z) in m.     Heliostat origin.
    aimpoint_xyz: list | np.ndarray | tuple,  # (x,y,z) in m.     Reflection aim point.
    sun_vector: Vxyz,
):  # Current direction of the sun
    #  Example: (2022, 7, 4, 11, 20, 0, -6)
    #              => July 4, 2022 at 11:20 am MDT (-6 hours)
    """
    !!!! DOES NOT WORK !!!!
    Computes heliostat surface normal which tracks the sun to the aimpoint.
    """
    # Sun position.
    sun_xyz = sun_vector.normalize().data.T.flatten()

    # Construct points as arrays.
    sun = np.array(sun_xyz)
    h = np.array(heliostat_xyz)
    aim = np.array(aimpoint_xyz)

    # Reflected ray to aimpoint.
    ha = aim - h
    ha = ha / np.linalg.norm(ha)

    # Surface normal
    h_sn = sun + ha
    h_sn = h_sn / np.linalg.norm(h_sn)
    n_x = h_sn[0]
    n_y = h_sn[1]
    n_z = h_sn[2]

    # Return.
    return [n_x, n_y, n_z]


def tracking_surface_normal_xy(
    heliostat_xyz,  # (x,y,z) in m.     Heliostat origin.
    aimpoint_xyz,  # (x,y,z) in m.     Reflection aim point.
    location_lon_lat,  # (lon,lat) in rad. Solar field origin.
    when_ymdhmsz,
):  # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6)
    #              => July 4, 2022 at 11:20 am MDT (-6 hours)
    """
    Computes heliostat surface normal which tracks the sun to the aimpoint.
    Returns only (x,y) components of the surface normal.
    """
    normal_xyz = tracking_surface_normal_xyz(heliostat_xyz, aimpoint_xyz, location_lon_lat, when_ymdhmsz)
    return [normal_xyz[0], normal_xyz[1]]


def tracking_nu(
    heliostat_xyz,  # (x,y,z) in m.     Heliostat origin.
    aimpoint_xyz,  # (x,y,z) in m.     Reflection aim point.
    location_lon_lat,  # (lon,lat) in rad. Solar field origin.
    when_ymdhmsz,
):  # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6)
    #              => July 4, 2022 at 11:20 am MDT (-6 hours)
    """
    Computes nu angle of the heliostat surface normal which tracks the sun to the aimpoint.

    nu is the angle to the projection of the surface normal onto the (x,y) plane, measured ccw from the x axis.
    """
    # Compute heliostat surface normal which tracks the sun to the aimpoint.
    n_xy = tracking_surface_normal_xy(heliostat_xyz, aimpoint_xyz, location_lon_lat, when_ymdhmsz)

    # Extract surface normal coordinates.
    n_x = n_xy[0]
    n_y = n_xy[1]

    # Compute nu.
    nu = math.atan2(n_y, n_x)

    # Return.
    return nu
