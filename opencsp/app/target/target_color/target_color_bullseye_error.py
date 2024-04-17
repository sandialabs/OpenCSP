"""
Calculations for 2f color targts.



"""

import math


def meters_given_pixels(d_pixel, dpi):
    """
    Placeholder
    """
    d_inch = d_pixel / dpi
    d_meter = d_inch / (1000.0 / 25.4)
    return d_meter


def surface_normal_error_magnitude_given_radius_in_meters(r_meter, focal_length_meter):
    """
    Computes the surface normal error of a reflection with a reflected image the given
     radius r_meter away from the target center.

     Correct if the mirror design is spherical, the view position is along the optical axis,
     and the target-to-mirror and camera-to-mirror distances are equal to the radius of curvature.

     Returns value in milliradians.
    """

    radius_of_curvature_meter = 2 * focal_length_meter
    reflected_ray_angle = math.atan(r_meter / radius_of_curvature_meter)  # radians
    surface_normal_error = reflected_ray_angle / 2.0  # radians
    return surface_normal_error * 1000.0  # Convert radians to milliradians.


def radius_in_mrad_given_row_col(n_rows, row, col, x_max, cx_offset_pix, y_offset_pix, focal_length_meter):
    """
    Placeholder
    """
    x = col
    y = n_rows - row
    cx = (x_max / 2) + cx_offset_pix
    cy = y_offset_pix
    dx = x - cx
    dy = y - cy
    r_pixel = math.sqrt((dx * dx) + (dy * dy))
    theta = math.atan2(dy, dx)
    # Lookup color bar entry.
    r_meter = meters_given_pixels(r_pixel)
    r_mrad = surface_normal_error_magnitude_given_radius_in_meters(r_meter, focal_length_meter)
    return r_mrad
