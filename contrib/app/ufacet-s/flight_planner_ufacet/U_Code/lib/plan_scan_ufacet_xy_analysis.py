"""
Analysis in (x,y) space to identify UFACET scan segments.



"""

import numpy as np

import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.


def ideal_gaze_xy_list(
    aimpoint_xyz, field_origin_lon_lat, when_ymdhmsz, key_xy, integration_step, bbox_xy
):  # Integration limits.
    # Integrate forward from the key point.
    post_x_list, post_y_list = ideal_gaze_xy_list_aux(
        aimpoint_xyz, field_origin_lon_lat, when_ymdhmsz, key_xy, integration_step, bbox_xy
    )
    # Integrate backward from the key point.
    pre_x_list, pre_y_list = ideal_gaze_xy_list_aux(
        aimpoint_xyz, field_origin_lon_lat, when_ymdhmsz, key_xy, -integration_step, bbox_xy  # Reverse direction
    )
    pre_x_list.reverse()
    pre_y_list.reverse()

    x_list = pre_x_list + post_x_list
    y_list = pre_y_list + post_y_list

    xy_list = [[x, y] for x, y, in zip(x_list, y_list)]
    # Return.
    return xy_list


def ideal_gaze_xy_list_aux(
    aimpoint_xyz, field_origin_lon_lat, when_ymdhmsz, key_xy, integration_step, bbox_xy
):  # Integration limits.
    # Extract integration limits.
    xy_min = bbox_xy[0]
    x_min = xy_min[0]
    y_min = xy_min[1]
    xy_max = bbox_xy[1]
    x_max = xy_max[0]
    y_max = xy_max[1]
    # Integrate from the key point.
    x_c = key_xy[0]
    y_c = key_xy[1]
    curve_x_list = []
    curve_y_list = []
    while ((x_min <= x_c) and (x_c <= x_max)) and ((y_min <= y_c) and (y_c <= y_max)):
        nu = sun_track.tracking_nu([x_c, y_c, 0], aimpoint_xyz, field_origin_lon_lat, when_ymdhmsz)
        gamma = nu + np.pi
        x_c += integration_step * np.cos(gamma)
        y_c += integration_step * np.sin(gamma)
        curve_x_list.append(x_c)
        curve_y_list.append(y_c)
    # Return.
    return curve_x_list, curve_y_list


def segment_approximating_ideal_gaze_curve(curve_xy_list):
    if len(curve_xy_list) == 0:
        return None
    elif len(curve_xy_list) == 1:
        xy0 = curve_xy_list[0]
        return [xy0, xy0]
    else:
        # # Deprecated:  Use endpoints.
        # xy0 = curve_xy_list[0]
        # xyN = curve_xy_list[-1]
        # return [xy0, xyN]

        # Compute best-fit.
        fit_segment_xy = g2d.best_fit_line_segment(curve_xy_list)

        # Orient segment.
        fit_xy0 = fit_segment_xy[0]
        fit_xy1 = fit_segment_xy[1]
        curve_xy0 = curve_xy_list[0]
        curve_xyN = curve_xy_list[-1]
        dx00 = fit_xy0[0] - curve_xy0[0]
        dy00 = fit_xy0[1] - curve_xy0[1]
        d00 = np.sqrt((dx00 * dx00) + (dy00 * dy00))
        dx0N = fit_xy0[0] - curve_xyN[0]
        dy0N = fit_xy0[1] - curve_xyN[1]
        d0N = np.sqrt((dx0N * dx0N) + (dy0N * dy0N))
        if d00 <= d0N:
            oriented_fit_segment = [fit_xy0, fit_xy1]
        else:
            oriented_fit_segment = [fit_xy1, fit_xy0]

        # Return.
        return oriented_fit_segment


def ufacet_xy_analysis(solar_field, aimpoint_xyz, when_ymdhmsz, curve_key_xy_list):
    # Notify progress.
    print("UFACET scan (x,y) analysis...")

    # Fetch solar_field origin.
    field_origin_lon_lat = solar_field.origin_lon_lat

    # Define integration limits.
    bbox_xy = solar_field.heliostat_bounding_box_xy()

    # Construct ideal gaze curves.
    integration_step = 0.1
    list_of_ideal_xy_lists = []
    list_of_best_fit_segment_xys = []
    for key_xy in curve_key_xy_list:
        # Construct ideal gaze curve.
        ideal_xy_list = ideal_gaze_xy_list(
            aimpoint_xyz, field_origin_lon_lat, when_ymdhmsz, key_xy, integration_step, bbox_xy
        )
        list_of_ideal_xy_lists.append(ideal_xy_list)
        # Construct segment approximating curve.
        segment_xy = segment_approximating_ideal_gaze_curve(ideal_xy_list)
        if segment_xy != None:
            list_of_best_fit_segment_xys.append(segment_xy)

    # Return.
    return list_of_ideal_xy_lists, list_of_best_fit_segment_xys
