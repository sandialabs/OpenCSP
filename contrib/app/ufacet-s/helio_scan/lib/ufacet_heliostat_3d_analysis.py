"""
Class for rendering plots and csv files evaluating 3-d heliostat shape estimates.



"""

import csv

# import glob
# import logging
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# from multiprocessing import Pool
import numpy as np

# from numpy.lib.function_base import sinc
import os
import pandas as pd
from math import sqrt as sqrt

import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.render.axis_3d as ax3d
import opencsp.common.lib.render.image_plot as ip
import opencsp.common.lib.render.PlotAnnotation as pa
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.list_tools as lt
from . import ufacet_pipeline_frame as upf
from .DEPRECATED_utils import *  # ?? SCAFFOLDING RCB -- ELIMINATE THIS


def distance2d(pt1, pt2):
    x1, y1, _ = pt1
    x2, y2, _ = pt2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def distance3d(pt1, pt2):
    x1, y1, z1 = pt1
    x2, y2, z2 = pt2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def read_txt_file_to_heliostat(filename, specifications):
    with open(filename, "r") as f:
        all_corners = []
        for line in f:
            line = line.split("\n")[0]
            corner = line.split(",")
            corner = [float(x) for x in corner]
            all_corners.append(corner)
        if len(all_corners) != specifications.corners_per_heliostat:
            return None
        facets_corners = []
        for facet_indx in range(0, len(all_corners), specifications.corners_per_facet):
            facet = []
            for indx_corner in range(0, specifications.corners_per_facet):
                facet.append(all_corners[facet_indx + indx_corner])
            facets_corners.append(facet)

        heliostat = {}
        for indx in range(0, len(facets_corners)):
            heliostat[indx + 1] = facets_corners[indx]
        return heliostat


def surface_normal_heliostat(heliostat, option, specifications):
    if option == "normalFacet":
        # Surface Normal of centered Facet
        centered_facet_corners = heliostat[specifications.centered_facet]
        top_right = centered_facet_corners[TOP_RIGHT_CORNER_INDX]
        bottom_right = centered_facet_corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left = centered_facet_corners[BOTTOM_LEFT_CORNER_INDX]
        a = np.array(top_right) - np.array(bottom_right)
        b = np.array(bottom_left) - np.array(bottom_right)
        surface_normal = np.cross(a, b)
    elif option == "normalHel":
        # Surface Normal of heliostat
        top_right = heliostat[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX]
        bottom_right = heliostat[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX]
        bottom_left = heliostat[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX]
        a = np.array(top_right) - np.array(bottom_right)
        b = np.array(bottom_left) - np.array(bottom_right)
        surface_normal = np.cross(a, b)
    elif option == "normalOptical":
        # Average Surface Normal
        surf_normals = []
        for _, corners in sorted(heliostat.items()):
            top_right = corners[TOP_RIGHT_CORNER_INDX]
            bottom_right = corners[BOTTOM_RIGHT_CORNER_INDX]
            bottom_left = corners[BOTTOM_LEFT_CORNER_INDX]
            a = np.array(top_right) - np.array(bottom_right)
            b = np.array(bottom_left) - np.array(bottom_right)
            surf_normals.append(np.cross(a, b))

        surface_normal = np.mean(np.array(surf_normals), axis=0)

    surface_normal = surface_normal / np.linalg.norm(surface_normal)
    return surface_normal


def facet_center(top_left, bottom_right, top_right, bottom_left):
    e = np.array(bottom_right) - np.array(top_left)  # direction vector of diagonal α
    f = np.array(bottom_left) - np.array(top_right)  # direction vector of diagonal β
    cdv = np.array(top_right) - np.array(top_left)  # vector from two points in the two lines respectively

    fcd_cross = np.cross(f, cdv)
    fe_cross = np.cross(f, e)
    norm_fcd_cross = np.linalg.norm(fcd_cross)
    norm_fe_cross = np.linalg.norm(fe_cross)

    M1 = np.array(top_left) - (norm_fcd_cross / norm_fe_cross) * e
    M2 = np.array(top_left) + (norm_fcd_cross / norm_fe_cross) * e

    # check if inside the four points
    AM1 = M1 - np.array(top_left)
    AB = np.array(top_right) - np.array(top_left)
    AD = np.array(bottom_left) - np.array(top_left)

    if (
        np.dot(AM1, AB) > 0
        and np.dot(AM1, AB) < np.dot(AB, AB)
        and np.dot(AM1, AD) > 0
        and np.dot(AM1, AD) < np.dot(AD, AD)
    ):
        return M1
    else:
        return M2


def rotate_align_vectors(a, b):  # ?? SCAFFOLDING RCB -- MOVE TO TRANSFORM_3D.PY?
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    u = np.cross(a, b)
    c = np.dot(a, b)

    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    ux_sq = np.matmul(ux, ux)

    R = np.eye(3) + ux + ux_sq * (1 / (1 + c))

    return R


def rotate_around_axis(axis, angle):
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    R1 = [t * ux * ux + c, t * ux * uy + uz * s, t * ux * uz - uy * s]
    R2 = [t * ux * uy - uz * s, t * uy * uy + c, t * uy * uz + ux * s]
    R3 = [t * ux * uz + uy * s, t * uy * uz - ux * s, t * uz * uz + c]

    R = [R1, R2, R3]

    return np.array(R)


def translation(heliostat, specifications, confirm=False):
    # heliostat center
    center = facet_center(
        heliostat[specifications.centered_facet][TOP_LEFT_CORNER_INDX],
        heliostat[specifications.centered_facet][BOTTOM_RIGHT_CORNER_INDX],
        heliostat[specifications.centered_facet][TOP_RIGHT_CORNER_INDX],
        heliostat[specifications.centered_facet][BOTTOM_LEFT_CORNER_INDX],
    )

    translated_heliostat = {}
    for key, corners in sorted(heliostat.items()):
        new_corners = []
        for corner in corners:
            new_corners.append([corner[0] - center[0], corner[1] - center[1], corner[2] - center[2]])

        translated_heliostat[key] = new_corners

    # print new center
    center = facet_center(
        translated_heliostat[specifications.centered_facet][TOP_LEFT_CORNER_INDX],
        translated_heliostat[specifications.centered_facet][BOTTOM_RIGHT_CORNER_INDX],
        translated_heliostat[specifications.centered_facet][TOP_RIGHT_CORNER_INDX],
        translated_heliostat[specifications.centered_facet][BOTTOM_LEFT_CORNER_INDX],
    )
    if confirm:
        print("new center after translation: ", center)
    return translated_heliostat


def find_best_translation(heliostat, heliostat_theoretical_dict=None):
    temp_corners = []
    temp_centers = []
    data_centers_x_sum = 0
    data_centers_y_sum = 0
    data_centers_z_sum = 0

    flat_centers_x_sum = 0
    flat_centers_y_sum = 0
    flat_centers_z_sum = 0
    for key, corners in sorted(heliostat.items()):
        top_left_corner = corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = corners[BOTTOM_LEFT_CORNER_INDX]

        x1, y1, z1 = top_left_corner
        x2, y2, z2 = bottom_right_corner
        mid_point1 = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
        # mid point of diagonal
        x1, y1, z1 = top_right_corner
        x2, y2, z2 = bottom_left_corner
        mid_point2 = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
        mid_point = [
            (mid_point1[0] + mid_point2[0]) / 2,
            (mid_point1[1] + mid_point2[1]) / 2,
            (mid_point1[2] + mid_point2[2]) / 2,
        ]

        data_centers_x_sum += mid_point[0]

        data_centers_y_sum += mid_point[1]

        data_centers_z_sum += mid_point[2]

        top_left_corner = heliostat_theoretical_dict[key][TOP_LEFT_CORNER_INDX]
        top_right_corner = heliostat_theoretical_dict[key][TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = heliostat_theoretical_dict[key][BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = heliostat_theoretical_dict[key][BOTTOM_LEFT_CORNER_INDX]

        flat_center = facet_center(top_left_corner, bottom_right_corner, top_right_corner, bottom_left_corner)
        flat_centers_x_sum += flat_center[0]
        flat_centers_y_sum += flat_center[1]
        flat_centers_z_sum += flat_center[2]

    tx = (data_centers_x_sum - flat_centers_x_sum) / 25
    ty = (data_centers_y_sum - flat_centers_y_sum) / 25
    tz = (data_centers_z_sum - flat_centers_z_sum) / 25
    return [tx, ty, tz]


def rotation(heliostat, surface_normal, confirm=False):
    zaxis = np.array([0, 0, 1])  # theoretical surface normal
    R = rotate_align_vectors(surface_normal, zaxis)
    rotated_heliostat = {}

    for key, corners in sorted(heliostat.items()):
        corners_arr = np.array(corners)
        corners_arr = corners_arr.T
        new_corners = np.matmul(R, corners_arr)
        new_corners = new_corners.T
        rotated_heliostat[key] = new_corners.tolist()

    if confirm:
        surface_normal = np.matmul(R, surface_normal)
        print("new surface normal after rotation :", surface_normal)

    return rotated_heliostat


def rotation_around_surface_normal(heliostat, heliostat_theoretical_dict=None):
    surface_normal = np.array([0, 0, 1])
    a = np.array(heliostat[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX]) - np.array(
        heliostat[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX]
    )
    a = a / np.linalg.norm(a)

    b = np.array(heliostat[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX]) - np.array(
        heliostat[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX]
    )
    b = b / np.linalg.norm(b)

    a_th = np.array(heliostat_theoretical_dict[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX]) - np.array(
        heliostat_theoretical_dict[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX]
    )
    a_th = a_th / np.linalg.norm(a_th)

    b_th = np.array(heliostat_theoretical_dict[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX]) - np.array(
        heliostat_theoretical_dict[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX]
    )
    b_th = b_th / np.linalg.norm(b_th)

    # angles
    theta1 = np.dot(a, a_th)
    theta2 = np.dot(b, b_th)
    # theta           = np.arccos(theta2) # one diagonal
    theta = np.arccos((theta1 + theta2) / 2.0)  # both of them
    cross = np.cross(a, a_th)
    cross = cross / np.linalg.norm(cross)
    if cross[2] < 0:  # positive rotation -- Left Hand Rule
        theta = abs(theta)
    else:
        theta = -abs(theta)

    R = rotate_around_axis(surface_normal, theta)
    rotated_heliostat = {}
    for key, corners in sorted(heliostat.items()):
        corners_arr = np.array(corners).reshape(-1, 3)
        corners_arr = corners_arr.T
        new_corners = np.matmul(R, corners_arr)
        new_corners = new_corners.T
        rotated_heliostat[key] = new_corners.tolist()
    return rotated_heliostat


def scaling(heliostat, specifications, heliostat_theoretical_dict=None):
    tl_th = heliostat_theoretical_dict[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX]
    tr_th = heliostat_theoretical_dict[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX]
    bl_th = heliostat_theoretical_dict[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX]

    tl_rec = heliostat[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX]
    tr_rec = heliostat[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX]
    br_rec = heliostat[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX]
    bl_rec = heliostat[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX]

    dx = (distance2d(tl_rec, tr_rec) + distance2d(bl_rec, br_rec)) / 2.0
    dy = (distance2d(tl_rec, bl_rec) + distance2d(tr_rec, br_rec)) / 2.0

    facet = heliostat[specifications.centered_facet - specifications.facets_per_row]
    center_above = facet_center(
        facet[TOP_LEFT_CORNER_INDX],
        facet[BOTTOM_RIGHT_CORNER_INDX],
        facet[TOP_RIGHT_CORNER_INDX],
        facet[BOTTOM_LEFT_CORNER_INDX],
    )

    facet = heliostat[specifications.centered_facet + specifications.facets_per_row]
    center_below = facet_center(
        facet[TOP_LEFT_CORNER_INDX],
        facet[BOTTOM_RIGHT_CORNER_INDX],
        facet[TOP_RIGHT_CORNER_INDX],
        facet[BOTTOM_LEFT_CORNER_INDX],
    )

    dz = (abs(center_above[2]) + abs(center_below[2])) / 2.0

    scale_x = distance2d(tl_th, tr_th) / dx
    scale_y = distance2d(tl_th, bl_th) / dy
    scale_z = specifications.z_offset / dz
    scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]])

    scaled_heliostat = {}
    for key, corners in sorted(heliostat.items()):
        corners_arr = np.array(corners).reshape(-1, 3)
        corners_arr = corners_arr.T
        new_corners = np.matmul(scale_matrix, corners_arr)
        new_corners = new_corners.T
        scaled_heliostat[key] = new_corners.tolist()

    return scaled_heliostat


def find_best_scaling(heliostat, specifications, single_factor=False):
    return 1.0  # ?? SCAFFOLDING RCB -- TEMPORARY.  PROBABLY ELIMINATE THIS ROUTINE.
    temp_xyz = []
    temp = []
    for _, corners in sorted(heliostat.items()):
        top_left_corner = corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = corners[BOTTOM_LEFT_CORNER_INDX]

        corners = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner, top_left_corner]
        for corner_indx in range(0, len(corners) - 1):
            corner = corners[corner_indx]
            next_corner = corners[corner_indx + 1]
            temp_x = (corner[0] - next_corner[0]) ** 2
            temp_y = (corner[1] - next_corner[1]) ** 2
            temp_z = (corner[2] - next_corner[2]) ** 2
            temp.append([temp_x + temp_y + temp_z])
            temp_xyz.append([temp_x, temp_y, temp_z])

    if single_factor:
        A = np.matrix(temp)
        b = np.matrix([specifications.facet_height**2 for _ in range(0, 100)]).T
        fit = (A.T * A).I * A.T * b
        one_scale = fit[0, 0]
        one_scale = sqrt(one_scale)
        errors = b - A
        residual = np.linalg.norm(errors)
        errors = b - A * fit
        residual = np.linalg.norm(errors)
        scale = one_scale
    else:
        A = np.matrix(temp_xyz)
        b = np.matrix([specifications.facet_height**2 for _ in range(0, 100)]).T
        fit = (A.T * A).I * A.T * b
        scale = [sqrt(abs(fit[0, 0])), sqrt(abs(fit[1, 0])), sqrt(abs(fit[2, 0]))]
        errors = b - A
        residual = np.linalg.norm(errors)
        errors = b - A * fit
        residual = np.linalg.norm(errors)

    return scale


def translate_rotate_scale(
    input_heliostat, option, specifications=None, heliostat_theoretical_dict=None, confirm=False
):
    heliostat = translation(input_heliostat, confirm=confirm, specifications=specifications)
    surface_normal = surface_normal_heliostat(heliostat, option, specifications=specifications)
    heliostat = rotation(heliostat, surface_normal, confirm=confirm)
    heliostat = rotation_around_surface_normal(heliostat, heliostat_theoretical_dict=heliostat_theoretical_dict)
    heliostat = scaling(heliostat, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict)
    return input_heliostat


def plot_heliostat_3d(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    theoretical_flag=False,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
    frame_camera_pose_dict=None,
    camera_suffix=None,
):
    figure_name = hel_name + "_3d_" + option
    if theoretical_flag:
        figure_name += "_displayed_with_flat"
    if frame_camera_pose_dict != None:
        figure_name += camera_suffix
    # figure_name += '.png'

    min_z = 10e2
    max_z = -1
    for _, corners in sorted(heliostat.items()):
        for corner in corners:
            z = corner[2]
            if z < min_z:
                min_z = z
            elif z > max_z:
                max_z = z
    for _, corners in heliostat_theoretical_dict.items():
        for corner in corners:
            z = corner[2]
            if z < min_z:
                min_z = z
            elif z > max_z:
                max_z = z

    fig = plt.figure()
    #    ax = plt.axes(projection='3d')  # Perspective projection
    ax = plt.axes(projection="3d", proj_type="ortho")  # Orthographic projection
    plt.title(title_prefix)
    if theoretical_flag:
        corners = [
            heliostat_theoretical_dict[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX],
            heliostat_theoretical_dict[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX],
            heliostat_theoretical_dict[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX],
            heliostat_theoretical_dict[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX],
        ]
        xdata = [corner[0] for corner in corners]
        ydata = [corner[1] for corner in corners]
        zdata = [corner[2] for corner in corners]
        ax.scatter3D(xdata[0], ydata[0], zdata[0], facecolor="tab:blue")
        ax.scatter3D(xdata[1], ydata[1], zdata[1], facecolor="tab:orange")
        ax.scatter3D(xdata[2], ydata[2], zdata[2], facecolor="tab:green")
        ax.scatter3D(xdata[3], ydata[3], zdata[3], facecolor="c")
        # draw center
        center = facet_center(
            heliostat_theoretical_dict[specifications.centered_facet][TOP_LEFT_CORNER_INDX],
            heliostat_theoretical_dict[specifications.centered_facet][BOTTOM_RIGHT_CORNER_INDX],
            heliostat_theoretical_dict[specifications.centered_facet][TOP_RIGHT_CORNER_INDX],
            heliostat_theoretical_dict[specifications.centered_facet][BOTTOM_LEFT_CORNER_INDX],
        )
        ax.scatter3D(center[0], center[1], center[2], facecolor="tab:green")
        # draw heliostat
        for _, corners in heliostat_theoretical_dict.items():
            temp_corners = corners.copy()
            temp_corners.append(temp_corners[0])  # cyclic
            xline, yline, zline = [], [], []
            # draw facets
            for indx in range(0, len(temp_corners) - 1):
                xline.append(temp_corners[indx][0])
                xline.append(temp_corners[indx + 1][0])
                yline.append(temp_corners[indx][1])
                yline.append(temp_corners[indx + 1][1])
                zline.append(temp_corners[indx][2])
                zline.append(temp_corners[indx + 1][2])
                ax.plot3D(xline, yline, zline, "m")

    # draw top corners
    corners = [
        heliostat[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX],
        heliostat[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX],
        heliostat[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX],
        heliostat[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX],
    ]
    xdata = [corner[0] for corner in corners]
    ydata = [corner[1] for corner in corners]
    zdata = [corner[2] for corner in corners]
    ax.scatter3D(xdata[0], ydata[0], zdata[0], facecolor="tab:blue")
    ax.scatter3D(xdata[1], ydata[1], zdata[1], facecolor="tab:orange")
    ax.scatter3D(xdata[2], ydata[2], zdata[2], facecolor="tab:green")
    ax.scatter3D(xdata[3], ydata[3], zdata[3], facecolor="c")
    # draw center
    center = facet_center(
        heliostat[specifications.centered_facet][TOP_LEFT_CORNER_INDX],
        heliostat[specifications.centered_facet][BOTTOM_RIGHT_CORNER_INDX],
        heliostat[specifications.centered_facet][TOP_RIGHT_CORNER_INDX],
        heliostat[specifications.centered_facet][BOTTOM_LEFT_CORNER_INDX],
    )
    ax.scatter3D(center[0], center[1], center[2], facecolor="tab:green")
    # draw heliostat
    for _, corners in sorted(heliostat.items()):
        temp_corners = corners.copy()
        temp_corners.append(temp_corners[0])  # cyclic
        xline, yline, zline = [], [], []
        # draw facets
        for indx in range(0, len(temp_corners) - 1):
            xline.append(temp_corners[indx][0])
            xline.append(temp_corners[indx + 1][0])
            yline.append(temp_corners[indx][1])
            yline.append(temp_corners[indx + 1][1])
            zline.append(temp_corners[indx][2])
            zline.append(temp_corners[indx + 1][2])
            ax.plot3D(xline, yline, zline, "tab:blue")

    # draw cameras
    if frame_camera_pose_dict != None:
        print("In plot_heliostat_3d(), figure_name = ", figure_name)
        print("In plot_heliostat_3d(), number of camera frames = ", len(frame_camera_pose_dict.keys()))
        idx = 0
        for rvec_tvec in dt.list_of_values_in_sorted_key_order(
            frame_camera_pose_dict
        ):  # ?? IF WE MAKE FRAME_CAMERA_POSE_DICT A CLASS, THIS WILL NEED TO BE UPDATED.
            idx += 1
            rvec = rvec_tvec[0]
            tvec = rvec_tvec[1]
            t_x = tvec[0][0]
            t_y = tvec[1][0]
            t_z = tvec[2][0]
            print("In plot_heliostat_3d(), t_x, t_y, t_z = ", t_x, t_y, t_z)
            ax.scatter3D(
                -t_x, t_y, t_z, s=0.5, label=str(idx)
            )  # 2021-11-14 4:00 AM.  Discovered that tvec_x coordinate is reversed compared to heliostat x axis.
    # ax.scatter3D(t_x, t_y, t_z, s=0.5, label=str(idx))
    # ax.scatter3D(t_x, t_y, t_z, facecolor='b', s=0.5, label=str(idx))
    # plt.legend()

    # if axes_equal:
    #     set_3d_axes_equal(ax)
    # else:
    #     ax.set_zlim3d(bottom=min_z-0.05, top=max_z+0.05)
    if frame_camera_pose_dict == None:
        ax.set_zlim3d(bottom=min_z - 0.1, top=max_z + 0.1)
    # ax.set_zlim3d(bottom=min_z-0.05, top=max_z+0.05)
    else:
        ax3d.set_3d_axes_equal(ax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    # plt.show()
    path_figure_base = os.path.join(saving_path, figure_name)
    path_figure = path_figure_base + ".png"
    print("In plot_heliostat_3d(), saving figure:", path_figure)  # ?? SCAFFOLDING RCB -- TEMPORARY ?
    plt.savefig(path_figure, dpi=1200)
    if frame_camera_pose_dict != None:
        # ?? SCAFFOLDING RCB -- CALL plot_and_save_plane_views(path_figure_base, ax) HERE INSTEAD.

        # (Az, el) describe a camera position, viewing back toward origin.
        # ax.view_init(-90, -90)  # x-negy plane view
        # path_figure = path_figure_base + '_-90_-90' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)
        # ax.view_init(-90, 0)  # y-x plane view
        # path_figure = path_figure_base + '_-90_0' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)
        # ax.view_init(-90, 90)  # negx-y plane view
        # path_figure = path_figure_base + '_-90_90' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)

        # ax.view_init(0, -90)  # x-z plane view
        # path_figure = path_figure_base + '_0_-90' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)
        # ax.view_init(0, 0)  # y-z plane view
        # path_figure = path_figure_base + '_0_0' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)
        # ax.view_init(0, 90)  # negx-z plane view
        # path_figure = path_figure_base + '_0_90' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)

        # ax.view_init(90, -90)  # x-y plane view
        # path_figure = path_figure_base + '_90_-90' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)
        # ax.view_init(90, 0)  # y-negx plane view
        # path_figure = path_figure_base + '_90_0' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)
        # ax.view_init(90, 90)  # negx-negy plane view
        # path_figure = path_figure_base + '_90_90' + '.png'
        # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
        # plt.savefig(path_figure, dpi=1200)

        # XY
        ax.view_init(90, -90)  # x-y plane view
        path_figure = path_figure_base + "_xy" + ".png"
        print("In plot_heliostat_3d(), saving figure:", path_figure)  # ?? SCAFFOLDING RCB -- TEMPORARY ?
        plt.savefig(path_figure, dpi=1200)

        # XZ
        ax.view_init(0, -90)  # x-z plane view
        path_figure = path_figure_base + "_xz" + ".png"
        print("In plot_heliostat_3d(), saving figure:", path_figure)  # ?? SCAFFOLDING RCB -- TEMPORARY ?
        plt.savefig(path_figure, dpi=1200)

        # YZ
        ax.view_init(0, 0)  # y-z plane view
        path_figure = path_figure_base + "_yz" + ".png"
        print("In plot_heliostat_3d(), saving figure:", path_figure)  # ?? SCAFFOLDING RCB -- TEMPORARY ?
        plt.savefig(path_figure, dpi=1200)

    plt.close()


def plot_heliostat_2d(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    theoretical_flag=False,
    axes_equal=True,
    plot_surface_normals=False,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
    normal_wrt_average=False,
):
    figure_name = hel_name + "_2d_" + option
    if plot_surface_normals:
        figure_name += "_displayed_normals"
    if theoretical_flag:
        figure_name += "_displayed_flat"
    figure_name += ".png"

    plt.figure()
    title = title_prefix
    if plot_surface_normals and normal_wrt_average:
        title = title + ", Normals Relative to Average Normal"
    plt.title(title)
    if theoretical_flag:
        corners = [
            heliostat_theoretical_dict[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX],
            heliostat_theoretical_dict[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX],
            heliostat_theoretical_dict[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX],
            heliostat_theoretical_dict[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX],
        ]
        xdata = [corner[0] for corner in corners]
        ydata = [corner[1] for corner in corners]
        plt.scatter(xdata[0], ydata[0], facecolor="tab:blue")
        plt.scatter(xdata[1], ydata[1], facecolor="tab:orange")
        plt.scatter(xdata[2], ydata[2], facecolor="tab:green")
        plt.scatter(xdata[3], ydata[3], facecolor="c")
        # draw center
        center = facet_center(
            heliostat_theoretical_dict[specifications.centered_facet][TOP_LEFT_CORNER_INDX],
            heliostat_theoretical_dict[specifications.centered_facet][BOTTOM_RIGHT_CORNER_INDX],
            heliostat_theoretical_dict[specifications.centered_facet][TOP_RIGHT_CORNER_INDX],
            heliostat_theoretical_dict[specifications.centered_facet][BOTTOM_LEFT_CORNER_INDX],
        )
        plt.scatter(center[0], center[1], facecolor="tab:green")
        for _, corners in heliostat_theoretical_dict.items():
            temp_corners = corners.copy()
            temp_corners.append(temp_corners[0])  # cyclic
            xline, yline, zline = [], [], []
            # draw facets
            for indx in range(0, len(temp_corners) - 1):
                xline.append(temp_corners[indx][0])
                xline.append(temp_corners[indx + 1][0])
                yline.append(temp_corners[indx][1])
                yline.append(temp_corners[indx + 1][1])
                plt.plot(xline, yline, "m", alpha=0.6)

    """
    Plotting Heliostat's corners, Centers, and Sides
    """
    corners = [
        heliostat[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX],
        heliostat[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX],
        heliostat[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX],
        heliostat[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX],
    ]
    xdata = [corner[0] for corner in corners]
    ydata = [corner[1] for corner in corners]
    plt.scatter(xdata[0], ydata[0], facecolor="tab:blue")
    plt.scatter(xdata[1], ydata[1], facecolor="tab:orange")
    plt.scatter(xdata[2], ydata[2], facecolor="tab:green")
    plt.scatter(xdata[3], ydata[3], facecolor="c")
    # draw center
    center = facet_center(
        heliostat[specifications.centered_facet][TOP_LEFT_CORNER_INDX],
        heliostat[specifications.centered_facet][BOTTOM_RIGHT_CORNER_INDX],
        heliostat[specifications.centered_facet][TOP_RIGHT_CORNER_INDX],
        heliostat[specifications.centered_facet][BOTTOM_LEFT_CORNER_INDX],
    )
    plt.scatter(center[0], center[1], facecolor="tab:green")
    for _, corners in sorted(heliostat.items()):
        temp_corners = corners.copy()
        temp_corners.append(temp_corners[0])  # cyclic
        xline, yline = [], []
        # draw facets
        for indx in range(0, len(temp_corners) - 1):
            xline.append(temp_corners[indx][0])
            xline.append(temp_corners[indx + 1][0])
            yline.append(temp_corners[indx][1])
            yline.append(temp_corners[indx + 1][1])
            plt.plot(xline, yline, "tab:blue")

    """
    Plotting Each Facet's surface normal
    """
    if plot_surface_normals:
        needle_scale = 40
        # needle_scale = 150
        # Compute average canting angle.
        if normal_wrt_average:
            ux_sum = 0
            uy_sum = 0
            u_count = 0
            for _, corners in sorted(
                heliostat.items()
            ):  # ?? SCAFFOLDING RCB -- THIS IS CRUFTY; REFACTOR TO COMPUTE IN A A SINGLE ANALYTIC PLACE, CONSISTENTLY.
                top_left = corners[TOP_LEFT_CORNER_INDX]
                top_right = corners[TOP_RIGHT_CORNER_INDX]
                bottom_right = corners[BOTTOM_RIGHT_CORNER_INDX]
                bottom_left = corners[BOTTOM_LEFT_CORNER_INDX]
                center = facet_center(top_left, bottom_right, top_right, bottom_left)
                diagonal_a = np.array(bottom_right) - np.array(top_left)
                diagonal_b = np.array(bottom_left) - np.array(top_right)
                surface_normal = np.cross(diagonal_b, diagonal_a)
                surface_normal = surface_normal / np.linalg.norm(surface_normal)
                plt.scatter(center[0], center[1], facecolor="k", s=1)

                ux, uy, _ = surface_normal
                ux_sum += ux
                uy_sum += uy
                u_count += 1
            ux_avg = ux_sum / u_count
            uy_avg = uy_sum / u_count
        for _, corners in sorted(heliostat.items()):
            top_left = corners[TOP_LEFT_CORNER_INDX]
            top_right = corners[TOP_RIGHT_CORNER_INDX]
            bottom_right = corners[BOTTOM_RIGHT_CORNER_INDX]
            bottom_left = corners[BOTTOM_LEFT_CORNER_INDX]
            center = facet_center(top_left, bottom_right, top_right, bottom_left)
            diagonal_a = np.array(bottom_right) - np.array(top_left)
            diagonal_b = np.array(bottom_left) - np.array(top_right)
            surface_normal = np.cross(diagonal_b, diagonal_a)
            surface_normal = surface_normal / np.linalg.norm(surface_normal)
            plt.scatter(center[0], center[1], facecolor="k", s=1)

            ux, uy, _ = surface_normal
            if normal_wrt_average:
                ux -= ux_avg
                uy -= uy_avg
            next_point = [center[0] + needle_scale * ux, center[1] + needle_scale * uy]
            plt.plot([center[0], next_point[0]], [center[1], next_point[1]], color="c")

    if axes_equal:
        plt.axis("equal")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig(saving_path + "/" + figure_name)
    # plt.show()
    plt.close()


def plot_canting_angles(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    directions_cosines=False,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
    plot=True,
    normal_wrt_average=False,
):
    anglesx = []
    anglesy = []
    anglesz = []
    canting_angles = {}
    for key, corners in sorted(heliostat.items()):
        top_left = corners[TOP_LEFT_CORNER_INDX]
        top_right = corners[TOP_RIGHT_CORNER_INDX]
        bottom_right = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left = corners[BOTTOM_LEFT_CORNER_INDX]
        diagonal_a = np.array(bottom_right) - np.array(top_left)
        diagonal_b = np.array(bottom_left) - np.array(top_right)
        surface_normal = np.cross(diagonal_b, diagonal_a)
        surface_normal = surface_normal / np.linalg.norm(surface_normal)
        if directions_cosines:
            ax = np.arccos(surface_normal[0])
            ay = np.arccos(surface_normal[1])
            az = np.arccos(surface_normal[2])
        else:
            ax = surface_normal[0]
            ay = surface_normal[1]
            az = surface_normal[2]

        anglesx.append(ax)
        anglesy.append(ay)
        anglesz.append(az)
        canting_angles[key] = [ax, ay, az]

    if (
        normal_wrt_average
    ):  # ?? SCAFFOLDING RCB -- THIS IS CRUFTY.  THIS SHOULD GO IN AN ANALYSIS PLACE, AND ONLY COMPUTE ONCE.  SEE ALSO SIMILAR CODE IN PLOT NORMALS FUNCTION.
        # Compute average canting angle x and y components.
        average_ax = sum(anglesx) / len(anglesx)
        average_ay = sum(anglesy) / len(anglesy)
        # Compute offsets.
        offset_anglesx = [(x - average_ax) for x in anglesx]
        offset_anglesy = [(y - average_ay) for y in anglesy]
        offset_anglesz = []
        for offset_ax, offset_ay in zip(offset_anglesx, offset_anglesy):
            offset_az = np.sqrt(1.0 - (offset_ax**2 + offset_ay**2))
            offset_anglesz.append(offset_az)
        offset_canting_angles = {}
        for key in dt.sorted_keys(heliostat):
            axyz = canting_angles[key]
            ax2 = axyz[0]
            ay2 = axyz[1]
            az2 = axyz[2]
            offset_ax2 = ax2 - average_ax
            offset_ay2 = ay2 - average_ay
            offset_az2 = np.sqrt(1.0 - (offset_ax2**2 + offset_ay2**2))
            offset_canting_angles[key] = [offset_ax2, offset_ay2, offset_az2]
        # Set output.
        anglesx = offset_anglesx
        anglesy = offset_anglesy
        anglesz = offset_anglesz
        canting_angles = offset_canting_angles

    if plot:
        df = pd.DataFrame({"Nx": anglesx}, index=[i + 1 for i in range(0, specifications.facets_per_heliostat)])
        ax = df.plot.bar(rot=0, color={"Nx": "tab:blue"}, figsize=(15, 10))
        title = title_prefix + ": X-Component of Surface Normal"
        if normal_wrt_average:
            title += ", Relative to Average Normal"
        plt.title(title)
        plt.xlabel("Facet id")
        plt.ylabel("X-component of units of Surface Normal")
        y_axis_max = 0.030  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        plt.ylim(
            -y_axis_max, y_axis_max
        )  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        #        plt.ylim(min(anglesx)-0.005, max(anglesx)+0.005)
        plt.grid(axis="y")
        figure_name = hel_name + "_canglesX_" + option + ".png"
        plt.savefig(saving_path + "/" + figure_name)
        plt.close()

        df = pd.DataFrame({"Ny": anglesy}, index=[i + 1 for i in range(0, specifications.facets_per_heliostat)])
        ax = df.plot.bar(rot=0, color={"Ny": "tab:orange"}, figsize=(15, 10))
        title = title_prefix + ": Y-Component of Surface Normal"
        if normal_wrt_average:
            title += ", Relative to Average Normal"
        plt.title(title)
        plt.xlabel("Facet id")
        plt.ylabel("Y-component of units of surface normal")
        plt.ylim(
            -y_axis_max, y_axis_max
        )  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        #        plt.ylim(min(anglesy)-0.001, max(anglesy)+0.001)
        plt.grid(axis="y")
        figure_name = hel_name + "_canglesY_" + option + ".png"
        plt.savefig(saving_path + "/" + figure_name)
        plt.close()

        df = pd.DataFrame({"Nz": anglesz}, index=[i + 1 for i in range(0, specifications.facets_per_heliostat)])
        ax = df.plot.bar(rot=0, color={"Nz": "magenta"}, figsize=(15, 10))
        title = title_prefix + ": Z-Component of Surface Normal"
        if normal_wrt_average:
            title += ", Relative to Average Normal"
        plt.title(title)
        plt.xlabel("Facet id")
        plt.ylabel("Z-component of units of Surface Normal")
        plt.ylim(min(anglesz) - 0.000001, 1)
        plt.grid(axis="y")
        figure_name = hel_name + "_canglesZ_" + option + ".png"
        plt.savefig(saving_path + "/" + figure_name)
        plt.close()

        # plt.show()
        plt.close()

    return canting_angles


def plot_corners_coordinates_differences(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    xy=True,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
):
    top_left = []
    top_right = []
    bottom_right = []
    bottom_left = []
    for key, corners in sorted(heliostat.items()):
        top_left_corner = corners[TOP_LEFT_CORNER_INDX]
        top_left_flat = heliostat_theoretical_dict[key][TOP_LEFT_CORNER_INDX]
        if xy:
            top_left.append(distance2d(top_left_corner, top_left_flat))
        else:
            top_left.append(abs(top_left_corner[2] - top_left_flat[2]))

        top_right_corner = corners[TOP_RIGHT_CORNER_INDX]
        top_right_flat = heliostat_theoretical_dict[key][TOP_RIGHT_CORNER_INDX]
        if xy:
            top_right.append(distance2d(top_right_corner, top_right_flat))
        else:
            top_right.append(abs(top_right_corner[2] - top_right_flat[2]))

        bottom_right_corner = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_right_flat = heliostat_theoretical_dict[key][BOTTOM_RIGHT_CORNER_INDX]
        if xy:
            bottom_right.append(distance2d(bottom_right_corner, bottom_right_flat))
        else:
            bottom_right.append(abs(bottom_right_corner[2] - bottom_right_flat[2]))

        bottom_left_corner = corners[BOTTOM_LEFT_CORNER_INDX]
        bottom_left_flat = heliostat_theoretical_dict[key][BOTTOM_LEFT_CORNER_INDX]
        if xy:
            bottom_left.append(distance2d(bottom_left_corner, bottom_left_flat))
        else:
            bottom_left.append(abs(bottom_left_corner[2] - bottom_left_flat[2]))

    df = pd.DataFrame(
        {
            "Top Left Corner": top_left,
            "Top Right Corner": top_right,
            "Bottom Right Corner": bottom_right,
            "Bottom Left Corner": bottom_left,
        },
        index=[i + 1 for i in range(0, specifications.facets_per_heliostat)],
    )
    ax = df.plot.bar(
        rot=0,
        color={
            "Top Left Corner": "blue",
            "Top Right Corner": "red",
            "Bottom Right Corner": "green",
            "Bottom Left Corner": "magenta",
        },
    )
    if xy:
        plt.title(title_prefix + ": XY-Difference")
    else:
        plt.title(title_prefix + ": Z-Difference")
    plt.xlabel("Facet id")
    plt.ylabel("Y (m)")
    plt.show()
    plt.close()


def plot_pose_estimation(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    best_translation=True,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
    plot=True,
):
    figure_name = hel_name + "_poseXYZ_" + option
    if best_translation:
        figure_name += "_bestTranslation"
    figure_name += ".png"
    T = [0, 0, 0]
    if best_translation:
        T = find_best_translation(heliostat, heliostat_theoretical_dict=heliostat_theoretical_dict)
    xdiff = []
    ydiff = []
    zdiff = []
    pose_estimations = {}
    for key, corners in sorted(heliostat.items()):
        top_left_corner = corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = corners[BOTTOM_LEFT_CORNER_INDX]
        center = facet_center(top_left_corner, bottom_right_corner, top_right_corner, bottom_left_corner)
        x, y, z = center
        x -= T[0]
        y -= T[1]
        z -= T[2]
        top_left_corner = heliostat_theoretical_dict[key][TOP_LEFT_CORNER_INDX]
        top_right_corner = heliostat_theoretical_dict[key][TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = heliostat_theoretical_dict[key][BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = heliostat_theoretical_dict[key][BOTTOM_LEFT_CORNER_INDX]
        center_th = facet_center(top_left_corner, bottom_right_corner, top_right_corner, bottom_left_corner)
        xth, yth, zth = center_th
        xdiff.append(x - xth)
        ydiff.append(y - yth)
        zdiff.append(z - zth)
        pose_estimations[key] = [x - xth, y - yth, z - zth]
    if plot:
        df = pd.DataFrame(
            {"X offset": xdiff, "Y offset": ydiff, "Z offset": zdiff},
            index=[i + 1 for i in range(0, specifications.facets_per_heliostat)],
        )
        ax = df.plot.bar(
            rot=0, color={"X offset": "tab:blue", "Y offset": "tab:orange", "Z offset": "cyan"}, figsize=(15, 10)
        )
        plt.grid(axis="y")
        plt.title(title_prefix + ": Centroid Difference")
        plt.xlabel("Facet id")
        plt.ylabel("Y (m)")
        plt.savefig(saving_path + "/" + figure_name)
        # plt.show()
        plt.close()
    return pose_estimations


def plot_pose_rotation_estimation(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    best_translation=True,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
    plot=True,
):
    figure_name = hel_name + "_poseRotZ_" + option
    figure_name += ".png"
    rot_z_deg_list = []
    pose_rotation_estimations = {}
    for key, corners in sorted(heliostat.items()):
        top_left_corner = corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = corners[BOTTOM_LEFT_CORNER_INDX]
        tl_x = top_left_corner[0]
        tl_y = top_left_corner[1]
        tr_x = top_right_corner[0]
        tr_y = top_right_corner[1]
        br_x = bottom_right_corner[0]
        br_y = bottom_right_corner[1]
        bl_x = bottom_left_corner[0]
        bl_y = bottom_left_corner[1]
        rot_z_bl2tl = math.atan2((tl_y - bl_y), (tl_x - bl_x)) - (math.pi / 2.0)
        rot_z_br2tr = math.atan2((tr_y - br_y), (tr_x - br_x)) - (math.pi / 2.0)
        rot_z_bl2br = math.atan2((br_y - bl_y), (br_x - bl_x))
        rot_z_tl2tr = math.atan2((tr_y - tl_y), (tr_x - tl_x))
        rot_z = (rot_z_bl2tl + rot_z_br2tr + rot_z_bl2br + rot_z_tl2tr) / 4.0
        rot_z_deg_list.append(np.degrees(rot_z))
        pose_rotation_estimations[key] = [rot_z]
    if plot:
        df = pd.DataFrame(
            {"Rotation About Z": rot_z_deg_list}, index=[i + 1 for i in range(0, specifications.facets_per_heliostat)]
        )
        ax = df.plot.bar(rot=0, color={"Rotation About Z": "tab:blue"}, figsize=(15, 10))
        plt.grid(axis="y")
        plt.title(title_prefix + ": Facet Z Rotation")
        plt.xlabel("Facet id")
        plt.ylabel("RotZ (degrees)")
        plt.savefig(saving_path + "/" + figure_name)
        # plt.show()
        plt.close()
    return pose_rotation_estimations


def plot_square_sides_quality(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    best_scaling=True,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
    plot=True,
):
    scale = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if best_scaling:
        scale = find_best_scaling(heliostat, specifications=specifications, single_factor=False)
        if isinstance(scale, list):
            scale_x, scale_y, scale_z = scale
            scale = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]])
        else:
            scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])

    rmse = 0
    residual = 0
    top_side, top_error = [], []
    right_side, right_error = [], []
    bottom_side, bottom_error = [], []
    left_side, left_error = [], []
    square_sides_errors = {}
    for key, corners in sorted(heliostat.items()):
        top_left_corner = corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = corners[BOTTOM_LEFT_CORNER_INDX]

        corners = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
        corners_arr = np.array(corners).reshape(-1, 3)
        corners_arr = corners_arr.T
        new_corners = np.matmul(scale, corners_arr)
        new_corners = new_corners.T
        new_corners = new_corners.tolist()

        top_left_corner = new_corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = new_corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = new_corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = new_corners[BOTTOM_LEFT_CORNER_INDX]

        top = distance3d(top_left_corner, top_right_corner)
        top_side.append(top)
        top_error.append(
            top - specifications.facet_width
        )  # ?? SCAFFOLDING RCB -- THESE BASIC ERROR DIFFERENCES ARE RECOMPUTED SEVERAL TIMES BELOW.  ELIMINATE REDUNDANT CALCULATION AND OPPORTUNITY FOR ERROR

        right = distance3d(top_right_corner, bottom_right_corner)
        right_side.append(right)
        right_error.append(right - specifications.facet_height)

        bottom = distance3d(bottom_left_corner, bottom_right_corner)
        bottom_side.append(bottom)
        bottom_error.append(bottom - specifications.facet_width)

        left = distance3d(bottom_left_corner, top_left_corner)
        left_side.append(left)
        left_error.append(left - specifications.facet_height)

        square_sides_errors[key] = [
            top - specifications.facet_width,
            right - specifications.facet_height,
            bottom - specifications.facet_width,
            left - specifications.facet_height,
        ]

        rmse += (
            (top - specifications.facet_width) ** 2
            + (right - specifications.facet_height) ** 2
            + (bottom - specifications.facet_width) ** 2
            + (left - specifications.facet_height) ** 2
        )
        mean_error = (
            abs(top - specifications.facet_width)
            + abs(right - specifications.facet_height)
            + abs(bottom - specifications.facet_width)
            + abs(left - specifications.facet_height)
        )

    if plot:
        rmse /= specifications.corners_per_heliostat
        mean_error /= specifications.corners_per_heliostat
        rmse = sqrt(rmse)
        # df = pd.DataFrame({'Top Side': top_side,'Bottom Side': bottom_side, 'Right Side': right_side, 'Left Side': left_side}, index=[i+1 for i in range(0, specifications.facets_per_heliostat)])
        # ax = df.plot.bar(rot=0, color={"Top Side": "tab:blue", "Right Side": "tab:orange", "Bottom Side": "tab:olive", "Left Side":"tab:purple"}, figsize=(15,10))
        # x      = [indx for indx in range(-1, specifications.facets_per_heliostat +1)]
        # y      = [specifications.facet_height  for _ in range(-1, specifications.facets_per_heliostat +1)]  # ?? SCAFFOLDING RCB -- SHOULD THIS SOMETIMES BE FACET WIDTH?
        # plt.plot(x, y, linewidth=1.5, color='black', linestyle='dashed',label='Facet Side')
        # plt.title(title_prefix)
        # plt.xlabel('Facet id')
        # plt.ylabel('Y (m)')
        # bottom = min([min(top_side), min(right_side), min(bottom_side), min(left_side)])
        # top    = max([max(top_side), max(right_side), max(bottom_side), max(left_side)])
        # plt.ylim(bottom - 0.001, top + 0.001)
        # plt.grid(axis='y')
        # plt.legend()
        # figure_name = hel_name + '_squareSides_' + option
        # if best_scaling:
        #     figure_name += '_bestScaling'
        # figure_name += '.png'
        # plt.savefig(saving_path + '/' + figure_name)
        # plt.close()

        df = pd.DataFrame(
            {
                "Top Side Error": top_error,
                "Bottom Side Error": bottom_error,
                "Right Side Error": right_error,
                "Left Side Error": left_error,
            },
            index=[i + 1 for i in range(0, specifications.facets_per_heliostat)],
        )
        ax = df.plot.bar(
            rot=0,
            color={
                "Top Side Error": "tab:blue",
                "Right Side Error": "tab:orange",
                "Bottom Side Error": "tab:olive",
                "Left Side Error": "tab:purple",
            },
            figsize=(15, 10),
        )
        plt.title(title_prefix)
        plt.xlabel("Facet id")
        plt.ylabel("Meters")
        bottom = min([min(top_error), min(right_error), min(bottom_error), min(left_error)])
        top = max([max(top_error), max(right_error), max(bottom_error), max(left_error)])
        x = [indx for indx in range(-1, specifications.facets_per_heliostat + 1)]
        y = [rmse for _ in range(-1, specifications.facets_per_heliostat + 1)]
        plt.plot(x, y, linewidth=1.5, color="red", linestyle="dashed")
        y = [-rmse for _ in range(-1, specifications.facets_per_heliostat + 1)]
        plt.plot(x, y, linewidth=1.5, color="red", linestyle="dashed", label="RMSE")
        plt.ylim(bottom + (10e-2 * bottom), top + 10e-2 * top)
        plt.grid(axis="y")
        plt.legend()
        # plt.show()
        figure_name = hel_name + "_squareSidesErrors_" + option
        if best_scaling:
            figure_name += "_bestScaling"
        figure_name += ".png"
        plt.savefig(saving_path + "/" + figure_name)
        plt.close()

    return square_sides_errors


def plot_square_diagonals_quality(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    best_scaling=True,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
    plot=True,
):
    scale = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if best_scaling:
        scale = find_best_scaling(heliostat, specifications=specifications, single_factor=False)
        if isinstance(scale, list):
            scale_x, scale_y, scale_z = scale
            scale = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]])
        else:
            scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])

    rmse = 0
    residual = 0
    bltr_dist_list, bltr_error_list = [], []
    brtl_dist_list, brtl_error_list = [], []
    square_diagonals_errors = {}
    for key, corners in sorted(heliostat.items()):
        top_left_corner = corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = corners[BOTTOM_LEFT_CORNER_INDX]

        # ?? SCAFFOLDING RCB -- WHAT IS THIS?  IS IT NECESSARY?  IF NOT, THEN RIP OUT, HERE AND IN SQUARE SIDE QUALITY FXN.
        corners = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
        corners_arr = np.array(corners).reshape(-1, 3)
        corners_arr = corners_arr.T
        new_corners = np.matmul(scale, corners_arr)
        new_corners = new_corners.T
        new_corners = new_corners.tolist()

        top_left_corner = new_corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = new_corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = new_corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = new_corners[BOTTOM_LEFT_CORNER_INDX]

        std_diagonal_dist = np.sqrt(specifications.facet_width**2 + specifications.facet_height**2)

        bltr_dist = distance3d(bottom_left_corner, top_right_corner)
        bltr_error = bltr_dist - std_diagonal_dist
        bltr_dist_list.append(bltr_dist)
        bltr_error_list.append(bltr_error)

        brtl_dist = distance3d(bottom_right_corner, top_left_corner)
        brtl_error = brtl_dist - std_diagonal_dist
        brtl_dist_list.append(brtl_dist)
        brtl_error_list.append(brtl_error)

        square_diagonals_errors[key] = [bltr_error, brtl_error]

        rmse += bltr_error**2 + brtl_error**2
        mean_error = abs(bltr_error) + abs(brtl_error)

    if plot:
        rmse /= specifications.corners_per_heliostat
        mean_error /= specifications.corners_per_heliostat
        rmse = sqrt(rmse)
        # df = pd.DataFrame({'BottomLeft-TopRight': bltr_dist_list, 'BottomRight-TopLeft': brtl_dist_list}, index=[i+1 for i in range(0, specifications.facets_per_heliostat)])
        # ax = df.plot.bar(rot=0, color={"BottomLeft-TopRight": "tab:blue", "BottomRight-TopLeft": "tab:orange"}, figsize=(15,10))
        # x      = [indx for indx in range(-1, specifications.facets_per_heliostat +1)]
        # y      = [std_diagonal_dist  for _ in range(-1, specifications.facets_per_heliostat +1)]
        # plt.plot(x, y, linewidth=1.5, color='black', linestyle='dashed',label='Diagonal')
        # plt.title(title_prefix)
        # plt.xlabel('Facet id')
        # plt.ylabel('Y (m)')
        # bottom = min([min(bltr_dist_list), min(brtl_dist_list)])
        # top    = max([max(bltr_dist_list), max(brtl_dist_list)])
        # plt.ylim(bottom - 0.001, top + 0.001)
        # plt.grid(axis='y')
        # plt.legend()
        # figure_name = hel_name + '_squareDiagonals_' + option
        # if best_scaling:
        #     figure_name += '_bestScaling'
        # figure_name += '.png'
        # plt.savefig(saving_path + '/' + figure_name)
        # plt.close()

        df = pd.DataFrame(
            {"BottomLeft-TopRight Error": bltr_error_list, "BottomRight-TopLeft Error": brtl_error_list},
            index=[i + 1 for i in range(0, specifications.facets_per_heliostat)],
        )
        ax = df.plot.bar(
            rot=0,
            color={"BottomLeft-TopRight Error": "tab:blue", "BottomRight-TopLeft Error": "tab:orange"},
            figsize=(15, 10),
        )
        plt.title(title_prefix)
        plt.xlabel("Facet id")
        plt.ylabel("Meters")
        bottom = min([min(bltr_error_list), min(brtl_error_list)])
        top = max([max(bltr_error_list), max(brtl_error_list)])
        x = [indx for indx in range(-1, specifications.facets_per_heliostat + 1)]
        y = [rmse for _ in range(-1, specifications.facets_per_heliostat + 1)]
        plt.plot(x, y, linewidth=1.5, color="red", linestyle="dashed")
        y = [-rmse for _ in range(-1, specifications.facets_per_heliostat + 1)]
        plt.plot(x, y, linewidth=1.5, color="red", linestyle="dashed", label="RMSE")
        plt.ylim(bottom + (10e-2 * bottom), top + 10e-2 * top)
        plt.grid(axis="y")
        plt.legend()
        # plt.show()
        figure_name = hel_name + "_squareDiagonalsErrors_" + option
        if best_scaling:
            figure_name += "_bestScaling"
        figure_name += ".png"
        plt.savefig(saving_path + "/" + figure_name)
        plt.close()

    return square_diagonals_errors


def plot_square_diagonal_offset_quality(
    heliostat,
    specifications,
    heliostat_theoretical_dict=None,
    saving_path=" ",
    hel_name=" ",
    title_prefix=" ",
    option="normalHel",
    plot=True,
):
    figure_name = hel_name + "_diagonalOffset_" + option
    diagonals_distance = []
    diagonals_distance_dict = {}
    for key, corners in sorted(heliostat.items()):
        top_left_corner = corners[TOP_LEFT_CORNER_INDX]
        top_right_corner = corners[TOP_RIGHT_CORNER_INDX]
        bottom_right_corner = corners[BOTTOM_RIGHT_CORNER_INDX]
        bottom_left_corner = corners[BOTTOM_LEFT_CORNER_INDX]
        # diagonals direction vectors
        e1 = np.array(bottom_right_corner) - np.array(top_left_corner)
        e2 = np.array(bottom_left_corner) - np.array(top_right_corner)
        n = np.cross(e1, e2)
        r1r2 = np.array(top_left_corner) - np.array(top_right_corner)
        distance = np.dot(n, r1r2) / np.linalg.norm(n)
        diagonals_distance.append(abs(distance))
        diagonals_distance_dict[key] = abs(distance)

    if plot:
        df = pd.DataFrame(
            {"Diagonal Offset Distance": diagonals_distance},
            index=[i + 1 for i in range(0, specifications.facets_per_heliostat)],
        )
        ax = df.plot.bar(rot=0, color={"Diagonal Offset Distance": "tab:blue"})
        plt.title(title_prefix)
        plt.xlabel("Facet id")
        plt.ylabel("Y (m)")
        plt.savefig(saving_path + "/" + figure_name)
        plt.close()

    return diagonals_distance_dict


# def plot_projection_analysis(heliostat_dict, heliostat_theoretical_dict=None, saving_path=' ', hel_name=' ', title_prefix=' ', option='normalHel', plot=True,
#                              camera_rvec=None, camera_tvec=None, camera_matrix=None, distortion_coefficients=None, invert=True):  # invert=True -> negate y, to match appearance of image.
#     if plot and                          \
#         (camera_rvec is not None) and    \
#         (camera_tvec is not None) and    \
#         (camera_matrix is not None) and  \
#         (distortion_coefficients is not None):
#         # Convert the heliostats from dictionary form to xyz lists.
#         corner_xyz_list = heliostat_xyz_list_given_dict(heliostat_dict)
#         flat_xyz_list   = heliostat_xyz_list_given_dict(heliostat_theoretical_dict)
#         # Construct a copy of the flat heliostat, offsetting the (x,y,z) points in the +z direction.
#         offset_flat_xyz_list = [[xyz[0], xyz[1], xyz[2]+0.5] for xyz in flat_xyz_list]
#         # Project, using input transform.
#         projected_corner_xys, jacobian = cv.projectPoints(np.array(corner_xyz_list), camera_rvec, camera_tvec, camera_matrix, distortion_coefficients)
#         projected_corner_xy_list = [x[0] for x in list(projected_corner_xys)]
#         projected_flat_xys, jacobian = cv.projectPoints(np.array(flat_xyz_list), camera_rvec, camera_tvec, camera_matrix, distortion_coefficients)
#         projected_flat_xy_list = [x[0] for x in list(projected_flat_xys)]
#         projected_offset_flat_xys, jacobian = cv.projectPoints(np.array(offset_flat_xyz_list), camera_rvec, camera_tvec, camera_matrix, distortion_coefficients)
#         projected_offset_flat_xy_list = [x[0] for x in list(projected_offset_flat_xys)]

#         # Invert, if desired.
#         if invert:
#             projected_corner_xy_list      = [[xy[0], -xy[1]] for xy in projected_corner_xy_list]
#             projected_flat_xy_list        = [[xy[0], -xy[1]] for xy in projected_flat_xy_list]
#             projected_offset_flat_xy_list = [[xy[0], -xy[1]] for xy in projected_offset_flat_xy_list]

#         # Construct annotations.
#         annotation_list = []
#         # # Data: Observed corners in image space.
#         # self.add_step_annotations(observed_corner_xy_list, hel_name, False, True, False, 'c', 0.2, 0.6, 5, annotation_list)
#         # Result: Current 3-d corners.
#         add_analysis_annotations(projected_corner_xy_list, hel_name, False, True, False, 'b', 0.2, 0.6, 5, annotation_list)
#         # Context: Flat frame plus needles.
#         add_analysis_annotations(projected_flat_xy_list, hel_name, True, False, True, 'm', 0.1, 0.3, 5, annotation_list)
#         for project_flat_xy, projected_offset_flat_xy in zip(projected_flat_xy_list, projected_offset_flat_xy_list):
#             needle_style = rcps.outline(color='g', linewidth=0.2)
#             annotation_list.append(pa.PlotAnnotation('point_seq', [project_flat_xy, projected_offset_flat_xy], None, needle_style))

#         # Draw figure.
#         plt.figure()
#         plt.title(title_prefix + ' Projection Analysis')
#         for annotation in annotation_list:
#             annotation.plot()

#         # Save.
#         figure_name = hel_name + '_projectionAnalysis_' + option
#         output_body_ext = figure_name + '.png'
#         output_dir_body_ext = os.path.join(saving_path, output_body_ext)
#         # print('In plot_projection_analysis(), writing ' + output_dir_body_ext)
#         plt.savefig(output_dir_body_ext, dpi=1500)
#         plt.close()


def draw_annotated_frame_figure(
    corner_xy_list, input_video_body, input_frame_dir, input_frame_id_format, frame_id, hel_name, explain, output_dir
):
    # Construct annotations.
    annotation_list = []
    # Data: Observed corners in image space.
    add_analysis_annotations(corner_xy_list, hel_name, False, True, False, "c", 0.2, 0.6, 5, annotation_list)
    # Construct figure.
    plt.figure()
    # Fetch and draw image file.
    frame_body_ext = upf.frame_file_body_ext_given_frame_id(input_video_body, frame_id, input_frame_id_format)
    frame_dir_body_ext = os.path.join(input_frame_dir, frame_body_ext)
    # print('In draw_annotated_frame_figure(), reading frame image file: ', frame_dir_body_ext)
    frame_img = cv.imread(frame_dir_body_ext)
    plt.imshow(cv.cvtColor(frame_img, cv.COLOR_BGR2RGB))
    # Prepare crop_box.
    max_row = frame_img.shape[0] - 1
    max_col = frame_img.shape[1] - 1
    crop_box = [[0, 0], [max_col, max_row]]
    # Draw.
    ip.plot_image_figure(
        frame_img,
        draw_image=False,  # Just show annotations.
        rgb=False,
        title=(hel_name + ", " + explain),
        annotation_list=annotation_list,
        crop_box=crop_box,
        context_str="Heliostats3dInference.save_annotated_frame_figure()",
        save=True,
        output_dir=output_dir,
        output_body=(hel_name + "_image_analysis_fig"),  # plot_image_figure() will add ".png"
        dpi=1000,  # 250,
        include_figure_idx_in_filename=False,
    )


def draw_annotated_frame_image(
    corner_xy_list, input_video_body, input_frame_dir, input_frame_id_format, frame_id, hel_name, note, output_dir
):
    # Construct annotations.
    annotation_list = []
    # Data: Observed corners in image space.
    add_analysis_annotations(corner_xy_list, hel_name, True, False, False, "r", 0.2, 0.6, 5, annotation_list)
    add_analysis_annotations(corner_xy_list, hel_name, False, True, False, "c", 0.2, 0.6, 5, annotation_list)
    if (note is not None) and len(note) > 0:
        note_xy = [150, 150]  # Upper left corner of the image.
        annotation_list.append(
            pa.PlotAnnotation(
                "text",
                [note_xy],
                note,
                rctxt.RenderControlText(fontsize=4, color="r", horizontalalignment="left", verticalalignment="top"),
            )
        )
    # Fetch image file.
    frame_body_ext = upf.frame_file_body_ext_given_frame_id(input_video_body, frame_id, input_frame_id_format)
    frame_dir_body_ext = os.path.join(input_frame_dir, frame_body_ext)
    # print('In draw_annotated_frame_image(), reading frame image file: ', frame_dir_body_ext)
    frame_img = cv.imread(frame_dir_body_ext)
    # Add annotations.
    if (annotation_list != None) and (len(annotation_list) > 0):
        for annotation in annotation_list:
            annotation.image_draw(frame_img)  # Automatically crops to image boundary.
    # Save.
    frame_img_body_ext = hel_name + "_image_analysis.png"
    frame_img_dir_body_ext = os.path.join(output_dir, frame_img_body_ext)
    # print('In draw_annotated_frame_image(), saving annotated image file: ', frame_img_dir_body_ext)
    cv.imwrite(frame_img_dir_body_ext, frame_img)


def add_analysis_annotations(
    corner_xy_list,
    text,
    draw_boundaries,
    draw_markers,
    draw_label,
    color,
    linewidth,
    markersize,
    fontsize,
    annotation_list,
):
    if draw_boundaries:
        facet_boundary_list = construct_facet_boundaries(corner_xy_list)
        for facet_boundary in facet_boundary_list:
            boundary_style = rcps.outline(color=color, linewidth=linewidth)
            annotation_list.append(pa.PlotAnnotation("point_seq", facet_boundary, None, boundary_style))
    if draw_markers:
        point_style = rcps.marker(color=color, marker=".", markersize=markersize)
        annotation_list.append(pa.PlotAnnotation("point_seq", corner_xy_list, None, point_style))
    if draw_label:
        label_xy = g2d.label_point(corner_xy_list)
        annotation_list.append(pa.PlotAnnotation("text", [label_xy], text, rctxt.bold(fontsize, color)))


def construct_facet_boundaries(corner_xy_list):
    corner_xy_list_2 = corner_xy_list.copy()
    facet_boundary_list = []
    while len(corner_xy_list_2) > 0:
        xy_1 = corner_xy_list_2.pop(0)
        xy_2 = corner_xy_list_2.pop(0)
        xy_3 = corner_xy_list_2.pop(0)
        xy_4 = corner_xy_list_2.pop(0)
        facet_boundary_list.append([xy_1, xy_2, xy_3, xy_4, xy_1])
    return facet_boundary_list


# ?? SCAFFOLDING RCB -- RE-ORDER THESE ROUTINES, ADD PAGE BREAK COMMENTS


def analyze_and_render_heliostat_3d(
    flat_hel_dir_body_ext,
    hel_dir_body_ext,
    explain,
    specifications,
    output_dir,
    camera_rvec=None,
    camera_tvec=None,
    camera_matrix=None,
    distortion_coefficients=None,
    tracked_frame_camera_pose_dict=None,
    processed_frame_camera_pose_dict=None,
):
    flat_hel_name = heliostat_name_given_heliostat_3d_dir_body_ext(flat_hel_dir_body_ext)
    hel_name = heliostat_name_given_heliostat_3d_dir_body_ext(hel_dir_body_ext)

    # Generate plots.
    flat_heliostat = read_txt_file_to_heliostat(flat_hel_dir_body_ext, specifications)
    msg = "In analyze_and_render_3d_heliostat_model(), calling generate_plots() for heliostat " + hel_name
    if explain is None:
        explain_2 = None
    else:
        explain_2 = explain.title()
        msg += ", " + explain
    msg += "."
    print(msg)
    generate_plots(
        hel_dir_body_ext,
        output_dir,
        specifications,
        flat_heliostat,
        explain=explain_2,
        camera_rvec=camera_rvec,
        camera_tvec=camera_tvec,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        tracked_frame_camera_pose_dict=tracked_frame_camera_pose_dict,
        processed_frame_camera_pose_dict=processed_frame_camera_pose_dict,
    )
    print("In analyze_and_render_3d_heliostat_model(), generate_plots() for heliostat " + hel_name + " finished.")

    # Generate csv files.
    msg.replace("generate_plots()", "generate_csv()")
    print(msg)
    generate_csv(hel_dir_body_ext, output_dir, specifications, flat_heliostat, option="noRotate")
    print("In analyze_and_render_3d_heliostat_model(), generate_csv() for heliostat " + hel_name + " finished.")


def generate_plots(
    filename,
    output_path,
    specifications,
    heliostat_theoretical_dict,
    explain=None,
    camera_rvec=None,
    camera_tvec=None,
    camera_matrix=None,
    distortion_coefficients=None,
    tracked_frame_camera_pose_dict=None,
    processed_frame_camera_pose_dict=None,
):
    heliostat = read_txt_file_to_heliostat(filename=filename, specifications=specifications)
    if heliostat is None:
        return None
    hel_name = heliostat_name_given_heliostat_3d_dir_body_ext(filename)  # ?? SCAFFOLDING RCB -- CRUFTY.  FIX THIS.
    heliostat_path = output_path  # ?? SCAFFOLDING RCB -- CLEAN THIS UP
    ft.create_directories_if_necessary(heliostat_path)

    # heliostat_facet         = translate_rotate_scale(heliostat,     specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, option='normalFacet',     confirm=False)  # ?? SCAFFOLDING RCB -- DISABLE, PROBABLY REMOVE
    # heliostat_hel           = translate_rotate_scale(heliostat,     specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, option='normalHel',       confirm=False)  # ?? SCAFFOLDING RCB -- DISABLE, PROBABLY REMOVE
    # heliostat_optical       = translate_rotate_scale(heliostat,     specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, option='normalOptical',   confirm=False)  # ?? SCAFFOLDING RCB -- DISABLE, PROBABLY REMOVE

    title_prefix = hel_name
    if explain is not None:
        title_prefix += ", " + explain
    """
    Plotting Heliostat 2D
    """
    # plot_heliostat_2d(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=True,  plot_surface_normals=False, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_heliostat_2d(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=True,  plot_surface_normals=False, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_heliostat_2d(heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=True,  plot_surface_normals=False, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    plot_heliostat_2d(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        theoretical_flag=True,
        plot_surface_normals=False,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotate",
    )

    """
    Plotting Heliostat 2D and 2D Surface Normals
    """
    # plot_heliostat_2d(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=False,  plot_surface_normals=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_heliostat_2d(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=False,  plot_surface_normals=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_heliostat_2d(heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=False,  plot_surface_normals=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    plot_heliostat_2d(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        theoretical_flag=False,
        plot_surface_normals=True,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotate",
    )
    plot_heliostat_2d(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        theoretical_flag=False,
        plot_surface_normals=True,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="wrtAvg",
        normal_wrt_average=True,
    )
    """
    Plotting Heliostat in 3D
    """
    # plot_heliostat_3d(heliostat=heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_heliostat_3d(heliostat=heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_heliostat_3d(heliostat=heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    plot_heliostat_3d(
        heliostat=heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        theoretical_flag=True,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotate",
    )
    plot_heliostat_3d(
        heliostat=heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        theoretical_flag=False,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotate",
    )
    # if tracked_frame_camera_pose_dict != None:  # ?? SCAFFOLDING RCB -- TEMPORARY DISABLE
    #     plot_heliostat_3d(heliostat=heliostat, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, theoretical_flag=False, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix + 'Camera locations tracking', option='noRotate', frame_camera_pose_dict=tracked_frame_camera_pose_dict, camera_suffix='track_cameras')  # ?? SCAFFOLDING RCB -- TEMPORARY DISABLE
    if processed_frame_camera_pose_dict != None:
        plot_heliostat_3d(
            heliostat=heliostat,
            specifications=specifications,
            heliostat_theoretical_dict=heliostat_theoretical_dict,
            theoretical_flag=False,
            saving_path=heliostat_path,
            hel_name=hel_name,
            title_prefix=title_prefix + "Processed camera locations for",
            option="noRotate",
            frame_camera_pose_dict=processed_frame_camera_pose_dict,
            camera_suffix="process_cameras",
        )

    """
    Canting Angles
    """
    # plot_canting_angles(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path,     hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_canting_angles(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path,     hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_canting_angles(heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path,     hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    plot_canting_angles(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotate",
    )
    plot_canting_angles(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="wrtAvg",
        normal_wrt_average=True,
    )

    """
    Pose Estimation
    """
    # plot_pose_estimation(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_translation=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_pose_estimation(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_translation=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_pose_estimation(heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_translation=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    # plot_pose_estimation(heliostat, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_translation=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='noRotate')
    plot_pose_estimation(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        best_translation=False,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotateNoTranslate",
    )
    """
    Pose Rotation Estimation
    """
    # plot_pose_rotation_estimation(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_translation=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_pose_rotation_estimation(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_translation=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_pose_rotation_estimation(heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_translation=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    # plot_pose_rotation_estimation(heliostat, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_translation=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='noRotate')
    plot_pose_rotation_estimation(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        best_translation=False,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotateNoTranslate",
    )

    """
    Square Side Quality
    """
    # plot_square_sides_quality(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_scaling=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_square_sides_quality(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_scaling=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_square_sides_quality(heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_scaling=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    # plot_square_sides_quality(heliostat, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_scaling=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='noRotate')
    plot_square_sides_quality(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        best_scaling=False,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotateNoScale",
    )

    """
    Square Diagonals Quality
    """
    # plot_square_diagonals_quality(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_scaling=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_square_diagonals_quality(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_scaling=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_square_diagonals_quality(heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_scaling=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    # plot_square_diagonals_quality(heliostat, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, best_scaling=True, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='noRotate')
    plot_square_diagonals_quality(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        best_scaling=False,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotateNoScale",
    )

    """
    Square Diagonal Offset Quality
    """
    # plot_square_diagonal_offset_quality(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
    # plot_square_diagonal_offset_quality(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
    # plot_square_diagonal_offset_quality(heliostat_optical, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical')
    plot_square_diagonal_offset_quality(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        saving_path=heliostat_path,
        hel_name=hel_name,
        title_prefix=title_prefix,
        option="noRotate",
    )


# """
# Projection Analysis
# """
# # plot_projection_analysis(heliostat_facet, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalFacet')
# # plot_projection_analysis(heliostat_hel, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalHel')
# # plot_projection_analysis(heliostat_optical, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='normalOptical',
# #                          camera_rvec=camera_rvec, camera_tvec=camera_tvec, camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)
# plot_projection_analysis(heliostat, heliostat_theoretical_dict=heliostat_theoretical_dict, saving_path=heliostat_path, hel_name=hel_name, title_prefix=title_prefix, option='noRotate',
#                          camera_rvec=camera_rvec, camera_tvec=camera_tvec, camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)


def generate_csv(hel_dir_body_ext, output_path, specifications, heliostat_theoretical_dict, option):
    def save_csv(dict, output_path=" ", name=" "):
        header = ["Heliostat"]
        for facet_id in range(0, specifications.facets_per_heliostat):
            header.append("Facet " + str(facet_id + 1))

        rows = []
        for hel_name, hel_facets in sorted(dict.items()):
            row = []
            row.append('"' + str(hel_name) + '"')
            for facet_id in range(0, specifications.facets_per_heliostat):
                xyz_value = hel_facets[facet_id + 1]
                row.append(xyz_value)

            rows.append(row)
        output_dir_body_ext = os.path.join(output_path, name)
        with open(output_dir_body_ext, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    hel_name = heliostat_name_given_heliostat_3d_dir_body_ext(
        hel_dir_body_ext
    )  # ?? SCAFFOLDING RCB -- CRUFTY.  FIX THIS.

    # print('Generating and Saving CSV Files ...')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # # options = ['normalFacet', 'normalHel', 'normalOptical']
    # options = ['normalOptical']
    # for option in options:  # ?? SCAFFOLDING RCB -- ELIMINATE THIS?
    canting_angles = {}
    pose_estimations = {}
    pose_rotation_estimations = {}
    square_sides_errors = {}
    square_diagonals_errors = {}
    square_diagonal_offsets = {}
    # ?? SCAFFOLDING RCB -- USED TO BE FOR LOOP HERE OVER HELIOSTAT NAMES, AND CSV FILES WERE COMPILATION ACROSS HELIOSTATS.  RETURN TO THIS?
    # try:
    heliostat = read_txt_file_to_heliostat(filename=hel_dir_body_ext, specifications=specifications)
    hel_name = heliostat_name_given_heliostat_3d_dir_body_ext(hel_dir_body_ext)
    # if heliostat is None:
    #     continue
    # heliostat  = translate_rotate_scale(heliostat, specifications=specifications, heliostat_theoretical_dict=heliostat_theoretical_dict, option=option, confirm=False)  # ?? SCAFFOLDING RCB -- DISABLE, PROBABLY REMOVE
    """
    Canting Angles
    """
    canting_angles[hel_name] = plot_canting_angles(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        hel_name=hel_name,
        plot=False,
        option=option,
    )
    save_csv(canting_angles, output_path=output_path, name=hel_name + "_canglesXYZ_" + option + ".csv")
    wrt_avg_option = "wrtAvg"
    canting_angles[hel_name] = plot_canting_angles(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        hel_name=hel_name,
        plot=False,
        option=wrt_avg_option,
        normal_wrt_average=True,
    )
    save_csv(canting_angles, output_path=output_path, name=hel_name + "_canglesXYZ_" + wrt_avg_option + ".csv")
    """
    Pose Estimation
    """
    pose_estimations[hel_name] = plot_pose_estimation(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        hel_name=hel_name,
        plot=False,
        option=option,
    )
    save_csv(pose_estimations, output_path=output_path, name=hel_name + "_poseXYZ_" + option + ".csv")
    """
    Pose Rotation Estimation
    """
    pose_rotation_estimations[hel_name] = plot_pose_rotation_estimation(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        hel_name=hel_name,
        plot=False,
        option=option,
    )
    save_csv(pose_rotation_estimations, output_path=output_path, name=hel_name + "_poseRotZ_" + option + ".csv")
    """
    Square Sides Quality Error
    """
    square_sides_errors[hel_name] = plot_square_sides_quality(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        hel_name=hel_name,
        plot=False,
        option=option,
    )
    save_csv(square_sides_errors, output_path=output_path, name=hel_name + "_serrorTRBL_" + option + ".csv")
    """
    Square Diagonals Quality Error
    """
    square_diagonals_errors[hel_name] = plot_square_diagonals_quality(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        hel_name=hel_name,
        plot=False,
        option=option,
    )
    save_csv(square_diagonals_errors, output_path=output_path, name=hel_name + "_diagerrors_" + option + ".csv")
    """
    Square Diagonal Offsets Quality
    """
    square_diagonal_offsets[hel_name] = plot_square_diagonal_offset_quality(
        heliostat,
        specifications=specifications,
        heliostat_theoretical_dict=heliostat_theoretical_dict,
        hel_name=hel_name,
        plot=False,
        option=option,
    )
    save_csv(square_diagonal_offsets, output_path=output_path, name=hel_name + "_diagoffsets_" + option + ".csv")


# except:
#     pass
# else:
#     pass

# ?? SCAFFOLDING RCB -- ORIGINAL CODE, WHICH WROTE CSV THAT FILES WERE COMPILATION ACROSS HELIOSTATS.  RETURN TO THIS?
# save_csv(canting_angles, output_path=output_path, name=hel_name+'_canglesXYZ_' + option + '.csv')
# save_csv(pose_estimations, output_path=output_path, name=hel_name+'_poseXYZ_' + option + '.csv')
# save_csv(pose_rotation_estimations, output_path=output_path, name=hel_name+'_poseRotZ_' + option + '.csv')
# save_csv(square_sides_errors, output_path=output_path, name=hel_name+'_serrorTRBL_' + option + '.csv')
# save_csv(square_diagonals_errors, output_path=output_path, name=hel_name+'_diagerrors_' + option + '.csv')
# save_csv(square_diagonal_offsets, output_path=output_path, name=hel_name+'_diagoffsets_' + option + '.csv')


def heliostat_name_given_heliostat_3d_dir_body_ext(
    heliostat_3d_dir_body_ext,
):  # ?? SCAFFOLDING RCB -- MAKE THIS GENERAL, CORRECT, ERROR-CHECKING.  (IT'S LATE, AND I'M OUT OF TIME.)
    heliostat_3d_dir, heliostat_3d_body, heliostat_3d_ext = ft.path_components(heliostat_3d_dir_body_ext)
    if heliostat_3d_body.find("distorted") == -1:
        # Case 1:  No projected/distorted substrings.
        tokens = heliostat_3d_body.split("_")
        three_d_str = tokens[-1]
        corners_str = tokens[-2]
        name_str = tokens[-3]
        return name_str
    else:
        # Case 2:  Includes projected/distorted substrings.
        tokens = heliostat_3d_body.split("_")
        three_d_str = tokens[-1]
        corners_str = tokens[-2]
        distorted_str = tokens[-3]
        confirmed_str = tokens[-4]
        name_str = tokens[-5]
        return name_str


def corners_3d_dir_body_ext(
    input_video_body, hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, corners_3d_dir
):
    # Assemble filename body.
    c_3d_body = hel_name
    if (input_video_body is not None) and (len(input_video_body) > 0):
        c_3d_body = input_video_body + "_" + c_3d_body
    if (projected_or_confirmed_str is not None) and (len(projected_or_confirmed_str) > 0):
        c_3d_body += "_" + projected_or_confirmed_str
    if (distorted_or_undistorted_str is not None) and (len(distorted_or_undistorted_str) > 0):
        c_3d_body += "_" + distorted_or_undistorted_str
    c_3d_body += "_" + "corners_3d"
    # Add extension and directory.
    c_3d_body_ext = c_3d_body + ".csv"
    c_3d_dir_body_ext = os.path.join(corners_3d_dir, c_3d_body_ext)
    # Return.
    return c_3d_dir_body_ext


# ?? SCAFFOLDING RCB -- RE-ORDER THESE ROUTINES, ADD PAGE BREAK COMMENTS


def heliostat_xyz_list_given_dict(heliostat_dict):
    xyz_list = []
    for key in dt.sorted_keys(heliostat_dict):
        facet_xyz_list = heliostat_dict[key]
        for xyz in facet_xyz_list:
            xyz_list.append(xyz)
    return xyz_list


def save_heliostat_3d(
    hel_name,
    corner_xyz_list,
    output_dir,
    input_video_body=None,
    projected_or_confirmed_str=None,
    distorted_or_undistorted_str=None,
):
    # Write the 3-d corner file.
    ft.create_directories_if_necessary(output_dir)
    output_heliostat_3d_dir_body_ext = corners_3d_dir_body_ext(
        input_video_body, hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, output_dir
    )
    print("In Heliostats3dInference.save_heliostat_3d(), writing file: ", output_heliostat_3d_dir_body_ext)
    with open(output_heliostat_3d_dir_body_ext, "w") as output_stream:
        wr = csv.writer(output_stream)
        wr.writerows(corner_xyz_list)
    return output_heliostat_3d_dir_body_ext


def plot_and_save_plane_views(path_figure_base, ax):
    # (Az, el) describe a camera position, viewing back toward origin.
    # ax.view_init(-90, -90)  # x-negy plane view
    # path_figure = path_figure_base + '_-90_-90' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)
    # ax.view_init(-90, 0)  # y-x plane view
    # path_figure = path_figure_base + '_-90_0' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)
    # ax.view_init(-90, 90)  # negx-y plane view
    # path_figure = path_figure_base + '_-90_90' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)

    # ax.view_init(0, -90)  # x-z plane view
    # path_figure = path_figure_base + '_0_-90' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)
    # ax.view_init(0, 0)  # y-z plane view
    # path_figure = path_figure_base + '_0_0' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)
    # ax.view_init(0, 90)  # negx-z plane view
    # path_figure = path_figure_base + '_0_90' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)

    # ax.view_init(90, -90)  # x-y plane view
    # path_figure = path_figure_base + '_90_-90' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)
    # ax.view_init(90, 0)  # y-negx plane view
    # path_figure = path_figure_base + '_90_0' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)
    # ax.view_init(90, 90)  # negx-negy plane view
    # path_figure = path_figure_base + '_90_90' + '.png'
    # print('In plot_heliostat_3d(), saving figure:', path_figure) # ?? SCAFFOLDING RCB -- TEMPORARY ?
    # plt.savefig(path_figure, dpi=1200)

    # XY
    ax.view_init(90, -90)  # x-y plane view
    path_figure = path_figure_base + "_xy" + ".png"
    print("In plot_heliostat_3d(), saving figure:", path_figure)  # ?? SCAFFOLDING RCB -- TEMPORARY ?
    plt.savefig(path_figure, dpi=1200)

    # XZ
    ax.view_init(0, -90)  # x-z plane view
    path_figure = path_figure_base + "_xz" + ".png"
    print("In plot_heliostat_3d(), saving figure:", path_figure)  # ?? SCAFFOLDING RCB -- TEMPORARY ?
    plt.savefig(path_figure, dpi=1200)

    # YZ
    ax.view_init(0, 0)  # y-z plane view
    path_figure = path_figure_base + "_yz" + ".png"
    print("In plot_heliostat_3d(), saving figure:", path_figure)  # ?? SCAFFOLDING RCB -- TEMPORARY ?
    plt.savefig(path_figure, dpi=1200)


def plot_heliostat_with_camera_poses(
    hel_name,
    design_hel_dir_body_ext,  # ?? SCAFFOLDING RCB -- REORDER PARAMETERS, AND ALSO CALLER PARAMETER ORDER.
    specifications,
    dict_of_frame_dicts,
    saving_path=" ",
    title_prefix=" ",
    explain=None,
    # frame_dict=None,
    # tracked_frame_camera_pose_dict=None,
    tracked_style=None,
    # processed_frame_camera_pose_dict=None,
    processed_style=None,
):
    # Load heliostat.
    # hel_name = heliostat_name_given_heliostat_3d_dir_body_ext(design_hel_dir_body_ext)
    heliostat = read_txt_file_to_heliostat(design_hel_dir_body_ext, specifications)

    # Announce.
    msg = "In plot_heliostat_with_camera_poses(), drawing heliostat " + hel_name
    if explain is None:
        explain_2 = None
    else:
        explain_2 = explain.title()
        msg += ", " + explain
    msg += "."
    print(msg)

    min_z = 10e2
    max_z = -1
    for _, corners in sorted(heliostat.items()):
        for corner in corners:
            z = corner[2]
            if z < min_z:
                min_z = z
            elif z > max_z:
                max_z = z

    figure_name = hel_name + "_camera_positions"

    fig = plt.figure()
    #    ax = plt.axes(projection='3d')  # Perspective projection
    ax = plt.axes(projection="3d", proj_type="ortho")  # Orthographic projection
    plt.title(title_prefix)

    # draw top corners
    corners = [
        heliostat[TOP_LEFT_FACET_INDX][TOP_LEFT_CORNER_INDX],
        heliostat[TOP_RIGHT_FACET_INDX][TOP_RIGHT_CORNER_INDX],
        heliostat[BOTTOM_RIGHT_FACET_INDX][BOTTOM_RIGHT_CORNER_INDX],
        heliostat[BOTTOM_LEFT_FACET_INDX][BOTTOM_LEFT_CORNER_INDX],
    ]
    xdata = [corner[0] for corner in corners]
    ydata = [corner[1] for corner in corners]
    zdata = [corner[2] for corner in corners]
    ax.scatter3D(xdata[0], ydata[0], zdata[0], facecolor="tab:blue")
    ax.scatter3D(xdata[1], ydata[1], zdata[1], facecolor="tab:orange")
    ax.scatter3D(xdata[2], ydata[2], zdata[2], facecolor="tab:green")
    ax.scatter3D(xdata[3], ydata[3], zdata[3], facecolor="c")
    # draw center
    center = facet_center(
        heliostat[specifications.centered_facet][TOP_LEFT_CORNER_INDX],
        heliostat[specifications.centered_facet][BOTTOM_RIGHT_CORNER_INDX],
        heliostat[specifications.centered_facet][TOP_RIGHT_CORNER_INDX],
        heliostat[specifications.centered_facet][BOTTOM_LEFT_CORNER_INDX],
    )
    ax.scatter3D(center[0], center[1], center[2], facecolor="tab:green")
    # draw heliostat
    for _, corners in sorted(heliostat.items()):
        temp_corners = corners.copy()
        temp_corners.append(temp_corners[0])  # cyclic
        xline, yline, zline = [], [], []
        # draw facets
        for indx in range(0, len(temp_corners) - 1):
            xline.append(temp_corners[indx][0])
            xline.append(temp_corners[indx + 1][0])
            yline.append(temp_corners[indx][1])
            yline.append(temp_corners[indx + 1][1])
            zline.append(temp_corners[indx][2])
            zline.append(temp_corners[indx + 1][2])
            ax.plot3D(xline, yline, zline, "tab:blue")

    # draw cameras
    if dict_of_frame_dicts != None:
        plot_heliostat_with_camera_poses_aux(ax, dict_of_frame_dicts, False, style=tracked_style)
        plot_heliostat_with_camera_poses_aux(ax, dict_of_frame_dicts, True, style=processed_style)
    # plt.legend()

    ax3d.set_3d_axes_equal(ax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    # plt.show()

    ft.create_directories_if_necessary(saving_path)
    path_figure_base = os.path.join(saving_path, figure_name)
    path_figure = path_figure_base + ".png"
    print("In plot_heliostat_3d(), saving figure:", path_figure)  # ?? SCAFFOLDING RCB -- TEMPORARY ?
    plt.savefig(path_figure, dpi=1200)
    plot_and_save_plane_views(path_figure_base, ax)

    plt.close()


def plot_heliostat_with_camera_poses_aux(ax, dict_of_frame_dicts, only_use_for_metrology, style, label=None):
    # Collect individual coordinate lists.
    x_list = []
    y_list = []
    z_list = []
    for frame_id in dt.sorted_keys(dict_of_frame_dicts):
        frame_dict = dict_of_frame_dicts[frame_id]
        # Determine whether to skip this point.
        if only_use_for_metrology and (frame_dict["use_for_metrology"] == False):
            continue
        # Plot this point.
        tvec = frame_dict["single_frame_camera_tvec"]
        t_x = tvec[0][0]
        t_y = tvec[1][0]
        t_z = tvec[2][0]
        # print('In plot_heliostat_with_camera_poses_aux(), t_x, t_y, t_z = ', t_x, t_y, t_z)
        x_list.append(
            -t_x
        )  # 2021-11-14 4:00 AM.  Discovered that tvec_x coordinate is reversed compared to heliostat x axis.
        y_list.append(t_y)
        z_list.append(t_z)
    # Plot.
    ax.plot3D(
        x_list,
        y_list,
        z_list,
        linestyle=style.linestyle,
        linewidth=style.linewidth,
        color=style.color,
        marker=style.marker,
        markersize=style.markersize,
        markeredgecolor=style.markeredgecolor,
        markeredgewidth=style.markeredgewidth,
        markerfacecolor=style.markerfacecolor,
        label=label,
    )
