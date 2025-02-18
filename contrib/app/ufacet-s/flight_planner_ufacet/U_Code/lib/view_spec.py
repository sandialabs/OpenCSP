import numpy as np

import opencsp.common.lib.geometry.geometry_2d as g2d


def view_spec_vplane(segment_xy) -> dict:  # A vertical plane containing the segment.
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
    spec["type"] = "vplane"
    spec["defining_segment_xy"] = segment_xy
    spec["line_intersecting_xy_plane"] = line
    spec["section_plane"] = plane
    spec["origin_xyz"] = origin_xyz
    spec["p_uxyz"] = p_uxyz
    spec["q_uxyz"] = q_uxyz
    spec["w_uxyz"] = w_uxyz
    # Return.
    return spec
