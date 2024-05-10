import os
import time
from functools import reduce
from multiprocessing.pool import Pool
from typing import Iterable
from warnings import warn

import numpy as np
from scipy.spatial.transform import Rotation

import opencsp.common.lib.tool.log_tools as lt

from opencsp.common.lib.csp import LightPath as lp
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.LightPathEnsemble import LightPathEnsemble
from opencsp.common.lib.csp.LightSource import LightSource
from opencsp.common.lib.csp.RayTrace import RayTrace
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.geometry.FunctionXYGrid import FunctionXYGrid
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlPointSeq import \
    RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlRayTrace import \
    RenderControlRayTrace



def plane_lines_intersection(
                        lines: tuple[Pxyz, Vxyz],
                        plane: tuple[Pxyz, Uxyz],  # used to be --> plane_point: Pxyz, plane_normal_vector: Uxyz,
                        epsilon: float = 1e-6,
                        verbose: bool = False,
                        ) -> Pxyz:
    """Vectorized plane intersection algorithm
    plane = (plane_point, plane_normal_vector)
        line intersection algorithm
    lines = (points, directions)

    Returns
    -------
    intersection_points: Pxyz
        The intersection (x,y,z) locations for each of the lines with the plane, one per line. Shape (3,N), where N is the number of lines.
        Disregards direction of line. 
    """

    # Unpack plane
    plane_point, plane_normal_vector = plane

    # Unpack lines
    points, directions = lines

    # normalize inputs
    lines_validate = np.squeeze(points.data), np.squeeze(directions.data)
    plane_validate = np.squeeze(plane_point.data), np.squeeze(plane_normal_vector.data)

    # validate inputs
    if np.ndim(plane_validate[0].data) != 1:
        lt.error_and_raise(ValueError, f"Error in plane_lines_intersection(): the 'plane' parameter should contain a single origin point, but instead contains {plane_validate[0].shape[1]} points")
    if np.ndim(plane_validate[1].data) != 1:
        lt.error_and_raise(ValueError, f"Error in plane_lines_intersection(): the 'plane' parameter should contain a single normal vector, but instead contains {plane_validate[1].shape[1]} points")
    for i in range(directions.data.shape[1]):
        if Vxyz.dot(plane_normal_vector, directions[i]) == 0:
            lt.error_and_raise(ValueError, f"Error in plane_lines_intersection(): the 'plane' parameter and 'line(s)' parameter(s) are parallel.")

    # finds where the light intersects the plane
    # algorithm explained at --- (??location for new pdf - email to ben and randy)
    # TODO tjlarki: upload explicitly vectorized algorithm proof 

    plane_normal_vector = plane_normal_vector.normalize()
    plane_vectorV = plane_normal_vector.data  # column vector
    plane_pointV = plane_point.data           # column vector

    # most recent points in light path ensemble
    lt.debug("setting up values...")
    P = points.data
    V = directions.data      # current vectors

    if verbose:
        print("finding intersections...")

    ########## Intersection Algorithm ###########
    # .op means to do the 'op' element wise
    d = np.matmul(plane_vectorV.T, V)           # (1 x N) <- (1 x 3)(3 x N)
    W = P - plane_pointV                        # (3 x N) <- (3 x N) -[broadcast] (3 x 1)
    f = -np.matmul(plane_vectorV.T, W) / d      # (1 x N) <- (1 x 3)(3 x N) ./ (1 x N)
    F = f * V                                   # (3 x N) <- (1 x N) .* (3 x N)
    intersection_matrix = P + F                 # (3 x N) <- (3 x N) .- (3 x N)
    #############################################
    intersection_points = Pxyz(intersection_matrix)

    # filter out points that miss the plane
    # if verbose:
    #     print("filtering out missed vectors")
    # filtered_intersec_points = intersection_points # Pxyz.merge(list(filter(lambda vec: not vec.hasnan(),intersection_points)))

    # if verbose:
    #     print("Rotating.")
    # # TODO: Do we want the inverse that takes that vector back into the up vector
    # # up_vec = Vxyz([0, 0, 1])
    # # rot = Vxyz.align_to(plane_normal_vector, up_vec)  # .inv()
    # # rotated_intersec_points: Pxyz = filtered_intersec_points.rotate(rot)

    # if verbose:
    #     print("Plane intersections calculated.")

    return intersection_points
    # return np.histogram2d(xyz[:,0], xyz[:,1], bins)
    # TODO tjlarki: create the histogram from this or bin these results

# if __name__ == "__main__":
    