import os
import time
from functools import reduce
from multiprocessing.pool import Pool
from typing import Iterable
from warnings import warn

import numpy as np
import psutil
from scipy.spatial.transform import Rotation

from opencsp_code.common.lib.csp import LightPath as lp
from opencsp_code.common.lib.csp.LightPath import LightPath
from opencsp_code.common.lib.csp.LightPathEnsemble import LightPathEnsemble
from opencsp_code.common.lib.csp.LightSource import LightSource
from opencsp_code.common.lib.csp.RayTrace import RayTrace
from opencsp_code.common.lib.csp.RayTraceable import RayTraceable
from opencsp_code.common.lib.csp.Scene import Scene
from opencsp_code.common.lib.geometry.FunctionXYGrid import FunctionXYGrid
from opencsp_code.common.lib.geometry.Pxy import Pxy
from opencsp_code.common.lib.geometry.Pxyz import Pxyz
from opencsp_code.common.lib.geometry.Uxyz import Uxyz
from opencsp_code.common.lib.geometry.Vxyz import Vxyz
from opencsp_code.common.lib.render.View3d import View3d
from opencsp_code.common.lib.render_control.RenderControlPointSeq import \
    RenderControlPointSeq
from opencsp_code.common.lib.render_control.RenderControlRayTrace import \
    RenderControlRayTrace



# TODO tjlarki: for maddie, make this better

def plane_intersec_vec_maddie(
                        lines: tuple[Pxyz, Vxyz],
                        plane: tuple[Pxyz, Uxyz],  # used to be --> plane_point: Pxyz, plane_normal_vector: Uxyz,
                        epsilon: float = 1e-6,
                        verbose: bool = False,
                        ):
    """Vectorized plane intersection algorithm
    plane = (plane_point, plane_normal_vector)
    
    Computes intersection points of a line and plane. Disregards direction of line. 
    """

    # Unpack plane
    plane_point, plane_normal_vector = plane

    # Unpack lines
    points, directions = lines

    # finds where the light intersects the plane
    # algorithm explained at \opencsp_code\doc\IntersectionWithPlaneAlgorithm.pdf
    # TODO tjlarki: upload explicitly vectorized algorithm proof

    plane_normal_vector = plane_normal_vector.normalize()
    plane_vectorV = plane_normal_vector.data  # column vector
    plane_pointV = plane_point.data           # column vector

    # most recent points in light path ensemble
    if verbose:
        print("setting up values...")
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
    if verbose:
        print("filtering out missed vectors")
    filtered_intersec_points = intersection_points # Pxyz.merge(list(filter(lambda vec: not vec.hasnan(),intersection_points)))

    if verbose:
        print("Rotating.")
    # TODO: Do we want the inverse that takes that vector back into the up vector
    # up_vec = Vxyz([0, 0, 1])
    # rot = Vxyz.align_to(plane_normal_vector, up_vec)  # .inv()
    # rotated_intersec_points: Pxyz = filtered_intersec_points.rotate(rot)

    if verbose:
        print("Plane intersections calculated.")

    return filtered_intersec_points
    # return np.histogram2d(xyz[:,0], xyz[:,1], bins)
    # TODO tjlarki: create the histogram from this or bin these results

# if __name__ == "__main__":
#     points = Pxyz([[-1, 0, 1],
#          [0, 0, 0],
#          [1, 1, 1]])
#     directions = Vxyz([[1, 0, 0],
#          [0, 1, 0],
#          [-1, -1, -1]])
    
#     plane = (Pxyz([0,0,0]), Vxyz([0,0,1]))

#     insec = plane_intersec_vec_maddie((points, directions), plane, verbose= True)
#     print(insec)


