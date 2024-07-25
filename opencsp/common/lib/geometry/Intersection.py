import os
import time
from functools import reduce
from multiprocessing.pool import Pool
from typing import Iterable
from warnings import warn

import numpy as np
import psutil
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

# from opencsp.common.lib.geometry.Plane import Plane
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace
from opencsp.common.lib.render_control.RenderControlIntersection import RenderControlIntersection
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets, save_hdf5_datasets
from opencsp.common.lib.tool.typing_tools import strict_types


class Intersection:
    def __init__(self, intersection_points: Pxyz):
        self.intersection_points = intersection_points

    @classmethod
    def plane_intersect_from_ray_trace(
        cls,
        ray_trace: RayTrace,
        plane: tuple[Pxyz, Uxyz],  # used to be --> plane_point: Pxyz, plane_normal_vector: Uxyz,
        epsilon: float = 1e-6,
        save_in_file: bool = False,
        save_name: str = None,
        max_ram_in_use_percent: float = 95.0,
        verbose: bool = False,
    ):
        """Vectorized plane intersection algorithm
        place = (plane_point, plane_normal_vector)"""

        # Unpack plane
        plane_point, plane_normal_vector = plane

        intersecting_points = []
        lpe = ray_trace.light_paths_ensemble
        batch = 0

        # ################# TODO TJL:draft for saving traces ######################
        # # TODO: add deletion if save_in_ram = False
        # if save_in_file:
        #     datasets = [
        #         f"Intersection/Information/PlaneLocation",
        #         f"Intersection/Information/PlaneNormalVector",
        #     ]
        #     data = [plane_point.data, plane_normal_vector.data]
        #     print("saving general information...")
        #     save_hdf5_datasets(data, datasets, save_name)
        # ##############################################################################

        # finds where the light intersects the plane
        # algorithm explained at \opencsp\doc\IntersectionWithPlaneAlgorithm.pdf
        # TODO TJL:upload explicitly vectorized algorithm proof

        plane_normal_vector = plane_normal_vector.normalize()
        plane_vectorV = plane_normal_vector.data  # column vector
        plane_pointV = plane_point.data  # column vector

        # most recent points in light path ensemble
        if verbose:
            print("setting up values...")
        P = Pxyz.merge(list(map(lambda xs: xs[-1], lpe.points_lists))).data
        V = lpe.current_directions.data  # current vectors

        if verbose:
            print("finding intersections...")

        ########## Intersection Algorithm ###########
        # .op means to do the 'op' element wise
        d = np.matmul(plane_vectorV.T, V)  # (1 x N) <- (1 x 3)(3 x N)
        W = P - plane_pointV  # (3 x N) <- (3 x N) -[broadcast] (3 x 1)
        f = -np.matmul(plane_vectorV.T, W) / d  # (1 x N) <- (1 x 3)(3 x N) ./ (1 x N)
        F = f * V  # (3 x N) <- (1 x N) .* (3 x N)
        intersection_matrix = P + F  # (3 x N) <- (3 x N) .- (3 x N)
        #############################################
        intersection_points = Pxyz(intersection_matrix)

        # filter out points that miss the plane
        if verbose:
            print("filtering out missed vectors")
        filtered_intersec_points = Pxyz.merge(list(filter(lambda vec: not vec.hasnan(), intersection_points)))

        # if verbose:
        #     print("Rotating.")
        # TODO: Do we want the inverse that takes that vector back into the up vector
        # up_vec = Vxyz([0, 0, 1])
        # rot = Vxyz.align_to(plane_normal_vector, up_vec)  # .inv()
        # rotated_intersec_points: Pxyz = filtered_intersec_points.rotate(rot)

        if verbose:
            print("Plane intersections caluculated.")

        ################# TODO TJL:draft for saving traces ######################
        if save_in_file:
            datasets = [f"Intersection/Batches/Batch{batch:08}"]
            if verbose:
                print(type(filtered_intersec_points.data))
            data = [filtered_intersec_points.data]
            if verbose:
                print(f"saving to {save_name}...")
            save_hdf5_datasets(data, datasets, save_name)
        ##############################################################################

        return Intersection(filtered_intersec_points)
        # return np.histogram2d(xyz[:,0], xyz[:,1], bins)
        # TODO TJL:create the histogram from this or bin these results

    plane_intersec_vec = plane_intersect_from_ray_trace

    # TODO TJL:for maddie, make this better

    @classmethod
    def plane_lines_intersection(
        cls,
        lines: tuple[Pxyz, Vxyz],
        plane: tuple[Pxyz, Uxyz],  # used to be --> plane_point: Pxyz, plane_normal_vector: Uxyz,
        epsilon: float = 1e-6,
        verbose: bool = False,
    ) -> Pxyz:
        """Vectorized plane intersection algorithm

        Parameters
        ----------
        lines : tuple[Pxyz, Vxyz]
            The lines to find intersections for with the given plane. The lines
            consist of two parts: a point (Pxyz) and a direction (Vxyz). Each
            index in the points should correspond to the same index in the
            directions. Note that only one Pxyz and Vxyz is needed to represent
            multiple lines.
        plane : tuple[Pxyz, Uxyz]
            The plane to find the intersections of. The plane definition
            consists of a point Pxyz that the plane goes through, and the normal
            vector to the plane Uxyz.

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
            lt.error_and_raise(
                ValueError,
                f"Error in plane_lines_intersection(): the 'plane' parameter should contain a single origin point, but instead contains {plane_validate[0].shape[1]} points",
            )
        if np.ndim(plane_validate[1].data) != 1:
            lt.error_and_raise(
                ValueError,
                f"Error in plane_lines_intersection(): the 'plane' parameter should contain a single normal vector, but instead contains {plane_validate[1].shape[1]} points",
            )
        for i in range(directions.data.shape[1]):
            if Vxyz.dot(plane_normal_vector, directions[i]) == 0:
                lt.error_and_raise(
                    ValueError,
                    f"Error in plane_lines_intersection(): the 'plane' parameter and 'line(s)' parameter(s) are parallel.",
                )

        plane_normal_vector = plane_normal_vector.normalize()
        plane_vectorV = plane_normal_vector.data  # column vector
        plane_pointV = plane_point.data  # column vector

        # most recent points in light path ensemble
        lt.debug("setting up values...")
        P = points.data
        V = directions.data  # current vectors

        lt.debug("finding intersections...")

        ########## Intersection Algorithm ###########
        # .+ means add element wise
        d = np.matmul(plane_vectorV.T, V)  # (1 x N) <- (1 x 3)(3 x N)
        W = P - plane_pointV  # (3 x N) <- (3 x N) -[broadcast] (3 x 1)
        f = -np.matmul(plane_vectorV.T, W) / d  # (1 x N) <- (1 x 3)(3 x N) ./ (1 x N)
        F = f * V  # (3 x N) <- (1 x N) .* (3 x N)
        intersection_matrix = P + F  # (3 x N) <- (3 x N) .- (3 x N)
        #############################################
        intersection_points = Pxyz(intersection_matrix)

        return intersection_points
        # return np.histogram2d(xyz[:,0], xyz[:,1], bins)
        # TODO TJL:create the histogram from this or bin these results

    @classmethod
    def from_hdf(cls, filename: str, intersection_name: str = "000"):
        # get the names of the batches to loop through
        intersection_points = Pxyz(
            list(load_hdf5_datasets([f"Intersection_{intersection_name}/Points"], filename).values())[0]
        )
        return Intersection(intersection_points)

    @classmethod
    def empty_intersection(cls):
        return cls(Pxyz.empty())

    def __add__(self, intersection: 'Intersection'):
        return Intersection(self.intersection_points.concatenate(intersection.intersection_points))

    def __len__(self):
        return len(self.intersection_points)

    def save_to_hdf(self, hdf_filename: str, intersection_name: str = "000"):
        datasets = [f"Intersection_{intersection_name}/Points", f"Intersection_{intersection_name}/Metatdata"]
        data = [self.intersection_points.data, "Placeholder"]
        save_hdf5_datasets(data, datasets, hdf_filename)

    def get_centroid(self) -> Pxyz:
        N = len(self)
        x = sum(self.intersection_points.x) / N
        y = sum(self.intersection_points.y) / N
        z = sum(self.intersection_points.z) / N
        return Pxyz([x, y, z])

    # flux maps

    def to_flux_mapXY(self, bins: int, resolution_type: str = None) -> FunctionXYGrid:
        pxy = Pxy([self.intersection_points.x, self.intersection_points.y])
        return Intersection._Pxy_to_flux_map(pxy, bins, resolution_type)

    def to_flux_mapXZ(self, bins: int, resolution_type: str = None) -> FunctionXYGrid:
        pxz = Pxy([self.intersection_points.x, self.intersection_points.z])
        return Intersection._Pxy_to_flux_map(pxz, bins, resolution_type)

    def to_flux_mapYZ(self, bins: int, resolution_type: str = None) -> FunctionXYGrid:
        pyz = Pxy([self.intersection_points.y, self.intersection_points.z])
        return Intersection._Pxy_to_flux_map(pyz, bins, resolution_type)

    def _Pxy_to_flux_map(points: Pxy, bins: int, resolution_type: str = "pixelX") -> FunctionXYGrid:

        xbins = bins
        x_low, x_high = min(points.x), max(points.x)
        y_low, y_high = min(points.y), max(points.y)

        x_range = x_high - x_low
        step = x_range / xbins
        y_range = y_high - y_low
        ybins = int(np.round(y_range / step))

        h, xedges, yedges = np.histogram2d(points.x, points.y, [xbins, ybins])

        # first and last midpoints
        x_mid_low = (xedges[0] + xedges[1]) / 2
        x_mid_high = (xedges[-2] + xedges[-1]) / 2
        y_mid_low = (yedges[0] + yedges[1]) / 2
        y_mid_high = (yedges[-2] + yedges[-1]) / 2

        return FunctionXYGrid(h, (x_mid_low, x_mid_high, y_mid_low, y_mid_high))

    # drawing

    def draw(self, view: View3d, style: RenderControlPointSeq = None):
        if style is None:
            style = RenderControlPointSeq()
        view.draw_single_Pxyz(self.intersection_points, style)

    def draw_subset(self, view: View3d, count: int, points_style: RenderControlPointSeq = None):
        for i in np.floor(np.linspace(0, len(self.intersection_points) - 1, count)):
            view.draw_single_Pxyz(self.intersection_points[int(i)])
