import numpy as np

import opencsp.common.lib.tool.log_tools as lt

from opencsp.common.lib.csp.RayTrace import RayTrace
from opencsp.common.lib.geometry.FunctionXYGrid import FunctionXYGrid

from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets, save_hdf5_datasets


class Intersection:
    """
    A class representing the intersection points of light paths with a plane.

    This class provides methods for calculating intersections of light paths with planes,
    as well as utilities for managing and analyzing these intersection points.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(self, intersection_points: Pxyz):
        """Initializes the Intersection instance with the given intersection points.

        Parameters
        ----------
        intersection_points : Pxyz
            The intersection points to be stored in this instance.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.intersection_points = intersection_points

    @classmethod
    def plane_intersect_from_ray_trace(
        cls,
        ray_trace: RayTrace,
        plane: tuple[Pxyz, Uxyz],  # used to be --> plane_point: Pxyz, plane_normal_vector: Uxyz,
        save_in_file: bool = False,
        save_name: str = None,
        verbose: bool = False,
    ):
        """
        Calculates intersection points of light paths with a specified plane.

        This method uses a vectorized algorithm to find where light paths intersect with a plane
        defined by a point and a normal vector.

        Parameters
        ----------
        ray_trace : RayTrace
            The RayTrace object containing the light paths to be analyzed.
        plane : tuple[Pxyz, Uxyz]
            A tuple containing a point on the plane (Pxyz) and the normal vector to the plane (Uxyz).
        save_in_file : bool, optional
            If True, saves the intersection data to a file. Defaults to False.
        save_name : str, optional
            The name of the file to save the intersection data. Required if save_in_file is True.
        verbose : bool, optional
            If True, enables verbose output for debugging purposes. Defaults to False.

        Returns
        -------
        Intersection
            An Intersection object containing the calculated intersection points.

        Raises
        ------
        ValueError
            If the intersection calculation fails or if the input parameters are invalid.
        """
        # "ChatGPT 4o" assisted with generating this docstring.

        # Unpack plane
        plane_point, plane_normal_vector = plane

        lpe = ray_trace.light_paths_ensemble
        batch = 0

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

        # Intersection Algorithm
        d = np.matmul(plane_vectorV.T, V)  # (1 x N) <- (1 x 3)(3 x N)
        W = P - plane_pointV  # (3 x N) <- (3 x N) -[broadcast] (3 x 1)
        f = -np.matmul(plane_vectorV.T, W) / d  # (1 x N) <- (1 x 3)(3 x N) ./ (1 x N)
        F = f * V  # (3 x N) <- (1 x N) .* (3 x N)
        intersection_matrix = P + F  # (3 x N) <- (3 x N) .- (3 x N)

        intersection_points = Pxyz(intersection_matrix)

        # filter out points that miss the plane
        if verbose:
            print("filtering out missed vectors")
        filtered_intersec_points = Pxyz.merge(list(filter(lambda vec: not vec.hasnan(), intersection_points)))
        if verbose:
            print("Plane intersections caluculated.")

        if save_in_file:
            datasets = [f"Intersection/Batches/Batch{batch:08}"]
            if verbose:
                print(type(filtered_intersec_points.data))
            data = [filtered_intersec_points.data]
            if verbose:
                print(f"saving to {save_name}...")
            save_hdf5_datasets(data, datasets, save_name)

        return Intersection(filtered_intersec_points)

    plane_intersec_vec = plane_intersect_from_ray_trace

    @classmethod
    def plane_lines_intersection(
        cls,
        lines: tuple[Pxyz, Vxyz],
        plane: tuple[Pxyz, Uxyz],  # used to be --> plane_point: Pxyz, plane_normal_vector: Uxyz,
    ) -> Pxyz:
        """
        Calculates intersection points of multiple lines with a specified plane.

        This method determines where each line intersects with the plane defined by a point and a normal vector.

        Parameters
        ----------
        lines : tuple[Pxyz, Vxyz]
            A tuple containing a point (Pxyz) and a direction vector (Vxyz) for the lines. Each
            index in the points should correspond to the same index in the directions. Note that
            only one Pxyz and Vxyz is needed to represent multiple lines.
        plane : tuple[Pxyz, Uxyz]
            A tuple containing a point on the plane (Pxyz) and the normal vector to the plane (Uxyz).
            Each index in the points should correspond to the same index in the directions. Note that
            only one Pxyz and Vxyz is needed to represent multiple lines.

        Returns
        -------
        Pxyz
            The intersection points (x, y, z) for each line with the plane. Shape (3, N), where N is the number of lines.

        Raises
        ------
        ValueError
            If the lines are parallel to the plane or if the input parameters are invalid.

        Notes
        -----
        Disregards direction of line.
        """
        # "ChatGPT 4o" assisted with generating this docstring.

        # Unpack plane
        plane_point, plane_normal_vector = plane

        # Unpack lines
        points, directions = lines

        # normalize inputs
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
                    "Error in plane_lines_intersection(): the 'plane' parameter and 'line(s)' parameter(s) are parallel.",
                )

        plane_normal_vector = plane_normal_vector.normalize()
        plane_vectorV = plane_normal_vector.data  # column vector
        plane_pointV = plane_point.data  # column vector

        # most recent points in light path ensemble
        lt.debug("setting up values...")
        P = points.data
        V = directions.data  # current vectors

        lt.debug("finding intersections...")

        # Intersection Algorithm
        d = np.matmul(plane_vectorV.T, V)  # (1 x N) <- (1 x 3)(3 x N)
        W = P - plane_pointV  # (3 x N) <- (3 x N) -[broadcast] (3 x 1)
        f = -np.matmul(plane_vectorV.T, W) / d  # (1 x N) <- (1 x 3)(3 x N) ./ (1 x N)
        F = f * V  # (3 x N) <- (1 x N) .* (3 x N)
        intersection_matrix = P + F  # (3 x N) <- (3 x N) .- (3 x N)

        intersection_points = Pxyz(intersection_matrix)

        return intersection_points

    @classmethod
    def from_hdf(cls, filename: str, intersection_name: str = "000"):
        """
        Loads intersection points from an HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file containing intersection data.
        intersection_name : str, optional
            The name of the intersection dataset to load. Defaults to "000".

        Returns
        -------
        Intersection
            An Intersection object containing the loaded intersection points.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # get the names of the batches to loop through
        intersection_points = Pxyz(
            list(load_hdf5_datasets([f"Intersection_{intersection_name}/Points"], filename).values())[0]
        )
        return Intersection(intersection_points)

    @classmethod
    def empty_intersection(cls):
        """
        Creates an empty Intersection object.

        Returns
        -------
        Intersection
            An Intersection object with no intersection points.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return cls(Pxyz.empty())

    def __add__(self, intersection: "Intersection"):
        return Intersection(self.intersection_points.concatenate(intersection.intersection_points))

    def __len__(self):
        return len(self.intersection_points)

    def save_to_hdf(self, hdf_filename: str, intersection_name: str = "000"):
        """
        Saves the intersection points to an HDF5 file.

        Parameters
        ----------
        hdf_filename : str
            The path to the HDF5 file where the intersection data will be saved.
        intersection_name : str, optional
            The name of the intersection dataset to save. Defaults to "000".
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        datasets = [f"Intersection_{intersection_name}/Points", f"Intersection_{intersection_name}/Metatdata"]
        data = [self.intersection_points.data, "Placeholder"]
        save_hdf5_datasets(data, datasets, hdf_filename)

    def get_centroid(self) -> Pxyz:
        """
        Calculates the centroid of the intersection points.

        Returns
        -------
        Pxyz
            The centroid of the intersection points as a Pxyz object.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        N = len(self)
        x = sum(self.intersection_points.x) / N
        y = sum(self.intersection_points.y) / N
        z = sum(self.intersection_points.z) / N
        return Pxyz([x, y, z])

    # flux maps

    def to_flux_mapXY(self, bins: int) -> FunctionXYGrid:
        """
        Generates a flux map in the XY plane from the intersection points.

        Parameters
        ----------
        bins : int
            The number of bins to use for the histogram.

        Returns
        -------
        FunctionXYGrid
            A FunctionXYGrid object representing the flux map in the XY plane.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        pxy = Pxy([self.intersection_points.x, self.intersection_points.y])
        return Intersection._Pxy_to_flux_map(pxy, bins)

    def to_flux_mapXZ(self, bins: int) -> FunctionXYGrid:
        """
        Generates a flux map in the XZ plane from the intersection points.

        Parameters
        ----------
        bins : int
            The number of bins to use for the histogram.

        Returns
        -------
        FunctionXYGrid
            A FunctionXYGrid object representing the flux map in the XZ plane.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        pxz = Pxy([self.intersection_points.x, self.intersection_points.z])
        return Intersection._Pxy_to_flux_map(pxz, bins)

    def to_flux_mapYZ(self, bins: int) -> FunctionXYGrid:
        """
        Generates a flux map in the YZ plane from the intersection points.

        Parameters
        ----------
        bins : int
            The number of bins to use for the histogram.

        Returns
        -------
        FunctionXYGrid
            A FunctionXYGrid object representing the flux map in the YZ plane.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        pyz = Pxy([self.intersection_points.y, self.intersection_points.z])
        return Intersection._Pxy_to_flux_map(pyz, bins)

    @staticmethod
    def _Pxy_to_flux_map(points: Pxy, bins: int) -> FunctionXYGrid:

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

    def draw(self, view: View3d, style: RenderControlPointSeq = None):
        """
        Draws the intersection points in a 3D view.

        Parameters
        ----------
        view : View3d
            The 3D view in which to draw the intersection points.
        style : RenderControlPointSeq, optional
            The style to use for rendering the points. Defaults to None.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if style is None:
            style = RenderControlPointSeq()
        self.intersection_points.draw_points(view, style)

    def draw_subset(self, view: View3d, count: int):
        """
        Draws a subset of intersection points in a 3D view.

        Parameters
        ----------
        view : View3d
            The 3D view in which to draw the intersection points.
        count : int
            The number of points to draw from the intersection points.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        for i in np.floor(np.linspace(0, len(self.intersection_points) - 1, count)):
            p = Pxyz(self.intersection_points[int(i)])
            p.draw_points(view)
