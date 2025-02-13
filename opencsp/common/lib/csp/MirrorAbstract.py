"""Abstract mirror representing a single reflective surface
"""

from abc import ABC, abstractmethod

from matplotlib.tri import Triangulation
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.RegionXY import Resolution

from opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract import VisualizeOrthorectifiedSlopeAbstract
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract


class MirrorAbstract(RayTraceable, VisualizeOrthorectifiedSlopeAbstract, OpticOrientationAbstract):
    """
    Abstract class inherited by all mirror classes
    """

    def __init__(self, shape: RegionXY) -> None:
        # super().__init__()
        OpticOrientationAbstract.__init__(self)

        self.region = shape
        self.comments = ["Mirror Comments:"]

        # self._set_optic_children()

    @property
    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        """Returns bounding box aligned to XY axes in mirror's base coordinate
        reference frame.

        Returns
        -------
        Tuple
            (left, right, bottom, top) bounding box. Mirror's base coordinate
            reference frame.
        """
        return self.region.axis_aligned_bounding_box()

    # override from OpticOrientationAbstract
    @property
    def children(self) -> list[OpticOrientationAbstract]:
        return None

    # # override from OpticOrientationAbstract
    def _add_child_helper(self, new_child: OpticOrientationAbstract):
        raise ValueError("Mirror does not accept new children.")

    def in_bounds(self, p: Pxy) -> np.ndarray[bool]:
        """
        Determines what points are valid points on the mirror.
        Input points are in the optic's base coordinate
        reference frame.

        Parameters
        -----------
        p : Pxy
            The set of points in the top-down view of the mirror
            in the mirror's base coordinate reference frame

        Returns
        --------
        np.ndarray[bool]
            1d ndarray with size equal to the length of the input p. Elements
            are booleans. True if point is within optic region, false otherwise.

        """
        return self.region.is_inside_or_on_border(p)

    @abstractmethod
    def surface_norm_at(self, p: Pxy) -> Vxyz:
        """Given an XY sample point in the mirror's base reference frame,
        returns the surface normal at the given location in the
        mirror's base coordinate reference frame.

        Parameters
        ----------
        p : Pxy
            Sample point in mirror's base coordinate reference frame.

        Returns
        -------
        Vxyz
            Normal vector of length len(p) in mirror's base coordinate
            reference frame.
        """

    def surface_normal_in_space(self, p: Pxy) -> Vxyz:
        """Given an XY sample point in the mirror's base reference frame,
        returns the surface normal at the given location in the
        mirror's parent coordinate reference frame.

        Parameters:
        -----------
        p : Pxy
            Sample point in mirror's base coordinate reference frame.

        Returns:
        --------
        Vxyz
            Normal vector of length len(p) in mirror's parent coordinate
            reference frame.
        """
        n = self.surface_norm_at(p)
        return n.rotate(self._self_to_parent_transform.R)

    @abstractmethod
    def surface_displacement_at(self, p: Pxy) -> np.ndarray[float]:
        """Given an XY sample point in the mirror's base reference frame,
        returns the z displacement at the given location in the
        mirror's base coordinate reference frame.

        Parameters
        ----------
        p : Pxy
            Sample point in mirror's base coordinate reference frame.

        Returns
        -------
        ndarray[float]
            Distance from the z=0 plane in mirror's base coordinate
            reference frame.
        """

    def location_at(self, p: Pxy) -> Pxyz:
        """Given an XY sample point in the mirror base reference frame,
        returns the XYZ point on the mirror's surface in the mirror's
        base reference frame.

        Parameters
        ----------
        p : Pxy
            Sample point in mirror's base coordinate reference frame.

        Returns
        -------
        Pxyz
            XYZ sample points on surface in mirror's base coordinate
            reference frame
        """
        z = self.surface_displacement_at(p)
        return Pxyz(np.array([p.x, p.y, z]))

    def location_in_space(self, p: Pxy) -> Pxyz:
        """Given an XY sample point in the mirror's base reference frame,
        returns the XYZ point on the mirror's surface in the mirror's
        parent reference frame.

        Parameters:
        -----------
        p : Pxy
            Sample point in mirror's base coordinate reference frame.

        Returns:
        --------
        Pxyz
            XYZ sample points on surface in mirror's parent coordinate
            reference frame
        """
        z = self.surface_displacement_at(p)
        og_point = Pxyz((p.x, p.y, z))
        return self._self_to_parent_transform.apply(og_point)

    def point_and_normal_in_space(self, p: Pxy) -> tuple[Pxyz, Vxyz]:
        """Given an XY sample point in the mirror's base reference frame,
        return the XYZ point on the mirror's surface and the surface normal
        in the mirror's parent reference frame.

        Parameters
        ----------
        p : Pxy
            Sample point in mirror's base coordinate reference frame.

        Returns
        -------
        tuple[Pxyz, Vxyz]
            Surface points and normal vectors in mirror's parent coordinate
            reference frame.
        """
        point = self.location_in_space(p)
        normal = self.surface_normal_in_space(p)
        return (point, normal)

    def survey_of_points(self, resolution: Resolution) -> tuple[Pxyz, Vxyz]:
        resolution.resolve_in_place(self.region)
        return self._survey_of_points_helper(resolution, TransformXYZ.identity())

    def _survey_of_points_helper(
        self, given_resolution: Resolution, frame_transform: TransformXYZ
    ) -> tuple[Pxyz, Vxyz]:
        resolution = given_resolution.change_frame_and_copy(frame_transform)
        resolution.resolve_in_place(self.region)
        # resolution = resolution.resolve_and_copy(self.region)

        # transformation_to_caller_frame = resolution.composite_transformation
        points = self.self_to_global_tranformation.apply(self.location_at(resolution.points))
        norms = self.surface_norm_at(resolution.points).rotate(self.self_to_global_tranformation.R)
        return points, norms

    # TODO this should use the Resolution class not an int and a string
    def survey_of_points_local(
        self, resolution: int, resolution_type: str = "pixelX", random_seed: int | None = None
    ) -> tuple[Pxyz, Vxyz]:
        """Returns a set of points sampled from inside the optic region in
        the mirror's base coordinate reference frame.

        See self.survey_of_points() for input descriptions

        Returns
        -------
        A tuple of the points (Pxyz) and normals at the respective points (Vxyz) in
        the object's base coordinate reference frame.
        """
        # Get points that will be on the mirror when lifted from the XY plane
        filtered_points = self.region.points_sample(Resolution.pixelX(resolution))
        # Return lifted points and normal vectors in local coordinates
        points = self.location_at(filtered_points)
        norms = self.surface_norm_at(filtered_points)
        return points, norms

    # override from VisualizeOrthorectifiedSlopeAbstract
    def orthorectified_slope_array(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """Returns X and Y surface slopes in ndarray format given X and Y
        sampling axes in the mirror's base coordinate reference frame.

        Parameters
        ----------
        x_vec/y_vec : ndarray
            X and Y grid sampling vectors in mirror's base coordinate reference frame

        Returns
        -------
        ndarray
            X and Y slope images of shape: (2, y_vec.size, x_vec.size), in the mirror's
            base coordinate reference system.
        """
        # Check vectors are 1 dimensional
        if (np.ndim(x_vec) != 1) or (np.ndim(y_vec) != 1):
            raise ValueError(f"X and Y vectors must be 1d, but had shapes: {x_vec.shape}, {y_vec.shape}.")

        # Create interpolation axes
        x_mat, y_mat = np.meshgrid(x_vec, y_vec)  # meters
        pts = Pxy((x_mat, y_mat))

        # Mask data
        mask = self.in_bounds(pts)

        # Calculate normals
        normals = np.zeros((3, len(pts))) * np.nan
        normals[:, mask] = self.surface_norm_at(pts[mask]).data  # 3 x M*N, normalized vectors

        # Calculate slopes
        slopes = -normals[:2] / normals[2:3]  # normalize z coordinate
        return slopes.reshape((2, y_vec.size, x_vec.size))  # 2 x M x N

    def draw(
        self, view: View3d, mirror_style: RenderControlMirror = None, transform: TransformXYZ | None = None
    ) -> None:
        """
        Draws a mirror onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure.
        mirror_styles : RenderControlMirror
            Holds attibutes about the 3d graph.
        transform : TransformXYZ
            3d transform used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the mirror's parent coordinate reference frame.
        """

        if mirror_style is None:
            mirror_style = RenderControlMirror()

        resolution = mirror_style.resolution
        if transform is None:
            transform = self.self_to_global_tranformation

        # Sample points within and on edge of region
        inner_values = self.region.points_sample(resolution)  # 2d, mirror coordinates
        edge_values = self.region.edge_sample(mirror_style.number_of_edge_points)  # 2d, mirror coordinates
        domain = edge_values.concatenate(inner_values)  # 2d, mirror coordinates

        points_surf = self.location_at(domain)  # 3d, mirror coordinates
        edge_values_lifted = self.location_at(edge_values)  # 3d, mirror coordinates

        points_surf = transform.apply(points_surf)  # 3d, current reference frame
        edge_values_lifted = transform.apply(edge_values_lifted)  # 3d, current reference frame

        # Draw surface triangulation
        tri = Triangulation(domain.x, domain.y)  # create triangles
        view.draw_xyz_trisurface(*points_surf.data, surface_style=mirror_style.surface_style, triangles=tri.triangles)

        # Draw surface boundary
        if mirror_style.point_styles is not None:
            mirror_style.point_styles.markersize = 0
            edge_values_lifted.draw_line(view, style=mirror_style.point_styles)

        # Draw surface normals
        if mirror_style.surface_normals:
            # Get sample points and normals
            points, normals = self.survey_of_points_local(mirror_style.norm_res, "pixelX", None)  # mirror coordinates
            points = transform.apply(points)  # current reference frame
            normals.rotate_in_place(transform.R)  # current reference frame

            # Put in list
            xyzdxyz = [[point.data, normal.data * mirror_style.norm_len] for point, normal in zip(points, normals)]
            # Draw on plot
            view.draw_xyzdxyz_list(xyzdxyz, close=False, style=mirror_style.norm_base_style)

    # override function from RayTraceable
    def most_basic_ray_tracable_objects(self) -> list[RayTraceable]:
        return [self]  # any mirror is in the set of most basic ray traceable objects
