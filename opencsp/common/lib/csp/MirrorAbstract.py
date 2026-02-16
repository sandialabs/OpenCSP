"""Abstract mirror representing a single reflective surface"""

from abc import ABC, abstractmethod

from typing import Callable

from matplotlib.tri import Triangulation
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.Resolution import Resolution

from opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract import VisualizeOrthorectifiedSlopeAbstract
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz, connection_lines
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror
import opencsp.common.lib.render_control.RenderControlMirrorProjected as rcmp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract

import opencsp.common.lib.tool.log_tools as lt


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

    # override function from RayTraceable
    def most_basic_ray_tracable_objects(self) -> list[RayTraceable]:
        return [self]  # any mirror is in the set of most basic ray traceable objects

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

    def lift_xy(self, projected_xy: Vxy) -> Vxyz:
        """Given a set of points on the (x,y) plane, construct the
        corresponding set of (x,y,z) points on the mirror surface.

        Parameters
        ----------
        projected_xy : Vxy
            Set of (x,y) points to lift.
            If any points are outside the domain of the surface function,
            their corresponding lifted z values will be zero.

        Returns
        -------
        Vxyz
            Set of (x,y,z) points, corresponding to the input (x,y) points lifted
            onto the mirror surface.  The order of points is preserved.
        """
        lifted_z = self.surface_displacement_at(projected_xy)
        # If any xy points aren't in range of the mirror's interpolation function, set to 0.
        if np.any(np.isnan(lifted_z)):
            lt.warn("In MirorParametric.lift_xy(), could not find z values for some input points; filling with zeros.")
            lifted_z = np.nan_to_num(lifted_z, nan=0)
        lifted_xyz = Vxyz((projected_xy.x, projected_xy.y, lifted_z))
        # Return
        return lifted_xyz

    def on_surface_xyz(self, general_xyz: Vxyz) -> Vxyz:
        """Given a set of points in (x,y,z) space, construct the
        corresponding (x,y,z) points on the mirror surface.

        Parameters
        ----------
        general_xyz : Vxyz
            Set of (x,y) points to lift.
            If any points are outside the domain of the surface function,
            their corresponding z values will be zero.

        Returns
        -------
        Vxyz
            Set of (x,y,z) points, corresponding to the input (x,y,z) points
            projected up or down onto the mirror surface.
            The order of points is preserved.
        """
        on_surface_z = self.surface_displacement_at(Pxy((general_xyz.x, general_xyz.y)))
        # If any xy points aren't in range of the mirror's interpolation function, set to 0.
        if np.any(np.isnan(on_surface_z)):
            lt.warn(
                "In MirorParametric.on_surface_xyz(), could not find z values for some input points; filling with zeros."
            )
            on_surface_z = np.nan_to_num(on_surface_z, nan=0)
        on_surface_xyz = Vxyz((general_xyz.x, general_xyz.y, on_surface_z))
        # Return
        return on_surface_xyz

    def number_of_boundary_vertices(self) -> int:
        """Returns the number of vertices in the region bounding the mirror.

        Returns
        -------
        int
            Number of vertices in the mirror boundary.
        """
        return self.projected_boundary_xy().len()

    def projected_boundary_xy(self) -> Vxy:
        """Returns a sequence of (x,y) points delineating the boundary
        of the mirror, projected onto the (x,y) plane.

        Returns
        -------
        Vxy
            Boundary of the mirror, projected onto the (x,y) plane.
        """
        return Vxy.merge([loop.vertices for loop in self.region.loops])

    def projected_boundary_xyz(self) -> Vxyz:
        """Returns a sequence of (x,y,z) points delineating the boundary
        of the mirror, projected onto the (x,y) plane.

        Returns
        -------
        Vxyz
            Boundary of the mirror, projected onto the (x,y) plane.
            z=0 for all points.
        """
        projected_boundary_xy = self.projected_boundary_xy()
        n_vertices = projected_boundary_xy.data.shape[1]
        projected_boundary_z = np.zeros(n_vertices)
        return Vxyz((projected_boundary_xy.x, projected_boundary_xy.y, projected_boundary_z))

    def constant_x_slice(self, x: float, y_min: float, y_max: float, n_points: int) -> Vxyz:
        """Returns a sequence of (x,y,z) points corresponding to a slice of constant x.
        Returns two sets of points: On the surface, and projected onto the (x,y) plane.

        Parameters
        ----------
        x : (float)
            x value for this constant-x slice.
        y_min : float
            y lower bound for this slice.
        y_max : float
            y upper bound for this slice.
        n_points : int
            Number of points to sample along the slice.

        Returns
        -------
        Vxyz
            Slice points on the mirror surface.
        Vxyz
            Slice points projected onto the (x,y) plane.
        """
        list_x = []
        list_y = []
        for y in np.linspace(y_min, y_max, n_points):
            list_x.append(x)
            list_y.append(y)
        projected_xyz = Vxyz((list_x, list_y, [0] * n_points))
        # Lifted slice in xyz space.
        lifted_xyz = self.on_surface_xyz(projected_xyz)
        # Return
        return lifted_xyz, projected_xyz

    def constant_y_slice(self, y: float, x_min: float, x_max: float, n_points: int) -> Vxyz:
        """Returns a sequence of (x,y,z) points corresponding to a slice of constant y.
        Returns two sets of points: On the surface, and projected onto the (x,y) plane.

        Parameters
        ----------
        y : (float)
            y value for this constant-y slice.
        x_min : float
            x lower bound for this slice.
        x_max : float
            x upper bound for this slice.
        n_points : int
            Number of points to sample along the slice.

        Returns
        -------
        Vxyz
            Slice points on the mirror surface.
        Vxyz
            Slice points projected onto the (x,y) plane.
        """
        list_x = []
        list_y = []
        for x in np.linspace(x_min, x_max, n_points):
            list_x.append(x)
            list_y.append(y)
        projected_xyz = Vxyz((list_x, list_y, [0] * n_points))
        # Lifted slice in xyz space.
        lifted_xyz = self.on_surface_xyz(projected_xyz)
        # Return
        return lifted_xyz, projected_xyz

    def draw(
        self,
        view: View3d,
        mirror_style: RenderControlMirror = None,
        draw_projection: bool = False,
        projected_style: rcmp.RenderControlMirrorProjected = None,
        transform: TransformXYZ | None = None,
    ) -> None:
        """
        Draws a mirror onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure.
        mirror_style : RenderControlMirror | None
            Holds attributes defining how to draw features, etc.  Default None.
        draw_projection: bool, optional
            Whether to draw the projection of the mirror onto the (x,y) plane.  Default False.
        projected_style: RenderControlMirrorProjected | None
            Holds attributes defining how to draw projection features.  Default None.
        transform : TransformXYZ
            3d transform used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the mirror's parent coordinate reference frame.
        """
        if mirror_style is None:
            mirror_style = RenderControlMirror()

        resolution = mirror_style.resolution

        # Ensure that transform is defined.
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

        # # Draw surface boundary
        # if mirror_style.point_styles is not None:
        #     mirror_style.point_styles.markersize = 0
        #     edge_values_lifted.draw_line(view, style=mirror_style.point_styles)

        # Draw projection onto (x,y) plane.
        if draw_projection:
            if projected_style is None:
                projected_style = rcmp.RenderControlMirrorProjected(
                    lifted_line_style=rcps.outline(color='red'),
                    lifted_vertex_style=rcps.marker(color='red', markersize=2),
                    projected_style=rcps.outline(color='blue'),
                    connection_style=rcps.outline(color='blue', linewidth=0.6),
                )
            # Draw defining region.
            # Projected boundary on xy plane.
            projected_boundary_xyz = self.projected_boundary_xyz()
            # Lifted boundary in xyz space.
            lifted_boundary_xyz = self.on_surface_xyz(projected_boundary_xyz)
            # Fine-resolution lifted boundary, showing edge curvature.
            fine_projected_boundary_xy = Vxy.merge([loop.edge_sample(20) for loop in self.region.loops])
            fine_lifted_boundary_xyz = self.lift_xy(fine_projected_boundary_xy)
            if projected_style.projected_style is not None:
                # Draw projected boundary.
                transform.apply(projected_boundary_xyz).draw_line(
                    view, close=True, style=projected_style.projected_style
                )
            if projected_style.connection_style is not None:
                # Draw lines connecting projected boundary vertices to lifted boundary vertices.
                projected_to_lifted_boundary_lines_xyz = connection_lines(projected_boundary_xyz, lifted_boundary_xyz)
                for line_xyz in projected_to_lifted_boundary_lines_xyz:
                    transform.apply(line_xyz).draw_line(view, close=False, style=projected_style.connection_style)
            if projected_style.lifted_line_style is not None:
                # Draw lifted boundary.
                transform.apply(fine_lifted_boundary_xyz).draw_line(
                    view, close=True, style=projected_style.lifted_line_style
                )
            if projected_style.lifted_vertex_style is not None:
                # Draw lifted vertices last.
                transform.apply(lifted_boundary_xyz).draw_line(
                    view, close=True, style=projected_style.lifted_vertex_style
                )

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

    def draw_slice_constant_x(
        self,
        view: View3d,
        slice_x: float,
        y_min: float,
        y_max: float,
        style: rcmp.RenderControlMirrorProjected = None,
        transform: TransformXYZ | None = None,
    ) -> None:
        """
        Draws a constant-x slice of a mirror onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure.
        slice_x: float
            x value defining the slice.
        y_min: float
            Minimum y value defining the slice lower bound endpoint.
        y_max: float
            Maximum y value defining the slice upper bound endpoint.
        style : RenderControlMirrorProjected
            Holds attributes defining how to draw features, etc.
        transform : TransformXYZ
            3d transform used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the mirror's parent coordinate reference frame.
        """
        # Ensure that the number of vertices is set.
        if style is None:
            n_vertices = 9
            fine_n_vertices = 41
        else:
            n_vertices = style.slice_n_vertices
            fine_n_vertices = style.slice_fine_n_vertices

        # Ensure that transform is defined.
        if transform is None:
            transform = self.self_to_global_tranformation

        # Construct.
        # Coarse-resolution lifted slice, with vertices to draw.
        lifted_slice_xyz, _ = self.constant_x_slice(slice_x, y_min, y_max, n_vertices)
        # Fine-resolution lifted slice, showing edge curvature.
        fine_lifted_slice_xyz, _ = self.constant_x_slice(slice_x, y_min, y_max, fine_n_vertices)

        # Draw.
        # Draw lifted slice.
        if style.lifted_line_style is not None:
            transform.apply(fine_lifted_slice_xyz).draw_line(view, close=False, style=style.lifted_line_style)
        # Draw lifted slice vertices last.
        if style.lifted_vertex_style is not None:
            transform.apply(lifted_slice_xyz).draw_line(view, close=False, style=style.lifted_vertex_style)

    def draw_slice_constant_y(
        self,
        view: View3d,
        slice_y: float,
        x_min: float,
        x_max: float,
        style: rcmp.RenderControlMirrorProjected = None,
        transform: TransformXYZ | None = None,
    ) -> None:
        """
        Draws a constant-y slice of a mirror onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure.
        slice_y: float
            y value defining the slice.
        x_min: float
            Minimum x value defining the slice lower bound endpoint.
        x_max: float
            Maximum x value defining the slice upper bound endpoint.
        style : RenderControlMirrorProjected
            Holds attributes defining how to draw features, etc.
        transform : TransformXYZ
            3d transform used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the mirror's parent coordinate reference frame.
        """
        # Ensure that the number of vertices is set.
        if style is None:
            n_vertices = 9
            fine_n_vertices = 41
        else:
            n_vertices = style.slice_n_vertices
            fine_n_vertices = style.slice_fine_n_vertices

        # Ensure that transform is defined.
        if transform is None:
            transform = self.self_to_global_tranformation

        # Construct.
        # Coarse-resolution lifted slice, with vertices to draw.
        lifted_slice_xyz, _ = self.constant_y_slice(slice_y, x_min, x_max, n_vertices)
        # Fine-resolution lifted slice, showing edge curvature.
        fine_lifted_slice_xyz, _ = self.constant_y_slice(slice_y, x_min, x_max, fine_n_vertices)

        # Draw.
        # Draw lifted slice.
        if style.lifted_line_style is not None:
            transform.apply(fine_lifted_slice_xyz).draw_line(view, close=False, style=style.lifted_line_style)
        # Draw lifted slice vertices last.
        if style.lifted_vertex_style is not None:
            transform.apply(lifted_slice_xyz).draw_line(view, close=False, style=style.lifted_vertex_style)

    def draw_projected_lifted_slice_constant_x(
        self,
        view: View3d,
        slice_x: float,
        y_min: float,
        y_max: float,
        style: rcmp.RenderControlMirrorProjected = None,
        transform: TransformXYZ | None = None,
    ) -> None:
        """
        Draws a constant-x slice of a mirror onto a View3d object, showing both the slice
        of the mirror surface, and the corresponding projected line on the (x,y) plane,
        with correspondence lines drawn at the slice coarse vertices.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure.
        slice_x: float
            x value defining the slice.
        y_min: float
            Minimum y value defining the slice lower bound endpoint.
        y_max: float
            Maximum y value defining the slice upper bound endpoint.
        style : RenderControlMirrorProjected
            Holds attributes defining how to draw features, etc.
        transform : TransformXYZ
            3d transform used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the mirror's parent coordinate reference frame.
        """
        # Ensure that the number of vertices is set.
        if style is None:
            n_vertices = 9
            fine_n_vertices = 41
        else:
            n_vertices = style.slice_n_vertices
            fine_n_vertices = style.slice_fine_n_vertices

        # Ensure that transform is defined.
        if transform is None:
            transform = self.self_to_global_tranformation

        # Construct.
        # Coarse-resolution lifted slice, with vertices to draw.
        lifted_slice_xyz, projected_slice_xyz = self.constant_x_slice(slice_x, y_min, y_max, n_vertices)
        # Fine-resolution lifted slice, showing edge curvature.
        fine_lifted_slice_xyz, _ = self.constant_x_slice(slice_x, y_min, y_max, fine_n_vertices)

        # Draw.
        # Projected slice.
        if style.projected_style is not None:
            transform.apply(projected_slice_xyz).draw_line(view, close=False, style=style.projected_style)
        # Connection lines.
        if style.connection_style is not None:
            # Construct and draw lines connecting projected slice vertices to lifted slice vertices.
            projected_to_lifted_slice_lines_xyz = connection_lines(projected_slice_xyz, lifted_slice_xyz)
            for line_xyz in projected_to_lifted_slice_lines_xyz:
                transform.apply(line_xyz).draw_line(view, close=False, style=style.connection_style)
        # Lifted slice.
        if style.lifted_line_style is not None:
            # Draw lifted slice.
            transform.apply(fine_lifted_slice_xyz).draw_line(view, close=False, style=style.lifted_line_style)
        if style.lifted_vertex_style is not None:
            # Draw lifted slice vertices last.
            transform.apply(lifted_slice_xyz).draw_line(view, close=False, style=style.lifted_vertex_style)

    def draw_projected_lifted_slice_constant_y(
        self,
        view: View3d,
        slice_y: float,
        x_min: float,
        x_max: float,
        style: rcmp.RenderControlMirrorProjected = None,
        transform: TransformXYZ | None = None,
    ) -> None:
        """
        Draws a constant-y slice of a mirror onto a View3d object, showing both the slice
        of the mirror surface, and the corresponding projected line on the (x,y) plane,
        with correspondence lines drawn at the slice coarse vertices.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure.
        slice_y: float
            y value defining the slice.
        x_min: float
            Minimum x value defining the slice lower bound endpoint.
        x_max: float
            Maximum x value defining the slice upper bound endpoint.
        style : RenderControlMirrorProjected
            Holds attributes defining how to draw features, etc.
        transform : TransformXYZ
            3d transform used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the mirror's parent coordinate reference frame.
        """
        # Ensure that the number of vertices is set.
        if style is None:
            n_vertices = 9
            fine_n_vertices = 41
        else:
            n_vertices = style.slice_n_vertices
            fine_n_vertices = style.slice_fine_n_vertices

        # Ensure that transform is defined.
        if transform is None:
            transform = self.self_to_global_tranformation

        # Construct.
        # Coarse-resolution lifted slice, with vertices to draw.
        lifted_slice_xyz, projected_slice_xyz = self.constant_y_slice(slice_y, x_min, x_max, n_vertices)
        # Fine-resolution lifted slice, showing edge curvature.
        fine_lifted_slice_xyz, _ = self.constant_y_slice(slice_y, x_min, x_max, fine_n_vertices)

        # Draw.
        # Projected slice.
        if style.projected_style is not None:
            transform.apply(projected_slice_xyz).draw_line(view, close=False, style=style.projected_style)
        # Connection lines.
        if style.connection_style is not None:
            # Construct and draw lines connecting projected slice vertices to lifted slice vertices.
            projected_to_lifted_slice_lines_xyz = connection_lines(projected_slice_xyz, lifted_slice_xyz)
            for line_xyz in projected_to_lifted_slice_lines_xyz:
                transform.apply(line_xyz).draw_line(view, close=False, style=style.connection_style)
        # Lifted slice.
        if style.lifted_line_style is not None:
            # Draw lifted slice.
            transform.apply(fine_lifted_slice_xyz).draw_line(view, close=False, style=style.lifted_line_style)
        if style.lifted_vertex_style is not None:
            # Draw lifted slice vertices last.
            transform.apply(lifted_slice_xyz).draw_line(view, close=False, style=style.lifted_vertex_style)

    def _draw_surfacemesh_aux_x(
        self,
        view: View3d,
        x: float,
        mirror_y_min: float,
        mirror_y_max: float,
        project: bool,
        style: rcmp.RenderControlMirrorProjected,
        transform: TransformXYZ | None = None,
    ) -> None:
        """Supports draw_surface_mesh() function by handling projected vs. non-projected
        cases for constant-x slices."""
        if project:
            self.draw_projected_lifted_slice_constant_x(
                view, slice_x=x, y_min=mirror_y_min, y_max=mirror_y_max, transform=transform, style=style
            )
        else:
            self.draw_slice_constant_x(
                view, slice_x=x, y_min=mirror_y_min, y_max=mirror_y_max, transform=transform, style=style
            )

    def _draw_surfacemesh_aux_y(
        self,
        view: View3d,
        y: float,
        mirror_x_min: float,
        mirror_x_max: float,
        project: bool,
        style: rcmp.RenderControlMirrorProjected,
        transform: TransformXYZ | None = None,
    ) -> None:
        """Supports draw_surface_mesh() function by handling projected vs. non-projected
        cases for constant-y slices."""
        if project:
            self.draw_projected_lifted_slice_constant_y(
                view, slice_y=y, x_min=mirror_x_min, x_max=mirror_x_max, transform=transform, style=style
            )
        else:
            self.draw_slice_constant_y(
                view, slice_y=y, x_min=mirror_x_min, x_max=mirror_x_max, transform=transform, style=style
            )

    def draw_surface_mesh(
        self,
        view: View3d,
        n_slices_x: int = 11,
        project_slices_x: bool = False,
        draw_special_origin_slice_x: bool = True,
        project_origin_slice_x: bool = True,
        n_slices_y: int = 11,
        project_slices_y: bool = False,
        draw_special_origin_slice_y: bool = True,
        project_origin_slice_y: bool = False,
        slice_style: rcmp.RenderControlMirrorProjected | None = None,
        origin_slice_style: rcmp.RenderControlMirrorProjected | None = None,
        transform: TransformXYZ | None = None,
    ) -> None:
        """Draws a mesh on the mirror surface, optionally with projected slice correspondence.

        If any arguments that control projection rendering are true (for example, project_slices_x),
        then the corresponding slices are drawn including their projection onto the (x,y) plane,
        with correspondence lines for coarse vertices.

        Parameters
        ----------
        view : View3d
            A view 3d object that holds the figure.
        n_slices_x : int
            Number of constant-x slices to draw.
            If you select an odd number and the mirror extent is balanced about x=0,
            then an ordinary slice is drawn at the origin x=0.
            Default 11.
        project_slices_x : bool
            Whether to draw projection of all constant-x slices.
        draw_special_origin_slice_x : bool
            Whether to draw a constant-x slice at x=0.  Overwrites an ordinary slice there.
        project_origin_slice_x : bool
            Whether to draw projection of origin constant-x slice.
        n_slices_y : int
            Number of constant-y slices to draw.
            If you select an odd number and the mirror extent is balanced about x=0,
            then an ordinary slice is drawn at the origin x=0.
            Default 11.
        project_slices_y : bool
            Whether to draw projection of all constant-y slices.
        draw_special_origin_slice_y : bool
            Whether to draw a constant-y slice at y=0.  Overwrites an ordinary slice there.
        project_origin_slice_y : bool
            Whether to draw projection of origin constant-y slice.
        slice_style : RenderControlMirrorProjected
            Style to draw the ordinary slices.  Holds attributes defining how to draw features.
            If None, default style parameters are used.
        origin_slice_style : RenderControlMirrorProjected
            Style to draw the origin slices.  Holds attributes defining how to draw features.
            If None, default style parameters are used.
        """
        # Ensure style control is set.
        if slice_style is None:
            slice_style = rcmp.mirror_contour()
        if origin_slice_style is None:
            origin_slice_style = rcmp.mirror_origin_contour()

        # Ensure that transform is defined.
        if transform is None:
            transform = self.self_to_global_tranformation

        # Get mirror extent.
        mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max = self.axis_aligned_bounding_box

        # Draw ordinary constant-x slices.
        if n_slices_x > 0:
            if n_slices_x == 1:
                # Linspace with n=1 returns the start point, not the middle,
                # so we handle this case separately.
                x_middle = (mirror_x_min, mirror_x_max) / 2.0
                self._draw_surfacemesh_aux_x(
                    view, x_middle, mirror_y_min, mirror_y_max, project_slices_x, slice_style, transform
                )
            else:
                for x in np.linspace(mirror_x_min, mirror_x_max, n_slices_x):
                    self._draw_surfacemesh_aux_x(
                        view, x, mirror_y_min, mirror_y_max, project_slices_x, slice_style, transform
                    )

        # Draw ordinary constant-y slices.
        if n_slices_y > 0:
            if n_slices_y == 1:
                # Linspace with n=1 returns the start point, not the middle,
                # so we handle this case separately.
                y_middle = (mirror_y_min, mirror_y_max) / 2.0
                self._draw_surfacemesh_aux_y(
                    view, y_middle, mirror_x_min, mirror_x_max, project_slices_y, slice_style, transform
                )
            else:
                for y in np.linspace(mirror_y_min, mirror_y_max, n_slices_y):
                    self._draw_surfacemesh_aux_y(
                        view, y, mirror_x_min, mirror_x_max, project_slices_y, slice_style, transform
                    )

        # Draw origin constant-x slice.
        if draw_special_origin_slice_x:
            self._draw_surfacemesh_aux_x(
                view, 0.0, mirror_y_min, mirror_y_max, project_origin_slice_x, origin_slice_style, transform
            )

        # Draw origin constant-y slice.
        if draw_special_origin_slice_y:
            self._draw_surfacemesh_aux_y(
                view, 0.0, mirror_x_min, mirror_x_max, project_origin_slice_y, origin_slice_style, transform
            )
