"""Facet class inherited by all facet classes"""

import numpy as np

from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract import VisualizeOrthorectifiedSlopeAbstract
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlFacet import RenderControlFacet
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
import opencsp.common.lib.tool.log_tools as lt

UP = Vxyz([0, 0, 1])


class Facet(RayTraceable, VisualizeOrthorectifiedSlopeAbstract, OpticOrientationAbstract):
    """Facet representation that contains a MirrorAbstract object."""

    def __init__(self, mirror: MirrorAbstract, name: str = None) -> "Facet":
        """Instantiates Facet class

        Parameters
        ----------
        mirror : MirrorAbstract
            Mirror object held inside Facet
        """
        self.mirror = mirror
        self.name = name

        # OpticOrientationAbstract information
        OpticOrientationAbstract.__init__(self)

        self.pointing_function = None

    @property
    def transform_mirror_to_self(self) -> TransformXYZ:
        """Returns the mirror to facet 3d transform"""
        return self.mirror._self_to_parent_transform

    # override from VisualizeOrthorectifiedSlopeAbstract
    @property
    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        """Returns bounding box aligned to XY axes in facet's child coordinate
        reference frame.

        Returns
        -------
        tuple[float, float, float, float]
            Left, right, bottom, top. Facet's child coordinate reference frame.
        """
        # Get XYZ locations of all points making up mirror region
        points_xy = Vxy.merge([loop.vertices for loop in self.mirror.region.loops])  # mirror frame
        points_z = self.mirror.surface_displacement_at(points_xy)  # mirror frame
        # If the corners aren't in range of the mirror's interpolation function, set to 0
        if np.any(np.isnan(points_z)):
            lt.warn("Could not find corner z values for facet; filling with zeros.")
            points_z = np.nan_to_num(points_z, nan=0)
        points_xyz = Vxyz((points_xy.x, points_xy.y, points_z))  # mirror frame

        # Transform "mirror base" to "facet child" coordinates
        xyz = self.mirror._self_to_parent_transform.apply(points_xyz)  # facet frame

        # Find bounding box
        return xyz.x.min(), xyz.x.max(), xyz.y.min(), xyz.y.max()  # facet frame

    # override from OpticOrientationAbstract
    @property
    def children(self) -> list[OpticOrientationAbstract]:
        return [self.mirror]

    # override from OpticOrientationAbstract
    def _add_child_helper(self, new_child: OpticOrientationAbstract):
        raise ValueError("Facet does not accept new children.")

    # override from RayTraceable
    def survey_of_points(self, resolution: Resolution) -> tuple[Pxyz, Vxyz]:
        # Get sample point locations (z=0 plane in "child" reference frame)
        resolution.resolve_in_place(self.axis_aligned_bounding_box)
        return self._survey_of_points_helper(resolution, TransformXYZ.identity())

    def _survey_of_points_helper(
        self, given_resolution: Resolution, frame_transform: TransformXYZ
    ) -> tuple[Pxyz, Vxyz]:
        resolution = given_resolution.change_frame_and_copy(frame_transform)
        resolution.resolve_in_place(self.axis_aligned_bounding_box)

        return self.mirror._survey_of_points_helper(resolution, self.mirror._self_to_parent_transform.inv())

    # override from VisualizeOrthorectifiedSlopeAbstract
    def orthorectified_slope_array(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """Returns X and Y surface slopes in ndarray format given X and Y
        sampling axes in the facet's child coordinate reference frame.

        Parameters
        ----------
        x_vec/y_vec : ndarray
            X and Y grid sampling vectors in facet's child coordinate reference frame

        Returns
        -------
        ndarray
            X and Y slope images of shape: (2, y_vec.size, x_vec.size), in the facet's
            child coordinate reference system.
        """
        # Get sample points
        x_mat, y_mat = np.meshgrid(x_vec, y_vec)  # facet child
        z_mat = np.zeros(x_mat.shape)  # facet child
        points_samp = Vxyz((x_mat, y_mat, z_mat))  # facet child

        # Get mask of points on mirror
        points_samp_mirror = self.transform_mirror_to_self.inv().apply(points_samp)  # mirror base
        mask = self.mirror.in_bounds(points_samp_mirror.projXY())
        points_samp_mirror = points_samp_mirror[mask]

        # Get normal vectors
        normals = self.mirror.surface_norm_at(points_samp_mirror.projXY())  # mirror base
        normals.rotate_in_place(self.transform_mirror_to_self.R)  # facet child

        # Calculate slopes and output as 2D array
        slope_data = np.zeros((2, len(points_samp))) * np.nan  # facet child
        slope_data[:, mask] = -normals.data[:2] / normals.data[2:3]  # facet child
        slope_data = np.reshape(slope_data, (2, y_vec.size, x_vec.size))  # facet child
        return slope_data  # facet child

    def get_2D_dimensions(self) -> tuple[float, float]:
        """Returns width and heightin facet's child coordinate
        reference frame.

        Returns
        -------
        tuple[float, float]
            Width: following (+)x infinity in the (-) direction for x_max(right) - following (-x) infinity in the (+) direction for x_min(left).
            Height: following (+)y infinity in the (-) direction for y_max(top) - following (-y) infinity in the (+) direction for y_min(bottom).
            Facet's child coordinate reference frame.
        """
        left, right, bottom, top = self.axis_aligned_bounding_box
        width = right - left
        height = top - bottom
        return width, height

    # override function from RayTraceable
    def most_basic_ray_tracable_objects(self) -> list[RayTraceable]:
        return self.mirror.most_basic_ray_tracable_objects()

    def draw(self, view: View3d, facet_style: RenderControlFacet = None, transform: TransformXYZ = None) -> None:
        """
        Draws facet mirror onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure.
        mirror_styles : RenderControlMirror
            Holds attibutes about the 3d graph.
        transform : TransformXYZ
            3d transform used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the facet's global coordinate reference frame.
        """

        if facet_style is None:
            facet_style = RenderControlFacet()

        if transform is None:
            transform = self.self_to_global_tranformation

        origin = transform.apply(Vxyz.origin())

        # Centroid.
        if facet_style.draw_centroid:
            origin.draw_points(view, style=facet_style.centroid_style)

        # Outline.
        if facet_style.draw_outline:
            # corners = [self.top_left_corner,
            #            self.top_right_corner,
            #            self.bottom_right_corner,
            #            self.bottom_left_corner]
            # view.draw_xyz_list(corners, close=True, style=facet_style.outline_style)
            left, right, bottom, top = self.axis_aligned_bounding_box
            border = Pxyz([[left, left, right, right], [top, bottom, bottom, top], [0, 0, 0, 0]])
            transform.apply(border).draw_line(view, close=True, style=facet_style.outline_style)

        # Surface normal.
        if facet_style.draw_surface_normal:
            # Construct ray.
            surface_normal_ray = transform.apply(UP * facet_style.surface_normal_length)
            # Draw ray and its base.
            origin.draw_points(view, style=facet_style.surface_normal_base_style)
            Vxyz.merge([origin, surface_normal_ray]).draw_line(view, style=facet_style.surface_normal_style)

        # # Surface normal drawn at corners.
        # # (Not the surface normal at the corner.  Facet curvature is not shown.)
        # if facet_style.draw_surface_normal_at_corners:
        #     # Construct rays.
        #     top_left_ray = self.surface_normal_ray(self.top_left_corner, facet_style.corner_normal_length)
        #     top_right_ray = self.surface_normal_ray(self.top_right_corner, facet_style.corner_normal_length)
        #     bottom_left_ray = self.surface_normal_ray(self.bottom_left_corner, facet_style.corner_normal_length)
        #     bottom_right_ray = self.surface_normal_ray(self.bottom_right_corner, facet_style.corner_normal_length)
        #     rays = [top_left_ray,
        #             top_right_ray,
        #             bottom_left_ray,
        #             bottom_right_ray]
        #     corners = [self.top_left_corner,
        #                self.top_right_corner,
        #                self.bottom_right_corner,
        #                self.bottom_left_corner]
        #     # Draw each ray and its base.
        #     for base, ray in zip(corners, rays):
        #         view.draw_xyz(base, style=facet_style.corner_normal_base_style)
        #         view.draw_xyz_list(ray, style=facet_style.corner_normal_style)

        # Name.
        if facet_style.draw_name:
            view.draw_xyz_text(origin.data.T[0], self.name, style=facet_style.name_style)

        if facet_style.draw_mirror_curvature:
            self.mirror.draw(view, facet_style.mirror_styles, transform * self.mirror._self_to_parent_transform)
