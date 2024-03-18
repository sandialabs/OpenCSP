"""

"""

from warnings import warn

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlFacet import RenderControlFacet


class Facet(RayTraceable):
    """
    Heliostat facet

    *Assumes a rectangular facet*

    [Args]:
        name    [str]: facet's name
        centroid_offset [list[float]] = facet's origin offset from centered facet's origin - Facing up
        width   [float]: facet width in meters (derived from the mirror width)
        height  [float]: facet height in meters (derived from the mirror height)
    """

    def __init__(
        self,
        name: str,
        mirror: MirrorAbstract,
        centroid_offset: np.ndarray = np.array([0, 0, 0]),
        canting: Rotation = Rotation.identity(),
    ):
        super(Facet, self).__init__()

        self.name = name
        self.mirror = mirror
        self.canting = canting
        self.centroid_offset = centroid_offset
        self.origin: np.ndarray = np.array([0, 0, 0])
        self.width: float = mirror.width
        self.height: float = mirror.height

        # Facet Corners [offsets in terms of facet's centoid]
        self.top_left_corner_offset = [-self.width / 2, self.height / 2, 0.0]
        self.top_right_corner_offset = [self.width / 2, self.height / 2, 0.0]
        self.bottom_right_corner_offset = [self.width / 2, -self.height / 2, 0.0]
        self.bottom_left_corner_offset = [-self.width / 2, -self.height / 2, 0.0]

        # Surface normals when heliostat is in face up postion.
        self.surface_normal_face_up = [0, 0, 1]

        self.set_position_in_space(self.origin, self.canting)

        # if additional information (backside structure, bolt locations, etc) is needed
        # Fill in here

    def set_position_in_space(self, heliostat_origin: np.ndarray, rotation: Rotation) -> None:  #TODO np.ndarray needs to be changed to a Pxyz
        # Sets facet's position given heliostat configuration.
        self.origin: np.ndarray = np.array(heliostat_origin) + rotation.apply(
            np.array(self.centroid_offset)
        )  # R_aiming * T_pivot * T_offset * R_canting * M_origin
        self.composite_rotation: Rotation = rotation * self.canting  # TODO tjlarki: is this right?
        self.surface_normal = self.composite_rotation.apply(
            [0, 0, 1]
        )  # TODO tjlarki: rename this center surface normal, or normal direction.

        self.mirror.set_position_in_space(self.origin, self.composite_rotation)

        self._update_corners_position_in_space()
        pass

    def set_facet_position_in_space(self, hel_centroid: np.ndarray, hel_rotation: Rotation) -> None:
        warn("Use Facet.set_position_in_space() instead.", category=DeprecationWarning, stacklevel=2)
        self.set_position_in_space(hel_centroid, hel_rotation)
        pass

    def _update_corners_position_in_space(self) -> None:
        # following are not set up for canting angle
        self.top_left_corner = self.origin + self.composite_rotation.apply(self.top_left_corner_offset)
        self.top_right_corner = self.origin + self.composite_rotation.apply(self.top_right_corner_offset)
        self.bottom_right_corner = self.origin + self.composite_rotation.apply(self.bottom_right_corner_offset)
        self.bottom_left_corner = self.origin + self.composite_rotation.apply(self.bottom_left_corner_offset)

    # def update_position_in_space(self):
    #     self.set_position_in_space(self.origin, self.canting)

    def set_facet_position_in_space(self, hel_centroid: np.ndarray, hel_rotation: Rotation) -> None:
        warn("Use Facet.set_position_in_space() instead.", category=DeprecationWarning, stacklevel=2)
        self.set_position_in_space(hel_centroid, hel_rotation)

    def surface_normal_ray(self, base: np.ndarray, length: float):
        # Constructs the head and tail of a vector of given length, placed at the base
        # position (computed after applying the heliostat configuration).
        #
        # This assumes that set_position_in_space() has already been called for
        # the current heliostat configuration.  This is required to set the internal
        # surface_normal.
        tail = base
        head = tail + (length * np.array(self.surface_normal))
        ray = [tail, head]
        return ray

    def draw(self, view: View3d, facet_styles: RenderControlFacet):
        # Compute and set the point positions for the current heliostat configuration.
        # self.set_facet_position_in_space(hel_centroid, hel_rotation)

        # Fetch draw style control.
        facet_style = facet_styles.style(self.name)

        if facet_style.draw_mirror_curvature:
            self.mirror.draw(view, facet_style.mirror_styles)

        # Centroid.
        if facet_style.draw_centroid:
            view.draw_xyz(self.origin, style=facet_style.centroid_style)

        # Outline.
        if facet_style.draw_outline:
            corners = [self.top_left_corner, self.top_right_corner, self.bottom_right_corner, self.bottom_left_corner]
            view.draw_xyz_list(corners, close=True, style=facet_style.outline_style)

        # Surface normal.
        if facet_style.draw_surface_normal:
            # Construct ray.
            surface_normal_ray = self.surface_normal_ray(self.origin, facet_style.surface_normal_length)
            # Draw ray and its base.
            view.draw_xyz(self.origin, style=facet_style.surface_normal_base_style)
            view.draw_xyz_list(surface_normal_ray, style=facet_style.surface_normal_style)

        # Surface normal drawn at corners.
        # (Not the surface normal at the corner.  Facet curvature is not shown.)
        if facet_style.draw_surface_normal_at_corners:
            # Construct rays.
            top_left_ray = self.surface_normal_ray(self.top_left_corner, facet_style.corner_normal_length)
            top_right_ray = self.surface_normal_ray(self.top_right_corner, facet_style.corner_normal_length)
            bottom_left_ray = self.surface_normal_ray(self.bottom_left_corner, facet_style.corner_normal_length)
            bottom_right_ray = self.surface_normal_ray(self.bottom_right_corner, facet_style.corner_normal_length)
            rays = [top_left_ray, top_right_ray, bottom_left_ray, bottom_right_ray]
            corners = [self.top_left_corner, self.top_right_corner, self.bottom_right_corner, self.bottom_left_corner]
            # Draw each ray and its base.
            for base, ray in zip(corners, rays):
                view.draw_xyz(base, style=facet_style.corner_normal_base_style)
                view.draw_xyz_list(ray, style=facet_style.corner_normal_style)

        # Name.
        if facet_style.draw_name:
            view.draw_xyz_text(self.origin, self.name, style=facet_style.name_style)

        if facet_style.draw_mirror_curvature:
            self.mirror.draw(view, facet_style.mirror_styles)

    def survey_of_points(self, resolution, random_dist: bool = False) -> tuple[Pxyz, Vxyz]:
        return self.mirror.survey_of_points(resolution, random_dist)

    # override function from RayTraceable
    def most_basic_ray_tracable_objects(self) -> list[RayTraceable]:
        return self.mirror.most_basic_ray_tracable_objects()
