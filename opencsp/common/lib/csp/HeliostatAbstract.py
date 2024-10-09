"""
Heliostat Class



"""

import copy
import csv
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from sympy import Symbol, diff

import opencsp.common.lib.csp.sun_track as st
import opencsp.common.lib.geometry.transform_3d as t3d
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.tool.math_tools as mt
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.HeliostatConfiguration import HeliostatConfiguration
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.FunctionXYContinuous import FunctionXYContinuous
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlEnsemble import RenderControlEnsemble
from opencsp.common.lib.render_control.RenderControlHeliostat import RenderControlHeliostat
from opencsp.common.lib.tool.typing_tools import strict_types

UP = Vxyz([0, 0, 1])


class HeliostatAbstract(RayTraceable, OpticOrientationAbstract, ABC):
    """
    Heliostat representation.
    Parameters:
        -----------
        facet_ensemble: FacetEnsemble
            List of facets, in order from top-left to bottom-right (row major order).
            The facet names should be their position in the list (1-indexed).
    """

    def __init__(
        self,
        facet_ensemble: FacetEnsemble,
        name: str = None,
        #  center_facet: Facet | str = None,
        #  use_center_facet_for_aiming: bool = False
    ) -> None:

        self.name = name
        self.facet_ensemble = facet_ensemble
        OpticOrientationAbstract.__init__(self)

        # # center facet is an optional field that can be used sometimes to
        # # allow for more accurate heliostat aiming
        # if center_facet != None:
        #     self.set_center_facet(center_facet)
        # else:
        #     self.center_facet = None
        # self.use_center_facet_for_aiming = use_center_facet_for_aiming

        pass  # end of __init__

    @property
    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        """Returns bounding box aligned to XY axes in heliostat's coordinate
        reference frame.

        Returns
        -------
        tuple[float, float, float, float]
            Left, right, bottom, top. Heliostat's coordinate reference frame.
        """
        # Get XYZ locations of all points making up mirror region
        xyz = []  # facet frame
        for facet in self.facet_ensemble.facets:
            # Get all mirror region vertices
            points_xy = Pxy.merge([loop.vertices for loop in facet.mirror.region.loops])  # mirror frame
            points_z = facet.mirror.surface_displacement_at(points_xy)  # mirror frame
            points_xyz = Pxyz((points_xy.x, points_xy.y, points_z))  # mirror frame
            points_xyz = facet.mirror.get_transform_relative_to(self).apply(points_xyz)  # facet frame
            xyz.append(points_xyz)  # facet frame
        xyz = Pxyz.merge(xyz)  # facet frame

        # Find bounding box
        return xyz.x.min(), xyz.x.max(), xyz.y.min(), xyz.y.max()  # facet frame

    # override OpticOrientationAbstract

    @property
    def children(self):
        return [self.facet_ensemble]

    # override OpticOrientationAbstract
    def _add_child_helper(self, new_child: OpticOrientationAbstract):
        raise ValueError("Heliostat does not accept new children.")

    # POSITIONING

    @abstractmethod
    def movement_transform(self, config: HeliostatConfiguration) -> TransformXYZ:
        """
        Instantiable classes that inheret HeliostatAbstract are required
        to have a functiont that takes in input and returns a TransformationXYZ
        object. This function can be called by another function that better
        describes the motion, but this one needs to exist to make sure the
        child class is a proper heliostat.

        Parameters
        ----------
        config: HeliostatConfiguration
            The arguments required by the Heliostat instance.
            Should be the same type as `self.current_configuration`.

        Returns
        -------
        TransformXYZ
            The transformation of the heliostat given the input `*args`.

        Example
        -------
        ```python
        # override movement_transform in HeliostatAbstract
        def movement_transform(self, az_angle: float, el_angle: float):
            '''possible movement_transform for an oversimplified
            azimuth and elevation based heliostat.'''
            az_rotation = Rotation.from_euler('z', az_angle)
            transform_az = TransformXYZ.from_R(az_rotation)
            el_rotation = # rotate about the proper vector
            transform_el = TransformXYZ.from_R(el_rotation)
            composite_transform = transform_el * transform_az
            return composite_transform
        ```
        """
        ...

    @property
    @abstractmethod
    def current_configuration(self) -> HeliostatConfiguration:
        """
        A tuple of the values that define the current state of the heliostat.

        Must adhere to the following property:
        ```python
        heliostat: HeliostatAbstract  # in some state
        current = helistat.current_configuration
        heliostat.set_orientation(current) # this should not change the heliostat
        ```
        """
        ...

    @current_configuration.setter
    @abstractmethod
    def current_configuration(self, new_current_configuration: HeliostatConfiguration) -> None:
        """Updates the value of current_configuration."""
        ...

    @abstractmethod
    def from_pointing_vector_to_configuration(self, pointing_vector: Vxyz) -> HeliostatConfiguration:
        """
        Gets the configuration for the heliostat that would move the heliostat
        so it is facing the given direction.

        Parameter
        ---------
        `pointing_vector`: Vxyz
            A vector that represents a direction the heliostat is pointing.

        Returns
        -------
        tuple
            The configuration that contains values that would move the heliostat
            to point on the direction of `pointing_vector` if they were used as arguments
            in `set_orientation`.
        """
        ...

    def set_orientation(self, config: HeliostatConfiguration) -> None:
        """
        Uses the `movement_transform(self, *args)` function to
        set the `_self_to_parent_transform` transformation of the FacetEnsemble in heliostat.
        """
        current_position = TransformXYZ.from_V(self.facet_ensemble._self_to_parent_transform.V)
        self.current_configuration = config
        self.facet_ensemble._self_to_parent_transform = current_position * self.movement_transform(config)

    def set_orientation_from_pointing_vector(self, pointing_vector: Vxyz) -> None:
        """
        Sets the pointing direction of the Heliostat to be the direction given.
        Note that this function depends on the individual implmentation of each heliostat.
        """
        configuration = self.from_pointing_vector_to_configuration(pointing_vector)
        self.set_orientation(configuration)

    # @strict_types
    def set_tracking_configuration(self, aimpoint: Pxyz, location_lon_lat: Iterable, when_ymdhmsz: tuple):
        """
        Orients the facet ensemble to point at the aimpoint given a location and time.
        """
        # Heliostat centroid coordinates.
        # Coordinates are (x,z) center, z=0 is at torque tube height.
        heliostat_origin = self.self_to_global_tranformation.apply(Pxyz.origin())

        # Compute heliostat surface normal which tracks the sun to the aimpoint.
        pointing_vector = st.tracking_surface_normal_xyz(heliostat_origin, aimpoint, location_lon_lat, when_ymdhmsz)
        self.set_orientation_from_pointing_vector(pointing_vector)

        pass  # end of set_tracking_configuration

    # MODIFICATION

    def set_facet_positions(self, positions: Pxyz):
        self.facet_ensemble.set_facet_positions(positions)

    def set_facet_canting(self, canting_rotations: list[Rotation]):
        self.facet_ensemble.set_facet_canting(canting_rotations)

    # TODO TJL:make this work and make it faster
    def set_canting_from_equation(self, func: FunctionXYContinuous) -> None:
        """
        Uses an equation to set the canting of the facets on the heliostat.

        Parameters:
        -----------
        func: FunctionXYContinuous
            The function that is used to set the canting angles.
            The function is of the form z = f(x, y) and the surface normal at
            (x0, y0) is the direction that a facet at (x0, y0) will point.

        """
        # equation for canting angles
        x_s = Symbol('x')
        y_s = Symbol('y')

        sym_func = func(x_s, y_s)

        dfdx = diff(sym_func, x_s)
        dfdy = diff(sym_func, y_s)

        facet_canting_rotations: list[TransformXYZ] = []

        for fac in self.facet_ensemble.facets:
            fac_origin = fac.get_transform_relative_to(self.facet_ensemble).apply(Pxyz.origin())
            x = fac_origin.x[0]
            y = fac_origin.y[0]

            # # Set the new z coordinate of this facet.
            # if move_centriods:
            #     ...

            dfdx_n = dfdx.subs([(x_s, x), (y_s, y)])
            dfdy_n = dfdy.subs([(x_s, x), (y_s, y)])

            if (dfdx_n == 0.0) and (dfdy_n == 0.0):
                # Then the surface normal is vertical, and there is no rotation.
                canting = Rotation.identity()

            else:
                # gradient of the surface
                surf_normal = -Uxyz([dfdx_n, dfdy_n, -1.0])
                UP = Uxyz([0, 0, 1])
                canting = UP.align_to(surf_normal)

            facet_canting_rotations.append(canting)

        self.facet_ensemble.set_facet_canting(facet_canting_rotations)

        return facet_canting_rotations

    # RENDERING

    def draw(self, view: View3d, heliostat_style: RenderControlHeliostat = None, transform: TransformXYZ = None):
        """
        Draws heliostat onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A View3d object that holds the figure.
        heliostat_style : RenderControlHeliostat
            Holds information on how to draw the heliostat
            and the objects that make up the heliostat.
        transform : TransformXYZ | None
            List of 3d transforms for each facet in ensemble.
            Used to position points in the Heliostat's base coordinate
            reference frame in space. If None, defaults to position points
            in the heliostat's global coordinate reference frame.
        """

        if heliostat_style is None:
            heliostat_style = RenderControlHeliostat()

        if transform is None:
            transform = self.self_to_global_tranformation

        origin = transform.apply(Pxyz.origin())

        # TODO TJL:do we want a default style?
        if heliostat_style is None:
            heliostat_style = rch.default()

        # Centroid.
        if heliostat_style.draw_centroid:
            view.draw_single_Pxyz(origin, style=heliostat_style.centroid_style)

        # # Outline.
        # if heliostat_style.draw_outline:
        #     left, right, top, bottom = self.axis_aligned_bounding_box
        #     corners = Pxyz([[left, left, right, right],
        #                    [top, bottom, bottom, top],
        #                    [0, 0, 0, 0]])
        #     corners_moved = transform.apply(corners)
        #     view.draw_Vxyz(corners_moved, close=True, style=heliostat_style.outline_style)

        # # Surface normal.
        # if heliostat_style.draw_surface_normal:
        #     # Construct ray.
        #     self.facet_ensemble.
        #     surface_normal_ray = transform.apply(UP * heliostat_style.corner_normal_length)
        #     # Draw ray and its base.
        #     view.draw_Vxyz(Vxyz.merge([origin, surface_normal_ray]),
        #                    close=False,
        #                    style=heliostat_style.surface_normal_style)

        # Facet Ensemble
        if heliostat_style.draw_facet_ensemble:
            facet_ensemble_transform = transform * self.facet_ensemble._self_to_parent_transform
            self.facet_ensemble.draw(view, heliostat_style.facet_ensemble_style, facet_ensemble_transform)

        # vertical post
        if heliostat_style.post != 0:
            DOWN = Vxyz([0, 0, -heliostat_style.post])
            direction = transform.apply(DOWN)
            view.draw_Vxyz(Vxyz.merge([origin + DOWN, origin]))

        # Name.
        if heliostat_style.draw_name:
            view.draw_xyz_text(origin.data.T[0], self.name, style=heliostat_style.name_style)

        # # Facets.
        # if heliostat_style.draw_facets:
        #     for facet in self.facets:
        #         facet.draw(view, heliostat_style.facet_styles)

        # # Surface normal drawn at corners.
        # # (Not the surface normal at the corner.  Facet curvature is not shown.)
        # if heliostat_style.draw_surface_normal_at_corners:
        #     # Construct rays.
        #     top_left_ray = self.surface_normal_ray(self.top_left_corner, heliostat_style.corner_normal_length)
        #     top_right_ray = self.surface_normal_ray(self.top_right_corner, heliostat_style.corner_normal_length)
        #     bottom_left_ray = self.surface_normal_ray(self.bottom_left_corner, heliostat_style.corner_normal_length)
        #     bottom_right_ray = self.surface_normal_ray(self.bottom_right_corner, heliostat_style.corner_normal_length)
        #     rays = [top_left_ray,
        #             top_right_ray,
        #             bottom_left_ray,
        #             bottom_right_ray]
        #     # Draw each ray and its base.
        #     for base, ray in zip(corners, rays):
        #         view.draw_xyz(base, style=heliostat_style.corner_normal_base_style)
        #         view.draw_xyz_list(ray, style=heliostat_style.corner_normal_style)

        # # Name.
        # if heliostat_style.draw_name:
        #     view.draw_xyz_text(self.origin, self.name, style=heliostat_style.name_style)

        pass  # end of draw

    # ACCESS

    def lookup_facet(self, facet_name: str):
        """Returns the first Facet in the Helisotat's FacetEnsemble that matches the given name.
        If there are no facets that match the given name it throws a KeyError."""
        return self.facet_ensemble.lookup_facet(facet_name)

    # @property
    # def aimpoint_xyz(self):
    #     if self._aimpoint_xyz == None:
    #         print('ERROR: In Heliostat.aimpoint_xyz(), attempt to fetch unset _aimpoint_xyz.')
    #         assert False
    #     return self._aimpoint_xyz

    # @property
    # def when_ymdhmsz(self):
    #     if self._when_ymdhmsz == None:
    #         print('ERROR: In Heliostat.when_ymdhmsz(), attempt to fetch unset _when_ymdhmsz.')
    #         assert False
    #     return self._when_ymdhmsz

    # RENDERING

    def survey_of_points(self, resolution: Resolution) -> tuple[Pxyz, Vxyz]:
        # Get sample point locations (z=0 plane in "child" reference frame)
        resolution.resolve_in_place(self.axis_aligned_bounding_box)
        return self._survey_of_points_helper(resolution, TransformXYZ.identity())

    def _survey_of_points_helper(
        self, given_resolution: Resolution, frame_transform: TransformXYZ
    ) -> tuple[Pxyz, Vxyz]:
        resolution = given_resolution.change_frame_and_copy(frame_transform)
        resolution.resolve_in_place(self.axis_aligned_bounding_box)
        points, normals = [], []
        facet_points, facet_normals = self.facet_ensemble._survey_of_points_helper(
            resolution, self.facet_ensemble._self_to_parent_transform.inv()
        )
        return (facet_points, facet_normals)


# GENERATOR


# # TODO TJL:update this to be a class method
# def h_from_facet_centroids(name: str, origin: np.ndarray, num_facets: int, num_rows: int = 0, num_cols: int = 0,
#                            file_centroids_offsets=None, pivot_height: float = 0, pivot_offset: float = 0,
#                            facet_width: float = 0, facet_height: float = 0, default_mirror_shape: Callable[[float, float], float] = lambda x, y: x * 0) -> 'Heliostat':

#     # Facets
#     facets, _ = facets_read_file(file_centroids_offsets, facet_height, facet_width, default_mirror_shape)

#     return Heliostat(name, origin, num_facets, num_rows, num_cols, facets, pivot_height, pivot_offset)


# # TODO TJL:update this to be a class method
# def facets_read_file(file: str, facet_width: float, facet_height: float,
#                      default_mirror_shape: Callable[[float, float], float] = lambda x, y: 0 * x * y) -> tuple[list[Facet], dict[str, Facet]]:
#     """ Read in a list of facets (and their relative positions on a standard heliostat) from a csv file.

#     Arguments:
#         file: one facet per row, with the format:
#             name, x, y, z """

#     with open(file) as csvfile:
#         readCSV = csv.reader(csvfile, delimiter=',')
#         id_row = 0
#         id_facet = 0
#         facets: list[Facet] = []
#         facet_dict: dict[str, Facet] = {}
#         for row in readCSV:
#             if not id_row:
#                 id_row += 1  # ignore the header
#                 continue
#             name = str(row[0])
#             x, y, z = (float(row[1]), float(row[2]), float(row[3]))

#             # creating facet and mirror
#             fn = default_mirror_shape  # flat surface
#             mirror = MirrorParametricRectangular(fn, (facet_width, facet_height))
#             facet = Facet(name=name, centroid_offset=[x, y, z],
#                           mirror=mirror)

#             # storing
#             facets.append(facet)
#             facet_dict[name] = id_facet
#             id_facet += 1

#     return facets, facet_dict
