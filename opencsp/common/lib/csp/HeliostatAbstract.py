"""Heliostat Class"""

from abc import ABC, abstractmethod
from typing import Iterable

from scipy.spatial.transform import Rotation
from sympy import Symbol, diff

import opencsp.common.lib.csp.sun_track as st
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.HeliostatConfiguration import HeliostatConfiguration
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
from opencsp.common.lib.render_control.RenderControlHeliostat import RenderControlHeliostat

UP = Vxyz([0, 0, 1])


class HeliostatAbstract(RayTraceable, OpticOrientationAbstract, ABC):
    """
    Heliostat representation.

    Parameters
    ----------
    facet_ensemble: FacetEnsemble
        List of facets, in order from top-left to bottom-right (row major order).
        The facet names should be their position in the list (1-indexed).
    """

    def __init__(self, facet_ensemble: FacetEnsemble, name: str = None) -> None:

        self.name = name
        self.facet_ensemble = facet_ensemble
        OpticOrientationAbstract.__init__(self)

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

    @property
    def children(self):
        return [self.facet_ensemble]

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
        .. code-block:: python

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

    @current_configuration.setter
    @abstractmethod
    def current_configuration(self, new_current_configuration: HeliostatConfiguration) -> None:
        """Updates the value of current_configuration."""

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

    def set_orientation(self, config: HeliostatConfiguration) -> None:
        """
        Uses the `movement_transform(self, *args)` function to
        set the `_self_to_parent_transform` transformation of the FacetEnsemble in heliostat.
        """
        self.current_configuration = config
        self.facet_ensemble._self_to_parent_transform = self.movement_transform(config)

    def set_orientation_from_pointing_vector(self, pointing_vector: Vxyz) -> None:
        """
        Sets the pointing direction of the Heliostat to be the direction given.
        Note that this function depends on the individual implmentation of each heliostat.
        """
        configuration = self.from_pointing_vector_to_configuration(pointing_vector)
        self.set_orientation(configuration)

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

    # MODIFICATION

    def set_facet_positions(self, positions: Pxyz):
        self.facet_ensemble.set_facet_positions(positions)

    def set_facet_cantings(self, canting_rotations: list[Rotation]):
        self.facet_ensemble.set_facet_cantings(canting_rotations)

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
        x_s = Symbol("x")
        y_s = Symbol("y")

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

        self.facet_ensemble.set_facet_cantings(facet_canting_rotations)

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
            origin.draw_points(view, style=heliostat_style.centroid_style)

        # Facet Ensemble
        if heliostat_style.draw_facet_ensemble:
            facet_ensemble_transform = transform * self.facet_ensemble._self_to_parent_transform
            self.facet_ensemble.draw(view, heliostat_style.facet_ensemble_style, facet_ensemble_transform)

        # vertical post
        if heliostat_style.post != 0:
            DOWN = Vxyz([0, 0, -heliostat_style.post])
            Vxyz.merge([origin + DOWN, origin]).draw_line(view)

        # Name.
        if heliostat_style.draw_name:
            view.draw_xyz_text(origin.data.T[0], self.name, style=heliostat_style.name_style)

    # ACCESS

    def lookup_facet(self, facet_name: str):
        """Returns the first Facet in the Helisotat's FacetEnsemble that matches the given name.
        If there are no facets that match the given name it throws a KeyError."""
        return self.facet_ensemble.lookup_facet(facet_name)

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
        facet_points, facet_normals = self.facet_ensemble._survey_of_points_helper(
            resolution, self.facet_ensemble._self_to_parent_transform.inv()
        )
        return (facet_points, facet_normals)
