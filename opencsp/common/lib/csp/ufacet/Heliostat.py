"""
Heliostat Class



"""

import copy
import csv
import math
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation
from sympy import Symbol, diff

import opencsp.common.lib.csp.ufacet.HeliostatConfiguration as hc
import opencsp.common.lib.csp.sun_track as st
import opencsp.common.lib.geometry.transform_3d as t3d
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.tool.math_tools as mt
from opencsp.common.lib.csp.ufacet.Facet import Facet
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlEnsemble import RenderControlEnsemble
from opencsp.common.lib.render_control.RenderControlHeliostat import RenderControlHeliostat
from opencsp.common.lib.tool.typing_tools import strict_types


class Heliostat(RayTraceable):
    """
    Heliostat (AKA Mirror Array) representation. This should be a 5x5 grid of facets for NSTTF.
    """

    def __init__(
        self,
        name: str,
        origin: np.ndarray,
        num_facets: int,
        num_rows: int = 1,
        num_cols: int = 1,
        facets: list[Facet] = None,
        pivot_height: float = 0,
        pivot_offset: float = 0,
        relative_facet_positions: Pxyz = None,
        center_facet: Facet | str = None,
        use_center_facet_for_aiming: bool = False,
    ):  # maybe make this default to true
        """Create a new heliostat instance.

        Parameters:
        -----------
            name    The name of this heliostat. Used for special styles given in the draw method.

            origin  The torque tube location for this heliostat, as a three vector xyz coordinate

            facets  List of facets, in order from top-left to bottom-right (row major order).
                    The facet names should be their position in the list (1-indexed).
        """
        # super(Heliostat, self).__init__()

        # setting default values
        if facets == None:
            facets = []
        if relative_facet_positions == None:
            relative_facet_positions = Pxyz.empty()

        # value checks
        # if len(facets) != len(relative_facet_positions):
        #     raise ValueError(f"number of facets must equal the number of facet positions. "
        #                      f"{len(facets)} facets wer given and {len(relative_facet_positions)} facet positions")

        # setting class variables
        self.name = name
        self.origin = origin
        # self.num_rows = num_rows
        # self.num_cols = num_cols
        self.num_facets = num_facets
        # self.ncenterfacet = (self.num_facets + 1) / 2
        self.pivot_height = pivot_height
        self.pivot_offset = pivot_offset

        # Validate the input
        # if (__debug__) and (self.num_facets != self.num_rows * self.num_cols):
        #     raise AssertionError("num_facets != num_rows * num_cols... is this correct? if so, then please remove this assertion") # TODO remove "num_facets" arg if this assertion is correct

        # Facets
        self.facets = facets
        self.relative_facet_positions = relative_facet_positions
        self.facet_dict = {
            f.name: f_i for f_i, f in enumerate(facets)
        }  # TODO tjlarki: check with RCB that this is what we want

        # center facet is an optional field that can be used sometimes to
        # allow for more accurate heliostat aiming
        if center_facet != None:
            self.set_center_facet(center_facet)
        else:
            self.center_facet = None
        self.use_center_facet_for_aiming = use_center_facet_for_aiming

        # Heliostat Edge Facets
        self.top_left_facet: Facet = self.facets[self.facet_dict[str(1)]]
        self.top_right_facet: Facet = self.facets[self.facet_dict[str(num_cols)]]
        self.bottom_right_facet: Facet = self.facets[self.facet_dict[str(num_facets)]]
        self.bottom_left_facet: Facet = self.facets[self.facet_dict[str(num_facets - num_cols + 1)]]

        # Heliostat Corners [offsets in terms of heliostat's centroid]
        self.top_left_corner_offset = [
            x + y for x, y in zip(self.top_left_facet.centroid_offset, self.top_left_facet.top_left_corner_offset)
        ]
        self.top_right_corner_offset = [
            x + y for x, y in zip(self.top_right_facet.centroid_offset, self.top_right_facet.top_right_corner_offset)
        ]
        self.bottom_right_corner_offset = [
            x + y
            for x, y in zip(self.bottom_right_facet.centroid_offset, self.bottom_right_facet.bottom_right_corner_offset)
        ]
        self.bottom_left_corner_offset = [
            x + y
            for x, y in zip(self.bottom_left_facet.centroid_offset, self.bottom_left_facet.bottom_left_corner_offset)
        ]

        # Centroid
        self.origin = np.array([origin[0], origin[1], origin[2]])  # Origin is at torque tube center.

        self.az = np.deg2rad(180)  # (az,el) = (180,90) degrees corresponds to pointing straight up,
        self.el = np.deg2rad(90)  # as if transitioned by tilting up from face south orientation.
        self.surface_normal = [0, 0, 1]  #
        self.rx_rotation = np.identity(3)
        self.rz_rotation = np.identity(3)
        self.rotation = Rotation.identity()  # Rz_rotation.dot(Rx_rotation)
        # self._set_corner_positions_in_space()

        # Tracking
        self._aimpoint_xyz = (
            None  # (x,y,y) in m. Do not access this member externally; use aimpoint_xyz() function instead.
        )
        self._when_ymdhmsz = (
            None  # (y,m,d,h,m,s,z). Do not access this member externally; use when_ymdhmsz() function instead.
        )

        # SET POSITION IN SPACE
        self.set_position_in_space(self.origin, self.rotation)

    # ACCESS

    @property
    def aimpoint_xyz(self):
        if self._aimpoint_xyz == None:
            print('ERROR: In Heliostat.aimpoint_xyz(), attempt to fetch unset _aimpoint_xyz.')
            assert False
        return self._aimpoint_xyz

    @property
    def when_ymdhmsz(self):
        if self._when_ymdhmsz == None:
            print('ERROR: In Heliostat.when_ymdhmsz(), attempt to fetch unset _when_ymdhmsz.')
            assert False
        return self._when_ymdhmsz

    def surface_normal_ray(self, base, length):
        # Constructs the head and tail of a vector of given length, placed at the base
        # position (computed after applying the heliostat configuration).
        #
        # This assumes that set_configuration() has already been called for
        # the current heliostat configuration.  This is required to set the internal
        # surface_normal.
        tail = base
        head = tail + (length * np.array(self.surface_normal))
        ray = [tail, head]
        return ray

    def compute_tracking_configuration(
        self, aimpoint_xyz: list | np.ndarray, location_lon_lat: tuple | list, when_ymdhmsz: tuple
    ):
        if self.use_center_facet_for_aiming:
            if self.center_facet == None:
                raise AttributeError(f"Helisotat (Name: {self.name}) does not have a center facet defined")

            # TODO tjlarki: centroid_offset needs to become a Pxyz eventually
            d: float = self.center_facet.centroid_offset[2]

            # Heliostat centroid coordinates.
            # Coordinates are (x,z) center, z=0 is at torque tube height.
            h_tube = self.origin
            h = Pxyz(h_tube)  # Later, add correction for center facet offset.

            # Compute heliostat surface normal which tracks the sun to the aimpoint.
            n = Vxyz(st.tracking_surface_normal_xyz(h_tube, aimpoint_xyz, location_lon_lat, when_ymdhmsz)).normalize()

            # iteratively find the normal vectors that actually
            # take into account the offset to the center facet
            for _ in range(10):
                n = Vxyz(
                    st.tracking_surface_normal_xyz(
                        (h + n * d).data.T.flatten(), aimpoint_xyz, location_lon_lat, when_ymdhmsz
                    )
                ).normalize()

            # Compute heliostat configuration.
            return hc.heliostat_configuration_given_surface_normal_xyz(n.data.T.flatten())
        else:
            # Heliostat centroid coordinates.
            # Coordinates are (x,z) center, z=0 is at torque tube height.
            h_tube = np.array(self.origin)
            h = h_tube  # Later, add correction for center facet offset.

            # Compute heliostat surface normal which tracks the sun to the aimpoint.
            n_xyz = st.tracking_surface_normal_xyz(h, aimpoint_xyz, location_lon_lat, when_ymdhmsz)

            # Compute heliostat configuration.
            return hc.heliostat_configuration_given_surface_normal_xyz(n_xyz)

    # @strict_types
    def compute_tracking_configuration_from_sun_vector(
        self, aimpoint_xyz: list | np.ndarray, sun_vector: Vxyz
    ) -> hc.HeliostatConfiguration:
        """!!! DOES NOT WORK !!!

        Similar to compute_tracking_configuration() but takes an arbitrary center sun vector
        instead of computing it from the time.

        Parameters
        ----------
        sun_vector: Vxyz
            The vector from the center of the sun.
        """
        if self.center_facet == None:
            raise AttributeError(f"Helisotat (Name: {self.name}) does not have a center facet defined")

        # TODO tjlarki: centroid_offset needs to become a Pxyz eventually
        d: float = self.center_facet.centroid_offset[2]

        # Heliostat centroid coordinates.
        # Coordinates are (x,z) center, z=0 is at torque tube height.
        h_tube = self.origin
        h = Pxyz(h_tube)  # Later, add correction for center facet offset.

        # Compute heliostat surface normal which tracks the sun to the aimpoint.
        n = Vxyz(st.tracking_surface_normal_xyz_given_sun_vector(h_tube, aimpoint_xyz, sun_vector)).normalize()

        # iteratively find the normal vectors that actually
        # take into account the offset to the center facet
        for _ in range(10):
            n = Vxyz(
                st.tracking_surface_normal_xyz_given_sun_vector((h + n * d).data.T.flatten(), aimpoint_xyz, sun_vector)
            ).normalize()

        # Compute heliostat configuration.
        return hc.heliostat_configuration_given_surface_normal_xyz(n.data.T.flatten())

    def compute_stow_configuration(self):
        # ?? TODO RCB -- MAKE THIS SENSITIVE TO INPUT DEFINITION.
        NSTTF = True
        if NSTTF:
            azimuth = np.deg2rad(270.0)  # ?? SCAFFOLDING RCB -- FIND OUT THE CORRECT NUMBER FOR THIS.
            elevation = np.deg2rad(-85.0)  # ?? SCAFFOLDING RCB -- FIND OUT THE CORRECT NUMBER FOR THIS.
        else:
            # ?? TODO RCB -- FOR NOW, ASSUME RADIAL STOW.  MAKE CONTROLLABLE.
            origin_x = self.origin[0]
            origin_y = self.origin[1]
            theta = math.atan2(-origin_y, -origin_x)
            azimuth = ((5.0 / 2.0) * math.pi) - theta
            if azimuth > (2.0 * math.pi):
                azimuth -= 2.0 * math.pi
            elevation = np.deg2rad(90.0)
        # Return.
        return hc.HeliostatConfiguration(az=azimuth, el=elevation)

    def compute_face_up_configuration(self):
        # ?? TODO RCB -- MAKE THIS SENSITIVE TO INPUT DEFINITION.
        NSTTF = True
        if NSTTF:
            azimuth = np.deg2rad(180.0)  # ?? SCAFFOLDING RCB -- FIND OUT THE CORRECT NUMBER FOR THIS.
            elevation = np.deg2rad(90.0)  # ?? SCAFFOLDING RCB -- FIND OUT THE CORRECT NUMBER FOR THIS.
        else:
            # ?? TODO RCB -- FOR NOW, ASSUME RADIAL STOW.  MAKE CONTROLLABLE.
            origin_x = self.origin[0]
            origin_y = self.origin[1]
            theta = math.atan2(-origin_y, -origin_x)
            azimuth = ((5.0 / 2.0) * math.pi) - theta
            if azimuth > (2.0 * math.pi):
                azimuth -= 2.0 * math.pi
            elevation = np.deg2rad(90.0)
        # Return.
        return hc.HeliostatConfiguration(az=azimuth, el=elevation)

    def corners(self):
        """Returns the list of corners in ul,ur,lr,ll order."""
        # Assumes that heliostat coordinates have been set, and the corners have been set.
        # Later we can add a more meaningful check for this.
        return [self.top_left_corner, self.top_right_corner, self.bottom_right_corner, self.bottom_left_corner]

    def get_configuration(self):
        return hc.HeliostatConfiguration(self.az, self.el)

    # CANTING MODIFICATION

    # TODO tjlarki: clean up surface normal code
    def set_canting_from_equation(self, func: Callable[[float, float], float], move_centriods: bool = False) -> None:
        """
        Uses an equation to set the canting of the facets on the heliostat.

        Parameters:
        -----------
            func            The function that is used to set the canting angles. Fuction origin and facet origin should be lined up.

            move_centroids  Should the facet offsets be set to the points on the function. False does nothing to the centroid offsets of the facets.

        """
        for fac in self.facets:
            x = fac.centroid_offset[0]
            y = fac.centroid_offset[1]

            # Set the new z coordinate of this facet.
            if move_centriods:
                fac.centroid_offset[2] = func(x, y)

            # Set the canting rotation for each facet.
            x_s = Symbol('x')
            y_s = Symbol('y')

            sym_func = func(x_s, y_s)

            dfdx = diff(sym_func, x_s)
            dfdy = diff(sym_func, y_s)

            dfdx_n = dfdx.subs([(x_s, x), (y_s, y)])
            dfdy_n = dfdy.subs([(x_s, x), (y_s, y)])

            # Check for the no-rotation case.  This is redundant with the check in the function
            # rodrigues_vector_from_partial_derivatives() below.
            # We could rely on that function by alling it and then checking its result for a zero norm,
            # but checking here allows us to avoid calling that function altogether.
            if (dfdx_n == 0.0) and (dfdy_n == 0.0):
                # Then the surface normal is vertical, and there is no rotation.
                fac.canting = Rotation.identity()

            else:
                # There is a rotation.
                # Make a Rodrigues vector.
                rotvec = self.rodrigues_vector_from_partial_derivatives(dfdx_n, dfdy_n)
                # Set the facet canting rotation.
                rotvec_norm = np.linalg.norm(rotvec)
                if rotvec_norm == 0:
                    fac.canting = Rotation.identity()
                else:
                    fac.canting = Rotation.from_rotvec(rotvec)

            self.set_position_in_space(self.origin, self.rotation)

    # TODO tjlarki: currently basis location off helisotat origin
    # @strict_types
    def set_on_axis_canting(self, aimpoint: Pxyz):
        focal_length = (aimpoint - Pxyz(self.origin)).magnitude()[0]
        fn = mt.lambda_symmetric_paraboloid(focal_length)
        self.set_canting_from_equation(fn)

        # # we need the Rotation R1 (canting rotation)
        # # we will call the heliostat roation R2
        # # R2 * R1 = R (total rotation)
        # # if we first rotate the helisotat then find the resulting canting required to get R
        # # then we are using the equation R1' * R2 = R, which means we are
        # # not getting the canting angle we want. Therefore we can
        # # set our two left side equal: R2 * R1 = R1' * R2
        # # where we know R2 and R1' and want R1
        # # R2 * R1 = R1' * R2 => R1 = inverse(R2) * R1' * R2
        # # This is implemented below:
        # aimpoint_xyz = list(aimpoint[0].data.T.flatten())  # convert aimpoint Pxyz to np.array  # TODO: convert all vectors and points to Vxyz system
        # current_configuration = self.get_configuration()  # remember current config
        # if self.use_center_facet_for_aiming:
        #     reference_point = self.center_facet.origin
        # else:
        #     reference_point = self.origin
        # sun_vector = Vxyz(reference_point - aimpoint_xyz).normalize()

        # self.set_tracking_from_sun_vector(aimpoint_xyz, sun_vector)  # aim heliostat

        # up_vector = Vxyz([0, 0, 1])
        # self.update_position_in_space()
        # heliostat_aiming_direction = up_vector.rotate(self.rotation)  # find
        # self.flatten()
        # R2 = self.rotation
        # R2_inv = R2.inv()  # chaching inverse for speed
        # for facet in self.facets:
        #     # facet_aiming_direction = up_vector.rotate(facet.composite_rotation)
        #     vector_the_normal_should_be = Vxyz(st.tracking_surface_normal_xyz_given_sun_vector(facet.origin,
        #                                                                                        aimpoint_xyz,
        #                                                                                        sun_vector))
        #     R1_prime = heliostat_aiming_direction.align_to(vector_the_normal_should_be)
        #     facet.canting = copy.deepcopy(R2_inv * R1_prime * R2)
        # self.set_configuration(current_configuration)
        # self.update_position_in_space()

    def set_off_axis_canting(self, long_lat: tuple, aimpoint: Pxyz, time_ymdhmsz: tuple):
        # we need the Rotation R1 (canting rotation)
        # we will call the heliostat roation R2
        # R2 * R1 = R (total rotation)
        # if we first rotate the helisotat then find the resulting canting required to get R
        # then we are using the equation R1' * R2 = R, which means we are
        # not getting the canting angle we want. Therefore we can
        # set our two left side equal: R2 * R1 = R1' * R2
        # where we know R2 and R1' and want R1
        # R2 * R1 = R1' * R2 => R1 = inverse(R2) * R1' * R2
        # This is implemented below:
        aimpoint_xyz = list(
            aimpoint[0].data.T.flatten()
        )  # convert aimpoint Pxyz to np.array  # TODO: convert all vectors and points to Vxyz system
        current_configuration = self.get_configuration()  # remember current config
        self.set_tracking(aimpoint_xyz, long_lat, time_ymdhmsz)  # aim heliostat

        up_vector = Vxyz([0, 0, 1])
        self.update_position_in_space()
        heliostat_aiming_direction = up_vector.rotate(self.rotation)  # find
        self.flatten()
        R2 = self.rotation
        R2_inv = R2.inv()  # chaching inverse for speed
        for facet in self.facets:
            # facet_aiming_direction = up_vector.rotate(facet.composite_rotation)
            vector_the_normal_should_be = Vxyz(
                st.tracking_surface_normal_xyz(facet.origin, aimpoint_xyz, long_lat, time_ymdhmsz)
            )
            R1_prime = heliostat_aiming_direction.align_to(vector_the_normal_should_be)
            facet.canting = copy.deepcopy(R2_inv * R1_prime * R2)
        self.set_configuration(current_configuration)
        self.update_position_in_space()

    def rodrigues_vector_from_partial_derivatives(self, dfdx_n: float, dfdy_n: float) -> np.ndarray:
        """
        Constructs a Rodrigues vector from partial derivatives of a surface normal that is
        pointing nearly straight up.
        The returned Rodriguez vector describes the rotation from vertical to the surface normal.

        Parameters:
            dfdx_n  The partial derivative of the surface in the x direction.

            dfdy_n  The partial derivative of the surface in the y direction.

        """
        # Check for the no-rotation case.  This is redundant with the check in set_canting_from_equation()
        # above, but someone might call this function from somewhere else.
        if (dfdx_n == 0) and (dfdy_n == 0):
            # Then the surface normal is vertical, and there is no rotation.
            # TODO:  THIS SHOULD RETURN AN ACTUAL Vxyz.
            return np.array([0.0, 0.0, 0.0])  # Zero magnitude implies zero rotation; direction doesn't matter.

        else:
            # There is a rotation.
            x_vec = [1.0, 0.0, float(dfdx_n)]  # direction of x partial derivative
            y_vec = [0.0, 1.0, float(dfdy_n)]  # direction of y partial derivative

            v = np.cross(x_vec, y_vec)

            v_norm = np.linalg.norm(v)
            if v_norm == 0.0:
                # TODO RCB: REPLACE THIS WITH LOG/EXCEPTION THROW.
                print('ERROR: In Heliostat.set_canting_from_equation(), encountered unexpected zero v_norm.')
                print('         x_vec = ', x_vec)
                print('         y_vec = ', y_vec)
                assert False

            u = v / v_norm

            v_rot = np.cross(u, np.array([0.0, 0.0, 1.0]))  # axis of rotation

            v_rot_norm = np.linalg.norm(v_rot)
            if v_rot_norm == 0.0:
                # TODO RCB: REPLACE THIS WITH LOG/EXCEPTION THROW.
                print('ERROR: In Heliostat.set_canting_from_equation(), encountered unexpected zero v_rot_norm.')
                print('         u = ', u)
                print('         v_rot = ', v_rot)
                assert False

            u_rot = v_rot / v_rot_norm  # axis of rotation normalized

            theta_rot = -mt.robust_arccos(np.dot(u, np.array([0.0, 0.0, 1.0])))  # angle of rotation

            # TODO:  THIS SHOULD RETURN AN ACTUAL Vxyz.
            return theta_rot * u_rot

    def flatten(self) -> None:
        """Flattens canting. Sets Canting equations to z = x*0 + y*0"""

        def fn(x, y):
            return x * 0

        # fn = FunctionXYContinuous
        self.set_canting_from_equation(fn)

    # @strict_types
    def set_canting_from_list(self, canting_rotations: list[Rotation]) -> None:
        # TODO: THIS SHOULD BE EITHER IMPLEMENTED, OR DISCARDED.
        for facet, canting in zip(self.facets, canting_rotations):
            if canting == None:
                continue
            facet.canting = canting
        self.set_position_in_space(self.origin, self.rotation)

    # MODIFICATION

    # @strict_types
    def set_center_facet(self, facet: Facet | str):
        """Sets the self.center_facet attribute.
        takes a string for the name of the facet, or a facet object."""
        if isinstance(facet, Facet):
            facet = facet.name
        self.center_facet = self.facets[self.facet_dict[facet]]

    def set_tracking(self, aimpoint_xyz, location_lon_lat, when_ymdhmsz):
        # checks
        if len(location_lon_lat) != 2:
            raise ValueError(f'{location_lon_lat} must be of length 2. (Lattitude, Longitude)')
        # Save tracking command.
        self._aimpoint_xyz = aimpoint_xyz
        self._when_ymdhmsz = when_ymdhmsz
        # Set tracking configuration.
        h_config = self.compute_tracking_configuration(aimpoint_xyz, location_lon_lat, when_ymdhmsz)
        self.set_configuration(h_config, clear_tracking=False)

    # @strict_types
    def set_tracking_from_sun_vector(self, aimpoint_xyz, sun_vector: Vxyz):
        sun_vector = sun_vector.normalize()
        # Save tracking command.
        self._aimpoint_xyz = aimpoint_xyz
        # Set tracking configuration.
        h_config = self.compute_tracking_configuration_from_sun_vector(aimpoint_xyz, sun_vector)
        self.set_configuration(h_config, clear_tracking=False)

    def set_stow(self):
        h_config = self.compute_stow_configuration()
        self.set_configuration(h_config, clear_tracking=True)

    def set_face_up(self):
        h_config = self.compute_face_up_configuration()
        self.set_configuration(h_config, clear_tracking=True)

    def set_configuration(self, h_config: hc.HeliostatConfiguration, clear_tracking=True):
        # TODO is this function safe to call multiple times? For example to update with small sun tracking changes.
        # Clear tracking command.
        if clear_tracking:
            self._aimpoint_xyz = None
            self._when_ymdhmsz = None
        # Fetch azimuth and elevation parameters.
        el = h_config.el
        az = h_config.az

        # The heliostat begins face up, with its final link z axis pointing straight up.
        # To rotate it ot teh desired az,el) configuration, we first rotate it about the
        # x axis, and then about the z axis.  These angles are right-hand-rule rotations
        # about these axes.
        #
        #   1. For example, to make the heliostat point to an elevation of 30 degrees, we rotate
        #      60 degrees about the x axis.
        #
        #   2. As another example, to achieve a final configuration where the heliostat is vertical,
        #      facing south, we rotate 90 degrees about the x axis, and do not rotate about the z axis.
        #
        #   3. As a third example, to achieve a final configuration where the heliostat is vertical,
        #      facing east, we rotate 90 degrees about the x axis, and then rotate 90 degrees about the z axis.
        #
        # Meanwhile, azimuth and elevation are defined differently.  Elevation is measured up from horizontal,
        # and azimuth is measured clockwise from north (compass headings).  These two lines convert these (az,el)
        # conventions into the proper rotation angles about the x and z axes.
        #
        hel_rotation = rotation_from_az_el(az, el)

        vector = np.array([0, 0, self.pivot_offset])
        vector_offset = hel_rotation.apply(vector)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TODO tjlarki: Line below is confusing to me
        origin = np.array(self.origin) + vector_offset  # Origin is at torque tube center.

        self.surface_normal = hel_rotation.apply([0, 0, 1])  # Before rotation, heliostat is face up.

        self.el = el
        self.az = az
        self.origin = origin  # ?? SCAFFOLDING RCB -- THESE LIST TO ARRAY AND BACK TYPE CONVERSIONS ARE INEFFICIENT.  RESOLVE THIS.
        # self.surface_normal = list(surface_normal)         # ?? SCAFFOLDING RCB -- THESE LIST TO ARRAY AND BACK TYPE CONVERSIONS ARE INEFFICIENT.  RESOLVE THIS.
        # self.rx_rotation      = Rx_rotation
        # self.rz_rotation      = Rz_rotation
        self.rotation = hel_rotation

        self.set_position_in_space(self.origin, self.rotation)

    def _set_corner_positions_in_space(self):
        """Updates corner positions given this heliostat's configuration."""
        self.top_left_corner = self.transform_xyz(self.top_left_corner_offset)
        self.top_right_corner = self.transform_xyz(self.top_right_corner_offset)
        self.bottom_right_corner = self.transform_xyz(self.bottom_right_corner_offset)
        self.bottom_left_corner = self.transform_xyz(self.bottom_left_corner_offset)

    def transform_xyz(self, xyz):
        """
        Accepts an [x,y,z] list, and returns a numpy array.  # ?? SCAFFOLDING RCB -- THIS IS ONE OF MANY PLACES WHERE EFFICIENCY COULD IMPROVE.  FOR EXAMPLE, WE COULD WORK WITH ARRAYS INSTEAD OF COORDINATE LISTS.  A CLASS OBJECT INSTEAD?
        """
        transformed_xyz = self.origin + self.rotation.apply(np.array(xyz))
        return transformed_xyz

    # override from RayTracable
    # @strict_types
    def set_position_in_space(self, location: np.ndarray, rotation: Rotation) -> None:
        self.origin = location  # + self.pivot_offset
        self.rotation = rotation
        for facet in self.facets:
            facet.set_position_in_space(location, rotation)
            # facet.set_position_in_space(location + self.pivot_offset, rotation)

        self._set_corner_positions_in_space()  # specifically orients the corners of the heliostat

    def _modification_check(self, func):  # TODO tjlarki: finish
        """Decorator that can runs self.set_position_in_space(self.origin, self.rotation)
        after any function decorated with it."""
        ...

    def update_position_in_space(self):  # TODO tjlarki: should this be in the RayTraceable object?
        """updates the heliostat position in space, and all RayTracables
        contained in self"""
        self.set_position_in_space(self.origin, self.rotation)

    # override function from RayTraceable
    def most_basic_ray_tracable_objects(self) -> list[RayTraceable]:
        basic_objects: RayTraceable = []
        for facet in self.facets:
            basic_objects += facet.most_basic_ray_tracable_objects()
        return basic_objects

    # RENDERING

    def draw(self, view: View3d, heliostat_styles: RenderControlHeliostat | RenderControlEnsemble = None):
        # Assumes that heliostat configuration has already been set.
        if heliostat_styles == None:
            heliostat_styles = RenderControlEnsemble(default_style=rch.mirror_surfaces())

        # Fetch draw style control.
        heliostat_style = heliostat_styles.style(self.name)

        # Centroid.
        if heliostat_style.draw_centroid:
            view.draw_xyz(self.origin, style=heliostat_style.centroid_style)

        # Outline.
        if heliostat_style.draw_outline:
            corners = [self.top_left_corner, self.top_right_corner, self.bottom_right_corner, self.bottom_left_corner]
            view.draw_xyz_list(corners, close=True, style=heliostat_style.outline_style)

        # Facets.
        if heliostat_style.draw_facets:
            for facet in self.facets:
                facet.draw(view, heliostat_style.facet_styles)

        # Surface normal.
        if heliostat_style.draw_surface_normal:
            # Construct ray.
            surface_normal_ray = self.surface_normal_ray(self.origin, heliostat_style.surface_normal_length)
            # Draw ray and its base.
            view.draw_xyz(self.origin, style=heliostat_style.surface_normal_base_style)
            view.draw_xyz_list(surface_normal_ray, style=heliostat_style.surface_normal_style)

        # Surface normal drawn at corners.
        # (Not the surface normal at the corner.  Facet curvature is not shown.)
        if heliostat_style.draw_surface_normal_at_corners:
            # Construct rays.
            top_left_ray = self.surface_normal_ray(self.top_left_corner, heliostat_style.corner_normal_length)
            top_right_ray = self.surface_normal_ray(self.top_right_corner, heliostat_style.corner_normal_length)
            bottom_left_ray = self.surface_normal_ray(self.bottom_left_corner, heliostat_style.corner_normal_length)
            bottom_right_ray = self.surface_normal_ray(self.bottom_right_corner, heliostat_style.corner_normal_length)
            rays = [top_left_ray, top_right_ray, bottom_left_ray, bottom_right_ray]
            # Draw each ray and its base.
            for base, ray in zip(corners, rays):
                view.draw_xyz(base, style=heliostat_style.corner_normal_base_style)
                view.draw_xyz_list(ray, style=heliostat_style.corner_normal_style)

        # Name.
        if heliostat_style.draw_name:
            view.draw_xyz_text(self.origin, self.name, style=heliostat_style.name_style)

    def survey_of_points(self, resolution, random_dist: bool = False) -> tuple[Pxyz, Vxyz]:
        """
        Returns a grid of equispaced points and the normal vectors at those points.

        Parameters
        ----------
        resolution:
            the rectangular resolution of the points gathered (add other forms later, like triangular or polar survey).

        Returns
        -------
            a tuple of the points (Pxyz) and normals at the respective points (Vxyz).

        """
        points = Pxyz.empty()
        normals = Vxyz.empty()
        for facet in self.facets:
            additional_points, additional_normals = facet.survey_of_points(resolution, random_dist)
            points = points.concatenate(additional_points)
            normals = normals.concatenate(additional_normals)

        return (points, normals)


# HELPER FUNCTIONS


def rotation_from_az_el(azimuth: float, elevation: float) -> Rotation:
    rotation_about_x = (np.pi / 2) - elevation
    rotation_about_z = np.pi - azimuth

    # TODO tjlarki: depricated approach, identical
    # Rx_rotation = t3d.axisrotation(np.array([1, 0, 0]), rotation_about_x)
    # Rz_rotation = t3d.axisrotation(np.array([0, 0, 1]), rotation_about_z)
    # hel_rotation: np.ndarray = np.dot(Rz_rotation, Rx_rotation)

    Rx_rotation = Rotation.from_euler('x', rotation_about_x)
    Rz_rotation = Rotation.from_euler('z', rotation_about_z)
    hel_rotation: Rotation = Rz_rotation * Rx_rotation  # composed rotation
    return hel_rotation


# GENERATOR


# TODO tjlarki: update this to be a class method
def h_from_facet_centroids(
    name: str,
    origin: np.ndarray,
    num_facets: int,
    num_rows: int = 0,
    num_cols: int = 0,
    file_centroids_offsets=None,
    pivot_height: float = 0,
    pivot_offset: float = 0,
    facet_width: float = 0,
    facet_height: float = 0,
    default_mirror_shape: Callable[[float, float], float] = lambda x, y: x * 0,
) -> 'Heliostat':
    # Facets
    facets, _ = facets_read_file(file_centroids_offsets, facet_height, facet_width, default_mirror_shape)

    return Heliostat(name, origin, num_facets, num_rows, num_cols, facets, pivot_height, pivot_offset)


# TODO tjlarki: update this to be a class method
def facets_read_file(
    file: str,
    facet_width: float,
    facet_height: float,
    default_mirror_shape: Callable[[float, float], float] = lambda x, y: 0 * x * y,
) -> tuple[list[Facet], dict[str, Facet]]:
    """Read in a list of facets (and their relative positions on a standard heliostat) from a csv file.

    Arguments:
        file: one facet per row, with the format:
            name, x, y, z"""

    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        id_row = 0
        id_facet = 0
        facets: list[Facet] = []
        facet_dict: dict[str, Facet] = {}
        for row in readCSV:
            if not id_row:
                id_row += 1  # ignore the header
                continue
            name = str(row[0])
            x, y, z = (float(row[1]), float(row[2]), float(row[3]))

            # creating facet and mirror
            fn = default_mirror_shape  # flat surface
            mirror = MirrorParametricRectangular(fn, (facet_width, facet_height))
            facet = Facet(name=name, centroid_offset=[x, y, z], mirror=mirror)

            # storing
            facets.append(facet)
            facet_dict[name] = id_facet
            id_facet += 1

    return facets, facet_dict
