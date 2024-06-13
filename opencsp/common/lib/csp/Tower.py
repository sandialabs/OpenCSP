"""
Tower Class

Copyright (c) 2021 Sandia National Laboratories.

"""

import csv
import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from sympy import Symbol, diff

import opencsp.common.lib.csp.ufacet.Heliostat as Heliostat
import opencsp.common.lib.csp.ufacet.HeliostatConfiguration as hc
import opencsp.common.lib.csp.sun_track as st
import opencsp.common.lib.geometry.transform_3d as t3d
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render_control.RenderControlTower as rct
import opencsp.common.lib.tool.math_tools as mt
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.geometry import Pxyz as OldPxyz
from opencsp.common.lib.geometry import Vxyz as OldVxyz
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlTower import RenderControlTower
from opencsp.common.lib.render.View3d import View3d


class Tower(RayTraceable):
    """
    Tower representation.
        renders nsttf tower

    """

    def __init__(
        self,
        name: str,
        origin: np.ndarray,
        parts: list[str] = ["whole tower"],
        height: float = 100,
        east: float = 8.8,
        west: float = -8.8,
        south: float = -8.8,
        north: float = 8.8,
        x_aim: float = 0,
        y_aim: float = 8.8,
        z_aim: float = 100,
    ):

        # parameters used for control tower at NSTTF
        # tower_control= Tower(name='Sandia NSTTF Control Tower',
        #             origin = np.array([0,0,0]),
        #             height=25,
        #             east = 8.8,
        #             west = -8.8,
        #             south = 284,
        #             north = 300)
        """Create a new Tower instance.

        Parameters:
        -----------
            name    The name of this Tower. Used for special styles given in the draw method.

            origin  The center of Tower, as a three vector xyz coordinate.

            all measurements in meters using ENU coordinate system.
        """
        super(Tower, self).__init__()
        self.name = name
        self.origin = origin
        self.parts = parts
        self.height = height
        self.east = east
        self.west = west
        self.south = south
        self.north = north
        self.x_aim = x_aim
        self.y_aim = y_aim
        self.z_aim = z_aim
        self.target_loc = Pxyz([x_aim, y_aim, z_aim])

        # Validate the input
        # if (__debug__) and (self.num_facets != self.num_rows * self.num_cols):
        #     raise AssertionError("num_facets != num_rows * num_cols... is this correct? if so, then please remove this assertion") # TODO remove "num_facets" arg if this assertion is correct

        # Tower faces, top, and bottom
        self.top = [
            [self.east, self.north, self.height],
            [self.west, self.north, self.height],
            [self.west, self.south, self.height],
            [self.east, self.south, self.height],
        ]
        self.northface = [
            [self.east, self.north, self.height],
            [self.west, self.north, self.height],
            [self.west, self.north, 0],
            [self.east, self.north, 0],
        ]
        self.southface = [
            [self.east, self.south, self.height],
            [self.west, self.south, self.height],
            [self.west, self.south, 0],
            [self.east, self.south, 0],
        ]
        self.bottom = [
            [self.east, self.north, 0],
            [self.west, self.north, 0],
            [self.west, self.south, 0],
            [self.east, self.south, 0],
        ]

        self.point = [self.x_aim, self.y_aim, self.z_aim]

        # Facets
        # self.facets = facets
        # self.parts_dict = {f.name: f_i for f_i, f in enumerate(parts)} # TODO tjlarki: check that this is what we want

        # # Tower Edge
        # self.top_left_facet:Facet     = self.facets[self.facet_dict[str(1)]]
        # self.top_right_facet:Facet    = self.facets[self.facet_dict[str(num_cols)]]
        # self.bottom_right_facet:Facet = self.facets[self.facet_dict[str(num_facets)]]
        # self.bottom_left_facet:Facet  = self.facets[self.facet_dict[str(num_facets - num_cols + 1)]]

        # # Tower Corners [offsets in terms of Tower's centroid]
        # self.top_left_corner_offset     = [x + y for x,y in zip(self.top_left_facet.centroid_offset, self.top_left_facet.top_left_corner_offset)]
        # self.top_right_corner_offset    = [x + y for x,y in zip(self.top_right_facet.centroid_offset, self.top_right_facet.top_right_corner_offset)]
        # self.bottom_right_corner_offset = [x + y for x,y in zip(self.bottom_right_facet.centroid_offset, self.bottom_right_facet.bottom_right_corner_offset)]
        # self.bottom_left_corner_offset  = [x + y for x,y in zip(self.bottom_left_facet.centroid_offset, self.bottom_left_facet.bottom_left_corner_offset)]

        # Centroid
        self.origin = np.array([origin[0], origin[1], origin[2]])  # Origin is at center of tower.

        # self.az             = np.deg2rad(180)  # (az,el) = (180,90) degrees corresponds to pointing straight up,
        # self.el             = np.deg2rad(90)   # as if transitioned by tilting up from face south orientation.
        # self.surface_normal = [0, 0, 1]        #
        self.rx_rotation = np.identity(3)
        self.rz_rotation = np.identity(3)
        self.rotation = Rotation.identity()  # Rz_rotation.dot(Rx_rotation)
        # self._set_wall_positions_in_space()

        # Tracking
        # self._aimpoint_xyz = None   # (x,y,y) in m. Do not access this member externally; use aimpoint_xyz() function instead.
        # self._when_ymdhmsz = None   # (y,m,d,h,m,s,z). Do not access this member externally; use when_ymdhmsz() function instead.

        # SET POSITION IN SPACE
        self.set_position_in_space(self.origin, self.rotation)

    # ACCESS

    def aimpoint_xyz(self):
        if self._aimpoint_xyz == None:
            print('ERROR: In Tower.aimpoint_xyz(), attempt to fetch unset _aimpoint_xyz.')
            assert False
        return self._aimpoint_xyz

    def when_ymdhmsz(self):
        if self._when_ymdhmsz == None:
            print('ERROR: In Tower.when_ymdhmsz(), attempt to fetch unset _when_ymdhmsz.')
            assert False
        return self._when_ymdhmsz

    def compute_tracking_configuration(self, aimpoint_xyz, location_lon_lat, when_ymdhmsz):
        # Tower centroid coordinates.
        # Coordinates are (x,z) center, z=0 is at torque tube height.
        h_tube = np.array(self.origin)
        h = h_tube  # Later, add correction for center facet offset.

        # Compute Tower surface normal which tracks the sun to the aimpoint.
        n_xyz = st.tracking_surface_normal_xyz(h, aimpoint_xyz, location_lon_lat, when_ymdhmsz)

        # Compute Tower configuration.
        return hc.Tower_configuration_given_surface_normal_xyz(n_xyz)

    def walls(self):
        """Returns the list of walls in ul,ur,lr,ll order."""
        # Assumes that Tower coordinates have been set, and the walls have been set.
        # Later we can add a more meaningful check for this.
        return [self.top_wall, self.north_wall, self.south_wall, self.bottom_wall]

    def rodrigues_vector_from_partial_derivatives(self, dfdx_n: float, dfdy_n: float) -> OldVxyz:
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
            # TODO:  THIS SHOULD RETURN AN ACTUAL OldVxyz.
            return np.array([0.0, 0.0, 0.0])  # Zero magnitude implies zero rotation; direction doesn't matter.

        else:
            # There is a rotation.
            x_vec = [1.0, 0.0, float(dfdx_n)]  # direction of x partial derivative
            y_vec = [0.0, 1.0, float(dfdy_n)]  # direction of y partial derivative

            v = np.cross(x_vec, y_vec)

            v_norm = np.linalg.norm(v)
            if v_norm == 0.0:
                # TODO RCB: REPLACE THIS WITH LOG/EXCEPTION THROW.
                print('ERROR: In Tower.set_canting_from_equation(), encountered unexpected zero v_norm.')
                print('         x_vec = ', x_vec)
                print('         y_vec = ', y_vec)
                assert False

            u = v / v_norm

            v_rot = np.cross(u, np.array([0.0, 0.0, 1.0]))  # axis of rotation

            v_rot_norm = np.linalg.norm(v_rot)
            if v_rot_norm == 0.0:
                # TODO RCB: REPLACE THIS WITH LOG/EXCEPTION THROW.
                print('ERROR: In Tower.set_canting_from_equation(), encountered unexpected zero v_rot_norm.')
                print('         u = ', u)
                print('         v_rot = ', v_rot)
                assert False

            u_rot = v_rot / v_rot_norm  # axis of rotation normalized

            theta_rot = -mt.robust_arccos(np.dot(u, np.array([0.0, 0.0, 1.0])))  # angle of rotation

            # TODO:  THIS SHOULD RETURN AN ACTUAL OldVxyz.
            return theta_rot * u_rot

    def set_canting_from_list() -> None:
        # TODO: THIS SHOULD BE EITHER IMPLEMENTED, OR DISCARDED.
        ...

    # MODIFICATION

    def set_tracking(self, aimpoint_xyz, location_lon_lat, when_ymdhmsz):
        # Save tracking command.
        self._aimpoint_xyz = aimpoint_xyz
        self._when_ymdhmsz = when_ymdhmsz
        # Set tracking configuration.
        h_config = self.compute_tracking_configuration(aimpoint_xyz, location_lon_lat, when_ymdhmsz)
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
        hel_rotation: Rotation = rotation_from_az_el(az, el)

        # rotation_about_x = (np.pi/2) - el
        # rotation_about_z = np.pi - az
        # Rx_rotation = t3d.axisrotation(np.array([1, 0, 0]), rotation_about_x)
        # Rz_rotation = t3d.axisrotation(np.array([0, 0, 1]), rotation_about_z)
        # hel_rotation   = Rz_rotation.dot(Rx_rotation)

        vector = np.array([0, 0, self.pivot_offset])
        vector_offset = hel_rotation.apply(vector)
        origin = np.array(self.origin) + vector_offset  # Origin is at torque tube center.

        surface_normal = hel_rotation.apply([0, 0, 1])  # Before rotation, heliostat is face up.

    # ?? SCAFFOLDING RCB -- ORIGINAL CODE
    # rotation_about_x = (np.pi/2) - el
    # rotation_about_z = np.pi - az

    # Rx_rotation = t3d.axisrotation(np.array([1, 0, 0]), rotation_about_x)
    # Rz_rotation = t3d.axisrotation(np.array([0, 0, 1]), rotation_about_z)

    # vector        = np.array([0, 0, self.pivot_offset])
    # vector_offset = Rz_rotation.dot(Rx_rotation).dot(vector)
    # centroid      = np.array(self.origin) + vector_offset  # Origin is at torque tube center.

    # hel_rotation   = Rz_rotation.dot(Rx_rotation)
    # surface_normal = hel_rotation.dot([0, 0, 1])  # Before rotation, heliostat is face up.

    #     self.el               = el
    #     self.az               = az
    #     self.origin = origin                     # ?? SCAFFOLDING RCB -- THESE LIST TO ARRAY AND BACK TYPE CONVERSIONS ARE INEFFICIENT.  RESOLVE THIS.
    #     self.surface_normal   = list(surface_normal)         # ?? SCAFFOLDING RCB -- THESE LIST TO ARRAY AND BACK TYPE CONVERSIONS ARE INEFFICIENT.  RESOLVE THIS.
    #     # self.rx_rotation      = Rx_rotation
    #     # self.rz_rotation      = Rz_rotation
    #     self.rotation         = hel_rotation

    #     self.set_position_in_space(self.origin, self.rotation)
    #     self._set_wall_positions_in_space()

    # def _set_wall_positions_in_space(self):
    #     """ Updates wall positions given this heliostat's configuration. """
    #     self.top_wall     = self.transform_xyz(self.top)
    #     self.north_wall    = self.transform_xyz(self.north)
    #     self.south_wall = self.transform_xyz(self.south)
    #     self.east_wall    = self.transform_xyz(self.east)
    #     self.west_wall = self.transform_xyz(self.west)
    #     self.bottom_wall  = self.transform_xyz(self.bottom)

    # def transform_xyz(self, xyz):
    #     """
    #     Accepts an [x,y,z] list, and returns a numpy array.  # ?? SCAFFOLDING RCB -- THIS IS ONE OF MANY PLACES WHERE EFFICIENCY COULD IMPROVE.  FOR EXAMPLE, WE COULD WORK WITH ARRAYS INSTEAD OF COORDINATE LISTS.  A CLASS OBJECT INSTEAD?
    #     """
    #     transformed_xyz = self.origin + self.rotation.apply(np.array(xyz))
    #     return transformed_xyz

    # def set_position_in_space(self, location:OldPxyz, rotation:Rotation) -> None:
    #     self.origin = location # + self.pivot_offset
    #     self.rotation = rotation
    #     for top in self.top:
    #         top.set_position_in_space(location, rotation)

    # RENDERING

    def draw(self, view: View3d, tower_style: RenderControlTower) -> None:
        # Assumes that heliostat configuration has already been set.

        tower_style = tower_style.style(self.name)

        # Whole tower
        if 'whole tower' in self.parts:
            self.parts += ['top', 'northface', 'southface', 'northface', 'bottom']

        # Top of tower
        if "top" in self.parts:
            view.draw_xyz_list(self.top, close=True, style=tower_style.wire_frame)

        # Northface of tower
        if "northface" in self.parts:
            view.draw_xyz_list(self.northface, close=True, style=tower_style.wire_frame)

        # target on northface of tower
        if "target" in self.parts:
            view.draw_xyz(self.point, style=tower_style.target)

        # Southface of tower
        if "southface" in self.parts:
            view.draw_xyz_list(self.southface, close=True, style=tower_style.wire_frame)

        # Bottom of tower
        if "bottom" in self.parts:
            view.draw_xyz_list(self.bottom, close=True, style=tower_style.wire_frame)

        return

    # def survey_of_points(self, resolution, random_dist:bool=False) -> tuple[Pxyz, Vxyz]:
    #     """
    #     Returns a grid of equispaced points and the normal vectors at those points.

    #     Parameters
    #     ----------
    #     resolution:
    #         the rectangular resolution of the points gathered (add other forms later, like triangular or polar survey).

    #     Returns
    #     -------
    #         a tuple of the points (Pxyz) and normals at the respective points (Vxyz).

    #     """
    #     points = Pxyz([[],[],[]])
    #     normals = Vxyz([[],[],[]])
    #     for facet in self.facets:
    #         additional_points, additional_normals = facet.survey_of_points(resolution, random_dist)
    #         points = points.concatenate(additional_points)
    #         normals = normals.concatenate(additional_normals)

    #     return (points, normals)


# HELPER FUNCTIONS

# def rotation_from_az_el(azimuth: float, elevation: float) -> Rotation:
#     rotation_about_x = (np.pi/2) - elevation
#     rotation_about_z = np.pi - azimuth

#     # TODO tjlarki: depricated approach, identical
#     # Rx_rotation = t3d.axisrotation(np.array([1, 0, 0]), rotation_about_x)
#     # Rz_rotation = t3d.axisrotation(np.array([0, 0, 1]), rotation_about_z)
#     # hel_rotation: np.ndarray = np.dot(Rz_rotation, Rx_rotation)

#     Rx_rotation = Rotation.from_euler('x', rotation_about_x)
#     Rz_rotation = Rotation.from_euler('z', rotation_about_z)
#     hel_rotation:Rotation =  Rz_rotation*Rx_rotation # composed rotation
#     return hel_rotation


# # GENERATOR

# def h_from_facet_centroids(name: str, origin: np.ndarray, num_facets:int, num_rows:int=0, num_cols:int=0,
#                 file_centroids_offsets=None, pivot_height:float=0, pivot_offset:float=0,
#                 facet_width:float=0, facet_height:float=0, default_mirror_shape:Callable[[float,float],float]=lambda x,y:x*0) -> 'Heliostat':

#     # Facets
#     facets, _ = facets_read_file(file_centroids_offsets, facet_height, facet_width, default_mirror_shape)

#     return Heliostat(name, origin, num_facets, num_rows, num_cols, facets, pivot_height, pivot_offset)

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
#                 # get rid of the header in csv
#                 id_row += 1
#                 continue
#             name, x, y, z = str(row[0]), float(row[1]), float(row[2]), float(row[3])

#             # creating facet and mirror
#             fn = default_mirror_shape # flat surface
#             mirror = MirrorParametricRectangular(fn, (facet_width, facet_height))
#             facet = Facet(name=name, centroid_offset=[x, y, z],
#                             mirror=mirror)

#             # storing
#             facets.append(facet)
#             facet_dict[name] = id_facet
#             id_facet += 1

#     return facets, facet_dict
