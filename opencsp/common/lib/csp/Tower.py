"""
Tower Class

"""

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.geometry.Pxyz import Pxyz
import opencsp.common.lib.render_control.RenderControlTower as rct
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlTower import RenderControlTower
from opencsp.common.lib.render.View3d import View3d


class Tower(RayTraceable):
    """
    Tower representation.

    All parameter values are in meters using ENU (East, North, Up)
    coordinate system. Values are relative to the field coordinates.

    Parameters:
    -----------
    name : str
        The name of this Tower. Used for special styles given in the draw method.
    origin : Pxyz
        The center of Tower. All other points in the tower are relative to this value.
    parts : list[str]
        The parts that build the Tower. Includes walls (top, northface, southface, bottom), and optionl target.
    height : float
        The height of Tower. Currently set for NSTTF Tower height of 63.5508 m.
    east : float
        The East wall of the Tower. Currently set for 5.485 m. TODO, MHH find dimensions of tower (in particular width)
    west : float
        The West wall of the Tower. Currently set for 5.485 m.
    south : float
        The South wall of the Tower. Currently set for 9.1168 m.
    north : float
        The North wall of the Tower. Currently set for 6.25 m.
    x_aim : float
        The x component of the target in relation to the Tower origin.
    y_aim : float
        The y component of the target in relation to the Tower origin.
    z_aim : float
        The z component of the target in relation to the Tower origin.
    """

    def __init__(
        self,
        name: str,
        origin: Pxyz,
        parts: list[str] = ["whole tower"],
        height: float = 63.5508,
        east: float = 5.485,
        west: float = -5.485,
        south: float = -9.1186,
        north: float = 6.25,
        height_g3p3: float = 53.89,
        east_g3p3: float = -33.47,
        west_g3p3: float = -46.63,
        south_g3p3: float = -23.054,
        north_g3p3: float = -9.90,
        x_aim: float = 0,
        y_aim: float = 6.25,
        z_aim: float = 63.5508,
        bcs_y_aim: float = 8.8,
        bcs_z_aim: float = 28.9,
        bcs_height: float = 50,
        g3p3_x_aim: float = -40,
        g3p3_y_aim: float = 8.5,
        g3p3_z_aim: float = 45.48,
    ):

        # parameters used for control tower at NSTTF
        # tower_control= Tower(name='Sandia NSTTF Control Tower',
        #             origin = np.array([0,0,0]),
        #             height=25,
        #             east = 8.8,
        #             west = -8.8,
        #             south = 284,
        #             north = 300
        #             x_aim: float = 0,
        #             y_aim: float = 6.25,
        #             z_aim: float = 63.5508,)
        # SOFAST_calculations:
        #             height: float = 63.5508,
        #             east: float = 5.485,
        #             west: float = -5.485,
        #             south: float = -9.1186,
        #             north: float = 6.25,
        #             x_aim: float = 0,
        #             y_aim: float = 6.25,
        #             z_aim: float = 64.8008,
        #             bcs_y_aim: float = 8.8,
        #             bcs_z_aim: float = 28.9,

        #         height_g3p3: float = 53.89,
        # east_g3p3: float = -33.42,
        # west_g3p3: float = -46.58,
        # south_g3p3: float = -4.674,
        # north_g3p3: float = 8.5,
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
        self.height_g3p3 = height_g3p3
        self.east_g3p3 = east_g3p3
        self.west_g3p3 = west_g3p3
        self.south_g3p3 = south_g3p3
        self.north_g3p3 = north_g3p3
        self.x_aim = x_aim
        self.y_aim = y_aim
        self.z_aim = z_aim
        self.bcs_y_aim = bcs_y_aim
        self.bcs_z_aim = bcs_z_aim
        self.bcs_height = bcs_height
        self.g3p3_x_aim = g3p3_x_aim
        self.g3p3_y_aim = g3p3_y_aim
        self.g3p3_z_aim = g3p3_z_aim
        self.target_loc = Pxyz([x_aim, y_aim, z_aim])
        self.bcs = Pxyz([x_aim, bcs_y_aim, bcs_z_aim])
        self.g3p3 = Pxyz([g3p3_x_aim, g3p3_y_aim, g3p3_z_aim])

        # Define the tilt angle in degrees and convert to radians
        tilt_angle = -26  # degrees
        theta = np.radians(tilt_angle)

        # Define the rotation matrix for a tilt around the Z-axis
        self.rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )

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
        """
        The top-left, top-right, bottom-right, bottom-left corners of the Tower,
        as viewed when standing north of the tower and facing south.
        """

        self.southface = [
            [self.east, self.south, self.height],
            [self.west, self.south, self.height],
            [self.west, self.south, 0],
            [self.east, self.south, 0],
        ]
        """
        The top-left, top-right, bottom-right, bottom-left corners of the Tower,
        as viewed when standing south of the tower and facing north.
        """

        self.bottom = [
            [self.east, self.north, 0],
            [self.west, self.north, 0],
            [self.west, self.south, 0],
            [self.east, self.south, 0],
        ]

        self.northface_bcs = [
            [self.east, self.bcs_y_aim, self.bcs_height],
            [self.west, self.bcs_y_aim, self.bcs_height],
            [self.west, self.bcs_y_aim, 0],
            [self.east, self.bcs_y_aim, 0],
        ]

        self.top_bcs = [
            [self.east, self.bcs_y_aim, self.bcs_height],
            [self.west, self.bcs_y_aim, self.bcs_height],
            [self.west, self.north, self.bcs_height],
            [self.east, self.north, self.bcs_height],
        ]

        self.bottom_bcs = [
            [self.east, self.bcs_y_aim, 0],
            [self.west, self.bcs_y_aim, 0],
            [self.west, self.north, 0],
            [self.east, self.north, 0],
        ]

        # G3P3 Tower faces, top, and bottom
        self.top_g3p3 = [
            [self.east_g3p3, self.north_g3p3, self.height_g3p3],
            [self.west_g3p3, self.north_g3p3, self.height_g3p3],
            [self.west_g3p3, self.south_g3p3, self.height_g3p3],
            [self.east_g3p3, self.south_g3p3, self.height_g3p3],
        ]

        self.northface_g3p3 = [
            [self.east_g3p3, self.north_g3p3, self.height_g3p3],
            [self.west_g3p3, self.north_g3p3, self.height_g3p3],
            [self.west_g3p3, self.north_g3p3, 0],
            [self.east_g3p3, self.north_g3p3, 0],
        ]
        """
        The top-left, top-right, bottom-right, bottom-left corners of the Tower,
        as viewed when standing north of the tower and facing south.
        """

        self.southface_g3p3 = [
            [self.east_g3p3, self.south_g3p3, self.height_g3p3],
            [self.west_g3p3, self.south_g3p3, self.height_g3p3],
            [self.west_g3p3, self.south_g3p3, 0],
            [self.east_g3p3, self.south_g3p3, 0],
        ]
        """
        The top-left, top-right, bottom-right, bottom-left corners of the Tower,
        as viewed when standing south of the tower and facing north.
        """

        self.bottom_g3p3 = [
            [self.east_g3p3, self.north_g3p3, 0],
            [self.west_g3p3, self.north_g3p3, 0],
            [self.west_g3p3, self.south_g3p3, 0],
            [self.east_g3p3, self.south_g3p3, 0],
        ]

        # Apply rotation to G3P3 tower coordinates
        self.top_g3p3 = self.rotate_coordinates(self.top_g3p3, self.rotation_matrix)
        self.northface_g3p3 = self.rotate_coordinates(self.northface_g3p3, self.rotation_matrix)
        self.southface_g3p3 = self.rotate_coordinates(self.southface_g3p3, self.rotation_matrix)
        self.bottom_g3p3 = self.rotate_coordinates(self.bottom_g3p3, self.rotation_matrix)

        self.normal_south_face = self.calculate_south_face_normal()

        self.point = [self.x_aim, self.y_aim, self.z_aim]
        """
        The target location given the x, y, and z components. 
        """
        self.bcs_point = [self.x_aim, self.bcs_y_aim, self.bcs_z_aim]

        self.g3p3_point = [self.g3p3_x_aim, self.g3p3_y_aim, self.g3p3_z_aim]

    def walls(self):
        """Returns the list of walls as top, north, south, and bottom."""
        # Assumes that Tower coordinates have been set, and the walls have been set.
        # # Later we can add a more meaningful check for this.
        return [self.top, self.northface, self.southface, self.bottom]

    def walls_g3p3(self):
        """Returns the list of walls as top, north, south, and bottom."""
        # Assumes that Tower coordinates have been set, and the walls have been set.
        # # Later we can add a more meaningful check for this.
        return [self.top_g3p3, self.northface_g3p3, self.southface_g3p3, self.bottom_g3p3]

    def rotate_coordinates(self, coordinates, rotation_matrix):
        """Rotate a list of coordinates using the given rotation matrix."""
        rotated_coordinates = []
        for coord in coordinates:
            coord = np.array(coord)  # Convert to numpy array
            rotated_coord = rotation_matrix @ coord  # Matrix multiplication
            rotated_coordinates.append(rotated_coord.tolist())  # Convert back to list
        return rotated_coordinates

    def draw(self, view: View3d, tower_style: RenderControlTower) -> None:
        # Assumes that heliostat configuration has already been set.

        tower_style = tower_style.style(self.name)

        # Whole tower
        if 'whole tower' in self.parts:
            self.parts += ['top', 'northface', 'southface', 'bottom', 'northface']

        # Top of tower
        if "top" in self.parts:
            view.draw_xyz_list(self.top, close=True, style=tower_style.wire_frame)

        # Northface of tower
        if "northface" in self.parts:
            view.draw_xyz_list(self.northface, close=True, style=tower_style.wire_frame)

        # Southface of tower
        if "southface" in self.parts:
            view.draw_xyz_list(self.southface, close=True, style=tower_style.wire_frame)

        # Bottom of tower
        if "bottom" in self.parts:
            view.draw_xyz_list(self.bottom, close=True, style=tower_style.wire_frame)

        # Northface of tower
        if "top_bcs" in self.parts:
            view.draw_xyz_list(self.top_bcs, close=True, style=tower_style.wire_frame)

        # Northface of tower
        if "northface_bcs" in self.parts:
            view.draw_xyz_list(self.northface_bcs, close=True, style=tower_style.wire_frame)

        # Northface of tower
        if "bottom_bcs" in self.parts:
            view.draw_xyz_list(self.bottom_bcs, close=True, style=tower_style.wire_frame)

        if 'BCS Tower' in self.parts:
            self.parts += ['northface_bcs', 'top_bcs', 'bottom_bcs']

        if 'G3P3 Tower' in self.parts:
            self.parts += ['top_g3p3', 'northface_g3p3', 'southface_g3p3', 'bottom_g3p3', 'northface_g3p3']

        # Top of tower
        if "top_g3p3" in self.parts:
            view.draw_xyz_list(self.top_g3p3, close=True, style=tower_style.wire_frame)

        # Northface of tower
        if "northface_g3p3" in self.parts:
            view.draw_xyz_list(self.northface_g3p3, close=True, style=tower_style.wire_frame)

        # Southface of tower
        if "southface_g3p3" in self.parts:
            view.draw_xyz_list(self.southface_g3p3, close=True, style=tower_style.wire_frame)

        # Bottom of tower
        if "bottom_g3p3" in self.parts:
            view.draw_xyz_list(self.bottom_g3p3, close=True, style=tower_style.wire_frame)

        # target on northface of tower
        if "TowerTop" in self.parts:
            view.draw_xyz(self.point, style=tower_style.target)

        if "BCS" in self.parts:
            view.draw_xyz(self.bcs_point, style=tower_style.bcs)

        if "G3P3" in self.parts:
            view.draw_xyz(self.g3p3_point, style=tower_style.g3p3)

        return

    def calculate_south_face_normal(self):
        """Calculate the normal vector of the south face."""
        # Extract points of the south face
        p1 = np.array(self.southface_g3p3[0])
        p2 = np.array(self.southface_g3p3[2])
        p3 = np.array(self.southface_g3p3[3])

        # Create two edge vectors
        edge1 = p2 - p3  # Vector from p1 to p2
        edge2 = p1 - p3  # Vector from p1 to p3

        # Calculate the normal vector using the cross product
        normal_vector = np.cross(edge1, edge2)

        # Normalize the normal vector
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # Apply the rotation to the normal vector
        # rotated_normal = self.rotation_matrix @ normal_vector
        return normal_vector
