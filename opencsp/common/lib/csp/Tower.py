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
        x_aim: float = 0,
        y_aim: float = 6.25,
        z_aim: float = 63.5508,
        bcs_y_aim: float = 8.8,
        bcs_z_aim: float = 28.9,
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
        self.bcs_y_aim = bcs_y_aim
        self.bcs_z_aim = bcs_z_aim
        self.target_loc = Pxyz([x_aim, y_aim, z_aim])
        self.bcs = Pxyz([x_aim, bcs_y_aim, bcs_z_aim])

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

        self.point = [self.x_aim, self.y_aim, self.z_aim]
        """
        The target location given the x, y, and z components. 
        """
        self.bcs_point = [self.x_aim, self.bcs_y_aim, self.bcs_z_aim]

    def walls(self):
        """Returns the list of walls as top, north, south, and bottom."""
        # Assumes that Tower coordinates have been set, and the walls have been set.
        # # Later we can add a more meaningful check for this.
        return [self.top, self.northface, self.southface, self.bottom]

    # RENDERING

    def draw(self, view: View3d, tower_style: RenderControlTower) -> None:
        # Assumes that heliostat configuration has already been set.

        tower_style = tower_style.style(self.name)

        # Whole tower
        if "whole tower" in self.parts:
            self.parts += ["top", "northface", "southface", "bottom", "northface"]

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

        # target on northface of tower
        if "target" in self.parts:
            view.draw_xyz(self.point, style=tower_style.target)

        if "bcs" in self.parts:
            view.draw_xyz(self.bcs_point, style=tower_style.bcs)

        return
