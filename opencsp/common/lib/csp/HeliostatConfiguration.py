"""


"""

import math
import numpy as np


class HeliostatConfiguration():
    """
    Container for the variables defining the heliostat configuration.
    User must provide the type of heliostat in the format in `valid_heliostat_types.

    """

    def __init__(self,
                 heliostat_type: str,
                 az: float = None,
                 el: float = None,

                 ) -> None:

        self.valid_heliostat_types = ['az-el']
        self.heliostat_type = heliostat_type

        if heliostat_type not in self.valid_heliostat_types:
            raise ValueError(f'Invalid type of heliostat. {self.heliostat_type} is '
                             f'not one of: {self.valid_heliostat_types}.')

        if heliostat_type == 'az-el':
            if (az is None) or (el is None):
                raise ValueError(f"Cannot have a HeliostatAzEl configuration"
                                 f" without az and el arguments")
            self.az = az
            self.el = el

    def get_values(self):
        if self.heliostat_type == 'az-el':
            return self.az, self.el
        raise ValueError(f'Invalid type of heliostat. {self.heliostat_type} is '
                         f'not one of: {self.valid_heliostat_types}.')


# CONFIGURATION CONSTRUCTION

def heliostat_configuration_given_surface_normal_xyz(n_xyz) -> HeliostatConfiguration:
    # Extract surface normal coordinates.
    n_x = n_xyz[0]
    n_y = n_xyz[1]
    n_z = n_xyz[2]

    # Convert heliostat surface normal to (az,el) coordinates.
    #   Elevation is measured up from horizontal,
    #   Azimuth is measured clockwise from north (compass headings).
    #
    # elevation
    n_xy_norm = math.sqrt((n_x * n_x) + (n_y * n_y))
    el = math.atan2(n_z, n_xy_norm)
    # azimuth
    # nu is the angle to the projection of the surface normal onto the (x,y) plane, measured ccw from the x axis.
    nu = math.atan2(n_y, n_x)
    az = (np.pi / 2) - nu  # Measured cw from the y axis.

    # Return the configuration.
    h_config = HeliostatConfiguration(el=el, az=az)
    return h_config


# COMMON CONFIGURATIONS

def face_north() -> HeliostatConfiguration:
    return HeliostatConfiguration('az-el', az=np.deg2rad(0), el=np.deg2rad(0))


def face_south() -> HeliostatConfiguration:
    return HeliostatConfiguration('az-el', az=np.deg2rad(180), el=np.deg2rad(0))


def face_east() -> HeliostatConfiguration:
    return HeliostatConfiguration('az-el', az=np.deg2rad(90), el=np.deg2rad(0))


def face_west() -> HeliostatConfiguration:
    return HeliostatConfiguration('az-el', az=np.deg2rad(270), el=np.deg2rad(0))


def face_up() -> HeliostatConfiguration:
    # Azinumth for UFACET scans.
    return HeliostatConfiguration('az-el', az=np.deg2rad(180), el=np.deg2rad(90))


def NSTTF_stow() -> HeliostatConfiguration:
    return HeliostatConfiguration('az-el', az=np.deg2rad(270), el=np.deg2rad(-85))
