"""


"""

import math
import numpy as np


class HeliostatConfiguration:
    """
    Container for the variables defining the heliostat configuration.
    """

    def __init__(
        self,
        az: float = np.deg2rad(
            180
        ),  # (az,el) = (180,90) degrees corresponde to pointing straight up,
        el: float = np.deg2rad(90),
    ) -> None:  # as if transitioned by tilting up from face south orientation.
        super(HeliostatConfiguration, self).__init__()

        # Axis control.
        self.az = az
        self.el = el


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
    return HeliostatConfiguration(az=np.deg2rad(0), el=np.deg2rad(0))


def face_south() -> HeliostatConfiguration:
    return HeliostatConfiguration(az=np.deg2rad(180), el=np.deg2rad(0))


def face_east() -> HeliostatConfiguration:
    return HeliostatConfiguration(az=np.deg2rad(90), el=np.deg2rad(0))


def face_west() -> HeliostatConfiguration:
    return HeliostatConfiguration(az=np.deg2rad(270), el=np.deg2rad(0))


def face_up() -> HeliostatConfiguration:
    # Azinumth for UFACET scans.
    return HeliostatConfiguration(az=np.deg2rad(180), el=np.deg2rad(90))


def NSTTF_stow() -> HeliostatConfiguration:
    return HeliostatConfiguration(az=np.deg2rad(270), el=np.deg2rad(-85))
