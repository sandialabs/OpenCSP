"""


"""

import math
import numpy as np


class HeliostatConfiguration:
    """
    Container for the variables defining the heliostat configuration.

    This class allows the user to define a heliostat configuration by specifying
    the type of heliostat and its associated parameters. Currently, only the 'az-el'
    type is supported, which requires azimuth and elevation angles.

    Parameters
    ----------
    heliostat_type : str
        The type of heliostat configuration. Must be one of the valid heliostat types.
    az : float, optional
        The azimuth angle in radians (required for 'az-el' type).
    el : float, optional
        The elevation angle in radians (required for 'az-el' type).

    Raises
    ------
    ValueError
        If the provided heliostat type is invalid or if az and el are not provided
        for the 'az-el' type.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, heliostat_type: str, az: float = None, el: float = None) -> None:
        """
        Initializes a HeliostatConfiguration object with the specified type and angles.

        Parameters
        ----------
        heliostat_type : str
            The type of heliostat configuration.
        az : float, optional
            The azimuth angle in radians (default is None).
        el : float, optional
            The elevation angle in radians (default is None).
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.valid_heliostat_types = ['az-el']
        self.heliostat_type = heliostat_type

        if heliostat_type not in self.valid_heliostat_types:
            raise ValueError(
                f'Invalid type of heliostat. {self.heliostat_type} is ' f'not one of: {self.valid_heliostat_types}.'
            )

        if heliostat_type == 'az-el':
            if (az is None) or (el is None):
                raise ValueError(f"Cannot have a HeliostatAzEl configuration" f" without az and el arguments")
            self.az = az
            self.el = el

    def get_values(self):
        """
        Retrieves the azimuth and elevation values for the heliostat configuration.

        Returns
        -------
        tuple
            A tuple containing the azimuth and elevation angles in radians.

        Raises
        ------
        ValueError
            If the heliostat type is invalid.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if self.heliostat_type == 'az-el':
            return self.az, self.el
        raise ValueError(
            f'Invalid type of heliostat. {self.heliostat_type} is ' f'not one of: {self.valid_heliostat_types}.'
        )


# CONFIGURATION CONSTRUCTION


def heliostat_configuration_given_surface_normal_xyz(n_xyz) -> HeliostatConfiguration:
    """
    Creates a HeliostatConfiguration from a given surface normal vector.

    This function converts the surface normal coordinates into azimuth and elevation
    angles.

    Parameters
    ----------
    n_xyz : array-like
        A 3-element array or list representing the surface normal vector in 3D space.

    Returns
    -------
    HeliostatConfiguration
        A HeliostatConfiguration object with the calculated azimuth and elevation angles.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
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
    """
    Creates a HeliostatConfiguration for a heliostat facing north.

    Returns
    -------
    HeliostatConfiguration
        A HeliostatConfiguration object with azimuth set to 0 radians and elevation set to 0 radians.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return HeliostatConfiguration('az-el', az=np.deg2rad(0), el=np.deg2rad(0))


def face_south() -> HeliostatConfiguration:
    """
    Creates a HeliostatConfiguration for a heliostat facing south.

    Returns
    -------
    HeliostatConfiguration
        A HeliostatConfiguration object with azimuth set to 180 radians and elevation set to 0 radians.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return HeliostatConfiguration('az-el', az=np.deg2rad(180), el=np.deg2rad(0))


def face_east() -> HeliostatConfiguration:
    """
    Creates a HeliostatConfiguration for a heliostat facing east.

    Returns
    -------
    HeliostatConfiguration
        A HeliostatConfiguration object with azimuth set to 90 radians and elevation set to 0 radians.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return HeliostatConfiguration('az-el', az=np.deg2rad(90), el=np.deg2rad(0))


def face_west() -> HeliostatConfiguration:
    """
    Creates a HeliostatConfiguration for a heliostat facing west.

    Returns
    -------
    HeliostatConfiguration
        A HeliostatConfiguration object with azimuth set to 270 radians and elevation set to 0 radians.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return HeliostatConfiguration('az-el', az=np.deg2rad(270), el=np.deg2rad(0))


def face_up() -> HeliostatConfiguration:
    """
    Creates a HeliostatConfiguration for a heliostat facing directly up.

    Returns
    -------
    HeliostatConfiguration
        A HeliostatConfiguration object with azimuth set to 180 radians and elevation set to 90 radians.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Azinumth for UFACET scans.
    return HeliostatConfiguration('az-el', az=np.deg2rad(180), el=np.deg2rad(90))


def NSTTF_stow() -> HeliostatConfiguration:
    """
    Creates a HeliostatConfiguration for the NSTTF stow position.

    Returns
    -------
    HeliostatConfiguration
        A HeliostatConfiguration object with azimuth set to 270 radians and elevation set to -85 radians.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return HeliostatConfiguration('az-el', az=np.deg2rad(270), el=np.deg2rad(-85))
