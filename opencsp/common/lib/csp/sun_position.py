"""
Sun Position Calculation

Adapted from John Clark Craig's post, "Python Sun Position for Solar Energy and Research."
https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777

See also:

    Pysolar: staring directly at the sun since 2007
    https://pysolar.readthedocs.io/en/latest/

    Source code for pvlib.solarposition
    https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/solarposition.html
"""

import math

import numpy as np


def sun_position_aux(
    location_lon_lat: tuple[float, float],  # radians.  (longitude, lattiude) pair.
    when_ymdhmsz: tuple[
        float, float, float, float, float, float, float
    ],  # (year, month, day, hour, minute, second, timezone) tuple.
    #  Example: (2022, 7, 4, 11, 20, 0, -6) => July 4, 2022 at 11:20 am MDT (-6 hours)
    refraction=True,
) -> tuple[float, float]:  # Boolean.  If True, apply refraction correction.
    """
    Calculates the sun's position (azimuth and elevation) based on the given location and time.

    This function computes the sun's apparent location in the sky using the provided
    longitude, latitude, and time information. It can also apply a refraction correction.

    Parameters
    ----------
    location_lon_lat : tuple[float, float]
        A tuple containing the longitude and latitude in radians.
    when_ymdhmsz : tuple[float, float, float, float, float, float, float]
        A tuple containing the year, month, day, hour, minute, second, and timezone.
    refraction : bool, optional
        If True, applies refraction correction to the elevation angle (default is True).

    Returns
    -------
    tuple[float, float]
        A tuple containing the azimuth and elevation angles in degrees.

    Notes
    -----
    The algorithm is based on John Clark Craig's implementation for calculating sun position in
    https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Extract the passed data
    year, month, day, hour, minute, second, timezone = when_ymdhmsz
    # Math typing shortcuts
    rad, deg = math.radians, math.degrees
    sin, cos, tan = math.sin, math.cos, math.tan
    asin, atan2 = math.asin, math.atan2
    # Fetch latitude and longitude in radians
    rlon = location_lon_lat[0]
    rlat = location_lon_lat[1]
    # Decimal hour of the day at Greenwich
    greenwichtime = hour - timezone + minute / 60 + second / 3600
    # Days from J2000, accurate from 1901 to 2099
    daynum = 367 * year - 7 * (year + (month + 9) // 12) // 4 + 275 * month // 9 + day - 730531.5 + greenwichtime / 24
    # Mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873
    # Mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768
    # Ecliptic longitude of the sun
    eclip_long = mean_long + 0.03342305518 * sin(mean_anom) + 0.0003490658504 * sin(2 * mean_anom)
    # Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum
    # Right ascension of the sun
    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))
    # Declination of the sun
    decl = asin(sin(obliquity) * sin(eclip_long))
    # Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon
    # Hour angle of the sun
    hour_ang = sidereal - rasc
    # Local elevation of the sun
    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))
    # Local azimuth of the sun
    azimuth = atan2(-cos(decl) * cos(rlat) * sin(hour_ang), sin(decl) - sin(rlat) * sin(elevation))
    # Convert azimuth and elevation to degrees
    azimuth = into_range(deg(azimuth), 0, 360)
    elevation = into_range(deg(elevation), -180, 180)
    # Refraction correction (optional)
    if refraction:
        targ = rad((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / tan(targ)) / 60
    # Return azimuth and elevation in degrees
    #    return (round(azimuth, 2), round(elevation, 2))  # ?? TODO RCB -- ORIGINAL CODE
    return (azimuth, elevation)


def into_range(x, range_min, range_max):
    """
    Adjusts a value to be within a specified range.

    This function takes a value and wraps it within the specified minimum and maximum range.

    Parameters
    ----------
    x : float
        The value to adjust.
    range_min : float
        The minimum value of the range.
    range_max : float
        The maximum value of the range.

    Returns
    -------
    float
        The adjusted value wrapped within the specified range.

    Notes
    -----
    The algorithm is based on John Clark Craig's implementation for calculating sun position in
    https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min


def sun_position(
    location_lon_lat: tuple[float, float], when_ymdhmsz: tuple  # radians.  (longitude, lattiude) pair.
) -> np.ndarray:  # (year, month, day, hour, minute, second, timezone) tuple.
    """
    Calculates the sun's apparent location in the sky based on the given location and time.

    This function retrieves the sun's azimuth and elevation angles and converts them into a unit vector.

    Parameters
    ----------
    location_lon_lat : tuple[float, float]
        A tuple containing the longitude and latitude in radians.
    when_ymdhmsz : tuple[float, float, float, float, float, float, float]
        A tuple containing the year, month, day, hour, minute, second, and timezone.

    Returns
    -------
    np.ndarray
        A unit vector representing the direction of the sun in 3D space.

    Notes
    -----
    The azimuth and elevation angles are calculated using the `sun_position_aux` function.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    #  Example: (2022, 7, 4, 11, 20, 0, -6) => July 4, 2022 at 11:20 am MDT (-6 hours)
    # Get the Sun's apparent location in the sky
    azimuth_deg, elevation_deg = sun_position_aux(location_lon_lat, when_ymdhmsz, True)  # John Clark Craig's version.
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)

    # Convert to a unit vector.
    sun_uxyz = direction_uxyz_given_azimuth_elevation(azimuth, elevation)

    # Return this routine's result.
    return sun_uxyz


def direction_uxyz_given_azimuth_elevation(azimuth: float, elevation: float):  # Both radians.
    """
    This function converts azimuth and elevation angles (in radians) into a unit vector
    in 3D space.

    Parameters
    ----------
    azimuth : float
        The azimuth angle in radians, measured from the positive x-axis.
    elevation : float
        The elevation angle in radians, measured from the xy-plane.

    Returns
    -------
    np.ndarray
        A 3D unit vector representing the direction corresponding to the given azimuth and elevation.

    Raises
    ------
    DeprecationWarning
        Indicates that this function is deprecated and should be migrated to another library.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Convert to degrees ccw from East.
    nu = (np.pi / 2.0) - azimuth
    # Construct the vector.
    z: float = np.sin(elevation)
    r: float = np.cos(elevation)
    x: float = r * np.cos(nu)
    y: float = r * np.sin(nu)
    uxyz = np.array([x, y, z])
    # Return.
    return uxyz
