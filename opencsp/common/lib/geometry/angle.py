"""
Angle Management
"""

import math
import numbers
import numpy as np
import numpy.typing as npt
from typing import Iterable, overload


# CONVERTING RADIANS TO DEGREES

# Use numpy.rad2deg(x) for scalars.


def coord2deg_aux(xy_or_xyz, idx):
    """
    Converts the specified coordinate (x, y, or z) from radians to degrees.

    Parameters
    ----------
    xy_or_xyz : np.ndarray
        A 1D array representing a point in 2D or 3D space.
    idx : int
        The index of the coordinate to convert (0 for x, 1 for y, 2 for z).

    Returns
    -------
    np.ndarray
        A copy of the input array with the specified coordinate converted to degrees.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    pt = xy_or_xyz.copy()
    pt[idx] = np.rad2deg(pt[idx])
    return pt


def coord2deg(xy_or_xyz_or_xy_seq_or_xyz_seq, idx):
    """
    Converts the specified coordinate from radians to degrees for a single point or a sequence of points.

    Parameters
    ----------
    xy_or_xyz_or_xy_seq_or_xyz_seq : np.ndarray or Iterable[np.ndarray]
        A single point (2D or 3D) or a sequence of points to convert.
    idx : int
        The index of the coordinate to convert (0 for x, 1 for y, 2 for z).

    Returns
    -------
    list[np.ndarray]
        A list of points with the specified coordinate converted to degrees.
        Returns an empty list if the input is empty.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    if len(xy_or_xyz_or_xy_seq_or_xyz_seq) == 0:
        # Then the input xy_or_xyz_or_xy_seq_or_xyz_seq can be interpreted
        # as an input sequence, that has zero length.
        # Given such an input, return the empty list.
        return []
    if isinstance(xy_or_xyz_or_xy_seq_or_xyz_seq[0], numbers.Number):
        # Then xy_or_xyz_or_xy_seq_or_xyz_seq is a single point.
        return coord2deg_aux(xy_or_xyz_or_xy_seq_or_xyz_seq, idx)
    else:
        # Else xy_or_xyz_or_xy_seq_or_xyz_seq is a sequence of points.
        output_list = []
        for xy_or_xyz in xy_or_xyz_or_xy_seq_or_xyz_seq:
            output_list.append(coord2deg_aux(xy_or_xyz, idx))
        return output_list


def x2deg(xy_or_xyz_or_xy_seq_or_xyz_seq):
    """
    Converts the x-coordinates of a point or sequence of points from radians to degrees.

    Parameters
    ----------
    xy_or_xyz_or_xy_seq_or_xyz_seq : np.ndarray or Iterable[np.ndarray]
        A single point (2D or 3D) or a sequence of points.

    Returns
    -------
    list[np.ndarray]
        A list of points with the x-coordinates converted to degrees.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return coord2deg(xy_or_xyz_or_xy_seq_or_xyz_seq, 0)


def y2deg(xy_or_xyz_or_xy_seq_or_xyz_seq):
    """
    Converts the y-coordinates of a point or sequence of points from radians to degrees.

    Parameters
    ----------
    xy_or_xyz_or_xy_seq_or_xyz_seq : np.ndarray or Iterable[np.ndarray]
        A single point (2D or 3D) or a sequence of points.

    Returns
    -------
    list[np.ndarray]
        A list of points with the y-coordinates converted to degrees.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return coord2deg(xy_or_xyz_or_xy_seq_or_xyz_seq, 1)


def z2deg(xyz_or_xyz_seq):
    """
    Converts the z-coordinates of a point or sequence of points from radians to degrees.

    Parameters
    ----------
    xyz_or_xyz_seq : np.ndarray or Iterable[np.ndarray]
        A single point (3D) or a sequence of points.

    Returns
    -------
    list[np.ndarray]
        A list of points with the z-coordinates converted to degrees.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return coord2deg(xyz_or_xyz_seq, 2)


def p2deg(pq_or_pq_seq):
    """
    Converts the p-coordinates of a point or sequence of points from radians to degrees.

    Parameters
    ----------
    pq_or_pq_seq : np.ndarray or Iterable[np.ndarray]
        A single point (2D) or a sequence of points.

    Returns
    -------
    list[np.ndarray]
        A list of points with the p-coordinates converted to degrees.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return coord2deg(pq_or_pq_seq, 0)


def q2deg(pq_or_pq_seq):
    """
    Converts the q-coordinates of a point or sequence of points from radians to degrees.

    Parameters
    ----------
    pq_or_pq_seq : np.ndarray or Iterable[np.ndarray]
        A single point (2D) or a sequence of points.

    Returns
    -------
    list[np.ndarray]
        A list of points with the q-coordinates converted to degrees.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return coord2deg(pq_or_pq_seq, 1)


@overload
def normalize(angle: float) -> float:
    """Normalizes a single angle to the range [0, 2π]."""
    # "ChatGPT 4o" assisted with generating this docstring.


@overload
def normalize(angles: npt.NDArray[np.float_] | Iterable) -> npt.NDArray[np.float_]:
    """Normalizes an array of angles to the range [0, 2π]."""
    # "ChatGPT 4o" assisted with generating this docstring.


def normalize(angle_or_angles: float | npt.NDArray[np.float_] | Iterable) -> float | npt.NDArray[np.float_]:
    """
    Adjusts the given angle or angles to be in the range [0, 2π].

    Note that because this function operates on floating point math,
    the result may not be exact (e.g., a value of -1e-16 could return 2π).

    Parameters
    ----------
    angle_or_angles : float or :py:meth:`npt.NDArray[np.float_]` or Iterable
        A single angle or an array/iterable of angles to normalize.

    Returns
    -------
    float or py:meth:`npt.NDArray[np.float_]`
        The normalized angle or array of normalized angles.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    if isinstance(angle_or_angles, np.ndarray):
        angles: np.ndarray = angle_or_angles
        # limit to the range +-pi
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        # for values < 0, add 2pi
        neg_angles = angles < 0
        angles[neg_angles] += np.pi * 2
        return angles
    elif isinstance(angle_or_angles, Iterable):
        return normalize(np.array(angle_or_angles))
    else:
        angle: float = angle_or_angles
        while angle < 0:
            angle += 2 * math.pi
        while angle > 2 * math.pi:
            angle -= 2 * math.pi
        return angle


def angle2_minus_angle_1(angle_1, angle_2):
    """
    Calculates the signed small angle between two input angles.

    This function corrects for cases where the angles wrap around 2π.

    Parameters
    ----------
    angle_1 : float
        The first angle in radians.
    angle_2 : float
        The second angle in radians.

    Returns
    -------
    float
        The signed angle difference between angle_2 and angle_1, adjusted for wrapping.

    Examples
    --------
    - angle_1 = 5 degrees, angle_2 = 10 degrees  ==>  5 degrees
    - angle_1 = 355 degrees, angle_2 = 10 degrees  ==>  15 degrees
    - angle_1 = -5 degrees, angle_2 = -10 degrees  ==>  -5 degrees
    - angle_1 = 5 degrees, angle_2 = -10 degrees  ==>  -15 degrees
    """
    two_pi = 2 * math.pi
    diff = angle_2 - angle_1
    while diff > two_pi:
        diff -= two_pi
    while diff < -two_pi:
        diff += two_pi
    return diff
