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
    pt = xy_or_xyz.copy()
    pt[idx] = np.rad2deg(pt[idx])
    return pt


def coord2deg(xy_or_xyz_or_xy_seq_or_xyz_seq, idx):
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
    return coord2deg(xy_or_xyz_or_xy_seq_or_xyz_seq, 0)


def y2deg(xy_or_xyz_or_xy_seq_or_xyz_seq):
    return coord2deg(xy_or_xyz_or_xy_seq_or_xyz_seq, 1)


def z2deg(xyz_or_xyz_seq):
    return coord2deg(xyz_or_xyz_seq, 2)


def p2deg(pq_or_pq_seq):
    return coord2deg(pq_or_pq_seq, 0)


def q2deg(pq_or_pq_seq):
    return coord2deg(pq_or_pq_seq, 1)


@overload
def normalize(angle: float) -> float:
    pass


@overload
def normalize(angles: npt.NDArray[np.float_] | Iterable) -> npt.NDArray[np.float_]:
    pass


def normalize(angle_or_angles: float | npt.NDArray[np.float_] | Iterable) -> float | npt.NDArray[np.float_]:
    """Adjusts the given angle_or_angles to be in the range 0-2π.
    Note that because this function operates on floating point math,
    your answer is not guaranteed to be exact (for example, a value
    of -1e-16 could return 2π)."""
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
    Returns the signed small angle between the two input angles.
    In most cases, this will simply be (angle_2 - angle_1).
    However, we correct for cases where the angles wrap around 2pi.
    Examples:
        angle 1 =   5deg, angle_2 =  10 deg  ==>   5 deg.
        angle 1 = 355deg, angle_2 =  10 deg  ==>  15 deg.
        angle 1 =  -5deg, angle_2 = -10 deg  ==>  -5 deg.
        angle 1 =   5deg, angle_2 = -10 deg  ==> -15 deg.
    """
    two_pi = 2 * math.pi
    diff = angle_2 - angle_1
    while diff > two_pi:
        diff -= two_pi
    while diff < -two_pi:
        diff += two_pi
    return diff
