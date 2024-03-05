import numpy as np
from numpy import ndarray


def rotate(points: ndarray, rot_vecs: ndarray) -> ndarray:
    """
    Rotates vectors according to given Rodrigez vector. N1 and N2 must
    broadcast together. Will return (1, 3) or (N, 3) vector.

    Parameters
    ----------
    points : ndarray
        (N1, 3) or (3,) array. Point(s) to rotate.
    rot_vecs : ndarray
        (N2, 3) or (3,) array. Rotation vector(s).

    Returns
    -------
    ndarray
        (N, 3) array. Rotated point(s).

    """
    # Perform checks
    points = points.reshape((-1, 3))
    rot_vecs = rot_vecs.reshape((-1, 3))

    # Perform rotation
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return (
        cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    )
