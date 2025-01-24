"""

Specifying a view of a three-dimensional space.

Options include general 3-d (allowing interactive rotation), xy, xz, yz, general section plane.

In all of the 2-d cases, there is an embedded (p,q) parameter space, which corresponds to
different projections of the 3-d coordinates.



"""

import numpy as np


# COMMON VIEWS


def view_spec_3d() -> dict:
    spec = {}
    spec['type'] = '3d'
    return spec


def view_spec_xy() -> dict:
    spec = {}
    spec['type'] = 'xy'
    return spec


def view_spec_xz() -> dict:
    spec = {}
    spec['type'] = 'xz'
    return spec


def view_spec_yz() -> dict:
    spec = {}
    spec['type'] = 'yz'
    return spec


def view_spec_im() -> dict:
    spec = {}
    spec['type'] = 'image'
    return spec


def view_spec_camera(camera, camera_xform) -> dict:
    spec = {}
    spec['type'] = 'camera'
    spec['camera'] = camera
    spec['camera_xform'] = camera_xform
    return spec


# XYZ <---> PQ CONVERSION


def xyz2pqw(xyz, view_spec):
    # Enforces the right-hand rule in all conversions.
    # That is, [p,q,w] is a right-handed coordinate system.
    if view_spec['type'] == '3d':
        return xyz
    elif view_spec['type'] == 'xy':
        return [xyz[0], xyz[1], xyz[2]]
    elif view_spec['type'] == 'xz':
        return [xyz[0], xyz[2], -xyz[1]]
    elif view_spec['type'] == 'yz':
        return [xyz[1], xyz[2], xyz[0]]
    elif view_spec['type'] == 'vplane':
        # Fetch section coordinate system.
        origin_xyz = np.array(view_spec['origin_xyz'])  # Make arrays so we can do simple vactor math.
        p_uxyz = np.array(view_spec['p_uxyz'])  #
        q_uxyz = np.array(view_spec['q_uxyz'])  #
        w_uxyz = np.array(view_spec['w_uxyz'])  #
        # Construct vector from origin to xyz.
        vxyz = np.array(xyz) - origin_xyz
        # Construct (p,q,w) components.
        p = vxyz.dot(p_uxyz)
        q = vxyz.dot(q_uxyz)
        w = vxyz.dot(w_uxyz)
        return [p, q, w]
    elif view_spec['type'] == 'camera':
        camera = view_spec['camera']
        camera_xform = view_spec['camera_xform']
        pq = camera_xform.pq_or_none(camera, xyz)
        if pq == None:
            # (x,y,z) point is behind camera and thus should not be drawn.
            return None
        else:
            p = pq[0]
            q = pq[1]
            w = 0
            # Return.
            return [p, q, w]
    else:
        print("ERROR: In xyz2pqw(), unrecognized view_spec['type'] = '" + str(view_spec['type']) + "' encountered.")
        assert False


def xyz2pq(xyz, view_spec):
    pqw = xyz2pqw(xyz, view_spec)
    if pqw == None:
        return None
    else:
        return pqw[0:2]


def pqw2xyz(pqw, view_spec):
    # Assumes the right-hand rule for all conversions.
    # That is, [p,q,w] is viewed as a right-handed coordinate system.
    if view_spec['type'] == '3d':
        return pqw
    elif view_spec['type'] == 'xy':
        return [pqw[0], pqw[1], pqw[2]]
    elif view_spec['type'] == 'xz':
        return [pqw[0], -pqw[2], pqw[1]]
    elif view_spec['type'] == 'yz':
        return [pqw[2], pqw[0], pqw[1]]
    elif view_spec['type'] == 'vplane':
        # Fetch section coordinate system.
        origin_xyz = np.array(view_spec['origin_xyz'])  # Make arrays so we can do simple vactor math.
        p_uxyz = np.array(view_spec['p_uxyz'])  #
        q_uxyz = np.array(view_spec['q_uxyz'])  #
        w_uxyz = np.array(view_spec['w_uxyz'])  #
        # Extract (p,q,w) components.
        p = pqw[0]
        q = pqw[1]
        w = pqw[2]
        # Construct (x,y,z) components.
        xyz = origin_xyz + (p * p_uxyz) + (q * q_uxyz) + (w * w_uxyz)
        return [xyz[0], xyz[1], xyz[2]]
    else:
        print("ERROR: In pqw2xyz(), unrecognized view_spec['type'] = '" + str(view_spec['type']) + "' encountered.")
        assert False


def pq2xyz(pq, view_spec):
    pqw = pq.copy()
    pqw.append(0)
    return pqw2xyz(pqw, view_spec)
