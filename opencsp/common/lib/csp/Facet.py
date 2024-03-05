"""Facet class inherited by all facet classes"""
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
from opencsp.common.lib.csp.OpticOrientation import OpticOrientation
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract import (
    VisualizeOrthorectifiedSlopeAbstract,
)
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror


class Facet(RayTraceable, VisualizeOrthorectifiedSlopeAbstract):
    """Facet representation that contains a MirrorAbstract object."""

    def __init__(self, mirror: MirrorAbstract) -> 'Facet':
        """Instantiates Facet class

        Parameters
        ----------
        mirror : MirrorAbstract
            Mirror object held inside Facet
        """
        self.mirror = mirror
        self.ori = OpticOrientation()
        self.pointing_function = None

    @property
    def transform_mirror_base_to_child(self) -> TransformXYZ:
        return self.mirror.ori.transform_base_to_parent

    @property
    def transform_mirror_base_to_base(self) -> TransformXYZ:
        return self.ori.transform_child_to_base * self.transform_mirror_base_to_child

    @property
    def transform_mirror_base_to_parent(self) -> TransformXYZ:
        return self.ori.transform_child_to_parent * self.transform_mirror_base_to_child

    @property
    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        """Returns bounding box aligned to XY axes in facet's child coordinate
        reference frame.

        Returns
        -------
        tuple[float, float, float, float]
            Left, right, bottom, top. Facet's child coordinate reference frame.
        """
        # Get XYZ locations of all points making up mirror region
        points_xy = Vxy.merge(
            [loop.vertices for loop in self.mirror.region.loops]
        )  # mirror base
        points_z = self.mirror.surface_displacement_at(points_xy)  # mirror base
        points_xyz = Vxyz((points_xy.x, points_xy.y, points_z))  # mirror base

        # Transform "mirror base" to "facet child" coordinates
        xyz = self.transform_mirror_base_to_child.apply(points_xyz)  # child

        # Find bounding box
        return xyz.x.min(), xyz.x.max(), xyz.y.min(), xyz.y.max()  # child

    def survey_of_points(
        self,
        resolution: int,
        resolution_type: str = 'pixelX',
        random_seed: int | None = None,
    ) -> tuple[Pxyz, Vxyz]:
        # Get sample point locations (z=0 plane in "child" reference frame)
        bbox = self.axis_aligned_bounding_box  # left, right, bottom, top, "child"
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        region = RegionXY(
            LoopXY.from_rectangle(bbox[0], bbox[2], width, height)
        )  # facet child
        points_child_xy = region.points_sample(
            resolution, resolution_type, random_seed
        )  # facet child
        points_child_xyz = Vxyz(
            (points_child_xy.x, points_child_xy.y, np.zeros(len(points_child_xy)))
        )  # facet child

        # Filter points that are inside mirror region
        points_mirror_base_xyz = self.transform_mirror_base_to_child.inv().apply(
            points_child_xyz
        )  # mirror base
        mask = self.mirror.region.is_inside_or_on_border(
            points_mirror_base_xyz.projXY()
        )
        points_mirror_base_xyz = points_mirror_base_xyz[mask]  # mirror base

        # Calculate points and normals at sample locations
        points_mirror_base, normals_mirror_base = self.mirror.point_and_normal_in_space(
            points_mirror_base_xyz.projXY()
        )  # facet child
        points_mirror_base = self.transform_mirror_base_to_child.inv().apply(
            points_mirror_base
        )  # mirror base
        normals_mirror_base.rotate_in_place(
            self.transform_mirror_base_to_child.inv().R
        )  # mirror base

        # Convert from mirror to fixed reference frame
        points = self.transform_mirror_base_to_parent.apply(
            points_mirror_base
        )  # facet parent
        normals = normals_mirror_base.rotate(
            self.transform_mirror_base_to_parent.R
        )  # facet parent

        return points, normals  # facet parent

    def orthorectified_slope_array(
        self, x_vec: np.ndarray, y_vec: np.ndarray
    ) -> np.ndarray:
        """Returns X and Y surface slopes in ndarray format given X and Y
        sampling axes in the facet's child coordinate reference frame.

        Parameters
        ----------
        x_vec/y_vec : ndarray
            X and Y grid sampling vectors in facet's child coordinate reference frame

        Returns
        -------
        ndarray
            X and Y slope images of shape: (2, y_vec.size, x_vec.size), in the facet's
            child coordinate reference system.
        """
        # Get sample points
        x_mat, y_mat = np.meshgrid(x_vec, y_vec)  # facet child
        z_mat = np.zeros(x_mat.shape)  # facet child
        points_samp = Vxyz((x_mat, y_mat, z_mat))  # facet child

        # Get mask of points on mirror
        points_samp_mirror = self.transform_mirror_base_to_child.inv().apply(
            points_samp
        )  # mirror base
        mask = self.mirror.in_bounds(points_samp_mirror.projXY())
        points_samp_mirror = points_samp_mirror[mask]

        # Get normal vectors
        normals = self.mirror.surface_norm_at(
            points_samp_mirror.projXY()
        )  # mirror base
        normals.rotate_in_place(self.transform_mirror_base_to_child.R)  # facet child

        # Calculate slopes and output as 2D array
        slope_data = np.zeros((2, len(points_samp))) * np.nan  # facet child
        slope_data[:, mask] = -normals.data[:2] / normals.data[2:3]  # facet child
        slope_data = np.reshape(slope_data, (2, y_vec.size, x_vec.size))  # facet child
        return slope_data  # facet child

    def draw(
        self,
        view: View3d,
        mirror_style: RenderControlMirror,
        transform: TransformXYZ | None = None,
    ) -> None:
        """
        Draws facet mirror onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure.
        mirror_styles : RenderControlMirror
            Holds attibutes about the 3d graph.
        transform : TransformXYZ
            3d transform used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the facet's parent coordinate reference frame.
        """
        if transform is None:
            transform = self.transform_mirror_base_to_parent
        self.mirror.draw(view, mirror_style, transform)

    def set_position_in_space(self, translation: Pxyz, rotation: Rotation) -> None:
        self.ori.transform_base_to_parent = TransformXYZ.from_R_V(rotation, translation)

    def define_pointing_function(self, func: Callable) -> None:
        """Sets the canting function to use. I.e., defines the
        "set_pointing" function.

        Parameters
        ----------
        func : Callable
            Function that returns a "child to base" TransformXYZ object.
        """
        self.pointing_function = func

    def set_pointing(self, *args) -> None:
        """Sets current facet canting (i.e. sets
        self.ori.transform_child_to_base using the given arguments.
        """
        if self.pointing_function is None:
            raise ValueError(
                'self.pointing_function is not defined. Use self.define_pointing_function.'
            )

        self.ori.transform_child_to_base = self.pointing_function(*args)

    @classmethod
    def generate_az_el(cls, mirror: MirrorAbstract) -> 'Facet':
        """Generates Facet object defined by a simple azimuth then elevation
        canting strategy. The "pointing_function" accessed by self.set_pointing
        has the following inputs
            - az - float - azimuth angle (rotation about z axis) in radians
            - el - float - elevation angle (rotation about x axis) in radians
        """

        def pointing_function(az: float, el: float) -> TransformXYZ:
            r = Rotation.from_euler('zx', [az, el], degrees=False)
            return TransformXYZ.from_R(r)

        # Create facet
        facet = cls(mirror)
        facet.define_pointing_function(pointing_function)

        return facet

    @classmethod
    def generate_rotation_defined(cls, mirror: MirrorAbstract) -> 'Facet':
        """Generates FacetCantable object defined by a given scipy Rotation object.
        The "pointing_function" accessed by self.set_pointing has the following input
            - rotation - scipy.spatial.transform.Rotation - rotation object
        """

        def pointing_function(rotation: Rotation) -> TransformXYZ:
            return TransformXYZ.from_R(rotation)

        # Create facet
        facet = cls(mirror)
        facet.define_pointing_function(pointing_function)

        return facet
