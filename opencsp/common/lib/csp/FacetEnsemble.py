"""Rigid ensemble of facets"""
from typing import Callable
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.OpticOrientation import OpticOrientation
from opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract import VisualizeOrthorectifiedSlopeAbstract
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror


class FacetEnsemble(RayTraceable, VisualizeOrthorectifiedSlopeAbstract):
    """Ensemble of facets that holds Facet objects.

    """

    def __init__(self, facets: list[Facet]):
        """Instantiates FacetEnsemble class

        Parameters
        ----------
        facets : list[Facet]
            List of located facets to place in facet ensemble.
        """
        self.facets = facets
        self.num_facets = len(facets)

        self.ori = OpticOrientation()

        self.pointing_function = None

    @property
    def transform_mirror_base_to_child(self) -> list[TransformXYZ]:
        return [f.transform_mirror_base_to_parent for f in self.facets]

    @property
    def transform_mirror_base_to_base(self) -> list[TransformXYZ]:
        return [self.ori.transform_child_to_base * trans for trans in self.transform_mirror_base_to_child]

    @property
    def transform_mirror_base_to_parent(self) -> list[TransformXYZ]:
        return [self.ori.transform_child_to_parent * trans for trans in self.transform_mirror_base_to_child]

    @property
    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        """Returns bounding box aligned to XY axes in ensemble's child coordinate
        reference frame.

        Returns
        -------
        tuple[float, float, float, float]
            Left, right, bottom, top. Ensemble's child coordinate reference frame.
        """
        # Get XYZ locations of all points making up mirror region
        xyz = []  # ensemble child
        for facet in self.facets:
            # Get all mirror region vertices
            points_xy = Pxy.merge([loop.vertices for loop in facet.mirror.region.loops])  # mirror base
            points_z = facet.mirror.surface_displacement_at(points_xy)  # mirror base
            points_xyz = Pxyz((points_xy.x, points_xy.y, points_z))  # mirror base
            points_xyz = facet.transform_mirror_base_to_parent.apply(points_xyz)  # ensemble child
            xyz.append(points_xyz)  # ensemble child
        xyz = Pxyz.merge(xyz)  # ensemble child

        # Find bounding box
        return xyz.x.min(), xyz.x.max(), xyz.y.min(), xyz.y.max()  # ensemble child

    def survey_of_points(self, resolution: int, resolution_type: str = 'pixelX', random_seed: int | None = None) -> tuple[Pxyz, Vxyz]:
        # Get sample point locations (z=0 plane in "ensemble child" reference frame)
        bbox = self.axis_aligned_bounding_box  # left, right, bottom, top
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        region = RegionXY(LoopXY.from_rectangle(bbox[0], bbox[2], width, height))  # ensemble child
        points_samp_xy = region.points_sample(resolution, resolution_type, random_seed)  # ensemble child
        points_samp_xyz = Vxyz((points_samp_xy.x, points_samp_xy.y, np.zeros(len(points_samp_xy))))  # ensemble child

        idx_facet = 0
        points_list = []
        normals_list = []
        for idx_facet in range(self.num_facets):
            # Filter points that are inside mirror region
            points_mirror_base = self.transform_mirror_base_to_child[idx_facet].inv().apply(points_samp_xyz)  # mirror base
            mask = self.facets[idx_facet].mirror.region.is_inside_or_on_border(points_mirror_base.projXY())
            points_mirror_base = points_mirror_base[mask]  # mirror base

            # Calculate points and normals at sample locations
            points_mirror_base = self.facets[idx_facet].mirror.location_at(points_mirror_base.projXY())  # mirror base
            normals_mirror_base = self.facets[idx_facet].mirror.surface_norm_at(points_mirror_base.projXY())  # mirror base

            # Convert from mirror to world reference frame
            points_list.append(self.transform_mirror_base_to_parent[idx_facet].apply(points_mirror_base))  # ensemble parent
            normals_list.append(normals_mirror_base.rotate(self.transform_mirror_base_to_parent[idx_facet].R))  # ensemble parent

        points = Vxyz.merge(points_list)
        normals = Vxyz.merge(normals_list)
        return points, normals  # world coordinates

    def orthorectified_slope_array(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        # Get sample points
        x_mat, y_mat = np.meshgrid(x_vec, y_vec)  # ensemble child
        z_mat = np.zeros(x_mat.shape)  # ensemble child
        points_samp = Vxyz((x_mat, y_mat, z_mat))  # ensemble child

        slope_data = np.zeros((2, len(points_samp))) * np.nan  # ensemble child
        for idx_facet in range(self.num_facets):
            # Get mask of points on mirror
            points_samp_mirror = self.transform_mirror_base_to_child[idx_facet].inv().apply(points_samp)  # mirror base
            mask = self.facets[idx_facet].mirror.in_bounds(points_samp_mirror.projXY())
            points_samp_mirror = points_samp_mirror[mask]

            # Get normal vectors
            normals = self.facets[idx_facet].mirror.surface_norm_at(points_samp_mirror.projXY())  # mirror base
            normals.rotate_in_place(self.transform_mirror_base_to_child[idx_facet].R)  # ensemble child

            # Calculate slopes and output as 2D array
            slope_data[:, mask] = -normals.data[:2] / normals.data[2:3]  # ensemble child

        slope_data = np.reshape(slope_data, (2, y_vec.size, x_vec.size))  # ensemble child
        return slope_data  # ensemble child

    def draw(self, view: View3d, mirror_style: RenderControlMirror, transform: list[TransformXYZ] | None = None) -> None:
        """
        Draws facet ensemble onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A view 3d object that holds the figure. 
        mirror_styles : RenderControlMirror
            Holds attibutes about the 3d graph.
        transform : list[TransformXYZ] | None
            List of 3d transforms for each facet in ensemble.
            Used to position points in the mirror's base coordinate
            reference frame in space. If None, defaults to position points
            in the ensemble's parent coordinate reference frame.
        """
        for idx, facet in enumerate(self.facets):
            if transform is None:
                transform_facet = self.transform_mirror_base_to_parent[idx]
            else:
                transform_facet = transform[idx]
            facet.draw(view, mirror_style, transform_facet)

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
        """Sets current facet ensemble canting (i.e. sets
        self.ori.transform_child_to_base using the given arguments.
        """
        if self.pointing_function is None:
            raise ValueError('self.pointing_function is not defined. Use self.define_pointing_function.')

        self.ori.transform_child_to_base = self.pointing_function(*args)

    @classmethod
    def generate_az_el(cls, facets: list[Facet]) -> 'FacetEnsemble':
        """Generates HeliostatCantable object defined by a simple azimuth then elevation
        canting strategy. The "pointing_function" accessed by self.set_pointing
        has the following inputs
            - az - float - azimuth angle (rotation about z axis) in radians
            - el - float - elevation angle (rotation about x axis) in radians
        """
        def pointing_function(az: float, el: float) -> TransformXYZ:
            r = Rotation.from_euler('zx', [az, el], degrees=False)
            return TransformXYZ.from_R(r)

        # Create heliostat
        heliostat = cls(facets)
        heliostat.define_pointing_function(pointing_function)

        return heliostat

    @classmethod
    def generate_rotation_defined(cls, facets: list[Facet]) -> 'FacetEnsemble':
        """Generates HeliostatCantable object defined by a given scipy Rotation object.
        The "pointing_function" accessed by self.set_pointing has the following input
            - rotation - scipy.spatial.transform.Rotation
        """
        def pointing_function(rotation: Rotation) -> TransformXYZ:
            return TransformXYZ.from_R(rotation)

        # Create heliostat
        heliostat = cls(facets)
        heliostat.define_pointing_function(pointing_function)

        return heliostat
