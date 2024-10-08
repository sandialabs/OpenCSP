"""Rigid ensemble of facets"""

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
from opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract import VisualizeOrthorectifiedSlopeAbstract
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlFacetEnsemble import RenderControlFacetEnsemble


class FacetEnsemble(RayTraceable, VisualizeOrthorectifiedSlopeAbstract, OpticOrientationAbstract):
    """Ensemble of facets that holds Facet objects."""

    def __init__(self, facets: list[Facet]):
        """Instantiates FacetEnsemble class

        Parameters
        ----------
        facets : list[Facet]
            List of located facets to place in facet ensemble.
        """
        self.facets = facets
        OpticOrientationAbstract.__init__(self)

        self.num_facets = len(facets)
        self.pointing_function = None

    @property
    def facet_positions(self) -> Pxyz:
        """The locations of the facets relative to the `FacetEnsemble` origin."""
        return Pxyz.merge([facet._self_to_parent_transform.apply(Pxyz.origin()) for facet in self.facets])

    @property
    def transform_mirror_to_self(self):
        """List of transforms from Mirror to self"""
        return [facet.mirror.get_transform_relative_to(self) for facet in self.facets]

    # override from VisualizeOrthorectifiedSlopeAbstract
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
        xyz = []  # facet frame
        for facet in self.facets:
            # Get all mirror region vertices
            points_xy = Pxy.merge([loop.vertices for loop in facet.mirror.region.loops])  # mirror frame
            points_z = facet.mirror.surface_displacement_at(points_xy)  # mirror frame
            points_xyz = Pxyz((points_xy.x, points_xy.y, points_z))  # mirror frame
            points_xyz = facet.mirror.get_transform_relative_to(self).apply(points_xyz)  # facet frame
            xyz.append(points_xyz)  # facet frame
        xyz = Pxyz.merge(xyz)  # facet frame

        # Find bounding box
        return xyz.x.min(), xyz.x.max(), xyz.y.min(), xyz.y.max()  # facet frame

    # override from OpticOrientationAbstract
    @property
    def children(self):
        return self.facets

    def lookup_facet(self, facet_name: str) -> Facet:
        """Returns the first Facet in the FacetEnsemble that matches the given name.
        If there are no facets that match the given name it throws a KeyError."""
        # Check input.
        matching_facets = filter(lambda facet: facet.name == facet_name, self.facets)
        first_matching = next(matching_facets, None)
        if first_matching is None:
            raise KeyError(f"No heliostat with the name '{facet_name}' appears in the SolarField {self}.")
        # Return.
        return first_matching

    # override from OpticOrientationAbstract
    def _add_child_helper(self, new_child: OpticOrientationAbstract):
        self.facets.append(new_child)

    def survey_of_points(self, resolution: Resolution) -> tuple[Pxyz, Vxyz]:
        # Get sample point locations (z=0 plane in "child" reference frame)
        resolution.resolve_in_place(self.axis_aligned_bounding_box)
        return self._survey_of_points_helper(resolution, TransformXYZ.identity())

    def _survey_of_points_helper(
        self, given_resolution: Resolution, frame_transform: TransformXYZ
    ) -> tuple[Pxyz, Vxyz]:
        resolution = given_resolution.change_frame_and_copy(frame_transform)
        resolution.resolve_in_place(self.axis_aligned_bounding_box)
        points, normals = [], []
        for facet in self.facets:
            facet_points, facet_normals = facet._survey_of_points_helper(
                resolution, facet._self_to_parent_transform.inv()
            )
            points.append(facet_points)
            normals.append(facet_normals)

        return Pxyz.merge(points), Vxyz.merge(normals)

    # override from VisualizeOrthorectifiedSlopeAbstract

    def orthorectified_slope_array(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        # Get sample points
        x_mat, y_mat = np.meshgrid(x_vec, y_vec)  # ensemble child
        z_mat = np.zeros(x_mat.shape)  # ensemble child
        points_samp = Vxyz((x_mat, y_mat, z_mat))  # ensemble child

        slope_data = np.zeros((2, len(points_samp))) * np.nan  # ensemble child
        for idx_facet, facet in enumerate(self.facets):
            # Get mask of points on mirror
            points_samp_mirror = self.get_transform_relative_to(facet.mirror).apply(points_samp)  # mirror base
            mask = self.facets[idx_facet].mirror.in_bounds(points_samp_mirror.projXY())
            points_samp_mirror = points_samp_mirror[mask]

            # Get normal vectors
            normals = self.facets[idx_facet].mirror.surface_norm_at(points_samp_mirror.projXY())  # mirror base
            normals.rotate_in_place(facet.mirror.get_transform_relative_to(self).R)  # ensemble child

            # Calculate slopes and output as 2D array
            slope_data[:, mask] = -normals.data[:2] / normals.data[2:3]  # ensemble child

        slope_data = np.reshape(slope_data, (2, y_vec.size, x_vec.size))  # ensemble child
        return slope_data  # ensemble child

    def draw(
        self, view: View3d, facet_ensemble_style: RenderControlFacetEnsemble = None, transform: TransformXYZ = None
    ) -> None:
        """
        Draws facet ensemble onto a View3d object.

        Parameters:
        -----------
        view : View3d
            A View3d object that holds the figure.
        facet_styles : RenderControlFacetEnsemble
            Holds information on how to draw each facet, inclusing
            information on how to draw specific facets.
        transform : TransformXYZ | None
            List of 3d transforms for each facet in ensemble.
            Used to position points in the FacetEnsemble's base coordinate
            reference frame in space. If None, defaults to position points
            in the ensemble's global coordinate reference frame.
        """

        if facet_ensemble_style is None:
            facet_ensemble_style = RenderControlFacetEnsemble()

        if transform is None:
            transform = self.self_to_global_tranformation

        origin = transform.apply(Pxyz.origin())
        normal_vector = transform.apply(Vxyz([0, 0, 1]) * facet_ensemble_style.normal_vector_length)

        # individual facets
        if facet_ensemble_style.draw_facets:
            for facet in self.facets:
                transform_facet = transform * facet._self_to_parent_transform
                facet_style = facet_ensemble_style.get_facet_style(facet.name)
                facet.draw(view, facet_style, transform_facet)

        # origin of the facet ensemble
        if facet_ensemble_style.draw_centroid:
            view.draw_single_Pxyz(origin)

        # pointing vector of the facet ensemble
        if facet_ensemble_style.draw_normal_vector:
            view.draw_Vxyz(Vxyz.merge([origin, normal_vector]), style=facet_ensemble_style.normal_vector_style)

        if facet_ensemble_style.draw_outline:
            left, right, top, bottom = self.axis_aligned_bounding_box
            corners = Pxyz([[left, left, right, right], [top, bottom, bottom, top], [0, 0, 0, 0]])
            corners_moved = transform.apply(corners)
            view.draw_Vxyz(corners_moved, close=True, style=facet_ensemble_style.outline_style)

    def set_facet_transform_list(self, transformations: list[TransformXYZ]):
        """
        Combines the `set_facet_positions` and `set_facet_canting` functions into a single action.
        """
        for transformation, facet in zip(transformations, self.facets):
            facet._self_to_parent_transform = transformation

    def set_facet_positions(self, positions: Pxyz):
        """Sets the positions of the facets relative to the ensemble.
        NOTE: Will remove previously set facet canting rotations
        """
        if len(positions) != len(self.facets):
            raise ValueError(
                f"This FacetEnsemble contains {len(self.facets)} and"
                f" the argument only gave {len(positions)} positions."
            )
        for pos, facet in zip(positions, self.facets):
            facet: Facet
            pos: Pxyz
            facet._self_to_parent_transform = TransformXYZ.from_V(pos)

    def set_facet_canting(self, canting_rotations: list[Rotation]):
        """Sets facet canting relative to ensemble.
        NOTE: Will remove previously set facet positionals
        """
        if len(canting_rotations) != len(self.facets):
            raise ValueError(
                f"This FacetEnsemble contains {len(self.facets)} and"
                f" the argument only gave {len(canting_rotations)} rotations."
            )
        for facet, pos, canting in zip(self.facets, self.facet_positions, canting_rotations):
            facet: Facet
            pos: Pxyz
            canting: Rotation
            facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, pos)
