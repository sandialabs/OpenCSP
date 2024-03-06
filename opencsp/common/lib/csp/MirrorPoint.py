"""Mirror class representing mirrors with scattered surface point
locations.
"""
from typing import Literal
from warnings import warn

import numpy as np
import scipy.interpolate as interp

import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
from opencsp.common.lib.geometry.FunctionXYDiscrete import FunctionXYDiscrete as FXYD
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror


class MirrorPoint(MirrorAbstract):
    def __init__(
        self,
        surface_points: Pxyz,
        normal_vectors: Uxyz,
        shape: RegionXY,
        interpolation_type: Literal[
            'given', 'bilinear', 'clough_tocher', 'nearest'
        ] = 'nearest',
    ) -> None:
        """Class representing a mirror defined by discrete, scattered points
        and corresponding normal vectors.

        Parameters
        ----------
        surface_points : Pxyz
            XYZ points on surface of mirror
        normal_vectors : Uxyz
            XYZ normal vectors corresponding to surface_points
        interpolation_type : str
            Interpolation type:
                - 'given' - Uses given XY points in look-up table
                - 'bilinear' - bilinear interpolation
                - 'clough_tocher' - Clough-Tocher interpolation
                - 'nearest' - nearest neighbor interpolation
        shape : RegionXY
            XY outline of mirror

        Raises
        ------
        ValueError
            If not all normal vectors have a positive z component
        """
        super().__init__(shape)  # initalizes the attributes universal to all mirrors

        # Save surface points and normal vectors
        self.surface_points = surface_points
        self.normal_vectors = normal_vectors

        # assert that all values of normal_vectors are pointing up:
        # this requires that the mirror can be defined as a function and is oriented face up.
        if normal_vectors.z.min() < 0:
            raise ValueError("All normal vectors must have a positive z component.")

        # Define interpolation type
        self._define_interpolation(interpolation_type)

    def _define_interpolation(
        self,
        interpolation_type: Literal['given', 'bilinear', 'clough_tocher', 'nearest'],
    ) -> None:
        """Defines the interpolation type to use

        Parameters
        ----------
        interpolation_type : str
            Interpolation type:
                - 'given' - Uses given XY points in look-up table
                - 'bilinear' - bilinear interpolation
                - 'clough_tocher' - Clough-Tocher interpolation
                - 'nearest' - nearest neighbor interpolation

        Raises
        ------
        ValueError
            If given interpolation type is not supported.
        """
        # Interpolate
        if interpolation_type == 'bilinear':
            # Z coordinate interpolation object
            points_xy = self.surface_points.projXY().data.T  # Nx2 array
            Z = self.surface_points.z
            self.surface_function = interp.LinearNDInterpolator(points_xy, Z, np.nan)
            # Normal vector interpolation object
            Z_N = self.normal_vectors.data.T
            self.normals_function = interp.LinearNDInterpolator(points_xy, Z_N, np.nan)
        elif interpolation_type == 'clough_tocher':
            # Z coordinate interpolation object
            points_xy = self.surface_points.projXY().data.T  # Nx2 array
            Z = self.surface_points.z
            self.surface_function = interp.CloughTocher2DInterpolator(
                points_xy, Z, np.nan
            )
            # Normal vector interpolation object
            Z_N = self.normal_vectors.data.T
            self.normals_function = interp.CloughTocher2DInterpolator(
                points_xy, Z_N, np.nan
            )
        elif interpolation_type == 'nearest':
            # Z coordinate interpolation object
            points_xy = self.surface_points.projXY().data.T  # Nx2 array
            Z = self.surface_points.z
            self.surface_function = interp.NearestNDInterpolator(points_xy, Z)
            # Normal vector interpolatin object
            Z_N = self.normal_vectors.data.T
            self.normals_function = interp.NearestNDInterpolator(points_xy, Z_N)
        elif interpolation_type == 'given':
            # Z coordinate lookup function
            points_lookup = {
                (x, y): z
                for x, y, z in zip(
                    self.surface_points.x, self.surface_points.y, self.surface_points.z
                )
            }
            self.surface_function = FXYD(points_lookup)
            # Normal vector lookup function
            normals_lookup = {
                (x, y): normal
                for x, y, normal in zip(
                    self.surface_points.x,
                    self.surface_points.y,
                    self.normal_vectors.data.T,
                )
            }
            self.normals_function = FXYD(normals_lookup)
            # Assert that there are no duplicate (x,y) pairs
            if len(points_lookup) != len(self.surface_points):
                raise ValueError("All (x,y) pairs must be unique.")
        else:
            raise ValueError(
                f"Interpolation type {str(interpolation_type)} does not exist."
            )

        # Save interpolation type
        self.interpolation_type = interpolation_type

    def _check_in_bounds(self, p_samp: Pxyz) -> None:
        """Checks that points are within mirror bounds"""
        if not all(self.in_bounds(p_samp)):
            raise ValueError("Not all points are within mirror perimeter.")

    def surface_norm_at(self, p: Pxy) -> Vxyz:
        self._check_in_bounds(p)
        pts = self.normals_function(p.x, p.y)
        return Vxyz(pts.T).normalize()

    def surface_displacement_at(self, p: Pxy) -> np.ndarray:
        self._check_in_bounds(p)
        return self.surface_function(p.x, p.y)

    def survey_of_points(
        self,
        resolution: int = 1,
        resolution_type: str = "pixelX",
        random_seed: int | None = None,
    ) -> tuple[Pxyz, Vxyz]:
        # If using "given" type samping
        if self.interpolation_type == 'given':
            if resolution_type != "given":
                warn(
                    "Resolution type becomes 'given' when using type 'given' interpolation.",
                    UserWarning,
                    stacklevel=2,
                )
            given_points_xy = self.surface_points.projXY()
            points = self.location_in_space(given_points_xy)
            normals = self.surface_normal_in_space(given_points_xy)

        # If surface is interpolated, sample using MirrorAbstact method
        else:
            points, normals = super().survey_of_points(
                resolution, resolution_type, random_seed
            )

        return points, normals

    def draw(
        self,
        view: View3d,
        mirror_style: RenderControlMirror,
        transform: TransformXYZ | None = None,
    ) -> None:
        # If no interpolation
        if self.interpolation_type == 'given':
            resolution = mirror_style.resolution
            edge_values = self.region.edge_sample(resolution)
            if transform is None:
                transform = self.ori.transform_base_to_parent

            domain = self.surface_points.projXY()
            p_space = self.location_in_space(domain)

            # Draw sample points
            view.draw_single_Pxyz(p_space, style=mirror_style.point_styles)

            # Calculate z height of boundary to draw (lowest z value)
            min_val = min(self.surface_displacement_at(domain))
            edge_values_lifted = Vxyz(
                [edge_values.x, edge_values.y, [min_val] * len(edge_values)]
            )

            # Convert edge values to global coordinate system
            edge_values_lifted = transform.apply(edge_values_lifted)

            # Draw edges
            if mirror_style.point_styles is not None:
                edge_style = mirror_style.point_styles
                edge_style.markersize = 0
                view.draw_Vxyz(edge_values_lifted, style=edge_style)

            # Draw surface normals
            if mirror_style.surface_normals:
                # Get points and normals in base coordinates
                points = self.surface_points
                if isinstance(self.normal_vectors, Uxyz):
                    normals = self.normal_vectors.as_Vxyz()
                else:
                    normals = self.normal_vectors

                # Convert points and normals to reference frame
                points = transform.apply(points)
                normals.rotate_in_place(transform.R)

                # Draw points and normals
                xyzdxyz = [
                    [point.data, normal.data * mirror_style.norm_len]
                    for point, normal in zip(points, normals)
                ]
                view.draw_xyzdxyz_list(
                    xyzdxyz, close=False, style=mirror_style.norm_base_style
                )

        # If surface is interpolated, draw mirror using MirrorAbstract method
        else:
            super().draw(view, mirror_style, transform)