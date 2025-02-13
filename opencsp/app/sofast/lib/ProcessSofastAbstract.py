"""
Abstract module for processing SOFAST data.

This module provides the `ProcessSofastAbstract` class, which defines common attributes and methods
used for processing SOFAST data. It includes functionality for saving data to HDF5 files, calculating
facet pointing, and generating OpenCSP representations of the optic under test.

Notes
-----
- The `ProcessSofastAbstract` class is designed to be subclassed and extended with specific
  processing logic.

ChatGPT 4o assisted in generating some docstrings in this module.
"""

from typing import Literal

import numpy as np


import opencsp.app.sofast.lib.calculation_data_classes as cdc
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import HDF5_SaveAbstract
import opencsp.common.lib.tool.log_tools as lt


class ProcessSofastAbstract(HDF5_SaveAbstract):
    """
    Abstract class for ProcessSofast classes defining common attributes and methods.

    Attributes
    ----------
    num_facets : int
        Number of facets in the optic.
    optic_type : Literal['undefined', 'single', 'multi']
        Type of optic being processed.
    data_facet_def : list[DefinitionFacet]
        List of facet definitions.
    data_ensemble_def : DefinitionEnsemble
        Ensemble definition of the optic.
    data_surfaces : list[Surface2DAbstract]
        List of surface definitions.
    data_geometry_general : cdc.CalculationDataGeometryGeneral
        General geometry data for the calculation.
    data_image_processing_general : cdc.CalculationImageProcessingGeneral
        General image processing data for the calculation.
    data_geometry_facet : list[cdc.CalculationDataGeometryFacet]
        List of geometry data for each facet.
    data_image_processing_facet : list[cdc.CalculationImageProcessingFacet]
        List of image processing data for each facet.
    data_error : cdc.CalculationError
        Error data for the calculation.
    data_calculation_facet : list[SlopeSolverData]
        List of slope solver data for each facet.
    data_calculation_ensemble : list[cdc.CalculationFacetEnsemble]
        List of calculation data for the ensemble.
    params : HDF5_SaveAbstract
        Parameters for the calculation.

    Methods
    -------
    save_to_hdf(file: str, prefix: str = '')
        Saves data to the given HDF file.
    get_optic(interp_type: Literal['bilinear', 'clough_tocher', 'nearest'] = 'nearest') -> FacetEnsemble | Facet
        Returns the OpenCSP representation of the optic under test.
    """

    def __init__(self):
        self.num_facets: int = None
        self.optic_type: Literal["undefined", "single", "multi"] = None
        self.data_facet_def: list[DefinitionFacet] = None
        self.data_ensemble_def: DefinitionEnsemble = None

        self.data_surfaces: list[Surface2DAbstract] = None

        self.data_geometry_general: cdc.CalculationDataGeometryGeneral = None
        self.data_image_processing_general: cdc.CalculationImageProcessingGeneral = None
        self.data_geometry_facet: list[cdc.CalculationDataGeometryFacet] = None
        self.data_image_processing_facet: list[cdc.CalculationImageProcessingFacet] = None
        self.data_error: cdc.CalculationError = None

        self.data_calculation_facet: list[SlopeSolverData] = None
        self.data_calculation_ensemble: list[cdc.CalculationFacetEnsemble] = None

        self.params: HDF5_SaveAbstract = None

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given file. Data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        # Log
        lt.info(f'Saving Sofast data to: {file:s}, in HDF5 folder: "{prefix:s}"')

        # One per measurement
        if self.data_error is not None:
            self.data_error.save_to_hdf(file, f"{prefix:s}DataSofastCalculation/general/")
        self.data_geometry_general.save_to_hdf(file, f"{prefix:s}DataSofastCalculation/general/")
        self.data_image_processing_general.save_to_hdf(file, f"{prefix:s}DataSofastCalculation/general/")

        # Sofast parameters
        self.params.save_to_hdf(file, f"{prefix:s}DataSofastInput/")

        # Facet definition
        if self.data_facet_def is not None:
            for idx_facet, facet_data in enumerate(self.data_facet_def):
                facet_data.save_to_hdf(file, f"{prefix:s}DataSofastInput/optic_definition/facet_{idx_facet:03d}/")

        # Ensemble definition
        if self.data_ensemble_def is not None:
            self.data_ensemble_def.save_to_hdf(file, f"{prefix:s}DataSofastInput/optic_definition/")

        # Surface definition
        for idx_facet, surface in enumerate(self.data_surfaces):
            surface.save_to_hdf(file, f"{prefix:s}DataSofastInput/optic_definition/facet_{idx_facet:03d}/")

        # Calculations, one per facet
        for idx_facet in range(self.num_facets):
            # Save facet slope data
            if self.data_calculation_facet is not None:
                self.data_calculation_facet[idx_facet].save_to_hdf(
                    file, f"{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/"
                )
            # Save facet geometry data
            if self.data_geometry_facet is not None:
                self.data_geometry_facet[idx_facet].save_to_hdf(
                    file, f"{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/"
                )
            # Save facet image processing data
            if self.data_image_processing_facet is not None:
                self.data_image_processing_facet[idx_facet].save_to_hdf(
                    file, f"{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/"
                )
            # Save ensemle data
            if self.data_calculation_ensemble is not None:
                self.data_calculation_ensemble[idx_facet].save_to_hdf(
                    file, f"{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/"
                )

    def _calculate_facet_pointing(self, reference: Literal["average"] | int = "average") -> None:
        """
        Calculates facet pointing relative to the given reference.

        Parameters
        ----------
        reference : 'average' | int
            If 'average', the pointing reference is the average of all
            facet pointing directions. If, int, that facet index is assumed
            to have perfect pointing.
        """
        if self.data_calculation_facet is None:
            lt.error_and_raise(ValueError, 'Slopes must be solved first by running "solve_slopes".')
        if reference != "average" and not isinstance(reference, int):
            lt.error_and_raise(ValueError, 'Given reference must be int or "average".')
        if isinstance(reference, int) and reference >= self.num_facets:
            lt.error_and_raise(
                ValueError, f"Given facet index, {reference:d}, is out of range of 0-{self.num_facets - 1:d}."
            )

        # Instantiate data list
        self.data_calculation_ensemble = []

        trans_facet_ensemble_list = []
        v_pointing_matrix = np.zeros((3, self.num_facets))
        for idx in range(self.num_facets):
            # Get transformation from user-input and slope solving
            trans_1 = TransformXYZ.from_R_V(
                self.data_ensemble_def.r_facet_ensemble[idx], self.data_ensemble_def.v_facet_locations[idx]
            )
            trans_2 = self.data_calculation_facet[idx].trans_alignment
            # Calculate inverse of slope solving transform
            trans_2 = TransformXYZ.from_V(-trans_2.V) * TransformXYZ.from_R(trans_2.R.inv())
            # Create local to global transformation
            trans_facet_ensemble_list.append(trans_2 * trans_1)

            # Calculate pointing vector in ensemble coordinates
            v_pointing = Vxyz((0, 0, 1)).rotate(trans_facet_ensemble_list[idx].R)
            v_pointing_matrix[:, idx] = v_pointing.data.squeeze()

        # Calculate reference pointing direction
        if isinstance(reference, int):
            v_pointing_ref = Vxyz(v_pointing_matrix[:, reference])
        elif reference == "average":
            v_pointing_ref = Vxyz(v_pointing_matrix.mean(1))
        # Calculate rotation to align pointing vectors
        r_align_pointing = v_pointing_ref.align_to(Vxyz((0, 0, 1)))
        trans_align_pointing = TransformXYZ.from_R(r_align_pointing)

        # Apply alignment rotation to total transformation
        trans_facet_ensemble_list = [trans_align_pointing * t for t in trans_facet_ensemble_list]

        # Calculate global slope and surface points
        for idx in range(self.num_facets):
            # Get slope data
            slopes = self.data_calculation_facet[idx].slopes_facet_xy  # facet coordinats

            # Calculate surface normals in local (facet) coordinates
            number_data_points = slopes.shape[1]
            u_surf_norms = np.ones((3, number_data_points))
            u_surf_norms[:2] = -slopes
            u_surf_norms = Uxyz(u_surf_norms).as_Vxyz()

            # Apply rotation to normal vectors
            u_surf_norms_global = u_surf_norms.rotate(trans_facet_ensemble_list[idx].R)
            # Convert normal vectors to global (ensemble) slopes
            slopes_ensemble_xy = -u_surf_norms_global.projXY().data / u_surf_norms_global.z

            # Convert surface points to global (ensemble) coordinates
            v_surf_points_ensemble = trans_facet_ensemble_list[idx].apply(
                self.data_calculation_facet[idx].v_surf_points_facet
            )

            # Calculate pointing vectors in ensemble coordinates
            v_facet_pointing_ensemble = Vxyz((0, 0, 1)).rotate(trans_facet_ensemble_list[idx].R)

            data = cdc.CalculationFacetEnsemble(
                trans_facet_ensemble_list[idx], slopes_ensemble_xy, v_surf_points_ensemble, v_facet_pointing_ensemble
            )
            self.data_calculation_ensemble.append(data)

    def get_optic(
        self, interp_type: Literal["bilinear", "clough_tocher", "nearest"] = "nearest"
    ) -> FacetEnsemble | Facet:
        """Returns the OpenCSP representation of the optic under test. Returns either
        a Facet or FacetEnsemble object depending on the optic type. Each mirror is
        represented by a MirrorPoint object. Each mirror origin is co-located with its
        parent facet origin.

        Parameters
        ----------
        interp_type : {'bilinear', 'clough_tocher', 'nearest'}, optional
            Mirror interpolation type, by default 'nearest'

        Returns
        -------
        FacetEnsemble | Facet
            Optic object
        """
        if self.data_calculation_facet is None:
            lt.error_and_raise(ValueError, "Sofast data must be processed before optic is available.")

        facets = []
        trans_list = []
        for idx_mirror in range(self.num_facets):
            # Get surface points
            pts: Vxyz = self.data_calculation_facet[idx_mirror].v_surf_points_facet
            # Get normals from slopes
            slopes: np.ndarray = self.data_calculation_facet[idx_mirror].slopes_facet_xy
            number_data_points = slopes.shape[1]
            norm_data = np.ones((3, number_data_points))
            norm_data[:2] = -slopes
            norm_vecs = Uxyz(norm_data)
            # Get mirror shape
            if self.optic_type == "undefined":
                # Find bounding box
                x1 = pts.x.min()
                x2 = pts.x.max()
                y1 = pts.y.min()
                y2 = pts.y.max()
                vertices = Vxy(([x1, x1, x2, x2], [y1, y2, y2, y1]))
            else:
                # Get optic region from optic definition
                vertices = self.data_facet_def[idx_mirror].v_facet_corners.projXY()
            shape = RegionXY.from_vertices(vertices)
            # Create mirror
            mirror = MirrorPoint(pts, norm_vecs, shape, interp_type)
            # Create facet
            facet = Facet(mirror)
            # Locate facet within ensemble
            if self.optic_type == "multi":
                trans: TransformXYZ = self.data_calculation_ensemble[idx_mirror].trans_facet_ensemble
                trans_list.append(trans)
            # Save facets
            facets.append(facet)

        # Return optics
        if self.optic_type == "multi":
            ensemble = FacetEnsemble(facets)
            ensemble.set_facet_transform_list(trans_list)
            return ensemble
        else:
            return facets[0]
