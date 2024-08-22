from typing import Literal

import opencsp.app.sofast.lib.calculation_data_classes as cdc
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.tool.hdf5_tools import HDF5_SaveAbstract
import opencsp.common.lib.tool.log_tools as lt


class ProcessSofastAbstract(HDF5_SaveAbstract):
    def __init__(self):
        self.num_facets: int = None
        self.optic_type: Literal['undefined', 'single', 'multi'] = None
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

        self.params = None

    def save_to_hdf(self, file: str, prefix: str = ''):
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
            self.data_error.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')
        self.data_geometry_general.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')
        self.data_image_processing_general.save_to_hdf(file, f'{prefix:s}DataSofastCalculation/general/')

        # Sofast parameters
        self.params.save_to_hdf(file, f'{prefix:s}DataSofastInput/')

        # Facet definition
        if self.data_facet_def is not None:
            for idx_facet, facet_data in enumerate(self.data_facet_def):
                facet_data.save_to_hdf(file, f'{prefix:s}DataSofastInput/optic_definition/facet_{idx_facet:03d}/')

        # Ensemble definition
        if self.data_ensemble_def is not None:
            self.data_ensemble_def.save_to_hdf(file, f'{prefix:s}DataSofastInput/optic_definition/')

        # Surface definition
        for idx_facet, surface in enumerate(self.data_surfaces):
            surface.save_to_hdf(file, f'{prefix:s}DataSofastInput/optic_definition/facet_{idx_facet:03d}/')

        # Calculations, one per facet
        for idx_facet in range(self.num_facets):
            # Save facet slope data
            if self.data_calculation_facet is not None:
                self.data_calculation_facet[idx_facet].save_to_hdf(
                    file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
                )
            # Save facet geometry data
            if self.data_geometry_facet is not None:
                self.data_geometry_facet[idx_facet].save_to_hdf(
                    file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
                )
            # Save facet image processing data
            if self.data_image_processing_facet is not None:
                self.data_image_processing_facet[idx_facet].save_to_hdf(
                    file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
                )
            # Save ensemle data
            if self.data_calculation_ensemble is not None:
                self.data_calculation_ensemble[idx_facet].save_to_hdf(
                    file, f'{prefix:s}DataSofastCalculation/facet/facet_{idx_facet:03d}/'
                )
