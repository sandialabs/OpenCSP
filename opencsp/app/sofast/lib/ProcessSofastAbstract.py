from typing import Literal

import numpy as np

import opencsp.app.sofast.lib.AbstractMeasurementSofast as ams
import opencsp.app.sofast.lib.calculation_data_classes as cdc
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ParamsSofastFringe import ParamsSofastFringe
import opencsp.app.sofast.lib.process_optics_geometry as po
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.deflectometry.SlopeSolver import SlopeSolver
from opencsp.common.lib.deflectometry.SlopeSolverData import SlopeSolverData
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import HDF5_SaveAbstract
import opencsp.common.lib.tool.log_tools as lt


class ProcessSofastAbstract:
    def __init__(self):
        self.num_facets: int = None
        self.optic_type: Literal['undefined', 'single', 'multi'] = None
        self.data_facet_def: list[DefinitionFacet] = None
        self.data_ensemble_def: DefinitionEnsemble = None

        self.data_surfaces: list[Surface2DAbstract] = None

        self.data_geometry_general: cdc.CalculationDataGeometryGeneral = None
        self.data_image_proccessing_general: cdc.CalculationImageProcessingGeneral = None
        self.data_geometry_facet: list[cdc.CalculationDataGeometryFacet] = None
        self.data_image_processing_facet: list[cdc.CalculationImageProcessingFacet] = None
        self.data_error: cdc.CalculationError = None

        self.data_calculation_facet: list[SlopeSolverData] = None
        self.data_calculation_ensemble: list[cdc.CalculationFacetEnsemble] = None
