import inspect
import unittest

import opencsp as opencsp
import example as example

# Assume opencsp is in PYHTONPATH
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.common.lib.cv import CacheableImage, SpotAnalysis
from opencsp.common.lib.cv.spot_analysis import ImagesStream, SpotAnalysisImagesStream, SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *
from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractSpotAnalysisImageProcessor

# Assume opencsp is in PYHTONPATH
import opencsp as opencsp
import example as example

# TODO: why aren't these imported from import opencsp as opencsp above
from opencsp.app.camera_calibration.lib.ViewAnnotatedImages import ViewAnnotatedImages
from opencsp.app.sofast.SofastGUI import SofastGUI
from opencsp.app.sofast.lib import *

<<<<<<< HEAD
=======
# from opencsp.app.target.target_color.target_color_2d_gradient import target_color_2d_gradient
>>>>>>> a785026 (Move target color files to contrib)
import opencsp.app.target.target_color.target_color as target_color


class test_Docstrings(unittest.TestCase):
    sofast_class_list = [
            opencsp.app.sofast.lib.AbstractMeasurementSofast,
            opencsp.app.sofast.lib.BlobIndex,
            opencsp.app.sofast.lib.calculation_data_classes.CalculationDataGeometryFacet,
            opencsp.app.sofast.lib.calculation_data_classes.CalculationDataGeometryGeneral,
            opencsp.app.sofast.lib.calculation_data_classes.CalculationError,
            opencsp.app.sofast.lib.calculation_data_classes.CalculationFacetEnsemble,
            opencsp.app.sofast.lib.calculation_data_classes.CalculationImageProcessingFacet,
            opencsp.app.sofast.lib.calculation_data_classes.CalculationImageProcessingGeneral,
            opencsp.app.sofast.lib.CalibrateDisplayShape.CalibrateDisplayShape,
            opencsp.app.sofast.lib.CalibrateSofastFixedDots.CalibrateSofastFixedDots,
            opencsp.app.sofast.lib.CalibrateDisplayShape.DataCalculation,
            opencsp.app.sofast.lib.CalibrateDisplayShape.DataInput,
            opencsp.app.sofast.lib.DebugOpticsGeometry.DebugOpticsGeometry,
            opencsp.app.sofast.lib.DefinitionEnsemble.DefinitionEnsemble,
            opencsp.app.sofast.lib.DefinitionFacet.DefinitionFacet,
            opencsp.app.sofast.lib.DisplayShape.DisplayShape,
            opencsp.app.sofast.lib.DistanceOpticScreen.DistanceOpticScreen,
            opencsp.app.sofast.lib.DotLocationsFixedPattern.DotLocationsFixedPattern,
            opencsp.app.sofast.lib.Fringes.Fringes,
            opencsp.app.sofast.lib.ImageCalibrationAbstract.ImageCalibrationAbstract,
            opencsp.app.sofast.lib.ImageCalibrationGlobal.ImageCalibrationGlobal,
            opencsp.app.sofast.lib.ImageCalibrationScaling.ImageCalibrationScaling,
            opencsp.app.sofast.lib.MeasurementSofastFixed.MeasurementSofastFixed,
            opencsp.app.sofast.lib.MeasurementSofastFringe.MeasurementSofastFringe,
            opencsp.app.sofast.lib.ParamsMaskCalculation.ParamsMaskCalculation,
            opencsp.app.sofast.lib.ParamsOpticGeometry.ParamsOpticGeometry,
            opencsp.app.sofast.lib.ParamsSofastFixed.ParamsSofastFixed,
            opencsp.app.sofast.lib.ParamsSofastFringe.ParamsSofastFringe,
            opencsp.app.sofast.lib.PatternSofastFixed.PatternSofastFixed,
            opencsp.app.sofast.lib.ProcessSofastFixed.ProcessSofastFixed,
            opencsp.app.sofast.lib.ProcessSofastFringe.ProcessSofastFringe,
            opencsp.app.sofast.lib.SofastConfiguration.SofastConfiguration,
            opencsp.app.sofast.lib.SpatialOrientation.SpatialOrientation,
            opencsp.app.sofast.lib.SystemSofastFixed.SystemSofastFixed,
            opencsp.app.sofast.lib.SystemSofastFringe.SystemSofastFringe,
            SofastGUI,
        ]
        opencsp.app.sofast.lib.AbstractMeasurementSofast,
        opencsp.app.sofast.lib.BlobIndex,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationDataGeometryFacet,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationDataGeometryGeneral,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationError,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationFacetEnsemble,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationImageProcessingFacet,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationImageProcessingGeneral,
        opencsp.app.sofast.lib.CalibrateDisplayShape.CalibrateDisplayShape,
        opencsp.app.sofast.lib.CalibrateSofastFixedDots.CalibrateSofastFixedDots,
        opencsp.app.sofast.lib.CalibrateDisplayShape.DataCalculation,
        opencsp.app.sofast.lib.CalibrateDisplayShape.DataInput,
        opencsp.app.sofast.lib.DebugOpticsGeometry.DebugOpticsGeometry,
        opencsp.app.sofast.lib.DefinitionEnsemble.DefinitionEnsemble,
        opencsp.app.sofast.lib.DefinitionFacet.DefinitionFacet,
        opencsp.app.sofast.lib.DisplayShape.DisplayShape,
        opencsp.app.sofast.lib.DistanceOpticScreen.DistanceOpticScreen,
        opencsp.app.sofast.lib.DotLocationsFixedPattern.DotLocationsFixedPattern,
        SofastGUI,
        opencsp.app.sofast.lib.Fringes.Fringes,
        opencsp.app.sofast.lib.ImageCalibrationAbstract.ImageCalibrationAbstract,
        opencsp.app.sofast.lib.ImageCalibrationGlobal.ImageCalibrationGlobal,
        opencsp.app.sofast.lib.ImageCalibrationScaling.ImageCalibrationScaling,
        opencsp.app.sofast.lib.MeasurementSofastFixed.MeasurementSofastFixed,
        opencsp.app.sofast.lib.MeasurementSofastFringe.MeasurementSofastFringe,
        opencsp.app.sofast.lib.ParamsMaskCalculation.ParamsMaskCalculation,
        opencsp.app.sofast.lib.ParamsOpticGeometry.ParamsOpticGeometry,
        opencsp.app.sofast.lib.ParamsSofastFixed.ParamsSofastFixed,
        opencsp.app.sofast.lib.ParamsSofastFringe.ParamsSofastFringe,
        opencsp.app.sofast.lib.PatternSofastFixed.PatternSofastFixed,
        opencsp.app.sofast.lib.ProcessSofastFixed.ProcessSofastFixed,
        opencsp.app.sofast.lib.ProcessSofastFringe.ProcessSofastFringe,
        opencsp.app.sofast.lib.SofastConfiguration.SofastConfiguration,
        opencsp.app.sofast.lib.SpatialOrientation.SpatialOrientation,
        opencsp.app.sofast.lib.SystemSofastFixed.SystemSofastFixed,
        opencsp.app.sofast.lib.SystemSofastFringe.SystemSofastFringe,
        SofastGUI,
    ]

    target_class_list = [opencsp.app.target.target_color.lib.ImageColor, target_color]

    camera_calibration_class_list = [
        opencsp.app.camera_calibration.lib.calibration_camera,
        opencsp.app.camera_calibration.lib.image_processing,
        ViewAnnotatedImages,
    ]
    scene_reconstruction_class_list = [opencsp.app.scene_reconstruction.lib.SceneReconstruction.SceneReconstruction]
    # TODO: example_camera_calibration_list
    # TODO: example_csp_list
    # TODO: example_scene_reconstruction_list
    # TODO: example_sofast_fixed_list
    # TODO: example_solarfield_list
    # TODO: example_camera_io_list
    # TODO: example_mirror_list
    # TODO: example_raytrace_list
    # TODO: example_sofast_calibration_list
    # TODO: example_sofast_fringe_list
    # TODO: example_targetcolor_list

    class_list = sofast_class_list + target_class_list + camera_calibration_class_list + scene_reconstruction_class_list
    #class_list = [
    #    Sofast,
    #    # Spot Analysis
    #    SpotAnalysis,
    #    SpotAnalysisOperable,
    #    ImagesStream,
    #    SpotAnalysisImagesStream,
    #    CacheableImage,
    #    AbstractAggregateImageProcessor,
    #    AbstractSpotAnalysisImageProcessor,
    #    AbstractVisualizationImageProcessor,
    #    AnnotationImageProcessor,
    #    AverageByGroupImageProcessor,
    #    BcsLocatorImageProcessor,
    #    ConvolutionImageProcessor,
    #    CroppingImageProcessor,
    #    EchoImageProcessor,
    #    ExposureDetectionImageProcessor,
    #    FalseColorImageProcessor,
    #    HotspotImageProcessor,
    #    LogScaleImageProcessor,
    #    NullImageSubtractionImageProcessor,
    #    PopulationStatisticsImageProcessor,
    #    SupportingImagesCollectorImageProcessor,
    #    View3dImageProcessor,
    #    ViewCrossSectionImageProcessor,
    # TODO: List all user-facing classes here.

<<<<<<< HEAD
    def test_docstrings_exist_for_methods(self):
        for class_module in self.class_list:
=======
    for class_module in class_list:
        print(class_module)
        method_list = []
        if inspect.isclass(class_module):
            method_list = [
                func
                for func in class_module.__dict__
                if callable(getattr(class_module, func))
                and not func.startswith("__")
                and not func.startswith("_")
                and not hasattr(super(class_module), func)
            ]
        else:
>>>>>>> 767739b (Check opencsp.app for doc strings)
            method_list = [
                func
                for func in dir(class_module)
                if callable(getattr(class_module, func)) and not func.startswith("__") and not func.startswith("_")
            ]

            undocumented_methods: list[str] = []

            for method in method_list:
                doc_exists = True
                if inspect.getdoc(getattr(class_module, method)) is None:
                    doc_exists = False

                method_name = f"{class_module.__name__}.{method}"
                print(f"doc_exists({method_name}): " f"{doc_exists}")
                if not doc_exists:
                    undocumented_methods.append(method)

            self.assertEqual(
                len(undocumented_methods),
                0,
                f"Found undocumented methods in {class_module}:\n\t" + "\n\t".join(undocumented_methods),
            )


if __name__ == '__main__':
    unittest.main()
