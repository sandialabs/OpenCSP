import inspect
import unittest

# Assume opencsp is in PYHTONPATH
import opencsp as opencsp
import example as example

from opencsp.app.camera_calibration import CameraCalibration
from opencsp.app.camera_calibration.lib.ViewAnnotatedImages import ViewAnnotatedImages
from opencsp.app.sofast.SofastGUI import SofastGUI
from opencsp.app.sofast.lib import *
from opencsp.app.select_image_points import SelectImagePoints
import opencsp.common.lib.cv.SpotAnalysis

import opencsp.app.target.target_color.target_color as target_color


class test_Docstrings(unittest.TestCase):
    camera_calibration_class_list = [
        CameraCalibration,
        opencsp.app.camera_calibration.lib.calibration_camera,
        opencsp.app.camera_calibration.lib.image_processing,
        ViewAnnotatedImages,
    ]

    scene_reconstruction_class_list = [opencsp.app.scene_reconstruction.lib.SceneReconstruction.SceneReconstruction]

    select_image_points_class_list = [SelectImagePoints]

    sofast_class_list = [
        SofastGUI,
        opencsp.app.sofast.lib.AbstractMeasurementSofast,
        opencsp.app.sofast.lib.BlobIndex,
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
        opencsp.app.sofast.lib.ProcessSofastAbstract.ProcessSofastAbstract,
        opencsp.app.sofast.lib.ProcessSofastFixed.ProcessSofastFixed,
        opencsp.app.sofast.lib.ProcessSofastFringe.ProcessSofastFringe,
        opencsp.app.sofast.lib.SofastConfiguration.SofastConfiguration,
        opencsp.app.sofast.lib.SpatialOrientation.SpatialOrientation,
        opencsp.app.sofast.lib.SystemSofastFixed.SystemSofastFixed,
        opencsp.app.sofast.lib.SystemSofastFringe.SystemSofastFringe,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationDataGeometryFacet,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationDataGeometryGeneral,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationError,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationFacetEnsemble,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationImageProcessingFacet,
        opencsp.app.sofast.lib.calculation_data_classes.CalculationImageProcessingGeneral,
        opencsp.app.sofast.lib.image_processing,
        opencsp.app.sofast.lib.load_sofast_hdf_data,
        opencsp.app.sofast.lib.process_optics_geometry,
        opencsp.app.sofast.lib.sofast_common_functions,
        opencsp.app.sofast.lib.spatial_processing,
    ]

    target_class_list = [target_color, opencsp.app.target.target_color.lib.ImageColor]

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

    app_class_list = (
        camera_calibration_class_list
        + scene_reconstruction_class_list
        + select_image_points_class_list
        + sofast_class_list
        + target_class_list
    )

    cv_class_list = [
        opencsp.common.lib.cv.CacheableImage,
        opencsp.common.lib.cv.SpotAnalysis,
        opencsp.common.lib.cv.spot_analysis.ImagesStream,
        opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream,
        opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable,
        opencsp.common.lib.cv.spot_analysis.image_processor.AbstractAggregateImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.AnnotationImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.AverageByGroupImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.BcsLocatorImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.ConvolutionImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.CroppingImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.EchoImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.ExposureDetectionImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.FalseColorImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.HotspotImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.LogScaleImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.NullImageSubtractionImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.PopulationStatisticsImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.SupportingImagesCollectorImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.View3dImageProcessor,
        opencsp.common.lib.cv.spot_analysis.image_processor.ViewCrossSectionImageProcessor,
    ]

    class_list = app_class_list + cv_class_list

    def test_docstrings_exist_for_methods(self):
        for class_module in self.class_list:
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

            if len((undocumented_methods)) != 0:
                print(f"Found undocumented methods in {class_module}:")
                for name in undocumented_methods:
                    print(f"\t{name}")
            assert len((undocumented_methods)) == 0


if __name__ == '__main__':
    unittest.main()
