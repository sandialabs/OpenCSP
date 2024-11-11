import inspect

# Assume opencsp is in PYHTONPATH
import opencsp as opencsp
import example as example

# TODO: why aren't these imported from import opencsp as opencsp above
from opencsp.app.camera_calibration import CameraCalibration
from opencsp.app.camera_calibration.lib.ViewAnnotatedImages import ViewAnnotatedImages
from opencsp.app.sofast.SofastGUI import SofastGUI
from opencsp.app.sofast.lib import *
from opencsp.app.select_image_points import SelectImagePoints
import opencsp.app.target.target_color.target_color as target_color

import opencsp.common.lib.camera.CameraTransform as CameraTransform
import opencsp.common.lib.camera.ImageAcquisition_DCAM_color as ImageAcquisition_DCAM_color
import opencsp.common.lib.camera.ImageAcquisition_MSMF as ImageAcquisition_MSMF
import opencsp.common.lib.camera.UCamera as UCamera
import opencsp.common.lib.cv.SpotAnalysis as SpotAnalysis
import opencsp.common.lib.deflectometry.ImageProjectionSetupGUI as ImageProjectionSetupGUI
import opencsp.common.lib.deflectometry.ParamsSlopeSolver as ParamsSlopeSolver
import opencsp.common.lib.deflectometry.ParamsSlopeSolverAbstract as ParamsSlopeSolverAbstract
import opencsp.common.lib.deflectometry.ParamsSlopeSolverParaboloid as ParamsSlopeSolverParaboloid
import opencsp.common.lib.deflectometry.ParamsSlopeSolverPlano as ParamsSlopeSolverPlano


def test_docstrings_exist_for_methods():
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

    camera_class_list = [
        opencsp.common.lib.camera.Camera.Camera,
        CameraTransform,
        opencsp.common.lib.camera.ImageAcquisitionAbstract.ImageAcquisitionAbstract,
        opencsp.common.lib.camera.ImageAcquisition_DCAM_color.ImageAcquisition,
        opencsp.common.lib.camera.ImageAcquisition_DCAM_mono.ImageAcquisition,
        opencsp.common.lib.camera.ImageAcquisition_MSMF.ImageAcquisition,
        opencsp.common.lib.camera.LiveView.LiveView,
        opencsp.common.lib.camera.UCamera.Camera,
        opencsp.common.lib.camera.UCamera.RealCamera,
        opencsp.common.lib.camera.image_processing,
    ]

    csp_class_list = [
        opencsp.common.lib.csp.Facet,
        opencsp.common.lib.csp.FacetEnsemble,
        opencsp.common.lib.csp.HeliostatAbstract,
        opencsp.common.lib.csp.HeliostatAzEl,
        opencsp.common.lib.csp.HeliostatConfiguration,
        opencsp.common.lib.csp.LightPath,
        opencsp.common.lib.csp.LightPathEnsemble,
        opencsp.common.lib.csp.LightSource,
        opencsp.common.lib.csp.LightSourcePoint,
        opencsp.common.lib.csp.LightSourceSun,
        opencsp.common.lib.csp.MirrorAbstract,
        opencsp.common.lib.csp.MirrorParametric,
        opencsp.common.lib.csp.MirrorParametricRectangular,
        opencsp.common.lib.csp.MirrorPoint,
        opencsp.common.lib.csp.OpticOrientationAbstract,
        opencsp.common.lib.csp.RayTrace,
        opencsp.common.lib.csp.RayTraceable,
        opencsp.common.lib.csp.Scene,
        opencsp.common.lib.csp.SolarField,
        opencsp.common.lib.csp.StandardPlotOutput,
        opencsp.common.lib.csp.Tower,
        opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract,
        opencsp.common.lib.csp.sun_position,
        opencsp.common.lib.csp.sun_track,
        opencsp.common.lib.csp.visualize_orthorectified_image,
    ]

    cv_class_list = [
        opencsp.common.lib.cv.CacheableImage,
        opencsp.common.lib.cv.OpticalFlow,
        opencsp.common.lib.cv.SpotAnalysis,
        opencsp.common.lib.cv.image_filters,
        opencsp.common.lib.cv.image_reshapers,
    ]

    deflectometry_class_list = [
        opencsp.common.lib.deflectometry.CalibrationCameraPosition,
        opencsp.common.lib.deflectometry.ImageProjection,
        opencsp.common.lib.deflectometry.ImageProjectionSetupGUI,
        opencsp.common.lib.deflectometry.ParamsSlopeSolver,
        opencsp.common.lib.deflectometry.ParamsSlopeSolverAbstract,
        opencsp.common.lib.deflectometry.ParamsSlopeSolverParaboloid,
        opencsp.common.lib.deflectometry.ParamsSlopeSolverPlano,
        opencsp.common.lib.deflectometry.SlopeSolver,
        opencsp.common.lib.deflectometry.SlopeSolverData,
        opencsp.common.lib.deflectometry.SlopeSolverDataDebug,
        opencsp.common.lib.deflectometry.Surface2DAbstract,
        opencsp.common.lib.deflectometry.Surface2DParabolic,
        opencsp.common.lib.deflectometry.Surface2DPlano,
        opencsp.common.lib.deflectometry.slope_fitting_2d,
    ]

    file_class_list = [
        opencsp.common.lib.file.AbstractAttributeParser,
        opencsp.common.lib.file.AttributesManager,
        opencsp.common.lib.file.CsvColumns,
        opencsp.common.lib.file.CsvInterface,
        opencsp.common.lib.file.SimpleCsv,
    ]

    common_class_list = camera_class_list + csp_class_list + cv_class_list + deflectometry_class_list + file_class_list

    class_list = app_class_list + common_class_list

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
            method_list = [
                func
                for func in dir(class_module)
                if callable(getattr(class_module, func))
                and not func.startswith("__")
                and not func.startswith("_")
                and not func.endswith("_UNVERIFIED")
            ]

        for method in method_list:
            doc_exists = True
            if getattr(class_module, method).__doc__ is None:
                doc_exists = False

            print(f"doc_exists({class_module.__name__}.{method}): " f"{doc_exists}")
            assert doc_exists
