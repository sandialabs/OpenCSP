import inspect
import unittest

# Assume opencsp is in PYTHONPATH
import opencsp as opencsp
import example.sofast_fringe.example_process_in_debug_mode
import example.sofast_calibration.example_calibration_screen_shape

# Examples
import example.sofast_fringe.example_make_rectangular_screen_definition
import example.sofast_fringe.sofast_temperature_analysis
import example.sofast_fringe.sofast_command_line_tool


# TODO: why aren't these imported from import opencsp as opencsp above
from opencsp.app.camera_calibration import CameraCalibration
from opencsp.app.camera_calibration.lib.ViewAnnotatedImages import ViewAnnotatedImages
from opencsp.app.sofast.SofastGUI import SofastGUI
from opencsp.app.sofast.lib import *
from opencsp.app.select_image_points import SelectImagePoints
import opencsp.common.lib.cv.SpotAnalysis


import opencsp.common.lib.camera.ImageAcquisition_DCAM_color
import opencsp.common.lib.camera.ImageAcquisition_MSMF
import opencsp.common.lib.camera.UCamera
import opencsp.common.lib.cv.SpotAnalysis
import opencsp.common.lib.deflectometry.ImageProjectionSetupGUI
import opencsp.common.lib.deflectometry.ParamsSlopeSolver
import opencsp.common.lib.deflectometry.ParamsSlopeSolverAbstract
import opencsp.common.lib.deflectometry.ParamsSlopeSolverParaboloid
import opencsp.common.lib.deflectometry.ParamsSlopeSolverPlano
import opencsp.common.lib.opencsp_path.optical_analysis_data_path
import opencsp.common.lib.process.ServerSynchronizer
import opencsp.common.lib.process.parallel_video_tools
import opencsp.common.lib.render.PlotAnnotation
import opencsp.common.lib.render.PowerpointSlide
import opencsp.common.lib.render.general_plot
import opencsp.common.lib.render.image_plot
import opencsp.common.lib.render.pandas_plot
import opencsp.common.lib.file.CsvColumns
import opencsp.common.lib.file.SimpleCsv
import opencsp.common.lib.render_control.RenderControlDeflectometryInstrument
import opencsp.common.lib.render_control.RenderControlEvaluateHeliostats3d
import opencsp.common.lib.render_control.RenderControlFramesNoDuplicates
import opencsp.common.lib.render_control.RenderControlHeliostatTracks
import opencsp.common.lib.render_control.RenderControlHeliostats3d
import opencsp.common.lib.render_control.RenderControlIntersection
import opencsp.common.lib.render_control.RenderControlKeyCorners
import opencsp.common.lib.render_control.RenderControlKeyFramesGivenManual
import opencsp.common.lib.render_control.RenderControlKeyTracks
import opencsp.common.lib.render_control.RenderControlPowerpointPresentation
import opencsp.common.lib.render_control.RenderControlTrajectoryAnalysis
import opencsp.common.lib.render_control.RenderControlVideoTracks
import opencsp.common.lib.tool.dict_tools
import opencsp.common.lib.uas.Scan
import opencsp.common.lib.uas.ScanPass
import opencsp.common.lib.uas.WayPoint


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

    target_class_list = [opencsp.app.target.target_color.lib.ImageColor]

    camera_calibration_class_list = [
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
        opencsp.app.sofast.lib.SofastInterface,
    ]

    target_class_list = [opencsp.app.target.target_color.lib.ImageColor]

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

    example_list = [
        example.sofast_fringe.sofast_temperature_analysis,
        example.sofast_fringe.sofast_command_line_tool,
        example.sofast_fringe.example_process_in_debug_mode,
        example.sofast_fringe.sofast_temperature_analysis,
        example.sofast_fringe.example_make_rectangular_screen_definition,
        example.sofast_calibration.example_calibration_screen_shape,
    ]

    app_class_list = (
        camera_calibration_class_list
        + scene_reconstruction_class_list
        + select_image_points_class_list
        + sofast_class_list
        + target_class_list
    )

    cv_class_list = [
        opencsp.common.lib.cv.CacheableImage,
        opencsp.common.lib.cv.OpticalFlow,
        opencsp.common.lib.cv.SpotAnalysis,
        opencsp.common.lib.cv.image_filters,
        opencsp.common.lib.cv.image_reshapers,
        opencsp.common.lib.cv.annotations.AbstractAnnotations,
        opencsp.common.lib.cv.annotations.HotspotAnnotation,
        opencsp.common.lib.cv.annotations.PointAnnotations,
        opencsp.common.lib.cv.fiducials.AbstractFiducials,
        opencsp.common.lib.cv.fiducials.BcsFiducial,
        opencsp.common.lib.cv.fiducials.PointFiducials,
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
    camera_class_list = [
        opencsp.common.lib.camera.Camera.Camera,
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

    geo_class_list = [opencsp.common.lib.geo.lon_lat_nsttf]

    geometry_class_list = [
        opencsp.common.lib.geometry.EdgeXY,
        opencsp.common.lib.geometry.FunctionXYAbstract,
        opencsp.common.lib.geometry.FunctionXYContinuous,
        opencsp.common.lib.geometry.FunctionXYDiscrete,
        opencsp.common.lib.geometry.FunctionXYGrid,
        opencsp.common.lib.geometry.Intersection,
        opencsp.common.lib.geometry.LineXY,
        opencsp.common.lib.geometry.LoopXY,
        opencsp.common.lib.geometry.Pxy,
        opencsp.common.lib.geometry.Pxyz,
        opencsp.common.lib.geometry.RegionXY,
        opencsp.common.lib.geometry.TransformXYZ,
        opencsp.common.lib.geometry.Uxy,
        opencsp.common.lib.geometry.Uxyz,
        opencsp.common.lib.geometry.Vxy,
        opencsp.common.lib.geometry.Vxyz,
        opencsp.common.lib.geometry.angle,
    ]

    opencsp_path_class_list = [
        opencsp.common.lib.opencsp_path.data_path_for_test,
        opencsp.common.lib.opencsp_path.opencsp_root_path,
        opencsp.common.lib.opencsp_path.optical_analysis_data_path,
    ]

    photogrammetry_class_list = [
        opencsp.common.lib.photogrammetry.ImageMarker,
        opencsp.common.lib.photogrammetry.bundle_adjustment,
        opencsp.common.lib.photogrammetry.photogrammetry,
    ]

    process_class_list = [
        opencsp.common.lib.process.MultiprocessNonDaemonic,
        opencsp.common.lib.process.ParallelPartitioner,
        opencsp.common.lib.process.ServerSynchronizer,
        opencsp.common.lib.process.parallel_file_tools,
        opencsp.common.lib.process.parallel_video_tools,
        opencsp.common.lib.process.subprocess_tools,
        opencsp.common.lib.process.lib.CalledProcessError,
        opencsp.common.lib.process.lib.ProcessOutputLine,
        opencsp.common.lib.process.lib.ServerSynchronizerError,
    ]

    render_class_list = [
        opencsp.common.lib.render.Color,
        opencsp.common.lib.render.ImageAttributeParser,
        opencsp.common.lib.render.PlotAnnotation,
        opencsp.common.lib.render.PowerpointSlide,
        opencsp.common.lib.render.VideoHandler,
        opencsp.common.lib.render.View3d,
        opencsp.common.lib.render.axis_3d,
        opencsp.common.lib.render.figure_management,
        opencsp.common.lib.render.general_plot,
        opencsp.common.lib.render.image_plot,
        opencsp.common.lib.render.pandas_plot,
        opencsp.common.lib.render.view_spec,
        opencsp.common.lib.render.lib.AbstractPlotHandler,
        opencsp.common.lib.render.lib.PowerpointImage,
        opencsp.common.lib.render.lib.PowerpointShape,
        opencsp.common.lib.render.lib.PowerpointText,
    ]

    render_control_class_list = [
        opencsp.common.lib.render_control.RenderControlAxis,
        opencsp.common.lib.render_control.RenderControlBcs,
        opencsp.common.lib.render_control.RenderControlDeflectometryInstrument,
        opencsp.common.lib.render_control.RenderControlEnsemble,
        opencsp.common.lib.render_control.RenderControlEvaluateHeliostats3d,
        opencsp.common.lib.render_control.RenderControlFacet,
        opencsp.common.lib.render_control.RenderControlFacetEnsemble,
        opencsp.common.lib.render_control.RenderControlFigure,
        opencsp.common.lib.render_control.RenderControlFigureRecord,
        opencsp.common.lib.render_control.RenderControlFramesNoDuplicates,
        opencsp.common.lib.render_control.RenderControlFunctionXY,
        opencsp.common.lib.render_control.RenderControlHeatmap,
        opencsp.common.lib.render_control.RenderControlHeliostat,
        opencsp.common.lib.render_control.RenderControlHeliostatTracks,
        opencsp.common.lib.render_control.RenderControlHeliostats3d,
        opencsp.common.lib.render_control.RenderControlIntersection,
        opencsp.common.lib.render_control.RenderControlKeyCorners,
        opencsp.common.lib.render_control.RenderControlKeyFramesGivenManual,
        opencsp.common.lib.render_control.RenderControlKeyTracks,
        opencsp.common.lib.render_control.RenderControlLightPath,
        opencsp.common.lib.render_control.RenderControlMirror,
        opencsp.common.lib.render_control.RenderControlPointSeq,
        opencsp.common.lib.render_control.RenderControlPowerpointPresentation,
        opencsp.common.lib.render_control.RenderControlPowerpointSlide,
        opencsp.common.lib.render_control.RenderControlRayTrace,
        opencsp.common.lib.render_control.RenderControlSolarField,
        opencsp.common.lib.render_control.RenderControlSurface,
        opencsp.common.lib.render_control.RenderControlText,
        opencsp.common.lib.render_control.RenderControlTower,
        opencsp.common.lib.render_control.RenderControlTrajectoryAnalysis,
        opencsp.common.lib.render_control.RenderControlVideo,
        opencsp.common.lib.render_control.RenderControlVideoFrames,
        opencsp.common.lib.render_control.RenderControlVideoTracks,
    ]

    common_target_class_list = [
        opencsp.common.lib.target.TargetAbstract,
        opencsp.common.lib.target.TargetColor,
        opencsp.common.lib.target.target_color_1d_gradient,
        opencsp.common.lib.target.target_color_2d_rgb,
        opencsp.common.lib.target.target_color_convert,
        opencsp.common.lib.target.target_image,
    ]

    tool_class_list = [
        opencsp.common.lib.tool.dict_tools,
        opencsp.common.lib.tool.exception_tools,
        opencsp.common.lib.tool.file_tools,
        opencsp.common.lib.tool.hdf5_tools,
        opencsp.common.lib.tool.image_tools,
        opencsp.common.lib.tool.list_tools,
        opencsp.common.lib.tool.log_tools,
        opencsp.common.lib.tool.math_tools,
        opencsp.common.lib.tool.string_tools,
        opencsp.common.lib.tool.system_tools,
        opencsp.common.lib.tool.time_date_tools,
        opencsp.common.lib.tool.tk_tools,
        opencsp.common.lib.tool.typing_tools,
        opencsp.common.lib.tool.unit_conversion,
    ]

    uas_class_list = [opencsp.common.lib.uas.Scan, opencsp.common.lib.uas.ScanPass, opencsp.common.lib.uas.WayPoint]

    common_class_list = (
        cv_class_list
        + camera_class_list
        + csp_class_list
        + cv_class_list
        + deflectometry_class_list
        + file_class_list
        + geo_class_list
        + geometry_class_list
        + opencsp_path_class_list
        + photogrammetry_class_list
        + process_class_list
        + render_class_list
        + render_control_class_list
        + common_target_class_list
        + tool_class_list
        + uas_class_list
    )

    class_list = app_class_list + common_class_list

    def test_docstrings_exist_for_methods(self):
        n_docstrings = 0
        for class_module in self.class_list:
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
                    and not func.endswith("_NOTWORKING")
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
                else:
                    n_docstrings += 1

            if len((undocumented_methods)) != 0:
                print(f"Found undocumented methods in {class_module}:")
                for name in undocumented_methods:
                    print(f"\t{name}")
            assert len((undocumented_methods)) == 0

        print(f"n_docstrings: {n_docstrings}")


if __name__ == "__main__":
    unittest.main()
