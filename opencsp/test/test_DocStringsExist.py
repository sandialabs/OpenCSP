import inspect
import unittest

# Assume opencsp is in PYTHONPATH
import opencsp as opencsp

# Examples
import example.sofast_calibration.example_calibration_screen_shape
import example.sofast_fringe.example_make_rectangular_screen_definition
import example.sofast_fringe.example_process_in_debug_mode
import example.sofast_fringe.sofast_command_line_tool
import example.sofast_fringe.sofast_temperature_analysis

import opencsp.app.camera_calibration.CameraCalibration
import opencsp.app.camera_calibration.lib.calibration_camera
import opencsp.app.camera_calibration.lib.image_processing
import opencsp.app.camera_calibration.lib.ViewAnnotatedImages
import opencsp.app.select_image_points.SelectImagePoints
import opencsp.app.scene_reconstruction.lib.SceneReconstruction
import opencsp.app.target.target_color.lib.ImageColor

import opencsp.app.sofast.SofastGUI
import opencsp.app.sofast.lib.AbstractMeasurementSofast
import opencsp.app.sofast.lib.BlobIndex
import opencsp.app.sofast.lib.CalibrateDisplayShape
import opencsp.app.sofast.lib.CalibrateSofastFixedDots
import opencsp.app.sofast.lib.CalibrateDisplayShape
import opencsp.app.sofast.lib.CalibrateDisplayShape
import opencsp.app.sofast.lib.DebugOpticsGeometry
import opencsp.app.sofast.lib.DefinitionEnsemble
import opencsp.app.sofast.lib.DefinitionFacet
import opencsp.app.sofast.lib.DisplayShape
import opencsp.app.sofast.lib.DistanceOpticScreen
import opencsp.app.sofast.lib.DotLocationsFixedPattern
import opencsp.app.sofast.lib.Fringes
import opencsp.app.sofast.lib.ImageCalibrationAbstract
import opencsp.app.sofast.lib.ImageCalibrationGlobal
import opencsp.app.sofast.lib.ImageCalibrationScaling
import opencsp.app.sofast.lib.MeasurementSofastFixed
import opencsp.app.sofast.lib.MeasurementSofastFringe
import opencsp.app.sofast.lib.ParamsMaskCalculation
import opencsp.app.sofast.lib.ParamsOpticGeometry
import opencsp.app.sofast.lib.ParamsSofastFixed
import opencsp.app.sofast.lib.ParamsSofastFringe
import opencsp.app.sofast.lib.PatternSofastFixed
import opencsp.app.sofast.lib.ProcessSofastAbstract
import opencsp.app.sofast.lib.ProcessSofastFixed
import opencsp.app.sofast.lib.ProcessSofastFringe
import opencsp.app.sofast.lib.SofastConfiguration
import opencsp.app.sofast.lib.SpatialOrientation
import opencsp.app.sofast.lib.SystemSofastFixed
import opencsp.app.sofast.lib.SystemSofastFringe
import opencsp.app.sofast.lib.calculation_data_classes
import opencsp.app.sofast.lib.image_processing
import opencsp.app.sofast.lib.load_sofast_hdf_data
import opencsp.app.sofast.lib.process_optics_geometry
import opencsp.app.sofast.lib.sofast_common_functions
import opencsp.app.sofast.lib.spatial_processing
import opencsp.app.sofast.lib.SofastInterface

import opencsp.common.lib.camera.Camera
import opencsp.common.lib.camera.image_processing
import opencsp.common.lib.camera.ImageAcquisition_DCAM_color
import opencsp.common.lib.camera.ImageAcquisition_DCAM_mono
import opencsp.common.lib.camera.ImageAcquisition_MSMF
import opencsp.common.lib.camera.ImageAcquisitionAbstract
import opencsp.common.lib.camera.LiveView
import opencsp.common.lib.camera.UCamera

import opencsp.common.lib.csp.Facet
import opencsp.common.lib.csp.FacetEnsemble
import opencsp.common.lib.csp.HeliostatAbstract
import opencsp.common.lib.csp.HeliostatAzEl
import opencsp.common.lib.csp.HeliostatConfiguration
import opencsp.common.lib.csp.LightPath
import opencsp.common.lib.csp.LightPathEnsemble
import opencsp.common.lib.csp.LightSource
import opencsp.common.lib.csp.LightSourcePoint
import opencsp.common.lib.csp.LightSourceSun
import opencsp.common.lib.csp.MirrorAbstract
import opencsp.common.lib.csp.MirrorParametric
import opencsp.common.lib.csp.MirrorParametricRectangular
import opencsp.common.lib.csp.MirrorPoint
import opencsp.common.lib.csp.OpticOrientationAbstract
import opencsp.common.lib.csp.RayTrace
import opencsp.common.lib.csp.RayTraceable
import opencsp.common.lib.csp.Scene
import opencsp.common.lib.csp.SolarField
import opencsp.common.lib.csp.StandardPlotOutput
import opencsp.common.lib.csp.Tower
import opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract
import opencsp.common.lib.csp.sun_position
import opencsp.common.lib.csp.sun_track
import opencsp.common.lib.csp.visualize_orthorectified_image

import opencsp.common.lib.cv.annotations.AbstractAnnotations
import opencsp.common.lib.cv.annotations.HotspotAnnotation
import opencsp.common.lib.cv.annotations.PointAnnotations
import opencsp.common.lib.cv.CacheableImage
import opencsp.common.lib.cv.fiducials.AbstractFiducials
import opencsp.common.lib.cv.fiducials.BcsFiducial
import opencsp.common.lib.cv.fiducials.PointFiducials
import opencsp.common.lib.cv.image_filters
import opencsp.common.lib.cv.image_reshapers
import opencsp.common.lib.cv.OpticalFlow
import opencsp.common.lib.cv.spot_analysis.ImagesStream
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable
import opencsp.common.lib.cv.SpotAnalysis
from opencsp.common.lib.cv.spot_analysis.image_processor import *

import opencsp.common.lib.deflectometry.CalibrationCameraPosition
import opencsp.common.lib.deflectometry.ImageProjection
import opencsp.common.lib.deflectometry.ImageProjectionSetupGUI
import opencsp.common.lib.deflectometry.ParamsSlopeSolver
import opencsp.common.lib.deflectometry.ParamsSlopeSolverAbstract
import opencsp.common.lib.deflectometry.ParamsSlopeSolverParaboloid
import opencsp.common.lib.deflectometry.ParamsSlopeSolverPlano
import opencsp.common.lib.deflectometry.slope_fitting_2d
import opencsp.common.lib.deflectometry.SlopeSolver
import opencsp.common.lib.deflectometry.SlopeSolverData
import opencsp.common.lib.deflectometry.SlopeSolverDataDebug
import opencsp.common.lib.deflectometry.Surface2DAbstract
import opencsp.common.lib.deflectometry.Surface2DParabolic
import opencsp.common.lib.deflectometry.Surface2DPlano

import opencsp.common.lib.file.AbstractAttributeParser
import opencsp.common.lib.file.AttributesManager
import opencsp.common.lib.file.CsvColumns
import opencsp.common.lib.file.CsvInterface
import opencsp.common.lib.file.SimpleCsv

import opencsp.common.lib.geo

import opencsp.common.lib.geometry.EdgeXY
import opencsp.common.lib.geometry.FunctionXYAbstract
import opencsp.common.lib.geometry.FunctionXYContinuous
import opencsp.common.lib.geometry.FunctionXYDiscrete
import opencsp.common.lib.geometry.FunctionXYGrid
import opencsp.common.lib.geometry.Intersection
import opencsp.common.lib.geometry.LineXY
import opencsp.common.lib.geometry.LoopXY
import opencsp.common.lib.geometry.Pxy
import opencsp.common.lib.geometry.Pxyz
import opencsp.common.lib.geometry.RegionXY
import opencsp.common.lib.geometry.TransformXYZ
import opencsp.common.lib.geometry.Uxy
import opencsp.common.lib.geometry.Uxyz
import opencsp.common.lib.geometry.Vxy
import opencsp.common.lib.geometry.Vxyz
import opencsp.common.lib.geometry.angle

import opencsp.common.lib.opencsp_path.optical_analysis_data_path
import opencsp.common.lib.opencsp_path.data_path_for_test
import opencsp.common.lib.opencsp_path.opencsp_root_path

import opencsp.common.lib.photogrammetry.ImageMarker
import opencsp.common.lib.photogrammetry.bundle_adjustment
import opencsp.common.lib.photogrammetry.photogrammetry

import opencsp.common.lib.process.lib.CalledProcessError
import opencsp.common.lib.process.lib.ProcessOutputLine
import opencsp.common.lib.process.lib.ServerSynchronizerError
import opencsp.common.lib.process.MultiprocessNonDaemonic
import opencsp.common.lib.process.parallel_file_tools
import opencsp.common.lib.process.parallel_video_tools
import opencsp.common.lib.process.ParallelPartitioner
import opencsp.common.lib.process.ServerSynchronizer
import opencsp.common.lib.process.subprocess_tools

import opencsp.common.lib.render.axis_3d
import opencsp.common.lib.render.Color
import opencsp.common.lib.render.figure_management
import opencsp.common.lib.render.general_plot
import opencsp.common.lib.render.image_plot
import opencsp.common.lib.render.ImageAttributeParser
import opencsp.common.lib.render.lib.AbstractPlotHandler
import opencsp.common.lib.render.lib.PowerpointImage
import opencsp.common.lib.render.lib.PowerpointShape
import opencsp.common.lib.render.lib.PowerpointText
import opencsp.common.lib.render.pandas_plot
import opencsp.common.lib.render.PlotAnnotation
import opencsp.common.lib.render.PowerpointSlide
import opencsp.common.lib.render.VideoHandler
import opencsp.common.lib.render.view_spec
import opencsp.common.lib.render.View3d

import opencsp.common.lib.render_control.RenderControlAxis
import opencsp.common.lib.render_control.RenderControlBcs
import opencsp.common.lib.render_control.RenderControlDeflectometryInstrument
import opencsp.common.lib.render_control.RenderControlEnsemble
import opencsp.common.lib.render_control.RenderControlEvaluateHeliostats3d
import opencsp.common.lib.render_control.RenderControlFacet
import opencsp.common.lib.render_control.RenderControlFacetEnsemble
import opencsp.common.lib.render_control.RenderControlFigure
import opencsp.common.lib.render_control.RenderControlFigureRecord
import opencsp.common.lib.render_control.RenderControlFramesNoDuplicates
import opencsp.common.lib.render_control.RenderControlFunctionXY
import opencsp.common.lib.render_control.RenderControlHeatmap
import opencsp.common.lib.render_control.RenderControlHeliostat
import opencsp.common.lib.render_control.RenderControlHeliostats3d
import opencsp.common.lib.render_control.RenderControlHeliostatTracks
import opencsp.common.lib.render_control.RenderControlIntersection
import opencsp.common.lib.render_control.RenderControlKeyCorners
import opencsp.common.lib.render_control.RenderControlKeyFramesGivenManual
import opencsp.common.lib.render_control.RenderControlKeyTracks
import opencsp.common.lib.render_control.RenderControlLightPath
import opencsp.common.lib.render_control.RenderControlMirror
import opencsp.common.lib.render_control.RenderControlPointSeq
import opencsp.common.lib.render_control.RenderControlPowerpointPresentation
import opencsp.common.lib.render_control.RenderControlPowerpointSlide
import opencsp.common.lib.render_control.RenderControlRayTrace
import opencsp.common.lib.render_control.RenderControlSolarField
import opencsp.common.lib.render_control.RenderControlSurface
import opencsp.common.lib.render_control.RenderControlText
import opencsp.common.lib.render_control.RenderControlTower
import opencsp.common.lib.render_control.RenderControlTrajectoryAnalysis
import opencsp.common.lib.render_control.RenderControlVideo
import opencsp.common.lib.render_control.RenderControlVideoFrames
import opencsp.common.lib.render_control.RenderControlVideoTracks

import opencsp.common.lib.target.target_color_1d_gradient
import opencsp.common.lib.target.target_color_2d_rgb
import opencsp.common.lib.target.target_color_convert
import opencsp.common.lib.target.target_image
import opencsp.common.lib.target.TargetAbstract
import opencsp.common.lib.target.TargetColor

import opencsp.common.lib.tool.dict_tools
import opencsp.common.lib.tool.exception_tools
import opencsp.common.lib.tool.file_tools
import opencsp.common.lib.tool.hdf5_tools
import opencsp.common.lib.tool.image_tools
import opencsp.common.lib.tool.list_tools
import opencsp.common.lib.tool.log_tools
import opencsp.common.lib.tool.math_tools
import opencsp.common.lib.tool.string_tools
import opencsp.common.lib.tool.system_tools
import opencsp.common.lib.tool.time_date_tools
import opencsp.common.lib.tool.tk_tools
import opencsp.common.lib.tool.typing_tools
import opencsp.common.lib.tool.unit_conversion

import opencsp.common.lib.uas.Scan
import opencsp.common.lib.uas.ScanPass
import opencsp.common.lib.uas.WayPoint


class test_Docstrings(unittest.TestCase):
    camera_calibration_class_list = [
        opencsp.app.camera_calibration.CameraCalibration.CalibrationGUI,
        opencsp.app.camera_calibration.lib.ViewAnnotatedImages.ViewAnnotatedImages,
    ]

    scene_reconstruction_class_list = [opencsp.app.scene_reconstruction.lib.SceneReconstruction.SceneReconstruction]

    select_image_points_class_list = [opencsp.app.select_image_points.SelectImagePoints.SelectImagePoints]

    sofast_class_list = [
        opencsp.app.sofast.SofastGUI.SofastGUI,
        opencsp.app.sofast.lib.AbstractMeasurementSofast.AbstractMeasurementSofast,
        opencsp.app.sofast.lib.BlobIndex.BlobIndex,
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
    ]

    target_class_list = [opencsp.app.target.target_color.lib.ImageColor.ImageColor]

    camera_calibration_class_list = [
        opencsp.app.camera_calibration.lib.calibration_camera,
        opencsp.app.camera_calibration.lib.image_processing,
        opencsp.app.camera_calibration.lib.ViewAnnotatedImages,
    ]

    scene_reconstruction_class_list = [opencsp.app.scene_reconstruction.lib.SceneReconstruction.SceneReconstruction]

    select_image_points_class_list = [opencsp.app.select_image_points.SelectImagePoints.SelectImagePoints]

    sofast_class_list = [
        opencsp.app.sofast.SofastGUI.SofastGUI,
        opencsp.app.sofast.lib.AbstractMeasurementSofast.AbstractMeasurementSofast,
        opencsp.app.sofast.lib.BlobIndex.BlobIndex,
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
        opencsp.app.sofast.lib.SofastInterface.SofastInterface,
    ]

    target_class_list = [opencsp.app.target.target_color.lib.ImageColor.ImageColor]

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
        opencsp.common.lib.cv.CacheableImage.CacheableImage,
        opencsp.common.lib.cv.OpticalFlow.OpticalFlow,
        opencsp.common.lib.cv.SpotAnalysis.SpotAnalysis,
        opencsp.common.lib.cv.image_filters,
        opencsp.common.lib.cv.image_reshapers,
        opencsp.common.lib.cv.annotations.AbstractAnnotations.AbstractAnnotations,
        opencsp.common.lib.cv.annotations.HotspotAnnotation.HotspotAnnotation,
        opencsp.common.lib.cv.annotations.PointAnnotations.PointAnnotations,
        opencsp.common.lib.cv.fiducials.AbstractFiducials.AbstractFiducials,
        opencsp.common.lib.cv.fiducials.BcsFiducial.BcsFiducial,
        opencsp.common.lib.cv.fiducials.PointFiducials.PointFiducials,
        opencsp.common.lib.cv.spot_analysis.ImagesStream.ImagesStream,
        opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream.SpotAnalysisImagesStream,
        opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable.SpotAnalysisOperable,
        AbstractAggregateImageProcessor,
        AbstractSpotAnalysisImageProcessor,
        AbstractVisualizationImageProcessor,
        AverageByGroupImageProcessor,
        BcsLocatorImageProcessor,
        ConvolutionImageProcessor,
        CroppingImageProcessor,
        EchoImageProcessor,
        ExposureDetectionImageProcessor,
        ViewFalseColorImageProcessor,
        HotspotImageProcessor,
        LogScaleImageProcessor,
        NullImageSubtractionImageProcessor,
        PopulationStatisticsImageProcessor,
        SupportingImagesCollectorImageProcessor,
        View3dImageProcessor,
        ViewCrossSectionImageProcessor,
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
        opencsp.common.lib.csp.Facet.Facet,
        opencsp.common.lib.csp.FacetEnsemble.FacetEnsemble,
        opencsp.common.lib.csp.HeliostatAbstract.HeliostatAbstract,
        opencsp.common.lib.csp.HeliostatAzEl.HeliostatAzEl,
        opencsp.common.lib.csp.HeliostatConfiguration.HeliostatConfiguration,
        opencsp.common.lib.csp.LightPath.LightPath,
        opencsp.common.lib.csp.LightPathEnsemble.LightPathEnsemble,
        opencsp.common.lib.csp.LightSource.LightSource,
        opencsp.common.lib.csp.LightSourcePoint.LightSourcePoint,
        opencsp.common.lib.csp.LightSourceSun.LightSourceSun,
        opencsp.common.lib.csp.MirrorAbstract.MirrorAbstract,
        opencsp.common.lib.csp.MirrorParametric.MirrorParametric,
        opencsp.common.lib.csp.MirrorParametricRectangular.MirrorParametricRectangular,
        opencsp.common.lib.csp.MirrorPoint.MirrorPoint,
        opencsp.common.lib.csp.OpticOrientationAbstract.OpticOrientationAbstract,
        opencsp.common.lib.csp.RayTrace.RayTrace,
        opencsp.common.lib.csp.RayTraceable.RayTraceable,
        opencsp.common.lib.csp.Scene.Scene,
        opencsp.common.lib.csp.SolarField.SolarField,
        opencsp.common.lib.csp.StandardPlotOutput.StandardPlotOutput,
        opencsp.common.lib.csp.Tower.Tower,
        opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract.VisualizeOrthorectifiedSlopeAbstract,
        opencsp.common.lib.csp.sun_position,
        opencsp.common.lib.csp.sun_track,
        opencsp.common.lib.csp.visualize_orthorectified_image,
    ]

    deflectometry_class_list = [
        opencsp.common.lib.deflectometry.CalibrationCameraPosition.CalibrationCameraPosition,
        opencsp.common.lib.deflectometry.ImageProjection.ImageProjection,
        opencsp.common.lib.deflectometry.ImageProjectionSetupGUI.ImageProjectionGUI,
        opencsp.common.lib.deflectometry.ImageProjectionSetupGUI.ImageProjectionData,
        opencsp.common.lib.deflectometry.ParamsSlopeSolver.ParamsSlopeSolver,
        opencsp.common.lib.deflectometry.ParamsSlopeSolverAbstract.ParamsSlopeSolverAbstract,
        opencsp.common.lib.deflectometry.ParamsSlopeSolverParaboloid.ParamsSlopeSolverParaboloid,
        opencsp.common.lib.deflectometry.ParamsSlopeSolverPlano.ParamsSlopeSolverPlano,
        opencsp.common.lib.deflectometry.SlopeSolver.SlopeSolver,
        opencsp.common.lib.deflectometry.SlopeSolverData.SlopeSolverData,
        opencsp.common.lib.deflectometry.SlopeSolverDataDebug.SlopeSolverDataDebug,
        opencsp.common.lib.deflectometry.Surface2DAbstract.Surface2DAbstract,
        opencsp.common.lib.deflectometry.Surface2DParabolic.Surface2DParabolic,
        opencsp.common.lib.deflectometry.Surface2DPlano.Surface2DPlano,
        opencsp.common.lib.deflectometry.slope_fitting_2d,
    ]

    file_class_list = [
        opencsp.common.lib.file.AbstractAttributeParser.AbstractAttributeParser,
        opencsp.common.lib.file.AttributesManager.AttributesManager,
        opencsp.common.lib.file.CsvColumns.CsvColumns,
        opencsp.common.lib.file.CsvInterface.CsvInterface,
        opencsp.common.lib.file.SimpleCsv.SimpleCsv,
    ]

    geo_class_list = [opencsp.common.lib.geo.lon_lat_nsttf]

    geometry_class_list = [
        opencsp.common.lib.geometry.EdgeXY.EdgeXY,
        opencsp.common.lib.geometry.FunctionXYAbstract.FunctionXYAbstract,
        opencsp.common.lib.geometry.FunctionXYContinuous.FunctionXYContinuous,
        opencsp.common.lib.geometry.FunctionXYDiscrete.FunctionXYDiscrete,
        opencsp.common.lib.geometry.FunctionXYGrid.FunctionXYGrid,
        opencsp.common.lib.geometry.Intersection.Intersection,
        opencsp.common.lib.geometry.LineXY.LineXY,
        opencsp.common.lib.geometry.LoopXY.LoopXY,
        opencsp.common.lib.geometry.Pxy.Pxy,
        opencsp.common.lib.geometry.Pxyz.Pxyz,
        opencsp.common.lib.geometry.RegionXY.RegionXY,
        opencsp.common.lib.geometry.TransformXYZ.TransformXYZ,
        opencsp.common.lib.geometry.Uxy.Uxy,
        opencsp.common.lib.geometry.Uxyz.Uxyz,
        opencsp.common.lib.geometry.Vxy.Vxy,
        opencsp.common.lib.geometry.Vxyz.Vxyz,
        opencsp.common.lib.geometry.angle,
    ]

    opencsp_path_class_list = [
        opencsp.common.lib.opencsp_path.data_path_for_test,
        opencsp.common.lib.opencsp_path.opencsp_root_path,
        opencsp.common.lib.opencsp_path.optical_analysis_data_path,
    ]

    photogrammetry_class_list = [
        opencsp.common.lib.photogrammetry.ImageMarker.ImageMarker,
        opencsp.common.lib.photogrammetry.bundle_adjustment,
        opencsp.common.lib.photogrammetry.photogrammetry,
    ]

    process_class_list = [
        opencsp.common.lib.process.MultiprocessNonDaemonic.MultiprocessNonDaemonic,
        opencsp.common.lib.process.ParallelPartitioner.ParallelPartitioner,
        opencsp.common.lib.process.ServerSynchronizer.ServerSynchronizer,
        opencsp.common.lib.process.parallel_file_tools,
        opencsp.common.lib.process.parallel_video_tools,
        opencsp.common.lib.process.subprocess_tools,
        opencsp.common.lib.process.lib.CalledProcessError.CalledProcessError,
        opencsp.common.lib.process.lib.ProcessOutputLine.ProcessOutputLine,
        opencsp.common.lib.process.lib.ServerSynchronizerError.ServerSynchronizerError,
    ]

    render_class_list = [
        opencsp.common.lib.render.Color.Color,
        opencsp.common.lib.render.ImageAttributeParser.ImageAttributeParser,
        opencsp.common.lib.render.PlotAnnotation.PlotAnnotation,
        opencsp.common.lib.render.PowerpointSlide.PowerpointSlide,
        opencsp.common.lib.render.VideoHandler.VideoHandler,
        opencsp.common.lib.render.View3d.View3d,
        opencsp.common.lib.render.axis_3d,
        opencsp.common.lib.render.figure_management,
        opencsp.common.lib.render.general_plot,
        opencsp.common.lib.render.image_plot,
        opencsp.common.lib.render.pandas_plot,
        opencsp.common.lib.render.view_spec,
        opencsp.common.lib.render.lib.AbstractPlotHandler.AbstractPlotHandler,
        opencsp.common.lib.render.lib.PowerpointImage.PowerpointImage,
        opencsp.common.lib.render.lib.PowerpointShape.PowerpointShape,
        opencsp.common.lib.render.lib.PowerpointText.PowerpointText,
    ]

    render_control_class_list = [
        opencsp.common.lib.render_control.RenderControlAxis.RenderControlAxis,
        opencsp.common.lib.render_control.RenderControlBcs.RenderControlBcs,
        opencsp.common.lib.render_control.RenderControlDeflectometryInstrument.RenderControlDeflectometryInstrument,
        opencsp.common.lib.render_control.RenderControlEnsemble.RenderControlEnsemble,
        opencsp.common.lib.render_control.RenderControlEvaluateHeliostats3d.RenderControlEvaluateHeliostats3d,
        opencsp.common.lib.render_control.RenderControlFacet.RenderControlFacet,
        opencsp.common.lib.render_control.RenderControlFacetEnsemble.RenderControlFacetEnsemble,
        opencsp.common.lib.render_control.RenderControlFigure.RenderControlFigure,
        opencsp.common.lib.render_control.RenderControlFigureRecord.RenderControlFigureRecord,
        opencsp.common.lib.render_control.RenderControlFramesNoDuplicates.RenderControlFramesNoDuplicates,
        opencsp.common.lib.render_control.RenderControlFunctionXY.RenderControlFunctionXY,
        opencsp.common.lib.render_control.RenderControlHeatmap.RenderControlHeatmap,
        opencsp.common.lib.render_control.RenderControlHeliostat.RenderControlHeliostat,
        opencsp.common.lib.render_control.RenderControlHeliostatTracks.RenderControlHeliostatTracks,
        opencsp.common.lib.render_control.RenderControlHeliostats3d.RenderControlHeliostats3d,
        opencsp.common.lib.render_control.RenderControlIntersection.RenderControlIntersection,
        opencsp.common.lib.render_control.RenderControlKeyCorners.RenderControlKeyCorners,
        opencsp.common.lib.render_control.RenderControlKeyFramesGivenManual.RenderControlKeyFramesGivenManual,
        opencsp.common.lib.render_control.RenderControlKeyTracks.RenderControlKeyTracks,
        opencsp.common.lib.render_control.RenderControlLightPath.RenderControlLightPath,
        opencsp.common.lib.render_control.RenderControlMirror.RenderControlMirror,
        opencsp.common.lib.render_control.RenderControlPointSeq.RenderControlPointSeq,
        opencsp.common.lib.render_control.RenderControlPowerpointPresentation.RenderControlPowerpointPresentation,
        opencsp.common.lib.render_control.RenderControlPowerpointSlide.RenderControlPowerpointSlide,
        opencsp.common.lib.render_control.RenderControlRayTrace.RenderControlRayTrace,
        opencsp.common.lib.render_control.RenderControlSolarField.RenderControlSolarField,
        opencsp.common.lib.render_control.RenderControlSurface.RenderControlSurface,
        opencsp.common.lib.render_control.RenderControlText.RenderControlText,
        opencsp.common.lib.render_control.RenderControlTower.RenderControlTower,
        opencsp.common.lib.render_control.RenderControlTrajectoryAnalysis.RenderControlTrajectoryAnalysis,
        opencsp.common.lib.render_control.RenderControlVideo.RenderControlVideo,
        opencsp.common.lib.render_control.RenderControlVideoFrames.RenderControlVideoFrames,
        opencsp.common.lib.render_control.RenderControlVideoTracks.RenderControlVideoTracks,
    ]

    common_target_class_list = [
        opencsp.common.lib.target.TargetAbstract.TargetAbstract,
        opencsp.common.lib.target.TargetColor.TargetColor,
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

    uas_class_list = [
        opencsp.common.lib.uas.Scan.Scan,
        opencsp.common.lib.uas.ScanPass.ScanPass,
        opencsp.common.lib.uas.WayPoint.WayPoint,
    ]

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
        undocumented_methods: dict[str, list[str]] = {}

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

            for method in method_list:
                doc_exists = True
                if inspect.getdoc(getattr(class_module, method)) is None:
                    doc_exists = False

                method_name = f"{class_module.__name__}.{method}"
                print(f"doc_exists({method_name}): " f"{doc_exists}")
                if not doc_exists:
                    if class_module.__name__ not in undocumented_methods:
                        undocumented_methods[class_module.__name__] = []
                    undocumented_methods[class_module.__name__].append(method)
                else:
                    n_docstrings += 1

            if len((undocumented_methods)) != 0:
                print(f"Found undocumented methods in {class_module}!")

        n_undocumented_methods = sum([len(v) for v in undocumented_methods.values()])
        if n_undocumented_methods > 0:
            print(f"\nFound {n_undocumented_methods} total undocumented metshods!\n")
            for class_module_name, class_ums in undocumented_methods.items():
                print(f"Undocumented methods in {class_module_name} ({len(class_ums)}):")
                for class_um in class_ums:
                    print(f"\t{class_um}")

        print(f"n_docstrings: {n_docstrings}")

        self.assertEqual(n_undocumented_methods, 0)


if __name__ == "__main__":
    unittest.main()
