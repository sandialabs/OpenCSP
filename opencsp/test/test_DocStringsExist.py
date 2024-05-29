import inspect

# Assume opencsp is in PYHTONPATH
import opencsp as opencsp
import example as example

# TODO: why aren't these imported from import opencsp as opencsp above
from opencsp.app.camera_calibration.lib.ViewAnnotatedImages import ViewAnnotatedImages
from opencsp.app.sofast.SofastGUI import SofastGUI
from opencsp.app.sofast.lib.visualize_setup import visualize_setup

# from opencsp.app.target.target_color.target_color_2d_gradient import target_color_2d_gradient
import opencsp.app.target.target_color.target_color_bullseye_error as target_color_bullseye_error
import opencsp.app.target.target_color.target_color_bullseye as target_color_bullseye
import opencsp.app.target.target_color.target_color_one_color as target_color_one_color
import opencsp.app.target.target_color.target_color_polar as target_color_polar
import opencsp.app.target.target_color.target_color as target_color


def test_docstrings_exist_for_methods():
    sofast_class_list = [
        SofastGUI,
        visualize_setup,
        opencsp.app.sofast.lib.Fringes.Fringes,
        opencsp.app.sofast.lib.ImageCalibrationAbstract.ImageCalibrationAbstract,
        opencsp.app.sofast.lib.ImageCalibrationGlobal.ImageCalibrationGlobal,
        opencsp.app.sofast.lib.ImageCalibrationScaling.ImageCalibrationScaling,
        opencsp.app.sofast.lib.MeasurementSofastFixed.MeasurementSofastFixed,
        opencsp.app.sofast.lib.MeasurementSofastFringe.MeasurementSofastFringe,
        opencsp.app.sofast.lib.ParamsSofastFringe.ParamsSofastFringe,
        opencsp.app.sofast.lib.ParamsSofastFixed.ParamsSofastFixed,
        opencsp.app.sofast.lib.SystemSofastFringe.SystemSofastFringe,
        opencsp.app.sofast.lib.SystemSofastFixed.SystemSofastFixed,
        opencsp.app.sofast.lib.CalibrateDisplayShape.CalibrateDisplayShape,
    ]
    target_class_list = [
        opencsp.app.target.target_color.lib.ImageColor,
        target_color_bullseye_error,
        target_color_bullseye,
        target_color_one_color,
        target_color_polar,
        target_color,
    ]
    camera_calibration_class_list = [
        ViewAnnotatedImages,
        opencsp.app.camera_calibration.lib.calibration_camera,
        opencsp.app.camera_calibration.lib.image_processing,
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
                if callable(getattr(class_module, func)) and not func.startswith("__") and not func.startswith("_")
            ]

        for method in method_list:
            doc_exists = True
            if getattr(class_module, method).__doc__ is None:
                doc_exists = False

            print(f"doc_exists({class_module.__name__}.{method}): " f"{doc_exists}")
            assert doc_exists
