from pathlib import Path

# Assume opencsp is in PYHTONPATH
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast

# TODO: import all user-facing classes here.


def test_docstrings_exist_for_methods():
    class_list = [
        Sofast
        # TODO: List all user-facing classes here.
    ]
    target_class_list = [target_color, target_color_polar, opencsp.app.target.target_color.lib.ImageColor]
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

    for class_module in class_list:
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
