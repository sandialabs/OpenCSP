"""
Functions to support common operational modes between the various objects used in SOFAST systems. These functions are
provided here as a common "inter-space" between objects, for cases where several object interact but the code to do so
doesn't belong in the object's classes.
"""

from typing import Callable, Literal

import numpy as np

from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
import opencsp.app.sofast.lib.SystemSofastFringe as ssf
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.tool.log_tools as lt


def check_system_fringe_loaded(sys_fringe: ssf.SystemSofastFringe | None, method_name: str) -> Literal[True]:
    """Checks if the system class has been instantiated, and that the camera and projector instances are still available.

    The main purpose in using this method is to have a consisten error message across the various SOFAST Fringe interfaces.

    Params
    ------
    sys_fringe: SystemSofastFringe
        The system which we wish to verify is not None
    method_name: str
        Name of the calling function or method, for use in error message

    Raises
    ------
    RuntimeError:
        If the system hasn't been loaded yet, or the system is loaded without the necessary prerequisites
    """
    if sys_fringe is not None:
        if ImageAcquisitionAbstract.instance() is None:
            # If we've done everything correctly then this should be unreachable code
            raise RuntimeError(f"In {method_name}: SOFAST system instance exists without a connected camera!")
        if ImageProjection.instance() is None:
            # If we've done everything correctly then this should be unreachable code
            raise RuntimeWarning(
                "In " + method_name + ": " + "SOFAST system instance exists without a loaded projector!"
            )
        return True
    else:
        raise RuntimeError(
            f"In {method_name}: Both ImageAcquisiton and ImageProjection must be loaded before using the SOFAST system."
        )


def check_calibration_loaded(sys_fringe: ssf.SystemSofastFringe, method_name: str) -> Literal[True]:
    """Checks if calibration is loaded. Returns True if loaded.

    Params
    ------
    sys_fringe: SystemSofastFringe
        The system which we wish to verify has a calibration loaded
    method_name: str
        Name of the calling function or method, for use in error message

    Raises:
    -------
    RuntimeError:
        If the system hasn't been loaded yet, or the system is loaded without the necessary prerequisites
    """
    check_system_fringe_loaded(sys_fringe, method_name)

    if sys_fringe.calibration is None:  # Not loaded
        raise RuntimeError(f"In {method_name}: Camera-Projector calibration must be loaded/performed.")
    else:  # Loaded
        return True


def check_camera_loaded(method_name: str, image_acquisition: ImageAcquisitionAbstract = None) -> Literal[True]:
    """Checks that the camera instance has been loaded.

    Params
    ------
    method_name: str
        Name of the calling function or method, for use in error message
    image_acquisition: ImageAcquisitionAbstract, optional
        The camera that we want to check is loaded. Default is the global instance

    Raises:
    -------
    RuntimeError:
        If the system hasn't been loaded yet, or the system is loaded without the necessary prerequisites
    image_acquisition: ImageAcquisitionAbstract
        The camera to check the existance of. If None, then the global instance is used.
    """
    image_acquisition, _ = get_default_or_global_instances(image_acquisition_default=image_acquisition)
    if image_acquisition is None:  # Not loaded
        raise RuntimeError("In " + method_name + ": Camera must be loaded.")
    else:  # Loaded
        return True


def check_projector_loaded(method_name: str, image_projection: ImageProjection = None) -> Literal[True]:
    """Checks that the projector instance has been loaded.

    Params
    ------
    method_name: str
        Name of the calling function or method, for use in error message
    image_projection: ImageProjection, optional
        The projector that we want to check is loaded. Default is the global instance

    Raises:
    -------
    RuntimeError:
        If the system hasn't been loaded yet, or the system is loaded without the necessary prerequisites
    image_projection: ImageProjection
        The projector to check the existance of. If None, then the global instance is used.
    """
    _, image_projection = get_default_or_global_instances(image_projection_default=image_projection)
    if image_projection is None:  # Not loaded
        raise RuntimeError("In " + method_name + ": Projector must be loaded.")
    else:  # Loaded
        return True


def get_default_or_global_instances(
    image_acquisition_default: ImageAcquisitionAbstract = None, image_projection_default: ImageProjection = None
) -> tuple[ImageAcquisitionAbstract | None, ImageProjection | None]:
    """Get the given values, or their global instance counterparts if not set.

    Args:
    image_acquisition_default (ImageAcquisitionAbstract, optional):
        The camera to be returned. If None, then the global camera instance will be returned. Defaults to None.
    image_projection_default (ImageProjection, optional):
        The projector to be returned. If None, then the global projector instance will be returned. Defaults to None.

    Returns:
    --------
    instances: tuple[ImageAcquisitionAbstract | None, ImageProjection | None]
        The given or global instances
    """
    ia_ret = image_acquisition_default
    ip_ret = image_projection_default
    if ia_ret is None:
        ia_ret = ImageAcquisitionAbstract.instance()
    if ip_ret is None:
        ip_ret = ImageProjection.instance()
    return ia_ret, ip_ret


def run_exposure_cal(
    image_acquisition: ImageAcquisitionAbstract = None,
    image_projection: ImageProjection = None,
    on_done: Callable = None,
) -> None:
    """Runs camera exposure calibration. This adjusts the exposure time of the camera to keep the pixels from being
    under or over saturated. Displays the crosshairs on the projector once finished.

    Params:
    -------
    image_acquisition: ImageAcquisitionAbstract
        The camera to auto-calibrate the exposure of. If None, then the global instance is used.
    image_projection: ImageProjection, optional
        The projector instance to display a flat white image during exposure calibration. If None and a global instance
        is available, then the global instance is used. Otherwise this step is skipped.
    on_done: Callable, optional
        If set, then this callback will be evaluated when the exposure calibration has completed

    Raises:
    -------
    RuntimeError:
        If image_acquisition isn't given and a global instance isn't available.
    """
    image_acquisition, image_projection = get_default_or_global_instances(image_acquisition, image_projection)

    # Check that a camera is available
    check_camera_loaded("run_exposure_calibration", image_acquisition)

    # Try to display a white image on the projector, if there is one
    if image_projection is None:
        lt.info("Running calibration without displayed white image.")
        image_acquisition.calibrate_exposure()
        if on_done is not None:
            on_done()
    else:
        lt.info("Running calibration with displayed white image.")

        def run_cal():
            image_acquisition.calibrate_exposure()
            image_projection.show_crosshairs()
            if on_done is not None:
                on_done()

        white_image = (
            np.array(image_projection.get_black_array_active_area()) + image_projection.display_data.projector_max_int
        )
        image_projection.display_image_in_active_area(white_image)
        image_projection.root.after(100, run_cal)


def get_exposure(image_acquisition: ImageAcquisitionAbstract = None) -> int | None:
    """Returns the exposure time of the camera (microseconds).

    Params
    ------
    image_acquisition: ImageAcquisitionAbstract
        The camera to auto-calibrate the exposure of. If None, then the global instance is used.

    Raises
    ------
    RuntimeError:
        If there is no camera instance
    """
    check_camera_loaded("get_exposure", image_acquisition=image_acquisition)
    image_acquisition, _ = get_default_or_global_instances(image_acquisition_default=image_acquisition)
    return image_acquisition.exposure_time


def set_exposure(new_exp: int, image_acquisition: ImageAcquisitionAbstract = None) -> None:
    """Sets camera exposure time value to the given value (microseconds)

    Params
    ------
    new_exp: int
        The exposure time to set, in microseconds.
    image_acquisition: ImageAcquisitionAbstract
        The camera to set the exposure of. If None, then the global instance is used.

    Raises
    ------
    RuntimeError:
        If there is no camera instance
    """
    check_camera_loaded("set_exposure", image_acquisition=image_acquisition)
    image_acquisition, _ = get_default_or_global_instances(image_acquisition_default=image_acquisition)
    image_acquisition.exposure_time = int(new_exp)
