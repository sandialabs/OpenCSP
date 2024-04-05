"""
Functions to support common operational modes between the various objects used in SOFAST systems. These functions are
provided here as a common "inter-space" between objects, for cases where several object interact but the code to do so
doesn't belong in the object's classes.
"""

from typing import Literal

import numpy as np

from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition as ImageAcquisition_DCAM
from opencsp.common.lib.camera.ImageAcquisition_DCAM_color import ImageAcquisition as ImageAcquisition_DCAM_color
from opencsp.common.lib.camera.ImageAcquisition_MSMF import ImageAcquisition as ImageAcquisition_MSMF
import opencsp.app.sofast.lib.SystemSofastFringe as ssf
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.tool.log_tools as lt


def check_system_fringe_loaded(sys_fringe: ssf.SystemSofastFringe | None, method_name: str) -> Literal[True]:
    """Checks if the system class has been instantiated, and that the camera and projector instances are still available.

    Raises:
    -------
    RuntimeError:
        If the system hasn't been loaded yet, or the system is loaded without the necessary prerequisites
    """
    if sys_fringe is not None:
        if ImageAcquisitionAbstract.instance() is None:
            # If we've done everything correctly then this should be unreachable code
            raise RuntimeError('In ' + method_name + ': ' +
                               'SOFAST system instance exists without a connected camera!')
        if ImageProjection.instance() is None:
            # If we've done everything correctly then this should be unreachable code
            raise RuntimeWarning('In ' + method_name + ': ' +
                                 'SOFAST system instance exists without a loaded projector!')
        return True
    else:
        raise RuntimeError('In ' + method_name + ': ' +
                           'Both ImageAcquisiton and ImageProjection must be loaded before using the SOFAST system.')


def check_calibration_loaded(sys_fringe: ssf.SystemSofastFringe, method_name: str) -> Literal[True]:
    """Checks if calibration is loaded. Returns True if loaded.

    Raises:
    -------
    RuntimeError:
        If the system hasn't been loaded yet, or the system is loaded without the necessary prerequisites
    """
    check_system_fringe_loaded(sys_fringe, method_name)

    if sys_fringe.calibration is None:  # Not loaded
        raise RuntimeError('In ' + method_name + ': ' +
                           'Camera-Projector calibration must be loaded/performed.')
    else:  # Loaded
        return True


def check_camera_loaded(method_name: str) -> Literal[True]:
    """Checks that the camera instance has been loaded.

    Raises:
    -------
    RuntimeError:
        If the system hasn't been loaded yet, or the system is loaded without the necessary prerequisites
    """
    if ImageAcquisitionAbstract.instance() is None:  # Not loaded
        raise RuntimeError('In ' + method_name + ': Camera must be loaded.')
        return False
    else:  # Loaded
        return True


def check_projector_loaded(method_name: str) -> Literal[True]:
    """Checks that the projector instance has been loaded.

    Raises:
    -------
    RuntimeError:
        If the system hasn't been loaded yet, or the system is loaded without the necessary prerequisites
    """
    if ImageProjection.instance() is None:  # Not loaded
        raise RuntimeError('In ' + method_name + ': Projector must be loaded.')
        return False
    else:  # Loaded
        return True


def get_default_or_global_instances(
    image_acquisition_default: ImageAcquisitionAbstract = None,
    image_projection_default: ImageProjection = None,
) -> tuple[ImageAcquisitionAbstract | None, ImageProjection | None]:
    """ Get the given values, or their global instance counterparts if not set.

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


def run_exposure_cal(image_acquisition: ImageAcquisitionAbstract = None, image_projection: ImageProjection = None) -> None:
    """Runs camera exposure calibration. This adjusts the exposure time of the camera to keep the pixels from being
    under or over saturated. Displays the crosshairs on the projector once finished.

    Params:
    -------
    image_acquisition: ImageAcquisitionAbstract
        The camera to auto-calibrate the exposure of. If None, then the global instance is used.
    image_projection: ImageProjection, optional
        The projector instance to display a flat white image during exposure calibration. If None and a global instance
        is available, then the global instance is used. Otherwise this step is skipped.

    Raises:
    -------
    RuntimeError:
        If image_acquisition isn't given and a global instance isn't available.
    """
    image_acquisition, image_projection = get_default_or_global_instances(image_acquisition, image_projection)

    # Check that a camera is available
    if image_acquisition is None:
        lt.error_and_raise(RuntimeError, 'Error in CommonMethods.run_exposure_cal(): ' +
                           'camera must be connected.')

    # Try to display a white image on the projector, if there is one
    if image_projection is None:
        lt.info('Running calibration without displayed white image.')
        image_acquisition.calibrate_exposure()
    else:
        lt.info('Running calibration with displayed white image.')

        def run_cal():
            image_acquisition.calibrate_exposure()
            image_projection.show_crosshairs()

        white_image = np.array(image_projection.zeros()) + image_projection.max_int
        image_projection.display_image_in_active_area(white_image)
        image_projection.root.after(100, run_cal)


def get_exposure(image_acquisition: ImageAcquisitionAbstract = None) -> int | None:
    """Returns the exposure time of the camera (microseconds).

    Params:
    -------
    image_acquisition: ImageAcquisitionAbstract
        The camera to auto-calibrate the exposure of. If None, then the global instance is used.
    """
    image_acquisition, _ = get_default_or_global_instances(image_acquisition=image_acquisition)

    if image_acquisition is None:
        lt.error_and_raise(
            RuntimeError,
            "Error in CommonMethods.get_exposure(): "
            + "must initialize image acquisition (camera) before attempting to get the exposure.",
        )
    return image_acquisition.exposure_time


def set_exposure(new_exp: int, image_acquisition: ImageAcquisitionAbstract = None) -> None:
    """Sets camera exposure time value to the given value (microseconds)

    Params:
    -------
    new_exp: int
        The exposure time to set, in microseconds.
    image_acquisition: ImageAcquisitionAbstract
        The camera to auto-calibrate the exposure of. If None, then the global instance is used.
    """
    image_acquisition, _ = get_default_or_global_instances(image_acquisition=image_acquisition)

    image_acquisition.exposure_time = int(new_exp)
