"""
Functions to support common operational modes between the various objects used in SOFAST systems. These functions are
provided here as a common "inter-space" between objects, for cases where several object interact but the code to do so
doesn't belong in the object's classes.
"""

import copy
from typing import Callable

import matplotlib.backend_bases
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

import opencsp.app.sofast.lib.MeasurementSofastFringe as msf
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition as ImageAcquisition_DCAM
from opencsp.common.lib.camera.ImageAcquisition_DCAM_color import ImageAcquisition as ImageAcquisition_DCAM_color
from opencsp.common.lib.camera.ImageAcquisition_MSMF import ImageAcquisition as ImageAcquisition_MSMF
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
import opencsp.app.sofast.lib.SofastServiceCallback as ssc
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.hdf5_tools as h5
import opencsp.common.lib.tool.log_tools as lt


def run_exposure_cal(image_acquisition: ImageAcquisitionAbstract, image_projection: ImageProjection) -> None:
    """Runs camera exposure calibration. This adjusts the exposure time of the camera to keep the pixels from being
    under or over saturated. Displays the crosshairs on the projector once finished."""
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


def get_exposure(image_acquisition: ImageAcquisitionAbstract) -> int | None:
    """Returns the exposure time of the camera (microseconds)."""
    if image_acquisition is None:
        lt.error_and_raise(
            RuntimeError,
            "Error in CommonMethods.get_exposure(): "
            + "must initialize image acquisition (camera) before attempting to get the exposure.",
        )
    return image_acquisition.exposure_time


def set_exposure(image_acquisition: ImageAcquisitionAbstract, new_exp: int) -> None:
    """Sets camera exposure time value to the given value (microseconds)"""
    image_acquisition.exposure_time = int(new_exp)


class SofastService:
    """Class that interfaces with SOFAST to run data acquisition and process results"""

    cam_options: dict[str, type[ImageAcquisitionAbstract]] = {
        'DCAM Mono': ImageAcquisition_DCAM,
        'DCAM Color': ImageAcquisition_DCAM_color,
        'MSMF Mono': ImageAcquisition_MSMF,
    }
    """ Defines camera objects to choose from (camera description, python type) """
