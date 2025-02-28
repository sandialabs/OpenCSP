"""Contains functions to save downsampled sofast measurement file
"""

import os
import sys

from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling as Calibration
import opencsp.app.sofast.lib.DistanceOpticScreen as osd
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir

sys.path.append(os.path.join(opencsp_code_dir(), ".."))
import contrib.test_data_generation.downsample_data_general as ddg  # nopep8


def downsample_measurement(file: str, n: int):
    """Returns downsampled measurement file

    Parameters
    ----------
    file : str
        High res measurement file
    n : int
        Downsample factor

    Returns
    -------
    Measurement
    """
    calibration_orig = Calibration.load_from_hdf(file)
    # Load measurement
    measurement_orig = Measurement.load_from_hdf(file)

    # Downsample measurement
    mask_images = ddg.downsample_images(measurement_orig.mask_images, n)
    fringe_images = ddg.downsample_images(measurement_orig.fringe_images, n)
    dist_optic_screen_measure = osd.DistanceOpticScreen(
        measurement_orig.v_measure_point_facet, measurement_orig.dist_optic_screen
    )
    new_meas = Measurement(
        mask_images=mask_images,
        fringe_images=fringe_images,
        fringe_periods_x=measurement_orig.fringe_periods_x,
        fringe_periods_y=measurement_orig.fringe_periods_y,
        dist_optic_screen_measure=dist_optic_screen_measure,
        date=measurement_orig.date,
        name=measurement_orig.name,
    )
    return [new_meas, calibration_orig]
