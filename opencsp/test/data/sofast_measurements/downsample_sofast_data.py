"""Contains functions to save downsampled sofast measurement file
"""

from opencsp.app.sofast.lib.Measurement import Measurement
from opencsp.common.lib.test.downsample_data import downsample_images


def downsample_measurement(file: str, n: int) -> Measurement:
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
    # Load measurement
    measurement_orig = Measurement.load_from_hdf(file)

    # Downsample measurement
    mask_images = downsample_images(measurement_orig.mask_images, n)
    fringe_images = downsample_images(measurement_orig.fringe_images, n)
    return Measurement(
        mask_images=mask_images,
        fringe_images=fringe_images,
        fringe_periods_x=measurement_orig.fringe_periods_x,
        fringe_periods_y=measurement_orig.fringe_periods_y,
        measure_point=measurement_orig.measure_point,
        optic_screen_dist=measurement_orig.optic_screen_dist,
        date=measurement_orig.date,
        name=measurement_orig.name
    )
