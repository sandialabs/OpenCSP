"""Downsamples full-resolution SOFAST datasets. Loads full-resolution
datasets from the Sofast sample data suite, downsamples, then saves downsampled
measurement files (and associated equivalent camera definition file) to the Sofast
test data suite.
"""
import os

import matplotlib.pyplot as plt

import opencsp
from   opencsp.common.lib.deflectometry.Display import Display
from   opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
import opencsp.app.sofast.test.downsample_sofast_data as ds
import opencsp.common.lib.test.downsample_data as dd


def downsample_dataset_1(base_dir):
    """Saves downsampled measurement and camera files. Also saves copies
    of display, camera, and calibration files.

    Parameters
    ----------
    base_dir : str
        Location of sample data
    """
    # Define downsample factor
    n_ds = 8

    # Define location of sample data
    file_measurement_facet = os.path.join(base_dir, 'measurement_facet.h5')
    file_measurement_ensemble = os.path.join(base_dir, 'measurement_ensemble.h5')
    file_camera = os.path.join(base_dir, 'camera.h5')
    file_display_1 = os.path.join(base_dir, 'display_distorted_2d.h5')
    file_display_2 = os.path.join(base_dir, 'display_distorted_3d.h5')
    file_display_3 = os.path.join(base_dir, 'display_rectangular.h5')
    file_calibration = os.path.join(base_dir, 'calibration.h5')

    dir_dataset_out = os.path.join(os.path.dirname(__file__), 'data')

    if not os.path.exists(dir_dataset_out):
        os.mkdir(dir_dataset_out)

    # Load data
    camera = dd.downsample_camera(file_camera, n_ds)
    measurement_facet = ds.downsample_measurement(file_measurement_facet, n_ds)
    measurement_ensemble = ds.downsample_measurement(file_measurement_ensemble, n_ds)
    display_1 = Display.load_from_hdf(file_display_1)
    display_2 = Display.load_from_hdf(file_display_2)
    display_3 = Display.load_from_hdf(file_display_3)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)

    # Plot data
    plt.figure()
    plt.imshow(measurement_facet.mask_images[..., 1])
    plt.title('Single Facet Mask Image')

    plt.figure()
    plt.imshow(measurement_ensemble.mask_images[..., 1])
    plt.title('Ensemble Mask Image')

    # Save data
    measurement_facet.save_to_hdf(os.path.join(dir_dataset_out, os.path.basename(file_measurement_facet)))
    measurement_ensemble.save_to_hdf(os.path.join(dir_dataset_out, os.path.basename(file_measurement_ensemble)))
    camera.save_to_hdf(os.path.join(dir_dataset_out, os.path.basename(file_camera)))
    display_1.save_to_hdf(os.path.join(dir_dataset_out, os.path.basename(file_display_1)))
    display_2.save_to_hdf(os.path.join(dir_dataset_out, os.path.basename(file_display_2)))
    display_3.save_to_hdf(os.path.join(dir_dataset_out, os.path.basename(file_display_3)))
    calibration.save_to_hdf(os.path.join(dir_dataset_out, os.path.basename(file_calibration)))

    plt.show()


if __name__ == '__main__':
    dir_sample_data = os.path.join(os.path.dirname(opencsp.__file__), '../../sample_data/sofast/measurement_set_1')
    downsample_dataset_1(dir_sample_data)
