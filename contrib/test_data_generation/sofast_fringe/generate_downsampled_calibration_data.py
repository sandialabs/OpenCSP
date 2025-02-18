"""Downsamples full-resolution SOFAST datasets. Loads full-resolution
datasets from the Sofast sample data suite, downsamples, then saves downsampled
measurement files (and associated equivalent camera definition file) to the Sofast
test data suite.
"""

from os.path import join, basename, exists, abspath
import sys

import matplotlib.pyplot as plt

from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir

sys.path.append(join(opencsp_code_dir(), ".."))
import contrib.test_data_generation.downsample_data_general as ddg  # nopep8
import contrib.test_data_generation.sofast_fringe.downsample_data as dds  # nopep8


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
    file_measurement_facet = abspath(join(base_dir, "sofast/measurement_facet.h5"))
    file_measurement_ensemble = abspath(join(base_dir, "sofast/measurement_facet_ensemble.h5"))
    file_calibration = abspath(join(base_dir, "sofast/image_calibration.h5"))
    file_camera = abspath(join(base_dir, "calibration_files/camera.h5"))
    file_display_1 = abspath(join(base_dir, "calibration_files/display_distorted_2d.h5"))
    file_display_2 = abspath(join(base_dir, "calibration_files/display_distorted_3d.h5"))
    file_display_3 = abspath(join(base_dir, "calibration_files/display_rectangular.h5"))

    dir_dataset_out = abspath(join(opencsp_code_dir(), "test/data/measurements_sofast_fringe"))

    if not exists(dir_dataset_out):
        raise FileNotFoundError(f"Output directory {dir_dataset_out} does not exist.")

    # Load data
    camera = ddg.downsample_camera(file_camera, n_ds)
    measurement_facet = dds.downsample_measurement(file_measurement_facet, n_ds)
    measurement_ensemble = dds.downsample_measurement(file_measurement_ensemble, n_ds)
    display_1 = Display.load_from_hdf(file_display_1)
    display_2 = Display.load_from_hdf(file_display_2)
    display_3 = Display.load_from_hdf(file_display_3)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)

    # Plot data
    plt.figure()
    plt.imshow(measurement_facet.mask_images[..., 1])
    plt.title("Single Facet Mask Image")

    plt.figure()
    plt.imshow(measurement_ensemble.mask_images[..., 1])
    plt.title("Ensemble Mask Image")

    # Save data
    measurement_facet.save_to_hdf(join(dir_dataset_out, basename(file_measurement_facet)))
    measurement_ensemble.save_to_hdf(join(dir_dataset_out, basename(file_measurement_ensemble)))
    camera.save_to_hdf(join(dir_dataset_out, basename(file_camera)))
    display_1.save_to_hdf(join(dir_dataset_out, basename(file_display_1)))
    display_2.save_to_hdf(join(dir_dataset_out, basename(file_display_2)))
    display_3.save_to_hdf(join(dir_dataset_out, basename(file_display_3)))
    calibration.save_to_hdf(join(dir_dataset_out, basename(file_calibration)))

    plt.show()


if __name__ == "__main__":
    # Create downsample dataset 1 (NSTTF Optics Lab data)
    dir_sample_data = join(opencsp_code_dir(), "../../sample_data/sofast/measurement_set_1")
    downsample_dataset_1(dir_sample_data)
