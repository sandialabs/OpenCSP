"""Downsamples full-resolution SOFAST calibration datasets. Loads full-resolution
datasets from the Sofast examples, downsamples, then saves downsampled
files to this test data suite.
"""

import os
from os.path import join
import shutil
import sys

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir

sys.path.append(join(opencsp_code_dir(), ".."))
import contrib.test_data_generation.downsample_data_general as ddg  # nopep8
import contrib.test_data_generation.sofast_fringe.downsample_data as dds  # nopep8


def downsample_dataset(dir_input: str, dir_output: str) -> None:
    """Creates downsampled Sofast calibration dataset.

    Parameters
    ----------
    dir_input : str
        Measurement file containing directory
    dir_output : str
        Output downsampled measuremnt file containing directory
    """
    # Define downsample factors
    n_sofast = 4

    # Copy files that don't need downsampling
    print("Copying files with no downsampling...")
    files = [
        "camera_sofast.h5",
        "screen_calibration_point_pairs.csv",
        "image_projection.h5",
        "image_sofast_camera.png",
        "point_locations.csv",
    ]
    for file in files:
        shutil.copy(join(dir_input, file), join(dir_output, file))

    # Downsample screen distortion measurements
    dir_output_screen_measurements = join(dir_output, "screen_shape_sofast_measurements")
    if not os.path.exists(dir_output_screen_measurements):
        os.makedirs(dir_output_screen_measurements)
    files_meas = [
        join(dir_input, "screen_shape_sofast_measurements/pose_1.h5"),
        join(dir_input, "screen_shape_sofast_measurements/pose_3.h5"),
        join(dir_input, "screen_shape_sofast_measurements/pose_4.h5"),
    ]
    for file_meas in files_meas:
        print(f"Downsampling sofast measurement: {os.path.basename(file_meas):s}...")
        meas_ds = dds.downsample_measurement(file_meas, n_sofast)
        meas_ds.save_to_hdf(join(dir_output_screen_measurements, os.path.basename(file_meas)))

    # Downsample screen distortion camera
    print("Downsampling sofast camera...")
    camera_sofast_ds = ddg.downsample_camera(join(dir_input, "camera_screen_shape.h5"), n_sofast)
    camera_sofast_ds.save_to_hdf(join(dir_output, "camera_screen_shape.h5"))


if __name__ == "__main__":
    downsample_dataset(
        dir_input=join(
            opencsp_code_dir(), "../../sample_data/sofast/data_photogrammetric_calibration/data_measurement"
        ),
        dir_output=join(opencsp_code_dir, "test/data/measurements_sofast_fringe"),
    )
