"""Generates downsampled dataset for dot_location_calibration
"""

from glob import glob
from os.path import join, basename, abspath
import sys

import imageio.v3 as imageio

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.photogrammetry import photogrammetry as ph

sys.path.append(join(opencsp_code_dir(), ".."))
import contrib.test_data_generation.downsample_data_general as ddg  # nopep8


def generate_data():
    """Downsamples and saves files"""
    # Define file locations
    dir_cal_data = join(
        opencsp_code_dir(), "../../sample_data/deflectometry/sandia_lab/dot_locations_calibration/data_measurement"
    )

    files_images = glob(abspath(join(dir_cal_data, "images/*.JPG")))
    file_camera_cal = abspath(join(dir_cal_data, "camera_calibration.h5"))

    dir_save = join(opencsp_code_dir(), "test/data/dot_location_calibration")

    # Downsample marker/dot images
    n_downsample = 4
    for file in files_images:
        print(f"Processing {basename(file):s}...")
        # Load image
        im = ph.load_image_grayscale(file)[..., None]
        # Downsample
        im_ds = ddg.downsample_images(im, n_downsample)
        # Save
        file_save = join(dir_save, "images", basename(file))
        imageio.imwrite(file_save, im_ds, quality=80)

    # Downsample cal camera
    print("Downsampling calibration camera...")
    cam_cal = ddg.downsample_camera(file_camera_cal, n_downsample)
    cam_cal.save_to_hdf(join(dir_save, basename(file_camera_cal)))


if __name__ == "__main__":
    generate_data()
