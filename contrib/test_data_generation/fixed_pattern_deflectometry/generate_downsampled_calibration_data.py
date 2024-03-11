"""Generates downsampled dataset used for calibrating the 3d locations of fixed
pattern dots.
"""
from glob import glob
from os.path import join, basename
import shutil
# import sys

import imageio.v3 as imageio

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.photogrammetry import photogrammetry as ph

# sys.path.append(join(opencsp_code_dir(), '..'))
# import contrib.test_data_generation.downsample_data_general as ddg  # nopep8


def generate_data():
    """Downsamples and saves files
    """
    # Define file locations
    dir_sample_data = join(
        opencsp_code_dir(), '../../sample_data/deflectometry/calibration_dot_locations/data_measurement')

    files_images = glob(join(dir_sample_data, 'images/*.JPG'))
    file_camera_cal = join(dir_sample_data, 'camera_calibration.h5')
    file_point_locs = join(dir_sample_data, 'point_locations.csv')
    file_camera_def = join(dir_sample_data, 'camera_deflectometry.h5')
    file_image_def = join(dir_sample_data, 'image_deflectometry_camera.png')
    file_points = join(dir_sample_data, 'point_locations.csv')

    dir_save = join(opencsp_code_dir(),
                    'test/data/fixed_pattern_deflectometry/dot_location_calibration/measurements')

    # Downsample marker/dot images
    # n_downsample = 5
    for file in files_images:
        print(f'Processing {basename(file):s}...')
        # Load image
        # im = ph.load_image_grayscale(file)[..., None]
        im = ph.load_image_grayscale(file)
        # Downsample
        # im_ds = ddg.downsample_images(im, n_downsample)
        # Save
        file_save = join(dir_save, 'images', basename(file))
        # imageio.imwrite(file_save, im_ds, quality=80)
        imageio.imwrite(file_save, im, quality=80)

    # Downsample cal camera
    print('Downsampling calibration camera...')
    # cam_cal = ddg.downsample_camera(file_camera_cal, n_downsample)
    # cam_cal.save_to_hdf(join(dir_save, basename(file_camera_cal)))

    # Save other files
    shutil.copy(file_camera_cal, join(dir_save, basename(file_camera_cal)))
    shutil.copy(file_point_locs, join(dir_save, basename(file_point_locs)))
    shutil.copy(file_camera_def, join(dir_save, basename(file_camera_def)))
    shutil.copy(file_image_def, join(dir_save, basename(file_image_def)))
    shutil.copy(file_points, join(dir_save, basename(file_points)))


if __name__ == '__main__':
    generate_data()
    print('Done')
