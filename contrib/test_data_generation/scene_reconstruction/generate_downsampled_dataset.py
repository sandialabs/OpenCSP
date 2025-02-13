"""Generates downsampled dataset used for calibrating the 3d locations of fixed
pattern dots.
"""

from glob import glob
from os.path import join, basename, exists
from os import mkdir
import shutil
import sys

import imageio.v3 as imageio

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.photogrammetry import photogrammetry as ph

sys.path.append(join(opencsp_code_dir(), ".."))
import contrib.test_data_generation.downsample_data_general as ddg  # nopep8


def generate_data():
    """Downsamples and saves files"""
    # Define file locations
    dir_sample_data = join(opencsp_code_dir(), "../../sample_data/scene_reconstruction/data_measurement")

    files_images = glob(join(dir_sample_data, "aruco_marker_images/*.JPG"))
    file_alignment_points = join(dir_sample_data, "alignment_points.csv")
    file_point_locs = join(dir_sample_data, "known_point_locations.csv")
    file_point_pair_dists = join(dir_sample_data, "point_pair_distances.csv")
    file_camera_cal = join(dir_sample_data, "camera.h5")

    dir_save = join(opencsp_code_dir(), "app/scene_reconstruction/test/data/data_measurement")

    # Downsample marker/dot images
    n_downsample = 5
    dir_save_images = join(dir_save, "aruco_marker_images")
    if not exists(dir_save_images):
        mkdir(dir_save_images)
    for file in files_images:
        print(f"Processing {basename(file):s}...")
        # Load image
        im = ph.load_image_grayscale(file)[..., None]
        # Downsample
        im_ds = ddg.downsample_images(im, n_downsample)
        # Save
        file_save = join(dir_save_images, basename(file))
        imageio.imwrite(file_save, im_ds, quality=75)

    # Downsample cal camera
    print("Downsampling calibration camera...")
    cam_cal = ddg.downsample_camera(file_camera_cal, n_downsample)
    cam_cal.save_to_hdf(join(dir_save, basename(file_camera_cal)))

    # Save other files
    shutil.copy(file_point_locs, join(dir_save, basename(file_point_locs)))
    shutil.copy(file_alignment_points, join(dir_save, basename(file_alignment_points)))
    shutil.copy(file_point_pair_dists, join(dir_save, basename(file_point_pair_dists)))


if __name__ == "__main__":
    generate_data()
    print("Done")
