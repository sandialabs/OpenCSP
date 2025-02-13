"""Downsamples a full-resolution scene reconstruction dataset. Loads full-resolution
datasets from the sample data directory, downsamples, then saves downsampled
files to this test data suite.

To update the expected data, run the corresponding example file.
"""

from glob import glob
import os
from os.path import join
import shutil

import imageio.v3 as imageio
from tqdm import tqdm

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.test.downsample_data as dd


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
    n_aruco = 5
    jpg_quality = 75

    # Copy files that don't need downsampling
    print("Copying files with no downsampling...")
    files = ["point_pair_distances.csv", "known_point_locations.csv", "alignment_points.csv"]
    for file in files:
        shutil.copy(join(dir_input, file), join(dir_output, file))

    # Downsample aruco marker images
    print("Downsampling aruco images...")
    dir_output_aruco_images = join(dir_output, "aruco_marker_images")
    if not os.path.exists(dir_output_aruco_images):
        os.makedirs(dir_output_aruco_images)
    files = glob(join(dir_input, "aruco_marker_images/*.JPG"))
    for file in tqdm(files):
        file_name = os.path.basename(file)
        im = imageio.imread(file)
        # Convert to monochrome and downsample
        im_ds = dd.downsample_images(im.astype(float).mean(2), n_aruco)
        # Save image
        imageio.imwrite(join(dir_output_aruco_images, file_name), im_ds, quality=jpg_quality)

    # Downsample aruco marker camera
    print("Downsampling camera...")
    camera_aruco_ds = dd.downsample_camera(join(dir_input, "camera.h5"), n_aruco)
    camera_aruco_ds.save_to_hdf(join(dir_output, "camera.h5"))


if __name__ == "__main__":
    downsample_dataset(
        dir_input=join(opencsp_code_dir(), "../../sample_data/scene_reconstruction/data_measurement"),
        dir_output=join(os.path.dirname(__file__), "data/data_measurement"),
    )
