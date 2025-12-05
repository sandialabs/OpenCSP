from os.path import join, dirname

import numpy as np

from opencsp.app.scene_reconstruction.lib.SceneReconstruction import SceneReconstruction
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def scene_reconstruction(dir_input, dir_output):
    """
    Reconstructs the XYZ locations of Aruco markers in a scene.

    Parameters
    ----------
    dir_input : str
        The directory containing the input files needed for scene reconstruction. This includes:

        - 'camera.h5': HDF5 file containing camera parameters.
        - 'known_point_locations.csv': CSV file with known point locations.
        - 'aruco_marker_images/NAME.JPG': Directory containing images of Aruco markers.
        - 'point_pair_distances.csv': CSV file with distances between point pairs.
        - 'alignment_points.csv': CSV file with alignment points.
    dir_output : str
        The directory where the output files, including point locations and calibration figures, will be saved.

    Notes
    -----
    This function performs the following steps:

    1. Loads the camera parameters from an HDF5 file.
    2. Loads known point locations, point pair distances, and alignment points from CSV files.
    3. Initializes the SceneReconstruction object with the camera parameters and known point locations.
    4. Runs the calibration process to determine the marker positions.
    5. Scales the points based on the provided point pair distances.
    6. Aligns the points using the provided alignment points.
    7. Saves the reconstructed point locations to a CSV file.
    8. Saves calibration figures as PNG files in the output directory.

    Examples
    --------
    >>> scene_reconstruction('/path/to/input', '/path/to/output')

    """
    # "ChatGPT 4o" assisted with generating this docstring.

    # Load components
    camera = Camera.load_from_hdf(join(dir_input, 'camera.h5'))
    known_point_locations = np.loadtxt(join(dir_input, 'known_point_locations.csv'), delimiter=',', skiprows=1)
    image_filter_path = join(dir_input, 'aruco_marker_images', '*.JPG')
    point_pair_distances = np.loadtxt(join(dir_input, 'point_pair_distances.csv'), delimiter=',', skiprows=1)
    alignment_points = np.loadtxt(join(dir_input, 'alignment_points.csv'), delimiter=',', skiprows=1)

    # Perform marker position calibration
    cal_scene_recon = SceneReconstruction(camera, known_point_locations, image_filter_path)
    cal_scene_recon.make_figures = True
    cal_scene_recon.run_calibration()

    # Scale points
    point_pairs = point_pair_distances[:, :2].astype(int)
    distances = point_pair_distances[:, 2]
    cal_scene_recon.scale_points(point_pairs, distances)

    # Align points
    marker_ids = alignment_points[:, 0].astype(int)
    alignment_values = Vxyz(alignment_points[:, 1:4].T)
    cal_scene_recon.align_points(marker_ids, alignment_values)

    # Save points as CSV
    cal_scene_recon.save_data_as_csv(join(dir_output, 'point_locations.csv'))

    # Save calibration figures
    for fig in cal_scene_recon.figures:
        figure_path_body = join(dir_output, fig.get_label() + '.png')
        lt.info('before overwrite check, figure_path_body = ' + figure_path_body)
        # Overwrite previous versions.
        if ft.file_exists(figure_path_body):
            ft.delete_file(figure_path_body)
        fig.savefig(figure_path_body)


def example_scene_reconstruction_driver(dir_input_fixture, dir_output_fixture):
    """
    Sets up and runs the scene_reconstruction() routine.

    Parameters
    ----------
    dir_input_fixture : str
        Directory to read input.  Called a fixture because it might be provided by pytest.
    dir_output_fixture : str
        Directory to write output.  Called a fixture because it might be provided by pytest.
    """
    dir_input = join(opencsp_code_dir(), 'app/scene_reconstruction/test/data/data_measurement')
    dir_output = join(dirname(__file__), 'data/output/scene_reconstruction')
    if dir_input_fixture:
        dir_input = dir_input_fixture
    if dir_output_fixture:
        dir_output = dir_output_fixture

    # Ensure output directory is ready
    ft.create_directories_if_necessary(dir_output)

    # Set up logger
    logfile_dir_body_ext = join(dir_output, 'log.txt')
    lt.logger(logfile_dir_body_ext, lt.log.INFO)
    lt.info('Starting program ' + __file__)

    lt.info('dir_input = ' + dir_input)
    lt.info('dir_output = ' + dir_output)
    lt.info('Calling routine scene_reconstruction(dir_input, dir_output)...')
    scene_reconstruction(dir_input, dir_output)


if __name__ == '__main__':
    # ?? RCB SCAFFOLDING RCB -- DELETE FOLLOWING
    import argparse
    import configparser
    import os

    # Start argparse
    # parser = argparse.ArgumentParser(prog=__file__.rstrip(".py"), description="Sensitive strings searcher")
    # Source - https://stackoverflow.com/a
    # Posted by Martijn Pieters, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-04, License - CC BY-SA 4.0
    parser = argparse.ArgumentParser(
        prog=__file__.rstrip(".py"),
        description="Example scene reconstruction calculation.  Given photos with Aruco markers, find marker and camera 3-d positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #     "--verbose",
    #     action="store_true",
    #     dest="verbose",
    #     help="Print more information while running. Overrides '--progress'.",
    # )
    parser.add_argument(
        "-s",
        "--settings_dir_body_ext",
        required=False,
        dest="settings_dir_body_ext",
        default=None,
        help="The directory root for reading data and writing output for this run.",
    )
    args = parser.parse_args()
    settings_dir_body_ext: str = args.settings_dir_body_ext
    print("arg: settings_dir_body_ext = ", settings_dir_body_ext)
    # End argparse

    if settings_dir_body_ext is None:
        dir_main_input = join(opencsp_code_dir(), 'app/scene_reconstruction/test/data/data_measurement')
        dir_main_output = join(dirname(__file__), 'data/output/scene_reconstruction')
        write_full_data = False
    else:
        print("current_working_directory = ", os.getcwd())
        print("settings_dir_body_ext = ", settings_dir_body_ext)
        print("ft.file_exists(settings_dir_body_ext) = ", ft.file_exists(settings_dir_body_ext))
        settings = configparser.ConfigParser()
        settings.read(settings_dir_body_ext)
        dir_main_input = settings["Default"]["dir_input"]
        dir_main_output = settings["Default"]["dir_output"]
        write_full_data = settings["Default"]["write_full_data"]
    # assert False
    # ?? RCB SCAFFOLDING RCB -- END SCAFFOLDING
    print("Before calling driver, dir_main_input  = ", dir_main_input)
    print("Before calling driver, dir_main_output = ", dir_main_output)
    print("Before calling driver, write_full_data = ", write_full_data)
    example_scene_reconstruction_driver(dir_main_input, dir_main_output)
