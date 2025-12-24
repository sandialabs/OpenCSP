r"""("r" prefix to ignore escape characters within docstring.)

Exercise scene reconstruction algorithms.

Supports these use cases:

1. Pytest execution.
   Purpose:  To verify code is still operating properly.

   a. Quick pytest unit tests.  Run pytest from opencsp directory.
      In OpenCSP directory:
      pytest

   b. More detailed pytest examples.
      In OpenCSP\example directory:
      pytest

   c. Automated tests for pull request management.

   d. Automated tests for nightly and weekly function checks.
      Run pytest from opencsp/example directory (with arguments to run full-scale data).

   e. Run pytest on only this example.
      In OpenCSP\example directory:
      pytest .\scene_reconstruction\example_scene_reconstruction.py

2. Running the example from the command line.
   Purpose:  To apply the example calculation to new data.

   a. With no arguments, to execute default behavior (for practice and study).
      In OpenCSP\example\scene_reconstruction directory:
      python .\example_scene_reconstruction.py

   b. With an argument to read a user-specified settings file to define input sources, output locations, and execution details.
      i. Data source/sink options:
         (1) Data built into the repository.  No argument.
             In OpenCSP\example\scene_reconstruction directory:
             python .\example_scene_reconstruction.py
         (2) Community data on local machine.  myfile_ctemp.ini
             In OpenCSP\example\scene_reconstruction directory:
             python .\example_scene_reconstruction.py --settings_dir_body_ext "C:\ctemp\OpenCSP_ctemp\example_scene_reconstruction_settings_ctemp.ini"
         (3) Data on local machine, but in user-owned location.  myfile_<user_id>.ini
             In OpenCSP\example\scene_reconstruction directory:
             python .\example_scene_reconstruction.py --settings_dir_body_ext "C:\Users\<user_id>\OpenCSP\OpenCSP_<user_id>\example_scene_reconstruction_settings_<user_id>.ini"
         (4) General network location.  myfile_<network_location>.ini
             In OpenCSP\example\scene_reconstruction directory:
             python .\example_scene_reconstruction.py --settings_dir_body_ext "\\<network_path>\OpenCSP_<net_name>\example_scene_reconstruction_settings_<net_name>.ini"
      ii. Output levels:
         (1) Output only minimal progress updates.  Omit --verbose flag.
         (2) Output full progress and calculation updates.  Add --verbose flag.

   c. Review command-line options:
      In OpenCSP\example\scene_reconstruction directory:
      python .\example_scene_reconstruction.py --help

3. Calling the example calculation from other code.
   Purpose:  To utilize the example calculation within a larger computation or application.

   a. Arbitrary context and general data.
      In calling file, import example_scenario_reconstruction, call driver.

4. Running an example with the Visual Studio Code debugger.
   Purpose:  To interact with the code execution (break points, check stack variables, etc), either for study or fixing an error.  Could apply to either 2 or 3 above.

   a. Use default settings, running on default data.  In VS Code, press F5 key.

   b. Temporarily modify internal variable values to test specific data computation.
      Make a scratch copy somewhere else, edit in VS Code, then press F5 key.

"""

from os.path import join, basename, dirname, splitext

import argparse
import configparser
import numpy as np

from opencsp.app.scene_reconstruction.lib.SceneReconstruction import SceneReconstruction
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def scene_reconstruction(dir_input, dir_output, verbose):
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
    verbose : bool
        If true, write out detailed information.

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
        # Overwrite previous versions.
        if ft.file_exists(figure_path_body):
            ft.delete_file(figure_path_body)
        fig.savefig(figure_path_body)


def example_scene_reconstruction_driver(arg_settings_dir_body_ext: str = None, verbose_param=None):
    """
    Sets up and runs the scene_reconstruction() routine.

    Parameters
    ----------
    arg_settings_dir_body_ext : str
        Full path and filename for settings file, containing inputand output directories, plot control settings, etc.
         Optional.  If not provided, internal defaults are used.
        See code for options sought within file.
    """
    # Get settings
    if arg_settings_dir_body_ext is None:
        print("Using default control settings.")
        dir_input = join(opencsp_code_dir(), 'app/scene_reconstruction/test/data/data_measurement')
        dir_output = join(dirname(__file__), 'data/output/scene_reconstruction')
        if verbose_param is None:
            verbose = False
        else:
            verbose = verbose_param
    else:
        print("Loading control from settings file:", arg_settings_dir_body_ext)
        if not ft.file_exists(arg_settings_dir_body_ext):
            print("ERROR: In " + basename(__file__) + ", settings file does not exist. Settings file:")
            print("   ", arg_settings_dir_body_ext)
            assert False
        settings = configparser.ConfigParser()
        settings.read(arg_settings_dir_body_ext)
        dir_input = settings["Default"]["dir_input"]
        dir_output = settings["Default"]["dir_output"]
        verbose_setting = settings["Default"]["verbose"]
        if verbose_param is None:
            verbose = verbose_setting
        else:
            verbose = verbose_param

    # Ensure output directory is ready
    ft.create_directories_if_necessary(dir_output)

    # Set up logger
    logfile_dir_body_ext = join(dir_output, splitext(basename(__file__))[0] + '_log.txt')
    print("logfile_dir_body_ext = ", logfile_dir_body_ext)
    lt.logger(logfile_dir_body_ext, lt.log.INFO)
    # Output standard lines.
    if verbose:
        lt.info_strings_from_file(join(dirname(__file__), splitext(basename(__file__))[0] + '_README.md'))
    lt.info('Starting program ' + __file__)
    lt.info('dir_input = ' + dir_input)
    lt.info('dir_output = ' + dir_output)
    lt.info('verbose = ' + str(verbose))
    if verbose:
        lt.info('Calling routine scene_reconstruction(dir_input, dir_output, verbose)...')
    scene_reconstruction(dir_input, dir_output, verbose)


if __name__ == '__main__':
    # Parse command-line arguments, if any.
    # Execute "python <this_file>.py --help" to see usage tips.
    #
    # Source - https://stackoverflow.com/a
    # Posted by Martijn Pieters, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-04, License - CC BY-SA 4.0
    parser = argparse.ArgumentParser(
        prog=__file__.rstrip(".py"),
        description='Example scene reconstruction calculation.  Given photos with Aruco markers, find marker and camera 3-d positions.  See "example_scene_reconstruction_README.md" for details.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--settings_dir_body_ext",
        required=False,
        dest="settings_dir_body_ext",
        default=None,
        help="Settings file defining run parameters (input/output directories, etc).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Output detailed information reporting run progress and calculations.",
    )
    args = parser.parse_args()
    arg_settings_dir_body_ext: str = args.settings_dir_body_ext
    verbose: bool = args.verbose

    # Manual override for use when debugging.  Comment this line for normal runs.
    # verbose: bool = True

    # Call driver.
    example_scene_reconstruction_driver(arg_settings_dir_body_ext, verbose)
