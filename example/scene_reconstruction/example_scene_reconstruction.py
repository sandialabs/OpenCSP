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

   f. Using pytest as the vehicle for full-scale example execution.
      In OpenCSP\example directory:
      pytest .\scene_reconstruction\example_scene_reconstruction.py --dir_input=C:\ctemp\OpenCSP_ctemp\example_data_large\scene_reconstruction\data_measurement --dir_output=C:\ctemp\OpenCSP_ctemp\example_data_large\scene_reconstruction\output  --write_full_data=True

   g. Using pytest as the vehicle for example execution on user data.
      In OpenCSP\example directory:
      pytest .\scene_reconstruction\example_scene_reconstruction.py --dir_input=<user_input_dir> --dir_output=<user_output_dir> --write_full_data=<user_choice_or_omit_argument>

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
         (1) Output only newly computed information.  In .ini: write_full_data = False
         (2) Output full beginning-to-end data corpus in a linear set of directories.
             In .ini: write_full_data = True

   c. Review command-line options:
      In OpenCSP\example\scene_reconstruction directory:
      python .\example_scene_reconstruction.py --help

3. Calling the example calculation from other code.
   Purpose:  To utilize the example calculation within a larger computation or application.

   a. Arbitrary context and general data.
      In calling file, import example_scenario_reconstruction, call driver.  (??other name??)

4. Running an example with the Visual Studio Code debugger.
   Purpose:  To interact with the code execution (break points, check stack variables, etc), either for study or fixing an error.  Could apply to either 2 or 3 above.

   a. Use default settings, running on default data.  In VS Code, press F5 key.

   b. Temporarily modify internal variable values to test specific data computation.
      Edit in VS Code, then press F5 key.

"""

from os.path import join, basename, dirname

import argparse
import configparser
import numpy as np

from opencsp.app.scene_reconstruction.lib.SceneReconstruction import SceneReconstruction
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def scene_reconstruction(dir_input, dir_output, write_full_data):
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
    write_full_data : bool
        If true, write out a directory structure including all input data.  Otherwise only write generated output.

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


def example_scene_reconstruction_driver(dir_input_fixture, dir_output_fixture, write_full_data_fixture):
    """
    Sets up and runs the scene_reconstruction() routine.

    Parameters
    ----------
    dir_input_fixture : str
        Directory to read input.  Has fixture suffix because it might be provided by pytest.
    dir_output_fixture : str
        Directory to write output.  Has fixture suffix because it might be provided by pytest.
    fixture_dir_write_full_data : bool
        If true, write out a directory structure including all input data.  Otherwise only write generated output.
        Has fixture suffix because it might be provided by pytest.
    """
    dir_input = join(opencsp_code_dir(), 'app/scene_reconstruction/test/data/data_measurement')
    dir_output = join(dirname(__file__), 'data/output/scene_reconstruction')
    write_full_data = False
    if dir_input_fixture:
        dir_input = dir_input_fixture
    if dir_output_fixture:
        dir_output = dir_output_fixture
    if write_full_data_fixture:
        write_full_data = write_full_data_fixture

    # Ensure output directory is ready
    ft.create_directories_if_necessary(dir_output)

    # Set up logger
    logfile_dir_body_ext = join(dir_output, 'log.txt')
    lt.logger(logfile_dir_body_ext, lt.log.INFO)
    lt.info('Starting program ' + __file__)

    lt.info('dir_input = ' + dir_input)
    lt.info('dir_output = ' + dir_output)
    lt.info('write_full_data = ' + str(write_full_data))
    lt.info('Calling routine scene_reconstruction(dir_input, dir_output, write_full_data)...')
    scene_reconstruction(dir_input, dir_output, write_full_data)


if __name__ == '__main__':
    # Parse command-line arguments, if any.
    # Execute "python <this_file>.py --help" to see usage tips.
    #
    # Source - https://stackoverflow.com/a
    # Posted by Martijn Pieters, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-04, License - CC BY-SA 4.0
    parser = argparse.ArgumentParser(
        prog=__file__.rstrip(".py"),
        description="Example scene reconstruction calculation.  Given photos with Aruco markers, find marker and camera 3-d positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--settings_dir_body_ext",
        required=False,
        dest="settings_dir_body_ext",
        default=None,
        help="The directory root for reading data and writing output for this run.",
    )
    args = parser.parse_args()
    arg_settings_dir_body_ext: str = args.settings_dir_body_ext
    print("arg: settings_dir_body_ext = ", arg_settings_dir_body_ext)

    # Get settings
    if arg_settings_dir_body_ext is None:
        print("Using default control settings.")
        dir_input_main = join(opencsp_code_dir(), 'app/scene_reconstruction/test/data/data_measurement')
        dir_output_main = join(dirname(__file__), 'data/output/scene_reconstruction')
        write_full_data_main = False
    else:
        print("Loading control from settings file:", arg_settings_dir_body_ext)
        if not ft.file_exists(arg_settings_dir_body_ext):
            print("ERROR: In " + basename(__file__) + ", settings file does not exist. Settings file:")
            print("   ", arg_settings_dir_body_ext)
            assert False
        settings = configparser.ConfigParser()
        settings.read(arg_settings_dir_body_ext)
        dir_input_main = settings["Default"]["dir_input"]
        dir_output_main = settings["Default"]["dir_output"]
        write_full_data_main = settings["Default"]["write_full_data"]

    # Call driver, noting status first.
    print("Calling driver:")
    print("    dir_main_input  = ", dir_input_main)
    print("    dir_main_output = ", dir_output_main)
    print("    write_full_data = ", write_full_data_main)
    example_scene_reconstruction_driver(dir_input_main, dir_output_main, write_full_data_main)
