from os.path import join, dirname

import numpy as np

from opencsp.common.lib.deflectometry.CalibrationCameraPosition import CalibrationCameraPosition

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.photogrammetry.photogrammetry import load_image_grayscale
import opencsp.common.lib.tool.file_tools as ft


def example_run_camera_position_calibration():
    """Calibrates the position of the Sofast camera. Saves the rvec/tvec that
    define the relative pose of the camera/screen to a CSV file located
    at ./data/output/camera_rvec_tvec.csv
    """
    # Define save dir
    save_dir = join(dirname(__file__), 'data/output/camera_pose')
    ft.create_directories_if_necessary(save_dir)

    # Define directory where screen shape calibration data is saved
    base_dir_sofast_cal = join(
        opencsp_code_dir(),
        'common/lib/deflectometry/test/data/data_measurement',
    )

    # Define inputs
    file_camera_sofast = join(base_dir_sofast_cal, 'camera_sofast.h5')
    file_cal_image = join(base_dir_sofast_cal, 'image_sofast_camera.png')
    file_pts_data = join(base_dir_sofast_cal, 'point_locations.csv')

    # Load input data
    camera = Camera.load_from_hdf(file_camera_sofast)
    image = load_image_grayscale(file_cal_image)

    # Load output data from Scene Reconstruction (Aruco marker xyz points)
    pts_marker_data = np.loadtxt(file_pts_data, delimiter=',', skiprows=1)
    pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
    corner_ids = pts_marker_data[:, 1]

    # Perform camera position calibraiton
    cal = CalibrationCameraPosition(camera, pts_xyz_marker, corner_ids, image)
    cal.verbose = 2
    cal.run_calibration()

    for fig in cal.figures:
        fig.savefig(join(save_dir, fig.get_label() + '.png'))

    # Save data
    cal.save_data_as_csv(join(save_dir, 'camera_rvec_tvec.csv'))


if __name__ == '__main__':
    example_run_camera_position_calibration()
