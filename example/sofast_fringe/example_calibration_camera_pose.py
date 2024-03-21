from os.path import join, dirname

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.deflectometry.CalibrationCameraPosition import CalibrationCameraPosition
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.photogrammetry.photogrammetry import load_image_grayscale
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_run_camera_position_calibration(save_dir: str):
    """Calibrates the relative position of the Sofast camera and display. 
    Saves the rvec/tvec in a SpatialOrientation file at ./data/output/spatial_orientation.h5
    """
    # Define directory where screen shape calibration data is saved
    base_dir_sofast_cal = join(opencsp_code_dir(), 'common/lib/deflectometry/test/data/data_measurement')

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
    cal.make_figures = True
    cal.run_calibration()

    for fig in cal.figures:
        file = join(save_dir, fig.get_label() + '.png')
        lt.info(f'Saving figure to: {file:s}')
        fig.savefig(file)

    # Get orientation
    r_screen_cam, v_cam_screen_screen = cal.get_data()
    r_screen_cam = Rotation.from_rotvec(r_screen_cam)
    v_cam_screen_screen = Vxyz(v_cam_screen_screen)

    r_cam_screen = r_screen_cam.inv()
    v_cam_screen_cam = v_cam_screen_screen.rotate(r_screen_cam)

    # Create spatial orientation object
    orientation = SpatialOrientation(r_cam_screen, v_cam_screen_cam)

    # Save data
    orientation.save_to_hdf(join(save_dir, 'spatial_orientation.h5'))


if __name__ == '__main__':
    # Define save dir
    save_path = join(dirname(__file__), 'data/output/camera_pose')
    ft.create_directories_if_necessary(save_path)

    # Set up logger
    lt.logger(join(save_path, 'log.txt'))

    example_run_camera_position_calibration(save_path)
