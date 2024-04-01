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


def example_camera_position_calibration():
    """Example Sofast calibration script

    Calibrates the position of the Sofast camera:
    1. Load measured calibration data
    2. Perform camera position calibration
    3. Save orientation as SpatialOrientation object
    4. Save calculation figures
    """
    # General setup
    # =============

    # Define save dir
    dir_save = join(dirname(__file__), 'data/output/camera_pose')
    ft.create_directories_if_necessary(dir_save)

    # Set up logger
    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # Define inputs
    file_camera_sofast = join(opencsp_code_dir(), 'test/data/sofast_common/camera_sofast.h5')
    file_cal_image = join(opencsp_code_dir(),
                          'test/data/camera_position_calibration/data_measurement/image_sofast_camera.png')
    file_pts_data = join(opencsp_code_dir(), 'test/data/sofast_common/aruco_corner_locations.csv')

    # 1. Load measured calibration data
    # =================================

    # Load input data
    camera = Camera.load_from_hdf(file_camera_sofast)
    image = load_image_grayscale(file_cal_image)

    # Load output data from Scene Reconstruction (Aruco marker xyz points)
    pts_marker_data = np.loadtxt(file_pts_data, delimiter=',', skiprows=1)
    pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
    corner_ids = pts_marker_data[:, 1]

    # 2. Perform camera position calibration
    # ======================================
    cal = CalibrationCameraPosition(camera, pts_xyz_marker, corner_ids, image)
    cal.make_figures = True
    cal.run_calibration()

    # 3. Save orientation as SpatialOrientation object
    # ================================================

    # Get orientation
    r_screen_cam, v_cam_screen_screen = cal.get_data()
    r_screen_cam = Rotation.from_rotvec(r_screen_cam)
    v_cam_screen_screen = Vxyz(v_cam_screen_screen)

    r_cam_screen = r_screen_cam.inv()
    v_cam_screen_cam = v_cam_screen_screen.rotate(r_screen_cam)

    # Create spatial orientation object
    orientation = SpatialOrientation(r_cam_screen, v_cam_screen_cam)

    # Save data
    orientation.save_to_hdf(join(dir_save, 'spatial_orientation.h5'))

    # 4. Save calculation figures
    # ===========================
    for fig in cal.figures:
        file = join(dir_save, fig.get_label() + '.png')
        lt.info(f'Saving figure to: {file:s}')
        fig.savefig(file)


if __name__ == '__main__':
    example_camera_position_calibration()
