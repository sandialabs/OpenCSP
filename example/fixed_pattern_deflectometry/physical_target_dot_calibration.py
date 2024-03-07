"""Example script that performs dot location calibration using photogrammetry.
"""
import os
from os.path import join, dirname, exists

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternSetupCalibrate import \
    FixedPatternSetupCalibrate
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.CalibrationCameraPosition import \
    CalibrationCameraPosition
from opencsp.common.lib.deflectometry.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


def example_perform_calibration():
    """Performs a dot-location calibration using photogrammetry"""
    # Define dot location images and origins
    base_dir = join(opencsp_code_dir(),
                    '../../sample_data/deflectometry/calibration_dot_locations/data_measurement/')
    files = [
        join(base_dir, 'images/DSC03965.JPG'),
        join(base_dir, 'images/DSC03967.JPG'),
        join(base_dir, 'images/DSC03970.JPG'),
        join(base_dir, 'images/DSC03972.JPG'),
    ]
    origins = Vxy(([4950, 4610, 4221, 3617], [3359, 3454, 3467, 3553]))

    # Define other files
    file_camera_position = join(base_dir, 'image_deflectometry_camera.png')
    file_camera_marker = join(base_dir, 'camera_calibration.h5')
    file_camera_system = join(base_dir, 'camera_deflectometry.h5')
    file_xyz_points = join(base_dir, 'point_locations.csv')
    dir_save = join(dirname(__file__), 'data/output/dot_location_calibration')

    if not exists(dir_save):
        os.makedirs(dir_save)

    # Load images
    images = []
    for file in files:
        images.append(cv.imread(file, cv.IMREAD_GRAYSCALE))
    image_camera_position = cv.imread(file_camera_position, cv.IMREAD_GRAYSCALE)

    # Load marker corner locations
    data = np.loadtxt(file_xyz_points, delimiter=',')
    pts_xyz_corners = Vxyz(data[:, 2:5].T)
    ids_corners = data[:, 1]

    # Load cameras
    camera_marker = Camera.load_from_hdf(file_camera_marker)
    camera_system = Camera.load_from_hdf(file_camera_system)

    # Perform dot location calibration
    cal_dot_locs = FixedPatternSetupCalibrate(
        images, origins, camera_marker, pts_xyz_corners, ids_corners, -32, 31, -31, 32
    )
    cal_dot_locs.verbose = 2
    cal_dot_locs.run()

    # Perform camera position calibration
    cal_camera = CalibrationCameraPosition(
        camera_system, pts_xyz_corners, ids_corners, image_camera_position
    )
    cal_camera.verbose = 2
    cal_camera.run_calibration()

    # Create spatial orientation object
    rvec_screen_cam, v_cam_screen_screen = cal_camera.get_data()
    r_screen_cam = Rotation.from_rotvec(rvec_screen_cam)
    r_cam_screen = r_screen_cam.inv()
    v_cam_screen_cam = Vxyz(v_cam_screen_screen).rotate(r_screen_cam)
    orientation = SpatialOrientation(r_cam_screen, v_cam_screen_cam)

    # Save data
    dot_locs = cal_dot_locs.get_dot_location_object()
    dot_locs.save_to_hdf(join(dir_save, 'fixed_pattern_dot_locations.h5'))
    orientation.save_to_hdf(join(dir_save, 'spatial_orientation.h5'))

    # Save figures
    for fig in cal_dot_locs.figures:
        fig.savefig(join(dir_save, fig.get_label() + '.png'))


if __name__ == '__main__':
    example_perform_calibration()
