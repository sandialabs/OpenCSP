"""Example script that performs dot location calibration using photogrammetry.
"""
import os
from os.path import join, dirname, exists

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.CalibrateSofastFixedDots import CalibrateSofastFixedDots
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.CalibrationCameraPosition import \
    CalibrationCameraPosition
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.photogrammetry.photogrammetry as ph
import opencsp.common.lib.tool.log_tools as lt


def example_perform_calibration():
    """Performs a dot-location calibration using photogrammetry"""
    # Define dot location images and origins
    base_dir = join(opencsp_code_dir(),
                    'test/data/fixed_pattern_deflectometry/dot_location_calibration/measurements')
    files_cal_images = [
        join(base_dir, 'images/DSC03965.JPG'),
        join(base_dir, 'images/DSC03967.JPG'),
        join(base_dir, 'images/DSC03970.JPG'),
        join(base_dir, 'images/DSC03972.JPG'),
    ]
    origins = np.array(([4950, 4610, 4221, 3617], [3359, 3454, 3467, 3553]), dtype=float) / 4
    origins = Vxy(origins.astype(int))

    # Define other files
    file_camera_position = join(base_dir, 'image_deflectometry_camera.png')
    file_camera_marker = join(base_dir, 'camera_calibration.h5')
    file_camera_system = join(base_dir, 'camera_deflectometry.h5')
    file_xyz_points = join(base_dir, 'point_locations.csv')
    dir_save = join(dirname(__file__), 'data/output/dot_location_calibration')

    if not exists(dir_save):
        os.makedirs(dir_save)

    # Set up logger
    lt.logger(log_dir_body_ext=join(dir_save, 'log.txt'), level=lt.log.INFO)

    # Load images
    image_camera_position = ph.load_image_grayscale(file_camera_position)

    # Load marker corner locations
    data = np.loadtxt(file_xyz_points, delimiter=',')
    pts_xyz_corners = Vxyz(data[:, 2:5].T)
    ids_corners = data[:, 1]

    # Load cameras
    camera_marker = Camera.load_from_hdf(file_camera_marker)
    camera_system = Camera.load_from_hdf(file_camera_system)

    # Perform dot location calibration
    cal_dot_locs = CalibrateSofastFixedDots(
        files_cal_images, origins, camera_marker, pts_xyz_corners, ids_corners, -32, 31, -31, 32
    )
    cal_dot_locs.plot = True
    cal_dot_locs.blob_search_threshold = 3.
    cal_dot_locs.blob_detector.minArea = 3.
    cal_dot_locs.blob_detector.maxArea = 30.
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
    cal_dot_locs.save_figures(dir_save)


if __name__ == '__main__':
    example_perform_calibration()
