import os
from os.path import join, dirname, exists

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.CalibrateSofastFixedDots import CalibrateSofastFixedDots
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.CalibrationCameraPosition import CalibrationCameraPosition
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.photogrammetry.photogrammetry as ph
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_calibrate_sofast_fixed_dot_locations():
    """Performs a printed fixed pattern dot location calibration
    using photogrammetry
    1. Load 
    """
    # General Setup
    # =============
    dir_save = join(dirname(__file__), 'data/output/calibrate_sofast_fixed_dot_locations')
    ft.create_directories_if_necessary(dir_save)

    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # Define dot location images
    dir_meas = join(opencsp_code_dir(), 'test/data/dot_location_calibration/data_measurement')
    dir_sofast = join(opencsp_code_dir(), 'test/data/sofast_common')

    files_cal_images = [
        join(dir_meas, 'images/DSC03965.JPG'),
        join(dir_meas, 'images/DSC03967.JPG'),
        join(dir_meas, 'images/DSC03970.JPG'),
        join(dir_meas, 'images/DSC03972.JPG'),
    ]

    # Define origin dot (here, using an LED)

    # Define blob detection params
    params = cv.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 2
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = 30
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = False
    params.filterByInertia = False

    origins = []
    for file in files_cal_images:
        # Load image
        image = ph.load_image_grayscale(file)
        # Detect origin
        origin = ip.detect_blobs_inverse(image, params)
        if len(origin) != 1:
            lt.error_and_raise(ValueError, f'Expected 1 blob but found {len(origin):d}')
        origins.append(origin)

        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.scatter(*origin.data)
        plt.show()

    origins = Vxy.merge(origins)

    # Define other files
    file_camera_marker = join(dir_meas, 'camera_image_calibration.h5')
    file_camera_system = join(dir_meas, 'camera_deflectometry.h5')
    file_xyz_points = join(dir_meas, 'point_locations.csv')
    dir_save = join(dirname(__file__), 'data/output/dot_location_calibration')

    if not exists(dir_save):
        os.makedirs(dir_save)

    # Set up logger
    lt.logger(log_dir_body_ext=join(dir_save, 'log.txt'), level=lt.log.INFO)

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
    cal_dot_locs.blob_search_threshold = 3.0
    cal_dot_locs.blob_detector.minArea = 3.0
    cal_dot_locs.blob_detector.maxArea = 30.0
    cal_dot_locs.run()

    # Perform camera position calibration
    cal_camera = CalibrationCameraPosition(camera_system, pts_xyz_corners, ids_corners, image_camera_position)
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
    example_calibrate_sofast_fixed_dot_locations()
