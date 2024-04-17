from os.path import join, dirname

import cv2 as cv
import numpy as np

from opencsp.app.sofast.lib.CalibrateSofastFixedDots import CalibrateSofastFixedDots
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.photogrammetry.photogrammetry as ph
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_calibrate_sofast_fixed_dot_locations():
    """Performs a printed fixed pattern dot location calibration
    using photogrammetry

    1. Load measured calibration data
    2. Find origin dots in calibration images
    3. Perform dot location calibration
    4. Save dot locations as DotLocationsFixedPattern file
    5. Save calculation figures
    """
    # General Setup
    # =============
    dir_save = join(dirname(__file__), 'data/output/calibrate_sofast_fixed_dot_locations')
    dir_save_figures = join(dir_save, 'figures')
    ft.create_directories_if_necessary(dir_save_figures)

    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # Define file locations
    dir_meas = join(opencsp_code_dir(), 'test/data/dot_location_calibration/data_measurement')

    file_camera = join(dir_meas, 'camera_calibration.h5')
    file_xyz_points = join(dir_meas, 'aruco_corner_locations.csv')

    files_cal_images = [
        join(dir_meas, 'images/DSC03992.JPG'),
        join(dir_meas, 'images/DSC03993.JPG'),
        join(dir_meas, 'images/DSC03994.JPG'),
        join(dir_meas, 'images/DSC03995.JPG'),
        join(dir_meas, 'images/DSC03996.JPG'),
    ]

    # 1. Load measured calibration data
    # =================================

    # Load marker corner locations
    data = np.loadtxt(file_xyz_points, delimiter=',')
    pts_xyz_corners = Vxyz(data[:, 2:5].T)
    ids_corners = data[:, 1]

    # Load camera
    camera = Camera.load_from_hdf(file_camera)

    # 2. Find origin dots in calibration images
    # =========================================

    # Define blob detection params
    params = cv.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 2
    params.minThreshold = 20
    params.maxThreshold = 50
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 2
    params.maxArea = 30
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = False
    params.filterByInertia = False

    # Define origin dot location (here, using an LED)
    origins = []
    for file in files_cal_images:
        # Load image
        image = ph.load_image_grayscale(file)
        # Detect origin
        origin = ip.detect_blobs_inverse(image, params)
        if len(origin) != 1:
            lt.error_and_raise(ValueError, f'Expected 1 blob but found {len(origin):d}')
        origins.append(origin)
    origins = Vxy.merge(origins)

    # 3. Perform dot location calibration
    # ===================================
    cal_dot_locs = CalibrateSofastFixedDots(
        files_cal_images, origins, camera, pts_xyz_corners, ids_corners, -32, 31, -31, 32
    )
    cal_dot_locs.plot = True
    cal_dot_locs.blob_search_threshold = 3.0
    cal_dot_locs.blob_detector.minArea = 3.0
    cal_dot_locs.blob_detector.maxArea = 30.0
    cal_dot_locs.run()

    # 4. Save dot locations as DotLocationsFixedPattern file
    # ======================================================
    dot_locs = cal_dot_locs.get_dot_location_object()
    dot_locs.save_to_hdf(join(dir_save, 'fixed_pattern_dot_locations.h5'))

    # 5. Save calculation figures
    # ===========================
    cal_dot_locs.save_figures(dir_save_figures)


if __name__ == '__main__':
    example_calibrate_sofast_fixed_dot_locations()
