"""Example script that performs dot location calibration using photogrammetry.

To create new test data, copy the results from the output folder into the
"calculations" folder.
"""
import os
from os.path import join, dirname, exists

import matplotlib
import numpy as np
import pytest

from opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternDotLocations import \
    FixedPatternDotLocations
from opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternSetupCalibrate import \
    FixedPatternSetupCalibrate
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.log_tools as lt


@pytest.mark.no_xvfb
def test_FixedPatternSetupCalibrate():
    """Tests dot-location calibration
    """
    # Define dot location images and origins
    base_dir = join(
        opencsp_code_dir(),
        'test',
        'data',
        'fixed_pattern_deflectometry',
        'dot_location_calibration',
    )
    files = [
        join(base_dir, 'measurements/images/DSC03965.JPG'),
        join(base_dir, 'measurements/images/DSC03967.JPG'),
        join(base_dir, 'measurements/images/DSC03970.JPG'),
        join(base_dir, 'measurements/images/DSC03972.JPG'),
    ]
    origins = np.array(([4950, 4610, 4221, 3617], [3359, 3454, 3467, 3553]), dtype=float) / 4
    origins = Vxy(origins.astype(int))

    # Define other files
    file_camera_marker = join(base_dir, 'measurements/camera_calibration.h5')
    file_xyz_points = join(base_dir, 'measurements/point_locations.csv')
    file_fpd_dot_locs_exp = join(base_dir, 'calculations/fixed_pattern_dot_locations.h5')
    dir_save = join(dirname(__file__), 'data/output/dot_location_calibration')

    if not exists(dir_save):
        os.makedirs(dir_save)

    # Set up logger
    lt.logger(log_dir_body_ext=join(dir_save, 'log.txt'), level=lt.log.INFO)

    # Load marker corner locations
    data = np.loadtxt(file_xyz_points, delimiter=',')
    pts_xyz_corners = Vxyz(data[:, 2:5].T)
    ids_corners = data[:, 1]

    # Load expedted FPD dot locations
    dot_locs_exp = FixedPatternDotLocations.load_from_hdf(file_fpd_dot_locs_exp)

    # Load cameras
    camera_marker = Camera.load_from_hdf(file_camera_marker)

    # Perform dot location calibration
    cal_dot_locs = FixedPatternSetupCalibrate(
        files, origins, camera_marker, pts_xyz_corners, ids_corners, -32, 31, -31, 32
    )
    cal_dot_locs.plot = True
    cal_dot_locs.blob_search_threshold = 3.
    cal_dot_locs.blob_detector.minArea = 3.
    cal_dot_locs.blob_detector.maxArea = 30.
    cal_dot_locs.run()

    # Save data
    dot_locs = cal_dot_locs.get_dot_location_object()
    dot_locs.save_to_hdf(join(dir_save, 'fixed_pattern_dot_locations.h5'))
    cal_dot_locs.save_figures(dir_save)

    # Test
    np.testing.assert_allclose(dot_locs.xyz_dot_loc, dot_locs_exp.xyz_dot_loc)
    np.testing.assert_allclose(dot_locs.x_dot_index, dot_locs_exp.x_dot_index)
    np.testing.assert_allclose(dot_locs.y_dot_index, dot_locs_exp.y_dot_index)


if __name__ == '__main__':
    test_FixedPatternSetupCalibrate()
