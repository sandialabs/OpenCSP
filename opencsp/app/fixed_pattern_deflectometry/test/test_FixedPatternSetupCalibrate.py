"""Example script that performs dot location calibration using photogrammetry.

"""
import os
from os.path import join, dirname, exists

import numpy as np

from opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternDotLocations import \
    FixedPatternDotLocations
from opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternSetupCalibrate import \
    FixedPatternSetupCalibrate
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.photogrammetry import photogrammetry as ph


def test_FixedPatternSetupCalibrate():
    """Tests dot-location calibration
    """
    # Define dot location images and origins
    base_dir = join(
        opencsp_code_dir(),
        'test',
        'data',
        'fixed_pattern_deflectometry_measurements',
        'dot_location_calibration',
    )
    files = [
        join(base_dir, 'measurements/images/DSC03965.JPG'),
        join(base_dir, 'measurements/images/DSC03967.JPG'),
        join(base_dir, 'measurements/images/DSC03970.JPG'),
        join(base_dir, 'measurements/images/DSC03972.JPG'),
    ]
    origins = Vxy(([4950, 4610, 4221, 3617], [3359, 3454, 3467, 3553]))

    # Define other files
    file_camera_marker = join(base_dir, 'measurements/camera_calibration.h5')
    file_xyz_points = join(base_dir, 'measurements/point_locations.csv')
    file_fpd_dot_locs_exp = join(base_dir, 'calculations/fixed_pattern_dot_locations.h5')
    dir_save = join(dirname(__file__), 'data/output/dot_location_calibration')

    if not exists(dir_save):
        os.makedirs(dir_save)

    # Load images
    images = []
    for file in files:
        images.append(ph.load_image_grayscale(file))

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
        images, origins, camera_marker, pts_xyz_corners, ids_corners, -32, 31, -31, 32
    )
    cal_dot_locs.run()

    # Save data
    dot_locs = cal_dot_locs.get_dot_location_object()
    dot_locs.save_to_hdf(join(dir_save, 'fixed_pattern_dot_locations.h5'))

    # Save figures
    for fig in cal_dot_locs.figures:
        fig.savefig(join(dir_save, fig.get_label() + '.png'))

    # Test
    np.testing.assert_allclose(dot_locs.xyz_dot_loc, dot_locs_exp.xyz_dot_loc)
    np.testing.assert_allclose(dot_locs.x_dot_index, dot_locs_exp.x_dot_index)
    np.testing.assert_allclose(dot_locs.y_dot_index, dot_locs_exp.y_dot_index)


if __name__ == '__main__':
    test_FixedPatternSetupCalibrate()
