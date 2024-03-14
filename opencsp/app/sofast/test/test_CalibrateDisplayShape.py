"""Tests Sofast screen distortion calibration
"""
from os.path import join
import unittest

from glob import glob
import numpy as np
import pytest

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.app.sofast.lib.CalibrateDisplayShape import (
    CalibrateDisplayShape,
    DataInput,
)
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


class TestCalibrateDisplayShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Tests the CalibrateDisplayShape process. If directories are None,
        uses default test data directory. All input files listed below must be
        on the dir_input path.
        """
        # Define default data directories
        dir_input_sofast = join(opencsp_code_dir(), 'app/sofast/test/data/data_measurement')
        dir_input_def = join(opencsp_code_dir(),
                             'common/lib/deflectometry/test/data/data_measurement')
        dir_output = join(opencsp_code_dir(), 'app/sofast/test/data/data_expected')

        verbose = 1  # 0=no output, 1=only print outputs, 2=print outputs and show plots, 3=plots only with no printing

        # Define input files
        resolution_xy = [100, 100]  # sample density of screen
        file_screen_cal_point_pairs = join(
            dir_input_sofast, 'screen_calibration_point_pairs.csv'
        )
        file_point_locations = join(dir_input_def, 'point_locations.csv')
        file_camera_distortion = join(dir_input_sofast, 'camera_screen_shape.h5')
        file_image_projection = join(dir_input_sofast, 'image_projection.h5')
        files_screen_shape_measurement = glob(
            join(dir_input_sofast, 'screen_shape_sofast_measurements/pose_*.h5')
        )

        # Load input data
        pts_marker_data = np.loadtxt(
            file_point_locations, delimiter=',', dtype=float, skiprows=1
        )
        pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
        corner_ids = pts_marker_data[:, 1]
        screen_cal_point_pairs = np.loadtxt(
            file_screen_cal_point_pairs, delimiter=',', skiprows=1
        ).astype(int)
        camera = Camera.load_from_hdf(file_camera_distortion)
        image_projection_data = ImageProjection.load_from_hdf(file_image_projection)

        # Store input data in data class
        data_input = DataInput(
            corner_ids,
            screen_cal_point_pairs,
            resolution_xy,
            pts_xyz_marker,
            camera,
            image_projection_data,
            [Measurement.load_from_hdf(f) for f in files_screen_shape_measurement],
        )

        # Perform screen position calibration
        cal_screen_position = CalibrateDisplayShape(data_input)
        cal_screen_position.run_calibration(verbose)

        # Get distortion data
        dist_data = cal_screen_position.get_data()

        # Test screen distortion information
        cls.data_exp = load_hdf5_datasets(
            ['pts_xy_screen_fraction', 'pts_xyz_screen_coords'],
            join(dir_output, 'screen_distortion_data_100_100.h5'),
        )
        cls.data_meas = dist_data

    @pytest.mark.no_xvfb
    def test_screen_distortion_data(self):
        """Tests screen calibration data"""
        np.testing.assert_allclose(
            self.data_meas['pts_xy_screen_fraction'].data,
            self.data_exp['pts_xy_screen_fraction'],
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            self.data_meas['pts_xyz_screen_coords'].data,
            self.data_exp['pts_xyz_screen_coords'],
            rtol=0,
            atol=1e-6,
        )
        print('Distortion data tested successfully.')


if __name__ == '__main__':
    unittest.main()
