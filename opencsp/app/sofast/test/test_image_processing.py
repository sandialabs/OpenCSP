"""Unit test suite to test image_processing library
"""

from os.path import join
import unittest

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ParamsSofastFringe import ParamsSofastFringe
from opencsp.common.lib.camera.Camera import Camera
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


class TestImageProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get test data location
        dir_sofast_fringe = join(opencsp_code_dir(), 'test/data/sofast_fringe')
        dir_sofast_common = join(opencsp_code_dir(), 'test/data/sofast_common')

        # Define calculation data files
        cls.data_file_facet = join(dir_sofast_fringe, 'data_expected_facet/data.h5')
        cls.data_file_undefined = join(dir_sofast_fringe, 'data_expected_undefined_mirror/data.h5')
        cls.data_file_multi = join(dir_sofast_fringe, 'data_expected_facet_ensemble/data.h5')

        # Define component files
        cls.data_file_camera = join(dir_sofast_common, 'camera_sofast_downsampled.h5')
        cls.data_file_measurement_facet = join(dir_sofast_fringe, 'data_measurement/measurement_facet.h5')
        cls.data_file_measurement_ensemble = join(dir_sofast_fringe, 'data_measurement/measurement_ensemble.h5')
        cls.data_file_calibration = join(dir_sofast_fringe, 'data_measurement/image_calibration.h5')

        # Sofast fixed dot image
        cls.sofast_fixed_meas = MeasurementSofastFixed.load_from_hdf(
            join(opencsp_code_dir(), 'test/data/sofast_fixed/data_measurement/measurement_facet.h5')
        )

    def test_calc_mask_raw(self):
        """Tests image_processing.calc_mask_raw()"""

        # Load test data
        datasets = ['MeasurementSofastFringe/mask_images']
        data = load_hdf5_datasets(datasets, self.data_file_measurement_facet)

        # Perform calculation
        mask_raw = ip.calc_mask_raw(data['mask_images'])

        # Test
        datasets = ['DataSofastCalculation/general/CalculationImageProcessingGeneral/mask_raw']
        data = load_hdf5_datasets(datasets, self.data_file_facet)
        np.testing.assert_allclose(data['mask_raw'], mask_raw)

    def test_mask_centroid(self):
        """Tests image_processing.centroid_mask()"""
        datasets = [
            'DataSofastCalculation/general/CalculationImageProcessingGeneral/v_mask_centroid_image',
            'DataSofastCalculation/general/CalculationImageProcessingGeneral/mask_raw',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)

        # Perform calculation
        v_mask_cent = ip.centroid_mask(data['mask_raw']).data.squeeze()

        # Test
        np.testing.assert_allclose(data['v_mask_centroid_image'], v_mask_cent)

    def test_keep_largest_mask_area(self):
        """Tests image_processing.keep_largest_mask_area()"""
        datasets = [
            'DataSofastCalculation/image_processing/facet_000/mask_processed',
            'DataSofastCalculation/image_processing/general/mask_raw',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_undefined)

        # Perform calculation
        mask_processed = ip.keep_largest_mask_area(data['mask_raw'])
        mask_processed = np.logical_and(mask_processed, data['mask_raw'])

        # Test
        np.testing.assert_allclose(data['mask_processed'], mask_processed)

    def test_edges_from_mask(self):
        """Tests image_processing.edges_from_mask()"""
        datasets = [
            'DataSofastCalculation/general/CalculationImageProcessingGeneral/v_edges_image',
            'DataSofastCalculation/general/CalculationImageProcessingGeneral/mask_raw',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)

        # Perform calculation
        v_edges = ip.edges_from_mask(data['mask_raw']).data.squeeze()

        # Test
        np.testing.assert_allclose(data['v_edges_image'], v_edges)

    def test_refine_mask_perimeter(self):
        """Tests image_processing.refine_mask_perimeter()"""
        datasets = [
            'DataSofastCalculation/general/CalculationImageProcessingGeneral/v_edges_image',
            'DataSofastCalculation/general/CalculationImageProcessingGeneral/loop_optic_image_exp',
            'DataSofastCalculation/facet/facet_000/CalculationImageProcessingFacet/loop_facet_image_refine',
        ]

        # Get default parameters from Sofast class
        params = ParamsSofastFringe()
        args = [
            params.geometry.perimeter_refine_axial_search_dist,
            params.geometry.perimeter_refine_perpendicular_search_dist,
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)

        # Perform calculation
        loop_facet_exp = LoopXY.from_vertices(Vxy(data['loop_optic_image_exp']))
        loop_facet_refine = ip.refine_mask_perimeter(
            loop_facet_exp, Vxy(data['v_edges_image']), *args
        ).vertices.data.squeeze()

        # Test
        np.testing.assert_allclose(data['loop_facet_image_refine'], loop_facet_refine)

    def test_refine_facet_corners(self):
        """Tests image_processing.refine_facet_corners()"""
        # Load edge data
        datasets = [
            'DataSofastCalculation/general/CalculationImageProcessingGeneral/v_edges_image',
            'DataSofastInput/optic_definition/DefinitionEnsemble/r_facet_ensemble',
        ]
        data = load_hdf5_datasets(datasets, self.data_file_multi)
        num_facets = data['r_facet_ensemble'].shape[0]
        v_edges_image = Vxy(data['v_edges_image'])

        # Loop through facets
        data_exp = []
        data_calc = []
        for facet_idx in range(num_facets):
            # Get sofast parameters
            datasets = [
                'DataSofastInput/ParamsSofastFringe/ParamsOpticGeometry/facet_corns_refine_step_length',
                'DataSofastInput/ParamsSofastFringe/ParamsOpticGeometry/facet_corns_refine_perpendicular_search_dist',
                'DataSofastInput/ParamsSofastFringe/ParamsOpticGeometry/facet_corns_refine_frac_keep',
            ]
            data = load_hdf5_datasets(datasets, self.data_file_multi)
            args = [
                data['facet_corns_refine_step_length'],
                data['facet_corns_refine_perpendicular_search_dist'],
                data['facet_corns_refine_frac_keep'],
            ]

            # Load input and expected output data
            datasets = [
                f'DataSofastCalculation/facet/facet_{facet_idx:03d}/CalculationImageProcessingFacet/loop_facet_image_refine',
                f'DataSofastCalculation/facet/facet_{facet_idx:03d}/CalculationImageProcessingFacet/v_facet_corners_image_exp',
                f'DataSofastCalculation/facet/facet_{facet_idx:03d}/CalculationImageProcessingFacet/v_facet_centroid_image_exp',
            ]
            data = load_hdf5_datasets(datasets, self.data_file_multi)
            v_facet_corners_image_exp = Vxy(data['v_facet_corners_image_exp'])
            v_facet_centroid_image_exp = Vxy(data['v_facet_centroid_image_exp'])

            # Save expected data
            data_exp.append(data['loop_facet_image_refine'])

            # Perform calculation
            reg = ip.refine_facet_corners(v_facet_corners_image_exp, v_facet_centroid_image_exp, v_edges_image, *args)
            data_calc.append(reg.vertices.data)

        data_exp = np.concatenate(data_exp, axis=1)
        data_calc = np.concatenate(data_calc, axis=1)

        # Test
        np.testing.assert_allclose(data_exp, data_calc)

    def test_unwrap_phase(self):
        """Tests image_processing.unwrap_phase()"""

        # Load test data
        datasets = [
            'DataSofastCalculation/facet/facet_000/CalculationDataGeometryFacet/v_screen_points_fractional_screens',
            'DataSofastCalculation/facet/facet_000/CalculationImageProcessingFacet/mask_processed',
        ]
        data = load_hdf5_datasets(datasets, self.data_file_facet)
        measurement = MeasurementSofastFringe.load_from_hdf(self.data_file_measurement_facet)
        calibration = ImageCalibrationScaling.load_from_hdf(self.data_file_calibration)

        measurement.calibrate_fringe_images(calibration)

        x_ims = measurement.fringe_images_x_calibrated
        y_ims = measurement.fringe_images_y_calibrated
        x_periods = measurement.fringe_periods_x
        y_periods = measurement.fringe_periods_y
        mask = data['mask_processed']

        # Perform calculation
        screen_xs = ip.unwrap_phase(x_ims[mask, :].T, x_periods)
        screen_ys = ip.unwrap_phase(y_ims[mask, :].T, y_periods)
        screen_ys = 1.0 - screen_ys
        v_display_pts = np.array([screen_xs, screen_ys])

        # Test
        np.testing.assert_allclose(data['v_screen_points_fractional_screens'], v_display_pts, rtol=1e-06)

    def test_calculate_active_pixel_pointing_vectors(self):
        """Tests image_processing.calculate_active_pixel_pointing_vectors()"""
        datasets = [
            'DataSofastCalculation/facet/facet_000/CalculationDataGeometryFacet/u_pixel_pointing_facet',
            'DataSofastCalculation/general/CalculationDataGeometryGeneral/r_optic_cam_refine_1',
            'DataSofastCalculation/facet/facet_000/CalculationImageProcessingFacet/mask_processed',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)
        camera = Camera.load_from_hdf(self.data_file_camera)

        # Perform calculation
        mask = data['mask_processed']
        r_cam_optic = Rotation.from_rotvec(data['r_optic_cam_refine_1']).inv()
        u_pixel_pointing_cam = ip.calculate_active_pixels_vectors(mask, camera)
        u_pixel_pointing_optic = u_pixel_pointing_cam.rotate(r_cam_optic).data.squeeze()

        # Test
        np.testing.assert_allclose(data['u_pixel_pointing_facet'], u_pixel_pointing_optic)

    def test_detect_blobs(self):
        params = cv.SimpleBlobDetector_Params()
        params.minDistBetweenBlobs = 2
        params.filterByArea = True
        params.minArea = 3
        params.maxArea = 30
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByConvexity = False
        params.filterByInertia = False

        blobs = ip.detect_blobs(self.sofast_fixed_meas.image, params)

        self.assertEqual(len(blobs), 3761, 'Test number of blobs')
        np.testing.assert_allclose(
            blobs[0].data.squeeze(),
            np.array([672.20654297, 1138.20654297]),
            rtol=0,
            atol=1e-6,
            err_msg='First blob pixel location does not match expected',
        )

    def test_detect_blobs_inverse(self):
        params = cv.SimpleBlobDetector_Params()
        params.minDistBetweenBlobs = 2
        params.filterByArea = True
        params.minArea = 3
        params.maxArea = 30
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByConvexity = False
        params.filterByInertia = False

        blobs = ip.detect_blobs_inverse(self.sofast_fixed_meas.image, params)

        self.assertEqual(len(blobs), 1, 'Test number of blobs')
        np.testing.assert_allclose(
            blobs[0].data.squeeze(),
            np.array([960.590515, 796.387695]),
            rtol=0,
            atol=1e-6,
            err_msg='blob pixel location does not match expected',
        )


if __name__ == '__main__':
    unittest.main()
