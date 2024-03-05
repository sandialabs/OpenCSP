"""Integration test. Testing processing of a 'single_facet' type optic.
"""
import glob
import os
import unittest

import numpy as np

import opencsp
from opencsp.common.lib.deflectometry.Display import Display
from opencsp.common.lib.deflectometry.FacetData import FacetData
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.Measurement import Measurement
from opencsp.app.sofast.lib.Sofast import Sofast
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


class TestSingle(unittest.TestCase):
    @classmethod
    def setUpClass(cls, base_dir: str | None = None):
        """Sets up class

        Parameters
        ----------
        base_dir : str | None, optional
            Sets base directory. If None, uses 'data' directory in directory
            contianing file, by default None
        """
        # Get test data location
        if base_dir is None:
            base_dir = os.path.join(
                os.path.dirname(opencsp.__file__), 'test/data/sofast_measurements'
            )

        # Find all test files
        cls.files_dataset = glob.glob(
            os.path.join(base_dir, 'calculations_facet/data*.h5')
        )
        if len(cls.files_dataset) == 0:
            raise ValueError('No single-facet datsets found.')
        else:
            print(f'Testing {len(cls.files_dataset)} single facet datasets')
            for file in cls.files_dataset:
                print(f'Using dataset: {os.path.abspath(file)}')

        # Define component files
        file_measurement = os.path.join(base_dir, 'measurement_facet.h5')

        # Load components
        measurement = Measurement.load_from_hdf(file_measurement)

        # Initialize data containers
        cls.slopes = []
        cls.surf_coefs = []
        cls.v_surf_points_facet = []

        # Load data from all datasets
        for file_dataset in cls.files_dataset:
            # Load display
            camera = Camera.load_from_hdf(file_dataset)
            calibration = ImageCalibrationScaling.load_from_hdf(file_dataset)
            display = Display.load_from_hdf(file_dataset)

            # Calibrate measurement
            measurement.calibrate_fringe_images(calibration)

            # Load surface definition
            surface_data = load_hdf5_datasets(
                [
                    'DataSofastInput/surface_params/facet_000/surface_type',
                    'DataSofastInput/surface_params/facet_000/robust_least_squares',
                    'DataSofastInput/surface_params/facet_000/downsample',
                ],
                file_dataset,
            )
            surface_data['robust_least_squares'] = bool(
                surface_data['robust_least_squares']
            )
            if surface_data['surface_type'] == 'parabolic':
                surface_data.update(
                    load_hdf5_datasets(
                        [
                            'DataSofastInput/surface_params/facet_000/initial_focal_lengths_xy'
                        ],
                        file_dataset,
                    )
                )

            # Load optic data
            facet_data = load_hdf5_datasets(
                [
                    'DataSofastInput/optic_definition/facet_000/v_centroid_facet',
                    'DataSofastInput/optic_definition/facet_000/v_facet_corners',
                ],
                file_dataset,
            )
            facet_data = FacetData(
                Vxyz(facet_data['v_facet_corners']),
                Vxyz(facet_data['v_centroid_facet']),
            )

            # Load sofast params
            datasets = [
                'DataSofastInput/sofast_params/mask_hist_thresh',
                'DataSofastInput/sofast_params/mask_filt_width',
                'DataSofastInput/sofast_params/mask_filt_thresh',
                'DataSofastInput/sofast_params/mask_thresh_active_pixels',
                'DataSofastInput/sofast_params/mask_keep_largest_area',
                'DataSofastInput/sofast_params/perimeter_refine_axial_search_dist',
                'DataSofastInput/sofast_params/perimeter_refine_perpendicular_search_dist',
                'DataSofastInput/sofast_params/facet_corns_refine_step_length',
                'DataSofastInput/sofast_params/facet_corns_refine_perpendicular_search_dist',
                'DataSofastInput/sofast_params/facet_corns_refine_frac_keep',
            ]
            params = load_hdf5_datasets(datasets, file_dataset)

            # Instantiate sofast object
            sofast = Sofast(measurement, camera, display)

            # Update parameters
            sofast.params.mask_hist_thresh = params['mask_hist_thresh']
            sofast.params.mask_filt_width = params['mask_filt_width']
            sofast.params.mask_filt_thresh = params['mask_filt_thresh']
            sofast.params.mask_thresh_active_pixels = params[
                'mask_thresh_active_pixels'
            ]
            sofast.params.mask_keep_largest_area = params['mask_keep_largest_area']

            sofast.params.geometry_params.perimeter_refine_axial_search_dist = params[
                'perimeter_refine_axial_search_dist'
            ]
            sofast.params.geometry_params.perimeter_refine_perpendicular_search_dist = (
                params['perimeter_refine_perpendicular_search_dist']
            )
            sofast.params.geometry_params.facet_corns_refine_step_length = params[
                'facet_corns_refine_step_length'
            ]
            sofast.params.geometry_params.facet_corns_refine_perpendicular_search_dist = params[
                'facet_corns_refine_perpendicular_search_dist'
            ]
            sofast.params.geometry_params.facet_corns_refine_frac_keep = params[
                'facet_corns_refine_frac_keep'
            ]

            # Run SOFAST
            sofast.process_optic_singlefacet(facet_data, surface_data)

            # Store test data
            cls.slopes.append(sofast.data_characterization_facet[0].slopes_facet_xy)
            cls.surf_coefs.append(
                sofast.data_characterization_facet[0].surf_coefs_facet
            )
            cls.v_surf_points_facet.append(
                sofast.data_characterization_facet[0].v_surf_points_facet.data
            )

    def test_slopes(self):
        datasets = ['DataSofastCalculation/facet/facet_000/slopes_facet_xy']
        for idx, file in enumerate(self.files_dataset):
            with self.subTest(i=idx):
                data = load_hdf5_datasets(datasets, file)
                np.testing.assert_allclose(
                    data['slopes_facet_xy'], self.slopes[idx], atol=1e-7, rtol=0
                )

    def test_surf_coefs(self):
        datasets = ['DataSofastCalculation/facet/facet_000/surf_coefs_facet']
        for idx, file in enumerate(self.files_dataset):
            with self.subTest(i=idx):
                data = load_hdf5_datasets(datasets, file)
                np.testing.assert_allclose(
                    data['surf_coefs_facet'], self.surf_coefs[idx], atol=1e-8, rtol=0
                )

    def test_int_points(self):
        datasets = ['DataSofastCalculation/facet/facet_000/v_surf_points_facet']
        for idx, file in enumerate(self.files_dataset):
            with self.subTest(i=idx):
                data = load_hdf5_datasets(datasets, file)
                np.testing.assert_allclose(
                    data['v_surf_points_facet'],
                    self.v_surf_points_facet[idx],
                    atol=1e-8,
                    rtol=0,
                )


if __name__ == '__main__':
    Test = TestSingle()
    Test.setUpClass()

    print('test_slopes', flush=True)
    Test.test_slopes()

    print('test_surf_coefs', flush=True)
    Test.test_surf_coefs()

    print('test_int_points', flush=True)
    Test.test_int_points()

    print('All tests run.')
