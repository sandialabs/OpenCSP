"""Integration test. Testing processing of a 'multi_facet' type optic.
"""
import os
import unittest

from scipy.spatial.transform import Rotation
import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
from opencsp.common.lib.camera.Camera import Camera
from opencsp.app.sofast.lib.Display import Display
from opencsp.app.sofast.lib.EnsembleData import EnsembleData
from opencsp.app.sofast.lib.FacetData import FacetData
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


class TestMulti(unittest.TestCase):
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
                opencsp_code_dir(), 'test/data/sofast_measurements'
            )

        # Directory Setup
        file_dataset = os.path.join(base_dir, 'calculations_facet_ensemble/data.h5')
        file_measurement = os.path.join(base_dir, 'measurement_ensemble.h5')
        print(f'Using dataset: {os.path.abspath(file_dataset)}')

        # Load data
        camera = Camera.load_from_hdf(file_dataset)
        display = Display.load_from_hdf(file_dataset)
        measurement = Measurement.load_from_hdf(file_measurement)
        calibration = ImageCalibrationScaling.load_from_hdf(file_dataset)

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

        # Calibrate measurement
        measurement.calibrate_fringe_images(calibration)

        # Instantiate sofast object
        sofast = Sofast(measurement, camera, display)

        # Update parameters
        sofast.params.mask_hist_thresh = params['mask_hist_thresh']
        sofast.params.mask_filt_width = params['mask_filt_width']
        sofast.params.mask_filt_thresh = params['mask_filt_thresh']
        sofast.params.mask_thresh_active_pixels = params['mask_thresh_active_pixels']
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
        sofast.params.geometry_params.facet_corns_refine_perpendicular_search_dist = (
            params['facet_corns_refine_perpendicular_search_dist']
        )
        sofast.params.geometry_params.facet_corns_refine_frac_keep = params[
            'facet_corns_refine_frac_keep'
        ]

        # Load array data
        datasets = [
            'DataSofastInput/optic_definition/ensemble/ensemble_perimeter',
            'DataSofastInput/optic_definition/ensemble/r_facet_ensemble',
            'DataSofastInput/optic_definition/ensemble/v_centroid_ensemble',
            'DataSofastInput/optic_definition/ensemble/v_facet_locations',
        ]
        ensemble_data = load_hdf5_datasets(datasets, file_dataset)
        ensemble_data = DefinitionEnsemble(
            Vxyz(ensemble_data['v_facet_locations']),
            [Rotation.from_rotvec(r) for r in ensemble_data['r_facet_ensemble']],
            ensemble_data['ensemble_perimeter'],
            Vxyz(ensemble_data['v_centroid_ensemble']),
        )

        # Load facet data
        facet_data = []
        for idx in range(len(ensemble_data.r_facet_ensemble)):
            datasets = [
                f'DataSofastInput/optic_definition/facet_{idx:03d}/v_centroid_facet',
                f'DataSofastInput/optic_definition/facet_{idx:03d}/v_facet_corners',
            ]
            data = load_hdf5_datasets(datasets, file_dataset)
            facet_data.append(
                FacetData(Vxyz(data['v_facet_corners']), Vxyz(data['v_centroid_facet']))
            )

        # Load surface data
        surface_data = []
        for idx in range(len(facet_data)):
            datasets = [
                f'DataSofastInput/surface_params/facet_{idx:03d}/downsample',
                f'DataSofastInput/surface_params/facet_{idx:03d}/initial_focal_lengths_xy',
                f'DataSofastInput/surface_params/facet_{idx:03d}/robust_least_squares',
                f'DataSofastInput/surface_params/facet_{idx:03d}/surface_type',
            ]
            data = load_hdf5_datasets(datasets, file_dataset)
            data['robust_least_squares'] = bool(data['robust_least_squares'])
            surface_data.append(data)

        # Run SOFAST
        sofast.process_optic_multifacet(facet_data, ensemble_data, surface_data)

        # Store data
        cls.data_test = {'slopes_facet_xy': [], 'surf_coefs_facet': []}

        cls.num_facets = sofast.num_facets
        cls.file_dataset = file_dataset

        for idx in range(sofast.num_facets):
            cls.data_test['slopes_facet_xy'].append(
                sofast.data_characterization_facet[idx].slopes_facet_xy
            )
            cls.data_test['surf_coefs_facet'].append(
                sofast.data_characterization_facet[idx].surf_coefs_facet
            )

    def test_slope(self):
        for idx in range(self.num_facets):
            with self.subTest(i=idx):
                # Get calculated data
                data_calc = self.data_test['slopes_facet_xy'][idx]

                # Get expected data
                datasets = [
                    f'DataSofastCalculation/facet/facet_{idx:03d}/slopes_facet_xy'
                ]
                data = load_hdf5_datasets(datasets, self.file_dataset)

                # Test
                np.testing.assert_allclose(
                    data['slopes_facet_xy'], data_calc, atol=1e-7, rtol=0
                )

    def test_surf_coefs(self):
        for idx in range(self.num_facets):
            with self.subTest(i=idx):
                # Get calculated data
                data_calc = self.data_test['surf_coefs_facet'][idx]

                # Get expected data
                datasets = [
                    f'DataSofastCalculation/facet/facet_{idx:03d}/surf_coefs_facet'
                ]
                data = load_hdf5_datasets(datasets, self.file_dataset)

                # Test
                np.testing.assert_allclose(
                    data['surf_coefs_facet'], data_calc, atol=1e-8, rtol=0
                )


if __name__ == '__main__':
    Test = TestMulti()
    Test.setUpClass()

    Test.test_slope()
    Test.test_surf_coefs()
    print('All tests run')
