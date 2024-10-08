"""Integration test. Testing processing of a 'multi_facet' type optic.

To update the test data, simply run this test and replace the input SOFAST
dataset HDF5 file with the output HDF5 file created when running this test.
"""

import os
import unittest

import numpy as np

from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets
import opencsp.common.lib.tool.file_tools as ft
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
        # Define save directory
        cls.dir_save = os.path.join(os.path.dirname(__file__), 'data/output/sofast_ensemble')
        ft.create_directories_if_necessary(cls.dir_save)

        # Get test data location
        if base_dir is None:
            base_dir = os.path.join(opencsp_code_dir(), 'test/data/sofast_fringe')

        # Directory Setup
        file_dataset = os.path.join(base_dir, 'data_expected_facet_ensemble/data.h5')
        file_measurement = os.path.join(base_dir, 'data_measurement/measurement_ensemble.h5')

        # Load data
        camera = Camera.load_from_hdf(file_dataset)
        display = DisplayShape.load_from_hdf(file_dataset)
        orientation = SpatialOrientation.load_from_hdf(file_dataset)
        measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
        calibration = ImageCalibrationScaling.load_from_hdf(file_dataset)
        data_ensemble = DefinitionEnsemble.load_from_hdf(file_dataset, 'DataSofastInput/optic_definition/')
        data_facets = []
        data_surfaces = []
        for idx_facet in range(data_ensemble.num_facets):
            prefix = f'DataSofastInput/optic_definition/facet_{idx_facet:03d}/'
            data_surfaces.append(Surface2DParabolic.load_from_hdf(file_dataset, prefix))
            data_facets.append(DefinitionFacet.load_from_hdf(file_dataset, prefix))

        # Calibrate measurement
        measurement.calibrate_fringe_images(calibration)

        # Instantiate sofast object
        sofast = ProcessSofastFringe(measurement, orientation, camera, display)

        # Update sofast processing parameters
        sofast.params = sofast.params.load_from_hdf(file_dataset, 'DataSofastInput/')

        # Run SOFAST
        sofast.process_optic_multifacet(data_facets, data_ensemble, data_surfaces)

        # Store data
        cls.data_test = {'slopes_facet_xy': [], 'surf_coefs_facet': [], 'facet_pointing': []}

        cls.num_facets = sofast.num_facets
        cls.file_dataset = file_dataset
        cls.sofast = sofast

        for idx in range(sofast.num_facets):
            cls.data_test['slopes_facet_xy'].append(sofast.data_characterization_facet[idx].slopes_facet_xy)
            cls.data_test['surf_coefs_facet'].append(sofast.data_characterization_facet[idx].surf_coefs_facet)
            cls.data_test['facet_pointing'].append(
                sofast.data_characterization_ensemble[idx].v_facet_pointing_ensemble.data.squeeze()
            )

    def test_slope(self):
        for idx in range(self.num_facets):
            with self.subTest(i=idx):
                # Get calculated data
                data_calc = self.data_test['slopes_facet_xy'][idx]

                # Get expected data
                datasets = [f'DataSofastCalculation/facet/facet_{idx:03d}/SlopeSolverData/slopes_facet_xy']
                data = load_hdf5_datasets(datasets, self.file_dataset)

                # Test
                np.testing.assert_allclose(data['slopes_facet_xy'], data_calc, atol=1e-7, rtol=0)

    def test_surf_coefs(self):
        for idx in range(self.num_facets):
            with self.subTest(i=idx):
                # Get calculated data
                data_calc = self.data_test['surf_coefs_facet'][idx]

                # Get expected data
                datasets = [f'DataSofastCalculation/facet/facet_{idx:03d}/SlopeSolverData/surf_coefs_facet']
                data = load_hdf5_datasets(datasets, self.file_dataset)

                # Test
                np.testing.assert_allclose(data['surf_coefs_facet'], data_calc, atol=1e-8, rtol=0)

    def test_facet_pointing(self):
        for idx in range(self.num_facets):
            with self.subTest(i=idx):
                # Get calculated data
                data_calc = self.data_test['facet_pointing'][idx]

                # Get expected data
                datasets = [
                    f'DataSofastCalculation/facet/facet_{idx:03d}/CalculationEnsemble/v_facet_pointing_ensemble'
                ]
                data = load_hdf5_datasets(datasets, self.file_dataset)

                # Test
                np.testing.assert_allclose(data['v_facet_pointing_ensemble'], data_calc, atol=1e-8, rtol=0)

    def test_save_as_hdf5(self):
        self.sofast.save_to_hdf(os.path.join(self.dir_save, 'sofast_processed_data_ensemble.h5'))


if __name__ == '__main__':
    unittest.main()
