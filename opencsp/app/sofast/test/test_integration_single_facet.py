"""Integration test. Testing processing of a 'single_facet' type optic.
"""

import glob
import os
import unittest

import numpy as np

from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ParamsSofastFringe import ParamsSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.Surface2DPlano import Surface2DPlano
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
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
            base_dir = os.path.join(opencsp_code_dir(), "test/data/sofast_fringe")

        # Find all test files
        cls.files_dataset = glob.glob(os.path.join(base_dir, "data_expected_facet/data*.h5"))
        cls.files_dataset.sort()
        if len(cls.files_dataset) == 0:
            raise ValueError("No single-facet datsets found.")

        # Define component files
        file_measurement = os.path.join(base_dir, "data_measurement/measurement_facet.h5")

        # Load components
        measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)

        # Initialize data containers
        cls.slopes = []
        cls.surf_coefs = []
        cls.v_surf_points_facet = []

        # Load data from all datasets
        for file_dataset in cls.files_dataset:
            # Load components
            camera = Camera.load_from_hdf(file_dataset)
            orientation = SpatialOrientation.load_from_hdf(file_dataset)
            calibration = ImageCalibrationScaling.load_from_hdf(file_dataset)
            display = DisplayShape.load_from_hdf(file_dataset)
            facet_data = DefinitionFacet.load_from_hdf(file_dataset, "DataSofastInput/optic_definition/facet_000/")
            params_sofast = ParamsSofastFringe.load_from_hdf(file_dataset, "DataSofastInput/")
            if "plano" in os.path.basename(file_dataset):
                surface = Surface2DPlano.load_from_hdf(file_dataset, "DataSofastInput/optic_definition/facet_000/")
            else:
                surface = Surface2DParabolic.load_from_hdf(file_dataset, "DataSofastInput/optic_definition/facet_000/")

            # Calibrate measurement
            measurement.calibrate_fringe_images(calibration)

            # Instantiate sofast object
            sofast = ProcessSofastFringe(measurement, orientation, camera, display)

            # Update parameters
            sofast.params = params_sofast

            # Run SOFAST
            sofast.process_optic_singlefacet(facet_data, surface)

            # Store test data
            cls.slopes.append(sofast.data_calculation_facet[0].slopes_facet_xy)
            cls.surf_coefs.append(sofast.data_calculation_facet[0].surf_coefs_facet)
            cls.v_surf_points_facet.append(sofast.data_calculation_facet[0].v_surf_points_facet.data)

    def test_slopes(self):
        datasets = ["DataSofastCalculation/facet/facet_000/SlopeSolverData/slopes_facet_xy"]
        for idx, file in enumerate(self.files_dataset):
            with self.subTest(i=idx):
                data = load_hdf5_datasets(datasets, file)
                np.testing.assert_allclose(data["slopes_facet_xy"], self.slopes[idx], atol=1e-7, rtol=0)

    def test_surf_coefs(self):
        datasets = ["DataSofastCalculation/facet/facet_000/SlopeSolverData/surf_coefs_facet"]
        for idx, file in enumerate(self.files_dataset):
            with self.subTest(i=idx):
                data = load_hdf5_datasets(datasets, file)
                np.testing.assert_allclose(data["surf_coefs_facet"], self.surf_coefs[idx], atol=1e-8, rtol=0)

    def test_int_points(self):
        datasets = ["DataSofastCalculation/facet/facet_000/SlopeSolverData/v_surf_points_facet"]
        for idx, file in enumerate(self.files_dataset):
            with self.subTest(i=idx):
                data = load_hdf5_datasets(datasets, file)
                np.testing.assert_allclose(
                    data["v_surf_points_facet"], self.v_surf_points_facet[idx], atol=1e-8, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
