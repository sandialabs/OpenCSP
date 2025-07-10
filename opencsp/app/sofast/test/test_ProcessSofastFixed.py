"""Test script that runs fixed pattern deflectometry analysis on saved data"""

from os.path import join
import unittest

import matplotlib.pyplot as plt
import numpy as np

from opencsp.app.sofast.lib.DefinitionEnsemble import DefinitionEnsemble
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ParamsSofastFixed import ParamsSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.hdf5_tools as h5
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestProcessSofastFixed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads data and runs ProcessSofastFixed"""
        cls.dir_sofast_fixed = join(opencsp_code_dir(), "test/data/sofast_fixed")
        cls.dir_sofast_common = join(opencsp_code_dir(), "test/data/sofast_common")
        cls.save_dir = join(opencsp_code_dir(), "app/sofast/test/data/output/process_sofast_fixed")
        ft.create_directories_if_necessary(cls.save_dir)

        lt.logger(join(cls.save_dir, "sofast_fixed_process_log.txt"), lt.log.ERROR)

        cls.sofast_single_facet: ProcessSofastFixed = None
        cls.sofast_facet_ensemble: ProcessSofastFixed = None
        cls.exp_slopes_xy_single_facet: np.ndarray = None
        cls.exp_slopes_xy_facet_ensemble: list[np.ndarray] = None
        cls.exp_facet_pointing_trans: list[np.ndarray] = None

        cls._process_facet_ensemble(cls)
        cls._process_single_facet(cls)

    def _process_facet_ensemble(self):
        # Definitions
        file_camera = join(self.dir_sofast_common, "camera_sofast.h5")
        file_facet = join(self.dir_sofast_common, "Facet_split_NSTTF.json")
        file_ensemble = join(self.dir_sofast_common, "Ensemble_split_NSTTF_facet.json")
        file_ori = join(self.dir_sofast_common, "spatial_orientation.h5")
        file_dot_locs = join(self.dir_sofast_fixed, "data_measurement/fixed_pattern_dot_locations.h5")
        file_meas = join(self.dir_sofast_fixed, "data_measurement/measurement_facet_ensemble.h5")
        file_exp = join(self.dir_sofast_fixed, "data_expected/calculation_facet_ensemble.h5")

        # Load data
        camera = Camera.load_from_hdf(file_camera)
        ensemble_data = DefinitionEnsemble.load_from_json(file_ensemble)
        facet_data = [DefinitionFacet.load_from_json(file_facet)] * ensemble_data.num_facets
        dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
        orientation = SpatialOrientation.load_from_hdf(file_ori)
        measurement = MeasurementSofastFixed.load_from_hdf(file_meas)
        surfaces = [
            Surface2DParabolic(initial_focal_lengths_xy=(150.0, 150), robust_least_squares=False, downsample=1)
        ] * ensemble_data.num_facets

        # Load expected data
        self.exp_slopes_xy_facet_ensemble = []
        self.exp_facet_pointing_trans = []
        for idx_facet in range(ensemble_data.num_facets):
            datasets = [
                f"DataSofastCalculation/facet/facet_{idx_facet:03d}/SlopeSolverData/slopes_facet_xy",
                f"DataSofastCalculation/facet/facet_{idx_facet:03d}/CalculationEnsemble/trans_facet_ensemble",
            ]
            data = h5.load_hdf5_datasets(datasets, file_exp)
            self.exp_slopes_xy_facet_ensemble.append(data["slopes_facet_xy"])
            self.exp_facet_pointing_trans.append(data["trans_facet_ensemble"])

        # Instantiate class
        params = ParamsSofastFixed.load_from_hdf(file_exp, "DataSofastInput/")
        self.sofast_facet_ensemble = ProcessSofastFixed(orientation, camera, dot_locs)
        self.sofast_facet_ensemble.params = params
        self.sofast_facet_ensemble.load_measurement_data(measurement)

        # Process
        pts_known = Vxy(((853, 1031), (680, 683)))
        xys_known = ((0, 0), (15, 0))
        self.sofast_facet_ensemble.process_multi_facet_optic(facet_data, surfaces, ensemble_data, pts_known, xys_known)

    def _process_single_facet(self):
        # Definitions
        file_camera = join(self.dir_sofast_common, "camera_sofast.h5")
        file_facet = join(self.dir_sofast_common, "Facet_NSTTF.json")
        file_ori = join(self.dir_sofast_common, "spatial_orientation.h5")
        file_dot_locs = join(self.dir_sofast_fixed, "data_measurement/fixed_pattern_dot_locations.h5")
        file_meas = join(self.dir_sofast_fixed, "data_measurement/measurement_facet.h5")
        file_exp = join(self.dir_sofast_fixed, "data_expected/calculation_facet.h5")

        # Load data
        surface = Surface2DParabolic(initial_focal_lengths_xy=(150.0, 150), robust_least_squares=False, downsample=1)
        camera = Camera.load_from_hdf(file_camera)
        facet_data = DefinitionFacet.load_from_json(file_facet)
        dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
        orientation = SpatialOrientation.load_from_hdf(file_ori)
        measurement = MeasurementSofastFixed.load_from_hdf(file_meas)

        # Load expected data
        datasets = ["DataSofastCalculation/facet/facet_000/SlopeSolverData/slopes_facet_xy"]
        data = h5.load_hdf5_datasets(datasets, file_exp)
        self.exp_slopes_xy_single_facet = data["slopes_facet_xy"]

        # Instantiate class
        params = ParamsSofastFixed.load_from_hdf(file_exp, "DataSofastInput/")
        self.sofast_single_facet = ProcessSofastFixed(orientation, camera, dot_locs)
        self.sofast_single_facet.params = params
        self.sofast_single_facet.load_measurement_data(measurement)

        # Process
        pt_known = measurement.origin
        xy_known = (0, 0)
        self.sofast_single_facet.process_single_facet_optic(facet_data, surface, pt_known, xy_known)

    def test_save_facet_ensemble_as_hdf(self):
        """Tests saving to HDF file"""
        self.sofast_facet_ensemble.save_to_hdf(join(self.save_dir, "data_calculation_facet_ensemble.h5"))

    def test_save_single_facet_as_hdf(self):
        """Tests saving to HDF file"""
        self.sofast_single_facet.save_to_hdf(join(self.save_dir, "data_calculation_single_facet.h5"))

    def test_save_facet_ensemble_slope_figure(self):
        """Tests genreating and saving a figure of a facet ensemble"""
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()

        fig_mng = fm.setup_figure(figure_control, axis_control_m, title="")
        ensemble = self.sofast_facet_ensemble.get_optic("bilinear")
        res_meter = 0.002  # meter, 2mm resolution
        clim_mrad = 7  # +/- 7 mrad color bar limits
        ensemble.plot_orthorectified_slope(res=res_meter, clim=clim_mrad, axis=fig_mng.axis)
        fig_mng.save(self.save_dir, "slope_magnitude_facet_ensemble", "png")

    def test_save_single_facet_slope_figure(self):
        """Tests generating and saving a figure of a single facet"""
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()

        fig_mng = fm.setup_figure(figure_control, axis_control_m, title="")
        facet = self.sofast_single_facet.get_optic("bilinear")
        res_meter = 0.002  # meter, 2mm resolution
        clim_mrad = 7  # +/- 7 mrad color bar limits
        facet.plot_orthorectified_slope(res=res_meter, clim=clim_mrad, axis=fig_mng.axis)
        fig_mng.save(self.save_dir, "slope_magnitude_single_facet", "png")

    def test_slopes_xy_facet_ensemble(self):
        """Tests slope data"""
        for idx_facet in range(self.sofast_facet_ensemble.num_facets):
            with self.subTest(i=idx_facet):
                np.testing.assert_allclose(
                    self.sofast_facet_ensemble.data_calculation_facet[idx_facet].slopes_facet_xy,
                    self.exp_slopes_xy_facet_ensemble[idx_facet],
                    rtol=0,  # relative tolerance (i.e. +/- a fixed fraction of the expected value)
                    atol=1e-6,  # absolute tolerance (i.e. +/- a fixed value)
                )

    def test_slopes_xy_single_facet(self):
        """Tests slope data"""
        np.testing.assert_allclose(
            self.sofast_single_facet.data_calculation_facet[0].slopes_facet_xy,
            self.exp_slopes_xy_single_facet,
            rtol=0,  # relative tolerance (i.e. +/- a fixed fraction of the expected value)
            atol=1e-6,  # absolute tolerance (i.e. +/- a fixed value)
        )

    def test_facet_pointing_ensemble(self):
        """Tests facet pointing"""
        for idx_facet in range(self.sofast_facet_ensemble.num_facets):
            with self.subTest(i=idx_facet):
                np.testing.assert_allclose(
                    self.sofast_facet_ensemble.data_calculation_ensemble[idx_facet].trans_facet_ensemble.matrix,
                    self.exp_facet_pointing_trans[idx_facet],
                    rtol=0,  # relative tolerance (i.e. +/- a fixed fraction of the expected value)
                    atol=1e-6,  # absolute tolerance (i.e. +/- a fixed value)
                )

    def tearDown(self) -> None:
        # Make sure we release all matplotlib resources.
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
