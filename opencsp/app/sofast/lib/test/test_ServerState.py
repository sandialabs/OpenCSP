"""
Unit test suite to test ServerState class
"""

from concurrent.futures import ThreadPoolExecutor
import os
import time
import unittest

import numpy as np
import pytest

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.ServerState import ServerState
from opencsp.app.sofast.test.test_DisplayShape import TestDisplayShape as t_display
import opencsp.app.sofast.lib.Executor as executor
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationGlobal import ImageCalibrationGlobal
import opencsp.app.sofast.lib.SpatialOrientation as so
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
import opencsp.app.sofast.test.ImageAcquisition_no_camera as ianc
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.camera.test.test_Camera import TestCamera as t_camera
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.deflectometry.Surface2DAbstract import Surface2DAbstract
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.deflectometry.Surface2DPlano import Surface2DPlano
from opencsp.common.lib.deflectometry.test.test_ImageProjection import _ImageProjection as _ip
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path import opencsp_settings
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.process.ControlledContext as cc
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


@pytest.mark.no_xvfb
class TestServerState(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Get data directories
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "ServerState")
        cls.out_dir = os.path.join(path, "data", "output", "ServerState")
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)

    def setUp(self):
        # make sure we can load everything
        sofast_settings = opencsp_settings["sofast_defaults"]
        sofast_settings["projector_file"] = os.path.join(self.data_dir, "projector.h5")
        sofast_settings["calibration_file"] = os.path.join(self.data_dir, "calibration.h5")
        sofast_settings["mirror_measure_point"] = [0, 0, 0]
        sofast_settings["mirror_screen_distance"] = 10
        sofast_settings["camera_calibration_file"] = os.path.join(self.data_dir, "camera_calibration.h5")
        sofast_settings["fixed_pattern_diameter_and_spacing"] = [10, 5]
        sofast_settings["spatial_orientation_file"] = os.path.join(self.data_dir, "spatial_orientation.h5")
        sofast_settings["display_shape_file"] = os.path.join(self.data_dir, "display_shape.h5")
        sofast_settings["facet_definition_files"] = [os.path.join(self.data_dir, "facet_definition.h5")]
        sofast_settings["surface_shape_file"] = os.path.join(self.data_dir, "surface_shape.h5")
        sofast_settings["num_fringe_periods"] = [4, 4]

        # build the default camera
        self.fringes = Fringes.from_num_periods()
        ianc.ImageAcquisitionWithFringes(self.fringes)

        # build the default server state instance
        ServerState()

    def tearDown(self):
        with et.ignored(Exception):
            ImageAcquisitionAbstract.instance().close()
        with et.ignored(Exception):
            ImageProjection.instance().close()
        with et.ignored(Exception):
            with ServerState.instance() as state:
                state.close_all()

    def _wait_on_state_busy(self, desired_busy_signal: bool):
        tstart = time.time()
        while time.time() < tstart + 15:
            time.sleep(0.1)
            with ServerState.instance() as state:
                if state.busy == desired_busy_signal:
                    return
        lt.error_and_raise(RuntimeError, f"Timed out while waiting for busy == {desired_busy_signal}")

    # def test_initial_state(self):
    #     with ServerState.instance() as state:
    #         self.assertFalse(state.is_closed)
    #         self.assertFalse(state.busy)
    #         self.assertTrue(state.projector_available)

    #         self.assertFalse(state.has_fixed_measurement)
    #         self.assertIsNone(state.last_measurement_fixed)
    #         # TODO add test data for fixed and uncomment
    #         # self.assertIsNotNone(state.system_fixed)
    #         # with state.system_fixed as sys:
    #         #     self.assertIsNotNone(sys)
    #         #     self.assertIsInstance(sys, SystemSofastFixed)

    #         self.assertFalse(state.has_fringe_measurement)
    #         self.assertIsNone(state.last_measurement_fringe)
    #         self.assertIsNotNone(state.system_fringe)
    #         with state.system_fringe as sys:
    #             self.assertIsNotNone(sys)
    #             self.assertIsInstance(sys, SystemSofastFringe)

    #         self.assertIsNone(state.processing_error)

    def _test_start_measure_fringes(self):
        """
        This is a fairly large test that checks the state values of the state instance before, during, and after a
        fringe measurement.

        There are several parts to this test:
            1. Check the state before the measurement
            2. Start the measurement
            3. Check the state during the measurement
            4. Check the state after the measurement
            5. Start a new measurement
            6. Check the state during a 2nd measurement
        """
        try:
            # 1. check the initial state
            with ServerState.instance() as state:
                self.assertFalse(state.busy)
                self.assertFalse(state.has_fringe_measurement)
                self.assertTrue(state.projector_available)

            # 2. start the measurement
            with ServerState.instance() as state:
                self.assertTrue(state.start_measure_fringes())

            # wait for the measurement to start
            self._wait_on_state_busy(True)

            # 3. check that the state changed to reflect that a measurement is running
            with ServerState.instance() as state:
                self.assertTrue(state.busy)
                self.assertFalse(state.projector_available)
                self.assertFalse(state.has_fringe_measurement)
                self.assertIsNone(state.last_measurement_fringe)
                # # starting a new measurement while the current measurement is running should fail
                # self.assertFalse(state.start_measure_fringes())

            # wait for the measurement to finish
            self._wait_on_state_busy(False)

            # 4. check that the measurement finished and we have some results
            with ServerState.instance() as state:
                self.assertFalse(state.busy)
                self.assertTrue(state.has_fringe_measurement)
                self.assertIsNotNone(state.last_measurement_fringe)

            # 5. start a 2nd measurement
            with ServerState.instance() as state:
                self.assertTrue(state.start_measure_fringes())

            # wait for the measurement to start
            self._wait_on_state_busy(True)

            # 6. check that the state changed to reflect that a measurement is running
            with ServerState.instance() as state:
                self.assertTrue(state.busy)
                self.assertFalse(state.projector_available)
                self.assertFalse(state.has_fringe_measurement)
                self.assertIsNone(state.last_measurement_fringe)
                # starting a new measurement while the current measurement is running should fail
                self.assertFalse(state.start_measure_fringes())

            # wait for the measurement to finish
            self._wait_on_state_busy(False)

        except Exception as ex:
            lt.error(ex)
            raise

        finally:
            # close the ImageProjection instance in order to exit the main loop
            ImageProjection.instance().close()

    def test_start_measure_fringes(self):
        pool = ThreadPoolExecutor(max_workers=1)
        result = pool.submit(self._test_start_measure_fringes)

        # check if we succeeded
        ex = result.exception()
        if ex is not None:
            raise ex


if __name__ == '__main__':
    lt.logger(level=lt.log.DEBUG)
    unittest.main()
