"""Unit test suite to test Executor class
"""

from concurrent.futures import ThreadPoolExecutor
import os
import time
from tkinter import Tk
import unittest

import numpy as np
import pytest

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DisplayShape import DisplayShape
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
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.process.ControlledContext as cc
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


@pytest.mark.no_xvfb
class TestExecutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Get data directories
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "Executor")
        cls.out_dir = os.path.join(path, "data", "output", "Executor")
        cls.so_file = os.path.join(orp.opencsp_code_dir(), 'test/data/sofast_fringe/data_expected_facet/data.h5')
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)

    def setUp(self):
        # placeholders
        self.sys_fringe: cc.ControlledContext[SystemSofastFringe] = None
        self.fringe_collected = False
        self.fringe_processed = False
        self.fringe_results: executor.FringeResults = None
        self.reported_error: Exception = None

        # build the default projector instance
        ImageProjection.in_new_window(_ip.display_dict)

        # build the default executor instance
        self.executor = executor.Executor()
        self.executor.on_fringe_collected = self._on_fringe_collected
        self.executor.on_fringe_processed = self._on_fringe_processed

        # build the default system instance
        self._build_sys_fringe()

        # build the default surface instances
        self.surface_plano = Surface2DPlano(False, 10)
        self.surface_parabolic = Surface2DParabolic((500, 500), False, 10)

    def tearDown(self):
        with et.ignored(Exception):
            self.executor.close()
        with et.ignored(Exception):
            ImageAcquisitionAbstract.instance().close()
        with et.ignored(Exception):
            ImageProjection.instance().close()
        with et.ignored(Exception):
            with self.sys_fringe as sys:
                sys.close_all()

    def _build_sys_fringe(self):
        # Build Fringe system for testing
        fringes = Fringes.from_num_periods()
        ianc.ImageAcquisitionWithFringes(fringes)
        self.sys_fringe = cc.ControlledContext(SystemSofastFringe())
        projector_values = np.arange(0, 255, (255 - 0) / 9)
        camera_response = np.arange(5, 50, (50 - 5) / 9)
        calibration_global = ImageCalibrationGlobal(camera_response, projector_values)
        with self.sys_fringe as sys:
            sys.calibration = calibration_global
            sys.set_fringes(fringes)

    def _start_processing(self, surface: Surface2DAbstract):
        mirror_measure_point = Vxyz((0, 0, 0))
        mirror_measure_dist = 10
        spatial_orientation = so.SpatialOrientation.load_from_hdf(self.so_file)
        camera_calibration = Camera(
            t_camera.intrinsic_mat, t_camera.distortion_coef_zeros, t_camera.image_shape_xy, "IdealCamera"
        )
        display_shape = DisplayShape(t_display.grid_data_rect2D, "Rectangular2D")
        facet = DefinitionFacet(
            Vxyz(np.array([[-2.4, 2.7, 2.7, -2.35], [-1.15, -1.3, 1.3, 1.15], [0, 0, 0, 0]])), Vxyz([0, 0, 0])
        )
        self.executor.start_process_fringe(
            self.sys_fringe,
            mirror_measure_point,
            mirror_measure_dist,
            spatial_orientation,
            camera_calibration,
            display_shape,
            facet,
            surface,
        )

    def _on_fringe_collected(self):
        self.fringe_collected = True
        ImageProjection.instance().close()

    def _on_fringe_processed(self, fringe_results: executor.FringeResults, ex: Exception):
        self.fringe_processed = True
        self.fringe_results = fringe_results
        self.reported_error = ex

    def test_collect_fringe_does_not_block(self):
        # Prep the collection
        self.executor.start_collect_fringe(self.sys_fringe)

        # If we get here, then that means the collection has been queued
        pass

    def test_collect_fringe_completes(self):
        # Prep the collection
        self.executor.start_collect_fringe(self.sys_fringe)
        self.assertFalse(self.fringe_collected)

        # Start the collection
        ImageProjection.instance().root.mainloop()

        # If we get here, then the collection should have finished and mainloop() exited
        self.assertTrue(self.fringe_collected)

    def test_collect_fringe_from_other_thread(self):
        pool = ThreadPoolExecutor(max_workers=1)
        result = pool.submit(self.executor.start_collect_fringe, self.sys_fringe)

        # Start the collection
        ImageProjection.instance().root.mainloop()

        # Check for errors
        if result.exception() is not None:
            raise result.exception()

        # If we get here, then the collection should have finished and mainloop() exited
        self.assertTrue(self.fringe_collected)

    def test_process_fringe_synchronous(self):
        # Run the collection
        self.executor.asynchronous_processing = False
        self.executor.start_collect_fringe(self.sys_fringe)
        ImageProjection.instance().root.mainloop()

        # Start the processing
        self.assertFalse(self.fringe_processed)
        self._start_processing(self.surface_plano)

        # Check that processing finished
        self.assertTrue(self.fringe_processed)
        if self.reported_error is not None:
            raise self.reported_error
        self.assertIsNotNone(self.fringe_results)

    def test_process_fringe_asynchronous(self):
        # Run the collection
        self.executor.start_collect_fringe(self.sys_fringe)
        ImageProjection.instance().root.mainloop()

        # Start the processing
        self.assertFalse(self.fringe_processed)
        with self.sys_fringe as sys:  # blocks the executor's process thread from continuing
            self._start_processing(self.surface_plano)
            self.assertFalse(self.fringe_processed)
        # processing can start at this point

        # Wait until processing is done
        tstart = time.time()
        while not self.fringe_processed:
            if time.time() > tstart + 10:
                break
            time.sleep(0.1)

        # Check that processing finished
        self.assertTrue(self.fringe_processed)
        if self.reported_error is not None:
            raise self.reported_error
        self.assertIsNotNone(self.fringe_results)

    def test_process_fringe_parbolic(self):
        # Run the collection
        self.executor.asynchronous_processing = False
        self.executor.start_collect_fringe(self.sys_fringe)
        ImageProjection.instance().root.mainloop()

        # Start the processing
        self.assertFalse(self.fringe_processed)
        self._start_processing(self.surface_parabolic)

        # Check that processing finished
        self.assertTrue(self.fringe_processed)
        if self.reported_error is not None:
            raise self.reported_error
        self.assertIsNotNone(self.fringe_results)
        self.assertIsInstance(self.fringe_results.focal_length_x, float)
        self.assertIsInstance(self.fringe_results.focal_length_y, float)


if __name__ == '__main__':
    unittest.main()
