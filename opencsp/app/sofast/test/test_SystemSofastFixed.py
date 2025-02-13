import unittest
from os.path import join, dirname

import pytest

from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.app.sofast.test.ImageAcquisition_no_camera import ImageAcquisition
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class TestSystemSofastFixed(unittest.TestCase):
    @pytest.mark.no_xvfb
    def test_system(self):
        # Get test data location
        file_im_proj = join(opencsp_code_dir(), "test/data/sofast_common/image_projection_test.h5")

        # Instantiate image projection class
        im_proj = ImageProjection.load_from_hdf(file_im_proj)

        # Instantiate image acquisition class
        im_aq = ImageAcquisition()

        # Set camera settings
        im_aq.frame_size = (100, 80)
        im_aq.frame_rate = 7
        im_aq.exposure_time = 300000
        im_aq.gain = 230

        # Create system class
        system = SystemSofastFixed(im_aq)
        system.image_delay = 300
        system.image_acquisition.exposure_time = 0.1  # seconds

        # Define pattern parameters
        system.set_pattern_parameters(5, 5)

        # Define functions to put in system queue
        funcs = [system.run_measurement, system.close_all]

        # Load function in queue
        system.set_queue(funcs)

        # Run
        system.run()

        # Get measurement
        v_measure_point_facet = Vxyz((0.0, 0.0, 0.0))
        dist_optic_screen = 10.0
        origin = Vxy((10, 200))
        measurement = system.get_measurement(v_measure_point_facet, dist_optic_screen, origin)


if __name__ == "__main__":
    save_dir = join(dirname(__file__), "data/output/system_fixed")
    ft.create_directories_if_necessary(save_dir)
    lt.logger(join(save_dir, "log.txt"), level=lt.log.DEBUG)

    unittest.main()
