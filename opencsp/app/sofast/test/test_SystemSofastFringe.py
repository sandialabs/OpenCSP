"""Unit test suite to test the System class
"""

import os
import unittest

import pytest

from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
from opencsp.app.sofast.test.ImageAcquisition_no_camera import ImageAcquisition
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


class TestSystemSofastFringe(unittest.TestCase):
    @pytest.mark.no_xvfb
    def test_SystemSofastFringe(self):
        # Get test data location
        file_im_proj = os.path.join(opencsp_code_dir(), 'test/data/sofast_common/image_projection_test.h5')

        # Create fringe object
        periods_x = [0.9, 3.9]
        periods_y = [15.9, 63.9]
        fringes = Fringes(periods_x, periods_y)

        # Instantiate image projection class
        im_proj = ImageProjection.load_from_hdf_and_display(file_im_proj)

        # Instantiate image acquisition class
        im_aq = ImageAcquisition()

        # Set camera settings
        im_aq.frame_size = (100, 80)
        im_aq.frame_rate = 7
        im_aq.exposure_time = 300000
        im_aq.gain = 230

        # Create system class
        system = SystemSofastFringe(im_proj, im_aq)

        # Load fringes
        system.load_fringes(fringes, 0)

        # Define functions to put in system queue
        def f1():
            system.capture_mask_and_fringe_images(system.run_next_in_queue)

        def f2():
            system.close_all()

        # Load function in queue
        system.set_queue([f1, f2])

        # Run
        system.run()


if __name__ == '__main__':
    unittest.main()
