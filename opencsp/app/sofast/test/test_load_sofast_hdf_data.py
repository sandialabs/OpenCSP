from os.path import join
import unittest

import opencsp.app.sofast.lib.load_sofast_hdf_data as lsd
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


class TestImageProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get test data location
        cls.file_fringe = join(opencsp_code_dir(), "test/data/sofast_fringe/data_expected_facet/data.h5")
        cls.file_fixed = join(opencsp_code_dir(), "test/data/sofast_fixed/data_expected/calculation_facet.h5")

    def test_load_fringe(self):
        lsd.load_mirror(self.file_fringe)

    def test_load_fringe_ideal(self):
        lsd.load_mirror_ideal(self.file_fringe, 100.0)

    def test_load_fixed(self):
        lsd.load_mirror(self.file_fixed)

    def test_load_fixed_ideal(self):
        lsd.load_mirror_ideal(self.file_fixed, 100.0)


if __name__ == "__main__":
    unittest.main()
