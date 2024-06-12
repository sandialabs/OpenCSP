from os.path import join, dirname
import unittest

import opencsp.common.lib.tool.file_tools as ft


class TestSpatialOrientation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup paths
        cls.dir_input = join(dirname(__file__), 'data/input/SofastConfiguration')
        ft.create_directories_if_necessary(cls.dir_input)
        cls.dir_output = join(dirname(__file__), 'data/output/SofastConfiguration')
        ft.create_directories_if_necessary(cls.dir_output)

        # Create sofast fringe instance
        cls.process_sofast_fringe = None

        # Create sofast fixed instance
        cls.process_sofast_fixed = None

    def test_visualize_setup_fringe(self):
        pass

    def test_measurement_stats_setup_fringe(self):
        pass

    def test_visualize_setup_fixed(self):
        pass

    def test_measurement_stats_setup_fixed(self):
        pass


if __name__ == '__main__':
    unittest.main()
