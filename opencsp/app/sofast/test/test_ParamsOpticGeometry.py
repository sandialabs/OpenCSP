import unittest
from os.path import join, dirname

from opencsp.app.sofast.lib.ParamsOpticGeometry import ParamsOpticGeometry
import opencsp.common.lib.tool.file_tools as ft


class TestParamsOpticGeometry(unittest.TestCase):
    def test_save_load_hdf(self):
        # Define save dir
        dir_save = join(dirname(__file__), "data/output/ParamsOpticGeometry")
        ft.create_directories_if_necessary(dir_save)
        file_save = join(dir_save, "params_optic_geometry.h5")

        # Instantiate with defaults
        params = ParamsOpticGeometry()

        # Test save
        params.save_to_hdf(file_save)

        # Test load
        ParamsOpticGeometry.load_from_hdf(file_save)


if __name__ == "__main__":
    unittest.main()
