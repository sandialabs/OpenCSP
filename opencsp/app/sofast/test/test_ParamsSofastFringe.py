import unittest
from os.path import join, dirname

from opencsp.app.sofast.lib.ParamsSofastFringe import ParamsSofastFringe
import opencsp.common.lib.tool.file_tools as ft


class TestParamsSofastFringe(unittest.TestCase):
    def test_save_load_hdf(self):
        # Define save dir
        dir_save = join(dirname(__file__), "data/output/ParamsSofastFringe")
        ft.create_directories_if_necessary(dir_save)
        file_save = join(dir_save, "params_sofast_fringe.h5")

        # Instantiate with defaults
        params = ParamsSofastFringe()

        # Test save
        params.save_to_hdf(file_save)

        # Test load
        ParamsSofastFringe.load_from_hdf(file_save)


if __name__ == "__main__":
    unittest.main()
