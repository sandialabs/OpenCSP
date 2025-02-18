import os
import sys
import time
import unittest
import unittest.mock

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.time_date_tools as tdt

# setting path
sys.path.append(os.path.join(orp.opencsp_code_dir(), ".."))
import contrib.scripts.FileCache as fc  # nopep8


class test_FileCache(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "FileCache")
        self.out_dir = os.path.join(path, "data", "output", "FileCache")
        ft.create_directories_if_necessary(self.out_dir)

    def _delay_1_second(self):
        """sleeps up to 1 second so that the file modification time looks different"""
        ts1 = tdt.current_time_string_forfile()
        while ts1 == tdt.current_time_string_forfile():
            time.sleep(0.05)

    def test_file_changed(self):
        outfile = "changing_file.txt"

        ft.write_text_file(outfile, self.out_dir, outfile, [])
        fc1 = fc.FileCache.for_file("", self.out_dir, outfile + ".txt")
        self._delay_1_second()
        ft.write_text_file(outfile, self.out_dir, outfile, [])
        fc2 = fc.FileCache.for_file("", self.out_dir, outfile + ".txt")

        self.assertNotEqual(fc1, fc2)

    def test_file_unchanged(self):
        outfile = "static_file.txt"

        ft.write_text_file(outfile, self.out_dir, outfile, [])
        fc1 = fc.FileCache.for_file("", self.out_dir, outfile + ".txt")
        self._delay_1_second()
        fc2 = fc.FileCache.for_file("", self.out_dir, outfile + ".txt")

        self.assertEqual(fc1, fc2)


if __name__ == "__main__":
    unittest.main()
