import random
import os
import sys
import unittest
import unittest.mock

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft

# setting path
sys.path.append(os.path.join(orp.opencsp_code_dir(), ".."))
import contrib.scripts.FileFingerprint as ff  # nopep8


class test_FileFingerprint(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "FileFingerprint")
        self.out_dir = os.path.join(path, "data", "output", "FileFingerprint")
        ft.create_directories_if_necessary(self.out_dir)

    def test_equal(self):
        d1 = "equal1"
        d2 = "equal2"
        f1 = "equal_file"
        f2 = "equal_file"
        contents = "%0.10f" % random.Random().random()

        ft.write_text_file(f1, f"{self.out_dir}/{d1}", f1, [contents], error_if_dir_not_exist=False)
        ft.write_text_file(f2, f"{self.out_dir}/{d2}", f2, [contents], error_if_dir_not_exist=False)
        ff1 = ff.FileFingerprint.for_file(f"{self.out_dir}/{d1}", "", f1 + ".txt")
        ff2 = ff.FileFingerprint.for_file(f"{self.out_dir}/{d2}", "", f2 + ".txt")

        self.assertEqual(ff1, ff2)

    def test_not_equal_relpath(self):
        d1 = "not_equal_relpath1"
        d2 = "not_equal_relpath2"
        f1 = "equal_file"
        f2 = "equal_file"
        contents = "%0.10f" % random.Random().random()

        ft.write_text_file(f1, f"{self.out_dir}/{d1}", f1, [contents], error_if_dir_not_exist=False)
        ft.write_text_file(f2, f"{self.out_dir}/{d2}", f2, [contents], error_if_dir_not_exist=False)
        ff1 = ff.FileFingerprint.for_file(self.out_dir, d1, f1 + ".txt")
        ff2 = ff.FileFingerprint.for_file(self.out_dir, d2, f2 + ".txt")

        self.assertNotEqual(ff1, ff2)

    def test_not_equal_filename(self):
        d1 = "not_equal_filename1"
        d2 = "not_equal_filename2"
        f1 = "equal_file1"
        f2 = "equal_file2"
        contents = "%0.10f" % random.Random().random()

        ft.write_text_file(f1, f"{self.out_dir}/{d1}", f1, [contents], error_if_dir_not_exist=False)
        ft.write_text_file(f2, f"{self.out_dir}/{d2}", f2, [contents], error_if_dir_not_exist=False)
        ff1 = ff.FileFingerprint.for_file(f"{self.out_dir}/{d1}", "", f1 + ".txt")
        ff2 = ff.FileFingerprint.for_file(f"{self.out_dir}/{d2}", "", f2 + ".txt")

        self.assertNotEqual(ff1, ff2)

    def test_not_equal_hash(self):
        d1 = "not_equal_hash1"
        d2 = "not_equal_hash2"
        f1 = "not_equal1"
        f2 = "not_equal2"
        contents = "%0.10f" % random.Random().random()
        contents1 = contents + " "
        contents2 = " " + contents

        ft.write_text_file(f1, f"{self.out_dir}/{d1}", f1, [contents1], error_if_dir_not_exist=False)
        ft.write_text_file(f2, f"{self.out_dir}/{d2}", f2, [contents2], error_if_dir_not_exist=False)
        ff1 = ff.FileFingerprint.for_file(f"{self.out_dir}/{d1}", "", f1 + ".txt")
        ff2 = ff.FileFingerprint.for_file(f"{self.out_dir}/{d2}", "", f2 + ".txt")

        self.assertNotEqual(ff1, ff2)


if __name__ == "__main__":
    unittest.main()
