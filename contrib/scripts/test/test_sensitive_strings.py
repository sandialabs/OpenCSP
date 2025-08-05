import os
import shutil
import sys
import unittest
import unittest.mock

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft

# setting path
sys.path.append(os.path.join(orp.opencsp_code_dir(), ".."))
import contrib.scripts.sensitive_strings as ss  # nopep8


class test_sensitive_strings(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "sensitive_strings")
        self.out_dir = os.path.join(path, "data", "output", "sensitive_strings")
        ft.create_directories_if_necessary(self.out_dir)

        allowed_binaries_dir_in = os.path.join(self.data_dir, "per_test_allowed_binaries")
        self.allowed_binaries_dir_tmp = os.path.join(self.out_dir, "per_test_allowed_binaries")
        ft.create_directories_if_necessary(self.allowed_binaries_dir_tmp)
        for f in ft.files_in_directory_by_extension(allowed_binaries_dir_in, ["csv"])["csv"]:
            ft.copy_file(os.path.join(allowed_binaries_dir_in, f), self.allowed_binaries_dir_tmp)

        self.root_search_dir = os.path.join(self.data_dir, "root_search_dir")
        self.ss_dir = os.path.join(self.data_dir, "per_test_sensitive_strings")
        self.all_binaries = os.path.join(self.allowed_binaries_dir_tmp, "all_binaries.csv")
        self.no_binaries = os.path.join(self.allowed_binaries_dir_tmp, "no_binaries.csv")
        self.single_binary = os.path.join(self.allowed_binaries_dir_tmp, "single_binary.csv")
        self.single_expected_not_found_binary = os.path.join(
            self.allowed_binaries_dir_tmp, "single_expected_not_found_binary.csv"
        )

    def tearDown(self):
        ft.delete_files_in_directory(self.allowed_binaries_dir_tmp, "*.csv")

    def test_no_matches(self):
        sensitive_strings_csv = os.path.join(self.ss_dir, "no_matches.csv")
        searcher = ss.SensitiveStringsSearcher(self.root_search_dir, sensitive_strings_csv, self.all_binaries)
        searcher.git_files_only = False
        self.assertEqual(searcher.search_files(), 0)

    def test_single_matcher(self):
        # based on file name
        sensitive_strings_csv = os.path.join(self.ss_dir, "test_single_matcher.csv")
        searcher = ss.SensitiveStringsSearcher(self.root_search_dir, sensitive_strings_csv, self.all_binaries)
        searcher.git_files_only = False
        self.assertEqual(searcher.search_files(), 1)

        # based on file content
        sensitive_strings_csv = os.path.join(self.ss_dir, "test_single_matcher_content.csv")
        searcher = ss.SensitiveStringsSearcher(self.root_search_dir, sensitive_strings_csv, self.all_binaries)
        searcher.git_files_only = False
        self.assertEqual(searcher.search_files(), 1)

    def test_directory_matcher(self):
        sensitive_strings_csv = os.path.join(self.ss_dir, "test_directory_matcher.csv")
        searcher = ss.SensitiveStringsSearcher(self.root_search_dir, sensitive_strings_csv, self.all_binaries)
        searcher.git_files_only = False
        self.assertEqual(searcher.search_files(), 1)

    def test_all_matches(self):
        sensitive_strings_csv = os.path.join(self.ss_dir, "test_all_matches.csv")
        searcher = ss.SensitiveStringsSearcher(self.root_search_dir, sensitive_strings_csv, self.no_binaries)
        searcher.git_files_only = False
        # 6 matches:
        #   files:   a.txt, b/b.txt, c/d/e.txt
        #   images:  c/img1.png, c/img2.jpg
        #   hdf5:    f.h5/f
        self.assertEqual(searcher.search_files(), 6)

    def test_single_unknown_binary(self):
        sensitive_strings_csv = os.path.join(self.ss_dir, "no_matches.csv")
        searcher = ss.SensitiveStringsSearcher(self.root_search_dir, sensitive_strings_csv, self.single_binary)
        searcher.git_files_only = False
        self.assertEqual(searcher.search_files(), 1)

    def test_single_expected_not_found_binary(self):
        sensitive_strings_csv = os.path.join(self.ss_dir, "no_matches.csv")
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.single_expected_not_found_binary
        )
        searcher.git_files_only = False
        # 2 unknown binaries, and 1 expected not found
        self.assertEqual(searcher.search_files(), 3)

    def test_hdf5_match(self):
        sensitive_strings_csv = os.path.join(self.ss_dir, "h5_match.csv")
        searcher = ss.SensitiveStringsSearcher(self.root_search_dir, sensitive_strings_csv, self.all_binaries)
        searcher.git_files_only = False
        # 2 unknown binaries, and 1 expected not found
        self.assertEqual(searcher.search_files(), 1)


if __name__ == "__main__":
    unittest.main()
