import os
import subprocess
import sys
import time
import unittest

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp


class TestFileTools(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "file_tools")
        self.out_dir = os.path.join(path, "data", "output", "file_tools")

    def tearDown(self) -> None:
        pass

    def test_files_in_directory(self):
        files_name_ext = ft.files_in_directory(self.data_dir)
        expected = [".dotfile", "a.a", "b.b", "d"]
        self.assertListEqual(expected, files_name_ext)

    def test_files_in_directory_files_only(self):
        files_name_ext = ft.files_in_directory(self.data_dir, files_only=True)
        expected = [".dotfile", "a.a", "b.b"]
        self.assertListEqual(expected, files_name_ext)

    def test_files_in_directory_recursive(self):
        files_name_ext = ft.files_in_directory(self.data_dir, recursive=True)
        files_name_ext = [f.replace("\\", "/") for f in files_name_ext]
        expected = [".dotfile", "a.a", "b.b", "d", "d/c.c", "d/e", "d/e/f.f"]
        self.assertListEqual(expected, files_name_ext)

    def test_files_in_directory_recursive_files_only(self):
        files_name_ext = ft.files_in_directory(self.data_dir, recursive=True, files_only=True)
        files_name_ext = [f.replace("\\", "/") for f in files_name_ext]
        expected = [".dotfile", "a.a", "b.b", "d/c.c", "d/e/f.f"]
        self.assertListEqual(expected, files_name_ext)

    def test_files_in_directory_by_extension(self):
        files_name_ext = ft.files_in_directory_by_extension(self.data_dir, [".a", ".b"])
        expected = {".a": ["a.a"], ".b": ["b.b"]}
        self.assertDictEqual(expected, files_name_ext)

    def test_files_in_directory_by_extension_case_sensity(self):
        files_name_ext = ft.files_in_directory_by_extension(self.data_dir, [".a", ".B"], case_sensitive=True)
        expected = {".a": ["a.a"], ".B": []}
        self.assertDictEqual(expected, files_name_ext)

    def test_binary_count_items_in_directory(self):
        test_dir = os.path.join(self.out_dir, "test_binary_count_items_in_directory")
        ft.create_directories_if_necessary(test_dir)
        ft.delete_files_in_directory(test_dir, "*.tmp")

        cnt = ft.binary_count_items_in_directory(test_dir, "%06d.tmp", start=0)
        self.assertEqual(cnt, 0)
        for i in range(100):
            ft.create_file(os.path.join(test_dir, "%06d.tmp" % i))
            cnt = ft.binary_count_items_in_directory(test_dir, "%06d.tmp", start=0)
            self.assertEqual(cnt, i + 1)

    def test_copy(self):
        test_dir = os.path.join(self.out_dir, "test_copy")
        ft.create_directories_if_necessary(test_dir)
        ft.delete_files_in_directory(test_dir, "*.tmp")

        ft.create_file(test_dir + "/copy_a.tmp")
        ft.copy_file(test_dir + "/copy_a.tmp", test_dir, "copy_b.tmp")
        self.assertTrue(ft.file_exists(test_dir + "/copy_b.tmp"))

        # this should be cought by copy_file() and throw an error
        ft.create_file(test_dir + "/copy_c.tmp")
        with self.assertRaises(FileExistsError):
            ft.copy_file(test_dir + "/copy_c.tmp", test_dir)
        with self.assertRaises(FileExistsError):
            ft.copy_file(test_dir + "/copy_c.tmp", test_dir, "copy_c.tmp")

    def test_rename(self):
        test_dir = os.path.join(self.out_dir, "test_rename")
        ft.create_directories_if_necessary(test_dir)
        ft.delete_files_in_directory(test_dir, "*.tmp")

        ft.create_file(test_dir + "/rename_a.tmp")
        ft.rename_file(test_dir + "/rename_a.tmp", test_dir + "/rename_b.tmp")
        self.assertTrue(ft.file_exists(test_dir + "/rename_b.tmp"))

        # this should be cought by rename_file() and not throw an error
        ft.create_file(test_dir + "/rename_c.tmp")
        ft.rename_file(test_dir + "/rename_c.tmp", test_dir + "/rename_c.tmp")

    def test_copy_and_delete(self):
        test_dir = os.path.join(self.out_dir, "test_copy_and_delete")
        ft.create_directories_if_necessary(test_dir)
        ft.delete_files_in_directory(test_dir, "*.tmp")

        ft.create_file(test_dir + "/copy_and_delete_a.tmp")
        ft.copy_and_delete_file(test_dir + "/copy_and_delete_a.tmp", test_dir + "/copy_and_delete_b.tmp")
        self.assertFalse(ft.file_exists(test_dir + "/copy_and_delete_a.tmp"))
        self.assertTrue(ft.file_exists(test_dir + "/copy_and_delete_b.tmp"))

        ft.create_file(test_dir + "/copy_and_delete_c.tmp")
        ft.copy_and_delete_file(test_dir + "/copy_and_delete_c.tmp", test_dir + "/copy_and_delete_d.tmp")
        self.assertFalse(ft.file_exists(test_dir + "/copy_and_delete_c.tmp"))
        self.assertTrue(ft.file_exists(test_dir + "/copy_and_delete_d.tmp"))

        # don't delete the source file if the source and destination are the same file
        ft.create_file(test_dir + "/copy_and_delete_e.tmp")
        ft.copy_and_delete_file(test_dir + "/copy_and_delete_e.tmp", test_dir + "/copy_and_delete_e.tmp")
        self.assertTrue(ft.file_exists(test_dir + "/copy_and_delete_e.tmp"))

    @unittest.skipIf('nt' not in os.name, "Testing slash normalization and path extension on windows only")
    def test_norm_path(self):
        actual = ft.norm_path("a/b/c/d")
        expected = "a\\b\\c\\d"
        self.assertEqual(actual, expected, "slashes not normalized")

        actual = ft.norm_path(
            "thisisaverylongpathnameitneedstobeatleast260characterslongbeforethelongpathnamenormalizationkicksinLoremipsumdolorsitametconsecteturadipiscingelitseddoeiusmodtemporincididuntutlaboreetdoloremagnaaliquaUtenimadminimveniamquisnostrudexercitationullamcolaborisnis"
        )
        normalized_long_path = "\\\\?\\thisisaverylongpathnameitneedstobeatleast260characterslongbeforethelongpathnamenormalizationkicksinLoremipsumdolorsitametconsecteturadipiscingelitseddoeiusmodtemporincididuntutlaboreetdoloremagnaaliquaUtenimadminimveniamquisnostrudexercitationullamcolaborisnis"
        expected = normalized_long_path
        self.assertEqual(actual, expected, "long path names not normalized")

        actual = ft.norm_path(normalized_long_path)
        expected = normalized_long_path
        self.assertEqual(actual, expected, "long path names that are already normalized are being modified")

        actual = ft.norm_path(
            "thisisaverylongpathnameitneedstobeatleast260characterslongbeforethelongpathnamenormalizationkicksin/LoremipsumdolorsitametconsecteturadipiscingelitseddoeiusmodtemporincididuntutlaboreetdoloremagnaaliquaUtenimadminimveniamquisnostrudexercitationullamcolaborisnis"
        )
        normalized_long_path_with_slash = "\\\\?\\thisisaverylongpathnameitneedstobeatleast260characterslongbeforethelongpathnamenormalizationkicksin\\LoremipsumdolorsitametconsecteturadipiscingelitseddoeiusmodtemporincididuntutlaboreetdoloremagnaaliquaUtenimadminimveniamquisnostrudexercitationullamcolaborisnis"
        expected = normalized_long_path_with_slash
        self.assertEqual(actual, expected, "long path names with a slash aren't being normalized")

    def test_join(self):
        actual = ft.join("a", "b/c", "d/e.txt")
        expected = ft.norm_path("a/b/c/d/e.txt")
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
