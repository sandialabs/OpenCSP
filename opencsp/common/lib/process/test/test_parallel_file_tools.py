from concurrent import futures
import os
import time
import unittest

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.process.parallel_file_tools as pft
import opencsp.common.lib.tool.file_tools as ft


class TestSubprocess(unittest.TestCase):
    path = os.path.join("common", "lib", "process", "test", "data", "output", "parallel_file_tools")

    def setUp(self):
        super().setUp()
        self.mypath = os.path.join(self.__class__.path, self._testMethodName)

    def _sleep_and_create(self, sleep_time: float, file_name: str):
        time.sleep(sleep_time)
        ft.create_file(file_name)

    def test_wait_on_files_no_files(self):
        tstart = time.time()
        pft.wait_on_files([], timeout=10)
        self.assertLess(time.time() - tstart, 10)

    def test_wait_on_files_timeout(self):
        with self.assertRaises(TimeoutError):
            tstart = time.time()
            pft.wait_on_files(["this_file_does_not_exist"], 0.1)

    def test_wait_on_files_pre_existing(self):
        # create the files to be tested
        files = [os.path.join(self.mypath, f"{i}.tmp") for i in range(3)]
        ft.create_directories_if_necessary(self.mypath)
        for path_name_ext in files:
            ft.create_file(path_name_ext, error_on_exists=False)

        # verify that wait_on_files returns these files
        found_files = pft.wait_on_files(files, timeout=1)

        self.assertEqual(files, found_files)

    def test_wait_on_files_returns_alternates(self):
        # create the files to be tested
        files, alternates = [], []
        alternates_dict: dict[str, list[str]] = {}
        for i in range(3):
            files.append(os.path.join(self.mypath, f"{i}.tmp"))
            alternates.append(os.path.join(self.mypath, f"{i+3}.tmp"))
            alternates_dict[files[-1]] = [alternates[-1]]
            alternates_path, _, _ = ft.path_components(alternates[-1])
            ft.create_directories_if_necessary(alternates_path)
            ft.create_file(alternates[-1], error_on_exists=False)

        # verify that wait_on_files returns the alternates files
        found_files = pft.wait_on_files(files, timeout=1, alternates=alternates_dict)

        self.assertEqual(alternates, found_files)

    def test_wait_on_files_create_file_after_delay(self):
        path_name_ext = os.path.join(self.mypath, "0.tmp")
        ft.create_directories_if_necessary(self.mypath)
        ft.delete_file(path_name_ext, error_on_not_exists=False)

        # spawn a new thread that will create the file after a 1 second delay
        executor = futures.ThreadPoolExecutor()
        executor.submit(self._sleep_and_create, 1.0, path_name_ext)

        # verify that the file doesn't exist yet
        self.assertFalse(
            ft.file_exists(path_name_ext),
            "File should not exist yet! (did the file creation thread run before this check?)",
        )

        # wait for the file to exist, and verify it exists just to be sure
        pft.wait_on_files([path_name_ext], timeout=10)
        self.assertTrue(ft.file_exists(path_name_ext))


if __name__ == "__main__":
    unittest.main()
