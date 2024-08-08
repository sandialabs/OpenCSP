import os
import unittest
import unittest.mock

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft


@unittest.mock.patch.dict(
    os.environ, {"OPENCSP_SETTINGS_DIRS": "None"}
)  # don't run unit tests with user specific settings
class test_opencsp_root_path(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_settings_file = os.path.join(os.path.expanduser("~"), ".opencsp", "settings.json")
        cls.tmp_settings_contents = (
            '{ "opencsp_root_path": { "example_data_dir": "e/f", "scratch_dir": "s/t", "scratch_name": "u" } }'
        )
        cls.did_create_settings_file = False
        if not ft.file_exists(cls.tmp_settings_file):
            path, _, _ = ft.path_components(cls.tmp_settings_file)
            if ft.directory_exists(path):
                open(cls.tmp_settings_file, 'w').write(cls.tmp_settings_contents)
                cls.did_create_settings_file = True

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.did_create_settings_file:
            if os.path.exists(cls.tmp_settings_file):
                os.unlink(cls.tmp_settings_file)

    def get_opencsp_path(self):
        # Try to determine the opencsp path via another means. Note that this
        # could be very wrong and that someone smarter will have to write this
        # test.
        self_path = os.path.dirname(__file__)
        self_path_dirs = os.path.normpath(self_path).split(os.sep)

        if "opencsp" not in self_path_dirs:
            self.skipTest(f"Can't find directory 'opencsp' in {self_path}.")
        opencsp_idx = len(self_path_dirs) - list(reversed(self_path_dirs)).index("opencsp")
        root_to_opencsp_dirs = self_path_dirs[:opencsp_idx]
        return os.path.sep.join(root_to_opencsp_dirs)

    def test_opencsp_code_dir(self):
        opencsp_path = self.get_opencsp_path()

        expected = os.path.normpath(opencsp_path).lower()
        actual = os.path.normpath(orp.opencsp_code_dir()).lower()
        self.assertEqual(expected, actual)

    def test_opencsp_doc_dir(self):
        """Just test that the opencsp_doc_dir() method works. TODO actually test the returned value."""
        self.assertIn("doc", orp.opencsp_doc_dir())

    def test_opencsp_example_dir(self):
        """Just test that the opencsp_example_dir() method works. TODO actually test the returned value."""
        self.assertIn("example", orp.opencsp_example_dir())

    def test_opencsp_scratch_dir(self):
        """Just test that the opencsp_scratch_dir() method works. TODO actually test the returned value."""
        self.assertIn("scratch", orp.opencsp_scratch_dir())

    def test_opencsp_cache_dir(self):
        """Just test that the opencsp_cache_dir() method works. TODO actually test the returned value."""
        self.assertIn("cache", orp.opencsp_cache_dir())

    def test_opencsp_temporary_dir(self):
        """Just test that the opencsp_temporary_dir() method works. TODO actually test the returned value."""
        self.assertTrue(("temp" in orp.opencsp_temporary_dir()) or ("tmp" in orp.opencsp_temporary_dir()))

    @unittest.skip("Can't get this test to work. Maybe someone smarter than me can make it work? :(")
    @unittest.mock.patch.dict(os.environ, {"OPENCSP_SETTINGS_DIRS": "~/.opencsp/"})
    def test_settings_file(self):
        """Creates a temporary "settings.json" file if one doesn't already
        exist and populates it with test values."""
        if os.path.exists(self.tmp_settings_file):
            file_contents = open(self.tmp_settings_file, 'r').read()
            if file_contents.strip() != self.tmp_settings_contents:
                self.skipTest(
                    f"File '{self.tmp_settings_file}' already exists. Not replacing file with a testing file."
                )

        self.assertEqual("e/f", orp.opencsp_data_example_dir())
        self.assertEqual("s/t/u", orp.opencsp_scratch_dir())


if __name__ == '__main__':
    unittest.main()
