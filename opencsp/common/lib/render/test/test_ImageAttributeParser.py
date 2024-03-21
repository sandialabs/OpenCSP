import datetime
import json
import os
import unittest
import unittest.mock
import pytest

import opencsp.common.lib.render.ImageAttributeParser as iap
import opencsp.common.lib.tool.file_tools as ft


class test_ImageAttributeParser(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "ImageAttributeParser")
        self.out_dir = os.path.join(path, "data", "output", "ImageAttributeParser")
        self.img_file = os.path.join(self.out_dir, f"nonexistant_image_{self._testMethodName}.png")
        attr_file_src = os.path.join(self.data_dir, f"nonexistant_image_{self._testMethodName}.txt")
        self.attr_file = os.path.join(self.out_dir, f"nonexistant_image_{self._testMethodName}.txt")

        ft.create_directories_if_necessary(self.out_dir)

        # Some tests have attribute files that need to be copied to self.out_dir.
        # Copy the attribute files, if they exist.
        if not ft.file_exists(self.attr_file) and ft.file_exists(attr_file_src):
            ft.copy_file(attr_file_src, self.out_dir)

    def test_no_attrfile(self):
        """Contructor succeeds even without attributes file"""
        parser = iap.ImageAttributeParser(self.img_file)
        self.assertEqual(self.img_file, parser.current_image_source)
        self.assertEqual(None, parser.original_image_source)
        self.assertEqual(None, parser.date_collected)
        self.assertEqual(None, parser.experiment_name)
        self.assertEqual(None, parser.notes)

    def test_has_contents(self):
        parser = iap.ImageAttributeParser()
        self.assertEqual(False, parser.has_contents())

        parser = iap.ImageAttributeParser(current_image_source=self.img_file)
        self.assertEqual(True, parser.has_contents())
        parser = iap.ImageAttributeParser(original_image_source=self.img_file)
        self.assertEqual(True, parser.has_contents())
        parser = iap.ImageAttributeParser(date_collected=datetime.datetime.now())
        self.assertEqual(True, parser.has_contents())
        parser = iap.ImageAttributeParser(experiment_name="")
        self.assertEqual(True, parser.has_contents())
        parser = iap.ImageAttributeParser(notes="")
        self.assertEqual(True, parser.has_contents())

    @pytest.mark.skip("See https://github.com/sandialabs/OpenCSP/issues/3")
    def test_with_attrfile(self):
        """Load all values from the associated attributes file. Use the new current_image_source value."""
        parser = iap.ImageAttributeParser(current_image_source=self.img_file)

        # Not 'a' as is in the attributes file associated with self.img_file,
        # but rather the current image source path self.img_file since it is
        # non-None in the constructor.
        self.assertEqual(self.img_file, parser.current_image_source)
        # The rest of these values should be set by the attributes file, since
        # they are not given in the ImageAttributeParser constructor.
        self.assertEqual(datetime.datetime.fromisoformat('2024-02-17'), parser.date_collected)
        self.assertEqual('c', parser.experiment_name)
        self.assertEqual('d', parser.notes)

        # Should raise an error when trying to replace the original_image_source.
        with self.assertRaises(ValueError):
            iap.ImageAttributeParser(current_image_source=self.img_file, original_image_source='z')

    def test_partial_attrfile(self):
        """Constructor pulls in non-None values from existing attributes file"""
        pass


if __name__ == '__main__':
    unittest.main()
