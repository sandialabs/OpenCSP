import numpy as np
import os
import random
import shutil
import unittest
import zipfile

from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis
from opencsp.common.lib.cv.spot_analysis.image_processor import ViewFalseColorImageProcessor
from contrib.common.lib.cv.spot_analysis.image_processor import PowerpointImageProcessor

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class TestPowerpointImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _, _ = ft.path_components(__file__)
        cls.data_dir = os.path.join(path, "data", "input", "PowerpointImageProcessor")
        cls.out_dir = os.path.join(path, "data", "output", "PowerpointImageProcessor")
        ft.create_directories_if_necessary(cls.data_dir)
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, "*.pptx")

    def setUp(self) -> None:
        # delete the previous results
        # ft.delete_file(self._get_default_ppt_file_path_name_ext()) # done in setUpClass
        self._delete_unzip_dir()

        # prepare the standard image processors
        self.false_color_processor = ViewFalseColorImageProcessor()
        self.powerpoint_processor = PowerpointImageProcessor(
            self.out_dir, self._testMethodName, processors_per_slide=[[self.false_color_processor]]
        )
        self.image_processors = [self.false_color_processor, self.powerpoint_processor]
        self.spot_analysis = SpotAnalysis(self._testMethodName, image_processors=self.image_processors)

        # prepare the example image
        large_grayscale_image = np.arange(1020, dtype=np.int16)
        large_grayscale_image = np.expand_dims(large_grayscale_image, axis=1)
        large_grayscale_image = np.broadcast_to(large_grayscale_image, (1020, 1020))
        self.large_grayscale_image = large_grayscale_image

    def _get_default_ppt_file_path_name(self) -> str:
        return ft.join(self.out_dir, self._testMethodName)

    def _get_default_ppt_file_path_name_ext(self) -> str:
        return ft.join(self.out_dir, self._testMethodName + ".pptx")

    def _get_unzip_dir(self, ppt_file_path_name_ext: str = None) -> str:
        if ppt_file_path_name_ext is None:
            ppt_file_path_name_ext = self._get_default_ppt_file_path_name_ext()
        ppt_file_path, ppt_file_name, _ = ft.path_components(ppt_file_path_name_ext)
        ppt_file_path_name = ft.join(ppt_file_path, ppt_file_name)
        return ppt_file_path_name

    def _delete_unzip_dir(self, ppt_file_path_name_ext: str = None):
        unzip_dir = self._get_unzip_dir(ppt_file_path_name_ext)
        if ft.directory_exists(unzip_dir):
            shutil.rmtree(unzip_dir)

    def _unzip_ppt(self, ppt_file_path_name_ext: str = None) -> str:
        if ppt_file_path_name_ext is None:
            ppt_file_path_name_ext = self._get_default_ppt_file_path_name_ext()
        unzip_dir = self._get_unzip_dir(ppt_file_path_name_ext)
        self.assertFalse(ft.directory_exists(unzip_dir))

        with zipfile.ZipFile(ppt_file_path_name_ext, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)

        return unzip_dir

    def _verify_ppt_has_images(self, ppt_extraction_dir: str, nimages: int):
        images_dir = ft.join(ppt_extraction_dir, "ppt/media")
        self.assertTrue(ft.directory_exists(images_dir))
        image_file_names_exts = it.image_files_in_directory(images_dir)
        self.assertEqual(
            len(image_file_names_exts),
            nimages,
            f"Expected powerpoint {ppt_extraction_dir} to contain {nimages} images for {self._testMethodName}",
        )

    def test_no_processors_specified(self):
        # build the spot analysis pipeline
        self.powerpoint_processor = PowerpointImageProcessor(self.out_dir, self._testMethodName)
        self.image_processors = [self.false_color_processor, self.powerpoint_processor]
        self.spot_analysis = SpotAnalysis(self._testMethodName, image_processors=self.image_processors)
        self.assertFalse(ft.file_exists(self._get_default_ppt_file_path_name_ext()))

        # evaluate for a single image
        self.spot_analysis.set_primary_images([self.large_grayscale_image])
        for result in self.spot_analysis:
            pass

        # verify the results in the generated powerpoint
        self.assertTrue(ft.file_exists(self._get_default_ppt_file_path_name_ext()))
        self._verify_ppt_has_images(self._unzip_ppt(), 1)

    def test_bad_save_dir(self):
        with self.assertRaises(FileNotFoundError):
            self.powerpoint_processor = PowerpointImageProcessor(ft.join(self.out_dir, "DNE"), "test_bad_save_dir")

    def test_bad_processors_per_slide(self):
        with self.assertRaises(ValueError):
            self.powerpoint_processor = PowerpointImageProcessor(
                self.out_dir, self._testMethodName, processors_per_slide="not a list"
            )
        with self.assertRaises(ValueError):
            self.powerpoint_processor = PowerpointImageProcessor(
                self.out_dir, self._testMethodName, processors_per_slide=[self.false_color_processor]
            )
        with self.assertRaises(ValueError):
            self.powerpoint_processor = PowerpointImageProcessor(
                self.out_dir, self._testMethodName, processors_per_slide=["not a list"]
            )
        with self.assertRaises(ValueError):
            self.powerpoint_processor = PowerpointImageProcessor(
                self.out_dir, self._testMethodName, processors_per_slide=[["not an image processor"]]
            )

    def test_single_image(self):
        self.assertFalse(ft.file_exists(self._get_default_ppt_file_path_name_ext()))

        # evaluate for a single image
        self.spot_analysis.set_primary_images([self.large_grayscale_image])
        for result in self.spot_analysis:
            pass

        # verify the results in the generated powerpoint
        self.assertTrue(ft.file_exists(self._get_default_ppt_file_path_name_ext()))
        self._verify_ppt_has_images(self._unzip_ppt(), 1)

    def test_several_images(self):
        self.assertFalse(ft.file_exists(self._get_default_ppt_file_path_name_ext()))

        # evaluate for a random number of images
        nimages = random.randint(2, 10)
        images = [self.large_grayscale_image + (i * 20) for i in range(nimages)]
        self.spot_analysis.set_primary_images(images)
        for result in self.spot_analysis:
            pass

        # verify the results in the generated powerpoint
        self.assertTrue(ft.file_exists(self._get_default_ppt_file_path_name_ext()))
        self._verify_ppt_has_images(self._unzip_ppt(), nimages)

    # def test_panick_save(self):
    #     """Test that results available are saved in the powerpoint in the case
    #     that the next image fails to load."""
    #     TODO


if __name__ == "__main__":
    unittest.main()
