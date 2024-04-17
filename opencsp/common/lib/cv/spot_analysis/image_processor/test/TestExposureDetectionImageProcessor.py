import numpy as np
import os
import unittest
from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.ExposureDetectionImageProcessor import (
    ExposureDetectionImageProcessor,
)

import opencsp.common.lib.tool.file_tools as ft


class TestExposureDetectionImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "ExposureDetectionImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "ExposureDetectionImageProcessor")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

        self.simple_image = np.array([100, 150, 200, 255])
        self.cacheable_simple_image = CacheableImage(self.simple_image, source_path="test_under_exposure")
        self.operable = SpotAnalysisOperable(self.cacheable_simple_image)

    def test_under_exposure_limit(self):
        # raises
        processor = ExposureDetectionImageProcessor(under_exposure_limit=0)
        processor._raise_on_error = True
        with self.assertRaises(RuntimeError) as ex:
            processor.process_image(self.operable)
            self.assertIn("under exposed", repr(ex))

        # raises
        processor = ExposureDetectionImageProcessor(under_exposure_limit=0.74)
        processor._raise_on_error = True
        with self.assertRaises(RuntimeError) as ex:
            processor.process_image(self.operable)
            self.assertIn("under exposed", repr(ex))

        # passes
        processor = ExposureDetectionImageProcessor(under_exposure_limit=0.75)
        processor._raise_on_error = True
        processor.process_image(self.operable)

    def test_under_exposure_threshold(self):
        # passes
        processor = ExposureDetectionImageProcessor(under_exposure_threshold=0)
        processor._raise_on_error = True
        processor.process_image(self.operable)

        # passes
        processor = ExposureDetectionImageProcessor(under_exposure_threshold=255)
        processor._raise_on_error = True
        processor.process_image(self.operable)

        # raises
        processor = ExposureDetectionImageProcessor(under_exposure_threshold=256)
        processor._raise_on_error = True
        with self.assertRaises(RuntimeError) as ex:
            processor.process_image(self.operable)
            self.assertIn("under exposed", repr(ex))

    def test_over_exposure_limit(self):
        # raises
        processor = ExposureDetectionImageProcessor(over_exposure_limit=0)
        processor._raise_on_error = True
        with self.assertRaises(RuntimeError) as ex:
            processor.process_image(self.operable)
            self.assertIn("over exposed", repr(ex))

        # raises
        processor = ExposureDetectionImageProcessor(over_exposure_limit=0.24)
        processor._raise_on_error = True
        with self.assertRaises(RuntimeError) as ex:
            processor.process_image(self.operable)
            self.assertIn("over exposed", repr(ex))

        # passes
        processor = ExposureDetectionImageProcessor(over_exposure_limit=0.25)
        processor._raise_on_error = True
        processor.process_image(self.operable)

    def test_max_pixel_value(self):
        # raises
        processor = ExposureDetectionImageProcessor(max_pixel_value=0)
        processor._raise_on_error = True
        with self.assertRaises(RuntimeError) as ex:
            processor.process_image(self.operable)
            self.assertIn("over exposed", repr(ex))

        # raises
        processor = ExposureDetectionImageProcessor(max_pixel_value=100)
        processor._raise_on_error = True
        with self.assertRaises(RuntimeError) as ex:
            processor.process_image(self.operable)
            self.assertIn("over exposed", repr(ex))

        # passes
        processor = ExposureDetectionImageProcessor(max_pixel_value=101)
        processor._raise_on_error = True
        processor.process_image(self.operable)


if __name__ == '__main__':
    unittest.main()
