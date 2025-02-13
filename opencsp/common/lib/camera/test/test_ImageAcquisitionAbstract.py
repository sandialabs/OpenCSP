import os

import unittest

import opencsp.app.sofast.test.ImageAcquisition_no_camera as ianc
import opencsp.common.lib.camera.ImageAcquisitionAbstract as iaa
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft


class test_ImageAcquisitionAbstract(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "ImageAcquisitionAbstract")
        self.out_dir = os.path.join(path, "data", "output", "ImageAcquisitionAbstract")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def tearDown(self):
        with et.ignored(Exception):
            iaa.ImageAcquisitionAbstract.instance().close()

    def test_cam_options(self):
        cam_options = iaa.ImageAcquisitionAbstract.cam_options()

        for k in cam_options:
            self.assertIsInstance(k, str)
            self.assertTrue(issubclass(cam_options[k], iaa.ImageAcquisitionAbstract))

    def test_set_image_acquisition(self):
        self.assertIsNone(iaa.ImageAcquisitionAbstract.instance())

        # Create a mock ImageAcquisition object
        image_acquisition = ianc.ImageAcquisition()

        # Test that the instance was set
        self.assertEqual(image_acquisition, iaa.ImageAcquisitionAbstract.instance())

        # Test un-setting the image_acquisition object
        image_acquisition.close()
        self.assertIsNone(iaa.ImageAcquisitionAbstract.instance())

    def test_on_close(self):
        global close_count
        close_count = 0

        def close_count_inc(image_acquisition):
            global close_count
            close_count += 1

        # Create a mock ImageAcquisition object with single on_close callback
        image_acquisition = ianc.ImageAcquisition()
        image_acquisition.on_close.append(close_count_inc)
        image_acquisition.close()
        self.assertEqual(close_count, 1)

        # Create a mock ImageAcquisition object with multiple on_close callback
        image_acquisition = ianc.ImageAcquisition()
        image_acquisition.on_close.append(close_count_inc)
        image_acquisition.on_close.append(close_count_inc)
        image_acquisition.close()
        self.assertEqual(close_count, 3)

        # Create a mock ImageAcquisition object without an on_close callback
        image_acquisition = ianc.ImageAcquisition()
        image_acquisition.close()
        self.assertEqual(close_count, 3)

    # # TODO
    # def test_calibrate_exposure(self):
    #     pass

    def test_get_set_exposure(self):
        # basic test: can we get and set the exposure time in microseconds?
        image_acquisition = ianc.ImageAcquisition()
        self.assertEqual(1, image_acquisition.exposure_time)
        image_acquisition.exposure_time = 1000
        self.assertEqual(1000, image_acquisition.exposure_time)

    def test_get_set_exposure_seconds(self):
        # more advanced test: do unit conversions work correctly when we get and set exposure time in seconds?
        image_acquisition = ianc.ImageAcquisition()

        # 1 microsecond
        self.assertEqual(1e-6, image_acquisition.exposure_time_seconds)

        # 1 millisecond
        image_acquisition.exposure_time = 1000
        self.assertEqual(1e-3, image_acquisition.exposure_time_seconds)
        self.assertEqual(1e3, image_acquisition.exposure_time)

        # 1 second
        image_acquisition.exposure_time_seconds = 1
        self.assertEqual(1, image_acquisition.exposure_time_seconds)
        self.assertEqual(1e6, image_acquisition.exposure_time)


if __name__ == "__main__":
    unittest.main()
