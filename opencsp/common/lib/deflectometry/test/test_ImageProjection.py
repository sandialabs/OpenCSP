import os

import numpy as np
import pytest
import unittest

import opencsp.common.lib.deflectometry.ImageProjection as ip
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft


@pytest.mark.no_xvfb
class test_ImageProjection(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "ImageProjection")
        self.out_dir = os.path.join(path, "data", "output", "ImageProjection")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def tearDown(self):
        with et.ignored(Exception):
            ip.ImageProjection.instance().close()

    def test_set_image_projection(self):
        self.assertIsNone(ip.ImageProjection.instance())

        # Create a mock ImageProjection object
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)

        # Test that the instance was set
        self.assertEqual(image_projection, ip.ImageProjection.instance())

        # Test un-setting the image_projection object
        image_projection.close()
        self.assertIsNone(ip.ImageProjection.instance())

    def test_on_close(self):
        global close_count
        close_count = 0

        def close_count_inc(image_projection):
            global close_count
            close_count += 1

        # Create a mock ImageProjection object with single on_close callback
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        image_projection.on_close.append(close_count_inc)
        image_projection.close()
        self.assertEqual(close_count, 1)

        # Create a mock ImageProjection object with multiple on_close callback
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        image_projection.on_close.append(close_count_inc)
        image_projection.on_close.append(close_count_inc)
        image_projection.close()
        self.assertEqual(close_count, 3)

        # Create a mock ImageProjection object without an on_close callback
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)
        image_projection.close()
        self.assertEqual(close_count, 3)

    def test_zeros(self):
        # Create a mock ImageProjection object
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)

        # Get the zeros array and verify its shape and values
        zeros = image_projection.zeros()
        self.assertEqual((480, 640, 3), zeros.shape)
        ones = zeros + 1
        self.assertEqual(np.sum(ones), 640 * 480 * 3)

    def test_to_from_hdf(self):
        h5file = os.path.join(self.out_dir, "test_to_from_hdf.h5")

        # Create a mock ImageProjection object
        image_projection = _ImageProjection.in_new_window(_ImageProjection.display_dict)

        # Save to HDF
        ft.delete_file(h5file, error_on_not_exists=False)
        self.assertFalse(ft.file_exists(h5file))
        image_projection.save_to_hdf(h5file)
        self.assertTrue(ft.file_exists(h5file))

        # Close, so that we don't have multiple windows open at a time
        image_projection.close()

        # Load from HDF
        image_projection2 = _ImageProjection.load_from_hdf_and_display(h5file)
        self.assertEqual(image_projection2.display_data, _ImageProjection.display_dict)


class _ImageProjection(ip.ImageProjection):
    display_dict = {
        'name': 'smoll_for_unit_tests',
        'win_size_x': 640,
        'win_size_y': 480,
        'win_position_x': 0,
        'win_position_y': 0,
        'size_x': 640,
        'size_y': 480,
        'position_x': 0,
        'position_y': 0,
        'projector_max_int': 255,
        'projector_data_type': "uint8",
        'shift_red_x': 0,
        'shift_red_y': 0,
        'shift_blue_x': 0,
        'shift_blue_y': 0,
        'image_delay': 1,
        'ui_position_x': 0,
    }


if __name__ == '__main__':
    unittest.main()
