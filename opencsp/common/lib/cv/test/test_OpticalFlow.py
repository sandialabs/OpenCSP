import copy
import math
import numpy as np
import os
import PIL.Image as Image
import time
import unittest

import opencsp.common.lib.cv.OpticalFlow as of
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.system_tools as st


class TestSubprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ret = super().setUpClass()

        cls.src_img_dir = os.path.join(
            orp.opencsp_code_dir(), "common", "lib", "cv", "test", "data", "input", "OpticalFlow"
        )
        cls.dst_dir = os.path.join(
            orp.opencsp_code_dir(), "common", "lib", "cv", "test", "data", "output", "OpticalFlow"
        )
        cls.src_img_file = os.path.join(cls.src_img_dir, "20210513F08f9800_w400.jpg")
        cls.tmp_img_dir = os.path.join(cls.dst_dir, "tmp")

        ft.create_directories_if_necessary(cls.tmp_img_dir)
        ft.create_directories_if_necessary(cls.dst_dir)

        return ret

    def setUp(self) -> None:
        ret = super().setUp()

        self.src_img_dir = self.__class__.src_img_dir
        self.dst_dir = self.__class__.dst_dir
        self.src_img_file = self.__class__.src_img_file
        self.tmp_img_dir = self.__class__.tmp_img_dir
        self.img1_name_ext = "f1_%s.jpg"
        self.img2_name_ext = "f2_%s.jpg"
        self.can_cache = not st.is_cluster()

        return ret

    def _prep_ref_img(self, travel: int, frame_motion_dir="up", secondary_frame_motion_dir=None):
        """Crop and save image files for testing.

        Args:
            travel (int): Number of pixels of translate by
            frame_motion_dir (str): Motion of the frame (camera). Example "up" means the frame has moved up, meaning that the image content has translated down. Defaults to "up".

        Returns:
            str: The first image name+ext (center of the primary image)
            str: The second image name+ext (translated imaged content)
        """
        img = Image.open(self.src_img_file)
        w, h = img.width, img.height
        tv = travel

        # Define the centered crop
        #                Left Top Right Bottom
        centered_crop = (tv, tv, w - tv, h - tv)

        # Define the translated crop
        translated_crop = centered_crop
        motions = [frame_motion_dir]
        if secondary_frame_motion_dir != None:
            motions.append(secondary_frame_motion_dir)
        for motion in motions:
            if motion == "up":
                translated_crop = (
                    translated_crop[0],
                    translated_crop[1] - tv,
                    translated_crop[2],
                    translated_crop[3] - tv,
                )
            elif motion == "down":
                translated_crop = (
                    translated_crop[0],
                    translated_crop[1] + tv,
                    translated_crop[2],
                    translated_crop[3] + tv,
                )
            elif motion == "right":
                translated_crop = (
                    translated_crop[0] + tv,
                    translated_crop[1],
                    translated_crop[2] + tv,
                    translated_crop[3],
                )
            elif motion == "left":
                translated_crop = (
                    translated_crop[0] - tv,
                    translated_crop[1],
                    translated_crop[2] - tv,
                    translated_crop[3],
                )

        # name to save these images with
        name = frame_motion_dir
        if secondary_frame_motion_dir != None:
            name += "_" + secondary_frame_motion_dir

        center = img.crop(centered_crop)
        moved = img.crop(translated_crop)
        img1_name_ext = self.img1_name_ext % name
        img2_name_ext = self.img2_name_ext % name
        center.save(os.path.join(self.tmp_img_dir, img1_name_ext))
        moved.save(os.path.join(self.tmp_img_dir, img2_name_ext))

        return img1_name_ext, img2_name_ext

    def _prep_limit_flow(self, magvals, angvals):
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "right")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=self.can_cache)
        flow.mag = np.array(magvals)
        flow.ang = np.array(angvals)
        return flow

    def test_left(self):
        """Frame has moved left, image content has translated right"""
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "left")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext)
        mag, ang = flow.dense()
        # left is a special case, because some of the angles will be ~0 and some will be ~2pi
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        self.assertAlmostEqual(10, np.average(mag), delta=0.3)
        self.assertAlmostEqual(np.pi * (0 / 2), np.average(ang), delta=np.pi / 15)

        # bonus test! can we save without error?
        Image.fromarray(flow.to_img()).save(
            os.path.join(self.dst_dir, "frame_translate_left__image_data_translate_right.jpg")
        )

    def test_left_down(self):
        """Frame has moved left and down, image content has translated right and up"""
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "left", "down")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext)
        mag, ang = flow.dense()
        self.assertAlmostEqual(np.pi * (1 / 4), np.median(ang), delta=np.pi / 15)
        self.assertAlmostEqual(math.sqrt(10**2 + 10**2), np.average(mag), delta=0.3)

        # bonus test! can we save without error?
        Image.fromarray(flow.to_img()).save(
            os.path.join(self.dst_dir, "frame_translate_left_down__image_data_translate_right_up.jpg")
        )

    def test_down(self):
        """Frame has moved down, image content has translated up"""
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "down")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext)
        mag, ang = flow.dense()
        self.assertAlmostEqual(10, np.average(mag), delta=0.3)
        self.assertAlmostEqual(np.pi * (1 / 2), np.average(ang), delta=np.pi / 15)

        # bonus test! can we save without error?
        Image.fromarray(flow.to_img()).save(
            os.path.join(self.dst_dir, "frame_translate_down__image_data_translate_up.jpg")
        )

    def test_right(self):
        """Frame has moved right, image content has translated left"""
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "right")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext)
        mag, ang = flow.dense()
        self.assertAlmostEqual(10, np.average(mag), delta=0.3)
        self.assertAlmostEqual(np.pi * (2 / 2), np.average(ang), delta=np.pi / 15)

        # bonus test! can we save without error?
        Image.fromarray(flow.to_img()).save(
            os.path.join(self.dst_dir, "frame_translate_right__image_data_translate_left.jpg")
        )

    def test_up(self):
        """Frame has moved up, image content has translated down"""
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "up")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext)
        mag, ang = flow.dense()
        self.assertAlmostEqual(10, np.average(mag), delta=0.3)
        self.assertAlmostEqual(np.pi * (3 / 2), np.average(ang), delta=np.pi / 15)

        # bonus test! can we save without error?
        Image.fromarray(flow.to_img()).save(
            os.path.join(self.dst_dir, "frame_translate_up__image_data_translate_down.jpg")
        )

    def test_cache_1(self):
        if not self.can_cache:
            return
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "right")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext)
        flow.clear_cache()

        # check that the cache doesn't exist
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=True)
        cache_file, cache_idx = flow._get_existing_cache_file()
        self.assertEqual(None, cache_file)

        # generate the cache files
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=True)
        flow.dense()

        # check that the cache exists
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=True)
        cache_file, cache_idx = flow._get_existing_cache_file()
        self.assertEqual(0, cache_idx)

    def test_cache_2(self):
        if not self.can_cache:
            return
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "right")
        img3, img4 = self._prep_ref_img(10, "left")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=True)
        flow.clear_cache()

        # check that the cache doesn't exist
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=True)
        cache_file, cache_idx = flow._get_existing_cache_file()
        self.assertEqual(None, cache_file)
        flow = of.OpticalFlow(self.tmp_img_dir, img3, self.tmp_img_dir, img4, cache=True)
        cache_file, cache_idx = flow._get_existing_cache_file()
        self.assertEqual(None, cache_file)

        # generate the cache file for A
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=True)
        flow.dense()
        # check that the cache exists/doesn't exist
        flow = of.OpticalFlow(self.tmp_img_dir, img3, self.tmp_img_dir, img4, cache=True)
        cache_file, cache_idx = flow._get_existing_cache_file()
        self.assertEqual(None, cache_file)
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=True)
        cache_file, cache_idx = flow._get_existing_cache_file()
        self.assertEqual(0, cache_idx)

        # generate the cache file for B
        flow = of.OpticalFlow(self.tmp_img_dir, img3, self.tmp_img_dir, img4, cache=True)
        flow.dense()
        # check that the cache files exist
        flow = of.OpticalFlow(self.tmp_img_dir, img3, self.tmp_img_dir, img4, cache=True)
        cache_file, cache_idx = flow._get_existing_cache_file()
        self.assertEqual(0, cache_idx)
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext, cache=True)
        cache_file, cache_idx = flow._get_existing_cache_file()
        self.assertEqual(1, cache_idx)

    def test_limit_by_magnitude_inside(self):
        flow = self._prep_limit_flow([1, 2, 3, 4, 5], [0, 1.5, 3, 4.5, 6])

        zeroed = flow.limit_by_magnitude(lower=2, upper=4, keep="inside")

        self.assertListEqual([True, False, False, False, True], zeroed.tolist())
        self.assertListEqual([0, 2, 3, 4, 0], flow.mag.tolist())
        self.assertListEqual([0, 1.5, 3, 4.5, 0], flow.ang.tolist())

    def test_limit_by_magnitude_outside(self):
        flow = self._prep_limit_flow([1, 2, 3, 4, 5], [0, 1.5, 3, 4.5, 6])

        zeroed = flow.limit_by_magnitude(lower=2, upper=4, keep="outside")

        self.assertListEqual([False, False, True, False, False], zeroed.tolist())
        self.assertListEqual([1, 2, 0, 4, 5], flow.mag.tolist())
        self.assertListEqual([0, 1.5, 0, 4.5, 6], flow.ang.tolist())

    def test_limit_by_angle_inside(self):
        flow = self._prep_limit_flow([1, 2, 3, 4, 5], [0, 1.5, 3, 4.5, 6])

        zeroed = flow.limit_by_angle(lower=2, upper=5, keep="inside")

        self.assertListEqual([True, True, False, False, True], zeroed.tolist())
        self.assertListEqual([0, 0, 3, 4, 0], flow.mag.tolist())
        self.assertListEqual([0, 0, 3, 4.5, 0], flow.ang.tolist())

    def test_limit_by_angle_outside(self):
        flow = self._prep_limit_flow([1, 2, 3, 4, 5], [0, 1.5, 3, 4.5, 6])

        zeroed = flow.limit_by_angle(lower=2, upper=5, keep="outside")

        self.assertListEqual([False, False, True, True, False], zeroed.tolist())
        self.assertListEqual([1, 2, 0, 0, 5], flow.mag.tolist())
        self.assertListEqual([0, 1.5, 0, 0, 6], flow.ang.tolist())

    def test_limit_by_angle_inside_neg(self):
        flow = self._prep_limit_flow([1, 2, 3, 4, 5], [0, 1.5, 3, 4.5, 6])

        # 5-2*pi = -1.28
        zeroed = flow.limit_by_angle(lower=-1.28, upper=2, keep="inside")

        self.assertListEqual([False, False, True, True, False], zeroed.tolist())
        self.assertListEqual([1, 2, 0, 0, 5], flow.mag.tolist())
        self.assertListEqual([0, 1.5, 0, 0, 6], flow.ang.tolist())

    def test_limit_by_angle_outside_neg(self):
        flow = self._prep_limit_flow([1, 2, 3, 4, 5], [0, 1.5, 3, 4.5, 6])

        # 5-2*pi = -1.28
        zeroed = flow.limit_by_angle(lower=-1.28, upper=2, keep="outside")

        self.assertListEqual([True, True, False, False, True], zeroed.tolist())
        self.assertListEqual([0, 0, 3, 4, 0], flow.mag.tolist())
        self.assertListEqual([0, 0, 3, 4.5, 0], flow.ang.tolist())

    def test_limit_by_angle_inside_bothneg(self):
        flow = self._prep_limit_flow([1, 2, 3, 4, 5], [0, 1.5, 3, 4.5, 6])

        # 2-2*pi = -4.28
        # 5-2*pi = -1.28
        zeroed = flow.limit_by_angle(lower=-4.28, upper=-1.28, keep="inside")

        self.assertListEqual([True, True, False, False, True], zeroed.tolist())
        self.assertListEqual([0, 0, 3, 4, 0], flow.mag.tolist())
        self.assertListEqual([0, 0, 3, 4.5, 0], flow.ang.tolist())

    def test_save_load(self):
        """When we save and load, do we get the same results back?"""
        img1_name_ext, img2_name_ext = self._prep_ref_img(10, "left")
        flow = of.OpticalFlow(self.tmp_img_dir, img1_name_ext, self.tmp_img_dir, img2_name_ext)
        flow.dense()

        # convert to integers to enable perfect comparisons before/after save
        _mag = (flow.mag * 100_000).astype(int)
        _ang = (flow.ang * 100_000).astype(int)
        flow._mag = _mag
        flow._ang = _ang
        flow.dense()  # forces the private matrices to be stored in the public matrices

        # start with a clean slate
        dir = self.dst_dir
        name_ext = "test_save_load_optflow.npy"
        dir_name_ext = os.path.join(dir, name_ext)
        if ft.file_exists(dir_name_ext):
            ft.delete_file(dir_name_ext)

        # save, verify files exist
        flow.save(dir, name_ext)
        self.assertTrue(ft.file_exists(dir_name_ext), "Failed to create save file")

        # make a copy, just to be sure that we're not loading into the already existing pointer
        _mag, _ang = copy.deepcopy(flow._mag), copy.deepcopy(flow._ang)
        mag, ang = copy.deepcopy(flow.mag), copy.deepcopy(flow.ang)

        # load, verify equal
        mag2, ang2 = flow.load(dir, name_ext)
        self.assertTrue(np.array_equal(_mag, flow._mag), "_Magnitude (private) matrices not equal after load+save")
        self.assertTrue(np.array_equal(_ang, flow._ang), "_Angle (private) matrices not equal after load+save")
        self.assertTrue(np.array_equal(mag, mag2), "Magnitude matrices not equal after load+save")
        self.assertTrue(np.array_equal(ang, ang2), "Angle matrices not equal after load+save")


if __name__ == "__main__":
    unittest.main()
