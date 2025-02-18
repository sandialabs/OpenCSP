import cv2
import numpy as np
import os
import unittest

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.render.VideoHandler as vh
import opencsp.common.lib.render_control.RenderControlVideo as rcv
import opencsp.common.lib.render_control.RenderControlVideoFrames as rcvf
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp


class test_VideoHandler(unittest.TestCase):
    dir_in = os.path.join(orp.opencsp_code_dir(), "common", "lib", "render", "test", "data", "input", "VideoHandler")
    dir_out = os.path.join(orp.opencsp_code_dir(), "common", "lib", "render", "test", "data", "output", "VideoHandler")

    @classmethod
    def setUpClass(cls) -> None:
        ret = super().setUpClass()
        if ft.directory_exists(test_VideoHandler.dir_out):
            ft.delete_files_in_directory(test_VideoHandler.dir_out, "*")
        ft.create_directories_if_necessary(test_VideoHandler.dir_out)
        return ret

    def setUp(self) -> None:
        # get the data paths, for easy access
        self.dir_in = test_VideoHandler.dir_in
        self.dir_out = test_VideoHandler.dir_out

        # prepare frame and video controllers
        self.video_control = rcv.RenderControlVideo(framerate=25)
        self.frame_control = rcvf.RenderControlVideoFrames(
            inframe_format="png", outframe_name="-%05d", outframe_format="png"
        )

    def set_output_dirs(self, handler: vh.VideoHandler):
        handler.dst_frames_dir = os.path.join(self.dir_out, "output_frames")
        handler.dst_example_frames_dir = os.path.join(self.dir_out, "example_frames")
        ft.create_directories_if_necessary(handler.dst_frames_dir)
        ft.create_directories_if_necessary(handler.dst_example_frames_dir)

    def test_get_duration(self):
        src_video_dir_name_ext = os.path.join(self.dir_in, "1s.mp4")
        handler = vh.VideoHandler.VideoInspector(src_video_dir_name_ext)
        self.set_output_dirs(handler)
        duration = handler.get_duration()
        self.assertAlmostEqual(duration, 1.0, delta=0.01)

    def test_get_width_height(self):
        src_video_dir_name_ext = os.path.join(self.dir_in, "1s.mp4")
        handler = vh.VideoHandler.VideoInspector(src_video_dir_name_ext)
        self.set_output_dirs(handler)
        width, height = handler.get_width_height()
        self.assertEqual(width, 640)
        self.assertEqual(height, 480)

    def test_extract_frames(self):
        frame_name_format = self.frame_control.get_outframe_name("1s.mp4")
        src_video_dir_name_ext = os.path.join(self.dir_in, "1s.mp4")
        test_dir = os.path.join(self.dir_out, "test_extract_frames")
        ft.create_directories_if_necessary(test_dir)

        handler = vh.VideoHandler.VideoExtractor(src_video_dir_name_ext, test_dir, None, self.frame_control)
        nextracted = handler.extract_frames()

        # verify that we extracted 25 frames from a 1 second clip at 25 fps
        self.assertEqual(nextracted, 25)

        # verify that the frames are blue, green, and red
        for frame_idx, color_idx in [(3, 0), (11, 1), (20, 2)]:
            img_name_ext = frame_name_format % frame_idx
            img_dir_name_ext = os.path.join(test_dir, img_name_ext)
            self.assertTrue(os.path.exists(img_dir_name_ext), f'Could not find image file "{img_dir_name_ext}"')

            img = cv2.imread(img_dir_name_ext)
            row_avg = np.average(img, axis=0)
            pix_avg = np.average(row_avg, axis=0)
            self.assertAlmostEqual(255, pix_avg[color_idx], delta=1)
            self.assertAlmostEqual(255, sum(pix_avg), delta=1)

    def test_extract_example_frames(self):
        src_video_dir_name_ext = os.path.join(self.dir_in, "1s.mp4")
        out_dir = os.path.join(self.dir_out, "test_extract_example_frames")
        example_dir = os.path.join(out_dir, "examples")
        ft.create_directories_if_necessary(example_dir)

        handler = vh.VideoHandler.VideoExtractor(src_video_dir_name_ext, out_dir, example_dir, self.frame_control)
        handler.extract_frames()

        # verify that we extracted 1 example frame from a 1 second clip
        nframes = ft.count_items_in_directory(example_dir)
        self.assertTrue(
            nframes == 1 or nframes == 2,
            f"Unexpected number of example frames. Expected either 1 or 2 from a 1s video clip. Found {nframes} in {example_dir}.",
        )

    def test_construct_video(self):
        dst_video_dir_name_ext = os.path.join(self.dir_out, "test_construct_video3.mp4")
        handler = vh.VideoHandler.VideoCreator(
            self.dir_in, dst_video_dir_name_ext, self.video_control, self.frame_control
        )
        self.set_output_dirs(handler)
        handler.construct_video(overwrite=True)
        actual_duration = handler.get_duration("output")
        expected_duration = 0.12  # 1 / 25 fps * 3 images
        self.assertAlmostEqual(actual_duration, expected_duration, delta=0.01)

    def test_frames_to_video_duplicates(self):
        dst_video_dir_name_ext = os.path.join(self.dir_out, "test_frames_to_video_duplicates.mp4")
        handler = vh.VideoHandler.VideoCreator(
            self.dir_in, dst_video_dir_name_ext, self.video_control, self.frame_control
        )
        self.set_output_dirs(handler)
        images_list = []
        for img_name in ["r.png", "g.png", "b.png"]:
            for i in range(25):
                images_list.append(img_name)
        handler.frames_to_video(images_list, overwrite=True)
        actual_duration = handler.get_duration("output")
        expected_duration = 3.0  # 1 / 25 fps * (25*3) images
        self.assertAlmostEqual(actual_duration, expected_duration, delta=0.01)

    def test_frames_to_video_exclusions(self):
        dst_video_dir_name_ext = os.path.join(self.dir_out, "test_frames_to_video_exclusions.mp4")
        handler = vh.VideoHandler.VideoCreator(
            self.dir_in, dst_video_dir_name_ext, self.video_control, self.frame_control
        )
        self.set_output_dirs(handler)
        handler.frames_to_video(["r.png", "g.png"], overwrite=True)
        actual_duration = handler.get_duration("output")
        expected_duration = 0.08  # 1 / 25 fps * 2 images
        self.assertAlmostEqual(actual_duration, expected_duration, delta=0.01)

    def test_transform_powerpoint(self):
        src_video_dir_name_ext = os.path.join(self.dir_in, "1s.mp4")
        dst_dir = os.path.join(self.dir_out, "test_transform_powerpoint")
        ft.create_directories_if_necessary(dst_dir)
        dst_video_dir_name_ext = vh.VideoHandler.transform_powerpoint(src_video_dir_name_ext, dst_dir, overwrite=True)

        # verify the width and height
        handler = vh.VideoHandler.VideoInspector(dst_video_dir_name_ext)
        self.set_output_dirs(handler)
        width, height = handler.get_width_height()
        self.assertEqual(width, 320)
        self.assertEqual(height, 240)

        # verify the file size
        src_size = os.path.getsize(src_video_dir_name_ext)
        dst_size = os.path.getsize(dst_video_dir_name_ext)
        self.assertLess(dst_size, src_size)


if __name__ == "__main__":
    unittest.main()
