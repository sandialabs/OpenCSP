"""
Developing a list of frames from a UFACET scan video, which eliminate repetitive frames.

(For some reason, it seems that ffmpeg occasionally spits out the same frame twice, with a new frame number.)



"""

import os

import opencsp.common.lib.render.video_manipulation as vm
import opencsp.common.lib.render_control.RenderControlFramesNoDuplicates as rcfnd
import opencsp.common.lib.tool.file_tools as ft
import lib.ufacet_pipeline_frame as upf


class FramesNoDuplicates:
    """
    Class for  a list of frames from a UFACET scan video, which eliminate repetitive frames.
    """

    def __init__(
        self,
        input_video_dir_body_ext,  # Where to find the video file.
        input_frame_dir,  # Where to find the video extracted frames (before removing duplicates).
        output_data_dir,  # Where to save the resulting summary data.
        output_render_dir,  # Where to save the resulting sample frame plots.
        output_frame_dir,  # Where to save the resulting frames.
        tolerance_image_size,  # Tolerance on size check for duplicate frames.
        tolerance_image_pixel,  # Tolerance on pixel check for duplicate frames.
        render_control,
    ):  # Flags to control rendering on this run.
        # Check input.
        if (input_video_dir_body_ext == None) or (len(input_video_dir_body_ext) == 0):
            raise ValueError("In FramesNoDuplicates.__init__(), null input_video_dir_body_ext encountered.")
        if (input_frame_dir == None) or (len(input_frame_dir) == 0):
            raise ValueError("In FramesNoDuplicates.__init__(), null input_frame_dir encountered.")
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In FramesNoDuplicates.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In FramesNoDuplicates.__init__(), null output_render_dir encountered.")
        if (output_frame_dir == None) or (len(output_frame_dir) == 0):
            raise ValueError("In FramesNoDuplicates.__init__(), null output_frame_dir encountered.")

        # Parse input video path components.
        input_video_dir, input_video_body, input_video_ext = ft.path_components(input_video_dir_body_ext)

        # Store input.
        self.input_video_dir_body_ext = input_video_dir_body_ext
        self.input_video_dir = input_video_dir
        self.input_video_body = input_video_body
        self.input_video_ext = input_video_ext
        self.input_frame_dir = input_frame_dir
        self.output_data_dir = output_data_dir
        self.output_render_dir = output_render_dir
        self.output_frame_dir = output_frame_dir
        self.tolerance_image_size = tolerance_image_size
        self.tolerance_image_pixel = tolerance_image_pixel
        self.render_control = render_control

        # Summary statistics file name.
        self.dict_body = self.input_video_body + "_frames_non_duplicate_statistics"
        self.dict_body_ext = self.dict_body + ".csv"
        self.dict_dir_body_ext = os.path.join(self.output_data_dir, self.dict_body_ext)

        # File listing non-duplicate file names.
        self.non_duplicate_body = self.input_video_body + "_non_duplicate_frame_files"
        self.non_duplicate_body_ext = self.non_duplicate_body + ".txt"
        self.non_duplicate_dir_body_ext = os.path.join(self.output_data_dir, self.non_duplicate_body_ext)

        # File listing duplicate file names.
        self.duplicate_body = self.input_video_body + "_duplicate_frame_files"
        self.duplicate_body_ext = self.duplicate_body + ".txt"
        self.duplicate_dir_body_ext = os.path.join(self.output_data_dir, self.duplicate_body_ext)

        # Extract frames, if not already.
        self.filter_frames_and_write_data()

        # Load the summary statistics.
        self.read_data()

        # Render, if desired.
        self.render()

    # CONSTRUCT RESULT
    def filter_frames_and_write_data(self):
        # Create the output frame directory if necessary.
        ft.create_directories_if_necessary(self.output_frame_dir)

        # Check if frames are already copied.
        if ft.directory_is_empty(self.output_frame_dir):
            print("In FramesNoDuplicates.filter_frames_and_write_data(), filtering frames...")

            # Identify duplicate frames.
            (non_duplicate_frame_files, duplicate_frame_files) = vm.identify_duplicate_frames(
                self.input_frame_dir, self.output_frame_dir, self.tolerance_image_size, self.tolerance_image_pixel
            )

            # Copy non-duplicate frames to frame output directory.
            for frame_file in non_duplicate_frame_files:
                print(
                    "In FramesNoDuplicates.filter_frames_and_write_data(), copying frame: "
                    + frame_file
                    + "  to  "
                    + output_frame_dir
                )
                ft.copy_file(os.path.join(input_frame_dir, frame_file), output_frame_dir)

            # Write the summary information.
            # Create the output data directory if necessary.
            ft.create_directories_if_necessary(self.output_data_dir)
            summary_dict = {}
            summary_dict["n_frames_non_duplicates"] = len(non_duplicate_frame_files)
            summary_dict["n_frames_duplicates"] = len(duplicate_frame_files)
            print("In FramesNoDuplicates.filter_frames_and_write_data(), writing frame summary statistics...")
            ft.write_dict_file(
                "frame summary statistics (duplicates removed)", self.output_data_dir, self.dict_body, summary_dict
            )
            ft.write_text_file(
                "non-duplicate frames", self.output_data_dir, self.non_duplicate_body, non_duplicate_frame_files
            )
            ft.write_text_file("duplicate frames", self.output_data_dir, self.duplicate_body, duplicate_frame_files)

    # LOAD RESULT
    def read_data(self):
        print("In FramesNoDuplicates.read_data(), reading frame statistics: ", self.dict_dir_body_ext)
        self.frame_statistics_dict = ft.read_dict(self.dict_dir_body_ext)

    # RENDER RESULT
    def render(self):
        if self.render_control.draw_example_frames:
            self.draw_example_frames()

    def draw_example_frames(self):
        print("In FramesNoDuplicates.draw_example_frames(), drawing example frames...")
        upf.draw_example_frames(
            self.output_frame_dir,
            self.output_render_dir,
            self.render_control,
            delete_suffix=".JPG_fig.png",
            n_intervals=10,
            include_figure_idx_in_filename=False,
        )


if __name__ == "__main__":
    input_video_dir_body_ext = (
        experiment_dir() + "2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4"
    )
    input_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/070c_ExtractedFrames/mavic_zoom/frames/"
    )
    output_data_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/080_FramesNoDuplicates/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/080_FramesNoDuplicates/mavic_zoom/render/"
    )
    output_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/"
    )
    tolerance_image_size = 0
    tolerance_image_pixel = 0
    render_control = rcfnd.default()

    extracted_frames_object = FramesNoDuplicates(
        input_video_dir_body_ext,
        input_frame_dir,
        output_data_dir,
        output_render_dir,
        output_frame_dir,
        tolerance_image_size,
        tolerance_image_pixel,
        render_control,
    )
