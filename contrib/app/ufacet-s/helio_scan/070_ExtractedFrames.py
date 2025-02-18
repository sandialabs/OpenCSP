"""
Extracting frames from a UFACET scan video.



"""

import os

import opencsp.common.lib.render_control.RenderControlExtractedFrames as rcef
import opencsp.common.lib.tool.file_tools as ft
import lib.ufacet_pipeline_frame as upf
import opencsp.common.lib.render.video_manipulation as vm


class ExtractedFrames:
    """
    Class for extracting all frames from a UFACET scan video.
    """

    def __init__(
        self,
        input_video_dir_body_ext,  # Where to find the video file.
        output_data_dir,  # Where to save the resulting summary data.
        output_render_dir,  # Where to save the resulting sample frame plots.
        output_frame_dir,  # Where to save the resulting frames.
        output_frame_id_format,  # Format to use for frame numbers in the generated frame filenames.
        render_control,
    ):  # Flags to control rendering on this run.
        # Check input.
        if (input_video_dir_body_ext == None) or (len(input_video_dir_body_ext) == 0):
            raise ValueError("In ExtractedFrames.__init__(), null input_video_dir_body_ext encountered.")
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In ExtractedFrames.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In ExtractedFrames.__init__(), null output_render_dir encountered.")
        if (output_frame_dir == None) or (len(output_frame_dir) == 0):
            raise ValueError("In ExtractedFrames.__init__(), null output_frame_dir encountered.")

        # Parse input video path components.
        input_video_dir, input_video_body, input_video_ext = ft.path_components(input_video_dir_body_ext)

        # Store input.
        self.input_video_dir_body_ext = input_video_dir_body_ext
        self.input_video_dir = input_video_dir
        self.input_video_body = input_video_body
        self.input_video_ext = input_video_ext
        self.output_data_dir = output_data_dir
        self.output_render_dir = output_render_dir
        self.output_frame_dir = output_frame_dir
        self.output_frame_id_format = output_frame_id_format
        self.render_control = render_control

        # Summary statistics file name.
        self.dict_body = self.input_video_body + "_frames_maybe_duplicates_statistics"
        self.dict_body_ext = self.dict_body + ".csv"
        self.dict_dir_body_ext = os.path.join(self.output_data_dir, self.dict_body_ext)

        # Extract frames, if not already.
        self.extract_frames_and_write_data()

        # Load the summary statistics.
        self.read_data()

        # Render, if desired.
        self.render()

    # CONSTRUCT RESULT
    def extract_frames_and_write_data(self):
        # Create the output frame directory if necessary.
        ft.create_directories_if_necessary(self.output_frame_dir)

        # Check if frames are already extracted.
        if ft.directory_is_empty(self.output_frame_dir):
            print("In ExtractedFrames.extract_frames_and_write_data(), extracting frames...")
            # Extract the frames.
            n_frames = vm.extract_frames(
                self.input_video_dir_body_ext, self.output_frame_dir, self.output_frame_id_format
            )

            # Write the summary information.
            # Create the output data directory if necessary.
            ft.create_directories_if_necessary(self.output_data_dir)
            summary_dict = {}
            summary_dict["n_frames_maybe_duplicates"] = n_frames
            print("In ExtractedFrames.extract_frames_and_write_data(), writing frame summary statistics...")
            ft.write_dict_file(
                "frame summary statistics (possibly includes duplicates)",
                self.output_data_dir,
                self.dict_body,
                summary_dict,
            )

    # LOAD RESULT
    def read_data(self):
        print("In ExtractedFrames.read_data(), reading frame statistics: ", self.dict_dir_body_ext)
        self.frame_statistics_dict = ft.read_dict(self.dict_dir_body_ext)

    # RENDER RESULT
    def render(self):
        if self.render_control.draw_example_frames:
            self.draw_example_frames()

    def draw_example_frames(self):
        print("In ExtractedFrames.draw_example_frames(), drawing example frames...")
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
    output_data_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/070_ExtractedFrames/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/070_ExtractedFrames/mavic_zoom/render/"
    )
    output_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/070c_ExtractedFrames/mavic_zoom/frames/"
    )
    output_frame_id_format = ".%06d"
    render_control = rcef.default()

    extracted_frames_object = ExtractedFrames(
        input_video_dir_body_ext,
        output_data_dir,
        output_render_dir,
        output_frame_dir,
        output_frame_id_format,
        render_control,
    )
