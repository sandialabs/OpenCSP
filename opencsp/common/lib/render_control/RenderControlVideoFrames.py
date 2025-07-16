import os
from typing import Optional

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class RenderControlVideoFrames:
    """
    Render control for extracting frames from videos.

    This class manages the extraction of frames from videos, allowing customization of frame formats,
    names, and extraction settings.

    Attributes
    ----------
    inframe_format : str
        The format for input frames (for video construction). Defaults to "png".
    outframe_name : str
        The format for frame names. Example "-%05d" with a video "foo.mp4" will produce "foo-00001", "foo-00002", etc. Defaults to "-%05d".
    outframe_format : str
        Format for extracted frames. Defaults to "png".
    outframe_dpi : Optional[int]
        Placeholder for dots per inch setting. Defaults to None.
    example_name : Optional[str]
        Name format for example frames. Defaults to outframe_name.
    example_format : Optional[str]
        Format for example (human consumable) extracted frames. Defaults to outframe_format.
    example_dpi : Optional[int]
        Placeholder for dpi setting for example frames. Defaults to outframe_dpi.
    example_freq : int
        The frequency of example frames. Defaults to 1 frame per second of video.
    draw_example_frames : bool
        If True, export example frames. Defaults to True.
    clear_dir : bool
        Whether to clear the existing directory before writing extracted frames. Defaults to True.
    concat_video_frame_names : bool
        Creates output names from concatenating the video and frame names. Defaults to True.
    """

    # ChatGPT 4o-mini assisted with generating this doc string
    def __init__(
        self,
        inframe_format: str = "png",
        outframe_name: str = "-%05d",
        outframe_format: str = "png",
        outframe_dpi: Optional[int] = None,
        example_name: Optional[str] = None,
        example_format: Optional[str] = None,
        example_dpi: Optional[int] = None,
        example_freq: int = 1,
        draw_example_frames=True,
        clear_dir=True,
        concat_video_frame_names=True,
    ):
        """
        Controls for how frames are used in VideoHandler.

        There are two types of extracted frames:
            - Example: some small number of example frames can optionally be
              extracted into a separate directory. These are intended for human
              consumption.
            - Outframe: All extracted frames from a video. These are more meant
              for machine processing and consumption.

        Parameters
        ----------
        inframe_format : str, optional
            The format for input frames (for video construction). Defaults to "png".
        outframe_name : str, optional
            The format for frame names. Defaults to "-%05d".
        outframe_format : str, optional
            Format for extracted frames. Defaults to "png".
        outframe_dpi : Optional[int], optional
            Placeholder for dots per inch setting. Defaults to None.
        example_name : Optional[str], optional
            Name format for example frames. Defaults to outframe_name.
        example_format : Optional[str], optional
            Format for example extracted frames. Defaults to outframe_format.
        example_dpi : Optional[int], optional
            Placeholder for dpi setting for example frames. Defaults to outframe_dpi.
        example_freq : int, optional
            The frequency of example frames. Defaults to 1 frame per second of video.
        draw_example_frames : bool, optional
            If True, export example frames. Defaults to True.
        clear_dir : bool, optional
            Whether to clear the existing directory before writing extracted frames. Defaults to True.
        concat_video_frame_names : bool, optional
            Creates output names from concatenating the video and frame names. Defaults to True.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        super(RenderControlVideoFrames, self).__init__()

        self.inframe_format = inframe_format
        self.outframe_name = outframe_name
        self.outframe_format = outframe_format
        self.outframe_dpi = outframe_dpi
        self.example_name = example_name if example_name is not None else outframe_name
        self.example_format = example_format if example_format is not None else outframe_format
        self.example_dpi = example_dpi if example_dpi is not None else outframe_dpi
        self.example_freq = example_freq
        self.draw_example_frames = draw_example_frames
        self.clear_dir = clear_dir
        self.concat_video_frame_names = concat_video_frame_names

    def clean_dir(self, dir: str, remove_only_images=False):
        """
        Remove images and/or all files in the specified directory.

        If `self.clear_dir` is True, this method will remove files based on the specified conditions.

        Parameters
        ----------
        dir : str
            The directory to clean.
        remove_only_images : bool, optional
            If True, only remove files with the specified image extensions. If False, remove all files. Default is False.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        # Sanity check
        if not ft.directory_exists(dir):
            lt.error_and_raise(RuntimeError, f"Directory {dir} does not exist!")

        # Clean the directory
        if not self.clear_dir:
            return
        if not remove_only_images:
            # Remove all files
            files_name_ext = ft.files_in_directory(dir, sort=False, files_only=True)
            for fn in files_name_ext:
                ft.delete_file(os.path.join(dir, fn), error_on_not_exists=False)
        else:
            # Remove only the files with the matching extension
            extensions = [self.outframe_format]
            if self.draw_example_frames and self.example_format != self.outframe_format:
                extensions.append(self.example_format)

            files_name_ext_dict = ft.files_in_directory_by_extension(dir, sort=False, extensions=extensions)
            for extension in files_name_ext_dict.keys():
                files_name_ext = files_name_ext_dict[extension]
                for fn in files_name_ext:
                    ft.delete_file(os.path.join(dir, fn), error_on_not_exists=False)

    def get_outframe_name(self, source_video_dir_body_ext: str = None, is_example_frames=False):
        """
        Returns the format string for generating frame names (name + ext only).

        Parameters
        ----------
        source_video_dir_body_ext : str, optional
            The source video directory and file name. If provided, it will be used to prefix the frame names.
        is_example_frames : bool, optional
            If True, generate names for example frames. Default is False.

        Returns
        -------
        str
            The formatted frame name with extension.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        # Get the name and extension
        name, ext = self.outframe_name, self.outframe_format
        if is_example_frames:
            name, ext = self.example_name, self.example_format

        # Name the frames
        if source_video_dir_body_ext is not None and self.concat_video_frame_names:
            _, file_name, _ = ft.path_components(source_video_dir_body_ext)
            name = file_name + name
        if not ext.startswith("."):
            ext = "." + ext
        name_ext = name + ext

        return name_ext

    def get_outframe_path_name_ext(
        self, destination_dir: str, source_video_dir_body_ext: str = None, is_example_frames=False
    ):
        """
        Get the full path for the output frame name including directory and extension.

        Parameters
        ----------
        destination_dir : str
            The directory where the extracted frames will be saved.
        source_video_dir_body_ext : str, optional
            The source video directory and file name. If provided, it will be used to prefix the frame names.
        is_example_frames : bool, optional
            If True, get the path for example frames. Default is False.

        Returns
        -------
        str
            The full path for the output frame name including directory and extension.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        name_ext = self.get_outframe_name(source_video_dir_body_ext, is_example_frames)
        return os.path.join(destination_dir, name_ext)

    def get_ffmpeg_args(
        self, destination_dir: str, source_video_dir_body_ext: str = None, is_example_frames=False
    ) -> tuple[str, dict[str, str]]:
        """
        Get the ffmpeg arguments for extracting either extracted or example frames.

        Example usage:
            ffmpeg_args = frame_control.get_ffmpeg_args(dest_dir, input_video_dir_body_ext)
            cmd = f"ffmpeg -i {input_video_dir_body_ext} {ffmpeg_args}"

        Parameters
        ----------
        destination_dir : str
            The directory where extracted frames will be located.
        source_video_dir_body_ext : str, optional
            If provided, the input video name will be used as the first part of the extracted frame names.
        is_example_frames : bool, optional
            If True, get the arguments for the example frames. Example frames
            are the same as standard frames, except that only a smaller
            fraction of total video frames are extraced. Default is False.

        Returns
        -------
        tuple[str, dict[str, str]]
            A tuple containing:
            - str: The arguments string to pass to ffmpeg.
            - dict: A dictionary with the key 'DESTDIR' pointing to the output path for the frames.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        args: list[str] = []
        paths: dict[str, str] = {}

        # Extract images at the highest quality
        args.append("-q:v 1 -qmin 1")

        # Set the framerate
        if is_example_frames:
            args.append(f"-filter:v fps={self.example_freq}")

        # Name the frames
        args.append("%DESTDIR%")
        paths["DESTDIR"] = self.get_outframe_path_name_ext(
            destination_dir, source_video_dir_body_ext, is_example_frames
        )

        return " ".join(args), paths

    @classmethod
    def default(cls):
        """
        Create a default instance of RenderControlVideoFrames.

        Returns
        -------
        RenderControlVideoFrames
            An instance of `RenderControlVideoFrames` with default settings.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        return cls()

    @classmethod
    def fast(cls):
        """
        Create a fast instance of RenderControlVideoFrames.

        This instance is configured to not draw example frames.

        Returns
        -------
        RenderControlVideoFrames
            An instance of `RenderControlVideoFrames` configured for fast processing.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        return cls(draw_example_frames=False)
