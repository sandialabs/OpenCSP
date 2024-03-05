"""


"""
import os
from typing import Optional

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

class RenderControlVideoFrames():
    """
    Render control for extracting frames from videos.
    """

    def __init__(self,
            inframe_format: str = "png",
            outframe_name: str = "-%05d",
            outframe_format: str = "png",
            outframe_dpi: Optional[int] = None,
            example_name: Optional[str] = None,
            example_format: Optional[str] = None,
            example_dpi: Optional[int] = None,
            example_freq: int = 1,
            draw_example_frames=True,
            clear_dir = True,
            concat_video_frame_names = True
        ):
        """ Controls for how frames get are used in VideoHandler.

        There are two types of extracted frames:
            - Example: some small number of example frames can optionally be
            extracted into a separate directory. These are intended for human
            consumption.
            - outframe: All extracted frames from a video. These are more meant
            for machine processing and consumption.

        Args:
            - inframe_format (str): The format for input frames (for video construction). Defaults to "png".
            - outframe_name (str): The format for frame names. Example "-%05d" with a video "foo.mp4" will produce "foo-00001", "foo-00002", etc. Defaults to "-%05d".
            - outframe_format (str): Format for etracted frames. Defaults to "png".
            - outframe_dpi (int|None): PLACEHOLDER. NOT IMPLEMENTED. What to se the dots per inch to. Defaults to None.
            - example_format (str|None): Format for example (human consumable) extracted frames. Defaults to outframe_format.
            - example_dpi (int|None): PLACEHOLDER. NOT IMPLEMENTED. What to set the dpi to. Defaults to outframe_dpi.
            - example_freq (float): The frequency of example frames. Defaults to 1 frame per second of video.
            - draw_example_frames (bool) If true, then also export example frames. Defaults to True.
            - clear_dir (bool): Whether to clear the existing directory before writing extracted frames to it. Defaults to True.
            - concat_video_frame_names (bool): Creates output names from concatenating the video+frame names. Defaults to True.
        """

        super(RenderControlVideoFrames, self).__init__()
        
        self.inframe_format      = inframe_format
        self.outframe_name       = outframe_name
        self.outframe_format     = outframe_format
        self.outframe_dpi        = outframe_dpi
        self.example_name        = example_name if example_name != None else outframe_name
        self.example_format      = example_format if example_format != None else outframe_format
        self.example_dpi         = example_dpi if example_dpi != None else outframe_dpi
        self.example_freq        = example_freq
        self.draw_example_frames = draw_example_frames
        self.clear_dir           = clear_dir
        self.concat_video_frame_names = concat_video_frame_names
    
    def clean_dir(self, dir: str, remove_only_images = False):
        """ If self.clean_dir is True, then removes images and/or all files in the directory.

        Args:
            dir (str): The directory to clean.
            images_ext (str): If this is not none, then remove files that have the same extension. If none then remove all files. Default None.
        """
        # sanity check
        if not ft.directory_exists(dir):
            lt.error_and_raise(RuntimeError, f"Directory {dir} does not exist!")
        
        # clean the directory
        # Note: in the case that we're running this script in parallel, then
        # there is a good chance that a file doesn't exist by the time we get
        # to it, so don't error out in that case.
        if not self.clean_dir:
            return
        if remove_only_images == False:
            # remove all files
            files_name_ext = ft.files_in_directory(dir, sort=False, files_only=True)
            for fn in files_name_ext:
                ft.delete_file(os.path.join(dir, fn), error_on_not_exists=False)
        else:
            # remove only the files with the matching extension
            extensions = [self.outframe_format]
            if self.draw_example_frames and self.example_format != self.outframe_format:
                extensions.append(self.example_format)

            files_name_ext_dict = ft.files_in_directory_by_extension(dir, sort=False, extensions=extensions)
            for extension in files_name_ext_dict.keys():
                files_name_ext = files_name_ext_dict[extension]
                for fn in files_name_ext:
                    ft.delete_file(os.path.join(dir, fn), error_on_not_exists=False)

    def get_outframe_name(self, source_video_dir_body_ext: str=None, is_example_frames=False):
        """ Returns the format string for generating frame names (name+ext only) """
        # get the name and extension
        name, ext = self.outframe_name, self.outframe_format
        if is_example_frames:
            name, ext = self.example_name, self.example_format

        # name the frames
        if source_video_dir_body_ext != None and self.concat_video_frame_names:
            _, file_name, _ = ft.path_components(source_video_dir_body_ext)
            name = file_name + name
        if not ext.startswith("."):
            ext = "." + ext
        name_ext = name + ext

        return name_ext
    
    def get_outframe_path_name_ext(self, destination_dir: str, source_video_dir_body_ext: str=None, is_example_frames=False):
        name_ext = self.get_outframe_name(source_video_dir_body_ext, is_example_frames)
        return os.path.join(destination_dir, name_ext)

    def get_ffmpeg_args(self, destination_dir: str, source_video_dir_body_ext: str=None, is_example_frames=False) -> tuple[str,dict[str,str]]:
        """ Get the ffmpeg arguments for extracting either extracted or example frames.

        Example usage::

            ffmpeg_args = frame_control.get_ffmpeg_args(dest_dir, input_video_dir_body_ext)
            cmd = f"ffmpeg -i {input_video_dir_body_ext} {ffmpeg_args}"
            subprocess_tools.run(cmd)

        Args:
            destination_dir (str): The directory where extracted frames will be located.
            source_video_dir_body_ext (str): If provided, then the input video name will be used as the first part of the extracted frame names.
            is_example_frames (bool): True to get the arguments for the example frames. Default False.

        Returns:
            str: The arguments string to pass to ffmpeg.
        """
        args: list[str] = []
        paths: dict[str,str] = {}

        # extract images at the highest quality
        args.append("-q:v 1 -qmin 1")

        # set the framerate
        if is_example_frames:
            args.append(f"-filter:v fps={self.example_freq}")

        # name the frames
        args.append("%DESTDIR%")
        paths["DESTDIR"] = self.get_outframe_path_name_ext(destination_dir, source_video_dir_body_ext, is_example_frames)

        return " ".join(args), paths

    # COMMON CASES
    
    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def fast(cls):
        return cls(draw_example_frames=False)