"""
Video and frames manipulation and creation.



"""

import cv2 as cv
import os

import opencsp.common.lib.process.subprocess_tools as subt
import opencsp.common.lib.render_control.RenderControlVideo as rcv
import opencsp.common.lib.render_control.RenderControlVideoFrames as rcvf
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.list_tools as listt
import opencsp.common.lib.tool.log_tools as lt


class VideoHandler:
    """Handle video creation, frame extraction, and video transcoding. The format for generated videos and
    frames is controlled by the video_control and frame_control render controllers.

    Not all of these arguments are required for every use case of this class. The generator methods
    VideoCreator(), VideoTransformer(), and VideoExtractor() can be used to simplify the required
    parameters down to the most common use cases."""

    _video_extensions = [
        "mp4",
        "avi",
        "wmv",
        "ogv",
        "4xm",
        "amv",
        "mtv",
        "m4v",
        "flv",
        "f4v",
        "f4p",
        "f4a",
        "f4b",
        "vlb",
        "m4v",
        "mpg",
        "mp2",
        "mpeg",
        "mpe",
        "mpv",
        "m4v",
        "webm",
        "mov",
        "qt",
        "rm",
        "rmvb",
    ]
    """ List of video formats that ffmpeg supports (probably not comprehensive, maybe some errors) """

    def __init__(
        self,
        src_video_dir_name_ext: str = None,
        dst_video_dir_name_ext: str = None,
        src_frames_dir: str = None,
        dst_frames_dir: str = None,
        dst_example_frames_dir: str = None,
        video_control: rcv.RenderControlVideo = None,
        frame_control: rcvf.RenderControlVideoFrames = None,
    ):
        """Handle video creation, frame extraction, and video transcoding. The format for generated videos and
        frames is controlled by the video_control and frame_control render controllers.

        Not all of these arguments are required for every use case of this class. The generator methods
        VideoCreator(), VideoTransformer(), and VideoExtractor() can be used to simplify the required
        parameters down to the most common use cases.

        Args:
            src_video_dir_name_ext (str, optional): The source file name for manipulation of an existing video. Not needed for creating a video from frames.
            dst_video_dir_name_ext (str, optional): The destination file name for creating a video. Not needed for extracting frames.
            src_frames_dir (str, optional): Where to find frames for video creation.
            dst_frames_dir (str, optional): Where to put frames for frame extraction. Defaults to default_output_path()/output_frames.
            dst_example_frames_dir (str, optional): Where to put example (debugging) frames for frame extraction. Defaults to default_output_path()/example_frames.
            video_control (rcv.RenderControlVideo, optional): Render controller for video creation.
            frame_control (rcvf.RenderControlVideoFrames, optional): Render controller for frames.
        """
        self.src_video_dir_name_ext = src_video_dir_name_ext
        self.dst_video_dir_name_ext = dst_video_dir_name_ext
        self.src_frames_dir = src_frames_dir
        self.dst_frames_dir = (
            dst_frames_dir if dst_frames_dir != None else os.path.join(ft.default_output_path(), "output_frames")
        )
        self.dst_example_frames_dir = (
            dst_example_frames_dir
            if dst_example_frames_dir != None
            else os.path.join(ft.default_output_path(), "example_frames")
        )
        self.video_control = video_control if video_control != None else rcv.RenderControlVideo.default()
        self.frame_control = frame_control if frame_control != None else rcvf.RenderControlVideoFrames.default()

    @classmethod
    def VideoInspector(cls, src_video_dir_name_ext: str):
        return cls(src_video_dir_name_ext=src_video_dir_name_ext)

    @classmethod
    def VideoCreator(
        cls,
        src_frames_dir: str,
        dst_video_dir_name_ext: str,
        video_control: rcv.RenderControlVideo,
        frame_control: rcvf.RenderControlVideoFrames,
    ):
        return cls(
            dst_video_dir_name_ext=dst_video_dir_name_ext,
            src_frames_dir=src_frames_dir,
            video_control=video_control,
            frame_control=frame_control,
        )

    @classmethod
    def VideoMerger(cls, src_videos_path, src_videos_ext, dst_video_dir_name_ext):
        return cls(
            src_video_dir_name_ext=os.path.join(src_videos_path, f"NA.{src_videos_ext}"),
            dst_video_dir_name_ext=dst_video_dir_name_ext,
        )

    @classmethod
    def VideoTransformer(
        cls, src_video_dir_name_ext: str, dst_video_dir_name_ext: str, video_control: rcv.RenderControlVideo
    ):
        return cls(
            src_video_dir_name_ext=src_video_dir_name_ext,
            dst_video_dir_name_ext=dst_video_dir_name_ext,
            video_control=video_control,
        )

    @classmethod
    def VideoExtractor(
        cls,
        src_video_dir_name_ext: str,
        dst_frames_dir: str,
        dst_example_frames_dir: str,
        frame_control: rcvf.RenderControlVideoFrames,
    ):
        return cls(
            src_video_dir_name_ext=src_video_dir_name_ext,
            dst_frames_dir=dst_frames_dir,
            dst_example_frames_dir=dst_example_frames_dir,
            frame_control=frame_control,
        )

    def _build_ffmpeg_cmd(self, args_str: str, paths: dict[str, str] = None):
        """Build a properly formatted ffmpeg command given the arguments string and file system paths.

        Example::

            indir = home_dir() + "/tmpth2t3p6u.txt"
            outfile = opencsp_dir() + "common/lib/test/output/render/VideoHandler\\test_construct_video.mp4"
            args_str = "-f concat -safe 0 -i %INDIR% -filter:v fps=5 -c:v libx264 %OUTFILE% 2>&1"
            cmd = self._build_ffmpeg_cmd(args_str, {"INDIR":indir, "OUTFILE":outfile})
            subt.run(cmd, cwd=src_dir)

        Args:
            args_str (str): The string of arguments to pass to ffmpeg.
            paths (dict[int,str], optional): The paths to pass to ffmpeg, to match and replace %KEY%s in args_str.

        Returns:
            str: A runnable command.
        """
        # replace path names
        for path_key in paths.keys():
            match = "%" + path_key.upper() + "%"
            path = ft.path_to_cmd_line(paths[path_key])
            if match in args_str:
                args_str = args_str.replace(match, path)
            else:
                lt.debug(
                    f"VideoHandler::_build_ffmpeg_cmd: could not find matches to path {match}:{path} in {args_str}"
                )

        # attach the executable
        ret = "ffmpeg"
        if args_str != "":
            ret += " " + args_str

        # ffmpeg outputs to stderr by default
        ret += " 2>&1"

        return ret

    # EXTRACTING FRAMES
    #
    def extract_frames(self, start_time: float = None, end_time: float = None):
        """Extracts all frames from a video file, placing them in the specified output directory.
        Creates the directory and its parents if it does not exist.
        The extracted frame format and directory overwrite option is controlled by the RenderControlVideoFrames.

        This takes about 7 hours to finish a 14m30s, 4K, 30fps video on a single 16-core server.
        This takes about 16 minutes to finish a similar video when parallelized across 20 16-core servers with parallel_video_tools.

        Raises:
            subprocess.CalledProcessError: Raised if the subprocess returns an error code.

        Args:
            - start_time (int): The time to start extracting at.
            - end_time (int): The time to stop extracting at.

        Returns:
            int: The number of frames extracted (not including example frames)
        """
        # Check input.
        if not ft.file_exists(self.src_video_dir_name_ext):
            lt.error_and_raise(
                FileNotFoundError,
                'ERROR: In VideoHandler.extract_frames(), src_video_dir_name_ext does not exist: "'
                + str(self.src_video_dir_name_ext)
                + '"',
            )
        if (start_time != None and end_time != None) and (end_time < start_time):
            lt.error_and_raise(
                ValueError,
                f"Error: in VideoHandler.extract_frames(), end_time < start_time ({end_time} < {start_time})",
            )

        # run once for example and extracted frames
        cleaned_dirs: list[str] = []
        for run in ["example", "output"]:
            if run == "example":
                if not self.frame_control.draw_example_frames:
                    continue
                is_example_frames = True
            else:
                is_example_frames = False
            frame_path_name_ext = self.get_extracted_frame_path_and_name_format(run)
            frame_dir, _, _ = ft.path_components(frame_path_name_ext)

            # Create the output frame directory if necessary.
            ft.create_directories_if_necessary(frame_dir)

            # Remove existing images with the same extension
            if frame_dir not in cleaned_dirs:
                self.frame_control.clean_dir(frame_dir)
                cleaned_dirs.append(frame_dir)

            # Assemble the ffmpeg command.
            ffmpeg_args, paths = self.frame_control.get_ffmpeg_args(
                frame_dir, self.src_video_dir_name_ext, is_example_frames
            )
            paths.update({"INFILE": self.src_video_dir_name_ext})
            time_arg = ""
            if start_time != None:
                time_arg += f"-ss {start_time} "
            if end_time != None:
                start_time_or_zero = start_time if start_time != None else 0
                time_arg += f"-t {end_time - start_time_or_zero} "
            cmd = self._build_ffmpeg_cmd(f"{time_arg}-i %INFILE% {ffmpeg_args}", paths)

            # Execute the ffmpeg command and create the frames.
            lt.debug("In ConstructExtractedFrames.extract_frames()")
            subt.run(cmd)

            # Count the number of extracted frames.
            if run == "output":
                frame_dir, _, frame_ext = ft.path_components(frame_path_name_ext)
                _, input_video_body, _ = ft.path_components(self.src_video_dir_name_ext)
                n_frames = ft.count_items_in_directory(frame_dir, name_prefix=input_video_body, name_suffix=frame_ext)

        # Return.
        return n_frames

    def get_extracted_frame_path_and_name_format(self, frame_type="output"):
        """Build the frame output name format, including the frame number (ex "-%05d").=

        Args:
            frame_type (str): Whether this should be for a normal "output" frame or an "example" frame. Defaults to "output".

        Returns:
            frame_format (str): The path/nameformat.ext of the output frame.
        """
        if frame_type == "example":
            frame_dir = self.dst_example_frames_dir
            is_example_frames = True
        else:
            frame_dir = self.dst_frames_dir
            is_example_frames = False
        _, input_video_body, _ = ft.path_components(self.src_video_dir_name_ext)
        return self.frame_control.get_outframe_path_name_ext(frame_dir, input_video_body, is_example_frames)

    # FILTERING DUPLICATE FRAMES
    #
    def identify_duplicate_frames(self, tolerance_image_size: int, tolerance_image_pixel: int):
        """Finds all frame duplicates in self.src_frames_dir.

        Args:
            - tolerance_image_size (int): Size difference, in bytes, for which the two files can be considered identical.
            - tolerance_image_pixel (int): How many pixels are allowed to be different and the images are still considered identical.

        Returns:
            - list[str]: The non-duplicate image's name+ext
            - list[str]: The duplicate image's name+ext
        """
        # Check input.
        if not ft.directory_exists(self.src_frames_dir):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In identify_duplicate_frames(), src_frames_dir does not exist: "'
                + str(self.src_frames_dir)
                + '"',
            )

        # Fetch list of all frame filenames.
        input_frame_file_size_pair_list = ft.files_in_directory_with_associated_sizes(self.src_frames_dir, sort=True)
        n_input_frames = len(input_frame_file_size_pair_list)

        # Construct the sequence of frames without duplicates, and also the list of duplicate frames omitted.
        if n_input_frames < 2:
            # Then there cannot be any duplicates.
            non_duplicate_frame_files = [ft.file_size_pair_name(pair) for pair in input_frame_file_size_pair_list]
            duplicate_frame_files = []
        else:
            # Loop through frame files, looking for duplicates.
            previous_frame_file_size_pair = input_frame_file_size_pair_list[0]
            non_duplicate_frame_files = [ft.file_size_pair_name(previous_frame_file_size_pair)]
            duplicate_frame_files: list[str] = []  # First frame is never a duplicate of preceding.
            for this_frame_file_size_pair in input_frame_file_size_pair_list[1:]:
                if self._this_frame_is_a_duplicate_of_previous(
                    previous_frame_file_size_pair,
                    this_frame_file_size_pair,
                    tolerance_image_size,
                    tolerance_image_pixel,
                ):
                    # Then this frame is a  duplicate.
                    duplicate_frame_files.append(ft.file_size_pair_name(this_frame_file_size_pair))
                    if len(duplicate_frame_files) == 1:
                        lt.info("Found at least one duplicate frame: " + duplicate_frame_files[0])
                else:
                    # This frame is not a duplicate.
                    non_duplicate_frame_files.append(ft.file_size_pair_name(this_frame_file_size_pair))
                    previous_frame_file_size_pair = this_frame_file_size_pair

        # Return.
        return non_duplicate_frame_files, duplicate_frame_files

    def _this_frame_is_a_duplicate_of_previous(
        self,
        previous_frame_file_size_pair,
        this_frame_file_size_pair,
        tolerance_image_size: int,
        tolerance_image_pixel: int,
    ):
        """Determine if the two images are duplicates.

        Args:
            previous_frame_file_size_pair (list|tuple): File name+ext and size for the first image to compare.
            this_frame_file_size_pair (list|tuple): File name+ext and size for the second image to compare.
            tolerance_image_size (int): Size difference, in bytes, for which the two files can be considered identical.
            tolerance_image_pixel (int): How many pixels are allowed to be different and the images are still considered identical.

        Returns:
            bool: True if the images are duplicates, False otherwise
        """
        # Fetch data components.
        previous_file: str = ft.file_size_pair_name(previous_frame_file_size_pair)
        previous_size: int = ft.file_size_pair_size(previous_frame_file_size_pair)
        this_file: str = ft.file_size_pair_name(this_frame_file_size_pair)
        this_size: int = ft.file_size_pair_size(this_frame_file_size_pair)

        # A first, fast test for duplicate images is to compare file size.  While we expect images of the
        # same (row, col) dimensions, file size varies due to JPEG compression of image content.  This is
        # why image files of constant-dimension images ahve varying size.  Identical images will result in
        # identical compressions, which will have identical size.  Therefore if two image files have
        # different size, then they must have different image content.  Thus we can use size as a first
        # check for duplicate images: If two files are of different size, they cannot be identical.
        # If they are of identical size, they may or may not be identical.
        #
        if abs(this_size - previous_size) <= tolerance_image_size:
            # Frame JPEG files are the same size.  They might be identical, so check content.
            if self._frames_are_identical(previous_file, this_file, tolerance_image_pixel):
                return True
            else:
                return False
        else:
            # Frame JPEG files are different sizes.  We conclude they cannot be identical.
            return False

    def _frames_are_identical(self, previous_frame_file: str, this_frame_file: str, tolerance_image_pixel: int):
        """Determine if the given frames are identical.

        Args:
            - previous_frame_file (str): File name+ext of the first frame to compare.
            - this_frame_file (str): File name+ext of the second frame to compare.
            - tolerance_image_pixel (int): How many pixels are allowed to be different and the images are still considered identical.

        Returns:
            - bool: True if identical, False otherwise
        """
        # Load frames.
        previous_dir_body_ext = os.path.join(
            self.src_frames_dir, previous_frame_file
        )  # Already includes the extension.
        this_dir_body_ext = os.path.join(self.src_frames_dir, this_frame_file)  # Already includes the extension.
        lt.debug("\nIn frames_are_identical(), loading image:", previous_dir_body_ext)
        previous_img = cv.imread(previous_dir_body_ext)
        lt.debug("In frames_are_identical(), loading image:", this_dir_body_ext)
        this_img = cv.imread(this_dir_body_ext)
        lt.debug("In frames_are_identical(), comparing images...")
        identical = it.images_are_identical(previous_img, this_img, tolerance_image_pixel)
        lt.debug("In frames_are_identical(), Done.  identical =", identical)
        # Return.
        return identical

    def _err_if_video_exists(self, do_err: bool = False):
        if not do_err:
            return
        if ft.file_exists(self.dst_video_dir_name_ext):
            lt.error_and_raise(RuntimeError, f"There is already an existing video file '{self.dst_video_dir_name_ext}'")

    def _remove_existing_video(self, do_remove: bool = False):
        if not do_remove:
            return
        if ft.file_exists(self.dst_video_dir_name_ext):
            lt.info(f"Removing existing file '{self.dst_video_dir_name_ext}'")
            ft.delete_file(self.dst_video_dir_name_ext)

    def _str_list_to_tmp_file(self, str_vals: list[str], tmp_dir: str = None):
        """Writes all the strings (plus a newline) to a temporary file and returns the file_dir_name_ext.

        Because this creates a temporary file, it is highly recommended to use
        this function as follow to prevent file clutter::

            try:
                tmp_dir_name_ext = self._str_list_to_tmp_file(str_vals)
                ...
            finally:
                ft.delete_file(tmp_dir_name_ext)

        Returns:
            str: The name of the temporary file.
        """
        fd, path_name_ext = ft.get_temporary_file(suffix=".txt", dir=tmp_dir)
        with open(fd, "w") as fout:
            fout.writelines(s + "\n" for s in str_vals)
        with open(path_name_ext, "r") as fin:
            lt.info("Temp file contents: " + fin.read())
        return path_name_ext

    def _files_list_to_video(
        self,
        src_dir: str,
        src_names_exts: list[str],
        dst_video_dir_name_ext: str,
        tmp_dir: str = None,
        overwrite: bool = False,
        widthheight_vidorimg_dir_name_ext="",
    ):
        """Generates a given video dst_video_dir_name_ext from the given images or videos src_names_exts.

        Args:
            src_dir (str): The directory in which to find all of the src_names_exts files.
            src_names_exts (list[str]): A list of either images or videos to concatenate into a single video.
            dst_video_dir_name_ext (str): The video file to be created.
            tmp_dir (str, optional): Where to save a temporary file with the list of filenames to. Defaults to user home or /tmp.
            overwrite (bool, optional): If True, then remove the existing video before creating a new one. Defaults to False.
            widthheight_vidorimg_dir_name_ext (str, optional): This can be a image or video to get the default width and height from. Defaults to "".
        """
        # create the directory for the video
        dst_video_dir, _, _ = ft.path_components(self.dst_video_dir_name_ext)
        ft.create_directories_if_necessary(dst_video_dir)

        # remove the existing video file
        self._err_if_video_exists(not overwrite)
        self._remove_existing_video(overwrite)

        # build the video
        try:
            # get the list of file names in the proper format for ffmpeg
            _, duration_str = self.video_control.get_frames_to_video_parameters()
            str_vals = []
            for src_name_ext in src_names_exts:
                _, _, src_ext = ft.path_components(src_name_ext)
                src_name_ext_norm = "file " + ft.path_to_cmd_line(os.path.join(src_dir, src_name_ext))
                src_name_ext_norm = src_name_ext_norm.replace("\\", "/")
                str_vals.append(src_name_ext_norm)
                if not (src_ext.strip(".").lower() in self._video_extensions):
                    str_vals.append(duration_str)

            # save the list of file names
            tmp_path_name_ext = self._str_list_to_tmp_file(str_vals, tmp_dir)

            # ffmpeg args
            args, paths = self.video_control.get_ffmpeg_args(widthheight_vidorimg_dir_name_ext)
            paths.update({"INDIR": tmp_path_name_ext, "DSTFILE": dst_video_dir_name_ext})
            cmd = self._build_ffmpeg_cmd(f"-f concat -safe 0 -i %INDIR% {args} %DSTFILE%", paths)

            # execute ffmpeg from the frames directory
            subt.run(cmd, cwd=src_dir)
        finally:
            ft.delete_file(tmp_path_name_ext)

    def frames_to_video(self, frame_names: list[str], tmp_dir: str = None, overwrite: bool = False):
        """Converts specific frames in into a video (vs construct_video() which uses all available frames).

        This takes about ~2 minutes for 450 frames with two Xeon E5-2695 36-thread cpus.
        This can also be parallelized with parallel_video_tools.

        Raises:
            subprocess.CalledProcessError: Raised if the subprocess returns an error code.

        Args:
            frame_names (list[str]): The list of frame names to generate a video from. This list will be temporarily saved as a file in tmp_dir.
            tmp_dir (str): Where to create a file to temporarily save the frame names to.
            overwrite (bool): If true, then remove the existing video before calling ffmpeg.

        Returns:
            str: The generated video file (name and extension). None if frame_names is empty.
        """
        lt.debug("In frames_to_video")
        frame_names_msg = ""
        try:
            # sanitize inputs
            if not ft.directory_exists(self.src_frames_dir):
                lt.error_and_raise(RuntimeError, f"Could not find the frames directory '{self.src_frames_dir}'!")
            if len(frame_names) == 0:
                return None
            frame_names_msg = f' from frames ["{frame_names[0]}", ...]'

            # get the example image for determining width/height
            img0_dir_path_ext = os.path.join(self.src_frames_dir, frame_names[0])

            # build the video
            self._files_list_to_video(
                self.src_frames_dir, frame_names, self.dst_video_dir_name_ext, tmp_dir, overwrite, img0_dir_path_ext
            )

            return self.dst_video_dir_name_ext
        except Exception:
            lt.error(
                f'Error: in VideoHandler.frames_to_video: failed to create video "{self.dst_video_dir_name_ext}"{frame_names_msg}'
            )
            raise

    def construct_video(self, tmp_dir: str = None, overwrite: bool = False):
        """Creates a video "self.dst_video_dir_name_ext" from the images in the "self.src_frames_dir" directory.
        To precisely control which image files are sourced, call frames_to_video(...) instead.

        Args:
            tmp_dir (str): Where to create a file to temporarily save the frame names to.
            overwrite (bool): If true, then remove the existing video before calling ffmpeg.

        Returns:
            str: The generated video file (name and extension). None if frame_names is empty.
        """
        lt.debug("In construct_video")
        ext = self.frame_control.inframe_format
        frame_names_dict = ft.files_in_directory_by_extension(self.src_frames_dir, [ext], sort=True)
        frame_names = frame_names_dict[ext]
        lt.debug(f"Found {len(frame_names)} frames to construct with ext '{ext}'")
        return self.frames_to_video(frame_names, tmp_dir, overwrite)

    def merge_videos(self, src_video_names: list[str] = None, tmp_dir: str = None, overwrite: bool = False):
        """Merges many videos into a single video.
        For H.265 videos, this is a very fast operation.

        Raises:
            subprocess.CalledProcessError: Raised if the subprocess returns an error code.

        Args:
            - src_video_names (list[str]): The list of videos names to generate a video from, or None to use all videos in the src_video_dir_name_ext directory. This list will be temporarily saved as a file in tmp_dir.
            - tmp_dir (str): Where to create a file to temporarily save the frame names to.
            - overwrite (bool): If true, then remove the existing video before calling ffmpeg.

        Returns:
            - str: The generated video file (name and extension). None if src_video_names is empty.
        """
        lt.debug("In merge_videos")
        # get some values
        src_video_dir, _, src_video_ext = ft.path_components(self.src_video_dir_name_ext)
        dst_video_dir, _, _ = ft.path_components(self.dst_video_dir_name_ext)

        # sanitize inputs
        if not ft.directory_exists(src_video_dir):
            lt.error_and_raise(RuntimeError, f"Could not find the source videos directory '{src_video_dir}'!")
        if src_video_names == None:
            src_video_names_dict = ft.files_in_directory_by_extension(src_video_dir, [src_video_ext])
            src_video_names = src_video_names_dict[src_video_ext]
            src_video_names = listt.natural_sort(src_video_names)
        if len(src_video_names) == 0:
            return None

        # create the output directory as necessary
        ft.create_directories_if_necessary(dst_video_dir)

        # build the video
        src_video0_dir_name_ext = os.path.join(src_video_dir, src_video_names[0])
        self._files_list_to_video(
            src_video_dir, src_video_names, self.dst_video_dir_name_ext, tmp_dir, overwrite, src_video0_dir_name_ext
        )

        return self.dst_video_dir_name_ext

    def transform_video(self, overwrite: bool = False):
        """Makes a copy of the given video as out_video_path_name_ext, with the various options from the render control.

        This can be used to, for example, create a video suitable for power point as with transform_powerpoint(...).

        Raises:
            subprocess.CalledProcessError: Raised if the subprocess returns an error code.
        """
        # validate input
        if not ft.file_exists(self.src_video_dir_name_ext):
            lt.error_and_raise(RuntimeError, f"Video '{self.src_video_dir_name_ext}' doesn't exist!")

        # remove the existing video file
        self._err_if_video_exists(not overwrite)
        self._remove_existing_video(overwrite)

        # build the ffmpeg command
        args, paths = self.video_control.get_ffmpeg_args(self.src_video_dir_name_ext)
        paths.update({"INFILE": self.src_video_dir_name_ext, "OUTFILE": self.dst_video_dir_name_ext})
        cmd = self._build_ffmpeg_cmd(f"-i %INFILE% {args} %OUTFILE%", paths)

        # execute ffmpeg
        lt.debug("In transform_video")
        subt.run(cmd)

        return self.dst_video_dir_name_ext

    @classmethod
    def transform_powerpoint(cls, src_video_dir_name_ext: str, dst_dir: str = None, overwrite: bool = False):
        """Makes a copy of the given video as '[path_and_name]_ppt[ext]' with
        a reduced size and codec suitable for inclusion in power point.
        This takes ~3 minutes for 26,000 1080p frames with two Xeon E5-2695 18-core cpus.
        This takes ~36 minutes for 26,000 4K frames with one Xeon E5-2670 16-core cpu.

        Returns:
            str: The dir_name_ext of the output video."""
        # get the new video name
        dirname, filename_ext = os.path.split(src_video_dir_name_ext)
        filename, fileext = os.path.splitext(filename_ext)
        dst_dir = dst_dir if dst_dir != None else dirname
        dst_video_dir_name_ext = os.path.join(dst_dir, filename + "_ppt" + fileext)

        # get the render control and video handler
        video_control = rcv.RenderControlVideo.power_point()
        handler = cls(
            src_video_dir_name_ext=src_video_dir_name_ext,
            dst_video_dir_name_ext=dst_video_dir_name_ext,
            video_control=video_control,
        )

        return handler.transform_video(overwrite)

    def get_width_height(self, input_or_output="input"):
        """Returns the width and height of the source video (or image), in pixels."""
        # from https://superuser.com/questions/841235/how-do-i-use-ffmpeg-to-get-the-video-resolution
        path = self.src_video_dir_name_ext if input_or_output == "input" else self.dst_video_dir_name_ext
        path = ft.path_to_cmd_line(path)
        lines = subt.run(
            f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {path}"
        )
        line = lines[0].val
        parts = line.split("x")
        w, h = int(parts[0]), int(parts[1])
        return w, h

    def get_duration(self, input_or_output="input"):
        """Returns the duration of the source video in seconds."""
        # from https://superuser.com/questions/650291/how-to-get-video-duration-in-seconds
        path = self.src_video_dir_name_ext if input_or_output == "input" else self.dst_video_dir_name_ext
        path = ft.path_to_cmd_line(path)
        lines = subt.run(
            f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path}"
        )
        line = lines[0].val
        seconds = float(line.strip())
        return seconds

    def get_num_frames(self, input_or_output="input"):
        """Returns the number of frames in the source video."""
        # from https://stackoverflow.com/questions/2017843/fetch-frame-count-with-ffmpeg
        path = self.src_video_dir_name_ext if input_or_output == "input" else self.dst_video_dir_name_ext
        path = ft.path_to_cmd_line(path)
        lines = subt.run(
            f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {path}"
        )
        line = lines[0].val
        num_frames = int(line.strip())
        return num_frames
