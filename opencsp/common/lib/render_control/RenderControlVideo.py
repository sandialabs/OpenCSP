class RenderControlVideo:
    """
    A class to control the rendering of videos with customizable parameters.

    This class allows users to specify various settings for video rendering, including
    framerate, resolution, codec, and bitrate options. It provides methods to retrieve
    ffmpeg arguments for video processing and to calculate parameters for generating
    videos from a set of images.

    Attributes
    ----------
    framerate : int
        The framerate for the video.
    width : int or None
        The width of the video.
    height : int or None
        The height of the video.
    min_scale : bool
        Indicates whether to apply minimum scaling.
    codec : str
        The codec used for encoding the video.
    low_bitrate : bool
        Indicates whether to use low bitrate encoding.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(
        self,
        framerate: int = 30,
        width=None,
        height=None,
        min_scale=False,
        codec: str = "H.264",
        low_bitrate: bool = False,
    ):
        """
        Initialize the RenderControlVideo object.

        Parameters
        ----------
        framerate : int, optional
            Framerate for the video in frames per second, or None (uses source framerate). Defaults to 30.
        width : int, optional
            Sets the width of the video. None to scale relative to height. Defaults to None.
        height : int, optional
            Sets the height of the video. None to scale relative to width. Defaults to None.
        min_scale : bool, optional
            When True, width=min(width, curr) and height=min(height, curr). Defaults to False.
        codec : str, optional
            How to encode new videos. Can be one of 'undefined', 'H.264', 'H.265', or 'copy'. Defaults to 'H.264'.
        low_bitrate : bool, optional
            Should videos be encoded for a reduced file size? If True, then this sets the '-crf' option. Defaults to False.

        Raises
        ------
        RuntimeError
            If an unrecognized codec option is provided or if low_bitrate is True with an undefined or copy codec.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.framerate = framerate
        self.codec = codec
        self.width = width
        self.height = height
        self.min_scale = min_scale
        self.low_bitrate = low_bitrate
        self._original_video_width_height = None

        if codec not in ["undefined", "H.264", "H.265", "copy"]:
            raise RuntimeError(f"Unrecognized codec option '{self.codec}'")
        if low_bitrate and codec in ["undefined", "copy"]:
            raise RuntimeError("Codec must be specified in order to use low_bitrate=True")

    def _get_original_video_width_height(self, video_or_image_path_name_ext: str):
        if self._original_video_width_height == None:
            import opencsp.common.lib.render.VideoHandler as vh

            handler = vh.VideoHandler.VideoInspector(video_or_image_path_name_ext)
            self._original_video_width_height = handler.get_width_height()
        return self._original_video_width_height

    def get_ffmpeg_args(self, video_or_image_path_name_ext: str = "") -> tuple[str, dict[str, str]]:
        """
        Get the arguments and directories to be passed to ffmpeg.

        Parameters
        ----------
        video_or_image_path_name_ext : str, optional
            Needed for width or height relative to the original video. Defaults to "".

        Returns
        -------
        tuple[str, dict[str, str]]
            The arguments string to pass to ffmpeg and any additional options as a dictionary.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        ret: list[str] = []

        if self.codec != "undefined":
            if self.codec == "copy":
                ret.append("-c copy")
            elif self.codec == "H.264":
                ret.append("-c:v libx264")
            elif self.codec == "H.265":
                ret.append("-c:v libx265")

        if self.framerate != None:
            ret.append(f"-filter:v fps={self.framerate}")

        if self.low_bitrate:
            if self.codec == "H.264" or self.codec == "copy":
                ret.append("-crf 30")
            elif self.code == "H.265":
                ret.append("-crf 40")

        if self.width != None or self.height != None:
            width, height = self.width, self.height
            if video_or_image_path_name_ext != "":

                def owidth():
                    return self._get_original_video_width_height(video_or_image_path_name_ext)[0]

                def oheight():
                    return self._get_original_video_width_height(video_or_image_path_name_ext)[1]

                if self.min_scale:
                    if width != None:
                        width = min(width, owidth())
                    if height != None:
                        height = min(height, oheight())
                if width == None:
                    width = int(owidth() * (height / oheight()))
                if height == None:
                    height = int(oheight() * (width / owidth()))
            ret.append(f'-vf "scale={width}:{height}"')
            self._original_video_width_height = None

        # pixel format: not necessary, let ffmpeg choose
        # ret.append("-pix_fmt yuv420p")

        return " ".join(ret), {}

    def get_frames_to_video_parameters(self):
        """
        Get the duration for the desired framerate to generate a video from a set of images.

        Returns
        -------
        tuple[float, str]
            How long each image should be shown for, and the ffmpeg files list line for that duration.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        framerate = self.framerate if self.framerate != None else 25
        duration = 1.0 / framerate
        duration_str = "duration %0.20F" % duration
        return duration, duration_str

    # COMMON CASES

    @classmethod
    def default(cls):
        """
        Create a default instance of RenderControlVideo.

        Returns
        -------
        RenderControlVideo
            A default instance of the class with preset parameters.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return cls()

    @classmethod
    def power_point(cls, framerate=10, width=320, height=None, codec='H.264', low_bitrate=False):
        """
        Returns a set of defaults suitable for embedding videos into PowerPoint.

        Parameters
        ----------
        framerate : int, optional
            Framerate for the video in frames per second. Defaults to 10.
        width : int, optional
            Sets the width of the video. Defaults to 320.
        height : int, optional
            Sets the height of the video. Defaults to None.
        codec : str, optional
            How to encode new videos. Defaults to 'H.264'.
        low_bitrate : bool, optional
            Should videos be encoded for a reduced file size? Defaults to False.

        Returns
        -------
        RenderControlVideo
            An instance of RenderControlVideo with parameters suitable for PowerPoint.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return cls(
            framerate=framerate, width=width, height=height, min_scale=True, codec=codec, low_bitrate=low_bitrate
        )
