class RenderControlKeyFramesGivenManual:
    """
    Render control for the UFACET pipeline step KeyFrames (manual input version).
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_key_frames=True,  # Whether to draw the key frame figures.
        key_frame_polygon_linewidth=3,  # Linewidth for expected heliostat polygon.
        key_frame_polygon_color="m",  # Color for expected heliostat polygon.
        key_frame_label_horizontalalignment="center",  # Horizontal alignment for heliostat label.
        key_frame_label_verticalalignment="center",  # Vertical alignment for heliostat label.
        key_frame_label_fontsize=10,  # Font size for heliostat label.
        key_frame_label_fontstyle="normal",  # Font style for heliostat label.
        key_frame_label_fontweight="bold",  # Font weight for heliostat label.
        key_frame_label_color="m",  # Color for heliostat label.
        key_frame_dpi=200,  # Dpi for saving figure to disk.
        key_frame_crop=False,  # Whether to crop annotations outside image frame.
    ):
        """
        Render control for the UFACET pipeline step KeyFrames (manual input version).

        This class manages the rendering settings for the KeyFrames step in the UFACET pipeline,
        allowing customization of various visual elements related to manually input key frames.

        Parameters
        ----------
        clear_previous : bool, optional
            Whether to remove any existing files in the designated output directory. By default, True.
        draw_key_frames : bool, optional
            Whether to draw the key frame figures. By default, True.
        key_frame_polygon_linewidth : float, optional
            Line width for the expected heliostat polygon. By default, 3.
        key_frame_polygon_color : str, optional
            Color for the expected heliostat polygon. By default, 'm' (magenta).
        key_frame_label_horizontalalignment : str, optional
            Horizontal alignment for the heliostat label. By default, 'center'.
        key_frame_label_verticalalignment : str, optional
            Vertical alignment for the heliostat label. By default, 'center'.
        key_frame_label_fontsize : int, optional
            Font size for the heliostat label. By default, 10.
        key_frame_label_fontstyle : str, optional
            Font style for the heliostat label. By default, 'normal'.
        key_frame_label_fontweight : str, optional
            Font weight for the heliostat label. By default, 'bold'.
        key_frame_label_color : str, optional
            Color for the heliostat label. By default, 'm' (magenta).
        key_frame_dpi : int, optional
            DPI (dots per inch) for saving figures to disk. By default, 200.
        key_frame_crop : bool, optional
            Whether to crop annotations outside the image frame. By default, False.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlKeyFramesGivenManual, self).__init__()

        self.clear_previous = clear_previous
        self.draw_key_frames = draw_key_frames
        self.key_frame_polygon_linewidth = key_frame_polygon_linewidth
        self.key_frame_polygon_color = key_frame_polygon_color
        self.key_frame_label_horizontalalignment = key_frame_label_horizontalalignment
        self.key_frame_label_verticalalignment = key_frame_label_verticalalignment
        self.key_frame_label_fontsize = key_frame_label_fontsize
        self.key_frame_label_fontstyle = key_frame_label_fontstyle
        self.key_frame_label_fontweight = key_frame_label_fontweight
        self.key_frame_label_color = key_frame_label_color
        self.key_frame_dpi = key_frame_dpi
        self.key_frame_crop = key_frame_crop


# COMMON CASES


def default():
    """
    Create a default render control for key frames.

    This function returns a `RenderControlKeyFramesGivenManual` instance with default settings.

    Returns
    -------
    RenderControlKeyFramesGivenManual
        An instance of `RenderControlKeyFramesGivenManual` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlKeyFramesGivenManual()


def fast():
    """
    Create a fast render control for key frames.

    This function returns a `RenderControlKeyFramesGivenManual` instance configured to skip
    drawing the key frame figures, which can speed up the rendering process.

    Returns
    -------
    RenderControlKeyFramesGivenManual
        An instance of `RenderControlKeyFramesGivenManual` configured for fast rendering.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlKeyFramesGivenManual(draw_key_frames=False)
