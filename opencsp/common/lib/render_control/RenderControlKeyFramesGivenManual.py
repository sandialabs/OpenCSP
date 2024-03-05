"""


"""


class RenderControlKeyFramesGivenManual:
    """
    Render control for the UFACET pipeline step KeyFrames (manual input version).
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_key_frames=True,  # Whether to draw the key frame figures.
        key_frame_polygon_linewidth=3,  # Linewidth for expected heliostat polygon.
        key_frame_polygon_color='m',  # Color for expected heliostat polygon.
        key_frame_label_horizontalalignment='center',  # Horizontal alignment for heliostat label.
        key_frame_label_verticalalignment='center',  # Vertical alignment for heliostat label.
        key_frame_label_fontsize=10,  # Font size for heliostat label.
        key_frame_label_fontstyle='normal',  # Font style for heliostat label.
        key_frame_label_fontweight='bold',  # Font weight for heliostat label.
        key_frame_label_color='m',  # Color for heliostat label.
        key_frame_dpi=200,  # Dpi for saving figure to disk.
        key_frame_crop=False,  # Whether to crop annotations outside image frame.
    ):
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
    return RenderControlKeyFramesGivenManual()


def fast():
    return RenderControlKeyFramesGivenManual(draw_key_frames=False)
