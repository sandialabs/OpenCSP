"""


"""


class RenderControlVideoTracks:
    """
    Render control for the UFACET pipeline step VideoTracks.
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_video_tracks=True,  # Whether to draw the video track figures.
        video_tracks_points_marker="o",  # Marker for video tracks.
        video_tracks_points_markersize=1.5,  # Marker size for video tracks.
        video_tracks_points_color="m",  # Color for video track points.
        video_tracks_label_horizontalalignment="center",  # Horizontal alignment for heliostat label.
        video_tracks_label_verticalalignment="center",  # Vertical alignment for heliostat label.
        video_tracks_label_fontsize=6,  # Font size for heliostat label.
        video_tracks_label_fontstyle="normal",  # Font style for heliostat label.
        video_tracks_label_fontweight="bold",  # Font weight for heliostat label.
        video_tracks_label_color="m",  # Color for heliostat label.
        video_tracks_dpi=200,  # Dpi for saving figure to disk.
        video_tracks_crop=True,  # Whether to crop annotations outside image frame.
    ):
        super(RenderControlVideoTracks, self).__init__()

        self.clear_previous = clear_previous
        self.draw_video_tracks = draw_video_tracks
        self.video_tracks_points_marker = video_tracks_points_marker
        self.video_tracks_points_markersize = video_tracks_points_markersize
        self.video_tracks_points_color = video_tracks_points_color
        self.video_tracks_label_horizontalalignment = video_tracks_label_horizontalalignment
        self.video_tracks_label_verticalalignment = video_tracks_label_verticalalignment
        self.video_tracks_label_fontsize = video_tracks_label_fontsize
        self.video_tracks_label_fontstyle = video_tracks_label_fontstyle
        self.video_tracks_label_fontweight = video_tracks_label_fontweight
        self.video_tracks_label_color = video_tracks_label_color
        self.video_tracks_dpi = video_tracks_dpi
        self.video_tracks_crop = video_tracks_crop


# COMMON CASES


def default(color="m"):
    return RenderControlVideoTracks(video_tracks_points_color=color, video_tracks_label_color=color)


def fast():
    return RenderControlVideoTracks(draw_video_tracks=False)
