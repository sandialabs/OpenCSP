class RenderControlVideoTracks:
    """
    Render control for the UFACET pipeline step VideoTracks.

    This class manages the rendering of video tracks within the UFACET pipeline.
    It provides options for customizing the appearance of video track markers and
    labels, as well as controlling the output settings for the rendered figures.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

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
        """
        Render control for the UFACET pipeline step VideoTracks.

        This class manages the rendering of video tracks within the UFACET pipeline.
        It provides options for customizing the appearance of video track markers and
        labels, as well as controlling the output settings for the rendered figures.

        Parameters
        ----------
        clear_previous : bool, optional
            If True, removes any existing files in the designated output directory.
            Defaults to True.
        draw_video_tracks : bool, optional
            If True, draws the video track figures. Defaults to True.
        video_tracks_points_marker : str, optional
            Marker style for video track points. Defaults to 'o'.
        video_tracks_points_markersize : float, optional
            Size of the marker for video track points. Defaults to 1.5.
        video_tracks_points_color : str, optional
            Color for video track points. Defaults to 'm'.
        video_tracks_label_horizontalalignment : str, optional
            Horizontal alignment for heliostat labels. Defaults to 'center'.
        video_tracks_label_verticalalignment : str, optional
            Vertical alignment for heliostat labels. Defaults to 'center'.
        video_tracks_label_fontsize : int, optional
            Font size for heliostat labels. Defaults to 6.
        video_tracks_label_fontstyle : str, optional
            Font style for heliostat labels. Defaults to 'normal'.
        video_tracks_label_fontweight : str, optional
            Font weight for heliostat labels. Defaults to 'bold'.
        video_tracks_label_color : str, optional
            Color for heliostat labels. Defaults to 'm'.
        video_tracks_dpi : int, optional
            Dots per inch (DPI) for saving figures to disk. Defaults to 200.
        video_tracks_crop : bool, optional
            If True, crops annotations that are outside the image frame. Defaults to True.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
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


def default(color='m'):
    """
    Create a default instance of RenderControlVideoTracks with specified color.

    Parameters
    ----------
    color : str, optional
        Color for video track points and labels. Defaults to 'm'.

    Returns
    -------
    RenderControlVideoTracks
        An instance of RenderControlVideoTracks with default settings
        and the specified color.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlVideoTracks(video_tracks_points_color=color, video_tracks_label_color=color)


def fast():
    """
    Create a fast instance of RenderControlVideoTracks with drawing disabled.

    Returns
    -------
    RenderControlVideoTracks
        An instance of RenderControlVideoTracks with drawing of video tracks disabled.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlVideoTracks(draw_video_tracks=False)
