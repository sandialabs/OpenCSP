class RenderControlHeliostatTracks:
    """
    Render control for the UFACET pipeline step HeliostatTracks.
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_heliostat_tracks=True,  # Whether to draw the video track figures.
        heliostat_tracks_points_marker="o",  # Marker for video tracks.
        heliostat_tracks_points_markersize=1.5,  # Marker size for video tracks.
        heliostat_tracks_points_color="m",  # Color for video track points.
        heliostat_tracks_label_horizontalalignment="center",  # Horizontal alignment for heliostat label.
        heliostat_tracks_label_verticalalignment="center",  # Vertical alignment for heliostat label.
        heliostat_tracks_label_fontsize=6,  # Font size for heliostat label.
        heliostat_tracks_label_fontstyle="normal",  # Font style for heliostat label.
        heliostat_tracks_label_fontweight="bold",  # Font weight for heliostat label.
        heliostat_tracks_label_color="m",  # Color for heliostat label.
        heliostat_tracks_dpi=200,  # Dpi for saving figure to disk.
        heliostat_tracks_crop=True,  # Whether to crop annotations outside image frame.
    ):
        """
        Render control for the UFACET pipeline step HeliostatTracks.

        This class manages the rendering settings for the HeliostatTracks step in the UFACET pipeline,
        allowing customization of various visual elements related to heliostat tracks.

        Parameters
        ----------
        clear_previous : bool, optional
            Whether to remove any existing files in the designated output directory. By default, True.
        draw_heliostat_tracks : bool, optional
            Whether to draw the video track figures for heliostats. By default, True.
        heliostat_tracks_points_marker : str, optional
            Marker style for the video track points. By default, 'o'.
        heliostat_tracks_points_markersize : float, optional
            Size of the marker for video track points. By default, 1.5.
        heliostat_tracks_points_color : str, optional
            Color for the video track points. By default, 'm' (magenta).
        heliostat_tracks_label_horizontalalignment : str, optional
            Horizontal alignment for the heliostat label. By default, 'center'.
        heliostat_tracks_label_verticalalignment : str, optional
            Vertical alignment for the heliostat label. By default, 'center'.
        heliostat_tracks_label_fontsize : int, optional
            Font size for the heliostat label. By default, 6.
        heliostat_tracks_label_fontstyle : str, optional
            Font style for the heliostat label. By default, 'normal'.
        heliostat_tracks_label_fontweight : str, optional
            Font weight for the heliostat label. By default, 'bold'.
        heliostat_tracks_label_color : str, optional
            Color for the heliostat label. By default, 'm' (magenta).
        heliostat_tracks_dpi : int, optional
            DPI (dots per inch) for saving figures to disk. By default, 200.
        heliostat_tracks_crop : bool, optional
            Whether to crop annotations outside the image frame. By default, True.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlHeliostatTracks, self).__init__()

        self.clear_previous = clear_previous
        self.draw_heliostat_tracks = draw_heliostat_tracks
        self.heliostat_tracks_points_marker = heliostat_tracks_points_marker
        self.heliostat_tracks_points_markersize = heliostat_tracks_points_markersize
        self.heliostat_tracks_points_color = heliostat_tracks_points_color
        self.heliostat_tracks_label_horizontalalignment = heliostat_tracks_label_horizontalalignment
        self.heliostat_tracks_label_verticalalignment = heliostat_tracks_label_verticalalignment
        self.heliostat_tracks_label_fontsize = heliostat_tracks_label_fontsize
        self.heliostat_tracks_label_fontstyle = heliostat_tracks_label_fontstyle
        self.heliostat_tracks_label_fontweight = heliostat_tracks_label_fontweight
        self.heliostat_tracks_label_color = heliostat_tracks_label_color
        self.heliostat_tracks_dpi = heliostat_tracks_dpi
        self.heliostat_tracks_crop = heliostat_tracks_crop


# COMMON CASES


def default(color='m'):
    """
    Create a default render control for heliostat tracks.

    This function returns a `RenderControlHeliostatTracks` instance with default settings,
    using the specified color for the points and labels.

    Parameters
    ----------
    color : str, optional
        Color for the video track points and heliostat labels. By default, 'm' (magenta).

    Returns
    -------
    RenderControlHeliostatTracks
        An instance of `RenderControlHeliostatTracks` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlHeliostatTracks(heliostat_tracks_points_color=color, heliostat_tracks_label_color=color)


def fast():
    """
    Create a fast render control for heliostat tracks.

    This function returns a `RenderControlHeliostatTracks` instance configured to skip
    drawing the heliostat tracks, which can speed up the rendering process.

    Returns
    -------
    RenderControlHeliostatTracks
        An instance of `RenderControlHeliostatTracks` configured for fast rendering.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlHeliostatTracks(draw_heliostat_tracks=False)
