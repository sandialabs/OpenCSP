class RenderControlKeyTracks:
    """
    Render control for the UFACET pipeline step KeyTracks.
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_key_tracks=True,  # Whether to draw the key frame track figures.
        key_tracks_points_marker="o",  # Marker for key frame tracks.
        key_tracks_points_markersize=1.5,  # Marker size for key frame tracks.
        key_tracks_points_color="m",  # Color for key frame track points.
        key_tracks_label_horizontalalignment="center",  # Horizontal alignment for heliostat label.
        key_tracks_label_verticalalignment="center",  # Vertical alignment for heliostat label.
        key_tracks_label_fontsize=6,  # Font size for heliostat label.
        key_tracks_label_fontstyle="normal",  # Font style for heliostat label.
        key_tracks_label_fontweight="bold",  # Font weight for heliostat label.
        key_tracks_label_color="m",  # Color for heliostat label.
        key_tracks_dpi=200,  # Dpi for saving figure to disk.
        key_tracks_crop=False,  # Whether to crop annotations outside image frame.
    ):
        """
        Render control for the UFACET pipeline step KeyTracks.

        This class manages the rendering settings for the KeyTracks step in the UFACET pipeline,
        allowing customization of various visual elements related to key frame tracks.

        Parameters
        ----------
        clear_previous : bool, optional
            Whether to remove any existing files in the designated output directory. By default, True.
        draw_key_tracks : bool, optional
            Whether to draw the key frame track figures. By default, True.
        key_tracks_points_marker : str, optional
            Marker style for the key frame track points. By default, 'o'.
        key_tracks_points_markersize : float, optional
            Size of the marker for key frame track points. By default, 1.5.
        key_tracks_points_color : str, optional
            Color for the key frame track points. By default, 'm' (magenta).
        key_tracks_label_horizontalalignment : str, optional
            Horizontal alignment for the key track label. By default, 'center'.
        key_tracks_label_verticalalignment : str, optional
            Vertical alignment for the key track label. By default, 'center'.
        key_tracks_label_fontsize : int, optional
            Font size for the key track label. By default, 6.
        key_tracks_label_fontstyle : str, optional
            Font style for the key track label. By default, 'normal'.
        key_tracks_label_fontweight : str, optional
            Font weight for the key track label. By default, 'bold'.
        key_tracks_label_color : str, optional
            Color for the key track label. By default, 'm' (magenta).
        key_tracks_dpi : int, optional
            DPI (dots per inch) for saving figures to disk. By default, 200.
        key_tracks_crop : bool, optional
            Whether to crop annotations outside the image frame. By default, False.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlKeyTracks, self).__init__()

        self.clear_previous = clear_previous
        self.draw_key_tracks = draw_key_tracks
        self.key_tracks_points_marker = key_tracks_points_marker
        self.key_tracks_points_markersize = key_tracks_points_markersize
        self.key_tracks_points_color = key_tracks_points_color
        self.key_tracks_label_horizontalalignment = key_tracks_label_horizontalalignment
        self.key_tracks_label_verticalalignment = key_tracks_label_verticalalignment
        self.key_tracks_label_fontsize = key_tracks_label_fontsize
        self.key_tracks_label_fontstyle = key_tracks_label_fontstyle
        self.key_tracks_label_fontweight = key_tracks_label_fontweight
        self.key_tracks_label_color = key_tracks_label_color
        self.key_tracks_dpi = key_tracks_dpi
        self.key_tracks_crop = key_tracks_crop


# COMMON CASES


def default():
    """
    Create a default render control for key tracks.

    This function returns a `RenderControlKeyTracks` instance with default settings.

    Returns
    -------
    RenderControlKeyTracks
        An instance of `RenderControlKeyTracks` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlKeyTracks()


def fast():
    """
    Create a fast render control for key tracks.

    This function returns a `RenderControlKeyTracks` instance configured to skip
    drawing the key frame track figures, which can speed up the rendering process.

    Returns
    -------
    RenderControlKeyTracks
        An instance of `RenderControlKeyTracks` configured for fast rendering.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlKeyTracks(draw_key_tracks=False)
