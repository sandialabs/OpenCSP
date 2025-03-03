class RenderControlHeliostats3d:
    """
    Render control for the UFACET pipeline step Heliostats3d.
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_heliostats_3d=True,  # Whether to draw the video track figures.
        heliostats_3d_points_marker="o",  # Marker for video tracks.
        heliostats_3d_points_markersize=1.5,  # Marker size for video tracks.
        heliostats_3d_points_color="m",  # Color for video track points.
        heliostats_3d_label_horizontalalignment="center",  # Horizontal alignment for heliostat label.
        heliostats_3d_label_verticalalignment="center",  # Vertical alignment for heliostat label.
        heliostats_3d_label_fontsize=6,  # Font size for heliostat label.
        heliostats_3d_label_fontstyle="normal",  # Font style for heliostat label.
        heliostats_3d_label_fontweight="bold",  # Font weight for heliostat label.
        heliostats_3d_label_color="m",  # Color for heliostat label.
        heliostats_3d_dpi=200,  # Dpi for saving figure to disk.
        heliostats_3d_crop=True,  # Whether to crop annotations outside image frame.
    ):
        """
        Render control for the UFACET pipeline step Heliostats3d.

        This class manages the rendering settings for the Heliostats3d step in the UFACET pipeline,
        allowing customization of various visual elements related to 3D heliostat tracks.

        Parameters
        ----------
        clear_previous : bool, optional
            Whether to remove any existing files in the designated output directory. By default, True.
        draw_heliostats_3d : bool, optional
            Whether to draw the 3D video track figures for heliostats. By default, True.
        heliostats_3d_points_marker : str, optional
            Marker style for the 3D video track points. By default, 'o'.
        heliostats_3d_points_markersize : float, optional
            Size of the marker for 3D video track points. By default, 1.5.
        heliostats_3d_points_color : str, optional
            Color for the 3D video track points. By default, 'm' (magenta).
        heliostats_3d_label_horizontalalignment : str, optional
            Horizontal alignment for the heliostat label. By default, 'center'.
        heliostats_3d_label_verticalalignment : str, optional
            Vertical alignment for the heliostat label. By default, 'center'.
        heliostats_3d_label_fontsize : int, optional
            Font size for the heliostat label. By default, 6.
        heliostats_3d_label_fontstyle : str, optional
            Font style for the heliostat label. By default, 'normal'.
        heliostats_3d_label_fontweight : str, optional
            Font weight for the heliostat label. By default, 'bold'.
        heliostats_3d_label_color : str, optional
            Color for the heliostat label. By default, 'm' (magenta).
        heliostats_3d_dpi : int, optional
            DPI (dots per inch) for saving figures to disk. By default, 200.
        heliostats_3d_crop : bool, optional
            Whether to crop annotations outside the image frame. By default, True.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlHeliostats3d, self).__init__()

        self.clear_previous = clear_previous
        self.draw_heliostats_3d = draw_heliostats_3d
        self.heliostats_3d_points_marker = heliostats_3d_points_marker
        self.heliostats_3d_points_markersize = heliostats_3d_points_markersize
        self.heliostats_3d_points_color = heliostats_3d_points_color
        self.heliostats_3d_label_horizontalalignment = heliostats_3d_label_horizontalalignment
        self.heliostats_3d_label_verticalalignment = heliostats_3d_label_verticalalignment
        self.heliostats_3d_label_fontsize = heliostats_3d_label_fontsize
        self.heliostats_3d_label_fontstyle = heliostats_3d_label_fontstyle
        self.heliostats_3d_label_fontweight = heliostats_3d_label_fontweight
        self.heliostats_3d_label_color = heliostats_3d_label_color
        self.heliostats_3d_dpi = heliostats_3d_dpi
        self.heliostats_3d_crop = heliostats_3d_crop


# COMMON CASES


def default(color='m'):
    """
    Create a default render control for 3D heliostats.

    This function returns a `RenderControlHeliostats3d` instance with default settings,
    using the specified color for the points and labels.

    Parameters
    ----------
    color : str, optional
        Color for the 3D video track points and heliostat labels. By default, 'm' (magenta).

    Returns
    -------
    RenderControlHeliostats3d
        An instance of `RenderControlHeliostats3d` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlHeliostats3d(heliostats_3d_points_color=color, heliostats_3d_label_color=color)


def fast():
    """
    Create a fast render control for 3D heliostats.

    This function returns a `RenderControlHeliostats3d` instance configured to skip
    drawing the 3D heliostat tracks, which can speed up the rendering process.

    Returns
    -------
    RenderControlHeliostats3d
        An instance of `RenderControlHeliostats3d` configured for fast rendering.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlHeliostats3d(draw_heliostats_3d=False)
