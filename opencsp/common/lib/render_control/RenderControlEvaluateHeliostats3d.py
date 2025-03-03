class RenderControlEvaluateHeliostats3d:
    """
    Render control for the UFACET pipeline step EvaluateHeliostats3d.
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_evaluate_heliostats_3d=True,  # Whether to draw the video track figures.
        evaluate_heliostats_3d_points_marker="o",  # Marker for video tracks.
        evaluate_heliostats_3d_points_markersize=1.5,  # Marker size for video tracks.
        evaluate_heliostats_3d_points_color="m",  # Color for video track points.
        evaluate_heliostats_3d_label_horizontalalignment="center",  # Horizontal alignment for heliostat label.
        evaluate_heliostats_3d_label_verticalalignment="center",  # Vertical alignment for heliostat label.
        evaluate_heliostats_3d_label_fontsize=6,  # Font size for heliostat label.
        evaluate_heliostats_3d_label_fontstyle="normal",  # Font style for heliostat label.
        evaluate_heliostats_3d_label_fontweight="bold",  # Font weight for heliostat label.
        evaluate_heliostats_3d_label_color="m",  # Color for heliostat label.
        evaluate_heliostats_3d_dpi=200,  # Dpi for saving figure to disk.
        evaluate_heliostats_3d_crop=True,  # Whether to crop annotations outside image frame.
    ):
        """
        Render control for the UFACET pipeline step EvaluateHeliostats3d.

        This class manages the rendering settings for the EvaluateHeliostats3d step in the UFACET pipeline,
        allowing customization of various visual elements related to heliostat evaluation.

        Parameters
        ----------
        clear_previous : bool, optional
            Whether to remove any existing files in the designated output directory. By default, True.
        draw_evaluate_heliostats_3d : bool, optional
            Whether to draw the video track figures. By default, True.
        evaluate_heliostats_3d_points_marker : str, optional
            Marker style for video track points. By default, 'o'.
        evaluate_heliostats_3d_points_markersize : float, optional
            Size of the marker for video track points. By default, 1.5.
        evaluate_heliostats_3d_points_color : str, optional
            Color for video track points. By default, 'm' (magenta).
        evaluate_heliostats_3d_label_horizontalalignment : str, optional
            Horizontal alignment for heliostat labels. By default, 'center'.
        evaluate_heliostats_3d_label_verticalalignment : str, optional
            Vertical alignment for heliostat labels. By default, 'center'.
        evaluate_heliostats_3d_label_fontsize : int, optional
            Font size for heliostat labels. By default, 6.
        evaluate_heliostats_3d_label_fontstyle : str, optional
            Font style for heliostat labels. By default, 'normal'.
        evaluate_heliostats_3d_label_fontweight : str, optional
            Font weight for heliostat labels. By default, 'bold'.
        evaluate_heliostats_3d_label_color : str, optional
            Color for heliostat labels. By default, 'm' (magenta).
        evaluate_heliostats_3d_dpi : int, optional
            DPI for saving figures to disk. By default, 200.
        evaluate_heliostats_3d_crop : bool, optional
            Whether to crop annotations outside the image frame. By default, True.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlEvaluateHeliostats3d, self).__init__()

        self.clear_previous = clear_previous
        self.draw_evaluate_heliostats_3d = draw_evaluate_heliostats_3d
        self.evaluate_heliostats_3d_points_marker = evaluate_heliostats_3d_points_marker
        self.evaluate_heliostats_3d_points_markersize = evaluate_heliostats_3d_points_markersize
        self.evaluate_heliostats_3d_points_color = evaluate_heliostats_3d_points_color
        self.evaluate_heliostats_3d_label_horizontalalignment = evaluate_heliostats_3d_label_horizontalalignment
        self.evaluate_heliostats_3d_label_verticalalignment = evaluate_heliostats_3d_label_verticalalignment
        self.evaluate_heliostats_3d_label_fontsize = evaluate_heliostats_3d_label_fontsize
        self.evaluate_heliostats_3d_label_fontstyle = evaluate_heliostats_3d_label_fontstyle
        self.evaluate_heliostats_3d_label_fontweight = evaluate_heliostats_3d_label_fontweight
        self.evaluate_heliostats_3d_label_color = evaluate_heliostats_3d_label_color
        self.evaluate_heliostats_3d_dpi = evaluate_heliostats_3d_dpi
        self.evaluate_heliostats_3d_crop = evaluate_heliostats_3d_crop


# COMMON CASES


def default(color='m'):
    """
    Create a default render control for evaluating heliostats in 3D.

    This function returns a `RenderControlEvaluateHeliostats3d` instance with default settings,
    using the specified color for the points and labels.

    Parameters
    ----------
    color : str, optional
        Color for the video track points and heliostat labels. By default, 'm' (magenta).

    Returns
    -------
    RenderControlEvaluateHeliostats3d
        An instance of `RenderControlEvaluateHeliostats3d` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlEvaluateHeliostats3d(
        evaluate_heliostats_3d_points_color=color, evaluate_heliostats_3d_label_color=color
    )


def fast():
    """
    Create a fast render control for evaluating heliostats in 3D.

    This function returns a `RenderControlEvaluateHeliostats3d` instance configured to skip
    drawing the video track figures, which can speed up the rendering process.

    Returns
    -------
    RenderControlEvaluateHeliostats3d
        An instance of `RenderControlEvaluateHeliostats3d` configured for fast rendering.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlEvaluateHeliostats3d(draw_evaluate_heliostats_3d=False)
