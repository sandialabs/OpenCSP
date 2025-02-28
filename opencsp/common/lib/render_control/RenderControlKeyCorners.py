class RenderControlKeyCorners:
    """
    Render control for the UFACET pipeline step KeyCorners.
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_key_corners=True,  # Whether to draw the key corner figures.
        key_corners_points_marker="o",  # Marker for key corners.
        key_corners_points_markersize=1.5,  # Marker size for key corners.
        key_corners_points_color="m",  # Color for key corner points.
        key_corners_label_horizontalalignment="center",  # Horizontal alignment for heliostat label.
        key_corners_label_verticalalignment="center",  # Vertical alignment for heliostat label.
        key_corners_label_fontsize=6,  # Font size for heliostat label.
        key_corners_label_fontstyle="normal",  # Font style for heliostat label.
        key_corners_label_fontweight="bold",  # Font weight for heliostat label.
        key_corners_label_color="m",  # Color for heliostat label.
        key_corners_dpi=200,  # Dpi for saving figure to disk.
        key_corners_crop=False,  # Whether to crop annotations outside image frame.
    ):
        """
        Render control for the UFACET pipeline step KeyCorners.

        This class manages the rendering settings for the KeyCorners step in the UFACET pipeline,
        allowing customization of various visual elements related to key corners.

        Parameters
        ----------
        clear_previous : bool, optional
            Whether to remove any existing files in the designated output directory. By default, True.
        draw_key_corners : bool, optional
            Whether to draw the key corner figures. By default, True.
        key_corners_points_marker : str, optional
            Marker style for the key corner points. By default, 'o'.
        key_corners_points_markersize : float, optional
            Size of the marker for key corner points. By default, 1.5.
        key_corners_points_color : str, optional
            Color for the key corner points. By default, 'm' (magenta).
        key_corners_label_horizontalalignment : str, optional
            Horizontal alignment for the key corner label. By default, 'center'.
        key_corners_label_verticalalignment : str, optional
            Vertical alignment for the key corner label. By default, 'center'.
        key_corners_label_fontsize : int, optional
            Font size for the key corner label. By default, 6.
        key_corners_label_fontstyle : str, optional
            Font style for the key corner label. By default, 'normal'.
        key_corners_label_fontweight : str, optional
            Font weight for the key corner label. By default, 'bold'.
        key_corners_label_color : str, optional
            Color for the key corner label. By default, 'm' (magenta).
        key_corners_dpi : int, optional
            DPI (dots per inch) for saving figures to disk. By default, 200.
        key_corners_crop : bool, optional
            Whether to crop annotations outside the image frame. By default, False.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlKeyCorners, self).__init__()

        self.clear_previous = clear_previous
        self.draw_key_corners = draw_key_corners
        self.key_corners_points_marker = key_corners_points_marker
        self.key_corners_points_markersize = key_corners_points_markersize
        self.key_corners_points_color = key_corners_points_color
        self.key_corners_label_horizontalalignment = key_corners_label_horizontalalignment
        self.key_corners_label_verticalalignment = key_corners_label_verticalalignment
        self.key_corners_label_fontsize = key_corners_label_fontsize
        self.key_corners_label_fontstyle = key_corners_label_fontstyle
        self.key_corners_label_fontweight = key_corners_label_fontweight
        self.key_corners_label_color = key_corners_label_color
        self.key_corners_dpi = key_corners_dpi
        self.key_corners_crop = key_corners_crop

        # Figure output.
        self.draw_img_box = True

        self.draw_edge_fig = True
        self.draw_edge_img_fig = False
        self.draw_edge_img = False
        self.draw_edge = True

        self.draw_sky_fig = True
        self.draw_sky_img_fig = False
        self.draw_sky_img = False
        self.draw_sky = True

        self.draw_skyhsv_fig = True
        self.draw_skyhsv_img_fig = False
        self.draw_skyhsv_img = False
        self.draw_skyhsv = True

        self.draw_boundaries_fig = True
        self.draw_boundaries = False

        self.draw_components_fig = True
        self.draw_components = False

        self.draw_filt_components_fig = True
        self.draw_filt_components = False

        self.draw_corners = True
        self.draw_facets = True
        self.draw_filtered_facets = True
        self.draw_filtered_heliostats = True
        self.draw_top_row_facets = True
        self.draw_top_row_facets_labels = True

        self.draw_confirmed_corners = True
        self.draw_projected_corners = True
        self.draw_projected_and_confirmed_corners = True

        # Csv file output.
        self.write_components = False
        self.write_filt_components = False
        self.write_fitted_lines_components = False
        self.write_fitted_lines_inliers_components = False
        self.write_top_left_corners = False
        self.write_top_right_corners = False
        self.write_bottom_left_corners = False
        self.write_bottom_right_corners = False
        self.write_facets = False
        self.write_all_confirmed_corners = True
        self.write_all_projected_corners = True
        self.write_confirmed_fnxl = True
        self.write_projected_fnxl = True


# COMMON CASES


def default():
    """
    Create a default render control for key corners.

    This function returns a `RenderControlKeyCorners` instance with default settings.

    Returns
    -------
    RenderControlKeyCorners
        An instance of `RenderControlKeyCorners` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlKeyCorners()


def fast():
    """
    Create a fast render control for key corners.

    This function returns a `RenderControlKeyCorners` instance configured to skip
    drawing the key corner figures, which can speed up the rendering process.

    Returns
    -------
    RenderControlKeyCorners
        An instance of `RenderControlKeyCorners` configured for fast rendering.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlKeyCorners(draw_key_corners=False)
