"""


"""


class RenderControlHeliostats3d:
    """
    Render control for the UFACET pipeline step Heliostats3d.
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_heliostats_3d=True,  # Whether to draw the video track figures.
        heliostats_3d_points_marker='o',  # Marker for video tracks.
        heliostats_3d_points_markersize=1.5,  # Marker size for video tracks.
        heliostats_3d_points_color='m',  # Color for video track points.
        heliostats_3d_label_horizontalalignment='center',  # Horizontal alignment for heliostat label.
        heliostats_3d_label_verticalalignment='center',  # Vertical alignment for heliostat label.
        heliostats_3d_label_fontsize=6,  # Font size for heliostat label.
        heliostats_3d_label_fontstyle='normal',  # Font style for heliostat label.
        heliostats_3d_label_fontweight='bold',  # Font weight for heliostat label.
        heliostats_3d_label_color='m',  # Color for heliostat label.
        heliostats_3d_dpi=200,  # Dpi for saving figure to disk.
        heliostats_3d_crop=True,  # Whether to crop annotations outside image frame.
    ):
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
    return RenderControlHeliostats3d(heliostats_3d_points_color=color, heliostats_3d_label_color=color)


def fast():
    return RenderControlHeliostats3d(draw_heliostats_3d=False)
