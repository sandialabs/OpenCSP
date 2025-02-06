""" """


class RenderControlEvaluateHeliostats3d:
    """
    Render control for the UFACET pipeline step EvaluateHeliostats3d.
    """

    def __init__(
        self,
        clear_previous=True,  # Remove any existing files in the designated output directory.
        draw_evaluate_heliostats_3d=True,  # Whether to draw the video track figures.
        evaluate_heliostats_3d_points_marker='o',  # Marker for video tracks.
        evaluate_heliostats_3d_points_markersize=1.5,  # Marker size for video tracks.
        evaluate_heliostats_3d_points_color='m',  # Color for video track points.
        evaluate_heliostats_3d_label_horizontalalignment='center',  # Horizontal alignment for heliostat label.
        evaluate_heliostats_3d_label_verticalalignment='center',  # Vertical alignment for heliostat label.
        evaluate_heliostats_3d_label_fontsize=6,  # Font size for heliostat label.
        evaluate_heliostats_3d_label_fontstyle='normal',  # Font style for heliostat label.
        evaluate_heliostats_3d_label_fontweight='bold',  # Font weight for heliostat label.
        evaluate_heliostats_3d_label_color='m',  # Color for heliostat label.
        evaluate_heliostats_3d_dpi=200,  # Dpi for saving figure to disk.
        evaluate_heliostats_3d_crop=True,  # Whether to crop annotations outside image frame.
    ):
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
    return RenderControlEvaluateHeliostats3d(
        evaluate_heliostats_3d_points_color=color, evaluate_heliostats_3d_label_color=color
    )


def fast():
    return RenderControlEvaluateHeliostats3d(draw_evaluate_heliostats_3d=False)
