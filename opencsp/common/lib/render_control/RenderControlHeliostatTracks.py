"""


"""


class RenderControlHeliostatTracks():
    """
    Render control for the UFACET pipeline step HeliostatTracks.
    """

    def __init__(self, 
                 clear_previous=True,                                 # Remove any existing files in the designated output directory.
                 draw_heliostat_tracks=True,                          # Whether to draw the video track figures.
                 heliostat_tracks_points_marker='o',                  # Marker for video tracks.
                 heliostat_tracks_points_markersize=1.5,              # Marker size for video tracks.
                 heliostat_tracks_points_color='m',                   # Color for video track points.
                 heliostat_tracks_label_horizontalalignment='center', # Horizontal alignment for heliostat label.
                 heliostat_tracks_label_verticalalignment='center',   # Vertical alignment for heliostat label.
                 heliostat_tracks_label_fontsize=6,                   # Font size for heliostat label.
                 heliostat_tracks_label_fontstyle='normal',           # Font style for heliostat label.
                 heliostat_tracks_label_fontweight='bold',            # Font weight for heliostat label.
                 heliostat_tracks_label_color='m',                    # Color for heliostat label.
                 heliostat_tracks_dpi=200,                            # Dpi for saving figure to disk.
                 heliostat_tracks_crop=True,                          # Whether to crop annotations outside image frame.
                 ):

        super(RenderControlHeliostatTracks, self).__init__()
        
        self.clear_previous                             = clear_previous
        self.draw_heliostat_tracks                      = draw_heliostat_tracks
        self.heliostat_tracks_points_marker             = heliostat_tracks_points_marker
        self.heliostat_tracks_points_markersize         = heliostat_tracks_points_markersize
        self.heliostat_tracks_points_color              = heliostat_tracks_points_color
        self.heliostat_tracks_label_horizontalalignment = heliostat_tracks_label_horizontalalignment
        self.heliostat_tracks_label_verticalalignment   = heliostat_tracks_label_verticalalignment
        self.heliostat_tracks_label_fontsize            = heliostat_tracks_label_fontsize
        self.heliostat_tracks_label_fontstyle           = heliostat_tracks_label_fontstyle
        self.heliostat_tracks_label_fontweight          = heliostat_tracks_label_fontweight
        self.heliostat_tracks_label_color               = heliostat_tracks_label_color
        self.heliostat_tracks_dpi                       = heliostat_tracks_dpi
        self.heliostat_tracks_crop                      = heliostat_tracks_crop


# COMMON CASES

def default(color='m'):
    return RenderControlHeliostatTracks(heliostat_tracks_points_color=color,
                                        heliostat_tracks_label_color=color)

def fast():
    return RenderControlHeliostatTracks(draw_heliostat_tracks=False)

