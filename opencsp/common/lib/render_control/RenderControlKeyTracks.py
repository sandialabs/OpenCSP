"""


"""


class RenderControlKeyTracks():
    """
    Render control for the UFACET pipeline step KeyTracks.
    """

    def __init__(self, 
                 clear_previous=True,                           # Remove any existing files in the designated output directory.
                 draw_key_tracks=True,                          # Whether to draw the key frame track figures.
                 key_tracks_points_marker='o',                  # Marker for key frame tracks.
                 key_tracks_points_markersize=1.5,              # Marker size for key frame tracks.
                 key_tracks_points_color='m',                   # Color for key frame track points.
                 key_tracks_label_horizontalalignment='center', # Horizontal alignment for heliostat label.
                 key_tracks_label_verticalalignment='center',   # Vertical alignment for heliostat label.
                 key_tracks_label_fontsize=6,                   # Font size for heliostat label.
                 key_tracks_label_fontstyle='normal',           # Font style for heliostat label.
                 key_tracks_label_fontweight='bold',            # Font weight for heliostat label.
                 key_tracks_label_color='m',                    # Color for heliostat label.
                 key_tracks_dpi=200,                            # Dpi for saving figure to disk.
                 key_tracks_crop=False,                         # Whether to crop annotations outside image frame.
                 ):

        super(RenderControlKeyTracks, self).__init__()
        
        self.clear_previous                        = clear_previous
        self.draw_key_tracks                      = draw_key_tracks
        self.key_tracks_points_marker             = key_tracks_points_marker
        self.key_tracks_points_markersize         = key_tracks_points_markersize
        self.key_tracks_points_color              = key_tracks_points_color
        self.key_tracks_label_horizontalalignment = key_tracks_label_horizontalalignment
        self.key_tracks_label_verticalalignment   = key_tracks_label_verticalalignment
        self.key_tracks_label_fontsize            = key_tracks_label_fontsize
        self.key_tracks_label_fontstyle           = key_tracks_label_fontstyle
        self.key_tracks_label_fontweight          = key_tracks_label_fontweight
        self.key_tracks_label_color               = key_tracks_label_color
        self.key_tracks_dpi                       = key_tracks_dpi
        self.key_tracks_crop                      = key_tracks_crop


# COMMON CASES

def default():
    return RenderControlKeyTracks()

def fast():
    return RenderControlKeyTracks(draw_key_tracks=False)

