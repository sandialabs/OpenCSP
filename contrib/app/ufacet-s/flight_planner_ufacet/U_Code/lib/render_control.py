"""
Render Control Container.

Provides a single container to pass around display parameters, such as line width, color, 
figure size, etc.  This eliminates the need to pass many parameters through a series of 
routine signatures.

Implemented as a dictionary, so applications can add dresised fields without modifying this code.

Features include:
  Figure management parameters, such as whether to tile the figures, the tile array, etc.



"""

from opencsp.common.lib.csp.Facet import Facet


class RenderControlFigure:
    """Facet class
    *Assuming rectangular facet*

    [Args]:
        name    [str]: facet's name
        centroid_offset [list[float]] = facet's centroid offset from centered facet's centroid - Facing up
        width   [float]: facet width in meters
        height  [float]: facet height in meters
    """

    def __init__(self, name, centroid_offset=[], width=0, height=0):
        super(Facet, self).__init__()

        self.name = name
        self.centroid_offset = centroid_offset
        self.width = width
        self.height = height

        # Facet Corners [offsets in terms of facet's centoid]
        self.top_left_corner_offset = [-width / 2, height / 2, 0]
        self.top_right_corner_offset = [width / 2, height / 2, 0]
        self.bottom_right_corner_offset = [width / 2, -height / 2, 0]
        self.bottom_left_corner_offset = [-width / 2, -height / 2, 0]

        # if additional information (backside structure, bolt locations, etc) is needed
        # Fill in here


def initialize_render_control(
    tile=True,  # True => Lay out figures in grid.  False => Place at upper_left or default screen center.
    tile_array=(3, 2),  # (n_x, n_y)
    tile_square=False,  # Set to True for equal-axis 3d plots.
    figsize=(6.4, 4.8),  # inch.
    upper_left_xy=None,  # pixel.  (0,0) --> Upper left corner of screen.
    grid=True,
):  # Whether or not to draw grid lines.
    render_control = {}
    # Figure management.
    render_control["figure_names"] = []
    render_control["tile"] = tile
    render_control["tile_array"] = tile_array
    render_control["tile_square"] = tile_square

    # Figure size and placement.
    render_control["figsize"] = figsize
    render_control["upper_left_xy"] = upper_left_xy

    # Axis control.
    render_control["x_label"] = "x (m)"
    render_control["y_label"] = "y (m)"
    render_control["z_label"] = "z (m)"
    render_control["grid"] = grid

    # Return.
    return render_control
