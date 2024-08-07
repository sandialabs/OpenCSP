"""


"""

import opencsp.common.lib.render.color as clr


class RenderControlPointSeq:
    """
    Render control for sequences of points.

    Controls style of point markers and lines connecting points.

    Choices from:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

        Line Styles
        -----------
        '-' 	solid line style
        '--' 	dashed line style
        '-.' 	dash-dot line style
        ':' 	dotted line style


        Colors
        ------
        'b' 	blue
        'g' 	green
        'r' 	red
        'c' 	cyan
        'm' 	magenta
        'y' 	yellow
        'k' 	black
        'w' 	white

        For more colors, see:
          https://matplotlib.org/stable/api/colors_api.html#module-matplotlib.colors

        Markers
        -------
        '.' 	   point marker
        ',' 	   pixel marker
        'o' 	   circle marker
        'v' 	   triangle_down marker
        '^' 	   triangle_up marker
        '<' 	   triangle_left marker
        '>' 	   triangle_right marker
        '1' 	   tri_down marker (three lines from the center to points on 30, 150, and 270 degrees)
        '2' 	   tri_up marker (three lines from the center to points on 90, 210, and 330 degrees)
        '3' 	   tri_left marker (three lines from the center to points on 60, 180, and 300 degrees)
        '4' 	   tri_right marker (three lines from the center to points on 0, 120, and 240 degrees)
        '8' 	   octagon marker
        's' 	   square marker
        'p' 	   pentagon marker
        'P' 	   plus (filled) marker
        '*' 	   star marker
        'h' 	   hexagon1 marker
        'H' 	   hexagon2 marker
        '+' 	   plus marker
        'x' 	   x marker
        'X' 	   x (filled) marker
        'D' 	   diamond marker
        'd' 	   thin_diamond marker
        '|' 	   vline marker
        '_' 	   hline marker
        'None'     no marker
        '$\u266B$' two quarter notes
        'arrow'    draws an arrow at the end of every line

        For more markers, see:
         https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html

    """

    def __init__(
        self,  # See above for details:
        linestyle='-',  # '-', '--', '-.', ':', '' or 'None'
        linewidth: float = 1,  # float
        color: str | clr.Color = 'b',  # line color
        marker='x',  # .,ov^<>12348sp*hH+xXDd|_ or None
        markersize: float = 6,  # float
        markeredgecolor: str | clr.Color = None,  # Defaults to color above if not set.
        markeredgewidth=None,  # Defaults to linewidth if not set.
        markerfacecolor: str | clr.Color = None,  # Defaults to color above if not set.
        vector_color: str | clr.Color = 'b',  # Used if points are in a vector field.
        vector_linewidth: float = 1,  # Used if points are in a vector field.
        vector_scale: float = 1.0,  # Facter to grow/srhink vector length, for points in a vector field.
    ):
        """
        linestyle : str
            Determines how lines are drawn. One of '-', '--', '-.', ':', '' or
            'None'. Default is '-' (solid line).
        linewidth : float
            Width of lines in the number of pixels. Default is 1.
        color : str | Color
            The primary color use for everything that doesn't have a color
            specified. If a Color object then the rgb() value of the Color
            object will be used. Default is 'b'.
        marker : str | None, optional
            The style of marker to use. See the class description for more
            information. Default is point '.'.
        markersize : float, optional
            Size of the marker, in pixels. Default is 6.
        markeredgecolor : str | Color | None, optional
            The color of the marker edges. Default is 'color'.
        markeredgewidth : float | None, optional
            Width of the marker edge in pixels. Defaults is 'linewidth'.
        markerfacecolor : str | Color | None, optional
            The color of the marker faces. Default is 'color'.
        vector_color : str | Color | None, optional
            The color for vectors. Only applies to points in a vector field.
            Default is 'b'.
        vector_linewidth : float, optional
            The line width for vectors, in pixels. Only applies to points in a
            vector field. default is 1.
        vector_scale : float, optional
            Facter to grow/srhink vector length. Only applies to points in a
            vector field. Default is 1.
        """
        super(RenderControlPointSeq, self).__init__()

        # Convert color values
        if isinstance(color, clr.Color):
            color = color.rgb()
        if isinstance(markeredgecolor, clr.Color):
            markeredgecolor = markeredgecolor.rgb()
        if isinstance(markerfacecolor, clr.Color):
            markerfacecolor = markerfacecolor.rgb()
        if isinstance(vector_color, clr.Color):
            vector_color = vector_color.rgb()

        # Set defaults.
        if markeredgecolor == None:
            markeredgecolor = color
        if markeredgewidth == None:
            markeredgewidth = linewidth
        if markerfacecolor == None:
            markerfacecolor = color

        # Set fields.
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.color = color
        self.marker = marker
        self.markersize = markersize
        self.markeredgecolor = markeredgecolor
        self.markeredgewidth = markeredgewidth
        self.markerfacecolor = markerfacecolor
        self.vector_color = vector_color
        self.vector_linewidth = vector_linewidth
        self.vector_scale = vector_scale

    # MODIFICATION

    def set_color(self, color):
        self.color = color
        self.markeredgecolor = color
        self.markerfacecolor = color


# COMMON CASES


def default(marker='.', color='b', linewidth=1, markersize=8):
    """
    What to draw if no particular preference is expressed.
    """
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker=marker, markersize=markersize)


def outline(color='k', linewidth=1):
    """
    Outlines of physical objects.
    """
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker='None')


def data_curve(color='b', linewidth=1, marker='.', markersize=3) -> RenderControlPointSeq:
    """
    A data curve with data points identified.
    """
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker=marker, markersize=markersize)


def marker(marker='o', color='b', markersize=3) -> RenderControlPointSeq:
    """
    A data curve with data points identified.
    """
    return RenderControlPointSeq(linestyle='None', color=color, marker=marker, markersize=markersize)


def vector_field(marker='.', color='b', markersize=3, vector_linewidth=1, vector_scale=1.0) -> RenderControlPointSeq:
    """
    A field of vector needles.
    """
    return RenderControlPointSeq(
        linestyle='None',
        color=color,
        marker=marker,
        markersize=markersize,
        vector_color=color,
        vector_linewidth=vector_linewidth,
        vector_scale=vector_scale,
    )


def thin(marker=',', linewidth=0.3, color='y') -> RenderControlPointSeq:
    return RenderControlPointSeq(color=color, marker=marker, linewidth=linewidth)
