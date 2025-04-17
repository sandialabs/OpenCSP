import opencsp.common.lib.render.Color as cl


class RenderControlPointSeq:
    """
    Render control for sequences of points.

    This class controls the style of point markers and lines connecting points.

    Choices for line styles and colors can be found in the Matplotlib documentation:
    - Line Styles: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    Line Styles
    -----------
    '-'  solid line style
    '--'  dashed line style
    '-.'  dash-dot line style
    ':'   dotted line style

    Colors
    ------
    'b'  blue
    'g'  green
    'r'  red
    'c'  cyan
    'm'  magenta
    'y'  yellow
    'k'  black
    'w'  white

    For more colors, see:
    https://matplotlib.org/stable/api/colors_api.html#module-matplotlib.colors

    Markers
    -------
    '.'   point marker
    ','   pixel marker
    'o'   circle marker
    'v'   triangle_down marker
    '^'   triangle_up marker
    '<'   triangle_left marker
    '>'   triangle_right marker
    '1'   tri_down marker
    '2'   tri_up marker
    '3'   tri_left marker
    '4'   tri_right marker
    '8'   octagon marker
    's'   square marker
    'p'   pentagon marker
    'P'   plus (filled) marker
    '*'   star marker
    'h'   hexagon1 marker
    'H'   hexagon2 marker
    '+'   plus marker
    'x'   x marker
    'X'   x (filled) marker
    'D'   diamond marker
    'd'   thin_diamond marker
    '|'   vline marker
    '_'   hline marker
    'None' no marker
    '$\u266b$' two quarter notes
    'arrow' draws an arrow at the end of every line

    For more markers, see:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
    """

    # ChatGPT 4o-mini assisted with generating this doc string
    def __init__(
        self,  # See above for details:
        linestyle='-',  # '-', '--', '-.', ':', '' or 'None'
        linewidth: float = 1,  # float
        color: str | cl.Color = 'b',  # line color
        marker='x',  # .,ov^<>12348sp*hH+xXDd|_ or None
        markersize: float = 6,  # float
        markeredgecolor: str | cl.Color = None,  # Defaults to color above if not set.
        markeredgewidth=None,  # Defaults to linewidth if not set.
        markerfacecolor: str | cl.Color = None,  # Defaults to color above if not set.
        markeralpha: float | None = None,
        vector_color: str | cl.Color = 'b',  # Used if points are in a vector field.
        vector_linewidth: float = 1,  # Used if points are in a vector field.
        vector_scale: float = 1.0,  # Facter to grow/srhink vector length, for points in a vector field.
    ):
        """
        Initialize the rendering control for point sequences.

        Parameters
        ----------
        linestyle : str, optional
            Determines how lines are drawn. One of '-', '--', '-.', ':', '' or 'None'. Default is '-'.
        linewidth : float, optional
            Width of lines in pixels. Default is 1.
        color : str | Color, optional
            The primary color used for everything that doesn't have a color specified. Default is 'b'.
        marker : str | None, optional
            The style of marker to use. See the class description for more information. Default is 'x'.
        markersize : float, optional
            Size of the marker in pixels. Default is 6.
        markeredgecolor : str | Color | None, optional
            The color of the marker edges. Default is the same as `color`.
        markeredgewidth : float | None, optional
            Width of the marker edge in pixels. Defaults to `linewidth`.
        markerfacecolor : str | Color | None, optional
            The color of the marker faces. Default is the same as `color`.
        markeralpha : float | None, optional
            The alpha value (transparency) for the markers, where 0 is fully transparent and 1 is fully opaque. Default is None.
        vector_color : str | Color | None, optional
            The color for vectors. Only applies to points in a vector field. Default is 'b'.
        vector_linewidth : float, optional
            The line width for vectors, in pixels. Only applies to points in a vector field. Default is 1.
        vector_scale : float, optional
            Factor to grow/shrink vector length. Only applies to points in a vector field. Default is 1.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        super(RenderControlPointSeq, self).__init__()

        # Set defaults.
        if markeredgecolor is None:
            markeredgecolor = color
        if markeredgewidth is None:
            markeredgewidth = linewidth
        if markerfacecolor is None:
            markerfacecolor = color

        # Set fields.
        self.linestyle = linestyle
        self.linewidth = linewidth
        self._color = color
        self.marker = marker
        self.markersize = markersize
        self._markeredgecolor = markeredgecolor
        self.markeredgewidth = markeredgewidth
        self._markerfacecolor = markerfacecolor
        self.markeralpha = markeralpha
        self._vector_color = vector_color
        self.vector_linewidth = vector_linewidth
        self.vector_scale = vector_scale

        self._standardize_color_values()

    @property
    def color(self) -> tuple[float, float, float, float] | None:
        """
        Get the RGBA color value for the primary color.

        Returns
        -------
        tuple[float, float, float, float] | None
            The RGBA color value or None if not set.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        if self._color is not None:
            return self._color.rgba()

    @property
    def markeredgecolor(self) -> tuple[float, float, float, float] | None:
        """
        Get the RGBA color value for the marker edge color.

        Returns
        -------
        tuple[float, float, float, float] | None
            The RGBA color value or None if not set.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        if self._markeredgecolor is not None:
            if self.markeralpha is not None:
                return self._markeredgecolor.rgba(self.markeralpha)

    @property
    def markerfacecolor(self) -> tuple[float, float, float, float] | None:
        """
        Get the RGBA color value for the marker face color.

        Returns
        -------
        tuple[float, float, float, float] | None
            The RGBA color value or None if not set.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        if self._markerfacecolor is not None:
            if self.markeralpha is not None:
                return self._markerfacecolor.rgba(self.markeralpha)

    @property
    def vector_color(self) -> tuple[float, float, float, float] | None:
        """
        Get the RGBA color value for the vector color.

        Returns
        -------
        tuple[float, float, float, float] | None
            The RGBA color value or None if not set.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        if self._vector_color is not None:
            return self._vector_color.rgba()

    def set_color(self, color):
        """
        Set the primary color and update marker edge and face colors.

        Parameters
        ----------
        color : str | Color
            The new color to set for the primary color and marker colors.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        self._color = color
        self._markeredgecolor = color
        self._markerfacecolor = color

        self._standardize_color_values()

    def _standardize_color_values(self):
        """Standardize color values to the Color class."""
        # ChatGPT 4o-mini assisted with generating this doc string
        self._color = cl.Color.convert(self._color)
        self._markeredgecolor = cl.Color.convert(self._markeredgecolor)
        self._markerfacecolor = cl.Color.convert(self._markerfacecolor)
        self._vector_color = cl.Color.convert(self._vector_color)


# COMMON CASES


def default(marker='.', color='b', linewidth=1, markersize=8):
    """
    Create a default render control for point sequences.

    This function returns a `RenderControlPointSeq` instance with default settings.

    Parameters
    ----------
    marker : str, optional
        Marker style for the points. By default, '.' (point marker).
    color : str, optional
        Color for the points. By default, 'b' (blue).
    linewidth : float, optional
        Line width for connecting lines. By default, 1.
    markersize : float, optional
        Size of the marker in pixels. By default, 8.

    Returns
    -------
    RenderControlPointSeq
        An instance of `RenderControlPointSeq` configured with default parameters.
    """
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker=marker, markersize=markersize)
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker=marker, markersize=markersize)


def outline(color='k', linewidth=1):
    """
    Create a render control for outlines of physical objects.

    This function returns a `RenderControlPointSeq` instance configured to draw outlines only.

    Parameters
    ----------
    color : str, optional
        Color for the outlines. By default, 'k' (black).
    linewidth : float, optional
        Line width for the outlines. By default, 1.

    Returns
    -------
    RenderControlPointSeq
        An instance of `RenderControlPointSeq` configured to display outlines only.
    """
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker='None')
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker='None')


def data_curve(color='b', linewidth=1, marker='.', markersize=3) -> RenderControlPointSeq:
    """
    Create a render control for a data curve with identified data points.

    This function returns a `RenderControlPointSeq` instance configured to draw a data curve
    with specified data points.

    Parameters
    ----------
    color : str, optional
        Color for the data curve. By default, 'b' (blue).
    linewidth : float, optional
        Line width for the data curve. By default, 1.
    marker : str, optional
        Marker style for the data points. By default, '.' (point marker).
    markersize : float, optional
        Size of the marker in pixels. By default, 3.

    Returns
    -------
    RenderControlPointSeq
        An instance of `RenderControlPointSeq` configured for a data curve.
    """
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker=marker, markersize=markersize)
    return RenderControlPointSeq(linestyle='-', linewidth=linewidth, color=color, marker=marker, markersize=markersize)


def marker(marker='o', color='b', markersize=3) -> RenderControlPointSeq:
    """
    Create a render control for displaying markers.

    This function returns a `RenderControlPointSeq` instance configured to display markers.

    Parameters
    ----------
    marker : str, optional
        Marker style for the points. By default, 'o' (circle marker).
    color : str, optional
        Color for the markers. By default, 'b' (blue).
    markersize : float, optional
        Size of the marker in pixels. By default, 3.

    Returns
    -------
    RenderControlPointSeq
        An instance of `RenderControlPointSeq` configured to display markers.
    """
    return RenderControlPointSeq(linestyle='None', color=color, marker=marker, markersize=markersize)
    return RenderControlPointSeq(linestyle='None', color=color, marker=marker, markersize=markersize)


def vector_field(marker='.', color='b', markersize=3, vector_linewidth=1, vector_scale=1.0) -> RenderControlPointSeq:
    """
    Create a render control for a field of vector needles.

    This function returns a `RenderControlPointSeq` instance configured to draw a field of vectors.

    Parameters
    ----------
    marker : str, optional
        Marker style for the points. By default, '.' (point marker).
    color : str, optional
        Color for the vector needles. By default, 'b' (blue).
    markersize : float, optional
        Size of the marker in pixels. By default, 3.
    vector_linewidth : float, optional
        Line width for the vector needles, in pixels. By default, 1.
    vector_scale : float, optional
        Factor to grow/shrink vector length. By default, 1.0.

    Returns
    -------
    RenderControlPointSeq
        An instance of `RenderControlPointSeq` configured for a vector field.
    """
    # ChatGPT 4o-mini assisted with generating this doc string
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
    """
    Create a thin render control point sequence.

    Parameters
    ----------
    marker : str, optional
        The marker style for the control points. Default is ','.
    linewidth : float, optional
        The width of the lines used to render the control points. Default is 0.3.
    color : str, optional
        The color of the control points. Default is 'y'.

    Returns
    -------
    RenderControlPointSeq
        An instance of RenderControlPointSeq configured with the specified 
        marker, linewidth, and color.
    """
    # ChatGPT 4o-mini assisted with writing this doc string
    return RenderControlPointSeq(color=color, marker=marker, linewidth=linewidth)


def thick(marker=',', linewidth=1.5, color='y') -> RenderControlPointSeq:
    """
    Create a thick render control point sequence.

    Parameters
    ----------
    marker : str, optional
        The marker style for the control points. Default is ','.
    linewidth : float, optional
        The width of the lines used to render the control points. Default is 1.5.
    color : str, optional
        The color of the control points. Default is 'y'.

    Returns
    -------
    RenderControlPointSeq
        An instance of RenderControlPointSeq configured with the specified 
        marker, linewidth, and color.
    """
    # ChatGPT 4o-mini assisted with writing this doc string
    return RenderControlPointSeq(color=color, marker=marker, linewidth=linewidth)
