import opencsp.common.lib.render.Color as cl


class RenderControlText:
    """
    Render control for text labels added to plots.

    This class manages the rendering of text labels in plots, allowing customization of font styles,
    colors, and alignments.

    Primary parameters from:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html

    Font Weight
    -----------
    A numeric value in range 0-1000
    'ultralight'
    'light'
    'normal'
    'regular'
    'book'
    'medium'
    'roman'
    'semibold'
    'demibold'
    'demi'
    'bold'
    'heavy'
    'extra bold'
    'black'

    zdir choices from:
    https://matplotlib.org/stable/gallery/mplot3d/text3d.html

    Color choices from:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

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
    """

    # ChatGPT 4o-mini assisted with generating this doc string
    def __init__(
        self,
        horizontalalignment: str = "center",
        verticalalignment: str = "center",
        fontsize: str | float = "medium",
        fontstyle: str = "normal",
        fontweight: int | str = "normal",
        zdir: str | tuple[int, int, int] | None = None,
        color: str | cl.Color = "b",
        rotation: float = 0,
    ):
        """
        Controls for how text gets rendered.

        Parameters
        ----------
        horizontalalignment : str, optional
            Horizontal alignment, one of 'center', 'right', 'left'. Default is 'center'.
        verticalalignment : str, optional
            Vertical alignment, one of 'center', 'top', 'bottom', 'baseline', 'center_baseline'. Default is 'center'.
        fontsize : str | float, optional
            Font size, specified as a float or one of 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'. Default is 'medium'.
        fontstyle : str, optional
            Font style, one of 'normal', 'italic', 'oblique'. Default is 'normal'.
        fontweight : int | str, optional
            Font weight, specified as an integer (0-1000) or one of 'light', 'normal', 'bold'. Default is 'normal'.
        zdir : str | tuple[int, int, int] | None, optional
            Direction that is considered "up" when rendering in 3D. Can be 'x', 'y', 'z', or a direction such as (1,1,0), (1,1,1), or None for the default. Default is None.
        color : str | Color, optional
            Color of the text, specified as a matplotlib named color or a Color instance. Default is 'b' (blue).
        rotation : float, optional
            Orientation of the text in radians, where 0 is horizontal and pi/2 is vertical. Default is 0.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        super(RenderControlText, self).__init__()

        # Set fields.
        self.horizontalalignment = horizontalalignment
        self.verticalalignment = verticalalignment
        self.fontsize = fontsize
        self.fontstyle = fontstyle
        self.fontweight = fontweight
        self.zdir = zdir
        self._color = color
        self.rotation = rotation

        self._standardize_color_values()

    @property
    def color(self) -> tuple[float, float, float, float] | None:
        """
        Get the RGBA color value for the text.

        Returns
        -------
        tuple[float, float, float, float] | None
            The RGBA color value or None if not set.
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        if self._color is not None:
            return self._color.rgba()

    def _standardize_color_values(self):
        # Standardize color values to the Color class.
        self._color = cl.Color.convert(self._color)


def default(fontsize="medium", color="b"):
    """
    Create a default text rendering control.

    This function returns a `RenderControlText` instance with default settings.

    Parameters
    ----------
    fontsize : str | float, optional
        Font size for the text. Default is 'medium'.
    color : str, optional
        Color for the text. Default is 'b' (blue).

    Returns
    -------
    RenderControlText
        An instance of `RenderControlText` configured with default parameters.
    """
    # ChatGPT 4o-mini assisted with generating this doc string
    return RenderControlText(fontsize=fontsize, fontstyle="normal", fontweight="normal", zdir=None, color=color)


def bold(fontsize="medium", color="b"):
    """
    Create a bold text rendering control for emphasis.

    This function returns a `RenderControlText` instance configured for bold text.

    Parameters
    ----------
    fontsize : str | float, optional
        Font size for the text. Default is 'medium'.
    color : str, optional
        Color for the text. Default is 'b' (blue).

    Returns
    -------
    RenderControlText
        An instance of `RenderControlText` configured for bold text.
    """
    # ChatGPT 4o-mini assisted with generating this doc string
    return RenderControlText(fontsize=fontsize, fontstyle="normal", fontweight="bold", zdir=None, color=color)
