"""


"""

import opencsp.common.lib.render.Color as cl


class RenderControlText:
    """
    Render control for text labels added to plots.

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

    """

    def __init__(
        self,  # See above for details:
        horizontalalignment: str = 'center',  # center, right, left
        verticalalignment: str = 'center',  # center, top, bottom, baseline, center_baseline
        fontsize: str | float = 'medium',  # float or xx-small, x-small, small, medium, large, x-large, xx-large
        fontstyle: str = 'normal',  # normal, italic, oblique
        fontweight: int | str = 'normal',  # 0-1000, or light, normal, bold (see above for full list)
        zdir: str | tuple[int, int, int] | None = None,  # None, 'x', 'y', 'z', (1,1,0), (1,1,1), ...
        color: str | cl.Color = 'b',  # bgrcmykw (see above)
        rotation: float = 0,  # radians, 0=horizontal, pi/2=vertical
    ):
        """
        Controls for how text gets rendered.

        Parameters
        ----------
        horizontalalignment: str, optional
            Horizontal alignment, one of 'center', 'right', 'left'. Default is 'center'.
        verticalalignment: str, optional
            Vertical alignment, one of 'center', 'top', 'bottom', 'baseline', 'center_baseline'. Default is 'center'.
        fontsize: str | float, optional
            float or xx-small, x-small, small, medium, large, x-large, xx-large. Default is 'medium'.
        fontstyle: str, optional
            normal, italic, oblique. Default is 'normal'.
        fontweight: int | str, optional
            0-1000, or light, normal, bold (see above for full list). Default is 'normal'.
        zdir: str | tuple[int, int, int] | None, optional
            Which direction is up when rendering in 3d. One of 'x', 'y', 'z', or
            a direction such as (1,1,0), (1,1,1), ..., or None for the
            matplotlib default. Default is None.
        color: str | Color, optional
            Color of the text, which can be specified as either a matplotlib
            named color or a Color instance. Default is 'b' (blue).
        rotation: float, optional
            The orientation of the text in radians where 0=horizontal and
            pi/2=vertical. Default is 0.
        """
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
        if self._color is not None:
            return self._color.rgba()

    def _standardize_color_values(self):
        # convert to 'Color' class
        self._color = cl.Color.convert(self._color)


def default(fontsize='medium', color='b'):
    """
    What to draw if no particular preference is expressed.
    """
    return RenderControlText(fontsize=fontsize, fontstyle='normal', fontweight='normal', zdir=None, color=color)


def bold(fontsize='medium', color='b'):
    """
    What to draw for emphasis.
    """
    return RenderControlText(fontsize=fontsize, fontstyle='normal', fontweight='bold', zdir=None, color=color)
