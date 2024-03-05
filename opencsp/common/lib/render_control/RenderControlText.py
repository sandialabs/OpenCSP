"""


"""


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
        horizontalalignment='center',  # center, right, left
        verticalalignment='center',  # center, top, bottom, baseline, center_baseline
        fontsize='medium',  # float or xx-small, x-small, small, medium, large, x-large, xx-large
        fontstyle='normal',  # normal, italic, oblique
        fontweight='normal',  # 0-1000, or light, normal, bold (see above for full list)
        zdir=None,  # None, 'x', 'y', 'z', (1,1,0), (1,1,1), ...
        color='b',  # bgrcmykw (see above)
    ):
        super(RenderControlText, self).__init__()

        # Set fields.
        self.horizontalalignment = horizontalalignment
        self.verticalalignment = verticalalignment
        self.fontsize = fontsize
        self.fontstyle = fontstyle
        self.fontweight = fontweight
        self.zdir = zdir
        self.color = color


def default(fontsize='medium', color='b'):
    """
    What to draw if no particular preference is expressed.
    """
    return RenderControlText(
        fontsize=fontsize,
        fontstyle='normal',
        fontweight='normal',
        zdir=None,
        color=color,
    )


def bold(fontsize='medium', color='b'):
    """
    What to draw for emphasis.
    """
    return RenderControlText(
        fontsize=fontsize, fontstyle='normal', fontweight='bold', zdir=None, color=color
    )
