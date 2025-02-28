import matplotlib.colors


class RenderControlHeatmap:
    """
    Render control for a heatmap.

    This class controls the style of the heatmap, allowing customization of line styles, line widths,
    and color maps.

    Choices for line styles and color maps can be found in the Matplotlib documentation:
    - Line styles: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    - Color maps: https://matplotlib.org/stable/gallery/color/colormap_reference.html

    Line styles
    -----------
    '-'      solid line style
    '--'     dashed line style
    '-.'     dash-dot line style
    ':'      dotted line style
    'None'   no lines

    Perceptually uniform color maps
    -------------------------------
    'Viridis'    purple -> turquoise -> yellow (default)
    'Plasma'     blue -> purple -> yellow-green
    'Inferno'    black -> rust -> yellow-white
    'Magma'      black -> maroon -> white
    'Cividis'    navy -> grey -> yellow

    Sequential color maps
    ---------------------
    'Greys'      white -> grey -> black
    'Purples'    white -> light purple -> royal purple
    'Blues'      white -> light blue -> dark blue
    'Greens'     white -> light green -> forest green
    'Oranges'    white -> orange -> burnt orange
    'Reds'       white -> tangerine -> dark red
    'YlOrBr'     white -> yellow -> orange -> brown
    'YlOrRd'     white -> yellow -> orange -> dark red
    'OrRd'       white -> orange -> dark red
    'PuRd'       white -> purple -> dark red
    'RdPu'       white -> pink -> purple
    'BuPu'       white -> light blue -> faded blue -> dark purple
    'GrBu'       white -> light green -> aqua -> dark blue
    'PuBu'       white -> light purple -> faded blue -> dark blue
    'YlGnBl'     white -> yellow -> green -> aqua -> blue -> dark blue
    'PuBlGn'     white -> light purple -> sky blue -> forest green
    'BuGn'       white -> aqua -> green -> forest green
    'YlGn'       white -> yellow -> yellow green -> forest green

    Miscellaneous colormaps
    -----------------------
    'jet'   navy -> green -> rust
    """

    # ChatGPT 4o-mini assisted with generating this doc string
    def __init__(
        self,
        linestyle_unimplemented="None",
        linewidth_unimplemented=1,
        cmap: str | matplotlib.colors.Colormap = "Viridis",
    ):
        """
        Initialize the RenderControlHeatmap instance.

        Parameters
        ----------
        linestyle_unimplemented : str, optional
            The line style to use for the heatmap. Default is "None".
        linewidth_unimplemented : int, optional
            The width of the lines in the heatmap. Default is 1.
        cmap : str or Colormap, optional
            The color map to use for the heatmap. Default is "Viridis".
        """
        # ChatGPT 4o-mini assisted with generating this doc string
        self.linestyle_unimplemented = linestyle_unimplemented
        self.linewidth_unimplemented = linewidth_unimplemented
        self.cmap = cmap
