import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz


class RenderControlIntersection:
    """
    A class to represent a render control intersection for plotting points in a specified style.

    Attributes
    ----------
    dots_style : rcps.RenderControlPointSeq
        The style of the dots used for rendering.
    plot_limits : tuple of tuple or None
        The limits for the plot, or None if not specified.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(
        self, dots_style=rcps.marker(), plot_limits: tuple[tuple, tuple] = None  # ((xmin, xmax), (ymin, ymax))
    ) -> None:
        """
            Initializes the RenderControlIntersection with specified dot style and plot limits.

            Parameters
            ----------
            dots_style : rcps.RenderControlPointSeq, optional
                The style of the dots to be used for rendering points. Defaults to `rcps.marker()`.
            plot_limits : tuple of tuple, optional
                The limits for the plot defined as ((xmin, xmax), (ymin, ymax)). 
                If None, the plot limits will be determined automatically.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.dots_style: rcps.RenderControlPointSeq = dots_style
        self.plot_limits = plot_limits


# Common Configurations
