import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz


class RenderControlIntersection:
    """
    A class for controlling the rendering of intersection points in a graphical representation.

    This class allows for the configuration of the style of dots used to represent intersection points
    and the limits of the plot area.

    Attributes
    ----------
    dots_style : rcps.RenderControlPointSeq
        The style of the dots used for rendering intersection points.
    plot_limits : tuple[tuple, tuple] or None
        The limits of the plot area, or None if not set.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(
        self, dots_style=rcps.marker(), plot_limits: tuple[tuple, tuple] = None  # ((xmin, xmax), (ymin, ymax))
    ) -> None:
        """
        Initializes the RenderControlIntersection instance with specified dot style and plot limits.

        Parameters
        ----------
        dots_style : rcps.RenderControlPointSeq, optional
            The style of the dots used for rendering intersection points. Defaults to a marker style.
        plot_limits : tuple[tuple, tuple], optional
            A tuple defining the limits of the plot area as ((xmin, xmax), (ymin, ymax)).
            If None, the plot limits are not set.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.dots_style: rcps.RenderControlPointSeq = dots_style
        self.plot_limits = plot_limits


# Common Configurations
