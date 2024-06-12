import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz


class RenderControlIntersection:
    def __init__(
        self, dots_style=rcps.marker(), plot_limits: tuple[tuple, tuple] = None  # ((xmin, xmax), (ymin, ymax))
    ) -> None:
        self.dots_style: rcps.RenderControlPointSeq = dots_style
        self.plot_limits = plot_limits


# Common Configurations
