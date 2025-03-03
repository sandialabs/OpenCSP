import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq


class RenderControlTower:
    """
    Render control for towers.

    This class manages the rendering settings for towers, allowing customization of various visual
    elements related to the tower's representation.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    def __init__(
        self,
        centroid=False,
        draw_name=False,
        name_style=None,
        draw_outline=True,
        point_styles: RenderControlPointSeq = None,
        wire_frame: RenderControlPointSeq = None,
        target: RenderControlPointSeq = None,
        bcs: RenderControlPointSeq = None,
    ) -> None:
        """
        Render control for towers.

        This class manages the rendering settings for towers, allowing customization of various visual
        elements related to the tower's representation.

        Parameters
        ----------
        centroid : bool, optional
            If True, the centroid will be drawn on the graph as the origin of the tower. Default is False.
        draw_name : bool, optional
            If True, the name of the tower will be drawn on the graph. Default is False.
        name_style : None | str, optional
            If `draw_name` is True, styles the name using `RenderControlText`. By default, it uses the default style from `RenderControlText`, with color 'black'.
        draw_outline : bool, optional
            If True, draws the outline of the tower using the wire frame style. Default is True.
        point_styles : RenderControlPointSeq, optional
            Styles for drawing the target as a point on the tower. Default is None.
        wire_frame : RenderControlPointSeq, optional
            Outline style of the tower that draws the walls and edges. Default is `RenderControlPointSeq` outline with color 'black' and linewidth '1'.
        target : RenderControlPointSeq, optional
            If provided, draws a point on the tower. Default is a red 'x' marker with a size of 6.
        """
        # "ChatGPT 4o" assisted with generating this docstring.

        # default values
        if name_style is None:
            namestyle = rctxt.default(color="k")
        if point_styles is None:
            point_styles = rcps.marker()
        if wire_frame is None:
            wire_frame = rcps.outline()
        if target is None:
            target = rcps.marker(marker="x", color="r", markersize=6)
        if bcs is None:
            bcs = rcps.marker(marker="+", color="b", markersize=6)

        super(RenderControlTower, self).__init__()

        self.centroid = centroid
        self.draw_name = draw_name
        self.name_style = name_style
        self.draw_outline = draw_outline
        self.point_styles = point_styles
        self.wire_frame = wire_frame
        self.target = target
        self.bcs = bcs

    def style(self, any):
        """ "style" is a method commonly used by RenderControlEnsemble.
        We add this method here so that RenderControlHeliostat can be used similarly to RenderControlEnsemble."""
        return self

    # Common Configurations


def normal_tower():
    """
    Create a render control for a normal tower.

    This function returns a `RenderControlTower` instance configured to display the overall outline
    of the tower.

    Returns
    -------
    RenderControlTower
        An instance of `RenderControlTower` configured for a normal tower outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall tower outline only.
    return RenderControlTower()


def no_target():
    """
    Create a render control for a tower outline without a target.

    This function returns a `RenderControlTower` instance configured to draw the tower outline
    without any target point.

    Returns
    -------
    RenderControlTower
        An instance of `RenderControlTower` configured to display the tower outline without a target.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # tower outline with no target.
    return RenderControlTower(wire_frame=rcps.outline(color="g"), target=False)


def no_bcs():
    """
    Create a render control for a tower outline without boundary control points (BCS).

    This function returns a `RenderControlTower` instance configured to draw the tower outline
    without any boundary control points.

    Returns
    -------
    RenderControlTower
        An instance of `RenderControlTower` configured to display the tower outline without boundary control points.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # tower outline with not bcs.
    return RenderControlTower(wire_frame=rcps.outline(color="g"), bcs=False)
