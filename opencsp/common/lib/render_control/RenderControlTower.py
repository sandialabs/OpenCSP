import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq


class RenderControlTower:
    """
    Render control for towers.
    """

    def __init__(
        self,
        centroid=False,
        draw_name=False,
        name_style=None,
        draw_outline=True,
        point_styles: RenderControlPointSeq = None,
        wire_frame: RenderControlPointSeq = None,
        target: RenderControlPointSeq = None,
    ) -> None:
        """
        Controls for rendering a tower.

        Parameters
        ----------
        centroid : bool, optional
            If True, centroid will be drawn on graph as origin of tower. Default is False.
        draw_name : bool, optional
            If True then the name will be drawn on graph. Default is False.
        name_style : None | str
            If draw_name = True, then styles name using RenderControlText. By default from RenderControlText, color 'black'.
        draw_outline : bool
            Draws outline of Tower using wire_frame style. Default is True.
        point_styles : RenderControlPointSeq
            Draws target as a point on the Tower.
        wire_frame : RenderControlPointSeq
            Outline style of Tower, that draws walls edges. Default is RenderControlSeq outline, color 'black', linewidth '1'.
        target : RenderControlPointSeq
            If target, draws a point on Tower. Default color is 'red', with shape 'x', markersize '6'.
        """

        # default values
        if name_style is None:
            namestyle = rctxt.default(color='k')
        if point_styles is None:
            point_styles = rcps.marker()
        if wire_frame is None:
            wire_frame = rcps.outline()
        if target is None:
            target = rcps.marker(marker='x', color='r', markersize=6)

        super(RenderControlTower, self).__init__()

        self.centroid = centroid
        self.draw_name = draw_name
        self.name_style = name_style
        self.draw_outline = draw_outline
        self.point_styles = point_styles
        self.wire_frame = wire_frame
        self.target = target

    def style(self, any):
        """ "style" is a method commonly used by RenderControlEnsemble.
        We add this method here so that RenderControlHeliostat can be used similarly to RenderControlEnsemble."""
        return self

    # Common Configurations


def normal_tower():
    # Overall tower outline only.
    return RenderControlTower()


def no_target():
    # tower outline with no target.
    return RenderControlTower(wire_frame=rcps.outline(color='g'), target=False)
