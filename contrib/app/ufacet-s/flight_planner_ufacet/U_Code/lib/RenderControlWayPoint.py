"""


"""

import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt


# Constants
DEFAULT_heading_scale = 2  # m
DEFAULT_CORNER_NORMAL_LENGTH = 1  # m


class RenderControlWayPoint:
    """
    Render control for flight path way points.
    """

    def __init__(
        self,
        draw_position=True,
        position_style=rcps.marker(marker="."),
        draw_stop=True,
        stop_style=rcps.marker(color="r", marker="x", markersize=7),  # Same color as heading.
        draw_heading=True,
        heading_scale=3,
        heading_style=rcps.outline(color="r"),
        draw_gaze=True,
        gaze_length=6,
        gaze_style=rcps.outline(color="g"),
        # draw_heading_at_corners = True,
        # corner_normal_length = 2,
        # corner_normal_style = rcps.outline(),
        # corner_normal_base_style = rcps.marker(),
        # draw_outline = True,
        # outline_style = rcps.outline(),
        draw_idx=True,
        idx_style=rctxt.RenderControlText(
            color="k", fontsize="small", horizontalalignment="right", verticalalignment="top"
        ),
    ):
        super(RenderControlWayPoint, self).__init__()

        self.draw_position = draw_position
        self.position_style = position_style
        self.draw_stop = draw_stop
        self.stop_style = stop_style
        self.draw_heading = draw_heading
        self.heading_scale = heading_scale
        self.heading_style = heading_style
        self.draw_gaze = draw_gaze
        self.gaze_length = gaze_length
        self.gaze_style = gaze_style
        # self.draw_heading_at_corners = draw_heading_at_corners
        # self.corner_normal_length = corner_normal_length
        # self.corner_normal_style = corner_normal_style
        # self.corner_normal_base_style = corner_normal_base_style
        # self.draw_outline = draw_outline
        # self.outline_style = outline_style
        self.draw_idx = draw_idx
        self.idx_style = idx_style


# COMMON CASES


def default():
    return RenderControlWayPoint()


# def outline(color='k'):
#     return RenderControlWayPoint(draw_position = False,
#                                draw_heading = False,
#                                draw_heading_at_corners = False,
#                                draw_outline = True,
#                                outline_style = rcps.outline(color=color),
#                                draw_idx=False)
#
#
# def outline_name(color='k'):
#     return RenderControlWayPoint(draw_position = False,
#                                draw_heading = False,
#                                draw_heading_at_corners = False,
#                                draw_idx=True,
#                                draw_outline = True,
#                                outline_style = rcps.outline(color=color),
#                                idx_style=rctxt.default(color=color))
#
#
# def normal_outline(color='k'):
#     return RenderControlWayPoint(draw_position = False,
#                                draw_heading = True,
#                                heading_scale=DEFAULT_heading_scale,
#                                heading_style = rcps.outline(color=color),
#                                heading_base_style = rcps.marker(color=color),
#                                draw_heading_at_corners = False,
#                                draw_outline = True,
#                                outline_style = rcps.outline(color=color),
#                                draw_idx=False)
#
#
# def corner_normals_outline(color='k'):
#     return RenderControlWayPoint(draw_position = False,
#                                draw_heading = False,
#                                draw_heading_at_corners = True,
#                                corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
#                                corner_normal_style = rcps.outline(color=color),
#                                corner_normal_base_style = rcps.marker(color=color),
#                                draw_outline = True,
#                                outline_style = rcps.outline(color=color),
#                                draw_idx=False)
#
#
# def corner_normals_outline_name(color='k'):
#     return RenderControlWayPoint(draw_position = False,
#                                draw_heading = False,
#                                draw_heading_at_corners = True,
#                                corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
#                                corner_normal_style = rcps.outline(color=color),
#                                corner_normal_base_style = rcps.marker(color=color),
#                                draw_idx=True,
#                                draw_outline = True,
#                                outline_style = rcps.outline(color=color),
#                                idx_style=rctxt.default(color=color))
#
#
# def highlight(color='b'):
#     return RenderControlWayPoint(position_style = rcps.marker(color=color),
#                                heading_base_style = rcps.marker(color=color),
#                                corner_normal_style = rcps.outline(color=color),
#                                corner_normal_base_style = rcps.marker(color=color),
#                                outline_style = rcps.outline(color=color),
#                                heading_style = rcps.outline(color=color),
#                                idx_style=rctxt.default(color=color))
#
