"""


"""

import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlScanPass as rcsp
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlWayPoint as rcwp


class RenderControlFlightPlan:
    """
    Render control for UAS flights.
    """

    def __init__(
        self,
        draw_outline=True,
        outline_style=rcps.outline(),
        draw_waypoints=True,
        waypoint_styles=rce.RenderControlEnsemble(rcwp.default()),
        draw_scan=True,
        scan_pass_styles=rce.RenderControlEnsemble(rcsp.default()),
        draw_name=False,
        name_style=rctxt.default(color="k"),
    ):
        super(RenderControlFlightPlan, self).__init__()

        self.draw_outline = draw_outline
        self.outline_style = outline_style
        self.draw_waypoints = draw_waypoints
        self.waypoint_styles = waypoint_styles
        self.draw_scan = draw_scan
        self.scan_pass_styles = scan_pass_styles
        self.draw_name = draw_name
        self.name_style = name_style


# COMMON CASES


def default():
    return RenderControlFlightPlan()


# def outline(color='k'):
#     # Overall field outline only.
#     return RenderControlFlightPlan(draw_waypoints = True,
#                                     waypoint_style = rcps.outline(color=color),
#                                     draw_sections=False,
#                                     draw_name=False)
#
#
# def heliostat_blanks():
#     # Draw nothing.  Heliostats will be added as special rendering categories.
#     return RenderControlFlightPlan(draw_waypoints = False,
#                                     draw_sections=True,
#                                     section_styles=rce.RenderControlEnsemble(rch.blank()),
#                                     draw_name=False)
#
#
# def heliostat_names(color='k'):
#     return RenderControlFlightPlan(draw_waypoints = False,
#                                     draw_sections=True,
#                                     heliostat_styles=rce.RenderControlEnsemble(rch.name(color=color)),
#                                     draw_name=False)
#
#
# def heliostat_centroids(color='k'):
#     return RenderControlFlightPlan(draw_waypoints = False,
#                                     draw_sections=True,
#                                     heliostat_styles=rce.RenderControlEnsemble(rch.centroid(color=color)),
#                                     draw_name=False)
#
#
# def heliostat_centroids_names(color='k'):
#     return RenderControlFlightPlan(draw_waypoints = False,
#                                     draw_sections=True,
#                                     heliostat_styles=rce.RenderControlEnsemble(rch.centroid_name(color=color)),
#                                     draw_name=False)
#
#
# def heliostat_outlines(color='k'):
#     return RenderControlFlightPlan(draw_waypoints = False,
#                                     draw_sections=True,
#                                     heliostat_styles=rce.RenderControlEnsemble(rch.outline(color=color)),
#                                     draw_name=False)
#
#
# def heliostat_vector_field(color='k', vector_length=9):
#     return RenderControlFlightPlan(draw_waypoints = False,
#                                     waypoint_style = rcps.outline(color=color),
#                                     draw_sections=True,
#                                     heliostat_styles=rce.RenderControlEnsemble(rch.normal(color=color,
#                                                                                           surface_normal_length=vector_length)),
#                                     draw_name=False)
#
#
