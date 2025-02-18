"""


"""

import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.RenderControlFlightPlan as rcfp
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf


class RenderControlFlightOverSolarField:
    """
    Render control for flights over solar fields.
    """

    def __init__(
        self,
        draw_solar_field=True,
        solar_field_style=rcsf.heliostat_centroids(color="grey"),
        draw_flight_plan=True,
        flight_plan_style=rcfp.default(),
    ):
        super(RenderControlFlightOverSolarField, self).__init__()

        self.draw_solar_field = draw_solar_field
        self.solar_field_style = solar_field_style
        self.draw_flight_plan = draw_flight_plan
        self.flight_plan_style = flight_plan_style


# COMMON CASES


def default():
    return RenderControlFlightOverSolarField()
