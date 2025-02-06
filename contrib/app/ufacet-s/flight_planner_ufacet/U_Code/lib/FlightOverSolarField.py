"""


"""

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca


class FlightOverSolarField:
    """
    Represents a flight over a solar field, for rendering and analysis.
    """

    def __init__(self, solar_field, flight_plan):  # SolarField class object.  # FlightPlan class object.
        super(FlightOverSolarField, self).__init__()

        self.solar_field = solar_field
        self.flight_plan = flight_plan

    def draw(self, view, flight_over_solar_field_style):
        # Solar field.
        if flight_over_solar_field_style.draw_solar_field:
            self.solar_field.draw(view, flight_over_solar_field_style.solar_field_style)
        # Flight plan.
        if flight_over_solar_field_style.draw_flight_plan:
            self.flight_plan.draw(view, flight_over_solar_field_style.flight_plan_style)


# -------------------------------------------------------------------------------------------------------
# TOP-LEVEL RENDERING ROUTINES
#


def draw_flight_over_solar_field(figure_control, flight_over_solar_field, flight_over_solar_field_style, view_spec):
    # Assumes that solar field and flight plan are already set up with heliosat configurations, waypoints, etc.

    # Construct title.
    title = flight_over_solar_field.flight_plan.name
    name = flight_over_solar_field.flight_plan.short_name
    # Setup figure.
    fig_record = fm.setup_figure_for_3d_data(figure_control, rca.meters(), view_spec, title=title, name=name)
    view = fig_record.view
    # Comment.
    fig_record.comment.append(title)
    # Draw.
    flight_over_solar_field.draw(view, flight_over_solar_field_style)
    # Finish.
    view.show()
    # Return.
    return view
