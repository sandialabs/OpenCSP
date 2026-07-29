""" """

import opencsp.common.lib.tool.file_tools as ft


class RenderControlTopLevel:
    """
    Overall output control for scan flight planning code.
    """

    def __init__(
        self,
        draw_ufacet_xy_analysis=True,
        draw_ufacet_section_construction=True,
        draw_ufacet_scan=True,
        draw_flight_plan=True,
        xy_solar_field_style=None,  # If defined, overrides default.
        flight_plan_output_path=None,  # Defaults to sister output directory with current date and time.
        save_flight_plan=True,
        summarize_figures=False,
        save_figures=True,
        figure_output_path=None,  # Defaults to sister output directory with current date and time.
    ):
        super(RenderControlTopLevel, self).__init__()

        self.draw_ufacet_xy_analysis = draw_ufacet_xy_analysis
        self.draw_ufacet_section_construction = draw_ufacet_section_construction
        self.draw_ufacet_scan = draw_ufacet_scan
        self.draw_flight_plan = draw_flight_plan
        self.xy_solar_field_style = xy_solar_field_style

        self.save_flight_plan = save_flight_plan
        if flight_plan_output_path != None:
            self.flight_plan_output_path = flight_plan_output_path
        else:
            self.flight_plan_output_path = ft.default_output_path()

        self.summarize_figures = summarize_figures

        self.save_figures = save_figures
        if figure_output_path != None:
            self.figure_output_path = figure_output_path
        else:
            self.figure_output_path = ft.default_output_path()


# COMMON CASES


def default():
    return RenderControlTopLevel()
