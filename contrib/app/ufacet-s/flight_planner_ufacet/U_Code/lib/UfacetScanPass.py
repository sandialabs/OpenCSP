"""


"""

import math

import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet_section_analysis as psusa
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_ufacet_section_analysis_render as psusar
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.uas.WayPoint as wp


class UfacetScanPass:
    """
    UFACET scan pass for measuring heliostats in a solar field.
    """

    def __init__(
        self,
        solar_field,  # SolarField class object.
        section,  # Dictionary containing defining parameters.
        ufacet_scan_parameters,
    ):  # Dictionary containing control information.
        super(UfacetScanPass, self).__init__()

        # Input parameters.
        self.solar_field = solar_field
        self.section = section
        self.ufacet_scan_parameters = ufacet_scan_parameters
        # Constructed members.
        self.setup_section()  # Sets up projection, heliostat names, and section context.
        self.pass_constraints = psusa.section_analysis(
            self.section_context, self.heliostat_name_list, self.assess_heliostat_name_list
        )

    def setup_section(self):
        # Construct section corresponding to this pass.
        # ?? SCAFFOLDING RCB -- ELIMINATE DUPLICATION, CONFUSION.
        # ?? SCAFFOLDING RCB -- RENAME, SO THAT "SECTION" IS NOT OVER-USED..
        self.section_context = self.ufacet_scan_parameters.copy()  # Copy because we will add to this.
        self.section_context["solar_field"] = self.solar_field
        self.section_context["view_spec"] = self.section[
            "view_spec"
        ]  # ?? SCAFFOLDING RCB -- ELIMINATE DUPLICATION, CONFUSION.
        # Heliostats along this pass.
        self.heliostat_name_list = self.section["selected_heliostat_name_list"]

        # Identify heliostats to assess along this pass.
        self.assess_heliostat_name_list = self.heliostat_name_list  # ?? SCAFFOLDING RCB -- ELIMINATE THIS DISTINCTION

        # Define the clipping box for rendering.
        p_min = 1e9
        p_max = -1e9
        view_spec = self.section["view_spec"]
        # Find the extremal (p_min, p_max) values.
        for heliostat_name in self.heliostat_name_list:
            heliostat = self.solar_field.lookup_heliostat(heliostat_name)
            corners = heliostat.corners()
            for corner_xyz in corners:
                corner_pq = vs.xyz2pq(corner_xyz, view_spec)
                if corner_pq[0] < p_min:
                    p_min = corner_pq[0]
                if corner_pq[0] > p_max:
                    p_max = corner_pq[0]
        q_min = 0
        q_max = 50
        p_margin = 20  # m.
        pq_min = [p_min - p_margin, q_min]
        pq_max = [p_max + p_margin, q_max]
        self.section_context["clip_pq_box"] = [pq_min, pq_max]

    def waypoints(self):
        # Fetch path parameters.
        locale = self.ufacet_scan_parameters["locale"]
        view_spec = self.section["view_spec"]
        pass_segment = self.pass_constraints["selected_cacg_segment"]  # "cacg" == "constant altitude, constant gaze"
        # Construct start and end (x,y,z) points.
        start_pq = pass_segment[0]  # Lead-in distance added later.
        end_pq = pass_segment[1]  # Run-past distance added later.
        start_xyz = vs.pq2xyz(start_pq, view_spec)
        end_xyz = vs.pq2xyz(end_pq, view_spec)
        # Compute heading.
        dx = end_xyz[0] - start_xyz[0]
        dy = end_xyz[1] - start_xyz[1]
        theta = math.atan2(dy, dx)
        # Fetch gaze angle.
        # Fork based on desired gaze control.
        gaze_type = self.ufacet_scan_parameters["gaze_type"]
        if gaze_type == "constant":
            # Constant gaze.
            selected_cacg_etaC = self.pass_constraints[
                "selected_cacg_etaC"
            ]  # "cacg" == "constant altitude, constant gaze"
            start_eta = selected_cacg_etaC[0]
            end_eta = selected_cacg_etaC[0]
        else:
            print('ERROR: In UfacetScanPass.waypoints(), unexpected gaze_type="' + str(gaze_type) + '" encountered.')
            assert False
            # Variable Gaze.
            # # ?? SCAFFOLDING RCB -- INCORRECT; SHOULD COMPUTE OPTIMUM COMPROMISE IN ANALYSIS.
            # per_heliostat_constraints = self.pass_constraints['per_heliostat_constraints']
            # heliostat_name_list = list(per_heliostat_constraints.keys())
            # first_constraints = per_heliostat_constraints[heliostat_name_list[0]]
            # last_constraints  = per_heliostat_constraints[heliostat_name_list[-1]]
            # start_etaC = first_constraints['selected_cacg_etaC']  # ?? SCAFFOLDING RCB -- INCORRECT; DOESN'T SELECT PATH TYPE.
            # start_eta = start_etaC[0]
            # end_etaC = last_constraints['selected_cacg_etaC']  # ?? SCAFFOLDING RCB -- INCORRECT; DOESN'T SELECT PATH TYPE.
            # end_eta = end_etaC[0]
        # Apply manual input gaze angle offset.
        start_eta += self.ufacet_scan_parameters["delta_eta"]
        end_eta += self.ufacet_scan_parameters["delta_eta"]

        # Determine speed.
        speed = self.ufacet_scan_parameters["speed"]  # m/sec.
        print(
            "WARNING: In UfacetScanPass.waypoints(), speed calculation not implemented yet. Setting speed to "
            + str(speed)
            + " m/sec."
        )  # ?? RCB SCAFFOLDING -- STUB
        # Construct way points.
        start_wpt = wp.WayPoint(locale, start_xyz, theta, start_eta, stop=False, speed=speed)
        end_wpt = wp.WayPoint(locale, end_xyz, theta, end_eta, stop=False, speed=speed)
        # Return.
        return [start_wpt, end_wpt]

    # RENDERING

    def draw_section_analysis(self, figure_control, analysis_render_control):
        psusar.draw_section_analysis(
            figure_control,
            self.section_context,
            self.heliostat_name_list,
            self.pass_constraints,
            analysis_render_control,
        )


# -------------------------------------------------------------------------------------------------------
# CONSTRUCTION FUNCTIONS
#


def construct_ufacet_passes(solar_field, section_list, ufacet_scan_parameters):
    # Notify progress.
    print("Constructing UFACET scan passes...")

    # Construct hte scan passes.
    scan_pass_list = []
    for section in section_list:
        scan_pass = UfacetScanPass(solar_field, section, ufacet_scan_parameters)
        scan_pass_list.append(scan_pass)

    # Return.
    return scan_pass_list
