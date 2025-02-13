"""
Parameters for defining a UFACET scan, in addition to the general scan parameters.



"""

import numpy as np

import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_parameters as psp


def construct_ufacet_scan_parameters(ufacet_scan_parameter_file, ufacet_control_parameters):
    # General scan parameters.
    scan_parameters = psp.construct_scan_parameters(ufacet_scan_parameter_file)

    # NSTTF
    if ufacet_scan_parameter_file == "NSTTF":
        # Scan section construction.
        scan_parameters["candidate_margin_w"] = (
            10.00  # m.  Margin on either side of section plane to bring in heliostats.
        )
        #     Should be larger than side-to-side heliostat distance.
        scan_parameters["discard_threshold_p"] = (
            9.00  # m.  Threshold to discard heliostats that are close together on a section, presumably abreast.
        )
        #     Should be smaller than minimum heliostat row spacing.
        # Section analysis.
        scan_parameters["p_margin"] = (
            0  # 2    # m.  Lateral distance to add to constraints to allow UAS postiion error.
        )
        scan_parameters["altitude_margin"] = 2.5  # m.   Clearance above highest possible heliostat point.
        scan_parameters["maximum_safe_altitude"] = (
            90.0  # meters.  Driven by safey considerations.  Control limit may be tighter.
        )
        scan_parameters["maximum_target_lookback"] = 3  # Number of heliostats to look back for reflection targets.
        scan_parameters["gaze_tolerance"] = np.deg2rad(
            1
        )  # Uncertainty in gaze angle.  True angle is +/- tolerance from nominal.

        # Scan flight.
        scan_parameters["eta_min"] = -(np.pi / 2.0)  # rad.
        scan_parameters["eta_max"] = np.pi / 4.0  # rad.
        scan_parameters["fly_forward_backward"] = True  # Always true for UFACET.
        scan_parameters["speed"] = 5  # m/sec.

        # Add control parameters.
        for key in ufacet_control_parameters.keys():
            if key in scan_parameters.keys():
                print('ERROR: In construct_ufacet_scan_parameters(1), duplicate key="' + str(key) + '" encountered.')
                assert False
            scan_parameters[key] = ufacet_control_parameters[key]

        # Ensure that maximum altitude does not exceed the maximum safe altitude.
        if scan_parameters["maximum_altitude"] > scan_parameters["maximum_safe_altitude"]:
            print(
                "NOTE: In construct_ufacet_scan_parameters(), input maximum altitude = "
                + str(scan_parameters["maximum_altitude"])
                + " m exceeds maximum safe altitude = "
                + str(scan_parameters["maximum_safe_altitude"])
                + " m.  Clamping to safe limit."
            )
            scan_parameters["maximum_altitude"] = scan_parameters["maximum_safe_altitude"]

        # Return.
        return scan_parameters

    else:
        print(
            "ERROR: In construct_ufacet_scan_parameters(), unexpected ufacet_scan_parameter_file = '"
            + ufacet_scan_parameter_file
            + "' encountered."
        )
        assert False
