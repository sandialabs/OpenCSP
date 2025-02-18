"""
Parameters for defining a vanity scan, in addition to the general scan parameters.



"""

import numpy as np

import opencsp.common.lib.camera.UCamera as cam
import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_parameters as psp


def check_eta(eta):
    # Check result.
    if np.rad2deg(eta) > 0:
        print(
            "WARNING: In construct_vanity_scan_parameters(), positive gaze angle = "
            + str(np.rad2deg(eta))
            + " deg encountered."
        )
        print("         Positive gaze angles look up.  Continuing...")
        print("         Continuing...")
    eta_max = np.deg2rad(30.0)
    if eta > eta_max:
        print(
            "ERROR: In construct_vanity_scan_parameters(), high positive gaze angle = "
            + str(np.rad2deg(eta))
            + "encountered."
        )
        print("       We reject positive gaze angles exceeding " + str(np.rad2deg(eta_max)) + " degrees.")
        assert False
    eta_min = np.rad2deg(-90.0)
    if eta < eta_min:
        print(
            "ERROR: In construct_vanity_scan_parameters(), excessive negative gaze angle = "
            + str(np.rad2deg(eta))
            + "encountered."
        )
        print("       We reject gaze angles less than " + str(np.rad2deg(eta_min)) + " degrees.")
        assert False


def construct_vanity_scan_parameters(vanity_scan_parameter_file, vanity_heliostat_name, vanity_heliostat_azimuth):
    # General scan parameters.
    scan_parameters = psp.construct_scan_parameters(vanity_scan_parameter_file)

    # Override general scan parameters.
    scan_parameters["lead_in"] = 7  # 18 # m.
    scan_parameters["run_past"] = 5  # 9 # m.
    scan_parameters["fly_forward_backward"] = True  # False

    if vanity_scan_parameter_file == "NSTTF":
        print("ERROR: In construct_vanity_scan_parameters(), NSTTF case not implemented yet.")
        assert False
        # Vanity scan parameters.
        eta = np.deg2rad(-35.0)  # Arbitrary test value.
        scan_parameters["n_horizontal"] = 10  # Number of horizontal passes.
        scan_parameters["n_vertical"] = 6  # Number of vertical passes.
        scan_parameters["eta"] = eta  # rad,  Gaze angle, measured relative to horizontal (positive ==> up).
        scan_parameters["relative_z"] = 20  # m.
        scan_parameters["speed"] = 10  # m/sec.
        # Check result and return.
        check_eta(eta)
        return scan_parameters

    else:
        print(
            "ERROR: In construct_vanity_scan_parameters(), unexpected vanity_scan_parameter_file = '"
            + vanity_scan_parameter_file
            + "' encountered."
        )
        assert False
