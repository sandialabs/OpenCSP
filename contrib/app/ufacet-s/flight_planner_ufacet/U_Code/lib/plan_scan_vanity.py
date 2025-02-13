"""
Construct a vanity scan.



"""

import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_vanity_parameters as psvp
import opencsp.common.lib.csp.SolarField as sf


def construct_vanity_scan(solar_field, vanity_scan_parameter_file, vanity_heliostat_name, vanity_heliostat_azimuth):
    # Notify progress.
    print("Constructing vanity scan...")

    # Fetch scan parameters.
    vanity_scan_parameters = psvp.construct_vanity_scan_parameters(
        vanity_scan_parameter_file, vanity_heliostat_name, vanity_heliostat_azimuth
    )

    # Construct the scan.
    scan = sf.construct_solar_field_vanity_scan(solar_field, vanity_scan_parameters)

    # Return.
    # Return the scan parameters, because they include information for converting the scan into a flight.
    return scan, vanity_scan_parameters
