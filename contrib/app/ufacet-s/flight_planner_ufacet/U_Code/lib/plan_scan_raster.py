"""
Construct a raster scan.



"""

import opencsp.app.ufacets.flight_planner_ufacet.U_Code.lib.plan_scan_raster_parameters as psrp
import opencsp.common.lib.csp.SolarField as sf


def construct_raster_scan(solar_field, raster_scan_parameter_file):
    # Notify progress.
    print("Constructing raster scan...")

    # Fetch scan parameters.
    raster_scan_parameters = psrp.construct_raster_scan_parameters(raster_scan_parameter_file)

    # Construct the scan.
    scan = sf.construct_solar_field_heliostat_survey_scan(solar_field, raster_scan_parameters)

    # Return.
    # Return the scan parameters, because they include information for converting the scan into a flight.
    return scan, raster_scan_parameters
