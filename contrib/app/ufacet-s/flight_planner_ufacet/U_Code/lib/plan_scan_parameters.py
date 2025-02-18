"""
Parameters for defining a scan, common across scan types.



"""

import opencsp.common.lib.camera.UCamera as cam


def construct_scan_parameters(scan_parameter_file):
    # Initialize container.
    scan_parameters = {}

    if scan_parameter_file == "NSTTF":
        # Location.
        scan_parameters["locale"] = (
            "NSTTF"  # Information needed to convert (x,y,z into global (longitude, latitude) coordinates.
        )

        # Camera.
        scan_parameters["camera"] = cam.sony_alpha_20mm_landscape()  # Camera model.
        # scan_parameters['camera'] = cam.sony_alpha_20mm_portrait()  # Camera model.
        # scan_parameters['camera'] = cam.ultra_wide_angle()  # Camera model.
        # scan_parameters['camera'] = cam.mavic_zoom()  # Camera model.

        # Scan flight.
        scan_parameters["lead_in"] = 18  # m.  # ** Overriden by vanity flights. **
        scan_parameters["run_past"] = 9  # m.  # ** Overriden by vanity flights. **
        scan_parameters["fly_forward_backward"] = False  # ** Overriden by vanity flights, raster flights. **

        # Return.
        return scan_parameters

    else:
        print(
            "ERROR: In construct_scan_parameters(), unexpected scan_parameter_file = '"
            + scan_parameter_file
            + "' encountered."
        )
        assert False
