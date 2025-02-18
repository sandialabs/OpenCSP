"""
Paths to directories and files providing data for unit and integration tests.
"""

import os

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp


# Sandia NSTTF
#
# Files related to the Sandia National Laboratories National Solar Thermal Test Facility.


def __sandia_nsttf_test_definition_dir():
    return os.path.join(
        orp.opencsp_code_dir(), "common", "lib", "test", "data", "input", "sandia_nsttf_test_definition"
    )


def sandia_nsttf_test_heliostats_origin_file():
    """
    Returns the path to the heliostats origin file for the Sandia NSTTF tests.

    This file contains data regarding the origin positions of heliostats at the Sandia
    National Solar Thermal Test Facility.

    Returns
    -------
    str
        The path to the heliostats origin file.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return os.path.join(__sandia_nsttf_test_definition_dir(), "NSTTF_Heliostats_origin_at_torque_tube.csv")


def sandia_nsttf_test_facet_centroidsfile():
    """
    Returns the path to the facet centroids file for the Sandia NSTTF tests.

    This file contains data regarding the centroids of facets at the Sandia National
    Solar Thermal Test Facility.

    Returns
    -------
    str
        The path to the facet centroids file.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return os.path.join(__sandia_nsttf_test_definition_dir(), "NSTTF_Facet_Centroids.csv")
