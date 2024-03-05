"""
Paths to directories and files providing data for unit and integration tests.
"""

import os

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp


# Sandia NSTTF
#
# Files related to the Sandia National Laboratories National Solar Thermal Test Facility.

def __sandia_nsttf_test_definition_dir():
    return os.path.join(orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'input', 'sandia_nsttf_test_definition')


def sandia_nsttf_test_heliostats_origin_file():
    return os.path.join(__sandia_nsttf_test_definition_dir(), 'NSTTF_Heliostats_origin_at_torque_tube.csv')

def sandia_nsttf_test_facet_centroidsfile():
    return os.path.join(__sandia_nsttf_test_definition_dir(), 'NSTTF_Facet_Centroids.csv')
