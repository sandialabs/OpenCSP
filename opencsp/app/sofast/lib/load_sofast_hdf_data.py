"""Library of functions that load saved Sofast HDF5 files and return OpenCSP optics classes.

NOTE: currently only returns data for single-optic measurements.
"""

import numpy as np
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


def load_single_mirror_ideal_parabolic(file: str, focal_length: float) -> MirrorParametric:
    """Uses the "OpticDefinition" in the given SOFAST HDF file and given
    focal length to create a reference Facet

    Parameters
    ----------
    file : str
        HDF file of saved Sofast facet data
    focal_length : float
        Ideal parabolic focal length of facet

    Returns
    -------
    Facet
        Reference facet representation. Defined as "rotation_defined."
    """
    # Load facet corners
    data = load_hdf5_datasets(['DataSofastInput/optic_definition/facet_000/DefinitionFacet/v_facet_corners'], file)

    # Create mirror
    v_facet_corners = Vxy(data['v_facet_corners'][:2])
    region_facet = RegionXY.from_vertices(v_facet_corners)
    return MirrorParametric.generate_symmetric_paraboloid(focal_length, region_facet)


def load_single_mirror(file: str) -> MirrorPoint:
    """Loads Sofast data of a single facet data collection into
    a Facet object.

    Parameters
    ----------
    file : str
        HDF file of saved Sofast facet data

    Returns
    -------
    Facet
        Representation of Facet
    """
    data = load_hdf5_datasets(
        [
            'DataSofastInput/optic_definition/facet_000/DefinitionFacet/v_facet_corners',
            'DataSofastCalculation/facet/facet_000/SlopeSolverData/slopes_facet_xy',
            'DataSofastCalculation/facet/facet_000/SlopeSolverData/v_surf_points_facet',
        ],
        file,
    )

    # Create facet region
    v_facet_corners = Vxy(data['v_facet_corners'][:2])
    region_facet = RegionXY.from_vertices(v_facet_corners)

    # Get mirror data
    surface_points = Vxyz(data['v_surf_points_facet'])
    slopes = data['slopes_facet_xy']

    surface_normals = np.ones((3, slopes.shape[1]))
    surface_normals[:2] = -slopes
    surface_normals = Uxyz(surface_normals)

    # Define mirror
    return MirrorPoint(surface_points, surface_normals, region_facet, 'nearest')
