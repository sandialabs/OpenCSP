"""Library of functions that load saved Sofast HDF5 files
and return OpenCSP optics classes.
"""

import numpy as np

from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets, get_groups_and_datasets


def load_mirror_ideal(file: str, focal_length: float) -> MirrorParametric:
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
    data = load_hdf5_datasets(["DataSofastInput/optic_definition/facet_000/DefinitionFacet/v_facet_corners"], file)

    # Create mirror
    v_facet_corners = Vxy(data["v_facet_corners"][:2])
    region_facet = RegionXY.from_vertices(v_facet_corners)
    return MirrorParametric.generate_symmetric_paraboloid(focal_length, region_facet)


def load_mirror(file: str) -> MirrorPoint:
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
    v_facet_corners_str = "DataSofastInput/optic_definition/facet_000/DefinitionFacet"
    if v_facet_corners_str not in get_groups_and_datasets(file)[0]:
        # Undefined mirror type
        data = load_hdf5_datasets(
            [
                "DataSofastCalculation/facet/facet_000/SlopeSolverData/slopes_facet_xy",
                "DataSofastCalculation/facet/facet_000/SlopeSolverData/v_surf_points_facet",
            ],
            file,
        )

        # Get facet bounding box
        xs = (
            data["v_surf_points_facet"][0].max(),
            data["v_surf_points_facet"][0].min(),
            data["v_surf_points_facet"][0].min(),
            data["v_surf_points_facet"][0].max(),
        )
        ys = (
            data["v_surf_points_facet"][1].min(),
            data["v_surf_points_facet"][1].min(),
            data["v_surf_points_facet"][1].max(),
            data["v_surf_points_facet"][1].max(),
        )
        v_facet_corners = Vxy((xs, ys))
    else:
        # Defined mirror type
        data = load_hdf5_datasets(
            [
                "DataSofastInput/optic_definition/facet_000/DefinitionFacet/v_facet_corners",
                "DataSofastCalculation/facet/facet_000/SlopeSolverData/slopes_facet_xy",
                "DataSofastCalculation/facet/facet_000/SlopeSolverData/v_surf_points_facet",
            ],
            file,
        )
        v_facet_corners = Vxy(data["v_facet_corners"][:2])

    # Create facet region
    region_facet = RegionXY.from_vertices(v_facet_corners)

    # Get mirror data
    surface_points = Vxyz(data["v_surf_points_facet"])
    slopes = data["slopes_facet_xy"]

    surface_normals = np.ones((3, slopes.shape[1]))
    surface_normals[:2] = -slopes
    surface_normals = Uxyz(surface_normals)

    # Define mirror
    return MirrorPoint(surface_points, surface_normals, region_facet, "nearest")
