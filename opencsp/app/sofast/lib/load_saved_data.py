"""Library of functions that load saved Sofast HDF5 files
and return OpenCSP optics classes.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from   opencsp.common.lib.csp.Facet import Facet
from   opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from   opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from   opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from   opencsp.common.lib.geometry.RegionXY import RegionXY
from   opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from   opencsp.common.lib.geometry.Uxyz import Uxyz
from   opencsp.common.lib.geometry.Vxy import Vxy
from   opencsp.common.lib.geometry.Vxyz import Vxyz
from   opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


def load_ideal_facet_ensemble_from_hdf(file: str, focal_length: float) -> FacetEnsemble:
    """Uses the "optic_definition" saved in SOFAST HDF file and given
    overall focal length to create a reference FacetEnsemble.

    Parameters
    ----------
    file : str
        HDF file of saved Sofast data
    focal_length : float
        Overall focal length (focal length of facets, not ensemble canting)

    Returns
    -------
    FacetEnsemble
        Ideal representation of FacetEnsemble made of MirrorParametric objects.
        Defined as "rotation_defined."
    """
    # Load ensemble definition data
    data_ensemble = load_hdf5_datasets(
        [
            'DataSofastInput/optic_definition/ensemble/r_facet_ensemble',
            'DataSofastInput/optic_definition/ensemble/v_facet_locations',
        ], file)
    vs_facet_loc = Vxyz(data_ensemble['v_facet_locations'])
    rs_facet_ensemble = [Rotation.from_rotvec(v) for v in data_ensemble['r_facet_ensemble']]

    # Load ensemble definition data
    num_facets = len(vs_facet_loc)
    facets = []
    for idx_facet in range(num_facets):
        data = load_hdf5_datasets(
            [
                f'DataSofastInput/optic_definition/facet_{idx_facet:03d}/v_facet_corners',
            ], file)

        # Create mirror region
        v_facet_corners = Vxy(data['v_facet_corners'][:2])
        region_facet = RegionXY.from_vertices(v_facet_corners)

        # Create mirror
        mirror = MirrorParametric.generate_symmetric_paraboloid(focal_length, region_facet)

        # Create facet
        facet = Facet(mirror)

        # Calculate facet position
        v_location = vs_facet_loc[idx_facet]

        # Calculate facet pointing rotation object (from +z vector)
        r_canting = rs_facet_ensemble[idx_facet]

        # Position facet in FacetEnsemble
        facet.set_position_in_space(v_location, r_canting)

        facets.append(facet)

    # Create FacetEnsemble
    return FacetEnsemble.generate_rotation_defined(facets)


def load_facet_ensemble_from_hdf(file: str) -> FacetEnsemble:
    """Loads SOFAST data as list of MirrorPoint objects

    Parameters
    ----------
    file : str
        HDF file of saved Sofast data

    Returns
    -------
    FacetEnsemble
        Measured data contained in OpenCSP optics class. Defined as a
        "rotation_defined" Ensemble.
    """
    # Get number of facets
    data_ensemble = load_hdf5_datasets(
        ['DataSofastInput/optic_definition/ensemble/v_facet_locations'], file)
    num_facets = data_ensemble['v_facet_locations'].shape[1]

    facets = []
    for idx_facet in range(num_facets):
        data = load_hdf5_datasets(
            [
                f'DataSofastInput/optic_definition/facet_{idx_facet:03d}/v_facet_corners',
                f'DataSofastCalculation/ensemble/facet_{idx_facet:03d}/trans_facet_ensemble',
                f'DataSofastCalculation/facet/facet_{idx_facet:03d}/slopes_facet_xy',
                f'DataSofastCalculation/facet/facet_{idx_facet:03d}/v_surf_points_facet',
            ], file)

        # Create facet region
        v_facet_corners = Vxy(data['v_facet_corners'][:2])
        region_facet = RegionXY.from_vertices(v_facet_corners)

        # Get mirror data
        surface_points = Vxyz(data['v_surf_points_facet'])
        slopes = data['slopes_facet_xy']

        surface_normals = np.ones((3, slopes.shape[1]))
        surface_normals[:2] = -slopes
        surface_normals = Uxyz(surface_normals)

        # Get measured facet orieintation
        trans_facet_ensemble = TransformXYZ(data['trans_facet_ensemble'])

        # Define mirror
        mirror = MirrorPoint(surface_points, surface_normals, region_facet, 'nearest')

        # Define facet
        facet = Facet(mirror)
        facet.set_position_in_space(trans_facet_ensemble.V, trans_facet_ensemble.R)

        facets.append(facet)

    # Create ensemble
    return FacetEnsemble.generate_rotation_defined(facets)


def load_ideal_facet_from_hdf(file: str, focal_length: float) -> Facet:
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
    data = load_hdf5_datasets(
        ['DataSofastInput/optic_definition/facet_000/v_facet_corners'], file)

    # Create mirror
    v_facet_corners = Vxy(data['v_facet_corners'][:2])
    region_facet = RegionXY.from_vertices(v_facet_corners)
    mirror = MirrorParametric.generate_symmetric_paraboloid(focal_length, region_facet)

    # Create facet
    return Facet.generate_rotation_defined(mirror)


def load_facet_from_hdf(file: str) -> Facet:
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
            'DataSofastInput/optic_definition/facet_000/v_facet_corners',
            'DataSofastCalculation/facet/facet_000/slopes_facet_xy',
            'DataSofastCalculation/facet/facet_000/v_surf_points_facet',
        ], file)

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
    mirror = MirrorPoint(surface_points, surface_normals, region_facet, 'nearest')

    # Define facet
    return Facet.generate_rotation_defined(mirror)
