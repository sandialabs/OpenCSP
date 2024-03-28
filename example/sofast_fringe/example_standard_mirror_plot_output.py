import os
from os.path import join

from scipy.spatial.transform import Rotation

import contrib.app.sofast.load_saved_data as lsd
import opencsp.common.lib.csp.standard_output as so
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


def plot_sofast_single_facet(data_file: str, dir_save: str, focal_length_paraboloid: float) -> None:
    """Loads and visualizes CSP optic from saved SOFAST HDF file
    containing measured data of an NSTTF Facet.

    Parameters
    ----------
    data_file : str
        Processed Sofast dataset
    dir_save : str
        Directory to save standard output image files
    focal_length_paraboloid : float
        Focal length of ideal symmetric parabolid
    """
    # Load data
    optic_meas = lsd.load_facet_from_hdf(data_file)
    optic_ref = lsd.load_ideal_facet_from_hdf(data_file, focal_length_paraboloid)

    # Define scene
    v_target_center = Vxyz((0, 0, 56.57))
    v_target_normal = Vxyz((0, 1, 0))
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=20)

    # Define reference optic position/orientation
    rot = Rotation.from_euler('x', [22.5], degrees=True)
    optic_ref.set_pointing(rot)
    r = Rotation.identity()
    v = Vxyz((0, 56.57, 0))
    optic_ref.set_position_in_space(v, r)

    # Define measured optic position/orientation
    rot = Rotation.from_euler('x', [22.5], degrees=True)
    optic_meas.set_pointing(rot)
    r = Rotation.identity()
    v = Vxyz((0, 56.57, 0))
    optic_meas.set_position_in_space(v, r)

    # Define visualization parameters
    options = so.VisualizationOptions()
    options.slope_clim = 7
    options.slope_error_clim = 1.5
    if dir_save is None:
        options.to_save = False
    else:
        options.to_save = True
        options.output_dir = dir_save

    # Create standard output plots
    so.standard_output(optic_meas, optic_ref, source, v_target_center, v_target_normal, options)


def plot_sofast_facet_ensemble(data_file: str, dir_save: str, focal_length_paraboloid: float) -> None:
    """Loads and visualizes CSP optic from saved SOFAST HDF file
    containing measured data of an NSTTF Heliostat.
    """
    # Load data
    optic_meas = lsd.load_facet_ensemble_from_hdf(data_file)
    optic_ref = lsd.load_ideal_facet_ensemble_from_hdf(data_file, focal_length_paraboloid)

    # Define scene
    v_target_center = Vxyz((0, 0, 56.57))
    v_target_normal = Vxyz((0, 1, 0))
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=20)

    # Define reference optic position/orientation
    rot = Rotation.from_euler('x', [22.5], degrees=True)
    optic_ref.set_pointing(rot)
    r = Rotation.identity()
    v = Vxyz((0, 56.57, 0))
    optic_ref.set_position_in_space(v, r)

    # Define measured optic position/orientation
    rot = Rotation.from_euler('x', [22.5], degrees=True)
    optic_meas.set_pointing(rot)
    r = Rotation.identity()
    v = Vxyz((0, 56.57, 0))
    optic_meas.set_position_in_space(v, r)

    # Define visualization parameters
    options = so.VisualizationOptions()
    options.slope_map_quiver_density = 0.4
    options.slope_clim = 30
    if dir_save is None:
        options.to_save = False
    else:
        options.to_save = True
        options.output_dir = dir_save

    # Create standard output plots
    so.standard_output(optic_meas, optic_ref, source, v_target_center, v_target_normal, options)


def example_driver():
    """Visualizes a sample optic slope map using a previously processed saved Sofast dataset.

    1) Creates an OpenCSP representation of optic measured by Sofast
    2) Creates an OpenCSP representation of an ideal optic
    3) Performs ray trace of FacetEnsembles
    4) Plot orthorectified slope maps
    5) Plot orthorectified slope error map
    6) Plot optic in 3d
    7) Plot sun images on receiver
    8) Plot ensquared energy curve

    """
    # Define measured and reference data
    sample_data_dir = join(opencsp_code_dir(), 'test/data/measurements_sofast_fringe/')

    save_dir = join(os.path.dirname(__file__), 'data/output/standard_output')

    # SOFAST collect of NSTTF Facet
    file_data = join(sample_data_dir, 'calculations_facet/data.h5')
    plot_sofast_single_facet(file_data, join(save_dir, 'facet'), 100.0)

    # SOFAST collect of lab heliostat
    file_data = join(sample_data_dir, 'calculations_facet_ensemble/data.h5')
    plot_sofast_facet_ensemble(file_data, join(save_dir, 'ensemble'), 1000.0)


if __name__ == '__main__':
    example_driver()
