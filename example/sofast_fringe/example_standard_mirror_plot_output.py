import os
from os.path import join

from scipy.spatial.transform import Rotation

import contrib.app.sofast.load_saved_data as lsd
import opencsp.common.lib.csp.standard_output as so
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft


def example_single_facet() -> None:
    """Loads and visualizes CSP facet from saved Sofast HDF file containing measured data of an NSTTF Facet.

    1. Load Sofast measurement data
    2. Define viewing/illumination geometry
    3. Create an OpenCSP representation of an ideal facet
    4. Create an OpenCSP representation of facet measured by Sofast
    5. Create standard output plots:
        - Perform ray trace of facet
        - Plot orthorectified slope maps
        - Plot orthorectified slope error map
        - Plot facet in 3d
        - Plot sun images on receiver
        - Plot ensquared energy curve
    """
    # General setup
    # =============

    dir_save = join(os.path.dirname(__file__), 'data/output/standard_output/facet')
    ft.create_directories_if_necessary(dir_save)

    # Define data file
    file_data = join(opencsp_code_dir(), 'test/data/sofast_fringe/calculations_facet/data.h5')

    # 1. Load Sofast measurement data
    # ===============================
    optic_meas = lsd.load_facet_from_hdf(file_data)
    optic_ref = lsd.load_ideal_facet_from_hdf(file_data, 100.0)

    # 2. Define viewing/illumination geometry
    # =======================================
    v_target_center = Vxyz((0, 0, 56.57))
    v_target_normal = Vxyz((0, 1, 0))
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=20)

    # 3. Create an OpenCSP representation of an ideal facet
    # =====================================================
    rot = Rotation.from_euler('x', [22.5], degrees=True)
    optic_ref.set_pointing(rot)
    r = Rotation.identity()
    v = Vxyz((0, 56.57, 0))
    optic_ref.set_position_in_space(v, r)

    # 4. Creates an OpenCSP representation of facet measured by Sofast
    # ================================================================
    rot = Rotation.from_euler('x', [22.5], degrees=True)
    optic_meas.set_pointing(rot)
    r = Rotation.identity()
    v = Vxyz((0, 56.57, 0))
    optic_meas.set_position_in_space(v, r)

    # 5. Create standard output plots
    # ===============================

    # Define visualization parameters
    options = so.VisualizationOptions()
    options.slope_clim = 7
    options.slope_error_clim = 1.5
    options.to_save = True
    options.output_dir = dir_save

    # Create standard output plots
    so.standard_output(optic_meas, optic_ref, source, v_target_center, v_target_normal, options)


def example_facet_ensemble() -> None:
    """Loads and visualizes CSP facet ensemble from saved Sofast HDF file containing measured data of an NSTTF Facet.

    1. Load Sofast measurement data
    2. Define viewing/illumination geometry
    3. Create an OpenCSP representation of an ideal facet ensemble
    4. Create an OpenCSP representation of facet ensemble measured by Sofast
    5. Create standard output plots:
        - Perform ray trace of facet ensemble
        - Plot orthorectified slope maps
        - Plot orthorectified slope error map
        - Plot facet in 3d
        - Plot sun images on receiver
        - Plot ensquared energy curve
    """
    # General setup
    # =============

    dir_save = join(os.path.dirname(__file__), 'data/output/standard_output/ensemble')
    ft.create_directories_if_necessary(dir_save)

    # Define data file
    file_data = join(opencsp_code_dir(), 'test/data/sofast_fringe/calculations_facet_ensemble/data.h5')

    # 1. Load Sofast measurement data
    # ===============================
    optic_meas = lsd.load_facet_ensemble_from_hdf(file_data)
    optic_ref = lsd.load_ideal_facet_ensemble_from_hdf(file_data, 1000.0)

    # 2. Define viewing/illumination geometry
    # =======================================
    v_target_center = Vxyz((0, 0, 56.57))
    v_target_normal = Vxyz((0, 1, 0))
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=20)

    # 3. Create an OpenCSP representation of an ideal facet ensemble
    # ==============================================================
    rot = Rotation.from_euler('x', [22.5], degrees=True)
    optic_ref.set_pointing(rot)
    r = Rotation.identity()
    v = Vxyz((0, 56.57, 0))
    optic_ref.set_position_in_space(v, r)

    # 4. Creates an OpenCSP representation of facet ensemble measured by Sofast
    # =========================================================================
    rot = Rotation.from_euler('x', [22.5], degrees=True)
    optic_meas.set_pointing(rot)
    r = Rotation.identity()
    v = Vxyz((0, 56.57, 0))
    optic_meas.set_position_in_space(v, r)

    # 5. Create standard output plots
    # ===============================

    # Define visualization parameters
    options = so.VisualizationOptions()
    options.slope_map_quiver_density = 0.4
    options.slope_clim = 30
    options.to_save = True
    options.output_dir = dir_save

    # Create standard output plots
    so.standard_output(optic_meas, optic_ref, source, v_target_center, v_target_normal, options)


if __name__ == '__main__':
    example_single_facet()
    example_facet_ensemble()
