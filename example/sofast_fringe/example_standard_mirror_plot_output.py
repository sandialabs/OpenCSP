import os
from os.path import join

import opencsp.app.sofast.lib.load_saved_data as lsd
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_single_facet() -> None:
    """Loads and visualizes CSP facet from saved Sofast HDF file containing measured data of an NSTTF Facet.

    1. Load Sofast measurement data
    2. Define viewing/illumination geometry
    3. Create standard output plots:
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

    lt.logger(os.path.join(dir_save, 'log.txt'), level=lt.log.INFO)

    # Define data file
    file_data = join(opencsp_code_dir(), 'test/data/sofast_fringe/data_expected_facet/data.h5')

    # 1. Load Sofast measurement data
    # ===============================
    optic_meas = lsd.load_mirror_from_hdf(file_data)
    optic_ref = lsd.load_ideal_mirror_from_hdf(file_data, 100.0)

    # 2. Define viewing/illumination geometry
    # =======================================
    v_target_center = Vxyz((0, 0, 100))
    v_target_normal = Vxyz((0, 0, -1))
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=40)

    # 3. Create standard output plots
    # ===============================

    # Define optics to plot
    output = StandardPlotOutput()
    output.optic_measured = optic_meas
    output.optic_reference = optic_ref

    # Update visualization parameters
    output.options_slope_vis.clim = 7
    output.options_slope_deviation_vis.clim = 1.5
    output.options_ray_trace_vis.enclosed_energy_max_semi_width = 1
    output.options_file_output.to_save = True
    output.options_file_output.output_dir = dir_save

    # Define ray trace parameters
    output.params_ray_trace.source = source
    output.params_ray_trace.v_target_center = v_target_center
    output.params_ray_trace.v_target_normal = v_target_normal

    # Create standard output plots
    output.plot()


if __name__ == '__main__':
    example_single_facet()
