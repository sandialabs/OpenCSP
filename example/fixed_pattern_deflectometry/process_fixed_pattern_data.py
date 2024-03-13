"""Example script that runs fixed pattern deflectometry analysis on saved data
"""
from os.path import join, dirname

from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.FacetData import FacetData
from opencsp.common.lib.deflectometry.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg


def process(
    file_camera: str,
    file_facet: str,
    file_dot_locs: str,
    file_ori: str,
    file_meas: str,
    save_dir: str,
    surface_data: dict,
) -> ProcessSofastFixed:
    """Performs fixed pattern deflectometry processing"""
    # Load data
    camera = Camera.load_from_hdf(file_camera)
    facet_data = FacetData.load_from_json(file_facet)
    fixed_pattern_dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
    orientation = SpatialOrientation.load_from_hdf(file_ori)
    measurement = MeasurementSofastFixed.load_from_hdf(file_meas)

    # Instantiate class
    fixed_pattern = ProcessSofastFixed(
        orientation, camera, fixed_pattern_dot_locs, facet_data
    )
    fixed_pattern.load_measurement_data(measurement)

    # Process
    fixed_pattern.process_single_facet_optic(surface_data)

    # Print focal lengths
    surf_coefs = fixed_pattern.data_slope_solver.surf_coefs_facet
    focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
    print('Parabolic fit focal lengths:')
    print(f'  X {focal_lengths_xy[0]:.3f} m')
    print(f'  Y {focal_lengths_xy[1]:.3f} m')

    # Plot slope image
    figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
    axis_control_m = rca.meters()

    fig_mng = fm.setup_figure(figure_control, axis_control_m, title='')
    mirror = fixed_pattern.get_mirror('bilinear')
    mirror.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_mng.axis)
    fig_mng.save(save_dir, 'orthorectified_slope_magnitude', 'png')

    # Save data
    fixed_pattern.save_to_hdf(join(save_dir, 'calculations.h5'))


def example_process_fixed_pattern_printed_target():
    """Example function that processes a fixed pattern data collect using a physical,
    printed dot target.
    """
    dir_base = join(
        opencsp_code_dir(), '../../sample_data/deflectometry/sandia_lab'
    )

    file_camera = join(dir_base, "calibration_files/camera.h5")
    file_facet = join(dir_base, "calibration_files/Facet_NSTTF.json")
    file_ori = join(dir_base, 'fixed_pattern/spatial_orientation.h5')
    file_dot_locs = join(dir_base, 'fixed_pattern/dot_locations_printed_target.h5')
    file_meas = join(dir_base, 'fixed_pattern/measurement_printed_target_1.h5')
    dir_output = join(dirname(__file__), 'data/output/printed_target')

    surface_data = dict(
        surface_type='parabolic',
        initial_focal_lengths_xy=(150.0, 150),
        robust_least_squares=False,
        downsample=1,
    )

    process(
        file_camera,
        file_facet,
        file_dot_locs,
        file_ori,
        file_meas,
        dir_output,
        surface_data,
    )


def example_process_fixed_pattern_screen_target():
    """Loads data and calls processing function"""
    dir_base = join(
        opencsp_code_dir(), '../../sample_data/deflectometry/sandia_lab'
    )

    # Define files
    file_camera = join(dir_base, "calibration_files/camera.h5")
    file_facet = join(dir_base, "calibration_files/Facet_NSTTF.json")
    file_ori = join(dir_base, 'fixed_pattern/spatial_orientation.h5')
    file_dot_locs = join(
        dir_base, 'fixed_pattern/dot_locations_screen_square_width3_space6.h5'
    )
    file_meas = join(
        dir_base, 'fixed_pattern/measurement_screen_square_width3_space6.h5'
    )
    dir_output = join(dirname(__file__), 'data/output/screen_target')

    surface_data = dict(
        surface_type='parabolic',
        initial_focal_lengths_xy=(150.0, 150),
        robust_least_squares=False,
        downsample=1,
    )

    process(
        file_camera,
        file_facet,
        file_dot_locs,
        file_ori,
        file_meas,
        dir_output,
        surface_data,
    )


if __name__ == '__main__':
    example_process_fixed_pattern_printed_target()
    example_process_fixed_pattern_screen_target()
