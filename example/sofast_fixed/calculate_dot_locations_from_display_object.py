"""Example script that creates and saves a DotLocationsFixedPattern flie and a SpaitalOrientation
file using a previously created Display object. This is only when displaying a fixed dot pattern
on a screen.
"""
import os
from os.path import join, dirname, exists

from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


def example_calculate_dot_locs_from_display():
    """Creates a DotLocationsFixedPattern object from a previously created Display object"""
    dir_base = join(
        opencsp_code_dir(),
        '../../sample_data/deflectometry/sandia_lab/calibration_files',
    )

    # Define files
    file_display = join(dir_base, 'display_distorted_3d.h5')
    file_image_projection = join(
        dir_base, 'Image_Projection_optics_lab_landscape_square.h5'
    )
    dir_save = join(dirname(__file__), 'data/output/dot_location_file')

    # Define dot parameters
    width_dot = 3  # pixels
    spacing_dot = 6  # pixels

    if not exists(dir_save):
        os.makedirs(dir_save)

    # Load data
    display = Display.load_from_hdf(file_display)
    im_proj_params = ImageProjection.load_from_hdf(file_image_projection)

    # Calculate fixed pattern display parameters
    projection = SystemSofastFixed(
        im_proj_params['size_x'], im_proj_params['size_y'], width_dot, spacing_dot
    )
    fixed_pattern_dot_locs = DotLocationsFixedPattern.from_projection_and_display(
        projection, display
    )

    # Calculate spatial orientation
    orientation = SpatialOrientation(display.r_cam_screen, display.v_cam_screen_cam)

    # Save data sets
    fixed_pattern_dot_locs.save_to_hdf(
        join(dir_save, f'fixed_pattern_display_w{width_dot:d}_s{spacing_dot:d}.h5')
    )
    orientation.save_to_hdf(join(dir_save, 'spatial_orientation.h5'))


if __name__ == '__main__':
    example_calculate_dot_locs_from_display()
