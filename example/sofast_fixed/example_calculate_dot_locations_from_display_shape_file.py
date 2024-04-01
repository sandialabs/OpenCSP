from os.path import join, dirname

from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_calculate_dot_locations_from_display_shape():
    """Creates a DotLocationsFixedPattern object from previously created DisplayShape and
    ImageProjection files.
    1. Load DisplayShape and image projection files
    2. Define dot projection object
    3. Define DotLocationsFixedPattern object
    4. Save DotLocationsFixedPattern to HDF5 file
    """
    # General Setup
    # =============
    dir_save = join(dirname(__file__), 'data/output/dot_locations_from_display_shape')
    ft.create_directories_if_necessary(dir_save)

    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # 1. Load DisplayShape and image projection files
    # ===============================================
    # Define files
    file_display = join(opencsp_code_dir(), 'test/data/sofast_common/display_distorted_3d.h5')
    file_image_projection = join(opencsp_code_dir(), 'test/data/sofast_common/image_projection.h5')

    # Load data
    display = DisplayShape.load_from_hdf(file_display)
    im_proj_params = ImageProjection.load_from_hdf(file_image_projection)

    # 2. Define dot projection object
    # ===============================
    width_dot = 3  # pixels
    spacing_dot = 6  # pixels

    # Create fixed pattern projection object
    projection = SystemSofastFixed(im_proj_params['size_x'], im_proj_params['size_y'], width_dot, spacing_dot)

    # 3. Define DotLocationsFixedPattern object
    # =========================================
    fixed_pattern_dot_locs = DotLocationsFixedPattern.from_projection_and_display(projection, display)

    # 4. Save DotLocationsFixedPattern to HDF5 file
    # =============================================
    file_dot_location_save = join(dir_save, 'fixed_pattern_dot_locations.h5')
    fixed_pattern_dot_locs.save_to_hdf(file_dot_location_save)
    lt.info(f'File saved to {file_dot_location_save:s}')


if __name__ == '__main__':
    example_calculate_dot_locations_from_display_shape()
