"""Script that displays a fixed pattern on a projector-screen system. Press the "escape"
key to close the window.
"""

from os.path import join

from opencsp.app.sofast.lib.PatternSofastFixed import PatternSofastFixed
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection


def run_fixed_pattern_projection():
    """Projects fixed pattern image on display"""

    # Load ImageProjection
    file_image_projection = join(
        opencsp_code_dir(), "../../sofast_calibration_files/Image_Projection_optics_lab_landscape_square.h5"
    )
    im_proj = ImageProjection.load_from_hdf_and_display(file_image_projection)

    # Define SofastFixedPattern object
    width_pattern = 3
    spacing_pattern = 6
    fixed_pattern = PatternSofastFixed(im_proj.size_x, im_proj.size_y, width_pattern, spacing_pattern)
    image = fixed_pattern.get_image("uint8", 255, "square")

    # Project image (press escape to close window)
    im_proj.display_image_in_active_area(image)
    im_proj.run()


if __name__ == "__main__":
    run_fixed_pattern_projection()
