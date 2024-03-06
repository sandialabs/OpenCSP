"""Script that displays a fixed pattern on a projector-screen system for 1 second

NOTE: This example requires a computer screen
"""
from os.path import join, dirname

import pytest

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternScreenProjection import (
    FixedPatternScreenProjection,
)
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection


@pytest.mark.no_xvfb
def example_project_fixed_pattern():
    """Projects fixed pattern image on display"""
    # Set pattern parameters
    file_image_projection = join(
        opencsp_code_dir(),
        "test/data/sofast_measurements/general/Image_Projection_test.h5",
    )
    width_pattern = 3
    spacing_pattern = 6

    # Load ImageProjection
    im_proj = ImageProjection.load_from_hdf_and_display(file_image_projection)

    fixed_pattern = FixedPatternScreenProjection(
        im_proj.size_x, im_proj.size_y, width_pattern, spacing_pattern
    )
    image = fixed_pattern.get_image('uint8', 255, 'square')

    # Project image
    im_proj.display_image_in_active_area(image)
    im_proj.root.after(1000, im_proj.close)
    im_proj.run()


if __name__ == '__main__':
    example_project_fixed_pattern()
