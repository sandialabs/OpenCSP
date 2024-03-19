"""Tests projecting a test dot pattern onto a screen
"""

import os

import pytest

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection


@pytest.mark.no_xvfb
def test_project_fixed_pattern_target():
    # Set pattern parameters
    file_image_projection = os.path.join(
        opencsp_code_dir(),
        "test/data/measurements_sofast_fringe/general/image_projection_test.h5",
    )

    # Load ImageProjection
    im_proj = ImageProjection.load_from_hdf_and_display(file_image_projection)

    fixed_pattern = SystemSofastFixed(
        im_proj.size_x, im_proj.size_y, width_pattern=3, spacing_pattern=6
    )
    image = fixed_pattern.get_image('uint8', 255, 'square')

    # Project image
    im_proj.display_image_in_active_area(image)
    im_proj.root.after(500, im_proj.close)
    im_proj.run()


if __name__ == '__main__':
    test_project_fixed_pattern_target()
