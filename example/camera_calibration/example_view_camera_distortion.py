from os.path import join, dirname

import matplotlib.pyplot as plt

from opencsp.app.camera_calibration.lib.calibration_camera import view_distortion
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft


def example_show_camera_distortion():
    """Plots visualization of camera distortion given a saved Camera HDF file

    1. Loads camera HDF file
    2. Plots distortion maps
    """
    # Define input camera file
    file = join(opencsp_code_dir(), 'test/data/sofast_common/camera_sofast.h5')

    # Load camera
    cam = Camera.load_from_hdf(file)

    # Create axes
    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    # View distortion
    view_distortion(cam, ax1, ax2, ax3)

    # Save image
    dir_save = join(dirname(__file__), 'data/output/camera_distortion')
    ft.create_directories_if_necessary(dir_save)
    fig.savefig(join(dir_save, 'distortion_plot.png'))
    plt.close('all')


if __name__ == '__main__':
    example_show_camera_distortion()
