import os

import matplotlib.pyplot as plt

import opencsp
from   opencsp.app.camera_calibration.lib.calibration_camera import view_distortion
from   opencsp.common.lib.camera.Camera import Camera

def example_driver():
    """Example SOFAST script

    Plots visualization of camera distortion given a saved Camera HDF file
    1. Loads camera HDF file
    2. Plots distortion maps

    """
    # Define input camera file
    file = os.path.join(os.path.dirname(opencsp.__file__), 'test/data/sofast_measurements/camera.h5')

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
    dir_save = os.path.join(os.path.dirname(__file__), 'data/output/camera_distortion')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    fig.savefig(os.path.join(dir_save, 'distortion_plot.png'))

if __name__ == '__main__':
    example_driver()
