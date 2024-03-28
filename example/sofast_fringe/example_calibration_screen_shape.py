from glob import glob
from os.path import join, dirname

import numpy as np

from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.CalibrateDisplayShape import CalibrateDisplayShape, DataInput
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def run_screen_shape_calibration(save_dir):
    """Runs screen shape calibration. Saves data to ./data/output/screen_shape"""
    # Load output data from Scene Reconstruction (Aruco marker xyz points)
    file_pts_data = join(
        opencsp_code_dir(), 'common/lib/deflectometry/test/data/data_measurement', 'point_locations.csv'
    )
    pts_marker_data = np.loadtxt(file_pts_data, delimiter=',', skiprows=1)
    pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
    corner_ids = pts_marker_data[:, 1]

    # Define desired resolution of screen sample grid
    resolution_xy = [100, 100]

    # Define directory where screen shape calibration data is saved
    base_dir_sofast_cal = join(opencsp_code_dir(), 'app/sofast/test/data/data_measurement')

    # Define input files
    file_screen_cal_point_pairs = join(base_dir_sofast_cal, 'screen_calibration_point_pairs.csv')
    file_camera_distortion = join(base_dir_sofast_cal, 'camera_screen_shape.h5')
    file_image_projection = join(base_dir_sofast_cal, 'image_projection.h5')
    files_screen_shape_measurement = glob(join(base_dir_sofast_cal, 'screen_shape_sofast_measurements/pose_*.h5'))

    # Load input data
    camera = Camera.load_from_hdf(file_camera_distortion)
    image_projection_data = ImageProjection.load_from_hdf(file_image_projection)
    screen_cal_point_pairs = np.loadtxt(file_screen_cal_point_pairs, delimiter=',', skiprows=1, dtype=int)

    # Store input data in data class
    data_input = DataInput(
        corner_ids,
        screen_cal_point_pairs,
        resolution_xy,
        pts_xyz_marker,
        camera,
        image_projection_data,
        [MeasurementSofastFringe.load_from_hdf(f) for f in files_screen_shape_measurement],
    )

    # Perform screen shape calibration
    cal = CalibrateDisplayShape(data_input)
    cal.make_figures = True
    cal.run_calibration()

    # Save screen shape data as HDF5 file
    cal.save_data_as_hdf(join(save_dir, 'screen_distortion_data.h5'))

    # Save calibration figures
    for fig in cal.figures:
        file = join(save_dir, fig.get_label() + '.png')
        lt.info(f'Saving figure to: {file:s}')
        fig.savefig(file)


def example_driver():
    # Define save directory
    save_path = join(dirname(__file__), 'data/output/screen_shape')
    ft.create_directories_if_necessary(save_path)

    # Set up logger
    lt.logger(join(save_path, 'log.txt'), lt.log.INFO)

    run_screen_shape_calibration(save_path)


if __name__ == '__main__':
    example_driver()
