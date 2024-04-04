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


def example_screen_shape_calibration():
    """Example Sofast calibration script

    Calibrates the 3d shape of a screen:
    1. Load measured calibration data
    2. Perform screen shpae calibration
    3. Save 3d shape data as DisplayShape object
    4. Save calculation figures
    """
    # General setup
    # =============

    # Define save directory
    dir_save = join(dirname(__file__), 'data/output/screen_shape')
    dir_save_figures = join(dir_save, 'figures')
    ft.create_directories_if_necessary(dir_save_figures)

    # Set up logger
    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # Define input files
    file_pts_data = join(opencsp_code_dir(), 'test/data/sofast_common/aruco_corner_locations.csv')
    file_screen_cal_point_pairs = join(
        opencsp_code_dir(), 'test/data/display_shape_calibration/data_measurement/screen_calibration_point_pairs.csv'
    )
    file_camera_distortion = join(
        opencsp_code_dir(), 'test/data/display_shape_calibration/data_measurement/camera_screen_shape.h5'
    )
    file_image_projection = join(opencsp_code_dir(), 'test/data/sofast_common/image_projection.h5')
    files_screen_shape_measurement = glob(
        join(
            opencsp_code_dir(),
            'test/data/display_shape_calibration/data_measurement/screen_shape_sofast_measurements/pose_*.h5',
        )
    )

    # 1. Load measured calibration data
    # =================================

    # Load output data from Scene Reconstruction (Aruco marker xyz points)
    pts_marker_data = np.loadtxt(file_pts_data, delimiter=',', skiprows=1)
    pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
    corner_ids = pts_marker_data[:, 1]

    # Define desired resolution of screen sample grid
    resolution_xy = [100, 100]

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

    # 2. Perform screen shape calibration
    # ====================================
    cal = CalibrateDisplayShape(data_input)
    cal.make_figures = True
    cal.run_calibration()

    # 3. Save 3d shape data as DisplayShape object
    # ============================================

    # Get screen shape data
    display_shape = cal.as_DisplayShape('Example display shape')

    # Save DisplayShape file
    file = join(dir_save, 'display_shape.h5')
    display_shape.save_to_hdf(file)
    lt.info(f'Saved DisplayShape file to {file:s}')

    # 4. Save calculation figures
    # ===========================
    for fig in cal.figures:
        file = join(dir_save_figures, fig.get_label() + '.png')
        lt.info(f'Saving figure to: {file:s}')
        fig.savefig(file)


if __name__ == '__main__':
    example_screen_shape_calibration()
