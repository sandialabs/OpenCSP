"""Example script that runs full Sofast manual calibration routine

1. Screen position calibration
2. Camera position calibration
3. Saves Display object

"""
from datetime import datetime as dt
from glob import glob
import os
from os.path import join

import numpy as np
from numpy import ndarray

from opencsp.common.lib.deflectometry.CalibrationCameraPosition import (
    CalibrationCameraPosition,
)
from opencsp.app.sofast.calibration.lib.CalibrationScreenShape import (
    CalibrationScreenShape,
    DataInput,
)
from opencsp.app.sofast.calibration.lib.save_physical_setup_file import (
    save_physical_setup_file,
)
from opencsp.app.sofast.lib.Measurement import Measurement
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.photogrammetry.photogrammetry import load_image_grayscale


def run_screen_shape_cal(
    pts_marker_data: ndarray, dir_input: str, verbose: int
) -> CalibrationScreenShape:
    """Runs screen shape calibration

    Parameters:
    ----------
    pts_marker_data : ndarray
        Output from SceneReconstruction()
    """
    # Define input files
    resolution_xy = [100, 100]  # sample density of screen
    file_screen_cal_point_pairs = join(dir_input, 'screen_calibration_point_pairs.csv')
    file_camera_distortion = join(dir_input, 'camera_screen_shape.h5')
    file_image_projection = join(dir_input, 'image_projection.h5')
    files_screen_shape_measurement = glob(
        join(dir_input, 'screen_shape_sofast_measurements/pose_*.h5')
    )

    # Load input data
    pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
    corner_ids = pts_marker_data[:, 1]
    screen_cal_point_pairs = np.loadtxt(
        file_screen_cal_point_pairs, delimiter=',', skiprows=1
    ).astype(int)
    camera = Camera.load_from_hdf(file_camera_distortion)
    image_projection_data = ImageProjection.load_from_hdf(file_image_projection)

    # Store input data in data class
    data_input = DataInput(
        corner_ids,
        screen_cal_point_pairs,
        resolution_xy,
        pts_xyz_marker,
        camera,
        image_projection_data,
        [Measurement.load_from_hdf(f) for f in files_screen_shape_measurement],
        False,
    )

    # Perform screen position calibration
    cal = CalibrationScreenShape(data_input)
    cal.run_calibration(verbose)

    return cal


def run_camera_position_cal(
    pts_marker_data: ndarray, dir_input: str, verbose: int
) -> CalibrationCameraPosition:
    """Calibrates the position of the camera"""
    # Define inputs
    file_camera_sofast = join(dir_input, 'camera_sofast.h5')
    file_cal_image = join(dir_input, 'image_sofast_camera.png')

    # Load input data
    camera = Camera.load_from_hdf(file_camera_sofast)
    pts_xyz_marker = Vxyz(pts_marker_data[:, 2:].T)
    corner_ids = pts_marker_data[:, 1]
    image = load_image_grayscale(file_cal_image)

    # Perform camera position calibraiton
    cal = CalibrationCameraPosition(camera, pts_xyz_marker, corner_ids, image)
    cal.verbose = verbose
    cal.run_calibration()

    return cal


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)

    # Define input file directories (high-res sample data)
    base_dir_sofast = join(
        base_dir, '../../../sample_data/sofast/data_manual_calibration/data_measurement'
    )

    save_dir = join(
        base_dir,
        f'../../../sample_data/sofast/data_manual_calibration/{dt.now().strftime(r"%Y_%m_%d-%H.%M.%S"):s}_data',
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    VERBOSITY = 2  # 0=no output, 1=only print outputs, 2=print outputs and show plots, 3=plots only with no printing

    # Load manually measured point data
    pts_data = np.loadtxt(
        join(base_dir_sofast, 'point_locations.csv'), delimiter=',', skiprows=1
    )

    # Run screen shape calibration
    cal_screen_shape = run_screen_shape_cal(pts_data, base_dir_sofast, VERBOSITY)

    # Run camera position calibration
    cal_camera_pose = run_camera_position_cal(pts_data, base_dir_sofast, VERBOSITY)

    # Save calibration figures
    for fig in cal_screen_shape.figures + cal_camera_pose.figures:
        fig.savefig(join(save_dir, fig.get_label() + '.png'))

    # Save raw data
    cal_screen_shape.save_data_as_hdf(join(save_dir, 'screen_distortion_data.h5'))
    cal_camera_pose.save_data_as_csv(join(save_dir, 'camera_rvec_tvec.csv'))

    # Save physical setup file
    file_save = join(save_dir, 'example_physical_setup_file.h5')
    NAME = 'Example physical setup file'
    screen_distortion_data = cal_screen_shape.get_data()
    rvec, tvec = cal_camera_pose.get_data()
    save_physical_setup_file(screen_distortion_data, NAME, rvec, tvec, file_save)
