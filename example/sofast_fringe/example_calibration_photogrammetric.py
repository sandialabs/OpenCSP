from glob import glob
import os
from os.path import join

import matplotlib
import numpy as np
from numpy import ndarray

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.app.scene_reconstruction.lib.SceneReconstruction import SceneReconstruction
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
from opencsp.common.lib.deflectometry.CalibrationCameraPosition import (
    CalibrationCameraPosition,
)
from opencsp.app.sofast.lib.CalibrateDisplayShape import (
    CalibrateDisplayShape,
    DataInput,
)
from opencsp.app.sofast.lib.save_DisplayShape_file import (
    save_DisplayShape_file,
)
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.photogrammetry.photogrammetry import load_image_grayscale


def run_scene_reconstruction(dir_input: str, verbose: int) -> SceneReconstruction:
    """Runs full Sofast calibration sequence"""
    # Define input file paths
    file_camera = join(dir_input, 'camera.h5')
    file_known_points = join(dir_input, 'known_point_locations.csv')
    images_pattern = join(dir_input, 'aruco_marker_images', '*.JPG')
    file_point_pair_distances = join(dir_input, 'point_pair_distances.csv')
    file_alignment_points = join(dir_input, 'alignment_points.csv')

    # Load components
    camera = Camera.load_from_hdf(file_camera)
    known_point_locations = np.loadtxt(
        file_known_points, delimiter=',', skiprows=1)
    point_pair_distances = np.loadtxt(
        file_point_pair_distances, delimiter=',', skiprows=1
    )
    alignment_points = np.loadtxt(
        file_alignment_points, delimiter=',', skiprows=1)

    # Perform marker position calibration
    cal = SceneReconstruction(camera, known_point_locations, images_pattern)
    cal.run_calibration(verbose)

    # Scale points
    point_pairs = point_pair_distances[:, :2].astype(int)
    distances = point_pair_distances[:, 2]
    cal.scale_points(point_pairs, distances, verbose=verbose)

    # Align points
    marker_ids = alignment_points[:, 0].astype(int)
    alignment_values = Vxyz(alignment_points[:, 1:4].T)
    cal.align_points(marker_ids, alignment_values, verbose=verbose)

    return cal


def run_screen_shape_cal(
    pts_marker_data: ndarray, dir_input: str, verbose: int
) -> CalibrateDisplayShape:
    """Runs screen shape calibration

    Parameters:
    -----------
    pts_marker_data : ndarray
        Output from SceneReconstruction()
    """
    # Define input files
    resolution_xy = [100, 100]  # sample density of screen
    file_screen_cal_point_pairs = join(
        dir_input, 'screen_calibration_point_pairs.csv')
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
    image_projection_data = ImageProjection.load_from_hdf(
        file_image_projection)

    # Store input data in data class
    data_input = DataInput(
        corner_ids,
        screen_cal_point_pairs,
        resolution_xy,
        pts_xyz_marker,
        camera,
        image_projection_data,
        [Measurement.load_from_hdf(f) for f in files_screen_shape_measurement],
    )

    # Perform screen position calibration
    cal = CalibrateDisplayShape(data_input)
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


def example_driver():
    """Example script that runs full Sofast photogrammetric calibration routine

    1. Marker position calibration
    2. Screen position calibration
    3. Camera position calibration
    4. Saves Display object

    """
    # Define input file directories
    base_dir_scene_recon = join(
        opencsp_code_dir(),
        'app/scene_reconstruction/test/data/data_measurement',
    )  # low-res test data
    base_dir_sofast = join(
        opencsp_code_dir(),
        'common/lib/deflectometry/test/data/data_measurement',
    )  # low-res test data

    # Define save path
    save_dir = join(
        os.path.dirname(__file__), 'data/output/photogrammetric_calibration'
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    VERBOSITY = 2  # 0=no output, 1=only print outputs, 2=print outputs and show plots, 3=plots only with no printing

    # Run scene reconstruction to get xyz point locations
    cal_scene_recon = run_scene_reconstruction(base_dir_scene_recon, VERBOSITY)
    pts_data = cal_scene_recon.get_data()

    # Run screen shape calibration
    cal_screen_shape = run_screen_shape_cal(
        pts_data, base_dir_sofast, VERBOSITY)

    # Run camera position calibration
    cal_camera_pose = run_camera_position_cal(
        pts_data, base_dir_sofast, VERBOSITY)

    # Get physical setup file data
    NAME = 'Example physical setup file'
    screen_distortion_data = cal_screen_shape.get_data()
    rvec, tvec = cal_camera_pose.get_data()

    # Save calibration figures
    for fig in (
        cal_scene_recon.figures + cal_screen_shape.figures + cal_camera_pose.figures
    ):
        fig.savefig(join(save_dir, fig.get_label() + '.png'))

    # Save data
    cal_scene_recon.save_data_as_csv(join(save_dir, 'point_locations.csv'))
    cal_screen_shape.save_data_as_hdf(
        join(save_dir, 'screen_distortion_data.h5'))
    cal_camera_pose.save_data_as_csv(join(save_dir, 'camera_rvec_tvec.csv'))

    # Save display file
    file_save = join(save_dir, 'example_physical_setup_file.h5')
    save_physical_setup_file(screen_distortion_data,
                             NAME, rvec, tvec, file_save)


if __name__ == '__main__':
    example_driver()
