import os
from os.path import join

import numpy as np

from opencsp.app.scene_reconstruction.lib.SceneReconstruction import SceneReconstruction
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


def example_scene_reconstruction():
    """Example script that reconstructs the XYZ locations of Aruco markers in a scene."""
    # Define input directory
    dir_input = join(
        opencsp_code_dir(),
        'app/scene_reconstruction/test/data/data_measurement',
    )

    # Define output directory
    save_dir = join(os.path.dirname(__file__), 'data/output')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    VERBOSITY = 2  # 0=no output, 1=only print outputs, 2=print outputs and show plots, 3=plots only with no printing

    # Load components
    camera = Camera.load_from_hdf(join(dir_input, 'camera.h5'))
    known_point_locations = np.loadtxt(
        join(dir_input, 'known_point_locations.csv'), delimiter=',', skiprows=1
    )
    image_filter_path = join(dir_input, 'aruco_marker_images', '*.JPG')
    point_pair_distances = np.loadtxt(
        join(dir_input, 'point_pair_distances.csv'), delimiter=',', skiprows=1
    )
    alignment_points = np.loadtxt(
        join(dir_input, 'alignment_points.csv'), delimiter=',', skiprows=1
    )

    # Perform marker position calibration
    cal_scene_recon = SceneReconstruction(
        camera, known_point_locations, image_filter_path
    )
    cal_scene_recon.run_calibration(VERBOSITY)

    # Scale points
    point_pairs = point_pair_distances[:, :2].astype(int)
    distances = point_pair_distances[:, 2]
    cal_scene_recon.scale_points(point_pairs, distances, verbose=VERBOSITY)

    # Align points
    marker_ids = alignment_points[:, 0].astype(int)
    alignment_values = Vxyz(alignment_points[:, 1:4].T)
    cal_scene_recon.align_points(marker_ids, alignment_values, verbose=VERBOSITY)

    # Save points as CSV
    cal_scene_recon.save_data_as_csv(join(save_dir, 'point_locations.csv'))

    # Save calibrtion figures
    for fig in cal_scene_recon.figures:
        fig.savefig(join(save_dir, fig.get_label() + '.png'))


if __name__ == '__main__':
    example_scene_reconstruction()
