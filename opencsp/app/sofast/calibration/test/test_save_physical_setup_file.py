import os
from os.path import join

import numpy as np

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.app.sofast.calibration.lib.save_physical_setup_file import (
    save_physical_setup_file,
)
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.file_tools as ft


def test_save_physical_setup_file():
    """Loads data and saves test Display file"""
    # Define input file directory
    base_dir = join(opencsp_code_dir(), 'common/lib/deflectometry/test')
    dir_input = join(base_dir, 'data', 'data_expected')
    dir_output = join(base_dir, 'data', 'output')
    file_save = join(dir_output, 'test_physical_setup_file.h5')

    ft.create_directories_if_necessary(dir_output)

    # Define data files
    file_screen_distortion_data = join(
        dir_input, 'screen_distortion_data_100_100.h5')
    file_cam = join(dir_input, 'camera_rvec_tvec.csv')

    # Load data
    name = 'Test Physical Setup File'
    data_dist = load_hdf5_datasets(
        ['pts_xy_screen_fraction', 'pts_xyz_screen_coords'], file_screen_distortion_data
    )
    screen_distortion_data = {
        'pts_xy_screen_fraction': Vxy(data_dist['pts_xy_screen_fraction']),
        'pts_xyz_screen_coords': Vxyz(data_dist['pts_xyz_screen_coords']),
    }
    data_cam = np.loadtxt(file_cam, delimiter=',')
    rvec = data_cam[0]
    tvec = data_cam[1]

    # Save physical setup file
    save_physical_setup_file(screen_distortion_data,
                             name, rvec, tvec, file_save)
    print('Test physical setup file save successfully.')


if __name__ == '__main__':
    test_save_physical_setup_file()
