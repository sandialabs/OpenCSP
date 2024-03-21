from os.path import join, dirname

import numpy as np

from opencsp.app.sofast.lib.save_DisplayShape_file import save_DisplayShape_file
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets
import opencsp.common.lib.tool.file_tools as ft


def example_save_display_shape_file():
    """Example script that saves a DisplayShape file from its components"""
    # Define save directory
    save_dir = join(dirname(__file__), 'data/output/save_DisplayShape_file')
    ft.create_directories_if_necessary(save_dir)

    # Load screen distortion data
    file_screen_distortion_data = join(
        opencsp_code_dir(), 'app/sofast/test/data/data_expected', 'screen_distortion_data_100_100.h5'
    )
    datasets = ['pts_xy_screen_fraction', 'pts_xyz_screen_coords']
    data = load_hdf5_datasets(datasets, file_screen_distortion_data)
    screen_distortion_data = {
        'pts_xy_screen_fraction': Vxy(data['pts_xy_screen_fraction']),
        'pts_xyz_screen_coords': Vxyz(data['pts_xyz_screen_coords']),
    }

    # Load rvec and tvec
    file_rvec_tvec = join(
        opencsp_code_dir(), 'common/lib/deflectometry/test/data/data_expected', 'camera_rvec_tvec.csv'
    )
    pose_data = np.loadtxt(file_rvec_tvec, delimiter=',')
    rvec = pose_data[0]
    tvec = pose_data[1]

    # Save DisplayShape file
    name = 'Example physical setup file'
    file_save = join(save_dir, 'example_display_shape_file.h5')
    save_DisplayShape_file(screen_distortion_data, name, rvec, tvec, file_save)


if __name__ == '__main__':
    example_save_display_shape_file()
