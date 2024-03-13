"""Script that saves a Sofast physical setup file from previously processed data
"""
from numpy import ndarray
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


def save_physical_setup_file(
    screen_distortion_data: dict,
    name: str,
    rvec: ndarray,
    tvec: ndarray,
    file_save: str,
) -> None:
    """Constructs and saves DisplayShape file

    Parameters
    ----------
    screen_distortion_data : dict
        Dict with following fields: 1) pts_xy_screen_fraction: Vxy, 2) pts_xyz_screen_coords: Vxyz
    name : str
        DisplayShape name
    rvec : ndarray
        Screen to camera rotation vector
    tvec : ndarray
        Camera to screen (in screen coordinates) translation vector
    file_save : str
        Output display file name to save to
    """
    # Load screen distortion data
    pts_xy_screen_fraction: Vxy = screen_distortion_data['pts_xy_screen_fraction']
    pts_xyz_screen_coords: Vxyz = screen_distortion_data['pts_xyz_screen_coords']

    # Gather rvec and tvec
    rot_screen_cam = Rotation.from_rotvec(rvec)
    v_cam_screen_screen = Vxyz(tvec)

    # Gather display grid data
    grid_data = dict(
        screen_model='distorted3D',
        Pxy_screen_fraction=pts_xy_screen_fraction,
        Pxyz_screen_coords=pts_xyz_screen_coords,
    )

    # Create display object
    display = DisplayShape(v_cam_screen_screen, rot_screen_cam, grid_data, name)

    # Save to HDF file
    display.save_to_hdf(file_save)
