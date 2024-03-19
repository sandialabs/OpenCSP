"""Library with function used to visualise a given Sofast setup in
3D given a display and camera file. Useful for debugging calibration errors.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


def visualize_setup(
    display: Display,
    camera: Camera,
    v_screen_object_screen: Vxyz = None,
    r_object_screen: Rotation = None,
    length_z_axis_cam: float = 8,
    axes_length: float = 2,
    min_axis_length_screen: float = 2,
    ax: plt.Axes | None = None,
):
    """Draws the given SOFAST setup components on a 3d axis.

    Parameters
    ----------
    display : Display
        SOFAST display object
    camera : Camera
        OpenCSP camera object
    v_screen_object_screen : Vxyz, optional
        Vector (m), screen to object in screen reference frame, by default None.
        If None, the object reference frame is not plotted.
    r_object_screen : Rotation, optional
        Rotation, object to screen reference frames, by default None.
        Only used if v_screen_object_screen is not None
    length_z_axis_cam : float, optional
        Length of camera z axis to draw (m), by default 8
    axes_length : float, optional
        Length of all other axes to draw (m), by default 2
    min_axis_length_screen : float, optional
        Minimum length of axes to draw (m), by default 2
    ax : plt.Axes | None, optional
        Matplotlib axes, if None, creates new axes, by default None
    """
    # Get axes
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(-15, 135, roll=180, vertical_axis='y')

    # Calculate camera position
    v_screen_cam_screen = -display.v_cam_screen_screen

    # Calculate camera FOV
    x = camera.image_shape_xy[0]
    y = camera.image_shape_xy[1]
    v_cam_fov_screen = (
        camera.vector_from_pixel(Vxy(([0, 0, x, x, 0], [0, y, y, 0, 0]))).as_Vxyz()
        * length_z_axis_cam
    )
    v_cam_fov_screen.rotate_in_place(display.r_cam_screen)
    v_cam_fov_screen += v_screen_cam_screen

    # Calculate camera X/Y axes
    v_cam_x_screen = (
        Vxyz(([0, axes_length], [0, 0], [0, 0])).rotate(display.r_cam_screen)
        + v_screen_cam_screen
    )
    v_cam_y_screen = (
        Vxyz(([0, 0], [0, axes_length], [0, 0])).rotate(display.r_cam_screen)
        + v_screen_cam_screen
    )
    v_cam_z_screen = (
        Vxyz(([0, 0], [0, 0], [0, length_z_axis_cam])).rotate(display.r_cam_screen)
        + v_screen_cam_screen
    )

    # Calculate object axes
    if v_screen_object_screen is not None:
        v_obj_x_screen = (
            Vxyz(([0, axes_length], [0, 0], [0, 0])).rotate(r_object_screen)
            + v_screen_object_screen
        )
        v_obj_y_screen = (
            Vxyz(([0, 0], [0, axes_length], [0, 0])).rotate(r_object_screen)
            + v_screen_object_screen
        )
        v_obj_z_screen = (
            Vxyz(([0, 0], [0, 0], [0, axes_length])).rotate(r_object_screen)
            + v_screen_object_screen
        )

    # Calculate screen outline
    p_screen_outline = display.interp_func(Vxy(([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])))

    # Calculate center of screen
    p_screen_cent = display.interp_func(Vxy((0.5, 0.5)))

    # Define positive xyz screen axes extent
    if v_screen_object_screen is None:
        obj_x = [np.nan]
        obj_y = [np.nan]
        obj_z = [np.nan]
    else:
        obj_x = v_screen_object_screen.x
        obj_y = v_screen_object_screen.y
        obj_z = v_screen_object_screen.z
    lx1 = max(
        np.nanmax(
            np.concatenate(
                (v_screen_cam_screen.x, v_cam_fov_screen.x, p_screen_outline.x, obj_x)
            )
        ),
        min_axis_length_screen,
    )
    ly1 = max(
        np.nanmax(
            np.concatenate(
                (v_screen_cam_screen.y, v_cam_fov_screen.y, p_screen_outline.y, obj_y)
            )
        ),
        min_axis_length_screen,
    )
    lz1 = max(
        np.nanmax(
            np.concatenate(
                (v_screen_cam_screen.z, v_cam_fov_screen.z, p_screen_outline.z, obj_z)
            )
        ),
        min_axis_length_screen,
    )
    # Define negative xyz screen axes extent
    lx2 = min(
        np.nanmin(
            np.concatenate(
                (v_screen_cam_screen.x, v_cam_fov_screen.x, p_screen_outline.x, obj_x)
            )
        ),
        -min_axis_length_screen,
    )
    ly2 = min(
        np.nanmin(
            np.concatenate(
                (v_screen_cam_screen.y, v_cam_fov_screen.y, p_screen_outline.y, obj_y)
            )
        ),
        -min_axis_length_screen,
    )
    lz2 = min(
        np.nanmin(
            np.concatenate(
                (v_screen_cam_screen.z, v_cam_fov_screen.z, p_screen_outline.z, obj_z)
            )
        ),
        -min_axis_length_screen,
    )
    # Add screen axes
    x = p_screen_cent.x[0]
    y = p_screen_cent.y[0]
    z = p_screen_cent.z[0]
    # Screen X axis
    ax.plot([x, x + lx1], [y, y], [z, z], color='red')
    ax.plot([x, x + lx2], [y, y], [z, z], color='black')
    ax.text(x + lx1, y, z, 'x')
    # Screen Y axis
    ax.plot([x, x], [y, y + ly1], [z, z], color='green')
    ax.plot([x, x], [y, y + ly2], [z, z], color='black')
    ax.text(x, y + ly1, z, 'y')
    # Screen Z axis
    ax.plot([x, x], [y, y], [z, z + lz1], color='blue')
    ax.plot([x, x], [y, y], [z, z + lz2], color='black')
    ax.text(x, y, z + lz1, 'z')

    # Add screen outline
    ax.plot(*p_screen_outline.data)

    # Add camera position origin
    ax.scatter(*v_screen_cam_screen.data, color='black')
    ax.text(*v_screen_cam_screen.data.squeeze(), 'camera')

    # Add camera XYZ axes
    ax.plot(*v_cam_x_screen.data, color='red')
    ax.text(*v_cam_x_screen[1].data.squeeze(), 'x', color='blue')
    ax.plot(*v_cam_y_screen.data, color='green')
    ax.text(*v_cam_y_screen[1].data.squeeze(), 'y', color='blue')
    ax.plot(*v_cam_z_screen.data, color='blue')
    ax.text(*v_cam_z_screen[1].data.squeeze(), 'z', color='blue')

    # Add camera FOV bounding box
    ax.plot(*v_cam_fov_screen.data)

    if v_screen_object_screen is not None:
        # Add object position origin
        ax.scatter(*v_screen_object_screen.data, color='black')
        ax.text(*v_screen_object_screen.data.squeeze(), 'object')

        # Add object XYZ axes
        ax.plot(*v_obj_x_screen.data, color='red')
        ax.text(*v_obj_x_screen[1].data.squeeze(), 'x', color='blue')
        ax.plot(*v_obj_y_screen.data, color='green')
        ax.text(*v_obj_y_screen[1].data.squeeze(), 'y', color='blue')
        ax.plot(*v_obj_z_screen.data, color='blue')
        ax.text(*v_obj_z_screen[1].data.squeeze(), 'z', color='blue')

    # Format and show
    plt.title('SOFAST Physical Setup\n(Screen Coordinates)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    plt.axis('equal')
