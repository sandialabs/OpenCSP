"""
3-d Axis Management



"""

import numpy as np

# from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D


# Set Axes Equal in 3d
def set_3d_axes_equal(ax: Axes3D, set_zmin_zero=False, box_aspect: None | tuple[int, int, int] = (1, 1, 1)):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
      set_zmin_zero: If true, then set the z axis lower limit to 0.
      box_aspect: If none, do nothing (use the standard 4:4:3 aspect). Otherwise, this should be a 3 tuple. For example, (1:1:1)

    Link: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """
    # Fetch limits.
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Check input.
    if set_zmin_zero and (z_limits[0] < 0):
        print("WARNING: Encountered negative values when attempting to set axis z limits relative to zero.")

    # Set z interval.
    if set_zmin_zero:
        z_limits[0] = 0

    # Find value range and mdpoint for each axis.
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    if set_zmin_zero:
        z_range = abs(z_limits[1] - z_limits[0])
    else:
        z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    if set_zmin_zero:
        ax.set_zlim3d([0, (2 * plot_radius)])
    else:
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    if box_aspect != None:
        ax.set_box_aspect(box_aspect)
