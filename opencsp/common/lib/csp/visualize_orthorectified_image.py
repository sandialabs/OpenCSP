"""Library that handles plotting orthorectified images (slope images, etc.)"""

import matplotlib.pyplot as plt
import numpy as np


def add_quivers(
    im_x: np.ndarray,
    im_y: np.ndarray,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    quiver_density: float,
    axis: plt.Axes | None = None,
    scale: float | None = None,
    color: str = "white",
) -> None:
    """
    Adds quiver arrows to data plot.

    Parameters
    ----------
    im_x/im_y : ndarray
        Images to sample x/y quiver directions from.
    x_vec/y_vec : ndarray
        X and Y data grid axes, meters.
    quiver_density : float
        Spacing of quiver arrows in meters.
    axis : [plt.Axes | None], optional
        Axes to plot on. The default is None. If None, uses plt.gca().
    scale : [float | None], optional
        Matplotlib "scale" for adding quiver arrows. The default is None.
        If None, uses the default scale.
    color : str
        Color of the quiver arrows.
    """
    if axis is None:
        axis = plt.gca()

    # Calculate quiver points
    res_x = np.mean(np.abs(np.diff(x_vec)))
    res_y = np.mean(np.abs(np.diff(y_vec)))
    Nx = int(quiver_density / res_x)
    Ny = int(quiver_density / res_y)
    x1 = int(Nx / 2)
    y1 = int(Ny / 2)

    x_locs, y_locs = np.meshgrid(x_vec[x1::Nx], y_vec[y1::Ny])
    u_dirs = -im_x[y1::Ny, x1::Nx]
    v_dirs = -im_y[y1::Ny, x1::Nx]

    # Add quiver arrows to axes
    axis.quiver(x_locs, y_locs, u_dirs, v_dirs, color=color, scale=scale, scale_units="x")


def plot_orthorectified_image(
    image: np.ndarray,
    axis: plt.Axes,
    cmap: str,
    extent: tuple[float, float, float, float],
    clims: tuple[float, float],
    cmap_title: str,
):
    """Plots orthorectified image on axes

    Parameters
    ----------
    image : np.ndarray
        2d image to plot
    axis : plt.Axes
        Matplotlib axis to plot on
    cmap : str
        Color map
    extent : tuple[float, float, float, float]
        Left, right, bottom, top
    clims : tuple[float, float]
        Color bar limits [low, high]
    cmap_title : str
        Title of colorbar
    """
    plt_im = axis.imshow(image, cmap, origin="lower", extent=extent)
    plt_im.set_clim(clims)
    plt_cmap = plt.colorbar(plt_im, ax=axis)
    plt_cmap.ax.set_ylabel(cmap_title, rotation=270, labelpad=15)
