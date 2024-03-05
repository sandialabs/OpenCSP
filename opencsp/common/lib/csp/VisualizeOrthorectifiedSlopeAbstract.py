"""Abstract class used for visualizing orthorectified slope looking
down from +z axis
"""
from abc import abstractmethod
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


class VisualizeOrthorectifiedSlopeAbstract:
    """Abstract class inherited by all objects which can have orthorectified slope
    visualization.
    """

    @abstractmethod
    def orthorectified_slope_array(
        self, x_vec: np.ndarray, y_vec: np.ndarray
    ) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        pass

    def plot_orthorectified_slope_error(
        self,
        reference: 'VisualizeOrthorectifiedSlopeAbstract',
        res: float = 0.1,
        type_: Literal['x', 'y', 'magnitude'] = 'magnitude',
        clim: float | None = None,
        axis: plt.Axes | None = None,
        quiver_density: float | None = None,
        quiver_scale: float | None = 10,
        quiver_color: str = 'white',
    ) -> None:
        """Plots slope difference with respect to a reference
        mirror on axes. Error defined as (self - reference).

        Parameters
        ----------
        reference : VisualizeOrthorectifiedSlopeAbstract
            CSP optic object supporting VisualizeOrthorectifiedSlopeAbstract
        res : float, optional
            The xy resolution of the plot, meters, by default 0.1
        type_ : str
            Type of slope image to generate - 'x', 'y', 'magnitude'
        clim : float | None
            Colorbar limit. Converts to [-clim, clim] for type 'x' and 'y'
            and [0, clim] for type 'magnitude.' Units in mrad. None to use default.
        axis : plt.Axes | None
            Axes to plot on. Default is None. If None, uses plt.gca().
        quiver_density : bool | None
            Spacing of quiver arrows in meters, None for no arrows.
        quiver_scale : float | None
            Scale of quiver arrows, None for default.
        quiver_color : str
            Color of quiver arrows.
        """
        # Check inputs
        if type_ not in ['x', 'y', 'magnitude']:
            raise ValueError(f'Given type_ {type_} not supported.')
        if (quiver_density is not None) and (res > quiver_density):
            raise ValueError('Quiver density cannot be less than image resolution.')

        # Get axes
        if axis is None:
            axis = plt.gca()

        # Create interpolation axes
        left, right, bottom, top = self.axis_aligned_bounding_box
        x_vec = np.arange(left, right, res)  # meters
        y_vec = np.arange(bottom, top, res)  # meters

        # Calculate reference mirror slope
        slopes_ref = reference.orthorectified_slope_array(x_vec, y_vec)  # radians

        # Calculate current mirror slope
        slopes_cur = self.orthorectified_slope_array(x_vec, y_vec)  # radians

        # Calculate slope difference (error)
        slopes_diff = slopes_cur - slopes_ref  # radians

        # Calculate slope image
        if type_ == 'x':
            image = slopes_diff[0] * 1000  # mrad
            x_image = image
            y_image = np.zeros(x_image.shape)
            if clim is not None:
                clims = [-clim, clim]
            else:
                clims = clim
            title = 'X Slope Error'
            cmap = 'seismic'
        elif type_ == 'y':
            image = slopes_diff[1] * 1000  # mrad
            y_image = image
            x_image = np.zeros(y_image.shape)
            if clim is not None:
                clims = [-clim, clim]
            else:
                clims = clim
            title = 'Y Slope Error'
            cmap = 'seismic'
        elif type_ == 'magnitude':
            x_image = slopes_diff[0] * 1000  # mrad
            y_image = slopes_diff[1] * 1000  # mrad
            image = np.sqrt((x_image**2 + y_image**2) / 2)  # mrad
            if clim is not None:
                clims = [0, clim]
            else:
                clims = clim
            title = 'Slope Error Magnitude'
            cmap = 'jet'

        # Plot image on axis
        extent = (left - res / 2, right + res / 2, bottom - res / 2, top + res / 2)
        self._plot_orthorectified_image(image, axis, cmap, extent, clims, 'mrad')

        # Add quiver arrows
        if quiver_density is not None:
            self._add_quivers(
                x_image,
                y_image,
                x_vec,
                y_vec,
                quiver_density,
                axis,
                quiver_scale,
                quiver_color,
            )

        # Label axes
        axis.set_title(title)

    def plot_orthorectified_slope(
        self,
        res: float = 0.1,
        type_: Literal['x', 'y', 'magnitude'] = 'magnitude',
        clim: float | None = None,
        axis: plt.Axes | None = None,
        quiver_density: float | None = None,
        quiver_scale: float | None = 50,
        quiver_color: str = 'white',
    ) -> None:
        """Plots orthorectified image of mirror slope

        Parameters
        ----------
        res : float, optional
            The xy resolution of the plot, meters, by default 0.1
        type_ : str
            Type of slope image to generate - 'x', 'y', 'magnitude'
        clim : float | None
            Colorbar limit. Converts to [-clim, clim] for type 'x' and 'y'
            and [0, clim] for type 'magnitude.' Units in mrad. None to use default.
        axis : plt.Axes | None
            Axes to plot on. Default is None. If None, uses plt.gca().
        quiver_density : bool | None
            Spacing of quiver arrows in meters, None for no arrows.
        quiver_scale : float | None
            Scale of quiver arrows, None for default.
        quiver_color : str
            Color of quiver arrows.
        """
        # Check inputs
        if type_ not in ['x', 'y', 'magnitude']:
            raise ValueError(f'Given type_ {type_} not supported.')
        if (quiver_density is not None) and (res > quiver_density):
            raise ValueError('Quiver density cannot be less than image resolution.')

        # Get axes
        if axis is None:
            axis = plt.gca()

        # Create interpolation axes
        left, right, bottom, top = self.axis_aligned_bounding_box
        x_vec = np.arange(left, right, res)  # meters
        y_vec = np.arange(bottom, top, res)  # meters

        # Calculate slope image
        slopes = self.orthorectified_slope_array(x_vec, y_vec)

        # Calculate slope image
        if type_ == 'x':
            image = slopes[0] * 1000  # mrad
            x_image = image
            y_image = np.zeros(x_image.shape)
            if clim is not None:
                clims = [-clim, clim]
            else:
                clims = clim
            title = 'X Slope'
        elif type_ == 'y':
            image = slopes[1] * 1000  # mrad
            y_image = image
            x_image = np.zeros(y_image.shape)
            if clim is not None:
                clims = [-clim, clim]
            else:
                clims = clim
            title = 'Y Slope'
        elif type_ == 'magnitude':
            x_image = slopes[0] * 1000  # mrad
            y_image = slopes[1] * 1000  # mrad
            image = np.sqrt(x_image**2 + y_image**2)  # mrad
            if clim is not None:
                clims = [0, clim]
            else:
                clims = clim
            title = 'Slope Magnitude'

        # Plot image on axes
        extent = (left - res / 2, right + res / 2, bottom - res / 2, top + res / 2)
        self._plot_orthorectified_image(image, axis, 'jet', extent, clims, 'mrad')

        # Add quiver arrows
        if quiver_density is not None:
            self._add_quivers(
                x_image,
                y_image,
                x_vec,
                y_vec,
                quiver_density,
                axis,
                quiver_scale,
                quiver_color,
            )

        # Label axes
        axis.set_title(title)

    def plot_orthorectified_curvature(
        self,
        res: float = 0.1,
        type_: Literal['x', 'y', 'combined'] = 'combined',
        clim: float | None = None,
        axis: plt.Axes | None = None,
    ):
        """Plots orthorectified curvature (1st derivative of slope) image
        on axes.

        Parameters
        ----------
        res : float, optional
            The xy resolution of the plot, meters, by default 0.1
        type_ : str
            Type of slope image to generate - 'x', 'y', 'combined'
        clim : float | None
            Colorbar limit. Converts to [-clim, clim] for type 'x' and 'y'
            and [0, clim] for type 'combined.' Units in mrad. None to use default.
        axis : plt.Axes | None
            Axes to plot on. Default is None. If None, uses plt.gca().
        """
        # Check inputs
        if type_ not in ['x', 'y', 'combined']:
            raise ValueError(f'Given type_ {type_} not supported.')

        # Get axes
        if axis is None:
            axis = plt.gca()

        # Create interpolation axes
        left, right, bottom, top = self.axis_aligned_bounding_box
        x_vec = np.arange(left, right, res)  # meters
        y_vec = np.arange(bottom, top, res)  # meters

        # Calculate slope image
        slopes = self.orthorectified_slope_array(x_vec, y_vec)  # slope

        # Calculate curvature image
        x_del_vec = np.diff(x_vec)  # meter
        y_del_vec = np.diff(y_vec)  # meter

        # Define clims
        if clim is not None:
            clims = [-clim, clim]
        else:
            clims = clim

        # Calculate slope image
        if type_ in ['x', 'combined']:
            image_x = np.diff(slopes[0] * 1000, axis=1)  # mrad / sample
            image_x /= x_del_vec[None, :]  # mrad / meter
            image = image_x
            title = 'X Curvature'
            extent = (left, right, bottom - res / 2, top + res / 2)

        if type_ in ['y', 'combined']:
            image_y = np.diff(slopes[1] * 1000, axis=0)  # mrad / sample
            image_y /= y_del_vec[:, None]  # mrad / meter
            image = image_y
            title = 'Y Curvature'
            extent = (left - res / 2, right + res / 2, bottom, top)

        if type_ == 'combined':
            image = (image_x[1:, :] + image_y[:, 1:]) / 2  # mrad / meter
            title = 'Combined Curvature'
            extent = (left, right, bottom, top)

        # Plot image on axes
        self._plot_orthorectified_image(
            image, axis, 'seismic', extent, clims, 'mrad/meter'
        )

        # Label axes
        axis.set_title(title)

    def _add_quivers(
        self,
        im_x: np.ndarray,
        im_y: np.ndarray,
        x_vec: np.ndarray,
        y_vec: np.ndarray,
        quiver_density: float,
        axis: plt.Axes | None = None,
        scale: float | None = None,
        color: str = 'white',
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
        axis.quiver(
            x_locs, y_locs, u_dirs, v_dirs, color=color, scale=scale, scale_units='x'
        )

    def _plot_orthorectified_image(
        self,
        image: np.ndarray,
        axis: plt.Axes,
        cmap: str,
        extent: tuple[float, float, float, float],
        clims: tuple[float, float],
        cmap_title: str,
    ):
        """Plots orthorectified image on axes"""
        plt_im = axis.imshow(image, cmap, origin='lower', extent=extent)
        plt_im.set_clim(clims)
        plt_cmap = plt.colorbar(plt_im, ax=axis)
        plt_cmap.ax.set_ylabel(cmap_title, rotation=270, labelpad=15)
