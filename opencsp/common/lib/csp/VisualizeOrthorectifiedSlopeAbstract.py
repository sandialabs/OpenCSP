"""Abstract class used for visualizing orthorectified slope looking
down from +z axis
"""

from abc import abstractmethod
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from opencsp.common.lib.csp.visualize_orthorectified_image import add_quivers, plot_orthorectified_image


class VisualizeOrthorectifiedSlopeAbstract:
    """Abstract class inherited by all objects which can have orthorectified slope
    visualization.
    """

    @abstractmethod
    def orthorectified_slope_array(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """Returns slope array at the given interpolation vectors.

        See also :py:meth:`get_orthorectified_slope_array`

        Parameters
        ----------
        x_vec : ndarray
            X interpolation vector of length n.
        y_vec : ndarray
            Y interpolation vector of length m.

        Returns
        -------
        slope_array : ndarray
            Shape (2, n, m) array of x/y slopes
        """
        pass

    @property
    @abstractmethod
    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        pass

    def get_orthorectified_slope_array(self, res: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns slope array given resolution

        Parameters
        ----------
        res : float, optional
            The xy resolution of the plot, meters, by default 0.1

        Returns
        -------
        slope_array : ndarray
            Shape (2, n, m) array of x/y slopes
        x_vec : ndarray
            X interpolation vector
        y_vec : ndarray
            Y interpolation vector
        """
        # Create interpolation axes
        left, right, bottom, top = self.axis_aligned_bounding_box
        x_vec = np.arange(left, right, res)  # meters
        y_vec = np.arange(bottom, top, res)  # meters

        # Calculate slope image
        return self.orthorectified_slope_array(x_vec, y_vec), x_vec, y_vec

    def plot_orthorectified_slope_error(
        self,
        reference: "VisualizeOrthorectifiedSlopeAbstract",
        res: float = 0.1,
        type_: Literal["x", "y", "magnitude"] = "magnitude",
        clim: float | None = None,
        axis: plt.Axes | None = None,
        quiver_density: float | None = None,
        quiver_scale: float | None = 10,
        quiver_color: str = "white",
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
        if type_ not in ["x", "y", "magnitude"]:
            raise ValueError(f"Given type_ {type_} not supported.")
        if (quiver_density is not None) and (res > quiver_density):
            raise ValueError("Quiver density cannot be less than image resolution.")

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
        if type_ == "x":
            image = slopes_diff[0] * 1000  # mrad
            x_image = image
            y_image = np.zeros(x_image.shape)
            if clim is not None:
                clims = [-clim, clim]
            else:
                clims = clim
            title = "X Slope Error"
            cmap = "seismic"
        elif type_ == "y":
            image = slopes_diff[1] * 1000  # mrad
            y_image = image
            x_image = np.zeros(y_image.shape)
            if clim is not None:
                clims = [-clim, clim]
            else:
                clims = clim
            title = "Y Slope Error"
            cmap = "seismic"
        elif type_ == "magnitude":
            x_image = slopes_diff[0] * 1000  # mrad
            y_image = slopes_diff[1] * 1000  # mrad
            image = np.sqrt((x_image**2 + y_image**2) / 2)  # mrad
            if clim is not None:
                clims = [0, clim]
            else:
                clims = clim
            title = "Slope Error Magnitude"
            cmap = "jet"

        # Plot image on axis
        extent = (left - res / 2, right + res / 2, bottom - res / 2, top + res / 2)
        plot_orthorectified_image(image, axis, cmap, extent, clims, "mrad")

        # Add quiver arrows
        if quiver_density is not None:
            add_quivers(x_image, y_image, x_vec, y_vec, quiver_density, axis, quiver_scale, quiver_color)

        # Label axes
        axis.set_title(title)

    def plot_orthorectified_slope(
        self,
        res: float = 0.1,
        type_: Literal["x", "y", "magnitude"] = "magnitude",
        clim: float | None = None,
        axis: plt.Axes | None = None,
        quiver_density: float | None = None,
        quiver_scale: float | None = 50,
        quiver_color: str = "white",
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
        if type_ not in ["x", "y", "magnitude"]:
            raise ValueError(f"Given type_ {type_} not supported.")
        if (quiver_density is not None) and (res > quiver_density):
            raise ValueError("Quiver density cannot be less than image resolution.")

        # Get axes
        if axis is None:
            axis = plt.gca()

        # Calculate slope image
        slopes, x_vec, y_vec = self.get_orthorectified_slope_array(res)

        # Calculate slope image
        if type_ == "x":
            image = slopes[0] * 1000  # mrad
            x_image = image
            y_image = np.zeros(x_image.shape)
            if clim is not None:
                clims = [-clim, clim]
            else:
                clims = clim
            title = "X Slope"
        elif type_ == "y":
            image = slopes[1] * 1000  # mrad
            y_image = image
            x_image = np.zeros(y_image.shape)
            if clim is not None:
                clims = [-clim, clim]
            else:
                clims = clim
            title = "Y Slope"
        elif type_ == "magnitude":
            x_image = slopes[0] * 1000  # mrad
            y_image = slopes[1] * 1000  # mrad
            image = np.sqrt(x_image**2 + y_image**2)  # mrad
            if clim is not None:
                clims = [0, clim]
            else:
                clims = clim
            title = "Slope Magnitude"

        # Plot image on axes
        left, right, bottom, top = self.axis_aligned_bounding_box
        extent = (left - res / 2, right + res / 2, bottom - res / 2, top + res / 2)
        plot_orthorectified_image(image, axis, "jet", extent, clims, "mrad")

        # Add quiver arrows
        if quiver_density is not None:
            add_quivers(x_image, y_image, x_vec, y_vec, quiver_density, axis, quiver_scale, quiver_color)

        # Label axes
        axis.set_title(title)

    def plot_orthorectified_curvature(
        self,
        res: float = 0.1,
        type_: Literal["x", "y", "combined"] = "combined",
        clim: float | None = None,
        axis: plt.Axes | None = None,
        processing: list[Literal["log", "smooth"]] = None,
        smooth_kernel_width: int = 1,
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
        processing : list[str]
            Apply processing steps to curvature plot.
            -Log: shows log of absolute value of image
            -Smooth: Smooths the image with a rectangular kernel of given width.
        smooth_kernel_width : int
            Dimension of kernel used to smooth slope image before creating curvature plot.
            By default, 1.
        """
        # Check inputs
        if type_ not in ["x", "y", "combined"]:
            raise ValueError(f"Given type_ {type_} not supported.")
        if processing is None:
            processing = []

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
            if isinstance(clim, (list, tuple)):
                clims = clim
            else:
                clims = [-clim, clim]
        else:
            clims = clim

        # Calculate slope image
        if type_ in ["x", "combined"]:
            image_x = np.diff(slopes[0] * 1000, axis=1)  # mrad / sample
            image_x /= x_del_vec[None, :]  # mrad / meter
            image = image_x
            title = "X Curvature"
            extent = (left, right, bottom - res / 2, top + res / 2)

        if type_ in ["y", "combined"]:
            image_y = np.diff(slopes[1] * 1000, axis=0)  # mrad / sample
            image_y /= y_del_vec[:, None]  # mrad / meter
            image = image_y
            title = "Y Curvature"
            extent = (left - res / 2, right + res / 2, bottom, top)

        if type_ == "combined":
            image = (image_x[1:, :] + image_y[:, 1:]) / 2  # mrad / meter
            title = "Combined Curvature"
            extent = (left, right, bottom, top)

        # Apply processing steps
        for proc in processing:
            if proc == "log":
                # Take log of image
                image = np.abs(image)
                image[image == 0] = np.nan
                image = np.log(image)
                image[np.isinf(image)] = np.nan
            elif proc == "smooth":
                # Smooth images
                ker = np.ones((smooth_kernel_width, smooth_kernel_width)) / smooth_kernel_width**2
                image = convolve2d(image, ker, mode="same", boundary="symm")

        # Plot image on axes
        plot_orthorectified_image(image, axis, "seismic", extent, clims, "mrad/meter")

        # Label axes
        axis.set_title(title)
