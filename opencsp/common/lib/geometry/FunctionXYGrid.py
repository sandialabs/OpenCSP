from typing import Iterable

import numpy as np

from opencsp.common.lib.geometry.FunctionXYAbstract import FunctionXYAbstract
from opencsp.common.lib.render.View3d import View3d
import opencsp.common.lib.render_control.RenderControlFunctionXY as rcfxy


class FunctionXYGrid(FunctionXYAbstract):
    """Discrete Fuction defined by a grid.
    This object uses x:column and y:row.
    When accessing, this looks like column major.

    Parameters
    ----------
    values: numpy array
        2 dimensional
    limits: 4-tuple
        in the form (smallest x, largest x, smallest y, largest y)
    """

    def __init__(self, values: np.ndarray, limits: tuple[float, float, float, float] = None) -> None:
        """Represents a discrete function of equispaced points in its domain.
        Defined by an array and the location of the 4 corners
        of that array in the funciton space.

        Parameters
        ----------
        values (np.ndarray):
            2d array representing the values of the function.
        limits (tuple[float, float, float, float], optional): Defaults to None.

        """
        self.values = values
        self.x_count = len(values[0]) + 0.0
        self.y_count = len(values) + 0.0
        if limits == None:  # default is to treat the array just as an array
            limits = (0.0, self.x_count, 0.0, self.y_count)
        # if we know the first and last x & y values in the domain
        # and we assume all values are equispaced, this gives all
        # the information needed
        self.x0, self.x1, self.y0, self.y1 = limits
        self.x_step = (self.x1 - self.x0) / (self.x_count - 1)
        self.y_step = (self.y1 - self.y0) / (self.y_count - 1)

    # override
    def value_at(self, x: float | Iterable[float], y: float | Iterable[float]) -> float | np.ndarray[float]:
        # array case
        if isinstance(x, Iterable) or isinstance(y, Iterable):
            if len(x) != len(y):
                raise ValueError(
                    f"The length of x and y must be the same. x length {len(x)} does not match y length {len(y)}"
                )
            return np.array([self.value_at(xi, yi) for xi, yi in zip(x, y)])
        # scalar case
        if self.in_domain(x, y):
            x_index, y_index = self.to_index_values(x, y)
            return self.values[y_index, x_index]
        else:
            raise IndexError(f"({x}, {y}) pair not within domain of grid function.")

    # override
    def in_domain(self, x: float, y: float) -> bool:
        """Takes in a pair of elements in the form (x:float, y:float) and returns true if the pair is in the domain of self."""
        if (not self.x0 <= x <= self.x1) or not (self.y0 <= y <= self.y1):
            return False
        x_index, y_index = self.to_index_values(x, y)
        if int(x_index) != x_index or int(y_index) != y_index:
            return False
        return True

    # override
    def __getstate__(self) -> dict:
        """Returns a serializable object for pickleing."""
        raise NotImplementedError("__getstate__ has not been implemented for FunctionXYGrid")

    # override
    def __setstate__(self, state: dict):
        """Takes in __getstate__(self)'s output to recreate the object `self` that was passed into __getstate__"""
        raise NotImplementedError("__setstate__ has not been implemented for FunctionXYGrid")

    def to_index_values(self, x: float, y: float) -> tuple[int, int]:
        x_index = (x - self.x0) / self.x_step
        y_index = (y - self.y0) / self.y_step
        if int(x_index) != x_index or int(y_index) != y_index:
            return False
        return (int(x_index), int(y_index))

    def draw(self, view: View3d, functionXY_style: rcfxy.RenderControlFunctionXY = None):
        if functionXY_style == None:
            functionXY_style = rcfxy.RenderControlFunctionXY()

        # extent
        extent_left = self.x0 - self.x_step
        extent_right = self.x1 + self.x_step
        extent_low = self.y0 - self.y_step
        extent_high = self.y1 + self.y_step
        extent = (extent_left, extent_right, extent_low, extent_high)

        # color values
        if functionXY_style.colorbar_min_max == None:
            vmin = min(map(min, self.values))
            vmax = max(map(max, self.values))
        else:
            vmin, vmax = functionXY_style.colorbar_min_max

        # draw
        if functionXY_style.draw_heatmap:
            view.imshow(
                self.values,
                colorbar=functionXY_style.colorbar,
                vmin=vmin,
                vmax=vmax,
                cmap=functionXY_style.cmap,
                extent=extent,
                # norm=colors.LogNorm()
            )

        if functionXY_style.draw_contours:
            view.contour(
                np.flipud(self.values),
                colorbar=functionXY_style.colorbar,
                vmin=vmin,
                levels=3,  # TODO TJL:add to render control
                vmax=vmax,
                colors="black",  # TODO TJL:add to render control
                #  cmap=functionXY_style.cmap,
                extent=extent,
            )
