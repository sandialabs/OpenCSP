from typing import Iterable

import numpy as np

from opencsp.common.lib.geometry.FunctionXYAbstract import FunctionXYAbstract
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.render.View3d import View3d


class FunctionXYDiscrete(FunctionXYAbstract):
    """
    A class representing a discrete function defined by scattered (x, y) points and their corresponding values.

    This class allows for the evaluation of function values at specified (x, y) coordinates
    and provides methods for checking if points are within the function's domain.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, values: dict[tuple[float, float], float]) -> None:
        """
        Initializes a FunctionXYDiscrete object with the specified values.

        Parameters
        ----------
        values : dict[tuple[float, float], float]
            A dictionary mapping (x, y) coordinate pairs to their corresponding function values.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        super().__init__()
        self.values = values
        x_domain, y_domain = tuple(zip(*values.keys()))
        self.x_domain = set(x_domain)
        self.y_domain = set(y_domain)
        self.domain = list(values.keys())

    def value_at(self, x: float | Iterable[float], y: float | Iterable[float]) -> float | np.ndarray[float]:
        """
        Retrieves the function value at the specified (x, y) coordinates.

        Parameters
        ----------
        x : float or Iterable[float]
            The x-coordinate(s) at which to evaluate the function.
        y : float or Iterable[float]
            The y-coordinate(s) at which to evaluate the function.

        Returns
        -------
        float or np.ndarray
            The function value(s) at the specified coordinates.

        Raises
        ------
        ValueError
            If the (x, y) pair is not within the domain of the function or if the lengths of x and y do not match.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if isinstance(x, Iterable) and isinstance(y, Iterable):
            if len(x) != len(y):
                raise ValueError(
                    f"The length of x and y must be the same. x length {len(x)} does not match y length {len(y)}"
                )
            return np.array([self.value_at(xi, yi) for xi, yi in zip(x, y)])
        else:
            if self.in_domain(x, y):
                return self.values[x, y]
            else:
                raise ValueError("(x,y) pair not within domain")

    def in_domain(self, x: float, y: float) -> bool:
        """
        Checks if the specified (x, y) coordinates are within the domain of the function.

        Parameters
        ----------
        x : float
            The x-coordinate to check.
        y : float
            The y-coordinate to check.

        Returns
        -------
        bool
            True if the (x, y) pair is within the domain, False otherwise.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return (x, y) in list(self.values.keys())

    def draw(self, view: View3d):
        """
        Draws the function in a 3D view.

        Parameters
        ----------
        view : View3d
            The 3D view in which to draw the function.

        Returns
        -------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if view.view_spec["type"] == "image":
            arr = np.zeros((len(self.y_domain), len(self.x_domain)))
            for ix, x in enumerate(sorted(self.x_domain)):
                for iy, y in enumerate(sorted(self.y_domain)):
                    arr[iy, ix] = self.value_at(x, y)
            view.imshow(arr, colorbar=True, cmap="jet")

    @classmethod
    def from_array(cls, x_domain: np.ndarray, y_domain: np.ndarray, values: np.ndarray):
        """
        Creates an instance of FunctionXYDiscrete using a 2D array.

        Parameters
        ----------
        x_domain : np.ndarray
            The values of x that will be used to access values of the array.
        y_domain : np.ndarray
            The values of y that will be used to access values of the array.
        values : np.ndarray
            A 2D array containing the values that will be returned when x and y are used.

        Returns
        -------
        FunctionXYDiscrete
            An instance of FunctionXYDiscrete initialized with the provided domains and values.

        Raises
        ------
        ValueError
            If the size of the domain does not match the size of the value array.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if len(values) != len(y_domain) or len(values[0]) != len(x_domain):
            raise ValueError("Size of the domain does not match size of the value array.")
        else:
            d = dict()
            for iy, y in enumerate(y_domain):
                for ix, x in enumerate(x_domain):
                    d[x, y] = values[iy, ix]
            return FunctionXYDiscrete(d)

    # override
    def __getstate__(self) -> tuple[str, list[RegionXY]]:
        return super().__getstate__()

    # override
    def __setstate__(self, state: tuple[str, list[RegionXY]]):
        return super().__setstate__(state)
