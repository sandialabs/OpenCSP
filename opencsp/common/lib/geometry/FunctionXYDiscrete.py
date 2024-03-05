from typing import Callable, Iterable

import numpy as np
import sympy

from opencsp.common.lib.geometry.FunctionXYAbstract import \
    FunctionXYAbstract
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.render.View3d import View3d


class FunctionXYDiscrete(FunctionXYAbstract):
    def __init__(self, values: dict[tuple[float, float], float]) -> None:
        super().__init__()
        self.values = values
        x_domain, y_domain = tuple(zip(*values.keys()))
        self.x_domain = set(x_domain)
        self.y_domain = set(y_domain)
        self.domain = list(values.keys())

    # def __add__(self, f2:'FunctionXYDiscrete') -> "FunctionXYDiscrete":
    #     sum = self.values + f2.values
    #     return FunctionXYDiscrete(sum)

    # def interpolate() -> FunctionXYAnalytic:
    #     ...

    def value_at(self, x: float | Iterable[float], y: float | Iterable[float]) -> float | np.ndarray[float]:
        if isinstance(x, Iterable) and isinstance(y, Iterable):
            if len(x) != len(y):
                raise ValueError(f"The length of x and y must be the same. x length {len(x)} does not match y length {len(y)}")
            return np.array([self.value_at(xi, yi) for xi, yi in zip(x, y)])
        else:
            if self.in_domain(x, y):
                return self.values[x, y]
            else:
                raise ValueError("(x,y) pair not within domain")

    def in_domain(self, x: float, y: float) -> bool:
        """Takes in a pair of elements in the form (x:float, y:float) and returns true if the pair is in the domain of self."""
        return (x, y) in list(self.values.keys())

    def draw(self, view: View3d, functionXY_style):
        if view.view_spec['type'] == 'image':
            # X, Y = np.meshgrid(self.x_domain, self.y_domain)
            arr = np.zeros((len(self.y_domain), len(self.x_domain)))
            for ix, x in enumerate(sorted(self.x_domain)):
                for iy, y in enumerate(sorted(self.y_domain)):
                    arr[iy, ix] = self.value_at(x, y)
            # A = self.as_callable()(X,Y)
            extent = [min(self.x_domain), max(self.x_domain), min(self.y_domain), max(self.y_domain)]
            # view.pcolormesh(list(self.x_domain), list(self.y_domain), arr, colorbar=True, cmap='jet', )
            view.imshow(arr, colorbar=True, cmap='jet')

    @classmethod
    def from_array(cls, x_domain: np.ndarray, y_domain: np.ndarray, values: np.ndarray):
        """
        Create an instance of FunctionXYDiscrete using a 2d array

        Parameters
        -----------
        x_domain: array, the values of x that will be used to access values of the array
        y_domain: array, the values of y that will be used to access values of the array
        values: 2d array, the values that will be returned when x and y are used
        """
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
