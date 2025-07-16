import copy
from typing import Callable, Iterable

import numpy as np
import sympy

from opencsp.common.lib.geometry.FunctionXYAbstract import FunctionXYAbstract
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.render.View3d import View3d
import opencsp.common.lib.render.lib.Drawable as dw


class FunctionXYContinuous(FunctionXYAbstract, dw.Drawable):
    """Wrapper for function that can be pickled. \n
    Extends the FunctionXYAbstract class.

    Constructor
    -----------
    FunctionXYContinuous(funcXY_string: str, domain: list[RegionXY])

    Parameters
    ----------
    funcXY_string: str
        String that contains the function that is supposed to be represented. \n
        `Note:` The function must use "x" and "y" as the vareiables.
    """

    def __init__(self, funcXY_string: str, domain: list[RegionXY] = None) -> None:
        super().__init__()
        # define the funciton's domain
        if domain == None:
            self.domain = []
        else:
            self.domain = domain

        # store the string representation of the function
        self.func_string = funcXY_string

        # store the lambda representaiton of the function
        self.func = lambda x, y: eval(self.func_string, None, {"x": x, "y": y})

    # def interpolate() -> FunctionXYAnalytic:
    #     ...

    def __repr__(self) -> str:
        return self.func_string

    # override
    def value_at(self, x: float, y: float) -> float:
        return self.func(x, y)

    # override
    def __getstate__(self) -> dict:
        d = copy.deepcopy(self.__dict__)
        d["func"] = None
        return d

    # override
    def __setstate__(self, d: dict):
        self.__dict__ = d
        self.func = lambda x, y: eval(self.func_string, None, {"x": x, "y": y})

    # override
    def in_domain(self, x: float, y: float) -> bool:
        raise NotImplementedError

    def draw(self, view: View3d):
        if view.view_spec["type"] == "image":
            # X, Y = np.meshgrid(self.x_domain, self.y_domain)
            arr = np.zeros((len(self.y_domain), len(self.x_domain)))
            for ix, x in enumerate(sorted(self.x_domain)):
                for iy, y in enumerate(sorted(self.y_domain)):
                    arr[iy, ix] = self.value_at(x, y)
            # A = self.as_callable()(X,Y)
            extent = [min(self.x_domain), max(self.x_domain), min(self.y_domain), max(self.y_domain)]
            # view.pcolormesh(list(self.x_domain), list(self.y_domain), arr, colorbar=True, cmap='jet', )
            view.imshow(arr, colorbar=True, cmap="jet")

    @classmethod
    def from_array(cls, x_domain: np.ndarray, y_domain: np.ndarray, values: np.ndarray):
        """
        Create an instance of FunctionXYContinuous using a 2d array

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
            return FunctionXYContinuous(d)
