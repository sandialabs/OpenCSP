import copy
from abc import ABC, abstractmethod

from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Pxyz import Pxyz


class FunctionXYAbstract(ABC):
    """Abstract class for a funciton of two variables

    Abstract Methods
    ----------------

    .. code-block:: python

        value_at(self, x: float, y: float) -> float:
            # "Returns the value of the function at the given x and y values."
        __getstate__(self):
            # "Returns a serializable object for pickleing."
        __setstate__(self, d):
            # "Takes in __getstate__(self)'s output to recreate the object `self` that was passed into __getstate__"
        in_domain(x, y) -> bool:
            # "Returns True if the (x,y) pair is in the domain, otherwise False."

    """

    def __init__(self):
        pass

    def __call__(self, x: float, y: float) -> float:
        return self.value_at(x, y)

    @abstractmethod
    def value_at(self, x: float, y: float) -> float:
        """Returns the value of the function at the given x and y values."""
        raise NotImplementedError

    @abstractmethod
    def __getstate__(self) -> dict:
        """Returns a serializable object for pickleing."""
        return super().__getstate__()

    @abstractmethod
    def __setstate__(self, state: dict):
        """Takes in __getstate__(self)'s output to recreate the object `self` that was passed into __getstate__"""
        return super().__setstate__()

    @abstractmethod
    def in_domain(self, x: float, y: float) -> bool:
        """Returns True if the (x,y) pair is in the domain, otherwise False."""
        raise NotImplementedError

    def lift(self, p: Pxy) -> Pxyz:
        """Takes in a Pxy point and returns the Pxyz point where the z values are the values
        that correspond to the functions results for value_at(p.x, p.y)"""
        z = self.value_at(p.x, p.y)
        xs, ys, zs = map(copy.deepcopy, (p.x, p.y, z))
        return Pxyz([xs, ys, zs])
