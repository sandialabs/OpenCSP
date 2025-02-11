import numbers

import numpy as np

import opencsp.common.lib.geometry.Pxy as p2
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.LoopXY import LoopXY
import opencsp.common.lib.tool.log_tools as lt


class RectXY(RegionXY):
    """
    Class representing a rectangle. This class adds convenience methods for
    accessing coordinates and applying offsets/scaling.
    """

    def __init__(self, tl: p2.Pxy, br: p2.Pxy):
        """
        Class representing a rectangle.

        Parameters
        ----------
        tl : p2.Pxy
            The top-left coordinate of this rectangle.
        br : p2.Pxy
            The bottom-left coordinate of this rectangle.
        """
        self.tl = tl
        self.br = br
        self.tr = p2.Pxy([br.x, tl.y])
        self.bl = p2.Pxy([tl.x, br.y])
        super().__init__(LoopXY.from_rectangle(self.left, self.top, self.width, self.height))

    @property
    def left(self) -> float:
        return self.tl.x[0]

    @property
    def right(self) -> float:
        return self.br.x[0]

    @property
    def top(self) -> float:
        return self.tl.y[0]

    @property
    def bottom(self) -> float:
        return self.br.y[0]

    @property
    def width(self) -> float:
        return abs(self.right - self.left)

    @property
    def height(self) -> float:
        return abs(self.top - self.bottom)

    def __repr__(self) -> str:
        return f"Rect<L{self.left}, R:{self.right}, T:{self.top}, B:{self.bottom}>"

    def __add__(self, other):
        if isinstance(other, p2.Pxy) or isinstance(other, np.ndarray):
            tl = self.tl + other
            br = self.br + other
            return RectXY(tl, br)

        else:
            lt.error_and_raise(ValueError, "Error in Rect.__add__():" + f"Trying to add {type(other)} to {type(self)}")

    def __sub__(self, other):
        if isinstance(other, p2.Pxy) or isinstance(other, np.ndarray):
            tl = self.tl - other
            br = self.br - other
            return RectXY(tl, br)

        else:
            lt.error_and_raise(ValueError, "Error in Rect.__sub__():" + f"Trying to add {type(other)} to {type(self)}")

    def __mul__(self, other):
        if isinstance(other, np.number) or isinstance(other, numbers.Number):
            tl = self.tl * other
            br = self.br * other
            return RectXY(tl, br)

        elif isinstance(other, np.ndarray) and other.size > 1:
            tl = self.tl * other
            br = self.br * other
            return RectXY(tl, br)

        else:
            lt.error_and_raise(ValueError, "Error in Rect.__sub__():" + f"Trying to add {type(other)} to {type(self)}")

    def __div__(self, other):
        if isinstance(other, np.number) or isinstance(other, numbers.Number):
            tl = self.tl / other
            br = self.br / other
            return RectXY(tl, br)

        elif isinstance(other, np.ndarray) and other.size > 1:
            tl = self.tl / other
            br = self.br / other
            return RectXY(tl, br)

        else:
            lt.error_and_raise(ValueError, "Error in Rect.__sub__():" + f"Trying to add {type(other)} to {type(self)}")
