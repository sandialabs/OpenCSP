import copy
import numbers

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Pxy import Pxy
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt


class RegionXY:
    """Representation of a 2D region"""

    def __init__(self, loop: LoopXY):
        """
        Instantiates a 2d region with a 2d loop

        Parameters
        ----------
        loop : LoopXY
            The first loop to define the region.

        NOTE: Currently only single-loop regions can be defined.

        """
        # Save loop in list
        self.loops = [loop]

    def add_loop(self, loop: LoopXY) -> None:
        """
        Appends a loop to the current region. Currently not implimented.

        Parameters
        ----------
        loop : LoopXY
            Loop to append to the region.

        """
        raise NotImplementedError("Cannot add more than one loop to a region currently.")

    def as_mask(self, vx: np.ndarray, vy: np.ndarray):
        """
        Returns the mask representation of a region given X and Y sample
        points.

        Parameters
        ----------
        vx/vy : np.ndarray
            The X and Y sample points.

        Returns
        -------
        np.ndarray
            Boolean 2d mask with shape (yv.size, xv.size). True for inside this
            region, False for outside this region.
        """
        # Create mask from first loop
        mask = self.loops[0].as_mask(vx, vy)

        # Update mask for remaining loops
        for loop in self.loops[1::]:
            mask = np.logical_xor(mask, loop.as_mask(vx, vy))

        return mask

    def is_inside(self, P: Vxy) -> np.ndarray:
        """
        Calculates if given points are inside the region.

        Parameters
        ----------
        P : Vxy
            Sample points

        Returns
        -------
        mask : np.ndarray
            1D array of booleans.

        """
        # Create mask from first loop
        mask = self.loops[0].is_inside(P)

        # Update mask for remaining loops
        for loop in self.loops[1::]:
            mask = np.logical_xor(mask, loop.is_inside(P))

        return mask

    def is_inside_or_on_border(self, P: Vxy, thresh: float = 1e-6) -> np.ndarray:
        """
        Calculates if given points are inside the region or
        ono the border.

        Parameters
        ----------
        P : Vxy
            Sample points

        Returns
        -------
        mask : np.ndarray
            1D array of booleans.

        """
        # Create mask from first loop
        mask = self.loops[0].is_inside_or_on_border(P, thresh)

        # Update mask for remaining loops
        for loop in self.loops[1::]:
            mask = np.logical_xor(mask, loop.is_inside_or_on_border(P, thresh))

        return mask

    def edge_sample(self, count: int):
        """Returns a Vxy of count points per edge per loop defining the region"""
        return Vxy.merge([loop.edge_sample(count) for loop in self.loops])

    def points_sample(self, resolution: "Resolution") -> Pxy:
        """Returns a Pxy object of points sampled from inside the region.

        Parameters
        ----------
        resolution : int
            Spacing between points
        resolution_type : str, optional
            {'random', 'pixelY', 'pixelX'}, by default 'pixelX'
        random_seed : int | None, optional
            _description_, by default None

        Returns
        -------
        Pxy
            _description_
        """
        resolution.resolve_in_place(self.axis_aligned_bounding_box())
        points = resolution.points
        filtered_points = self.filter_points(points)
        return filtered_points

    def draw(self, ax: plt.Axes = None, style: rcps.RenderControlPointSeq = None):
        """
        Draws all loops on given axes.

        Parameters
        ----------
        ax : plt.Axes
            Axes to draw on. If no Axes given, draws on current axes.
        style : RenderControlPointSequence, optional
            The style used to draw this region.

        """
        for loop in self.loops:
            loop.draw(ax, style)

    def axis_aligned_bounding_box(self) -> tuple[float, float, float, float]:
        """
        Gives the minnimum bounding envelope for the region. The minnimum
        bounding envelope is the smallest rectangle that can fit the RegionXY
        in question where all sides of the rectangle are parrallel to either the X or Y axes.

        Returns
        ---------
        (left, right, bottom, top): tuple[float, float, float, float]
            the x and y values for the bounds
        """
        if len(self.loops) == 1:
            return self.loops[0].axis_aligned_bounding_box()
        else:
            raise NotImplementedError("RegionXY.axis_aligned_bounding_box is only implemented for single loop regions.")

    # alias for easy use axis_aligned_bounding_box()
    aabbox = axis_aligned_bounding_box

    def filter_points(self, p: Pxy) -> Pxy:
        """Returns only the subset of given points that are inside or on
        the border of the region."""
        return p[self.is_inside_or_on_border(p)]

    @classmethod
    def rectangle(cls, size: float | tuple[float, float]):
        """Create a rectangular region centered at (0,0)

        Parameters
        ----------
        size: float | Iterable[float]
            If 'size' is a scalar the region will be a square with side lengths of 'size'.
            If it is an iterable of length 2 the width (x direction) will be length size[0]
            and the height (y direction) will be of length size[1].
        """
        if np.isscalar(size):
            width = size
            height = size
        elif len(size) == 2:
            width = size[0]
            height = size[1]
        else:
            raise ValueError("size must be either a scalar or a 2 element list-like.")

        x = width / 2
        y = height / 2

        vertices = Pxy([[-x, -x, x, x], [y, -y, -y, y]])

        loop = LoopXY.from_vertices(vertices)
        return RegionXY(loop)

    @classmethod
    def from_vertices(cls, vertices: Pxy):
        """Creates a single loop region defined by the vertices given."""
        loop = LoopXY.from_vertices(vertices)
        return RegionXY(loop)

    def __add__(self, other: Vxy | numbers.Number) -> "RegionXY":
        if isinstance(other, Vxy) or isinstance(other, numbers.Number):
            pass
        else:
            lt.error_and_raise(
                TypeError,
                "Error in RegionXY.__add__(): "
                + "secondary value in addition must be of type Vxy or Number, "
                + f"but is {type(other)}",
            )
        if isinstance(other, Vxy) and len(other) != 1:
            lt.error_and_raise(
                ValueError,
                "Error in RegionXY.__add__(): " + "other value Vxy must have length 1, " + f"but {len(other)=}",
            )

        ret = RegionXY(self.loops[0] + other)
        for loop in self.loops[1:]:
            ret.add_loop(loop + other)

        return ret

    def __sub__(self, other: Vxy | numbers.Number) -> "RegionXY":
        return self + (-other)
