import copy
import numbers

import matplotlib.pyplot as plt
import numpy as np

# import opencsp.common.lib.geometry.Resolution as Resolution

from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
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
        raise NotImplementedError('Cannot add more than one loop to a region currently.')

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

    def points_sample(self, resolution: 'Resolution') -> Pxy:
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
            raise NotImplementedError('RegionXY.axis_aligned_bounding_box is only implemented for single loop regions.')

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
                + f"secondary value in addition must be of type Vxy or Number, "
                + f"but is {type(other)}",
            )
        if isinstance(other, Vxy) and len(other) != 1:
            lt.error_and_raise(
                ValueError,
                "Error in RegionXY.__add__(): " + f"other value Vxy must have length 1, " + f"but {len(other)=}",
            )

        ret = RegionXY(self.loops[0] + other)
        for loop in self.loops[1:]:
            ret.add_loop(loop + other)

        return ret

    def __sub__(self, other: Vxy | numbers.Number) -> "RegionXY":
        return self + (-other)


class Resolution:
    """
    Allows options for defining a set of points needed. To choose a type of
    Resolution use a class method with keeps the type of unresolved resolution
    stored until the bouning box containing the resolution is known.

    Attributes
    ----------
    self.points: Pxy
        The points that the resolution cares about. These are the xy points
        that will be used in whatever the resolution is for.
    self.unresolved: tuple[str, ...]
        The description of the properties the resolution should have once
        it is given a bounding box to axt on.
    """

    def __init__(self, points: Pxy) -> None:
        self._points = points
        self.unresolved: tuple[str, ...] = None
        self.composite_transformation = TransformXYZ.identity()

    @classmethod
    def separation(cls, separation: float) -> 'Resolution':
        """Separates the points along x and y by the separation."""
        res = Resolution(None)
        res.unresolved = ("separation", separation)
        return res

    @classmethod
    def pixelX(cls, points_along_x: int) -> 'Resolution':
        """Will have `points_along_x` points along x and
        equispaced points along y."""
        res = Resolution(None)
        res.unresolved = ("pixelX", points_along_x)
        return res

    @classmethod
    def random(cls, number_of_points: int, seed: int = None) -> 'Resolution':
        """There will be `number_of_points` uniformly randomly in the region.
        Can choose to add a seed."""
        res = Resolution(None)
        res.unresolved = ("random", number_of_points, seed)
        return res

    @classmethod
    def center(cls) -> 'Resolution':
        """Gives the center point of a bounding box.
        This resolution cannot be 'resolved' since there is not a set of points that can
        represent what it is trying to do."""
        res = Resolution(None)
        res.unresolved = ("center",)
        return res

    @property
    def points(self) -> Pxy:
        if self._points is None:
            RuntimeError("Resolution has no points, probably due to an unresolved tag.")
        return self._points

    def is_resolved(self) -> bool:
        return self.unresolved is None

    def is_unresolved(self) -> bool:
        return self.unresolved is not None

    def change_frame_and_copy(self, frame_transform: TransformXYZ) -> 'Resolution':
        res = copy.deepcopy(self)
        frame_translation = frame_transform.V.projXY()
        if res.is_unresolved():
            return res
        res._points = res.points + frame_translation
        res.composite_transformation = frame_transform * res.composite_transformation
        return res

    def resolve_in_place(self, bounding_box: tuple[float, float, float, float] | RegionXY):
        """RegionXY
        If the Resolution object is "unresolved" this is the function that resolves it to a
        set of points in xy space (Pxy). If there is no unresolved tag, this function
        just filters out points that do not fall in bounding box. Acts in place.
        """
        # if a RegionXY is given:
        region: RegionXY = None
        if isinstance(bounding_box, RegionXY):
            region = bounding_box
            bounding_box = region.axis_aligned_bounding_box()

        left, right, bottom, top = bounding_box

        if self.is_resolved():  # add `and region is None` if this is too slow
            self._points = Pxy.merge([p for p in self.points if left <= p.x[0] <= right and bottom <= p.y[0] <= top])
        elif self.is_unresolved():  # self is unresolved
            width = right - left
            height = top - bottom

            match self.unresolved[0]:

                case "separation":
                    separation = float(self.unresolved[1])

                    width -= separation / 1e10
                    height -= separation / 1e10

                    x_shift = (width % separation) / 2
                    x_start, x_end = (left + x_shift, right - x_shift)
                    y_shift = (height % separation) / 2
                    y_start, y_end = (bottom + y_shift, top - y_shift)

                    xs = np.arange(x_start, x_end, separation)
                    ys = np.arange(y_start, y_end, separation)
                    X = [x for x in xs for y in ys]
                    Y = [y for x in xs for y in ys]
                    self._points = Pxy([X, Y])
                    self.unresolved = None

                case "pixelX":
                    points_along_x = int(self.unresolved[1])

                    separation = (right - left) / points_along_x
                    self.unresolved = ("separation", separation)
                    self.resolve_in_place(bounding_box)

                case "random":
                    number_of_points = int(self.unresolved[1])
                    random_seed = int(self.unresolved[2]) if self.unresolved[2] is not None else None

                    rng = np.random.default_rng(random_seed)
                    xs = rng.uniform(left, right, number_of_points)
                    ys = rng.uniform(bottom, top, number_of_points)
                    self._points = Pxy([xs, ys])
                    self.unresolved = None

                case "center":
                    self._points = Pxy([left + width / 2, bottom + height / 2])
                    # does not resolve

                case _:
                    ValueError(f"{self.unresolved[0]} is not a valid resolution type.")

        if region is not None:
            self._points = region.filter_points(self.points)

        pass  # end of resolve

    def resolve_and_copy(self, bounding_box: tuple[float, float, float, float] | RegionXY) -> 'Resolution':
        """
        If the Resolution object is "unresolved" this is the function that resolves it to a
        set of points in xy space (Pxy). If there is no unresolved tag, this function
        just filters out points that do not fall in bounding box. Produces a new Resolution object.
        """
        new_res = copy.deepcopy(self)
        new_res.resolve_in_place(bounding_box)
        return new_res
