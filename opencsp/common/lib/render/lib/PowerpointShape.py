import inspect
import pptx
from typing import Iterable, overload


class PowerpointShape:
    """
    A class that supplements the shape class from python-pptx, facilitating custom layouts.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(self, cell_dims: tuple[float, float, float, float] = None, code_location: str = None):
        """Initializes the PowerpointShape with specified dimensions and code location.

        Args:
            cell_dims (tuple[float, float, float, float], optional):
                The left, right, top, and bottom dimensions of the confining area of this shape (in inches). Defaults to None.
            code_location (str, optional):
                The location in the code where the instance was created. Defaults to None.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.cell_dims = cell_dims
        self.code_location = code_location

        if self.code_location == None:
            trace = []
            frame = inspect.currentframe().f_back
            while frame != None:
                trace.append(str(frame))
                frame = frame.f_back
            self.code_location = "\n".join(reversed(trace))

    @overload
    def _pptx_inches(self, val: int | float) -> int:
        pass

    @overload
    def _pptx_inches(self, val: Iterable) -> list[int]:
        pass

    def _pptx_inches(self, vals: int | float | Iterable):
        # Converts the given values to PowerPoint-style inches for placement on a slide.
        #
        # Parameters
        # ----------
        # vals : int, float, or Iterable
        #    The value(s) to convert to inches. Can be a single value or an iterable of values.
        #
        # Returns
        # -------
        # int or list[int]
        #    The converted value(s) in PowerPoint inches.
        #
        # Notes
        # -----
        # If a single value is provided, it returns a single integer. If an iterable is provided, it returns a list of integers.
        # "ChatGPT 4o" assisted with generating this doc
        try:
            ret = []
            for val in vals:
                ret.append(pptx.util.Inches(val))
            return ret
        except:
            return pptx.util.Inches(val)

    def cell_dims_pptx(self):
        """Returns the PowerPoint-style inches that bound this shape (left, top, width, height).

        Returns
        -------
        list[int]
            A list of the dimensions in PowerPoint inches.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return self._pptx_inches(self.cell_dims)

    def _dims_to_str(self, dims: Iterable[float]):
        # Converts a tuple of dimensions to a comma-separated string.
        #
        # Parameters
        # ----------
        # dims : Iterable[float]
        #    The dimensions to convert.
        #
        # Returns
        # -------
        # str or None
        #    A comma-separated string of dimensions, or None if the input is None.
        # "ChatGPT 4o" assisted with generating this doc
        if dims == None:
            return None
        return ",".join([str(v) for v in dims])

    def _str_to_dims(self, sval: str):
        # Converts a comma-separated string of dimensions back to a tuple of floats.
        #
        # Parameters
        # ----------
        # sval : str
        #    The string to convert.
        #
        # Returns
        # -------
        # tuple[float] or None
        #    A tuple of floats representing the dimensions, or None if the input is None or the string representation of None.
        # "ChatGPT 4o" assisted with generating this doc

        if sval == None or sval == str(None):
            return None
        return tuple([float(v) for v in sval.split(",")])
