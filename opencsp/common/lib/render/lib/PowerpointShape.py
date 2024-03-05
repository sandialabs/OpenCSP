import inspect
import pptx
from typing import Iterable, overload

class PowerpointShape():
    def __init__(self, cell_dims: tuple[float,float,float,float]=None, code_location: str=None):
        """ This class supplements the shape class from python-pptx. It allows us to do our custom layouts a little bit easier.

        Args:
            cell_dims (tuple[float,float,float,float]): left, right, top, bottom of the confining area of this shape (inches). Default None.
        """
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
    def _pptx_inches(self, val: int|float) -> int:
        pass
    
    @overload
    def _pptx_inches(self, val: Iterable) -> list[int]:
        pass
    
    def _pptx_inches(self, vals: int|float|Iterable):
        """ Converts the given values to powerpoint-style inches, for placement on a slide. """
        try:
            ret = []
            for val in vals:
                ret.append(pptx.util.Inches(val))
            return ret
        except:
            return pptx.util.Inches(val)
    
    def cell_dims_pptx(self):
        """ Returns the powerpoint-style inches that bound this shape (left, top, width, height). """
        return self._pptx_inches(self.cell_dims)
    
    def _dims_to_str(self, dims: Iterable[float]):
        if dims == None:
            return None
        return ",".join([str(v) for v in dims])
    
    def _str_to_dims(self, sval: str):
        if sval == None or sval == str(None):
            return None
        return tuple([float(v) for v in sval.split(",")])