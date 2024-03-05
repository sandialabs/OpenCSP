from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class SpotAnalysisPopulationStatistics():
    maxf: npt.NDArray[np.float_] = None
    """ Maximum value seen across images. None if not yet calculated. """
    minf: npt.NDArray[np.float_] = None
    """ Minimum value seen across images. None if not yet calculated. """
    avgf_rolling_window: npt.NDArray[np.float_] = None
    """ Average value seen across images. None if not yet calculated. """
    
    window_size: int = 1
    """ Current window size, for statistics that are calculated as a rolling window. """
    population_size: int = 0
    """ How many images have been sampled by the time this operable was seen. """
    
    @property
    def maxi(self) -> npt.NDArray[np.int_]:
        """ Like fmax, but returns the rounded integer result. """
        ret = np.round(self.maxf)
        if not np.issubdtype(ret.dtype, np.integer):
            ret = ret.astype(np.int32)
        return ret
    
    @property
    def mini(self) -> npt.NDArray[np.int_]:
        """ Like fmin, but returns the rounded integer result. """
        ret = np.round(self.minf)
        if not np.issubdtype(ret.dtype, np.integer):
            ret = ret.astype(np.int32)
        return ret
    
    @property
    def avgi_rolling_window(self) -> npt.NDArray[np.int_]:
        """ Like favg_rolling_window, but returns the rounded integer result. """
        ret = np.round(self.avgf_rolling_window)
        if not np.issubdtype(ret.dtype, np.integer):
            ret = ret.astype(np.int32)
        return ret