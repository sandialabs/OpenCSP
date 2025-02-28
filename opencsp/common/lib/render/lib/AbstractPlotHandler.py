from abc import ABC
import copy

import matplotlib.backend_bases
import matplotlib.figure
import matplotlib.pyplot as plt

import opencsp.common.lib.tool.exception_tools as et


class AbstractPlotHandler(ABC):
    """Class to automatically track and close matplotlib plot windows.

    Note that even through this will ensure that all registered plots are closed when the this instance is destructed,
    it is almost always better to close the figure as soon as it's not needed any more via the close() method.

    Implementing classes need to make calls to:
    - super().__init__()
    - super().__del__()
    - self._register_plot(fig)"""

    def __init__(self, *vargs, **kwargs):
        self._open_plots: list[matplotlib.figure.Figure] = []

    def __del__(self):
        self._free_plots()

    def close(self):
        self._free_plots()

    def _on_plot_closed(self, event: matplotlib.backend_bases.CloseEvent):
        # Stop tracking plots that are still open when the plots get closed.
        to_remove = None
        for fig in self._open_plots:
            if fig.canvas == event.canvas:
                to_remove = fig
                break
        if to_remove is not None:
            self._open_plots.remove(to_remove)

    def _register_plot(self, fig: matplotlib.figure.Figure):
        # Registers the given figure, to be closed when this instance is closed or destructed.
        self._open_plots.append(fig)
        fig.canvas.mpl_connect("close_event", self._on_plot_closed)

    def _free_plots(self):
        for fig in copy.copy(self._open_plots):
            with et.ignored(Exception):
                plt.close(fig)
