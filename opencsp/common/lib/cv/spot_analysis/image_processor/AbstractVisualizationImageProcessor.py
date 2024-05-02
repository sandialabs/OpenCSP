from abc import ABC, abstractmethod
from typing import Callable

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr


class AbstractVisualizationImageProcessor(AbstractSpotAnalysisImagesProcessor, ABC):
    """
    An AbstractSpotAnalysisImagesProcessor that is used to generate visualizations.

    The visualizations that these processors create can be used either for
    debugging or monitoring, depending on the value of the "interactive"
    initialization parameter.

    VisualizationCoordinator
    ------------------------
    Certain elements of the visualization are handled by the
    VisualizationCoordinator, including at least:

        - tiled layout of visualization windows
        - user interaction that is common to all visualization windows

    The life cycle for this class is::

        - __init__()
        - register_visualization_coordinator()*
        - num_figures()*
        - init_figure_records()*
        - process()
        -     _execute()
        -     visualize_operable()*
        - close_figures()*

    In the above list, one star "*" indicates that this method is called by the
    coordinator.
    """

    def __init__(self, name: str, interactive: bool | Callable[[SpotAnalysisOperable], bool]):
        # import here to avoid circular dependencies
        from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator

        super().__init__(name)

        # register arguments
        self.interactive = interactive

        # internal values
        self.visualization_coordinator: VisualizationCoordinator = None
        self.initialized_figure_records = False

    @property
    @abstractmethod
    def num_figures(self) -> int:
        pass

    @abstractmethod
    def _init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        pass

    @abstractmethod
    def visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool) -> None:
        pass

    @abstractmethod
    def close_figures(self):
        pass

    @property
    def has_visualization_coordinator(self) -> bool:
        return self.visualization_coordinator is not None

    def register_visualization_coordinator(self, coordinator):
        self.visualization_coordinator = coordinator

    def init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> None:
        self._init_figure_records(render_control_fig)
        self.initialized_figure_records = True

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        if self.has_visualization_coordinator:
            if self.visualization_coordinator.is_visualize(self, operable, is_last):
                self.visualization_coordinator.visualize(self, operable, is_last)
        else:
            if not self.initialized_figure_records:
                self.init_figure_records(rcf.RenderControlFigure(tile=False))
            self.visualize_operable(operable, is_last)

        return [operable]
