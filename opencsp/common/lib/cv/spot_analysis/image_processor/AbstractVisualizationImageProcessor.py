from abc import ABC, abstractmethod

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.render_control.RenderControlFigure as rcf


class AbstractVisualizationImageProcessor(AbstractSpotAnalysisImagesProcessor, ABC):
    def __init__(self, name: str):
        # import here to avoid circular dependencies
        from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator

        super().__init__(name)

        self.visualization_coordinator: VisualizationCoordinator = None
        self.initialized_figure_records = False

    @property
    @abstractmethod
    def num_figures(self) -> int:
        pass

    @abstractmethod
    def _init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> None:
        pass

    @abstractmethod
    def _visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool) -> None:
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
            self._visualize_operable(operable, is_last)

        return [operable]
