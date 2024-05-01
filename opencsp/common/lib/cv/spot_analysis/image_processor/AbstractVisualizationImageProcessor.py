from abc import ABC, abstractmethod

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import AbstractSpotAnalysisImagesProcessor


class AbstractVisualizationImageProcessor(AbstractSpotAnalysisImagesProcessor, ABC):
    def __init__(self, name: str):
        # import here to avoid circular dependencies
        from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator
        super().__init__(name)

        self.visualization_coordinator: VisualizationCoordinator = None

    @property
    def has_visualization_coordinator(self):
        return self.visualization_coordinator is not None

    def register_visualization_coordinator(self, coordinator):
        self.visualization_coordinator = coordinator

    @abstractmethod
    def _visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool) -> None:
        pass

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        if self.has_visualization_coordinator:
            if self.visualization_coordinator.is_visualize(self, operable, is_last):
                self.visualization_coordinator.visualize(self, operable, is_last)
        else:
            self._visualize_operable(operable, is_last)

        return [operable]
