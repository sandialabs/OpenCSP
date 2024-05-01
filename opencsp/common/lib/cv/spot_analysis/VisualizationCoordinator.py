from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *


class VisualizationCoordinator:
    """
    Coordinates visualizations across many image processors, so that the same
    image is seen from each processor at the same time.
    """

    def __init__(self):
        self.visualization_processors: list[AbstractVisualizationImageProcessor] = []

    def clear(self):
        self.visualization_processors.clear()

    def register_visualization_processors(self, all_processors: list[AbstractSpotAnalysisImagesProcessor]):
        for processor in all_processors:
            if isinstance(processor, AbstractVisualizationImageProcessor):
                visualization_processor: AbstractVisualizationImageProcessor = processor
                visualization_processor.register_visualization_coordinator(self)
                self.visualization_processors.append(visualization_processor)

    def is_visualize(self, visualization_processor: AbstractVisualizationImageProcessor, operable: SpotAnalysisOperable, is_last: bool) -> bool:
        if visualization_processor == self.visualization_processors[-1]:
            return True
        return False

    def visualize(self, visualization_processor: AbstractVisualizationImageProcessor, operable: SpotAnalysisOperable, is_last: bool):
        for processor in self.visualization_processors:
            processor._visualize_operable(operable, is_last)
