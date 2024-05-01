import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.render_control.RenderControlFigure as rcf


class VisualizationCoordinator:
    """
    Coordinates visualizations across many image processors, so that the same
    image is seen from each processor at the same time.
    """
    max_tiles_x = 4
    """ How many tiles we can have in the horizontal direction """
    max_tiles_y = 2
    """ How many tiles we can have in the vertical direction """

    def __init__(self):
        self.visualization_processors: list[AbstractVisualizationImageProcessor] = []
        self.render_control_fig: rcf.RenderControlFigure = None

    def clear(self):
        self.visualization_processors.clear()

    def register_visualization_processors(self, all_processors: list[AbstractSpotAnalysisImagesProcessor]):
        # find and register all visualization processors
        for processor in all_processors:
            if isinstance(processor, AbstractVisualizationImageProcessor):
                visualization_processor: AbstractVisualizationImageProcessor = processor
                visualization_processor.register_visualization_coordinator(self)
                self.visualization_processors.append(visualization_processor)

        # determine the tiling arangement
        num_figures = 0
        for processor in self.visualization_processors:
            num_figures += processor.num_figures
        if num_figures <= 1:
            tiles_x = 1
            tiles_y = 1
        elif num_figures <= 2:
            tiles_x = 2
            tiles_y = 1
        elif num_figures <= 8:
            tiles_x = int(np.ceil(num_figures / 2))
            tiles_y = 2
        elif num_figures <= 12:
            tiles_x = int(np.ceil(num_figures / 3))
            tiles_y = 3
        else:
            tiles_y = int(np.floor(np.sqrt(num_figures)))
            tiles_x = int(np.ceil(num_figures / tiles_y))
        tiles_x = np.min([tiles_x, self.max_tiles_x])
        tiles_y = np.min([tiles_y, self.max_tiles_y])

        # build the figure manager
        if tiles_x == 1 and tiles_y == 1:
            self.render_control_fig = rcf.RenderControlFigure(tile=False)
        else:
            self.render_control_fig = rcf.RenderControlFigure(tile=True, tile_array=(tiles_x, tiles_y))

        # initialize the visualizers
        for processor in self.visualization_processors:
            processor._init_figure_records(self.render_control_fig)

    def is_visualize(self, visualization_processor: AbstractVisualizationImageProcessor, operable: SpotAnalysisOperable, is_last: bool) -> bool:
        if visualization_processor == self.visualization_processors[-1]:
            return True
        return False

    def visualize(self, visualization_processor: AbstractVisualizationImageProcessor, operable: SpotAnalysisOperable, is_last: bool):
        for processor in self.visualization_processors:
            processor._visualize_operable(operable, is_last)
