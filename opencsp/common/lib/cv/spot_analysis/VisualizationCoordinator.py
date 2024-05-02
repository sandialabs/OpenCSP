import weakref

import matplotlib
import matplotlib.backend_bases
import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.tool.log_tools as lt


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
        self.figures: list[weakref.ref[rcfr.RenderControlFigureRecord]] = []

        # used to ensure a valid internal state
        self.has_registered_visualization_processors = False

        # user interaction
        self.shift_down = False
        self.enter_pressed = False
        self.enter_shift_pressed = False
        self.closed = False

    def clear(self):
        self.visualization_processors.clear()
        self.render_control_fig = None
        self.figures.clear()

        self.has_registered_visualization_processors = False
        self.enter_shift_pressed = False

    def on_key_release(self, event: matplotlib.backend_bases.KeyEvent):
        shift_down = self.shift_down
        key = event.key

        if "shift+" in key:
            shift_down = True
            key = key.replace("shift+", "")

        if key == "enter" or key == "return":
            if shift_down:
                self.enter_shift_pressed = True
            else:
                self.enter_pressed = True
        elif key == "shift":
            self.shift_down = False

    def on_key_press(self, event: matplotlib.backend_bases.KeyEvent):
        if event.key == "shift":
            self.shift_down = True

    def on_close(self, event: matplotlib.backend_bases.CloseEvent):
        self.closed = True

    def register_visualization_processors(self, all_processors: list[AbstractSpotAnalysisImagesProcessor]):
        # this method is not safe to be called multiple times
        if self.has_registered_visualization_processors:
            lt.warning("Warning in VisualizationCoordinator.register_visualization_processors(): " +
                       "attempting to register processors again without calling 'clear()' first.")
            return
        self.has_registered_visualization_processors = True

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
            processor_figures = processor._init_figure_records(self.render_control_fig)
            for fig_record in processor_figures:
                fig_record.figure.canvas.mpl_connect('close_event', self.on_close)
                fig_record.figure.canvas.mpl_connect('key_release_event', self.on_key_release)
                fig_record.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
                self.figures.append(weakref.ref(fig_record))

    def _get_figures(self) -> list[rcfr.RenderControlFigureRecord]:
        figures = [(ref, ref()) for ref in self.figures]
        alive = [fr for ref, fr in filter(lambda ref, fr: fr is not None, figures)]
        dead = [ref for ref, fr in filter(lambda ref, fr: fr is None, figures)]

        for ref in dead:
            self.figures.remove(ref)

        return alive

    def is_visualize(
        self,
        visualization_processor: AbstractVisualizationImageProcessor,
        operable: SpotAnalysisOperable,
        is_last: bool,
    ) -> bool:
        if self.closed:
            return False
        elif visualization_processor == self.visualization_processors[-1]:
            return True
        return False

    def visualize(
        self,
        visualization_processor: AbstractVisualizationImageProcessor,
        operable: SpotAnalysisOperable,
        is_last: bool,
    ):
        for processor in self.visualization_processors:
            processor._visualize_operable(operable, is_last)

        # if interactive, then block until the user presses "enter" or closes one or more visualizations
        interactive = False
        for processor in self.visualization_processors:
            if isinstance(processor.interactive, bool):
                interactive |= processor.interactive
            else:
                interactive |= processor.interactive(operable)
        if interactive:
            self.enter_pressed = False
            self.closed = False

            first_iteration = True
            while True:
                # if shift+enter was ever pressed, that will disable interactive mode
                if self.enter_shift_pressed:
                    break

                # wait for up to total_wait_time for the user to interact with the visualizations
                old_raise = matplotlib.rcParams["figure.raise_window"]
                matplotlib.rcParams["figure.raise_window"] = first_iteration
                figures = list(filter(lambda fr: fr is not None, [ref() for ref in self.figures]))
                total_wait_time = 0.1  # seconds
                per_record_wait_time = total_wait_time / len(figures)
                for fig_record in figures:
                    if fig_record.figure.waitforbuttonpress(per_record_wait_time) is not None:
                        break
                matplotlib.rcParams["figure.raise_window"] = old_raise

                # check for interaction
                if self.enter_pressed:
                    break
                if self.closed:
                    # UI design decision: it feels more natural to me (Ben) for
                    # the plot to not be shown again when it has been closed
                    # instead of being reinitialized and popping back up.
                    for processor in self.visualization_processors:
                        processor._close_figures()

                first_iteration = False
