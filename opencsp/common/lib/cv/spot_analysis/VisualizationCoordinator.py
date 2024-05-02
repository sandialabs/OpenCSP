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
    Coordinates visualizations across many visualization image processors.

    This class coordinates the following common visualization activities:

        - automatic tiling scaling for many visualization figures/windows
        - display of visualizations for one operable at a time
        - "enter", "shift+enter", and close operations on visualizations
    """

    max_tiles_x = 4
    """ How many tiles we can have in the horizontal direction """
    max_tiles_y = 2
    """ How many tiles we can have in the vertical direction """

    def __init__(self):
        # visualization handlers
        self.visualization_processors: list[AbstractVisualizationImageProcessor] = []
        """ List of all visualization processors registered with this instance. """
        self.figures: list[weakref.ref[rcfr.RenderControlFigureRecord]] = []
        """
        List of figures returned from init_figure_records() for each of the
        registered visualization handlers. Note that this is a list of weak
        references so that figure records can be released by the responsible
        party and this class won't hang onto resources unnecessarily.
        """

        # render control
        self.render_control_fig: rcf.RenderControlFigure = None
        """
        The render control handler. Sets the tiling status, rows, and columns
        for all registered visualization handlers.
        """

        # used to ensure a valid internal state
        self.has_registered_visualization_processors = False
        """ True if register_visualization_processors() has been evaluated. """

        # user interaction
        self.shift_down = False
        """ Monitors the state of the shift key """
        self.enter_pressed = False
        """
        True if enter has been pressed since the latest call to visualize().
        Only used in interactive mode.
        """
        self.enter_shift_pressed = False
        """
        True if shift+enter has been pressed an odd number of times. Only used
        in interactive mode.
        """
        self.closed = False
        """
        True if any visualization window has ever been closed. Only used in
        interactive mode.
        """

    def clear(self):
        """
        Closes all visualization windows, and resets the state for this coordinator.
        """
        # close all visualization windows
        for processor in self.visualization_processors:
            processor.close_figures()

        # reset internal state
        self.visualization_processors.clear()
        self.render_control_fig = None
        self.figures.clear()

        self.has_registered_visualization_processors = False
        self.enter_shift_pressed = False

    def on_key_release(self, event: matplotlib.backend_bases.KeyEvent):
        """
        Key release event handler for all visualization figure windows.
        """
        shift_down = self.shift_down
        key = event.key

        if "shift+" in key:
            shift_down = True
            key = key.replace("shift+", "")

        if key == "enter" or key == "return":
            self.enter_pressed = True
            if shift_down:
                self.enter_shift_pressed = not self.enter_shift_pressed
                lt.info("Interactive mode: " + ("Disabled" if self.enter_shift_pressed else "Enabled"))
        elif key == "shift":
            self.shift_down = False

    def on_key_press(self, event: matplotlib.backend_bases.KeyEvent):
        """
        Key press event handler for all visualization figure windows.
        """
        if event.key == "shift":
            self.shift_down = True

    def on_close(self, event: matplotlib.backend_bases.CloseEvent):
        """
        Window close event handler for all visualization figure windows.
        """
        self.closed = True

    def register_visualization_processors(self, all_processors: list[AbstractSpotAnalysisImagesProcessor]):
        """
        Finds all AbstractVisualizationImageProcessors in the given list of
        all_processors and registers this coordinator with them. This
        coordinator will then tell the processors to initialize their figures
        and register event handlers with the newly created figures.

        Parameters
        ----------
        all_processors: list[AbstractSpotAnalysisImagesProcessor]
            Processors to search through for visualization processors, some of
            which may be visualization processor and some not.
        """
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
        num_figures_dict: dict[AbstractVisualizationImageProcessor, int] = {}
        for processor in self.visualization_processors:
            num_figures_dict[processor] = processor.num_figures
        num_figures_total = sum(num_figures_dict.values())
        if num_figures_total <= 1:
            tiles_x = 1
            tiles_y = 1
        elif num_figures_total <= 2:
            tiles_x = 2
            tiles_y = 1
        elif num_figures_total <= 8:
            tiles_x = int(np.ceil(num_figures_total / 2))
            tiles_y = 2
        elif num_figures_total <= 12:
            tiles_x = int(np.ceil(num_figures_total / 3))
            tiles_y = 3
        else:
            tiles_y = int(np.floor(np.sqrt(num_figures_total)))
            tiles_x = int(np.ceil(num_figures_total / tiles_y))
        tiles_x = np.min([tiles_x, self.max_tiles_x])
        tiles_y = np.min([tiles_y, self.max_tiles_y])

        # build the figure manager
        if tiles_x == 1 and tiles_y == 1:
            self.render_control_fig = rcf.RenderControlFigure(tile=False)
        else:
            self.render_control_fig = rcf.RenderControlFigure(tile=True, tile_array=(tiles_x, tiles_y))

        # initialize the visualizers
        for processor in self.visualization_processors:
            processor_figures = processor.init_figure_records(self.render_control_fig)
            if len(processor_figures) != num_figures_dict[processor]:
                lt.warning("Warning in VisualizationCoordinator.register_visualization_processors(): " +
                           f"Unexpected number of visualization windows for processor {processor.name}. " + f" Expected {num_figures_dict[processor]} but received {len(processor_figures)}!")
            for fig_record in processor_figures:
                fig_record.figure.canvas.mpl_connect('close_event', self.on_close)
                fig_record.figure.canvas.mpl_connect('key_release_event', self.on_key_release)
                fig_record.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
                self.figures.append(weakref.ref(fig_record))

    def _get_figures(self) -> list[rcfr.RenderControlFigureRecord]:
        """
        Get strong references to the figure_records from the registered
        visualization handlers.

        Because the self.figures list is a list of weak references, we need to
        check each time we access the list if the figure_records are still
        available and haven't been garbage collected already. This method also
        removes any stale references from the list.
        """
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
        """
        Checks if now is an appropriate time to trigger visualizing for all
        registered visualization handlers. Typically called from each
        AbstractVisualizationImageProcessor's _execute() method.

        Parameters
        ----------
        visualization_processor : AbstractVisualizationImageProcessor
            The processor that is calling this method.
        operable : SpotAnalysisOperable
            The "operable" value in the calling processor's _execute() method,
            passed through to here.
        is_last : bool
            The "is_last" value in the calling processor's _execute() method,
            passed through to here.

        Returns
        -------
        bool
            True if now is a good time to trigger visualization of the given
            operable, False to skip visualization.
        """
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
        """
        Calls visualize_operable() on each of the registered visualization
        processors.

        This method is typically called from the _execute() method of a
        AbstractVisualizationImageProcessor after a call to is_visualize().

        After all visualizations have been updated, if interactive, then we
        block until the user has either pressed "enter" or closed the
        visualization windows.

        Parameters
        ----------
        visualization_processor : AbstractVisualizationImageProcessor
            The processor that is calling this method.
        operable : SpotAnalysisOperable
            The "operable" value in the calling processor's _execute() method,
            passed through to here. This is the operable to be visualized.
        is_last : bool
            The "is_last" value in the calling processor's _execute() method,
            passed through to here.
        """
        for processor in self.visualization_processors:
            processor.visualize_operable(operable, is_last)

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

                # if any plot is closed, then every plot was closed, and we can just continue
                if self.closed:
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
                        processor.close_figures()

                first_iteration = False
