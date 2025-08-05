import copy
import dataclasses
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
        - "enter", "shift+enter", and close operations on visualizations
    """

    max_tiles_x = 4
    """ How many tiles we can have in the horizontal direction """
    max_tiles_y = 2
    """ How many tiles we can have in the vertical direction """

    def __init__(self):
        # visualization handlers
        self.visualization_processors: list[AbstractVisualizationImageProcessor] = []
        """ List of all visualization processors registered with this instance.
        They will be in the same order as given in
        :py:meth:`register_visualization_processors`. """
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
        self.has_initialized_vis_processors = False
        """ True if initialize_vis_processors() has been evaluated. """

        # user interaction
        self.shift_down = False
        """
        Monitors the state of the shift key.
        Only used in interactive mode.
        """
        self.enter_pressed = False
        """
        True if enter has been pressed since the latest call to visualize().
        Only used in interactive mode.
        """
        self.enter_shift_pressed = False
        """
        True if shift+enter has been pressed an odd number of times.
        Only used in interactive mode.
        """
        self.closed = False
        """
        True if any visualization window has ever been closed.
        Only used in interactive mode.
        """

    def clear(self):
        """
        Closes all visualization windows, and resets the state for this coordinator.
        """
        # close all visualization windows
        if self.has_initialized_vis_processors:
            for processor in self.visualization_processors:
                processor.close_figures()

        # reset internal state
        self.visualization_processors.clear()
        self.render_control_fig = None
        self.figures.clear()

        self.has_registered_visualization_processors = False
        self.has_initialized_vis_processors = False

        self.shift_down = False
        self.enter_pressed = False
        self.enter_shift_pressed = False

        self.closed = False

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

    def register_visualization_processors(self, all_processors: list[AbstractSpotAnalysisImageProcessor]):
        """
        Finds all AbstractVisualizationImageProcessors in the given list of
        all_processors and registers this coordinator with them. This
        coordinator will then tell the processors to initialize their figures
        and register event handlers with the newly created figures.

        Parameters
        ----------
        all_processors: list[AbstractSpotAnalysisImageProcessor]
            Processors to search through for visualization processors, some of
            which may be visualization processor and some not.
        """
        # Developer's note: this method is not safe to be called multiple times

        if self.has_registered_visualization_processors:
            lt.warning(
                "Warning in VisualizationCoordinator.register_visualization_processors(): "
                + "attempting to register processors again without calling 'clear()' first."
            )
            return
        self.has_registered_visualization_processors = True

        # find and register all visualization processors
        for processor in all_processors:
            if isinstance(processor, AbstractVisualizationImageProcessor):
                visualization_processor: AbstractVisualizationImageProcessor = processor
                visualization_processor.register_visualization_coordinator(self)
                self.visualization_processors.append(visualization_processor)

    def initialize_vis_processors(self, operable: SpotAnalysisOperable):
        if self.has_initialized_vis_processors:
            return
        self.has_initialized_vis_processors = True

        # determine the tiling arangement
        num_figures_dict: dict[AbstractVisualizationImageProcessor, int] = {}
        for processor in self.visualization_processors:
            num_figures_dict[processor] = processor.num_figures
        num_figures_total = sum(num_figures_dict.values())
        tiles_y, tiles_x = rcf.RenderControlFigure.num_tiles_4x3aspect(num_figures_total)
        tiles_x = np.min([tiles_x, self.max_tiles_x])
        tiles_y = np.min([tiles_y, self.max_tiles_y])

        # build the figure manager
        self.render_control_fig = AbstractVisualizationImageProcessor.default_render_control_figure_for_operable(
            operable
        )
        if tiles_x == 1 and tiles_y == 1:
            pass
        else:
            self.render_control_fig.tile = True
            self.render_control_fig.tile_array = (tiles_x, tiles_y)

        # initialize the visualizers
        for processor in self.visualization_processors:
            processor_figures = processor._init_figure_records(self.render_control_fig)
            if len(processor_figures) != num_figures_dict[processor]:
                lt.warning(
                    "Warning in VisualizationCoordinator.register_visualization_processors(): "
                    + f"Unexpected number of visualization windows for processor {processor.name}. "
                    + f" Expected {num_figures_dict[processor]} but received {len(processor_figures)}!"
                )
            for fig_record in processor_figures:
                # register callbacks for figures
                fig_record.figure.canvas.mpl_connect('close_event', self.on_close)
                fig_record.figure.canvas.mpl_connect('key_release_event', self.on_key_release)
                fig_record.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

                # register this figure for coordinated management
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

    def is_last_visualization_processor(self, visualization_processor: AbstractVisualizationImageProcessor) -> bool:
        """
        Returns True if the given processor is the last visualization processor
        in the list of :py:attr:`visualization_processors`.
        """
        return visualization_processor is self.visualization_processors[-1]

    def is_interactive(self, operable: SpotAnalysisOperable) -> bool:
        """
        Returns False if interactive mode was disabled by pressing shift+enter.
        Otherwise returns True if any of the registered
        :py:attr:`visualization_processors` is interactive.
        """
        for processor in self.visualization_processors:
            if isinstance(processor.interactive, bool):
                if processor.interactive:
                    return True
            else:
                if processor.interactive(operable):
                    return True

        return False

    def wait_after_visualization(
        self, visualization_processor: AbstractVisualizationImageProcessor, operable: SpotAnalysisOperable
    ) -> bool:
        """
        Returns True if image processing should be blocked after the given
        visualization_processor has rendered the given operable.
        """
        if self.is_interactive(operable):
            if self.is_last_visualization_processor(visualization_processor):
                if not self.closed:
                    if not self.enter_shift_pressed:
                        return True

        return False

    def visualize(
        self,
        visualization_processor: AbstractVisualizationImageProcessor,
        operable: SpotAnalysisOperable,
        is_last: bool,
    ) -> SpotAnalysisOperable:
        """
        Calls :py:meth:`visualize_operable` on the given visualization_processor
        and assigns the returned images as visualization_images on the returned
        operable. Then, if interactive, continued execution is blocked until
        "enter" is pressed or any visualization windows is closed.

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

        Returns
        -------
        operable: SpotAnalysisOperable
            A copy of the given operable with all visualization images (if any)
            appended to the operable.
        """
        # initialize the visualization processors, as necessary
        self.initialize_vis_processors(operable)

        # render the visualization image processor
        processor_visualizations = visualization_processor._visualize_operable(operable, is_last)

        # compile all visualizations together into a single operable to be returned
        if len(processor_visualizations) > 0:
            # make a copy of the latest operable's visualizations
            all_vis_images: dict[AbstractSpotAnalysisImageProcessor, list] = {}
            for processor2 in operable.visualization_images:
                all_vis_images[processor2] = copy.copy(operable.visualization_images[processor2])

            # append the new visualizations
            if visualization_processor not in all_vis_images:
                all_vis_images[visualization_processor] = []
            all_vis_images[visualization_processor] += processor_visualizations

            # update the operable
            operable = dataclasses.replace(operable, visualization_images=all_vis_images)
        else:
            # update the operable
            operable = dataclasses.replace(operable)

        # if interactive, then block until the user presses "enter" or closes one or more visualizations
        if self.wait_after_visualization(visualization_processor, operable):
            lt.info(
                "Starting interactive visualization. Press 'enter' on any visualization window to continue...", end=''
            )
            self.enter_pressed = False

            first_iteration = True
            while True:
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
                if self.enter_pressed or self.enter_shift_pressed:
                    break
                if self.closed:
                    # UI design decision: it feels more natural to me (Ben) for
                    # the plot to not be shown again when it has been closed
                    # instead of being reinitialized and popping back up.
                    for processor in self.visualization_processors:
                        processor.close_figures()

                first_iteration = False

            lt.info("continuing execution")

        return operable
