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

    By convention subclasses are named "View*ImageProcessor" (their name starts
    with "View" and ends with "ImageProcessor"). Note that subclasses should not
    implement their own _execute() methods, but should instead implement
    num_figures, _init_figure_records(), visualize_operable(), and
    close_figures().

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

    Examples
    --------
    An example class that simply renders operables as an image might be implemented as::

        class ViewSimpleImageProcessor(AbstractVisualizationImageProcessor):
            def __init__(self, name, interactive):
                super().__init__(name, interactive)

                self.figure_rec: RenderControlFigureRecord = None

            @property
            def num_figures(self):
                return 1

            def _init_figure_records(self, render_control_fig):
                self.fig_record = fm.setup_figure(
                    render_control_fig,
                    rca.image(),
                    equal=False,
                    name=self.name,
                    code_tag=f"{__file__}._init_figure_records()",
                )
                return [self.fig_record]

            def visualize_operable(self, operable, is_last):
                image = operable.primary_image.nparray
                self.fig_record.view.imshow(image)

            def close_figures(self):
                with exception_tools.ignored(Exception):
                    self.fig_record.close()
                self.fig_record = None
    """

    def __init__(self, name: str, interactive: bool | Callable[[SpotAnalysisOperable], bool]):
        """
        Parameters
        ----------
        name : str
            Passed through to AbstractSpotAnalysisImagesProcessor.__init__()
        interactive : bool | Callable[[SpotAnalysisOperable], bool], optional
            If True then the spot analysis pipeline is paused until the user presses the "enter" key, by default False
        """
        # import here to avoid circular dependencies
        from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator

        super().__init__(name)

        # register arguments
        self.interactive = interactive

        # internal values
        self.visualization_coordinator: VisualizationCoordinator = None
        """
        The coordinator registered with this instance through
        register_visualization_coordinator(). If None, then it is assumed that
        we should draw the visualization during the _execute() method.
        """
        self.initialized_figure_records = False
        """ True if init_figure_records() has been called, False otherwise. """

    @property
    @abstractmethod
    def num_figures(self) -> int:
        """
        How many figure windows this instance intends to create. Must be
        available at all times after this instance has been initialized.
        """
        pass

    @abstractmethod
    def _init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        """
        Initializes the figure windows (via figure_management.setup_figure*) for
        this instance and returns the list of initialized figures. The length of
        this list ideally should match what was previously returned for
        num_figures.

        Parameters
        ----------
        render_control_fig : rcf.RenderControlFigure
            The render controller to use during figure setup.

        Returns
        -------
        figures: list[rcfr.RenderControlFigureRecord]
            The list of newly created figure windows.
        """
        pass

    @abstractmethod
    def visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool) -> None:
        """
        Updates the figures for this instance with the data from the given operable.
        """
        pass

    @abstractmethod
    def close_figures(self):
        """
        Closes all visualization windows created by this instance.
        """
        pass

    @property
    def has_visualization_coordinator(self) -> bool:
        """
        True if this instance is registered with a visualization coordinator.
        False otherwise.
        """
        return self.visualization_coordinator is not None

    def register_visualization_coordinator(self, coordinator):
        """
        Registers the given coordinator with this visualization processor instance.

        Parameters
        ----------
        coordinator : VisualizationCoordinator
            The coordinator that is registering against this instance.
        """
        # Note: no type hint for coordinator to avoid a circular import dependency
        self.visualization_coordinator = coordinator

    def init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        """
        Called by the registered coordinator to create any necessary
        visualization windows. If there is no registered coordinator by the time
        _execute is called, then this method will be evaluated by this instance
        internally.

        Parameters
        ----------
        render_control_fig : rcf.RenderControlFigure
            The controller to use with figure_management.setup_figure*

        Returns
        -------
        list[rcfr.RenderControlFigureRecord]
            The list of newly created visualization windows.
        """
        ret = self._init_figure_records(render_control_fig)
        self.initialized_figure_records = True
        return ret

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        if self.has_visualization_coordinator:
            if self.visualization_coordinator.is_visualize(self, operable, is_last):
                self.visualization_coordinator.visualize(self, operable, is_last)
        else:
            if not self.initialized_figure_records:
                self.init_figure_records(rcf.RenderControlFigure(tile=False))
            self.visualize_operable(operable, is_last)

        return [operable]
