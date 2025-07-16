from abc import ABC, abstractmethod
import copy
import dataclasses
from typing import Callable

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class AbstractVisualizationImageProcessor(AbstractSpotAnalysisImageProcessor, ABC):
    """
    An AbstractSpotAnalysisImageProcessor that is used to generate visualizations.

    By convention subclasses are named "View*ImageProcessor" (their name starts
    with "View" and ends with "ImageProcessor"). Note that subclasses should not
    implement their own _execute() methods, but should instead implement
    num_figures, init_figure_records(), visualize_operable(), and
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

            def init_figure_records(self, render_control_fig):
                self.fig_record = fm.setup_figure(
                    render_control_fig,
                    rca.image(),
                    equal=False,
                    name=self.name,
                    code_tag=f"{__file__}.init_figure_records()",
                )
                return [self.fig_record]

            def visualize_operable(self, operable, is_last):
                image = operable.primary_image.nparray
                self.fig_record.view.imshow(image)
                return [self.fig_record]

            def close_figures(self):
                with exception_tools.ignored(Exception):
                    self.fig_record.close()
                self.fig_record = None
    """

    def __init__(
        self,
        interactive: bool | Callable[[SpotAnalysisOperable], bool],
        base_image_selector: str | ImageType = None,
        name: str = None,
    ):
        """
        Parameters
        ----------
        interactive : bool | Callable[[SpotAnalysisOperable], bool], optional
            If True then the spot analysis pipeline is paused until the user presses the "enter" key, by default False
        base_image_selector : ImageType, optional
            Which image to draw the visualization on top of. The latest
            available of the given type is used. Can also be one of None,
            'Visualization', or 'Algorithm'. Default is the latest primary
            image.
        name : str
            Passed through to AbstractSpotAnalysisImageProcessor.__init__()
        """
        # import here to avoid circular dependencies
        from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator

        super().__init__(name)

        # validate arguments
        if isinstance(base_image_selector, str):
            acceptable_values = ["visualization", "algorithm"]
            if base_image_selector.lower() not in acceptable_values:
                lt.error_and_raise(
                    ValueError,
                    "Error in AbstractVisualizationImageProcessor(): "
                    + f"base_image_selector must be either an ImageType or one of {acceptable_values}, "
                    + "but it is '{base_image_selector}'",
                )

        # register arguments
        self.interactive = interactive
        self.base_image_selector = base_image_selector
        """
        Determines the image returned from
        :py:meth:`_get_image_for_visualizing`. Typically this will be one of
        None/ImageType.PRIMARY or 'visualization'.
        """

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
    def init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
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

    def _get_image_for_visualizing(self, operable: SpotAnalysisOperable) -> CacheableImage:
        """
        Chooses one of the operable's images to use to draw visualizations on
        top of based on self.:py:attr:`base_image_selector`. The returned value
        will be passed through to :py:meth:`visualize_operable` as the
        base_image.
        """
        if self.base_image_selector is None or self.base_image_selector == ImageType.PRIMARY:
            return operable.primary_image
        elif isinstance(self.base_image_selector, str):
            if self.base_image_selector.lower() == 'visualization':
                return list(operable.visualization_images.values())[-1][0]
            elif self.base_image_selector.lower() == 'algorithm':
                return list(operable.algorithm_images.values())[-1][0]
            else:
                lt.error_and_raise(
                    RuntimeError,
                    "Error in AbstractVisualizationImageProcessor._get_image_for_visualization(): "
                    + f"unknown base_image_selector string value '{self.base_image_selector}'",
                )
        elif self.base_image_selector in [
            ImageType.REFERENCE,
            ImageType.NULL,
            ImageType.COMPARISON,
            ImageType.BACKGROUND_MASK,
        ]:
            return operable.supporting_images[self.base_image_selector]
        else:
            lt.error_and_raise(
                RuntimeError,
                "Error in AbstractVisualizationImageProcessor._get_image_for_visualization(): "
                + f"unknown base_image_selector of type {type(self.base_image_selector)}: {self.base_image_selector}",
            )

    @abstractmethod
    def visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool, base_image: CacheableImage
    ) -> list[CacheableImage | rcfr.RenderControlFigureRecord]:
        """
        Updates the figures for this instance with the data from the given operable.

        The implementing visualization image processor has the option of
        returning visualizations as cacheable images, figure records, or a mix
        of both.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable to draw the visualization for.
        is_last : bool
            True if this is the last operable to be drawn by this processor.
        base_image : CacheableImage
            The base image on which to draw the visualization. Value is
            determined by :py:attr:`base_image_selector` and retrieved with
            :py:meth:`_get_image_for_visualizing`.

        Returns
        -------
        visualizations: list[CacheableImage|rcfr.RenderControlFigureRecord]
            Visualizations from this image processor as cacheable images or as
            figure records. Empty list if there aren't any.
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

    def _init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
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
        ret = self.init_figure_records(render_control_fig)
        self.initialized_figure_records = True
        return ret

    @staticmethod
    def default_render_control_figure_for_operable(operable: SpotAnalysisOperable):
        """
        Create a default render control figure for the given operable.

        This static method generates a render control figure based on the dimensions
        and number of channels of the primary image associated with the provided
        `SpotAnalysisOperable`. The figure is configured with specific settings
        such as tile layout and whitespace padding.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            An instance of `SpotAnalysisOperable` containing the primary image
            for which the render control figure is to be created.

        Returns
        -------
        rcf.RenderControlFigure
            A configured render control figure that can be used for visualizing
            the operable's primary image.

        Notes
        -----
        - The figure size is determined based on the pixel dimensions of the
           primary image.
        - The method assumes that the `primary_image` attribute of the operable
          contains a valid NumPy array representation of the image.

        """
        # ChatGPT 4o-mini assisted with generating this docstring
        (height_px, width_px), nchannel = it.dims_and_nchannels(operable.primary_image.nparray)
        figsize = rcf.RenderControlFigure.pixel_resolution_inches(width_px, height_px)
        figure_control = rcf.RenderControlFigure(tile=False, figsize=figsize, grid=False, draw_whitespace_padding=False)
        return figure_control

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        """
        Execute the visualization process for the given operable.

        This method performs the visualization of the provided `SpotAnalysisOperable`.
        It checks for the presence of a visualization coordinator and either visualizes
        the operable through the coordinator or directly if no coordinator is available.
        The method also manages the visualization images associated with the operable.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable instance to be visualized. It may contain visualization
            images that will be updated during the execution.

        is_last : bool
            A flag indicating whether this is the last operable to be processed.
            This may affect how the visualization is handled.

        Returns
        -------
        list[SpotAnalysisOperable]
            A list containing the updated `SpotAnalysisOperable` instance with
            the visualization images included.

        Notes
        -----
        - If a visualization coordinator is present, the operable is visualized
        through it. If not, the visualization is performed immediately.
        - The method initializes figure records if they have not been set up
        previously.
        - The visualization images are copied and updated to ensure that the
        original operable remains unchanged.
        """
        # ChatGPT 4o-mini assisted with generating this docstring
        ret: SpotAnalysisOperable = None

        if self.has_visualization_coordinator:
            # Visualize the operable and block (if interactive).
            op_with_vis = self.visualization_coordinator.visualize(self, operable, is_last)
            if op_with_vis is not None:
                ret = op_with_vis
            else:
                ret = dataclasses.replace(operable)
        else:
            # no coordinator for synchronized visualization, always visualize
            # the operable immediately
            if not self.initialized_figure_records:
                # Create the figure to plot to
                render_control = self.default_render_control_figure_for_operable(operable)
                self._init_figure_records(render_control)
            new_visualizations = self._visualize_operable(operable, is_last)

            # get the visualization images list
            visualization_images = copy.copy(ret.visualization_images)
            if self not in visualization_images:
                visualization_images[self] = []
            else:
                visualization_images[self] = copy.copy(visualization_images[self])
            visualization_images[self] += new_visualizations

            # update the return value
            ret = dataclasses.replace(operable, visualization_images=visualization_images)

        return [ret]

    def _visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool) -> list[CacheableImage]:
        """
        Calls :py:meth:`visualize_operable` and collects the visualziation images.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable to visualize.
        is_last : bool
            True if this is the last operable that this method will be evaluated for.

        Returns
        -------
        list[CacheableImage]
            This processor's visualizations.
        """
        # visualize the operable
        base_image = self._get_image_for_visualizing(operable)
        visualizations = self.visualize_operable(operable, is_last, base_image)

        # build the list of visualization images
        all_vis_images: list[CacheableImage] = []
        for cacheable_or_figure_rec in visualizations:
            if isinstance(cacheable_or_figure_rec, CacheableImage):
                all_vis_images.append(cacheable_or_figure_rec)

            else:
                # get the figure as an numpy array, using the standard 8 inches figure height
                np_image = cacheable_or_figure_rec.to_array(8.0)

                # add the image
                cacheable_image = CacheableImage(np_image)
                all_vis_images.append(cacheable_image)

        return all_vis_images
