from abc import ABC, abstractmethod
import copy
import dataclasses
from typing import Callable
import weakref

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.render.figure_management as fm
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
        -     _get_image_for_visualizing()
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
        self._render_control_fig: weakref.ref[rcf.RenderControlFigure] = None
        """
        The figure control used in init_figure_records(). Managed as a weak
        reference so that this reference doesn't complicating garbage collection.
        """
        self._include_visualization_image_no_axes: bool = False
        """
        If True, then the _visualization_image_no_axes image is set on the
        returned SpotAnalysisOperable. This value is used by the next
        visualization image processor that has its base_image_selector set to
        'visualization', and then the value is unset.
        """

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
                if operable._visualization_image_no_axes is not None:
                    print(self.name)
                    return operable._visualization_image_no_axes
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
            figure records. Figure records preferred. Empty list if there aren't any.
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
        self._render_control_fig = weakref.ref(render_control_fig)
        ret = self.init_figure_records(render_control_fig)
        self.initialized_figure_records = True
        return ret

    @staticmethod
    def default_render_control_figure_for_operable(operable: SpotAnalysisOperable):
        """
        Create a default render control figure for the given operable.

        This static method generates a render control figure based on the dimensions
        and number of channels of the primary image associated with the provided
        `SpotAnalysisOperable`. This default figure does not use tiling and does not
        include the typical matplotlib whitespace padding.

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
        """
        # ChatGPT 4o-mini assisted with generating this docstring, reviewed by a human
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
        # ChatGPT 4o-mini assisted with generating this docstring, reviewed by a human
        ret: SpotAnalysisOperable = None

        # Remember the _visualization_image_no_axes value so that it can be unset if unchanged.
        previs_visualization_image_no_axes = operable._visualization_image_no_axes

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
            new_visualizations, _visualization_image_no_axes = self._visualize_operable(operable, is_last)

            # get the visualization images list
            visualization_images = copy.copy(operable.visualization_images)
            if self not in visualization_images:
                visualization_images[self] = []
            else:
                visualization_images[self] = copy.copy(visualization_images[self])
            visualization_images[self] += new_visualizations

            # update the return value
            ret = dataclasses.replace(
                operable,
                visualization_images=visualization_images,
                _visualization_image_no_axes=_visualization_image_no_axes,
            )

        # Unset the _visualization_image_no_axes if unchanged.
        # We do this because that value is intended to be used for the next processor only.
        if ret._visualization_image_no_axes == previs_visualization_image_no_axes:
            ret = dataclasses.replace(ret, _visualization_image_no_axes=None)

        return [ret]

    def prepare_figure_records(
        self,
        figure_records: list[rcfr.RenderControlFigureRecord],
        figure_shapes: list[list[int | float] | tuple | None] = (),
    ):
        """
        Clears the figure records, sets the layout margins to "tight", and
        sets the figure record shape.

        Parameters
        ----------
        figure_records : list[rcfr.RenderControlFigureRecord]
            The figure records to be prepared.
        figure_shapes : list[list[int | float] | tuple | None], optional
            The shapes of the figure records. This can either be the desired
            height/width of the figure in inches (if < 50), or the number of
            rows/columns in the figure (if > 50). The figure will be resized so
            that the width-to-height ratio matches this value.
        """
        for fig_record in figure_records:
            fig_record.figure.tight_layout()
            fig_record.clear()

        for i in range(min(len(figure_records), len(figure_shapes))):
            fig_record, fig_shape = figure_records[i], figure_shapes[i]
            if fig_shape is None:
                continue
            if len(fig_shape) < 2:
                lt.error_and_raise(
                    ValueError,
                    f"In AbstractVisualizationImageProcessor.prepare_figure_records ({self.name}):"
                    + f"expected figure shape to have at least two dimensions, but {figure_shapes[i]=}",
                )
            curr_height = fig_record.figure.get_size_inches()[1]

            if fig_shape[0] >= 50 or fig_shape[1] >= 50:
                # Assume that fig_shape is pixel dimensions with
                # rows (0) and columns (1), such as from array.shape
                nrows, ncols = fig_shape[0], fig_shape[1]
                width_to_height = ncols / nrows
                fig_record.figure.set_size_inches(width_to_height * curr_height, curr_height)

            else:
                # Assume the the fig_shape is the height/width in inches.
                height, width = fig_shape[0], fig_shape[1]
                width_to_height = width / height
                fig_record.figure.set_size_inches(width_to_height * curr_height, curr_height)

    def _hide_vis_window_dressings(self, fig_record: rcfr.RenderControlFigureRecord):
        # hide all the window dressings
        if (render_control_fig := self._render_control_fig()) is not None:
            fm.hide_axes(fig_record, render_control_fig, force_hide=True)
        old_title = fig_record.title
        fig_record.title = ""
        fig_record.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        return {"old_title": old_title}

    def _apply_window_dressings(self, fig_record: rcfr.RenderControlFigureRecord, window_dressings: dict[str, any]):
        # re-apply all the window dressings
        fig_record.figure.tight_layout()
        if (render_control_fig := self._render_control_fig()) is not None:
            fm.show_axes(fig_record, render_control_fig)
        old_title: str = window_dressings["old_title"]
        fig_record.title = old_title

    def _visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool
    ) -> tuple[list[CacheableImage], CacheableImage | None]:
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

        # verify the returned type
        if not isinstance(visualizations, list):
            raise TypeError(
                f"Error in {self.name}.visualize_operable(): "
                + "should have returned a list of visualizations but instead returned a "
                + str(type(visualizations))
            )
        for i, visualization in enumerate(visualizations):
            if not (
                isinstance(visualization, CacheableImage) or isinstance(visualization, rcfr.RenderControlFigureRecord)
            ):
                raise TypeError(
                    f"Error in {self.name}.visualize_operable(): "
                    + "should have returned a list of CacheableImage and RenderControlFigureRecord, but "
                    + f"visualization {i} is a "
                    + str(type(visualizations))
                )

        # build the list of visualization images
        all_vis_images: list[CacheableImage] = []
        _visualization_image_no_axes: CacheableImage = None
        for i, cacheable_or_figure_rec in enumerate(visualizations):
            if isinstance(cacheable_or_figure_rec, CacheableImage):
                cacheable = cacheable_or_figure_rec

                all_vis_images.append(cacheable)

            else:
                fig_record = cacheable_or_figure_rec

                # get the figure as an numpy array, using the standard 8 inches figure height
                np_image = fig_record.to_array(8.0)

                # assign the figure as the latest visualization
                if self._include_visualization_image_no_axes:
                    if i == 0:
                        # hide window dressings
                        window_dressings = self._hide_vis_window_dressings(fig_record)

                        # render the undressed visualization
                        _visualization_image_no_axes = CacheableImage.from_single_source(fig_record.to_array(8.0))

                        # re-apply the window dressings
                        self._apply_window_dressings(fig_record, window_dressings)

                # add the image
                cacheable_image = CacheableImage(np_image)
                all_vis_images.append(cacheable_image)

        return all_vis_images, _visualization_image_no_axes
