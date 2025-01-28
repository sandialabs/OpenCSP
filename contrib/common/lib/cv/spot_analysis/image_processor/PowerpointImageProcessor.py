import copy
import os
from typing import TypeVar

import PIL.Image
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractVisualizationImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.render.lib.PowerpointImage as pi
import opencsp.common.lib.render.PowerpointSlide as ps
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlPowerpointPresentation as rcpp
import opencsp.common.lib.render_control.RenderControlPowerpointSlide as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.string_tools as st


ProcOrImg = TypeVar('ProcOrImage', AbstractSpotAnalysisImageProcessor, CacheableImage, str, np.ndarray, PIL.Image.Image)


class ProcessorSelector:
    def __init__(self, processor_or_image: ProcOrImg, caption: str = None, image_types: list[ImageType] = None):
        self.processor_or_image = processor_or_image
        self.caption: str = caption

        self.processor: AbstractSpotAnalysisImageProcessor = None
        self.image: CacheableImage = None
        self.is_processor: bool = False
        self.has_caption: bool = caption is not None
        self.image_types: list[ImageType] = []

        if isinstance(processor_or_image, AbstractSpotAnalysisImageProcessor):
            self.processor = processor_or_image
            self.is_processor = True
        else:
            self.image = CacheableImage.from_single_source(processor_or_image)
            self.is_processor = False

        if image_types is None:
            self.image_types = ImageType.ALL()
        elif isinstance(image_types, list) or isinstance(image_types, tuple) or isinstance(image_types, set):
            self.image_types = list(image_types)
        elif isinstance(image_types, ImageType):
            self.image_types = [image_types]
        else:
            lt.error_and_raise(
                TypeError,
                "Error in PowerpointImageProcessor.ProcessorSelector(): "
                + f"image_types should be a list of ImageType, but {type(image_types)=}",
            )

    @classmethod
    def from_tuple(cls, processor_sel: tuple[ProcOrImg, str, ImageType]):
        caption = None
        image_types = None

        if len(processor_sel) > 1:
            if isinstance(processor_sel[1], str):
                # processor_sel is a tuple (ProcOrImg, 'caption', ...)
                caption = processor_sel[1]
                if len(processor_sel) > 2:
                    # processor_sel is a tuple (ProcOrImg, 'caption', ImageType...)
                    image_types = processor_sel[2:]
            else:
                # processor_sel is a tuple (ProcOrImg, ImageType...)
                image_types = processor_sel[1:]
        else:
            # processor_sel is a tuple (ProcOrImg)
            pass

        return cls(processor_sel[0], caption, image_types)

    def get_caption(self):
        if self.has_caption:
            caption = self.caption

        elif self.is_processor:
            if self.processor.name.endswith("ImageProcessor"):
                caption = self.processor.name[: -len("ImageProcessor")]
                caption = " ".join(st.camel_case_split(caption))
            else:
                caption = self.processor.name

        else:
            caption = "Image"
            if self.image.source_path is not None:
                caption = "".join(ft.path_components(self.image.source_path)[1:])

        return caption


class PowerpointImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Saves the results of the spot analysis image processing pipeline as a PowerPoint deck.
    """

    def __init__(
        self,
        save_dir: str,
        save_name: str,
        overwrite=False,
        operable_title_slides_todo=False,
        processors_per_slide: (
            list[list[ProcessorSelector | ProcOrImg | tuple[ProcOrImg, str, ImageType]]] | None
        ) = None,
    ):
        super().__init__()

        # register some of the input arguments
        self.save_dir = save_dir
        self.save_name = save_name if save_name.lower().endswith(".pptx") else save_name + ".pptx"
        self.dest_path_name_ext = ft.norm_path(os.path.join(self.save_dir, self.save_name))

        # validate input
        if not ft.directory_exists(self.save_dir):
            lt.error_and_raise(
                FileNotFoundError,
                "Error in PowerpointImageProcessor.__init__(): "
                + f"destination directory \"{self.save_dir}\" does not exist!",
            )
        if ft.file_exists(self.dest_path_name_ext):
            if not overwrite:
                lt.error_and_raise(
                    FileExistsError,
                    "Error in PowerpointImageProcessor.__init__(): "
                    + f"destination file \"{self.dest_path_name_ext}\" already exists!",
                )
        if processors_per_slide is not None:
            if not hasattr(processors_per_slide, "__iter__"):
                lt.error_and_raise(
                    TypeError,
                    "Error in PowerpointImageProcessor.__init__(): "
                    + f"\"processors_per_slide\" is a {type(processors_per_slide)}, but should be a list of lists",
                )
            else:
                for i, processor_set in enumerate(processors_per_slide):
                    if not hasattr(processor_set, "__iter__") or isinstance(
                        processor_set, AbstractSpotAnalysisImageProcessor
                    ):
                        lt.error_and_raise(
                            TypeError,
                            "Error in PowerpointImageProcessor.__init__(): "
                            + f"\"processors_per_slide[{i}]\" is a {type(processor_set)}, but should be a list!",
                        )
                    for j, processor_sel in enumerate(processor_set):
                        processor_sel_type = type(processor_sel)
                        # normalize to use ProcOrImg
                        if not isinstance(processor_sel, ProcessorSelector):
                            if not isinstance(processor_sel, tuple):
                                processor_sel = tuple([processor_sel])
                            try:
                                processors_per_slide[i][j] = ProcessorSelector.from_tuple(processor_sel)
                            except TypeError:
                                lt.error_and_raise(
                                    TypeError,
                                    "Error in PowerpointImageProcessor.__init__(): "
                                    + f"\"processors_per_slide[{i}][{j}]\" is a {processor_sel_type}, "
                                    + f"but should be an {type(AbstractSpotAnalysisImageProcessor)}, "
                                    + f"{type(CacheableImage)}, or image-like!",
                                )

        # register the rest of the input arguments
        self.overwrite = overwrite
        self.operable_title_slides = operable_title_slides_todo
        self.processors_per_slide = processors_per_slide

        # internal values
        self.is_first_operable = True
        self.presentation = rcpp.RenderControlPowerpointPresentation()
        self.slide_control = rcps.RenderControlPowerpointSlide()

    def _add_operable_title_slide(self, operable: SpotAnalysisOperable):
        # check if we should have title slides
        if not self.operable_title_slides:
            return

        # create a title slide and add it to the presentation
        slide = ps.PowerpointSlide.template_title(operable.best_primary_pathnameext, "", self.slide_control)
        self.presentation.add_slide(slide)

    def _get_processors_in_order(self, operable: SpotAnalysisOperable) -> list[AbstractSpotAnalysisImageProcessor]:
        previous_operables = operable.previous_operables[0]
        previous_processor: AbstractSpotAnalysisImageProcessor = operable.previous_operables[1]

        if previous_operables is None or previous_processor is None:
            return []

        else:
            ret = self._get_processors_in_order(previous_operables[0])
            ret.append(previous_processor)
            return ret

    def _populate_processor_images_dict(
        self,
        operable: SpotAnalysisOperable,
        processor_images_dict: dict[AbstractSpotAnalysisImageProcessor, list[tuple[CacheableImage, ImageType]]],
        include_processors: list[ProcessorSelector],
        recursion_index=0,
    ):
        """
        Follows the chain of operables backwards to get a list of all images
        that affected the given operable.
        """

        ############################################
        # Get images related to the current operable
        ############################################

        # get the values for one step back
        previous_operables = operable.previous_operables[0]
        """ The operables that preceeded this current operable """
        previous_processor: AbstractSpotAnalysisImageProcessor = operable.previous_operables[1]
        """ The processor that produced this current operable """
        if previous_operables is None or previous_processor is None:
            return
        if True:  # one operable per processor, to avoid a spanning tree for many-to-one processors
            previous_operables = previous_operables[:1]

        # make sure we have a list to add to
        if previous_processor not in processor_images_dict:
            processor_images_dict[previous_processor] = []

        # Add the visualization images for this step
        found_algo_images = False
        if previous_processor in operable.visualization_images:
            for image in operable.visualization_images[previous_processor]:
                processor_images_dict[previous_processor].append((image, ImageType.VISUALIZATION))
                found_algo_images = True

        # If we didn't find any images, then add the primary image for this step
        if not found_algo_images:
            processor_images_dict[previous_processor].append((operable.primary_image, ImageType.PRIMARY))

        # add the supporting images for this step
        # TODO

        ######################################
        # Get images for the operable's parent
        ######################################

        # repeat for the previous step in the operable chain
        for previous_operable in previous_operables:
            self._populate_processor_images_dict(
                previous_operable, processor_images_dict, include_processors, recursion_index=recursion_index + 1
            )

        #################################################################
        # Special Case: processors that have special visualization images
        #################################################################

        # add missing visualization images
        if recursion_index == 0:
            vis_processors = filter(
                lambda proc: isinstance(proc, AbstractVisualizationImageProcessor), processor_images_dict.keys()
            )
            for processor in vis_processors:
                processor_images_dict[processor] = []
                for vis_image in operable.visualization_images[processor]:
                    processor_images_dict[processor].append((vis_image, ImageType.VISUALIZATION))

        ######################################################
        # Get images that explain how determinations were made
        ######################################################

        if self.is_first_operable:
            for processor in operable.algorithm_images:
                algorithm_images = operable.algorithm_images[processor]
                if len(algorithm_images) == 0:
                    continue

                if processor not in processor_images_dict:
                    processor_images_dict[processor] = []
                for algo_image in algorithm_images:
                    processor_images_dict[processor].append((algo_image, ImageType.ALGORITHM))

        ###############################
        # Limit to the given processors
        ###############################

        # collect the list of include processors
        select_processors = [processor_sel.processor for processor_sel in include_processors]

        # remove any processors that don't match the limited types
        for processor in list(processor_images_dict.keys()):
            if processor not in select_processors:
                del processor_images_dict[processor]

        ##################################
        # Limit to the desired image types
        ##################################

        for processor_sel in include_processors:
            if processor_sel.processor in processor_images_dict:
                for image_imagetype in copy.copy(processor_images_dict[processor_sel.processor]):
                    image, image_type = image_imagetype
                    if image_type not in processor_sel.image_types:
                        processor_images_dict[processor_sel.processor].remove(image_imagetype)

        ###############################
        # Remove duplicate images
        ###############################

        for processor in processor_images_dict:
            keep_images: list[tuple[CacheableImage, ImageType]] = []
            for cacheable_imagetype in processor_images_dict[processor]:
                if cacheable_imagetype not in keep_images:
                    keep_images.append(cacheable_imagetype)
            processor_images_dict[processor] = keep_images

    def save(self):
        """Saves this presentation out to disk"""
        self.presentation.save(self.dest_path_name_ext, self.overwrite)

    def fetch_input_operable(self):
        """Get the next operable. If an unexpected exception gets thrown, then
        panic save the presentation as it exists up to this point."""
        try:
            return super().fetch_input_operable()
        except Exception as ex:
            if not isinstance(ex, StopIteration):
                self.save()
            raise

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # add a title slide for the operable, as necessary
        self._add_operable_title_slide(operable)

        # build the slides for this operable
        processor_images_dict_template: dict[
            AbstractSpotAnalysisImageProcessor | CacheableImage, list[tuple[CacheableImage, ImageType]]
        ] = {}

        # Initialize the processor_images dict. Initializing it in this way will
        # will preserve processor ordering when the dictionary is filled.
        all_processors = self._get_processors_in_order(operable)
        for processor in all_processors:
            processor_images_dict_template[processor] = []

        # check if this should be for all processors
        processors_per_slide = self.processors_per_slide
        if processors_per_slide is None:
            processors_per_slide = [[tuple([processor])] for processor in all_processors]

        # add one slide for each set of processors
        for processor_set in processors_per_slide:
            processor_images_dict = copy.copy(processor_images_dict_template)

            # get the images per processor
            processor_filter = lambda processor_sel: processor_sel.is_processor
            processor_only_set = list(filter(processor_filter, processor_set))
            self._populate_processor_images_dict(operable, processor_images_dict, processor_only_set)

            # get the images
            images_list: list[tuple[ProcessorSelector, CacheableImage]] = []
            for processor_sel in processor_set:
                if processor_sel.is_processor:
                    for image, image_type in processor_images_dict[processor_sel.processor]:
                        images_list.append((processor_sel, image))
                else:
                    images_list.append((processor_sel, processor_sel.image))

            # prepare the slide
            n_rows, n_cols = rcf.RenderControlFigure.num_tiles_4x3aspect(len(images_list))
            slide = ps.PowerpointSlide.template_content_grid(n_rows, n_cols, self.slide_control)

            # add information to the slide
            slide.set_title(operable.best_primary_nameext)
            for processor_sel, image in images_list:
                caption = processor_sel.get_caption()
                slide.add_image(pi.PowerpointImage(image.nparray, caption=caption))

            # add the slide to the presentation
            slide.save_and_bake()
            self.presentation.add_slide(slide)

        # save the powerpoint presentation
        if is_last:
            self.save()

        self.is_first_operable = False
        return [operable]
