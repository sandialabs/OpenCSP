import os
from typing import Iterator

from PIL import Image as Image
import numpy as np

from opencsp import opencsp_settings
from opencsp.common.lib.cv.CacheableImage import CacheableImage

# from opencsp.common.lib.cv.spot_analysis.image_processor import * # I suggest importing these dynamically as needed, to reduce startup time
from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractSpotAnalysisImageProcessor
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.ImagesStream import ImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import SpotAnalysisImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperablesStream import _SpotAnalysisOperablesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperableAttributeParser import SpotAnalysisOperableAttributeParser
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisPipeline import SpotAnalysisPipeline
from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator
import opencsp.common.lib.render.VideoHandler as vh
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class SpotAnalysis(Iterator[SpotAnalysisOperable]):
    """Spot Analysis class for characterizing beams of light.

    This is meant to be a general beam characterization tool that receives as
    input an image (or video) of a single beam, and generates annotations and
    numerical statistics as output. In general, there is an image processing
    step, an image analysis step, and a visualization step. Any number of
    processors and analyzers can be chained together to form a pipeline for
    evaluation.

    Once defined, a SpotAnalysis instance can evaluate any number of input
    images with either the :py:meth:`set_primary_images` or
    :py:meth:`set_input_operables` methods.

    A list of possible use cases for this class include:
        a. BCS
        b. BCS-based heliostat calibration
        c. Vertical laser characterization
        d. Laser beam spread characterization
        e. 2f returned spot analysis
        f. Motion characterization video analysis
        g. Wind BCS video trajectory analysis
        h. Laser-based cross-check
        i. Earth mover's metric for comparisons (possible implementations (1) scipy has a method or (2) https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf)
        j. Enclosed power

    These are our expected outputs:
        - Annotated images
        - Visualizations (ex 2d or 3d heat-flux map, side-by-side comparisons)
        - Accompanying statistics INI files
        - Other details text file (heliostat name and location, date and time, heliostat operator, etc...)
        - Powerpoint slides

    The features necessary to support these use cases include:
        1. BCS target location (:py:class:`.BcsLocatorImageProcessor`)
        2. Square target location (TargetBoardLocatorImageProcessor)
        3. NULL image subtraction (:py:class:`.NullImageSubtractionImageProcessor`)
        4. Ambient and gradient light subtraction (NOT IMPLEMENTED YET)
        5. Lens correction (NOT IMPLEMENTED YET)
        6. Intensity per pixel generation (NOT IMPLEMENTED YET)
        7. Centroid calculation (a-h) (NOT IMPLEMENTED YET)
        8. FWHM analysis (a,b,d,e,j) (NOT IMPLEMENTED YET)
        9. Astigmatism identification (a-e,j) (NOT IMPLEMENTED YET)
        10. Automatic beam identification (b) (NOT IMPLEMENTED YET)
        11. Video-to-frame extraction (f,g) (:py:class:`.ImagesStream`)
        12. Video streaming (f,g) (NOT IMPLEMENTED YET)
        13. Networked image collection (b) (NOT IMPLEMENTED YET)
        14. Color-aware image subtraction (c,d,e,h) (NOT IMPLEMENTED YET)
        15. Color-aware light subtraction (c,d,e,h) (NOT IMPLEMENTED YET)
        16. Time series analysis (f,g) (NOT IMPLEMENTED YET)
        17. Laser placement and pointing angle as image metadata (e,h) (NOT IMPLEMENTED YET)
        18. Laser placement and pointing angle as separate input (e,h) (NOT IMPLEMENTED YET)
        19. Earth mover's metric (i) (NOT IMPLEMENTED YET)
        20. Multi-image comparison (f,g,i) (NOT IMPLEMENTED YET)
        21. Intensity-to-power mapping (a,d,i,j) (NOT IMPLEMENTED YET)
        22. Noise smoothing (:py:class:`.ConvolutionImageProcessor`)
        23. Noise removal and quanitification (NOT IMPLEMENTED YET)
        24. Multi-image averaging over time (a-e,h-j) (:py:class:`.AverageByGroupImageProcessor`)
        25. Fiducial identification (e,18,21,23,24) (NOT IMPLEMENTED YET)
        26. Screen/camera plane homography (TargetBoardLocatorImageProcessor)
        27. Self-PnP and stabilization (:py:class:`.StabilizationImageProcessor`)
        28. Annotate power envelopes (NOT IMPLEMENTED YET)
        29. Orthorectify wrt. target (TargetBoardLocatorImageProcessor)
        30. Orthorectify wrt. beam (NOT IMPLEMENTED YET)
        31. Spot radius in mrad, at angle Beta relative to centroid (NOT IMPLEMENTED YET)
        32. Spot radius in mrad, at a half angle relative to the heliostat location (NOT IMPLEMENTED YET)
        33. Beam radius in mrad, at angle Beta relative to centroid (NOT IMPLEMENTED YET)
        34. Beam radius in mrad, at a half angle relative to the heliostat location (NOT IMPLEMENTED YET)
        35. Cropping (:py:class:`.CroppingImageProcessor`)
        36. Over and under exposure detection (:py:class:`.ExposureDetectionImageProcessor`)
        37. Tagging images as NULL images (NOT IMPLEMENTED YET)
        38. Filters (gaussian, box, etc) (:py:class:`.ConvoluationImageProcessor`)
        39. Peak value pixel identification (:py:class:`.HotspotImageProcessor`)
        40. Logorithmic scaling (:py:class:`.LogScaleImageProcessor`)
        41. False color visualization (:py:class:`.FalseColorImageProcessor`)
        42. Over/under exposure visualization (NOT IMPLEMENTED YET)
        43. Image fill/inpainting (:py:class:`.InpaintImageProcessor`)

    The inputs to support these features include:
        - primary image or images (:py:meth:`set_primary_images`)
        - reference image (7,8,22) (:py:meth:`set_default_support_images`)
        - null image (1,2,12,13,20) (:py:meth:`set_default_support_images`)
        - comparison image (18) (:py:meth:`set_default_support_images`)
        - background selection (2,11,13) (NOT IMPLEMENTED YET)
        - camera lens characterization (3) (NOT IMPLEMENTED YET)
        - image server network address (11) (NOT IMPLEMENTED YET)
        - laser color/wavelength (12,13) (NOT IMPLEMENTED YET)
        - camera color characterization (12,13) (NOT IMPLEMENTED YET)
        - laser placement and pointing angle (16) (NOT IMPLEMENTED YET)
        - intensity-to-power mapping (19) (NOT IMPLEMENTED YET)
        - multiple primary images, or a primary video (20,24) (:py:meth:`set_primary_images`)
        - fiducial definition and location (25) (NOT IMPLEMENTED YET)
        - manual 3D point identification from 2D images (25) (NOT IMPLEMENTED YET)

    Example usage::

        group_assigner = AverageByGroupImageProcessor.group_by_name(re.compile(r"(foo|bar)"))
        group_trigger = AverageByGroupImageProcessor.group_trigger_on_change()

        image_processors = {
            'Crop': CroppingImageProcessor(x1=200, x2=600, y1=100, y2=400),
            'AvgG': AverageByGroupImageProcessor(group_assigner, group_trigger),
            'Echo': EchoImageProcessor(),
            'Noiz': ConvolutionImageProcessor(kernel="box", diameter=3),
            'Ve3d': View3dImageProcessor(crop_to_threshold=20, max_resolution=(100, 100)),
            'Stat': PopulationStatisticsImageProcessor(min_pop_size=1, initial_min=0, initial_max=255),
            'Fclr': FalseColorImageProcessor(),
        }
        pptx_processor = PowerpointImageProcessor(
            save_dir=outdir,
            save_name="processing_pipeline",
            processors_per_slide=[
                [image_processors['Noiz'], image_processors['Ve3d']],
                [image_processors['Fclr']],
            ],
        )
        image_processors_list = list(image_processors.values()) + [pptx_processor]

        spot_analysis = sa.SpotAnalysis(
            experiment_name, image_processors_list, save_dir=outdir
        )
        spot_analysis.set_primary_images(images_path_name_exts)

        for result in spot_analysis:
            pass
    """

    def __init__(
        self,
        name: str,
        image_processors: list[AbstractSpotAnalysisImageProcessor],
        save_dir: str = None,
        save_overwrite=False,
    ):
        """
        Parameters
        ----------
        name: str
            The name of this instance. For example, this could be one of the use
            cases listed above.
        image_processors: list[AbstractSpotAnalysisImageProcessor]
            The list of processors that will be pipelined together and used to
            process input images.
        save_dir: str
            If not None, then primary images will be saved to the given
            directory as a PNG after having been fully processed. Defaults to None.
        save_overwrite: bool
            If True, then overwrite any existing images in the save_dir with the
            new output. Defaults to False.
        """
        self.name = name
        """ The name of this instance. For example, this could be one of the use
        cases listed above. """
        self.image_processors: list[AbstractSpotAnalysisImageProcessor] = []
        """ List of processors, one per step of the analysis. The output from
        each processor can include one or more operables, and is made available
        to all subsequent processors. """
        self.input_stream: _SpotAnalysisOperablesStream = None
        """ The images to be processed. """
        self._processing_pipeline: SpotAnalysisPipeline = None
        """ The current iterator and evaluation pipeline. """
        self.save_dir: str = save_dir
        """ If not None, then primary images will be saved to the given
        directory as a PNG after having been fully processed. """
        self.saved_names: set[str] = set()
        """ The names of image files saved to, to ensure unique image file names
        are used. """
        self.save_idx: int = 0
        """ The counter used to create unique image file names to save to, as
        necessary. """
        self.save_overwrite = save_overwrite
        """ If True, then overwrite any existing images in the save_dir with the
        new output. Defaults to False. """
        self.default_support_images: dict[ImageType, CacheableImage] = None
        """ Other supporting images for processing input images. If not None, then
        all values here will be made available for processing as the default
        values. """
        self.default_data: SpotAnalysisOperable = None
        """ Other supporting data for processing input images. If not None, then
        all values here will be made available for processing as the default
        values. """
        self.visualization_coordinator = VisualizationCoordinator()
        """ Manages interactive viewing of all visualization image processors. """

        self.set_image_processors(image_processors)

    def set_image_processors(self, image_processors: list[AbstractSpotAnalysisImageProcessor]):
        self.image_processors = image_processors

        # register the visualization processors
        self.visualization_coordinator.clear()
        self.visualization_coordinator.register_visualization_processors(image_processors)

        # register the image processors for in-flight results
        self._operables_in_flight = {processor: [] for processor in image_processors}

    @staticmethod
    def _images2stream(
        images: list[str] | list[np.ndarray] | vh.VideoHandler | ImagesStream,
    ) -> ImagesStream | ImagesIterable:
        if isinstance(images, ImagesStream):
            return images
        else:
            return ImagesIterable(images)

    def _assign_inputs(self, input_operables: Iterator[SpotAnalysisOperable]):
        # sanitize arguments
        if not isinstance(input_operables, _SpotAnalysisOperablesStream):
            input_operables = _SpotAnalysisOperablesStream(input_operables)

        # assign inputs
        self.input_stream = input_operables

        # reset temporary values
        self._prev_result = None
        self.set_image_processors(self.image_processors)

    def set_primary_images(self, images: list[str] | list[np.ndarray] | vh.VideoHandler | ImagesStream):
        """Assigns the images of the spot to be analyzed, in preparation for process_next().

        Parameters
        ----------
        images : list[str] | list[np.ndarray] | vh.VideoHandler | ImagesStream
            The primary input images to be processed. Can be a list of images
            path/name.ext, list of images loaded into numpy arrays, a video
            handler instance set with a video to be broken into frames, or an
            images stream.

        Notes
        -----
        Only one of set_primary_images or :py:meth:`set_input_operables` should
        be used to assign input images.
        """
        primary_images = self._images2stream(images)
        images_stream = SpotAnalysisImagesStream(primary_images, {})
        self._assign_inputs(images_stream)

    def set_input_operables(
        self,
        input_operables: _SpotAnalysisOperablesStream | list[SpotAnalysisOperable] | Iterator[SpotAnalysisOperable],
    ):
        """Assigns primary and supporting images, and other necessary data, in preparation for process_next().

        Notes
        -----
        Only one of set_primary_images or :py:meth:`set_input_operables` should
        be used to assign input images.
        """
        self._assign_inputs(input_operables)

    def set_default_support_images(self, support_images: dict[ImageType, CacheableImage]):
        """Provides extra support images for use during image processing, as a
        default for when the support images are not otherwise from the input
        operables. Note that this does not include the primary images or other
        data."""
        # check for state consistency
        if self.input_stream is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in SpotAnalysis.set_default_support_images(): "
                + "must call set_input_operables() or set_primary_images() first.",
            )

        self.input_stream.set_defaults(support_images, self.input_stream.default_data)

    def set_default_data(self, operable: SpotAnalysisOperable):
        """Provides extra data for use during image processing, as a default
        for when the data is not otherwise from the input operables. Note that
        this does not include the primary or supporting images."""
        # check for state consistency
        if self.input_stream is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in SpotAnalysis.set_default_data(): "
                + "must call set_input_operables() or set_primary_images() first.",
            )

        self.input_stream.set_defaults(self.input_stream.default_support_images, operable)

    def process_next(self):
        """Attempts to get the next processed image and results data from the
        image processors pipeline.

        Returns
        -------
        result: SpotAnalysisOperable | None
            The processed primary image and other associated data. None if done
            processing.
        """
        last_processor = self.image_processors[-1]

        # sanity check
        if self.input_stream is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in SpotAnalysis.process_next(): "
                + "must assign inputs via set_primary_images() or set_input_operables() first.",
            )

        # Create the processing pipeline, as necessary
        if self._processing_pipeline is None:
            self._initialize_processing_pipeline()

        # Get the next result
        return self._processing_pipeline.process_next()

    def _save_image(self, save_path_name_ext: str, image: CacheableImage, description: str):
        # check for overwrite
        if ft.file_exists(save_path_name_ext):
            if self.save_overwrite:
                lt.debug(
                    f"In SpotAnalaysis._save_image(): saving over existing {description} file {save_path_name_ext}"
                )
            else:
                lt.info(
                    f"In SpotAnalaysis._save_image(): not saving over existing {description} file {save_path_name_ext}"
                )
                return False

        # save the image
        _, _, save_ext = ft.path_components(save_path_name_ext)
        if save_ext in ["np", "npy"]:
            np.save(save_path_name_ext, image.nparray, allow_pickle=False)
        else:
            image.to_image().save(save_path_name_ext)

        return True

    def save_image(
        self,
        operable: SpotAnalysisOperable,
        save_dir: str = None,
        save_ext: str = "png",
        also_save_supporting_images=True,
        also_save_attributes_file=True,
    ):
        """Saves the primary image from the given operable.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The primary image to be saved
        save_dir : str, optional
            The directory to save the image to. Defaults to self.save_dir.
        save_ext : str, optional
            The type to save the image as. Can include any standard image type,
            as well as "np" or "npy" for numpy. Defaults to png.
        also_save_supporting_images : bool, optional
            If True, then any supporting images are saved next to the primary
            image with additional extensions to their names. Defaults to True.
        also_save_attributes_file : bool, optional
            If True, then any attributes from the SpotAnalysisOperable will
            be saved next to the primary image as a .txt file in JSON format.

        Returns
        -------
        image_path_name_ext: str | None
            The path to the saved image, if self.save_dir != None. None if the
            image wasn't saved, or if the image exists and save_overwrite is
            False, or if done processing.
        """
        save_dir = save_dir if save_dir != None else self.save_dir
        save_ext = save_ext.lstrip(".")
        image_path_name_ext = None

        # Save the resulting processed image
        if save_dir != None:
            # Get the original file name
            orig_image_name = ""
            orig_image_path, orig_image_name_ext = operable.get_primary_path_nameext()
            if orig_image_name_ext != None:
                _, orig_image_name, orig_image_ext = ft.path_components(orig_image_name_ext)
                orig_image_name += "_"

            # Get the output name of the file to save to
            sa_name_appendix = ft.convert_string_to_file_body(self.name)
            image_name = f"{orig_image_name}{sa_name_appendix}"
            if image_name in self.saved_names:
                image_name = f"{orig_image_name}{sa_name_appendix}_{self.save_idx}"
                self.save_idx += 1
            self.saved_names.add(image_name)
            image_path_name_ext = os.path.join(save_dir, f"{image_name}.{save_ext}")

            # Try to save the image
            if not self._save_image(image_path_name_ext, operable.primary_image, "primary image"):
                return

            # Save supporting images
            if also_save_supporting_images:
                supporting_images = operable.supporting_images
                for image_type in supporting_images:
                    # get the file name
                    supporting_image = supporting_images[image_type]
                    if supporting_image == None:
                        continue
                    supporting_image_name = ImageType(image_type).name
                    supporting_image_path_name_ext = os.path.join(
                        save_dir, f"{image_name}_{supporting_image_name}.{save_ext}"
                    )

                    # save the image
                    self._save_image(
                        supporting_image_path_name_ext, supporting_image, f"{supporting_image_name} supporting image"
                    )

            # Save associated attributes
            if also_save_attributes_file:
                attr_path_name_ext = os.path.join(save_dir, f"{image_name}.txt")
                attr = SpotAnalysisOperableAttributeParser(operable, self)
                attr.save(attr_path_name_ext, overwrite=self.save_overwrite)

        return image_path_name_ext

    def _initialize_processing_pipeline(self):
        self._processing_pipeline = SpotAnalysisPipeline(self.image_processors, iter(self.input_stream))

    def __iter__(self):
        self._initialize_processing_pipeline()
        return self

    def __next__(self) -> SpotAnalysisOperable:
        ret = self.process_next()
        if ret == None:
            raise StopIteration
        return ret


if __name__ == "__main__":
    # TODO write at least three input examples (eg laser, bcs, something else) to make sure this architecture isn't limited
    from opencsp.common.lib.cv.spot_analysis.image_processor import *

    lt.logger()

    collaborative_dir = opencsp_settings["opencsp_root_path"]["collaborative_dir"]
    experiment_dir = os.path.join(
        collaborative_dir, "Experiments", "2023-05-12_SpringEquinoxMidSummerSolstice", "2_Data", "BCS_data"
    )
    # indir = os.path.join(experiment_dir, "Measure_01", "20230512_071442 5W01_off", "Raw Images")
    indir = os.path.join(experiment_dir, "Measure_01", "20230512_071638 5W01_000_880_2890", "Raw Images")
    outdir = orp.opencsp_temporary_dir()

    image_processors = [
        PopulationStatisticsImageProcessor(min_pop_size=-1),
        EchoImageProcessor(),
        LogScaleImageProcessor(),
        ViewFalseColorImageProcessor(),
    ]
    sa = SpotAnalysis("BCS Test", image_processors, save_dir=outdir, save_overwrite=True)
    image_name_exts = ft.files_in_directory(indir)
    image_path_name_exts = [os.path.join(indir, image_name_ext) for image_name_ext in image_name_exts]
    sa.set_primary_images(image_path_name_exts)

    # Iterating through the returned results causes the images to be processed
    # by "pulling" on the last image processor, which pulls on the previous
    # image processor, etc up the chain.
    for i, result in enumerate(sa):
        # save out the images and associated attributes
        save_path = sa.save_image(result)
        if save_path is None:
            lt.warn(f"Failed to save image. Maybe SpotAnalaysis.save_overwrite is False? ({sa.save_overwrite=})")
        else:
            lt.info(f"Saved image to {save_path}")

        # check that the attributes were saved
        parser = SpotAnalysisOperableAttributeParser(result, sa)
        if i == 0:
            print(f"{parser.image_processors=}")
