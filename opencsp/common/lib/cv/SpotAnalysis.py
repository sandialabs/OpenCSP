import numpy as np
import os
from PIL import Image as Image
from typing import Iterator

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.VisualizationCoordinator import VisualizationCoordinator
import opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor as asaip

from opencsp import opencsp_settings

# from opencsp.common.lib.cv.spot_analysis.image_processor import * # I suggest importing these dynamically as needed, to reduce startup time
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
from opencsp.common.lib.cv.spot_analysis.ImagesStream import ImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import ImageType, SpotAnalysisImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperablesStream import SpotAnalysisOperablesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperableAttributeParser import SpotAnalysisOperableAttributeParser
import opencsp.common.lib.render.VideoHandler as vh
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class SpotAnalysis(Iterator[tuple[SpotAnalysisOperable]]):
    """Spot Analysis class for characterizing beams of light.

    This is meant to be a general beam characterization tool that receives as
    input an image (or video) of a single beam, and generates annotations and
    numerical statistics as output. In general, there is an image processing
    step, an image analysis step, and a visualization step. Any number of
    processors and analyzers can be chained together to form a pipeline for
    evaluation. Once defined, a SpotAnalysis instance can be used to evaluate
    one image/video per evaluation.

    A list of possible use cases for this class include::

        1. BCS
        2. BCS-based heliostat calibration
        3. Vertical laser characterization
        4. Laser beam spread characterization
        5. 2f returned spot analysis
        6. Motion characterization video analysis
        7. Wind BCS video trajectory analysis
        8. Laser-based cross-check
        9. Earth mover's metric for comparisons (scipy has a method, also maaaaaybe look at https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf)
        10. Enclosed power

    These are our expected outputs::

        - Annotated images
        - Visualizations (ex 2d or 3d heat-flux map, side-by-side comparisons)
        - Accompanying statistics INI files
        - Other details text file (heliostat name and location, date and time, heliostat operator, etc...)
        - Powerpoint slides

    The features necessary to support these use cases include::

        a. NULL image subtraction (TODO)
        b. Ambient and gradient light subtraction (TODO)
        c. Lens correction (TODO)
        d. Intensity per pixel generation (TODO)
        e. Centroid calculation (1-8) (TODO)
        f. FWHM analysis (1,2,4,5,10) (TODO)
        g. Astigmatism identification (1-5,10) (TODO)
        h. Automatic beam identification (2) (TODO)
        i. Video-to-frame extraction (6,7) (TODO)
        j. Video streaming (6,7) (TODO)
        k. Networked image collection (2) (TODO)
        l. Color-aware image subtraction (3,4,5,8) (TODO)
        m. Color-aware light subtraction (3,4,5,8) (TODO)
        n. Time series analysis (6,7) (TODO)
        o. Laser placement and pointing angle as image metadata (5,8) (TODO)
        p. Laser placement and pointing angle as separate input (5,8) (TODO)
        q. Earth mover's metric (9) (TODO)
        r. Multi-image comparison (6,7,9) (TODO)
        s. Intensity-to-power mapping (1,4,9,10) (TODO)
        t. Noise removal and quantification (TODO)
        u. Multi-image averaging over time (1-5,8-10) (TODO)
        v. Fiducial identification (5,r,u,w,x) (TODO)
        w. Screen/camera plane homography (TODO)
        x. Self-PnP (TODO)
        y. Annotate power envelopes (TODO)
        z. Orthorectify wrt. target (TODO)
        aa. Orthorectify wrt. beam (TODO)
        ab. Spot radius in mrad, at angle Beta relative to centroid (TODO)
        ac. Spot radius in mrad, at a half angle relative to the heliostat location (TODO)
        ad. Beam radius in mrad, at angle Beta relative to centroid (TODO)
        ae. Beam radius in mrad, at a half angle relative to the heliostat location (TODO)
        af. Cropping (TODO)
        ag. Over and under exposure detection (TODO)
        ah. Tagging images as NULL images (TODO)
        ai. Filters (gaussian, box, etc) (TODO)
        aj. Peak value pixel identification (TODO)
        ak. Logorithmic scaling (LogScaleImageProcessor)
        al. False color visualization (FalseColorImageProcessor)
        am. Over/under exposure visualization (TODO)

    The inputs to support these features include::

        - primary image or images (TODO)
        - reference image (g,h,v) (TODO)
        - null image (a,b,l,m,t) (TODO)
        - comparison image (r) (TODO)
        - background selection (b,l,m) (TODO)
        - camera lens characterization (c) (TODO)
        - image server network address (k) (TODO)
        - laser color/wavelength (l,m) (TODO)
        - camera color characterization (l,m) (TODO)
        - laser placement and pointing angle (p) (TODO)
        - intensity-to-power mapping (s) (TODO)
        - multiple primary images, or a primary video (u,x) (TODO)
        - fiducial definition and location (v) (TODO)
        - manual 3D point identification from 2D images (w) (TODO)
    """

    def __init__(
        self,
        name: str,
        image_processors: list[asaip.AbstractSpotAnalysisImagesProcessor],
        save_dir: str = None,
        save_overwrite=False,
    ):
        """
        Parameters
        ----------
        name: str
            The name of this instance. For example, this could be one of the use
            cases listed above.
        image_processors: list[AbstractSpotAnalysisImagesProcessor]
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
        self.image_processors: list[asaip.AbstractSpotAnalysisImagesProcessor] = []
        """ List of processors, one per step of the analysis. The output from
        each processor can include one or more images (or numeric values), and
        is made available to all subsequent processors. """
        self._results_iter: Iterator[SpotAnalysisOperable] = None
        """ The returned value from iter(self.image_processors[-1]). Initialized
        on the first call to process_next(). """
        self._prev_result: SpotAnalysisOperable = None
        """ The previously returned result. """
        self.input_stream: SpotAnalysisOperablesStream = None
        """ The images to be processed. """
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
        """ Shows the same image from all visualization processors at the same time. """

        self.set_image_processors(image_processors)

    def set_image_processors(self, image_processors: list[asaip.AbstractSpotAnalysisImagesProcessor]):
        self.image_processors = image_processors

        # chain the image processors together
        for i, image_processor in enumerate(self.image_processors):
            if i == 0:
                continue
            image_processor.assign_inputs(self.image_processors[i - 1])

        # register the visualization processors
        self.visualization_coordinator.clear()
        self.visualization_coordinator.register_visualization_processors(image_processors)

        # limit the amount of memory that image processors utilize
        from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessorLeger import (
            image_processors_persistant_memory_total,
        )

        mem_per_image_processor = image_processors_persistant_memory_total / len(self.image_processors)
        for image_processor in self.image_processors:
            image_processor._allowed_memory_footprint = mem_per_image_processor

        # assign the input stream to the first image processor
        if self.input_stream != None:
            self._assign_inputs(self.input_stream)

    @staticmethod
    def _images2stream(
        images: list[str] | list[np.ndarray] | vh.VideoHandler | ImagesStream,
    ) -> ImagesStream | ImagesIterable:
        if isinstance(images, ImagesStream):
            return images
        else:
            return ImagesIterable(images)

    def _assign_inputs(self, input_operables: Iterator[SpotAnalysisOperable]):
        if not isinstance(input_operables, SpotAnalysisOperablesStream):
            input_operables = SpotAnalysisOperablesStream(input_operables)
        self.input_stream = input_operables
        self._prev_result = None
        self.image_processors[0].assign_inputs(self.input_stream)

    def set_primary_images(self, images: list[str] | list[np.ndarray] | vh.VideoHandler | ImagesStream):
        """Assigns the images of the spot to be analyzed, in preparation for process_next().

        See also: set_input_operables()"""
        primary_images = self._images2stream(images)
        images_stream = SpotAnalysisImagesStream(primary_images, {})
        self._assign_inputs(SpotAnalysisOperablesStream(images_stream))

    def set_input_operables(
        self, input_operables: SpotAnalysisOperablesStream | list[SpotAnalysisOperable] | Iterator[SpotAnalysisOperable]
    ):
        """Assigns primary and supporting images, and other necessary data, in preparation for process_next().

        See also: set_primary_images()"""
        self._assign_inputs(input_operables)

    def set_default_support_images(self, support_images: dict[ImageType, CacheableImage]):
        """Provides extra support images for use during image processing, as a
        default for when the support images are not otherwise from the input
        operables. Note that this does not include the primary images or other
        data."""
        self.input_stream.set_defaults(support_images, self.input_stream.default_data)

    def set_default_data(self, operable: SpotAnalysisOperable):
        """Provides extra data for use during image processing, as a default
        for when the data is not otherwise from the input operables. Note that
        this does not include the primary or supporting images."""
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
        if self._results_iter is None:
            self._results_iter = iter(self.image_processors[-1])

        # Release memory from the previous result
        if self._prev_result is not None:
            self.image_processors[-1].cache_image_to_disk_as_necessary(self._prev_result)
            self._prev_result = None

        # Attempt to get the next image. Raises StopIteration if there are no
        # more results available.
        try:
            result = next(self._results_iter)
        except StopIteration:
            return None

        self._prev_result = result
        return result

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
            it.numpy_to_image(image.nparray).save(save_path_name_ext)

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
            orig_image_path_name = ""
            if operable.primary_image.source_path != None:
                _, orig_image_path_name, _ = ft.path_components(operable.primary_image.source_path)
                orig_image_path_name += "_"

            # Get the output name of the file to save to
            sa_name_appendix = ft.convert_string_to_file_body(self.name)
            image_name = f"{orig_image_path_name}{sa_name_appendix}"
            if image_name in self.saved_names:
                image_name = f"{orig_image_path_name}{sa_name_appendix}_{self.save_idx}"
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

    def __iter__(self):
        self.save_idx = 0
        self._prev_result = None
        return self

    def __next__(self):
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
        FalseColorImageProcessor(),
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
