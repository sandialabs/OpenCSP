from abc import abstractmethod
import copy
import dataclasses
import os
from typing import TypeVar

import numpy as np
from PIL import Image

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

image_processors_persistant_memory_total: int = 1 * pow(2, 30)  # default total of 1GiB
""" The amount of system memory that image processors are allowed to retain
as cache between calls to their 'run()' method. The most recently used results
are prioritized for maining in memory. Default (1 GiB). """


ImageLike = TypeVar('ImageLike', CacheableImage, np.ndarray, Image.Image)


class AbstractSpotAnalysisImageProcessor:
    """
    Class to perform one step of image processing before spot analysis is performed.

    This is an abstract class. Implementations can be found in the same
    directory. To create a new implementation, inherit from one of the existing
    implementations or this class. The most basic implementation need only
    implement the :py:meth:`_execute` method::

        def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
            raise NotImplementedError()
    """

    def __init__(self, name: str = None):
        """
        Parameters
        ----------
        name : str, optional
            The name to use for this image processor instance, or None to
            default to the class name. By default None.
        """

        # set default parameter values
        if name is None:
            name = self.__class__.__name__

        self._name = name
        # Name of this instance, probably the class name.
        self._num_images_processed: int = 0
        # The running total of the number of resulting images this instance
        # produced, since the last time that inputs were assigned.
        self.save_to_disk: int = False
        """ True to save results to the hard drive instead of holding them in
        memory. If False, then this is dynamically determined at runtime during
        image processing based on image_processors_persistant_memory_total. """
        self.cached = False
        """ True if we've ever cached the processed results of this processor to
        disk since processing was started. """
        self._my_tmp_dir = None
        # The directory where temporary images from this instance are saved to.
        self._tmp_images_saved = 0
        # How many images have been saved by this instance since it was created.
        self._clear_tmp_on_deconstruct = True
        # If true, then delete all png images in _my_tmp_dir, and then also
        # the directory if empty.
        self.operables_in_flight: list[SpotAnalysisOperable] = []
        """
        For most processors, :py:meth:`_execute` will return one resultant
        operable for each input operable. For these standard cases, this will
        contain one value during the :py:meth:`process_operable` method, and
        will be empty otherwise.
        
        Sometimes _execute() may return zero results. In this case, this value
        will contain all the operables passed to _execute() since the last time
        that _execute() returned at least one operable. These "in-flight"
        operables are remembered so that history can be correctly assigned to
        the resultant operables, once they become available.
        """

    @property
    def name(self) -> str:
        """Name of this processor"""
        return self._name

    def _register_processed_result(self, is_last: bool):
        """Updates internal variables for tracking the number of processed operables."""
        self._num_images_processed += 1

    def cache_images_to_disk_as_necessary(self):
        """
        Check memory usage and convert images to files (aka file path strings)
        as necessary in order to reduce memory usage.
        """
        allowed_memory_footprint = image_processors_persistant_memory_total
        CacheableImage.cache_images_to_disk_as_necessary(allowed_memory_footprint, self._get_tmp_path)

    def _get_save_dir(self):
        # Finds a temporary directory to save to for the processed output images from this instance.
        if self._my_tmp_dir == None:
            scratch_dir = os.path.join(orp.opencsp_scratch_dir(), "spot_analysis_image_processing")
            i = 0
            while True:
                dirname = self.name + str(i)
                self._my_tmp_dir = os.path.join(scratch_dir, dirname)
                if not ft.directory_exists(self._my_tmp_dir):
                    try:
                        os.makedirs(self._my_tmp_dir)
                        break  # success!
                    except FileExistsError:
                        # probably just created this directory in another thread
                        pass
                else:
                    i += 1
        return self._my_tmp_dir

    def _get_tmp_path(self) -> str:
        # Get the path+name+ext to save a cacheable image to, in our temporary
        # directory, in numpy format.
        #
        # Returns:
        #     path_name_ext: str
        #         Where to save the image.
        # get the path
        path_name_ext = os.path.join(self._get_save_dir(), f"{self._tmp_images_saved}.npy")
        self._tmp_images_saved += 1
        return path_name_ext

    def process_operable(
        self, input_operable: SpotAnalysisOperable, is_last: bool = False
    ) -> list[SpotAnalysisOperable]:
        """
        Executes this instance's image processing on a single given input
        primary image, with the supporting other images.

        This method is typically called from SpotAnalysis. Other users should
        consider using :py:meth:`process_images`.

        Parameters
        ----------
        input_operable : SpotAnalysisOperable
            The primary input image to be processed, other supporting images,
            and any necessary data.
        is_last : bool
            True if this is the last input image to be processed.

        Returns
        -------
        results : list[SpotAnalysisOperable]
            Zero, one, or more than one results from running image processing.
        """
        self.operables_in_flight.append(input_operable)

        try:
            ret: list[SpotAnalysisOperable] = self._execute(input_operable, is_last)
        except Exception as ex:
            lt.error(
                "Error in AbstractSpotAnalysisImageProcessor.process_operable(): "
                + f"encountered {ex.__class__.__name__} exception while processing image {input_operable.primary_image_source_path}"
            )
            raise
        if not isinstance(ret, list):
            lt.error_and_raise(
                TypeError,
                f"Error in AbstractSpotAnalysisImageProcessor.process_operable() ({self.name}): "
                + f"_execute() should return a list[SpotAnalysisOperable] but instead returned a {type(ret)}",
            )
        for operable in ret:
            if not isinstance(operable, SpotAnalysisOperable):
                lt.error_and_raise(
                    TypeError,
                    f"Error in AbstractSpotAnalysisImageProcessor.process_operable() ({self.name}): "
                    + f"expected return value from _execute() to be list[SpotAnalysisOperable] but is instead list[{type(operable)}]!",
                )

        # record processed results
        if is_last:
            self._register_processed_result(is_last)

        # register the new operable's "previous_operable" value based on operables_in_flight
        for i in range(len(ret)):
            # Don't register the returned operable as its own previous operable.
            # This shouldn't happen with image processors that follow the
            # convention of producing a new operable as their output, but it is
            # possible.
            # Do nothing image processors can also return the same input
            # operable as output, such as for the EchoImageProcessor.
            ret_is_in_flight = np.any([ret[i] is op_in_flight for op_in_flight in self.operables_in_flight])
            if not ret_is_in_flight:
                ret[i] = dataclasses.replace(ret[i], previous_operables=(copy.copy(self.operables_in_flight), self))

        # de-register any operable on which we're waiting for results
        if len(ret) > 0 or is_last:
            self.operables_in_flight.clear()

        # release memory by cacheing images to disk
        self.cache_images_to_disk_as_necessary()

        return ret

    def process_images(self, images: list[ImageLike]) -> list[ImageLike]:
        """
        Processes the given images with this processor and returns 0, 1, or more
        than 1 resulting images.

        This method is provided for convenience, to allow for use of the spot
        analysis image processors as if they were simple functions. The
        following is an example of a more standard use of an image processor::

            processors = [
                EchoImageProcessor(),
                LogScaleImageProcessor(),
                FalseColorImageProcessor()
            ]
            spot_analysis = SpotAnalysis("Log Scale Images", processors)
            spot_analysis.set_primary_images(images)
            results = [result for result in spot_analysis]

        Parameters
        ----------
        images : list[CacheableImage  |  np.ndarray  |  Image.Image]
            The images to be processed.

        Returns
        -------
        list[CacheableImage | np.ndarray | Image.Image]
            The resulting images after processing. Will be the same type as the
            first of the input images.
        """
        # import here to avoid cyclic imports
        from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis

        # set up a new spot analysis instance
        spot_analysis = SpotAnalysis(self.name, [self])
        spot_analysis.set_primary_images(images)

        # evaluate
        ret: list[CacheableImage] = []
        for result in spot_analysis:
            # return the same type
            if isinstance(images[0], CacheableImage):
                ret.append(result)
            elif isinstance(images[0], np.ndarray):
                ret.append(result.nparray)
            elif isinstance(images[0], Image.Image):
                ret.append(result.to_image())
            else:
                lt.error_and_raise(
                    TypeError,
                    "Error in AbstractSpotAnalysisImageProcessor.process_images(): "
                    + f"expected input to be one of CacheableImage, Numpy array, or Pillow image, but is instead of type {type(images[0])}",
                )

        return ret

    @abstractmethod
    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        """Evaluate an input primary image (and other images/data), and generate the output processed image(s) and data.

        This is the actual image processing method implemented by each image
        processing class. It is called from :py:meth:`process_operable`.

        In most cases, the resulting operable(s) should be different from the
        input operable(s). This keeps operables (mostly) immutable and makes
        following the chain of logic easier for debugging an image processing
        pipeline. The exception is for do-nothing image processors, such as the
        :py:class:`EchoImageProcessor` or the
        :py:class:`SaveToFileImageProcessor`, which return the same operable
        instance that is given as the input. Note that do-nothing image
        processors won't be added to the operable's
        :py:attr:`.previous_operables` history.

        Parameters
        ----------
        primary_image : SpotAnalysisOperable
            The original image that should be processed, any supporting images
            to be utilized during processing, and other necessary data.
        is_last : bool
            True when this is the last image to be processed

        Returns
        -------
        results : list[SpotAnalysisOperable]
            The processed primary image(s).

            If zero operables are returned, then it will be assumed that more
            than one input operable is required for processing and _execute will
            be called again with the next operable.

            If one or more operables are returned, then they will be made
            available immediately from the process_operable method.
        """
        pass

    def __len__(self) -> int:
        # Get the current number of processed output images from this instance.
        #
        # This number will potentially be increasing with every call to
        # process_operable() or _execute().
        #
        # This will return 0 immediately after being created.
        return self._num_images_processed
