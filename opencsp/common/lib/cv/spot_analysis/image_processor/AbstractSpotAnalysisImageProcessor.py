from abc import abstractmethod
import copy
import dataclasses
import os
import sys
from typing import Callable, Iterable, Iterator, Union

import numpy as np
from PIL import Image

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
from opencsp.common.lib.cv.spot_analysis.ImagesStream import ImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperablesStream import _SpotAnalysisOperablesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import SpotAnalysisImagesStream
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.typing_tools as tt

image_processors_persistant_memory_total: int = 1 * pow(2, 30)  # default total of 1GiB
""" The amount of system memory that image processors are allowed to retain
as cache between calls to their 'run()' method. The most recently used results
are prioritized for maining in memory. Default (1 GiB). """


class AbstractSpotAnalysisImageProcessor(Iterator[SpotAnalysisOperable]):
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
        self.input_operables: Iterable[SpotAnalysisOperable] = None
        """ The input iterable given to assign_inputs. """
        self.input_iter: Iterator[SpotAnalysisOperable] = None
        """ Iterator over the input images. None until the iterator has been primed. """
        self.is_iterator_primed = False
        """ True if __iter__ has been called since assign_inputs. False otherwise. """
        self._finished_processing = False
        # True if we've finished iterating over all input images. This gets set
        # when we get a StopIteration error from next(input_iter).
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
        self.next_item: SpotAnalysisOperable = None
        """
        The next fetched item from the input_iter, held in anticipation for the
        following call to __next__(). None if we haven't retrieved the next
        value from the input_iter yet.
        """
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
        self.results_on_deck: list[SpotAnalysisOperable] = []
        """
        Sometimes :py:meth:`_execute` may return multiple results. In this case,
        we hold on to the processed operables and return only one of them per
        iteration in __next__(). This gaurantees that each image processor in
        the chain consumes and produces single images.
        """
        self._on_image_processed: list[Callable[[SpotAnalysisOperable]]] = []
        # A list of callbacks to be evaluated when an image is finished processing.

        # initialize some of the state
        self.assign_inputs([])

    @property
    def name(self) -> str:
        """Name of this processor"""
        return self._name

    @property
    def _finished_processing(self):
        return self.__finished_processing

    @_finished_processing.setter
    def _finished_processing(self, val):
        self.__finished_processing = val

    @property
    def finished(self):
        """True if we've finished iterating over all input images. This gets set
        when we get a StopIteration error from next(input_iter)."""
        return self.__finished_processing

    def assign_inputs(
        self,
        operables: Union[
            "AbstractSpotAnalysisImageProcessor", list[SpotAnalysisOperable], Iterator[SpotAnalysisOperable]
        ],
    ):
        """
        Register the input operables to be processed either with the run()
        method, or as an iterator.

        Parameters
        ----------
        operables : Union[ AbstractSpotAnalysisImageProcessor, list[SpotAnalysisOperable], Iterator[SpotAnalysisOperable] ]
            The operables to be processed.
        """
        # initialize the state for a new set of inputs
        self.input_operables = operables
        self._num_images_processed = 0
        self._finished_processing = False
        self.next_item = None
        self.is_iterator_primed = False

        # check for an empty list [], indicating no inputs
        if isinstance(operables, list):
            if len(operables) == 0:
                self.input_iter = None
                self._finished_processing = True
                self.operables_in_flight.clear()
                self.results_on_deck.clear()

    def _register_processed_result(self, is_last: bool):
        """Updates internal variables for tracking the number of processed operables."""
        self._num_images_processed += 1
        if is_last:
            self._finished_processing = True

    def cache_images_to_disk_as_necessary(self):
        """Check memory usage and convert images to files (aka file path
        strings) as necessary in order to reduce memory usage."""
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

    def _save_image(self, im: CacheableImage, idx_list: list[int], dir: str, name_prefix: str = None, ext="jpg") -> str:
        # Saves the given image to the given path.
        #
        # Parameters
        # ----------
        # im : CacheableImage
        #     The image to be saved.
        # idx_list : list[int]
        #     Length-1 list where idx_list[0] is the count of images saved with
        #     this method. Used for naming the saved images. This value is updated
        #     as part of the execution of this method.
        # dir : str
        #     The directory to save the image to.
        # name_prefix : str, optional
        #     A prefix to prepend to the image name, by default empty string
        # ext : str, optional
        #     The extension/type to save the image with, by default "jpg"
        #
        # Returns
        # -------
        # str
        #     The path/name.ext of the newly saved image.
        idx = idx_list[0]
        image_name = ("" if name_prefix == None else f"{name_prefix}_") + f"SA_preprocess_{self.name}{idx}"
        image_path_name_ext = os.path.join(dir, image_name + "." + ext)
        lt.debug("Saving SpotAnalysis processed image to " + image_path_name_ext)
        im.to_image().save(image_path_name_ext)
        idx_list[0] = idx + 1
        return image_path_name_ext

    def run(
        self,
        operables: (
            ImagesIterable
            | ImagesStream
            | SpotAnalysisImagesStream
            | list[SpotAnalysisOperable]
            | Iterator[SpotAnalysisOperable]
            | Union["AbstractSpotAnalysisImageProcessor"]
        ),
    ) -> list[SpotAnalysisOperable]:
        """
        Performs image processing on the input operables and returns the results.

        This is provided as a convenience method. The more typical way to use
        this class is to create a SpotAnalysis instance, assign this image
        processor to that instance, and then iterate over the results.

        See also: :py:meth:`process_images` as another convenience method.

        Parameters
        ----------
        operables : ImagesIterable  |  ImagesStream  |  SpotAnalysisImagesStream  |  list[SpotAnalysisOperable]  |  Iterator[SpotAnalysisOperable]  |  Union[AbstractSpotAnalysisImageProcessor]
            The input operables to be processed. If these are images, then they
            will be wrapped in a SpotAnalysisOperablesStream.

        Returns
        -------
        list[SpotAnalysisOperable]
            The resulting operables after processing.
        """
        if isinstance(operables, (ImagesIterable, ImagesStream)):
            operables = SpotAnalysisImagesStream(operables)
        if isinstance(operables, SpotAnalysisImagesStream):
            operables = _SpotAnalysisOperablesStream(operables)
        self.assign_inputs(operables)
        ret = [result for result in self]
        return ret

    def process_operable(
        self, input_operable: SpotAnalysisOperable, is_last: bool = False
    ) -> list[SpotAnalysisOperable]:
        """Should probably not be called by external classes. Evaluate this instance as an iterator instead.

        Executes this instance's image processing on a single given input
        primary image, with the supporting other images.

        When processed with the run() method, this function will be called for
        all input images.

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

        # execute any registered callbacks
        for operable in ret:
            for callback in self._on_image_processed:
                callback(operable)

        # register the new operable's "previous_operable" value based on operables_in_flight
        for i in range(len(ret)):
            # Don't register the returned operable as its own previous operable.
            # This shouldn't happen with image processors that follow the
            # convention of producing a new operable as their output, but it is
            # possible.
            # Do nothing image processors can also return the same input
            # operable as output, such as for the EchoImageProcessor.
            if ret[i] not in self.operables_in_flight:
                ret[i] = dataclasses.replace(ret[i], previous_operables=(copy.copy(self.operables_in_flight), self))

        # de-register any operable on which we're waiting for results
        if len(ret) > 0 or is_last:
            self.operables_in_flight.clear()

        # release memory by cacheing images to disk
        self.cache_images_to_disk_as_necessary()

        return ret

    def process_images(self, images: list[CacheableImage | np.ndarray | Image.Image]) -> list[CacheableImage]:
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
        list[CacheableImage]
            The resulting images after processing.
        """
        # import here to avoid cyclic imports
        from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis

        spot_analysis = SpotAnalysis(self.name, [self])
        spot_analysis.set_primary_images(images)
        ret: list[CacheableImage] = []
        for result in spot_analysis:
            ret += result.get_all_images(supporting=False)

        return ret

    @abstractmethod
    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        """Evaluate an input primary image (and other images/data), and generate the output processed image(s) and data.

        This is the actual image processing method implemented by each image
        processing class. It is called from :py:meth:`process_operable`.

        In most cases, the resulting operable(s) should be different from the
        input operable(s). This keeps operables (mostly) immutable and makes
        following the chain of logic easier for debugging an image processing
        pipeline. The exception is for do-nothing image processors, such as for
        the EchoImageProcessor. Note that image processors that return the same
        input operable as an output won't be added to the operable's
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

            If one operable is returned, then it will be made available
            immediately to the next image processor in the spot analysis
            pipeline.

            If more than one operable is returned, then the first will be made
            available to the next image processor and the rest will be staged,
            to be made available one at a time. This means that this _execute()
            method won't be called again until all returned operables have been
            exhausted.
        """
        pass

    def __len__(self) -> int:
        # Get the current number of processed output images from this instance.
        #
        # This number will potentially be increasing with every call to
        # process_operable() or _execute().
        #
        # This will return 0 immediately after a call to assign_inputs().
        return self._num_images_processed

    def __iter__(self):
        if not self.is_iterator_primed:
            # Operables have been assigned via assign_inputs(), but we haven't
            # started iterating yet. We need to prime self.input_iter and
            # self.next_item before returning.
            try:
                input_iter: Iterator[SpotAnalysisOperable] = iter(self.input_operables)
                self.input_iter = input_iter
                self.is_iterator_primed = True
                return iter(self)
            except StopIteration:
                self.assign_inputs([])
                raise
        elif self.finished:
            # We must have already finished processing all input images, either
            # through the run() method or by simply having iterated through them
            # all.
            return self
        elif self.input_iter != None:
            # We must be iterating through the input images already.
            return self
        else:
            lt.error_and_raise(
                RuntimeError,
                "Error in AbstractSpotAnalysisImageProcessor.__iter__(): "
                + "unexpected state encountered, "
                + "expected is_iterator_primed to be False, finished to be True, or input_iter to be set, but "
                + f"{self.is_iterator_primed=}, {self.finished=}, and {self.input_iter=}",
            )

    def __next__(self):
        # Get the next processed image and it's support images and data. Since
        # this is only utilized when processing as an iterator instead of by using
        # the run() method, then calling this method will cause one or more input
        # image(s) to be fetched so that it can be executed upon.
        if not self.has_next():
            raise StopIteration

        # Check if we already have a result from a previous iteration staged and ready to be returned
        if len(self.results_on_deck) > 0:
            return self.results_on_deck.pop(0)

        # Process the next image.
        #
        # It is possible that it requires more than one input image before a
        # result becomes available, so keep executing on input images until at
        # least one result is returned.
        #
        # It is also possible that more than one result is returned, in which
        # case we need to stage the extra processed results.
        is_last = False
        output_operables: list[SpotAnalysisOperable] = []
        while not is_last:
            # Get the operable to be processed
            self.fetch_input_operable()
            input_operable = self.next_item
            self.next_item = None
            if input_operable == None:
                lt.error_and_raise(
                    RuntimeError,
                    f"Programmer error in AbstractSpotAnalysisImageProcessor.__next__() ({self.name}): "
                    + "input_operable should never be None but it is!",
                )

            # Determine if this is the last operable we're going to be receiving
            is_last = not self.has_next()

            # Process the input operable and get the output results
            output_operables = self.process_operable(input_operable, is_last)

            # Stop if we have results to return
            if len(output_operables) > 0:
                break

        if is_last:
            self._finished_processing = True

        # Did execute return any results?
        if len(output_operables) == 0:
            if not is_last:
                lt.error_and_raise(
                    RuntimeError,
                    f"Programmer error in AbstractSpotAnalysisImageProcessor.__next__() ({self.name}): "
                    + "as long as there are input image available (aka is_last is False) we should keep executing until "
                    + f"at least one result becomes available, but {is_last=} and {len(output_operables)=}",
                )
            raise StopIteration

        # Did execute return exactly one result?
        elif len(output_operables) == 1:
            ret = output_operables[0]

        # Did execute return more than one result?
        else:
            ret = output_operables[0]
            self.results_on_deck = output_operables[1:]

        return ret

    def has_next(self) -> bool:
        """
        Returns True if this image processor will return another result when
        __next__() is called, or False if it won't. This might result in a call
        to the prior image processor's __next__() method.
        """
        if self.finished:
            return False
        if len(self.results_on_deck) > 0:
            return True
        if (self.input_operables is None) or (
            isinstance(self.input_operables, list) and len(self.input_operables) == 0
        ):
            return False
        if self.next_item is not None:
            return True
        if hasattr(self.input_operables, "has_next"):
            return self.input_operables.has_next()

        # We tried every other method of determining if there is a next value
        # available from the input iterable. The only possible way left to
        # determine if there is a next value available is to try and retrieve
        # it.
        if not self.is_iterator_primed:
            iter(self)  # primes the iterator
        try:
            self.fetch_input_operable()
        except StopIteration:
            pass
        return self.next_item is not None

    def fetch_input_operable(self):
        """
        Retrieves the operable to operate on. Populates self.next_item.

        The input operable might be the prefetched self.next_item or it might
        need to be requested as the next result from the input_iter.

        Raises
        ------
        StopIteration:
            The input_iter doesn't have any more results available
        """
        # check for invalid state
        if self.finished:
            lt.error_and_raise(
                RuntimeError,
                "Error in AbstractSpotAnalysisImageProcessor.get_input_operable() ({self.name}): "
                + "Trying to retrieve an operable in an invalid state.",
            )

        # get the operable, as necessary
        if self.next_item is None:
            try:
                self.next_item = next(self.input_iter)
            except StopIteration:
                raise

    @tt.strict_types
    def get_processed_image_save_callback(
        self, dir: str, name_prefix: str = None, ext="jpg"
    ) -> Callable[[SpotAnalysisOperable], str]:
        """
        Saves images to the given directory with the file name
        "[name_prefix+'_']SA_preprocess_[self.name][index].[ext]". The returned
        function takes an ndarray image as input, and returns the saved
        path_name_ext.

        This method is designed to be used as a callback with self.on_image_processed().
        """
        idx_list = [0]
        return lambda operable: self._save_image(operable.primary_image, idx_list, dir, name_prefix, ext)

    def on_image_processed(self, callback: Callable[[SpotAnalysisOperable], None]):
        """Registers the given callback, to be evaluated for each processed
        image returned from _evaluate().

        Parameters
        ----------
        callback : Callable[[SpotAnalysisOperable], None]
            The function to be evaluated. Requires one input, which will be the
            result from process_operable().
        """
        self._on_image_processed.append(callback)
