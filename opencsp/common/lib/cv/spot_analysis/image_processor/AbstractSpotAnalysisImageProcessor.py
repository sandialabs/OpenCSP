from abc import abstractmethod
import copy
from typing import Callable, Iterator, Union

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.ImagesIterable import ImagesIterable
from opencsp.common.lib.cv.spot_analysis.ImagesStream import ImagesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperablesStream import SpotAnalysisOperablesStream
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisImagesStream import SpotAnalysisImagesStream
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessorLeger import (
    AbstractSpotAnalysisImagesProcessorLeger,
)
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.typing_tools as tt


class AbstractSpotAnalysisImagesProcessor(Iterator[SpotAnalysisOperable], AbstractSpotAnalysisImagesProcessorLeger):
    """Class to perform one step of image processing before spot analysis is performed.

    This is an abstract class. Implementations can be found in the same
    directory. To create a new implementation, inherit from one of the existing
    implementations or this class.
    """

    def __init__(self, name: str):
        AbstractSpotAnalysisImagesProcessorLeger.__init__(self, name)
        self.next_item: SpotAnalysisOperable = None
        """ The next fetched item from the input_iter, held in anticipation for the
        following call to __next__(). None if the next item hasn't been fetched yet,
        or we've reached the end of the input_iter. """
        self.inmem_inputs: list[SpotAnalysisOperable] = []
        """ Operables retrieved from from input_iter that have been _execute()'ed
        on, but _execute() hasn't returned as many results as the number of times
        that it's been called. In other words::
        
            len(inmem_inputs) == num_executes - len(cummulative_processed_results) """
        self.results_on_deck: list[SpotAnalysisOperable] = []
        """ Sometimes _execute() may return multiple results. In this case,
        we hold on to the processed operables and return only one of them per
        iteration in __next__(). """
        self._on_image_processed: list[Callable[[SpotAnalysisOperable]]] = []
        """ A list of callbacks to be evaluated when an image is finished processing. """

    def run(
        self,
        operables: (
            ImagesIterable
            | ImagesStream
            | SpotAnalysisImagesStream
            | Union['AbstractSpotAnalysisImagesProcessor', list[SpotAnalysisOperable], Iterator[SpotAnalysisOperable]]
        ),
    ) -> list[CacheableImage]:
        """Performs image processing on the input images."""
        if isinstance(operables, (ImagesIterable, ImagesStream)):
            operables = SpotAnalysisImagesStream(operables)
        if isinstance(operables, SpotAnalysisImagesStream):
            operables = SpotAnalysisOperablesStream(operables)
        self.assign_inputs(operables)
        for result in self:
            pass
        return copy.copy(self.all_processed_results)

    def process_image(self, input_operable: SpotAnalysisOperable, is_last: bool = False) -> list[SpotAnalysisOperable]:
        """Should probably not be called by external classes. Evaluate this instance as an iterator instead.

        Executes this instance's image processing on a single given input
        primary image, with the supporting other images. If enough images have
        been processed as to exceed this instance's memory limitations, then all
        processed primary images will be stored to disk instead of being kept in
        memory. The resulting processed images or paths to said images will be
        recorded in self.cummulative_processed_results and can be accessed by
        self.all_results after all input images have been processed (aka when
        is_last=True).

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
        if self.cummulative_processed_results == None:
            self.initialize_cummulative_processed_results()

        self.inmem_inputs.append(input_operable)
        ret: list[SpotAnalysisOperable] = self._execute(input_operable, is_last)
        if not isinstance(ret, list):
            lt.error_and_raise(
                TypeError,
                f"Error in AbstractSpotAnalysisImageProcessor.process_image() ({self.name}): "
                + f"_execute() should return a list[SpotAnalysisOperable] but instead returned a {type(ret)}",
            )
        for operable in ret:
            if not isinstance(operable, SpotAnalysisOperable):
                lt.error_and_raise(
                    TypeError,
                    f"Error in AbstractSpotAnalysisImageProcessor.process_image() ({self.name}): "
                    + f"expected return value from _execute() to be list[SpotAnalysisOperable] but is instead list[{type(operable)}]!",
                )

        # record processed results
        for result in ret:
            self.register_processed_result(result, is_last)

        # execute any registered callbacks
        for operable in ret:
            for callback in self._on_image_processed:
                callback(operable)

        # release memory by cacheing images to disk
        for operable in ret:
            # release one per returned value
            self.cache_image_to_disk_as_necessary(self.inmem_inputs.pop(0))
        if is_last:
            # release all
            while len(self.inmem_inputs) > 0:
                self.cache_image_to_disk_as_necessary(self.inmem_inputs.pop(0))

        return ret

    @abstractmethod
    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        """Evaluate an input primary image (and other images/data), and generate the output processed image(s) and data.

        The actual image processing method. Called from process_image().

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

    def __iter__(self):
        if self.all_processed_results != None:
            # We must have already finished processing all input images, either
            # through the run() method or by simply having iterated through them
            # all.
            if not self.finished_processing:
                lt.error_and_raise(
                    RuntimeError,
                    f"Programmer error in AbstractSpotAnalaysisImageProcess.__iter__: "
                    + f"self.all_processed_results != None but {self.finished_processing=}!",
                )
            raise StopIteration
        else:
            if self.input_iter != None:
                # We must be iterating through the input images already.
                return self
            else:
                # We must be iterating through the input images and haven't
                # started processing them yet.
                # Or, we're restarting iteration.
                self.assign_inputs(self._original_operables)  # initializes the leger
                self.input_iter = iter(self._original_operables)
                self.inmem_inputs = []
                try:
                    self.next_item = next(self.input_iter)
                except StopIteration:
                    self.next_item = {}
                    self.all_processed_results = []
                    self.finished_processing = True
                return self

    def __next__(self):
        """Get the next processed image and it's support images and data. Since
        this is only utilized when processing as an iterator instead of by using
        the run() method, then calling this method will cause one or more input
        image(s) to be fetched so that it can be executed upon."""
        # Check if we already have a result from a previous iteration staged and ready to be returned
        if len(self.results_on_deck) > 0:
            return self.results_on_deck.pop(0)
        else:
            if self.finished_processing:
                raise StopIteration

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
            # Get the the values for:
            # - the current input operable
            # - the input operable for the next cycle
            # - and the value of is_last
            input_operable = self.next_item
            if input_operable == None:
                lt.error_and_raise(
                    RuntimeError,
                    f"Programmer error in AbstractSpotAnalysisImageProcessor.__next__() ({self.name}): "
                    + "input_operable should never be None but it is!",
                )
            try:
                self.next_item = next(self.input_iter)
            except StopIteration:
                self.next_item = None
                is_last = True

            output_operables = self.process_image(input_operable, is_last)
            if len(output_operables) > 0:
                break

        if is_last:
            self.finished_processing = True

        # Did execute return any results?
        if len(output_operables) == 0:
            if not is_last:
                lt.error_and_raise(
                    RuntimeError,
                    f"Programmer error in SpotAnalysisAbstractImagesProcessor.__next__() ({self.name}): "
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

    @tt.strict_types
    def get_processed_image_save_callback(
        self, dir: str, name_prefix: str = None, ext="jpg"
    ) -> Callable[[SpotAnalysisOperable], str]:
        """Saves images to the given directory with the file name
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
            result from process_image().
        """
        self._on_image_processed.append(callback)
