from typing import Iterator

from PIL import Image as Image

from opencsp.common.lib.cv.spot_analysis.image_processor import AbstractSpotAnalysisImageProcessor
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.tool.log_tools as lt


class SpotAnalysisPipeline:
    """
    Manages the propogation of operables, starting from the input operables,
    through the image processors, to the output results.
    """

    def __init__(
        self, image_processors: list[AbstractSpotAnalysisImageProcessor], input_stream: Iterator[SpotAnalysisOperable]
    ):
        """
        Parameters
        ----------
        image_processors: list[AbstractSpotAnalysisImageProcessor]
            The list of processors that will be pipelined together and used to
            process input images.
        """
        self.image_processors: list[AbstractSpotAnalysisImageProcessor] = image_processors
        """ List of processors, one per step of the analysis. The output from
        each processor can include one or more operables, and is made available
        to all subsequent processors. """
        self._prev_result: SpotAnalysisOperable = None
        # The most recently returned result.
        self._intermediary_operables: dict[AbstractSpotAnalysisImageProcessor, list[SpotAnalysisOperable]] = None
        # The resulting operables that have been returned from processors but
        # are not the final results. This is used for book keeping for both
        # standard processors that return a single operable, and for processors
        # that return more than one result from their _execute method.
        self.input_stream: Iterator[SpotAnalysisOperable] = input_stream
        """ The current iterator, as from a call to iter(SpotAnalysisOperablesStream). """
        self.__next_input_operable: SpotAnalysisOperable = None
        # The next operable to be processed by the first processor in the
        # pipeline. We retrieve this operable ahead of time so that we know
        # whether to set the is_last flag when calling process_operable().
        self.__next_input_operable_is_primed: bool = False
        # Tracks the state of whether we have tried to retrieve a value for
        # _next_input_operable at least once. By tracking this we can avoid
        # polling the input_stream upon instance creation.

        # register the image processors for in-flight results
        self._intermediary_operables = {processor: [] for processor in image_processors}

    @property
    def _next_input_operable(self) -> SpotAnalysisOperable:
        if self.__next_input_operable is None:
            if not self.__next_input_operable_is_primed:
                self.__next_input_operable_is_primed = True
                self._next_input_operable = next(self.input_stream)

        return self.__next_input_operable

    @_next_input_operable.setter
    def _next_input_operable(self, operable: SpotAnalysisOperable):
        self.__next_input_operable = operable

    @property
    def is_operable_available(self) -> bool:
        """
        Returns True if there are any intermediary operables, or if there is at
        least one more input operable. Returns False if all operables have been
        processed.

        This value is used in :py:meth:`process_next` to determine the value of
        the is_last flag when calling process_operable.
        """
        if self._next_input_operable is not None:
            return True

        if sum(len(ops) for ops in self._intermediary_operables.values()) > 0:
            return True

        return False

    def _feed_intermediary_operable(self) -> bool:
        """
        Feeds a single intermediary operable to the next image processor in the
        pipeline and returns True. If there are no intermediary operables then
        this returns False.
        """
        for processor_1_idx in range(len(self.image_processors) - 2, -1, -1):
            processor_1 = self.image_processors[processor_1_idx]

            if len(self._intermediary_operables[processor_1]) > 0:
                # There are results from processor_1 waiting to be processed
                # by the next processor (processor_2).
                # Send the first of the results to processor_2.

                # get the in-flight operable and remove it from the in-flight list
                in_flight_operable = self._intermediary_operables[processor_1].pop(0)

                # check if this is the last remaining operable to be evaluated
                is_last = not self.is_operable_available

                # Send the result from processor_1 to processor_2.
                processor_2_idx = processor_1_idx + 1
                processor_2 = self.image_processors[processor_2_idx]
                self._intermediary_operables[processor_2] += processor_2.process_operable(in_flight_operable, is_last)

                return True

        return False

    def _feed_pipeline(self):
        """
        Feeds a single intermediary operable to the next image processor in the
        pipeline. If there are no intermediary operables, then the next input
        operable is fed into the first processor in the pipeline.

        Raises
        ------
        StopIteration:
            If there are no more pending intermediary or input operables.
        """
        # Check for intermediary operables that still need to be processed.
        if self._feed_intermediary_operable():
            return

        # No pending intermediary operables.
        # Check if there is an input operable.
        if self._next_input_operable is None:
            raise StopIteration()

        # get the next input
        input_operable = self._next_input_operable
        try:
            self._next_input_operable = next(self.input_stream)
        except StopIteration:
            self._next_input_operable = None

        # process the input
        first_processor = self.image_processors[0]
        is_last = not self.is_operable_available
        self._intermediary_operables[first_processor] += first_processor.process_operable(input_operable, is_last)

    def process_next(self) -> None | SpotAnalysisOperable:
        """
        Attempts to get the next processed image and results data from the image
        processors pipeline.

        Returns
        -------
        result: SpotAnalysisOperable | None
            The processed primary image and other associated data. None if all
            input operables have been fully processed.
        """
        last_processor = self.image_processors[-1]

        # Release memory from the previous result
        if self._prev_result is not None:
            last_processor.cache_images_to_disk_as_necessary()
            self._prev_result = None

        # Process input and intermediary operables until we have results
        # available from the last processor.
        results = self._intermediary_operables[last_processor]
        while len(results) == 0:
            try:
                self._feed_pipeline()
                results = self._intermediary_operables[last_processor]
            except StopIteration:
                return None

        # return the result
        result_operable = results.pop(0)
        self._prev_result = result_operable
        return result_operable
