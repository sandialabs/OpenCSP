from abc import ABC
from collections.abc import Sized
import copy
import os
from typing import Iterator, Union
import sys

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.file_tools as ft

image_processors_persistant_memory_total: int = 4*pow(2, 30) # default total of 4GB
""" The amount of static memory that image processors are allowed to retain
as cache between calls to their 'run()' method. They always keep the most recent
results in memory, up until the data has been used by the next image processor
in the pipeline. """

class AbstractSpotAnalysisImagesProcessorLeger(ABC, Sized):
    """ Holds all the data flowing through an
    AbstractSpotAnalysisImageProcessor, broken out so as to reduce that class's
    complexity. """
    def __init__(self, name: str):
        self._name = name
        """ Name of this instance, probably the class name. """
        self._original_operables: Iterator[SpotAnalysisOperable] = None
        """ The input images to be processed, as given in the run() method. """
        self.all_processed_results: list[SpotAnalysisOperable] = None
        """ The results from evaluating this instance. None if processing hasn't
        finished yet. """
        self.cummulative_processed_results: list[SpotAnalysisOperable] = None
        """ The results currently available from evaluating this instance. None if
        processing hasn't started yet. """
        self._allowed_memory_footprint: int = None
        """ How much memory this instance is allowed to consume while idle. This
        affects whether primary and supporting images are held in memory as
        numpy arrays, or if they are saved to disk and returned as file path
        strings. """
        self.save_to_disk: int = False
        """ True to save results to the hard drive instead of holding them in
        memory. Dynamically determined at runtime during image processing based on
        self._allowed_memory_footprint. """
        self.input_iter: Iterator[SpotAnalysisOperable] = None
        """ Iterator over the input images. """
        self.finished_processing = False
        """ True if we've finished iterating over all input images. This gets set
        when we get a StopIteration error from next(input_iter). """
        self.cached = False
        """ True if we've ever cached the processed results of this processor to
        disk since processing was started. """
        self._my_tmp_dir = None
        """ The directory where temporary images from this instance are saved to. """
        self._tmp_images_saved = 0
        """ How many images have been saved by this instance since it was created. """
        self._clear_tmp_on_deconstruct = True
        """ If true, then delete all png images in _my_tmp_dir, and then also
        the directory if empty. """
        
        self.assign_inputs([])
        
    def __sizeof__(self) -> int:
        if self.cached:
            return 0
        elif len(self.cummulative_processed_results) == 0:
            return 0
        else:
            return sys.getsizeof(self.cummulative_processed_results[0]) * len(self.cummulative_processed_results)
        
    def __del__(self):
        # delete cached numpy files
        if ft.directory_exists(self._get_tmp_path()):
            ft.delete_files_in_directory(self._get_tmp_path(), "*.npy", error_on_dir_not_exists=False)
            if ft.directory_is_empty(self._get_tmp_path()):
                os.rmdir(self._get_tmp_path())
        
        # delete output png files
        if self._my_tmp_dir != None:
            if self._clear_tmp_on_deconstruct:
                ft.delete_files_in_directory(self._my_tmp_dir, "*.png", error_on_dir_not_exists=False)
                if ft.directory_is_empty(self._my_tmp_dir):
                    os.rmdir(self._my_tmp_dir)
            self._my_tmp_dir = None
    
    @property
    def name(self) -> str:
        """ Name of this processor """
        return self._name
    
    @property
    def input_operable(self) -> CacheableImage | None:
        """ The first of the input operables that was given to this instance
        before it did any image processing. """
        originals = self.input_operables
        if originals == None or len(originals) == 0:
            return None
        return originals[0]
    
    @property
    def input_operables(self) -> list[SpotAnalysisOperable] | None:
        """ The input operables that were given to this instance before it did
        any image processing. None if the input wasn't a list or image processor
        type. """
        if isinstance(self._original_operables, AbstractSpotAnalysisImagesProcessorLeger):
            predecessor: AbstractSpotAnalysisImagesProcessorLeger = self._original_operables
            if predecessor.finished:
                return predecessor.all_results
        elif isinstance(self._original_operables, list):
            return copy.copy(self._original_operables)
        return None
    
    @property
    def finished(self):
        if not (self.finished_processing == (self.all_processed_results == None)):
            lt.error_and_raise(RuntimeError, f"Programmer error in AbstractSpotAnalysisImageProcessor.finished: {self.finished_processing=} but {len(self.all_processed_results)=}")
        return self.finished_processing

    @property
    def all_results(self):
        """ Returns the list of resultant output images from this instance's image processing.

        Raises
        -------
        RuntimeError
            If finished == False
        """
        if not self.finished_processing:
            lt.error_and_raise(RuntimeError, "Can't get the list of processed images from this instance until all input images have been processed.")
        return copy.copy(self.all_processed_results)
    
    def assign_inputs(self, operables: Union['AbstractSpotAnalysisImagesProcessorLeger',list[SpotAnalysisOperable],Iterator[SpotAnalysisOperable]]):
        """ Register the input operables to be processed either with the run() method, or as an iterator. """
        self._original_operables = operables
        self.all_processed_results: list[SpotAnalysisOperable] = None
        self.cummulative_processed_results: list[SpotAnalysisOperable] = None
        self.input_iter: Iterator[SpotAnalysisOperable] = None
        self.finished_processing = False
    
    def initialize_cummulative_processed_results(self):
        if self.cummulative_processed_results != None and len(self.cummulative_processed_results) > 0:
            lt.error_and_raise(RuntimeError, f"Programmer error: initialized cummulative_processed_results at incorrect time. There are current {len(self.cummulative_processed_results)} in-flight results when there should be 0.")
        self.cummulative_processed_results = []
    
    def register_processed_result(self, operable: SpotAnalysisOperable, is_last: bool):
        # remember this processed result so that we can reference it again during a later computation
        self.cummulative_processed_results.append(operable)
        if is_last:
            self.all_processed_results = copy.copy(self.cummulative_processed_results)
    
    def cache_image_to_disk_as_necessary(self, operable: SpotAnalysisOperable):
        """ Check memory usage and convert images to files (aka file path
        strings) as necessary in order to reduce memory usage. """
        # import here to avoid import loops, since AbstractSpotAnalysisImageProcessor inherits from this class
        from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import AbstractSpotAnalysisImagesProcessor
        
        total_mem_size = sys.getsizeof(operable)*2 + sys.getsizeof(self)
        allowed_memory_footprint = image_processors_persistant_memory_total
        if self._allowed_memory_footprint != None:
            allowed_memory_footprint = self._allowed_memory_footprint
            
        if (total_mem_size > allowed_memory_footprint) or (self.save_to_disk):
            image_processor = self
            if isinstance(self.input_iter, AbstractSpotAnalysisImagesProcessor):
                image_processor = self.input_iter
            
            operable.primary_image.cache(image_processor._get_tmp_path())
            if not self.cached:
                for result in self.cummulative_processed_results:
                    result.primary_image.cache(image_processor._get_tmp_path())
                self.cached = True
    
    def _get_save_dir(self):
        """ Finds a temporary directory to save to for the processed output images from this instance. """
        if self._my_tmp_dir == None:
            scratch_dir = os.path.join(orp.opencsp_scratch_dir(), "spot_analysis_image_processing")
            i = 0
            while True:
                dirname = self.name + str(i)
                self._my_tmp_dir = os.path.join(scratch_dir, dirname)
                if not ft.directory_exists(self._my_tmp_dir):
                    try:
                        os.makedirs(self._my_tmp_dir)
                        break # success!
                    except FileExistsError:
                        # probably just created this directory in another thread
                        pass
                else:
                    i += 1
        return self._my_tmp_dir
    
    def _get_tmp_path(self) -> str:
        """ Get the path+name+ext to save a cacheable image to, in our temporary
        directory, in numpy format.

        Returns:
            path_name_ext: str
                Where to save the image.
        """
        # get the path
        path_name_ext = os.path.join(self._get_save_dir(), f"{self._tmp_images_saved}.npy")
        self._tmp_images_saved += 1
        return path_name_ext
    
    def __len__(self) -> int:
        """ Get the number of processed output images from this instance.

        Raises
        -------
        RuntimeError
            If the input images haven't finished being processed yet.
        """
        if self.all_processed_results != None:
            return len(self.all_processed_results)
        lt.error_and_raise(RuntimeError, "Can't get the length of this instance until all input images have been processed.")

    def save_processed_images(self, dir: str, name_prefix: str=None, ext="jpg"):
        """ Saves the processed images to the given directory with the file name
        "[name_prefix+'_']SA_preprocess_[self.name][index].[ext]". If this
        instance is being used as an image stream, then use
        on_image_processed(get_processed_image_save_callback()) instead.
        
        This method is designed to be used as a callback with self.on_image_processed(). """
        for idx, operable in enumerate(self.all_processed_results):
            self._save_image(operable.primary_image, [idx], dir, name_prefix, ext)
    
    def _save_image(self, im: CacheableImage, idx_list: list[int], dir: str, name_prefix: str=None, ext="jpg"):
        idx = idx_list[0]
        image_name = ("" if name_prefix == None else f"{name_prefix}_") + f"SA_preprocess_{self.name}{idx}"
        image_path_name_ext = os.path.join(dir, image_name+"."+ext)
        lt.debug("Saving SpotAnalysis processed image to " + image_path_name_ext)
        im.to_image().save(image_path_name_ext)
        idx_list[0] = idx + 1
        return image_path_name_ext