import os
import sys
from typing import Callable, Optional, Union
import weakref

import numpy as np
import numpy.typing as npt
from PIL import Image

import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class CacheableImage:
    """
    An image container that allows for caching an image when the image
    data isn't in use, or for retrieval of an image from the cached file
    when the data is in use.

    The intended use for this class is to reduce memory usage by caching
    images to disk while not in use. Therefore, there is an inherent
    priority order for the data that is returned from various methods:
    (1) in-memory array, (2) numpy cache file, (3) image source file.

    Only one of the inputs (array, cache_path, source_path) are required. The
    "array" parameter should be the raw image data, the cache_path should be a
    string to a .npy file, and the source_path should be a string to an image
    file. Note that the cache_path file doesn't need to exist yet, but can
    instead be an indicator for where to cache the image to.

    The following table determines how images are retrieved based on the given
    parameters ([X] = given and file exists, [.] = given and file doesn't
    exist, * = any):

        +=======+=========+==========+========================================================================+
        | array | cache_p | source_p | retrieval method                                                       |
        +=======+=========+==========+========================================================================+
        |   X   |         |          | array                                                                  |
        |       |         |          | cache_path after cacheing (a temporary cache_path will be assigned)    |
        +-------+---------+----------+------------------------------------------------------------------------+
        |   X   |    X    |    *     | array                                                                  |
        |       |         |          | cache_path after cacheing (array contents will then be ignored)        |
        +-------+---------+----------+------------------------------------------------------------------------+
        |   X   |   [X]   |          | array                                                                  |
        |       |         |          | cache_path after cacheing (array contents will be saved to cache_path) |
        +-------+---------+----------+------------------------------------------------------------------------+
        |   X   |   [X]   |    X     | array                                                                  |
        |       |         |          | source_path after cacheing (array contents will then be ignored)       |
        +-------+---------+----------+------------------------------------------------------------------------+
        |   X   |         |    X     | array                                                                  |
        |       |         |          | source_path after cacheing (array contents will then be ignored)       |
        +-------+---------+----------+------------------------------------------------------------------------+
        |       |    X    |    *     | cache_path                                                             |
        +-------+---------+----------+------------------------------------------------------------------------+
        |       |   [X]   |    X     | source_path                                                            |
        +-------+---------+----------+------------------------------------------------------------------------+
        |       |         |    X     | source_path                                                            |
        +-------+---------+----------+------------------------------------------------------------------------+

    In addition, the following cases will raise a FileNotFoundError during the
    __init__ method():

        +=======+=========+==========+
        | array | cache_p | source_p |
        +=======+=========+==========+
        |   X   |         |   [X]    |
        +-------+---------+----------+
        |   X   |   [X]   |   [X]    |
        +-------+---------+----------+
        |       |   [X]   |          |
        +-------+---------+----------+
        |       |         |   [X]    |
        +-------+---------+----------+
    """

    _cacheable_images_registry: weakref.WeakKeyDictionary["CacheableImage", int] = {}
    """
    Class variable that tracks the existance of each cacheable image and the
    order of accesses.
    """
    _inactive_registry: weakref.WeakKeyDictionary["CacheableImage", int] = {}
    """ Like _cacheable_images_registry, but for instances that have been deregistered and not reregistered. """
    _cacheable_images_last_access_index: int = 0
    """ The last value used in the _cacheable_images_registry for maintaining access order. """

    def __init__(self, array: np.ndarray = None, cache_path: str = None, source_path: str = None):
        """
        Parameters
        ----------
        array: np.ndarray, optional
            The image, as it exists in memory. Should be compatible with
            image_tools.numpy_to_image().
        cache_path: str, optional
            Where to find and/or save the cached version of the image. Should be
            a numpy (.npy) file.
        source_path: str, optional
            The source file for the image (an image file, for example jpg or png).
        """
        # check that all the necessary inputs have been provided
        err_msg = "Error in CacheableImage.__init__(): must provide one of array, cache_path, or source_path!"
        if array is None:
            if cache_path is None:
                if source_path is None:
                    lt.error_and_raise(ValueError, err_msg)
                elif not ft.file_exists(source_path):
                    lt.error_and_raise(FileNotFoundError, err_msg)
            elif not ft.file_exists(cache_path):
                if source_path is None or not ft.file_exists(source_path):
                    lt.error_and_raise(FileNotFoundError, err_msg)
        else:
            if cache_path is None or not ft.file_exists(cache_path):
                if source_path is not None and not ft.file_exists(source_path):
                    lt.error_and_raise(FileNotFoundError, err_msg)

        # verify that the cache path is valid
        self.validate_cache_path(cache_path, "__init__")

        self._array = array
        """ The in-memory numpy array version of this image. None if not
        assigned, or if the data exists as a Pillow image, or if cached. """
        self._image = None
        """ The in-memory Pillow version of this image. None if not assigned, or
        if the data exists as a numpy array, or if cached. """
        self.cache_path = cache_path
        """ The path/name.ext to the cached numpy array. """
        self.source_path = source_path
        """ The path/name.ext to the source image file. """
        self.cached = False
        """ True if the numpy version of this image is cached to a file """

        self._register_access(self)

    def __del__(self):
        if not hasattr(self, "cache_path"):
            # this can happen when an error is raised during __init__()
            pass
        else:
            if self.cache_path is not None:
                with et.ignored(Exception):
                    ft.delete_file(self.cache_path)

    @classmethod
    def _register_access(cls, instance: "CacheableImage"):
        """
        Inserts this cacheable image as in index in the registry. This
        should be called every time the cacheable image is::

            - created
            - loaded from cache
            - accessed via nparray()
            - accessed via to_image()

        Parameters
        ----------
        instance : CacheableImage
            The instance to be registered.
        """
        if instance in cls._cacheable_images_registry:
            with et.ignored(KeyError):
                del cls._cacheable_images_registry[instance]
        if instance in cls._inactive_registry:
            with et.ignored(KeyError):
                del cls._inactive_registry[instance]
        cls._cacheable_images_registry[instance] = cls._cacheable_images_last_access_index + 1
        cls._cacheable_images_last_access_index += 1

    @classmethod
    def _register_inactive(cls, instance: "CacheableImage"):
        """
        Removes the given instance from the active registry and inserts it into
        the inactive registry. The inactive registry is useful for when a
        cacheable image has been cached and likely won't be active again for a
        while.
        """
        if instance in cls._cacheable_images_registry:
            with et.ignored(KeyError):
                del cls._cacheable_images_registry[instance]
        if instance in cls._inactive_registry:
            with et.ignored(KeyError):
                del cls._inactive_registry[instance]
        cls._inactive_registry[instance] = 0

    @classmethod
    def lru(cls, deregister=True) -> Optional["CacheableImage"]:
        """
        Returns the least recently used cacheable instance, where "use" is
        counted every time the image is loaded from cache.

        If deregister is true, then the returned instance is removed frmo the
        "active" list to the "inactive" list of cacheable images.

        This does not load any cached data from disk.
        """
        for instance_ref in cls._cacheable_images_registry:
            if instance_ref is not None:
                if deregister:
                    cls._register_inactive(instance_ref)
                return instance_ref

    def __sizeof__(self) -> int:
        """
        Returns the number of bytes in use by the in-memory numpy array and
        Pillow image.

        This does not load any cached data from disk.
        """
        return sys.getsizeof(self._array) + sys.getsizeof(self._image)

    @classmethod
    def all_cacheable_images_size(cls):
        """
        The number of bytes of system memory used by all cacheable images for
        in-memory numpy arrays and in-memory Pillow images.
        """
        ret = 0
        for instance_ref in cls._cacheable_images_registry:
            if instance_ref is not None:
                ret += sys.getsizeof(instance_ref)
        return ret

    @classmethod
    def from_single_source(
        cls, array_or_path: Union[np.ndarray, str, 'CacheableImage', Image.Image]
    ) -> 'CacheableImage':
        """Generates a CacheableImage from the given numpy array, numpy '.npy' file, or image file."""
        if isinstance(array_or_path, CacheableImage):
            return array_or_path
        elif isinstance(array_or_path, str):
            path: str = array_or_path
            if path.lower().endswith(".npy"):
                return cls(cache_path=path)
            return cls(source_path=path)
        elif isinstance(array_or_path, np.ndarray):
            array: np.ndarray = array_or_path
            return cls(array=array)
        elif isinstance(array_or_path, Image.Image):
            array: np.ndarray = np.array(array_or_path)
            return cls(array=array)
        else:
            lt.error_and_raise(
                TypeError, f"Error in CacheableImage.from_single_source(): unexpected type {type(array_or_path)}"
            )

    def validate_cache_path(self, cache_path: Optional[str], caller_name: str):
        """Ensures that the given cache_path ends with ".npy", or is None."""
        if cache_path == None:
            return

        if not cache_path.lower().endswith(".npy"):
            _, _, ext = ft.path_components(cache_path)
            lt.error_and_raise(
                ValueError,
                f"Error in CacheableImage.{caller_name}(): "
                + f"cache_path must end with '.npy' but instead the extension is {ext}",
            )

        path, name, ext = ft.path_components(cache_path)
        if not ft.directory_exists(path):
            lt.error_and_raise(
                FileNotFoundError,
                f"Error in CacheableImage.{caller_name}(): " + f"cache_path directory {path} does not exist",
            )

    @staticmethod
    def _load_image(im: str | np.ndarray) -> npt.NDArray[np.int_]:
        """Loads the cached numpy data or image file. Returns "im" if a numpy array."""
        if isinstance(im, np.ndarray):
            return im
        elif im.lower().endswith(".npy"):
            return np.load(im)
        else:
            im = Image.open(im)
            return np.array(im)

    def __load_image(self) -> npt.NDArray[np.int_] | None:
        """Loads the numpy array from the cache or image file, as necessary."""
        # self._register_access(self) # registered in self.nparray
        if self._array is not None:
            return self._array
        elif self.cache_path is not None and ft.file_exists(self.cache_path):
            self.cached = True
            return self._load_image(self.cache_path)
        elif ft.file_exists(self.source_path):
            return self._load_image(self.source_path)
        else:
            lt.error_and_raise(
                RuntimeError,
                f"Error in CacheableImage.__load_image(): Can't load image! {self._array=}, {self.cache_path=}, {self.source_path=}",
            )

    @property
    def nparray(self) -> npt.NDArray[np.int_] | None:
        """The image data for this image, as a numpy array."""
        self._register_access(self)
        self._image = None

        if self._array is None:
            if not self.cached:
                self._array = self.__load_image()

        return self.__load_image()

    def to_image(self) -> Image.Image:
        """The image data for this image, as a Pillow image."""
        # self._register_access(self) # registered in self.nparray
        if self._image == None:
            self._image = it.numpy_to_image(self.nparray)
        return self._image

    def _does_source_image_match(self, nparray: np.ndarray):
        """Returns true if this image's source_path image file matches the data
        in the given numpy array."""
        if self.source_path is not None:
            imarray = np.array(Image.open(self.source_path))
            try:
                arrays_are_equal = np.equal(nparray, imarray).all()
            except Exception:
                return False

            enforce_source_image_matches = True
            # Debugging: check that the programmer didn't misuse CacheableImage
            # by setting the source_path to a file that doesn't actually match
            # the numpy array or cache path.
            if enforce_source_image_matches and not arrays_are_equal:
                try:
                    import os
                    import opencsp.common.lib.opencsp_path.opencsp_root_path as orp

                    debug_dir = ft.norm_path(os.path.join(orp.opencsp_temporary_dir(), "debug"))
                    ft.create_directories_if_necessary(debug_dir)
                    np_path_name_ext = ft.norm_path(os.path.join(debug_dir, "CacheableImageNparray.png"))
                    im_path_name_ext = ft.norm_path(os.path.join(debug_dir, "CacheableImageImage.png"))
                    Image.fromarray(nparray).save(np_path_name_ext)
                    Image.fromarray(imarray).save(im_path_name_ext)
                    errtxt = f" These images have been saved to {debug_dir} for debugging."
                except:
                    errtxt = ""
                    pass

                lt.error_and_raise(
                    ValueError,
                    "Error in CacheableImage(): "
                    + "the cacheable image array data and image file must be identical, but they are not! "
                    + f"{self.cache_path=}, {self.source_path=}"
                    + errtxt,
                )

            return arrays_are_equal

    def cache(self, cache_path: str = None):
        """
        Stores this instance to the cache and releases the handle to the in-memory image.
        Note that the memory might not be abailable for garbage collection, if
        there are other parts of the code that still have references to the
        in-memory image or array.
        """
        # get the path to the numpy file
        if self.cache_path == None:
            if cache_path == None:
                lt.error_and_raise(
                    ValueError,
                    "Error in CacheableImage.cache(): "
                    + "this instance does not have a pre-programmed cache_path and the provided cache_path is None. "
                    + "Caching requires at least one path to be non-None!",
                )
            else:
                self.validate_cache_path(cache_path, "cache")
                self.cache_path = cache_path

        # check that this instance isn't already cached
        if self._array is None and self._image == None:
            return

        # Cache this instance.
        if self._does_source_image_match(self.nparray):
            # This instance was created from an image file, so we can simply
            # depend on that image file instead of writing a new numpy file to
            # disk.
            pass
        elif ft.file_exists(cache_path) and os.path.getsize(cache_path) > 10:
            # This instance has already been cached to disk at least once. Don't
            # need to cache it again.
            pass
        else:
            # Write the numpy array to disk.
            try:
                np.save(cache_path, self.nparray)
            except Exception as ex:
                lt.info(
                    "In CacheableImage.cache(): "
                    + f"exception encountered while trying to save to file {cache_path}. "
                    + f"Exception: {repr(ex)}"
                )

        # Indicate that this instance is cached
        self._array = None
        self._image = None
        self.cached = True

    def save_image(self, image_path_name_ext: str):
        """
        Saves this image as an image file to the given file.

        Note: this replaces the internal reference to source_path, if any, with
        the newly given path.

        Parameters
        ----------
        image_path_name_ext : str
            The file path/name.ext to save to. For example "image.png".
        """
        self.to_image().save(image_path_name_ext)
        self.source_path = image_path_name_ext

    @staticmethod
    def cache_images_to_disk_as_necessary(
        memory_limit_bytes: int, tmp_path_generator: Callable[[], str] = None, log_level=lt.log.DEBUG
    ):
        """Check memory usage and convert images to files (aka file path
        strings) as necessary in order to reduce memory usage."""
        total_mem_size = CacheableImage.all_cacheable_images_size()
        if total_mem_size <= memory_limit_bytes:
            return

        log_method = lt.get_log_method_for_level(log_level)
        target_mem_size = total_mem_size / 2
        log_method(f"Hit total memory size of {int(total_mem_size / 1024 / 1024)}MB")

        while total_mem_size > target_mem_size:
            # Get the least recently used cacheable image.
            # By cacheing the LRU instance, we are most likely to maintain
            # images in memory that are going to be used again in the near
            # future.
            cacheable_image = CacheableImage.lru()
            if cacheable_image is None:
                break

            # free the LRU instance's memory by cacheing it to disk
            cacheable_image_size = sys.getsizeof(cacheable_image)
            if cacheable_image_size == 0:
                pass  # already cached to disk
            if cacheable_image.cache_path is not None:
                cacheable_image.cache(None)
            else:  # cache_path is None
                if tmp_path_generator is not None:
                    cacheable_image.cache(tmp_path_generator())
                else:
                    lt.error_and_raise(
                        RuntimeError,
                        f"Attempting to cache CacheableImage {cacheable_image}, " + "but cache_path hasn't been set!",
                    )
            total_mem_size -= cacheable_image_size

        log_method(f"New total memory size after cacheing images: {int(total_mem_size / 1024 / 1024)}MB")
