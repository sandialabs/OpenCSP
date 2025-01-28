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

        1. in-memory array
        2. numpy cache file
        3. image source file

    Sources
    -------
    Only one of the inputs (array, cache_path, source_path) are required in the
    constructor. In fact, there is a method :py:meth:`from_single_source` that
    tries to guess which input is being provided. The "array" parameter should
    be the raw image data, the cache_path should be a string to a .npy file, and
    the source_path should be a string to an image file. Note that the
    cache_path file doesn't need to exist yet, but can instead be an indicator
    for where to cache the image to.

    The following table determines how images are retrieved based on the given
    parameters ([X] = given and file exists, [X] = given and file doesn't
    exist, * = any):

        +-------+---------+----------+-----------------------------------------------------------------------------------+
        | array | cache_p | source_p | retrieval method                                                                  |
        +=======+=========+==========+===================================================================================+
        |   X   |         |          | **array** (cache_path after cacheing, a temporary cache_path will be assigned)    |
        +-------+---------+----------+-----------------------------------------------------------------------------------+
        |   X   |    X    |    \*    | **array** (cache_path after cacheing, array contents will then be ignored)        |
        +-------+---------+----------+-----------------------------------------------------------------------------------+
        |   X   |   [X]   |          | **array** (cache_path after cacheing, array contents will be saved to cache_path) |
        +-------+---------+----------+-----------------------------------------------------------------------------------+
        |   X   |   [X]   |    X     | **array** (source_path after cacheing, array contents will then be ignored)       |
        +-------+---------+----------+-----------------------------------------------------------------------------------+
        |   X   |         |    X     | **array** (source_path after cacheing, array contents will then be ignored)       |
        +-------+---------+----------+-----------------------------------------------------------------------------------+
        |   X   |         |   [X]    | **array** (cache_path after cacheing, array contents will be saved to cache_path) |
        +-------+---------+----------+-----------------------------------------------------------------------------------+
        |       |    X    |    \*    | **cache_path**                                                                    |
        +-------+---------+----------+-----------------------------------------------------------------------------------+
        |       |   [X]   |    X     | **source_path**                                                                   |
        +-------+---------+----------+-----------------------------------------------------------------------------------+
        |       |         |    X     | **source_path**                                                                   |
        +-------+---------+----------+-----------------------------------------------------------------------------------+

    In addition, the following cases will raise a FileNotFoundError during the
    __init__ method():

        +-------+---------+----------+
        | array | cache_p | source_p |
        +=======+=========+==========+
        |       |   [X]   |          |
        +-------+---------+----------+
        |       |         |   [X]    |
        +-------+---------+----------+


    sys.getsizeof
    -------------
    This class overrides the default __sizeof__ dunder method, meaning that the
    size returned by sys.getsizeof(cacheable_image) is not just the size of all
    variables tracked by the instance. Rather, the size of the Numpy array and
    Pillow image are returned. This metric better represents the
    memory-conserving use case that is intended for this class.

    __sizeof__ returns close to 0 if the array and image attributes have been
    set to None (aka the cache() method has been executed). Note that this does
    not depend on the state of the garbage collector, which might not actually
    free the memory for some time after it is no longer being tracked by this
    class. An attempt at freeing the memory can be done immediately with
    `gc.collect()` but the results of this are going to be implementation
    specific.

    The size of the source path, image path, and all other attributes are not
    included in the return value of __sizeof__. This decision was made for
    simplicity. Also the additional memory has very little impact. For example a
    256 character path uses ~0.013% as much memory as a 1920x1080 monochrome
    image.
    """

    _cacheable_images_registry: weakref.WeakKeyDictionary["CacheableImage", int] = {}
    # Class variable that tracks the existence of each cacheable image and the
    # order of accesses.
    _inactive_registry: weakref.WeakKeyDictionary["CacheableImage", int] = {}
    # Like _cacheable_images_registry, but for instances that have been deregistered and not reregistered.
    _cacheable_images_last_access_index: int = 0
    # The last value used in the _cacheable_images_registry for maintaining access order.
    _expected_cached_size: int = (48 * 2) * 2
    # Upper bound on the anticipated return value from __sizeof__ after cache()
    # has just been evaluated. Each python variable uses ~"48" bytes, there are
    # "*2" variables included (array and image), and we don't care about
    # specifics so add some buffer "*2".

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
            The source file for the image (an image file, for example jpg or
            png). If provided, then this file will be used as the backing cache
            instead of creating a new cache file.
        """
        # check that all the necessary inputs have been provided
        err_msg = "Error in CacheableImage.__init__(): must provide at least one of array, cache_path, or source_path!"
        fnfe_msg = err_msg + " %s file %s does not exist!"
        if array is None:
            if cache_path is None:
                if source_path is None:
                    lt.error_and_raise(ValueError, err_msg)
                elif not ft.file_exists(source_path):
                    lt.error_and_raise(FileNotFoundError, fnfe_msg % ("source", source_path))
            elif not ft.file_exists(cache_path):
                if source_path is None:
                    lt.error_and_raise(FileNotFoundError, fnfe_msg % ("cache", source_path))
                elif not ft.file_exists(source_path):
                    lt.error_and_raise(FileNotFoundError, fnfe_msg % ("source/cache", source_path + "/" + cache_path))
        elif not isinstance(array, np.ndarray):
            lt.error_and_raise(
                TypeError, "Error in CacheableImage.__init__(): " + "given array must be a numpy array, if given!"
            )

        # verify that the paths are valid
        self.validate_cache_path(cache_path, "__init__")
        self.validate_source_path(source_path, "__init__")

        self._array = array
        # The in-memory numpy array version of this image. None if not assigned or
        # if cached. Should always be available whenever self._image is available.
        self._image = None
        # The in-memory Pillow version of this image. None if not assigned, or if
        # the data exists as a numpy array but not as an image, or if this
        # instance is cached.
        self.cache_path = cache_path
        """ The path/name.ext to the cached numpy array. """
        self.source_path = source_path
        """ The path/name.ext to the source image file. """

        self._register_access(self)

    def __del__(self):
        if not hasattr(self, "cache_path"):
            # this can happen when an error is raised during __init__()
            pass
        else:
            if self._cache_path is not None:
                with et.ignored(Exception):
                    ft.delete_file(self._cache_path)
                    self._cache_path = None

    @property
    def cache_path(self) -> str | None:
        """
        The path/name.ext to the cached numpy array, if set. Guaranteed to pass
        validate_cache_path().
        """
        return self._cache_path

    @cache_path.setter
    def cache_path(self, new_val: str | None):
        """
        Parameters
        ----------
        new_val : str
            The path/name.ext to the cached numpy array. This file doesn't need
            to exist yet, but if it does exist, then its contents should match
            self.nparray. Must pass validate_cache_path().
        """
        if new_val is not None:
            # verify the ending to the cache path file
            new_val = ft.norm_path(new_val)
            self.validate_cache_path(new_val, "cache_path")

            # verify the file contents equal the array contents
            try:
                arrval = self.nparray
            except Exception:
                arrval = None
            if arrval is not None:
                if ft.file_exists(new_val):
                    cache_val = self._load_image(new_val)
                    if not np.equal(arrval, cache_val).all():
                        lt.error_and_raise(
                            ValueError,
                            "Error in CacheableImage.source_path(): "
                            + f"the contents of self.nparray and {new_val} do not match!"
                            + f" ({self.cache_path=}, {self.source_path=})",
                        )

        self._cache_path = new_val

    @property
    def source_path(self) -> str | None:
        """
        The path/name.ext to the source image file, if set. Guaranteed to pass
        validate_source_path().
        """
        return self._source_path

    @source_path.setter
    def source_path(self, new_val: str | None):
        """
        Parameters
        ----------
        new_val : str
            The path/name.ext to the source image file. This file doesn't need
            to exist yet, but if it does exist, then it must be readable by
            Pillow and its contents should match self.nparray. Must pass
            validate_source_path().
        """
        if new_val is not None:
            # verify we can read the file
            new_val = ft.norm_path(new_val)
            self.validate_source_path(new_val, "source_path")

            # verify the file contents equal the array contents
            try:
                arrval = self.nparray
            except Exception:
                arrval = None
            if arrval is not None:
                if ft.file_exists(new_val):
                    image = self._load_image(new_val)
                    if not np.equal(arrval, image).all():
                        lt.error_and_raise(
                            ValueError,
                            "Error in CacheableImage.source_path(): "
                            + f"the contents of self.nparray and {new_val} do not match!"
                            + f" ({self.cache_path=}, {self._source_path=})",
                        )

        self._source_path = new_val

    @staticmethod
    def _register_access(instance: "CacheableImage"):
        # Inserts this cacheable image as in index in the registry. This should be
        # called every time the cacheable image is accessed for tracking the most
        # recently used instances. This should be called at least during::
        #
        #    - creation
        #    - loading into memory
        #    - access via nparray()
        #    - access via to_image()
        #
        # Parameters
        # ----------
        # instance : CacheableImage
        #    The instance to be registered.
        images_registry = CacheableImage._cacheable_images_registry
        inactive_registry = CacheableImage._inactive_registry

        if instance in images_registry:
            with et.ignored(KeyError):
                del images_registry[instance]
        if instance in inactive_registry:
            with et.ignored(KeyError):
                del inactive_registry[instance]
        images_registry[instance] = CacheableImage._cacheable_images_last_access_index + 1
        CacheableImage._cacheable_images_last_access_index += 1

    @staticmethod
    def _register_inactive(instance: "CacheableImage"):
        # Removes the given instance from the active registry and inserts it into
        # the inactive registry. The inactive registry is useful for when a
        # cacheable image has been cached and likely won't be active again for a
        # while.
        images_registry = CacheableImage._cacheable_images_registry
        inactive_registry = CacheableImage._inactive_registry

        if instance in images_registry:
            with et.ignored(KeyError):
                del images_registry[instance]
        if instance in inactive_registry:
            with et.ignored(KeyError):
                del inactive_registry[instance]
        inactive_registry[instance] = 0

    @staticmethod
    def lru(deregister=True) -> Optional["CacheableImage"]:
        """
        Returns the least recently used cacheable instance, where "use" is
        counted every time the image is loaded from cache.

        If deregister is true, then the returned instance is moved from the
        "active" list to the "inactive" list of cacheable images. This is useful
        when we anticipate that the returned instance isn't going to be used for
        a while, such as from the method cache_images_to_disk_as_necessary().

        This does not load any cached data from disk.
        """
        images_registry = CacheableImage._cacheable_images_registry

        for instance_ref in images_registry:
            if instance_ref is not None:
                if deregister:
                    CacheableImage._register_inactive(instance_ref)
                return instance_ref
            else:
                # the CacheableImage has been garbage collected, remove its
                # entry from the weak references dict
                with et.ignored(KeyError):
                    del images_registry[instance_ref]

    def __sizeof__(self) -> int:
        # Returns the number of bytes in use by the in-memory numpy array and
        # Pillow image for this instance.
        #
        # This does not load any cached data from disk.
        return sys.getsizeof(self._array) + it.getsizeof_approx(self._image)

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
                return cls(cache_path_name_ext=path)
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
        """
        Verifies that the cache_path, if set, ends with ".npy". Raises a ValueError otherwise.

        Parameters
        ----------
        cache_path : str | None
            The path/name.ext to save this image to, as a numpy array. For example
            ft.join(orp.opencsp_scratch_dir(), "/OpenCSP/tmp001.npy").
        caller_name : str
            Name of the calling method. This is used as part of the generated error message.
        """
        if cache_path == None:
            return

        if not cache_path.lower().endswith(".npy"):
            _, _, ext = ft.path_components(cache_path)
            lt.error_and_raise(
                ValueError,
                f"Error in CacheableImage.{caller_name}(): "
                + f"cache_path must end with '.npy' but instead the extension is {ext}",
            )

    def validate_source_path(self, source_path: Optional[str], caller_name: str):
        """Ensures that the given source_path has one of the readable file extensions, or is None."""
        if source_path == None:
            return

        _, _, ext = ft.path_components(source_path)
        ext = ext[1:]  # strip off the leading period "."
        allowed_exts = sorted(it.pil_image_formats_rw + it.pil_image_formats_readable)
        if ext.lower() not in allowed_exts:
            lt.error_and_raise(
                ValueError,
                f"Error in CacheableImage.{caller_name}(): "
                + f"{source_path} must be readable by Pillow, but "
                + f"the extension {ext} isn't in the known extensions {allowed_exts}!",
            )

    @staticmethod
    def _load_image(im: str | np.ndarray) -> npt.NDArray[np.int_]:
        # Loads the cached numpy data or image file. If the given "im" is a numpy
        # array then it will be returned as is.
        if isinstance(im, np.ndarray):
            return im
        elif im.lower().endswith(".npy"):
            return np.load(im)
        else:
            im = Image.open(im)
            return np.array(im)

    def __load_image(self) -> npt.NDArray[np.int_] | None:
        # Loads the numpy array from the cache or image file, as necessary.
        # self._register_access(self) # registered in self.nparray
        if self._array is not None:
            return self._array
        elif self.cache_path is not None and ft.file_exists(self.cache_path):
            return self._load_image(self.cache_path)
        elif self._source_path is not None and ft.file_exists(self._source_path):
            return self._load_image(self._source_path)
        else:
            lt.error_and_raise(
                RuntimeError,
                f"Error in CacheableImage.__load_image(): Can't load image! {self._array=}, {self.cache_path=}, {self._source_path=}",
            )

    @property
    def nparray(self) -> npt.NDArray[np.int_] | None:
        """The data for this CacheableImage, as a numpy array. This is the
        default internal representation of data for this class."""
        self._register_access(self)

        if self._array is None:
            self._array = self.__load_image()

        return self._array

    def to_image(self) -> Image.Image:
        """Converts the numpy array representation of this image into a Pillow
        Image class and returns the converted value."""
        if self._image == None:
            # self._register_access(self) # registered in self.nparray
            self._image = it.numpy_to_image(self.nparray)
        else:
            self._register_access(self)

        return self._image

    def _does_source_image_match(self, nparray: np.ndarray):
        # Returns true if this image's source_path image file matches the data
        # in the given numpy array.
        if self._source_path is not None and ft.file_exists(self._source_path):
            imarray = np.array(Image.open(self._source_path))
            try:
                arrays_are_equal = np.equal(nparray, imarray).all()
            except Exception:
                return False

            # Check that the programmer didn't misuse CacheableImage by setting
            # the source_path to a file that doesn't actually match the numpy
            # array or cache path.
            if not arrays_are_equal:
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
                    + f"{self.cache_path=}, {self._source_path=}"
                    + errtxt,
                )

            return arrays_are_equal

    def cache(self, cache_path: str = None):
        """
        Stores this instance to the cache and releases the handle to the in-memory image.
        Note that the memory might not be abailable for garbage collection, if
        there are other parts of the code that still have references to the
        in-memory image or array.

        Parameters
        ----------
        cache_path : str, optional
            The cache path/name.ext to use for storing this image, in case this
            image doesn't already have a cache path. Can be None if a new cache
            file isn't needed (the source image file is accessible or this
            instance has been cached before). By default None
        """
        # use either self.cache_path or cache_path, depending on:
        # 1. self.cache_path exists
        # 2. cache_path exists
        # 3. self.cache_path is set and cache_path does not exist
        # 4. self.cache_path is not set cache_path is set
        # 5. self.source_path exists
        if self.cache_path is not None:
            if ft.file_exists(self.cache_path):
                # 1. do nothing
                pass
            elif cache_path is not None:
                if ft.file_exists(cache_path):
                    # 2. update self.cache_path to the already existing file
                    self.cache_path = cache_path
                else:
                    # 3. do nothing
                    pass
            else:
                # 3. do nothing
                pass
        else:
            if cache_path is not None:
                # 4. use the non-None value
                self.cache_path = cache_path
            else:
                if self._source_path is not None and ft.file_exists(self._source_path):
                    # 5. don't need to create a numpy file if we can just read from the source file
                    pass
                else:
                    # We don't have enough information about where to put the
                    # contents for this instance.
                    lt.error_and_raise(
                        ValueError,
                        "Error in CacheableImage.cache(): "
                        + "this instance was not created with a cache_path and the provided cache_path is None. "
                        + "Cacheing requires at least one path (cache_path or source_path) to be non-None!",
                    )

        # Cache this instance.
        if self._does_source_image_match(self.nparray):
            # This instance was created from an image file, so we can simply
            # depend on that image file instead of writing a new numpy file to
            # disk.
            pass
        elif ft.file_exists(self.cache_path) and os.path.getsize(self.cache_path) > 10:
            # This instance has already been cached to disk at least once. Don't
            # need to cache it again. We check the size to make sure that the
            # file was actually cached and doesn't just exist as a placeholder.
            # I chose '> 10' instead of '> 0' because I'm paranoid that
            # getsize() will return a small number of bytes on some systems.
            pass
        else:
            # Write the numpy array to disk.
            try:
                np.save(self.cache_path, self.nparray)
            except Exception:
                lt.error(
                    "In CacheableImage.cache(): " + f"exception encountered while trying to save to file {cache_path}. "
                )
                raise

        # Indicate that this instance is cached
        self._array = None
        self._image = None

    def save_image(self, image_path_name_ext: str):
        """
        Saves this image as an image file to the given file. This method is best
        used when an image is intended to be kept after a computation, in which
        case the newly saved image file can be the on-disk reference instead of
        an on-disk cache file.

        Note: this replaces the internal reference to source_path, if any, with
        the newly given path. It is therefore suggested to not use this method
        unless you are using this class as part of an end-use application, in
        order to avoid unintended side effects.

        Parameters
        ----------
        image_path_name_ext : str
            The file path/name.ext to save to. For example "image.png".
        """
        self.to_image().save(image_path_name_ext)
        self._source_path = image_path_name_ext

    @staticmethod
    def cache_images_to_disk_as_necessary(
        memory_limit_bytes: int, tmp_path_generator: Callable[[], str], log_level=lt.log.DEBUG
    ):
        """
        Check memory usage and convert images to files (aka file path strings)
        as necessary in order to reduce memory usage.

        Note that due to the small amount of necessary memory used by each
        CacheableImage instance, all instances can be cached and still be above
        the `memory_limit_bytes` threshold. This can happen either when
        memory_limit_bytes is sufficiently small, or the number of live
        CacheableImages is sufficiently large. In these cases, this method may
        not be able to lower the amount of memory in use.

        Parameters
        ----------
        memory_limit_bytes : int
            The total number of bytes of memory that all CacheableImages are
            allowed to use for their in-memory arrays and images, in sum. Note
            that each CachableImage instance will still use some small amount of
            memory even after it has been cached.
        tmp_path_generator : Callable[[], str]
            A function that returns a path/name.ext for a file that does not
            exist yet. This file will be used to save the numpy array out to.
        log_level : int, optional
            The level to print out status messages to, including the amount of
            memory in use before and after caching images. By default
            lt.log.DEBUG.
        """
        # By providing the memory_limit_bytes as a parameter, we're effectively
        # enabling the user to choose a lower memory threshold than is the
        # default. There's also the benefit of requiring the user to think about
        # how much memory they want to use, which is going to be system and
        # application specific.
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
            if cacheable_image_size <= cacheable_image._expected_cached_size:
                continue  # already cached to disk, probably
            if cacheable_image.cache_path is not None:
                cacheable_image.cache(None)
            else:  # cache_path is None
                cacheable_image.cache(tmp_path_generator())

            bytes_cached_to_disk = cacheable_image_size - sys.getsizeof(cacheable_image)
            total_mem_size -= bytes_cached_to_disk

        log_method(f"New total memory size after cacheing images: {int(total_mem_size / 1024 / 1024)}MB")
