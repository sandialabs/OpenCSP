from abc import abstractmethod, ABC
import functools
from typing import Callable, Optional

import numpy as np

import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.log_tools as lt


class ImageAcquisitionAbstract(ABC):
    """
    Abstract base class for image acquisition from cameras.

    This class defines the interface for acquiring images from various camera types.
    It implements a multiton design pattern to ensure that only one instance of a
    camera is active at a time. The class provides methods for exposure calibration,
    frame retrieval, and managing camera settings.

    Attributes
    ----------
    _instances : dict[int, ImageAcquisitionAbstract]
        A dictionary of all instantiated camera instances, ensuring that each index
        corresponds to a unique camera instance.
    _next_instance_idx : int
        The index to use for the next camera instance added to the `_instances` dictionary.
    on_close : list[Callable[[ImageAcquisitionAbstract], None]]
        A list of callback functions to be executed when the camera is closed.
    is_closed : bool
        A flag indicating whether the camera connection is closed.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    _instances: dict[int, "ImageAcquisitionAbstract"] = {}
    """ All instantiated camera instances. We use a dictionary to ensure that once an index is assigned,
    then all references to that same index will return the same camera (or None if the camera was closed). """
    _next_instance_idx: int = 0
    """ The index to next use when adding a camera to the _instances dict. """

    @staticmethod
    @functools.cache
    def cam_options() -> dict[str, type["ImageAcquisitionAbstract"]]:
        """Defines camera objects to choose from (camera description, python type)"""
        # import here to avoid circular reference
        from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition as IA_DCAM_mono
        from opencsp.common.lib.camera.ImageAcquisition_DCAM_color import ImageAcquisition as IA_DCAM_color
        from opencsp.common.lib.camera.ImageAcquisition_MSMF import ImageAcquisition as IA_MSMF

        cam_options: dict[str, type[ImageAcquisitionAbstract]] = {
            "DCAM Mono": IA_DCAM_mono,
            "DCAM Color": IA_DCAM_color,
            "MSMF Mono": IA_MSMF,
        }
        return cam_options

    def __init__(self):
        if not self.instance_matches(ImageAcquisitionAbstract.instances()):
            idx = ImageAcquisitionAbstract._next_instance_idx
            ImageAcquisitionAbstract._instances[idx] = self
            ImageAcquisitionAbstract._next_instance_idx += 1
        else:
            lt.error_and_raise(
                RuntimeError,
                f"Error in {self.__class__.__name__}(): "
                + "we expect to only have one camera instance per connected camera, but one already exists!",
            )

        self.on_close: list[Callable[[ImageAcquisitionAbstract], None]] = []
        self.is_closed = False

    @staticmethod
    def instance(idx: int = None) -> Optional["ImageAcquisitionAbstract"]:
        """Get one of the global camera (ImageAcquisition) instances, if available.

        If the camera has been closed, then this function will return None, even if the instance exists.

        We use the multiton design pattern (several static global instances) for this class because we should only ever
        have one open instance per camera.

        Parameters
        ----------
        idx: int, None
            Which instance to return. If None, then the first non-closed instance is returned.
        """
        if idx is None:
            for idx in ImageAcquisitionAbstract._instances:
                camera = ImageAcquisitionAbstract._instances[idx]
                if not camera.is_closed:
                    return camera
            return None
        elif idx in ImageAcquisitionAbstract._instances:
            camera = ImageAcquisitionAbstract._instances[idx]
            if camera.is_closed:
                return None
            return camera
        else:
            return None

    @staticmethod
    def instances(subclass: type["ImageAcquisitionAbstract"] = None) -> list["ImageAcquisitionAbstract"]:
        """Get all global camera (ImageAcquisition) instances, as available.

        If a camera has been closed, then this function will not include it, even if the instance exists.

        We use the multiton design pattern (several static global instances) for this class because we should only ever
        have one open ImageAcquisitionAbstract instance per camera.

        Parameters
        ----------
        subclass: type[ImageAcquisitionAbstract], optional
            Limits the type of camera instances returned. If None, then all available are returned.
        """
        ret: list[ImageAcquisitionAbstract] = []
        for idx in ImageAcquisitionAbstract._instances:
            camera = ImageAcquisitionAbstract.instance(idx)
            if camera is None:
                continue
            if subclass is not None and not isinstance(camera, subclass):
                continue
            ret.append(camera)
        return ret

    @abstractmethod
    def instance_matches(self, possible_matches: list["ImageAcquisitionAbstract"]) -> bool:
        """
        Returns true if there's another instance in the list of possible matches that is equal to this instance. False
        if no other instances match.

        Parameters
        ----------
        possible_matches: list[ImageAcquisitionAbstract]
            The other cameras to match against. Does not include cameras that have been closed.
        """

    def calibrate_exposure(self):
        """
        Sets the camera's exposure so that only 1% of pixels are above the set
        saturation threshold. Uses a binary search algorithm.
        """
        lt.info("Starting exposure calibration.")

        def _get_exposure_idxs():
            """Returns indices of under/over exposed images"""
            # Under exposed
            idx_0 = np.where(exposure_out == -1)[0][-1]
            # Over exposed
            idx_1 = np.where(exposure_out == 1)[0][0]
            return idx_0, idx_1

        def _check_exposure_set():
            """Returns True if set, False if not set"""
            idx_0, idx_1 = _get_exposure_idxs()
            if idx_1 - idx_0 == 1:
                return True
            else:
                return False

        def _get_next_exposure_idx():
            """Returns the next exposure index to test"""
            idx_0, idx_1 = _get_exposure_idxs()
            return int(np.mean([idx_0, idx_1]))

        def _check_saturated(im):
            """Returns True if more than 1% of image is saturated"""
            return (im.astype(float) > (self.max_value * 0.95)).sum() > (0.01 * im.size)

        # Get exposure values to test
        exposure_values = self.shutter_cal_values

        # Create array to contain flag if exposure was too high (1) or low (-1)
        exposure_out = np.zeros(exposure_values.size)
        exposure_out[0] = -1
        exposure_out[-1] = 1

        # Checks that the minimum value is under-exposed
        self.exposure_time = exposure_values[0]
        lt.debug(f"Trying minimum exposure: {exposure_values[0]}")
        im = self.get_frame()
        if _check_saturated(im):
            lt.error_and_raise(ValueError, "Minimum exposure value is too high; image still saturated.")

        # Checks that the maximum value is over-exposed
        self.exposure_time = exposure_values[-1]
        lt.debug(f"Trying maximum exposure: {exposure_values[-1]}")
        im = self.get_frame()
        if not _check_saturated(im):
            lt.error_and_raise(ValueError, "Maximum exposure value is too low; image not saturated.")

        # Check if exposure is set
        max_iters = int(np.ceil(np.log2(exposure_values.size)) + 1)
        for _ in range(max_iters):
            # Get next exposure index to test
            idx = _get_next_exposure_idx()

            # Set exposure
            self.exposure_time = exposure_values[idx]
            lt.debug(f"Trying: {exposure_values[idx]}")

            # Capture image
            im = self.get_frame()

            # Check if saturated
            if _check_saturated(im):
                # Saturated
                exposure_out[idx : idx + 1] = np.array([1])
            else:
                # Under-saturated
                exposure_out[idx : idx + 1] = np.array([-1])

            # Check if exposure is set
            if _check_exposure_set():
                break

        # Check exposure was set successfully
        if not _check_exposure_set():
            lt.error_and_raise(ValueError, "Error with setting exposure.")

        # Set final exposure and log results
        exposure_value_set = exposure_values[_get_exposure_idxs()[0]]
        self.exposure_time = exposure_value_set
        lt.info(f"Exposure set to: {exposure_value_set}")

    @abstractmethod
    def get_frame(self):
        """Gets a single frame from the camera"""

    @property
    @abstractmethod
    def gain(self):
        """Camera gain value"""

    @property
    @abstractmethod
    def exposure_time(self) -> int:
        """Camera exposure_time value (microseconds)"""

    @property
    def exposure_time_seconds(self) -> float:
        """Camera exposure_time value (seconds)"""
        return self.exposure_time / 1_000_000

    @exposure_time_seconds.setter
    def exposure_time_seconds(self, exposure_time_seconds: float):
        self.exposure_time = int(exposure_time_seconds * 1_000_000)

    @property
    @abstractmethod
    def frame_size(self):
        """camera frame size (X pixels by Y pixels)"""

    @property
    @abstractmethod
    def frame_rate(self):
        """Camera frame rate (units of FPS)"""

    @property
    @abstractmethod
    def max_value(self):
        """
        Camera's maximum saturation value.
        Example: If camera outputs 8-bit images, max_value = 255
        """

    @property
    @abstractmethod
    def shutter_cal_values(self) -> np.ndarray:
        """
        Returns camera exposure_time values to use when calibrating the exposure_time. These
        values should be monotonically increasing from lowest shutter values to largest shutter
        values. The values need not be evely spaced, but the camera shutter is set to one of
        these values.
        """

    @abstractmethod
    def close(self):
        """Closes the camera connection"""
        # Prevent close() from being evaluated multiple times.
        # Also used as a check in get_instance().
        if self.is_closed:
            return
        self.is_closed = True

        # Callbacks
        for callback in self.on_close:
            with et.ignored(Exception):
                callback(self)
