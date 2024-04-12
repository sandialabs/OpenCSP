"""Representation of a notional camera for image acquisition"""

from typing import Callable

import numpy as np

from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract


class ImageAcquisition(ImageAcquisitionAbstract):
    def __init__(self):
        super().__init__()

        # Define max saturation value
        self._max_value = 250

        # Define gain
        self._gain = 100

        # Define exposure_time
        self._shutter = 1

        # Define frame size
        self._frame_size = (640, 320)

        # Define frame rate
        self._frame_rate = 30

        # Define exposure_time calibration values
        self._shutter_cal_values = np.arange(1, 100, 1, dtype=float)

    def instance_matches(self, possible_matches: list[ImageAcquisitionAbstract]) -> bool:
        # allow for unlimited cameras during unit tests
        return False

    def get_frame(self) -> np.ndarray:
        # Return test image
        x, y = self._frame_size
        return np.zeros((y, x), dtype=np.uint8)

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, gain: float):
        self._gain = gain

    @property
    def exposure_time(self) -> float:
        return self._shutter

    @exposure_time.setter
    def exposure_time(self, exposure_time: float):
        self._shutter = exposure_time

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    @frame_size.setter
    def frame_size(self, frame_size: tuple[int, int]):
        self._frame_size = frame_size

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, frame_rate: float):
        self._frame_rate = frame_rate

    @property
    def max_value(self) -> int:
        return self._max_value

    @property
    def shutter_cal_values(self) -> np.ndarray:
        return self._shutter_cal_values

    def close(self):
        super().close()


class IA_No_Calibrate(ImageAcquisition):
    def __init__(self):
        super().__init__()
        self.is_calibrated = False

    def calibrate_exposure(self):
        self.is_calibrated = True


class ImageAcquisitionWithFringes(ImageAcquisition):
    """Class for unit testing. Mimics a camera by returning first a light image for the mask, then a dark image
    for the mask, then cycling through all fringe images."""

    def __init__(self, fringes: Fringes):
        super().__init__()
        self.index = -2
        self.fringes = fringes
        self.fringe_images = None

    def get_frame(self) -> np.ndarray:
        x, y = self.frame_size

        if self.index < 0:
            # mask images
            frame = np.zeros((y, x), "uint8")
            if self.index == -1:
                frame[10:-10, 10:-10] = self.max_value
        else:
            # fringe images
            if self.fringe_images is None:
                self.fringe_images = self.fringes.get_frames(x, y, "uint8", [0, self.max_value])
            frame = self.fringe_images[:, :, self.index]

        self.index += 1
        if self.index >= self.fringes.num_images:
            self.index = 0

        return frame
