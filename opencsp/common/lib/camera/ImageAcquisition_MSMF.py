import cv2 as cv
import numpy as np

from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract


class ImageAcquisition(ImageAcquisitionAbstract):
    def __init__(self, instance: int = 0):
        # Connect to camera using MicroSoft Media Foundation API
        self.cap = cv.VideoCapture(instance, cv.CAP_MSMF)

        # Set auto exposure off
        self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.0)

        # Save max saturation value
        self._max_value = int(220)

        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            raise IOError("Error opening webcam")

    def get_frame(self) -> np.ndarray:
        # Capture image
        ret, frame = self.cap.read()

        # Check frame was captured successfully
        if not ret:
            raise Exception('Frame was not captured successfully.')

        # Format image
        if np.ndim(frame) == 3:
            frame = frame.mean(axis=2)
        elif np.ndim(frame) != 2:
            raise ValueError(
                f'Output frame must have 2 or 3 dimensions, not {np.ndim(frame):d}.'
            )

        return frame

    @property
    def gain(self) -> float:
        return self.cap.get(cv.CAP_PROP_GAIN)

    @gain.setter
    def gain(self, gain: float):
        self.cap.set(cv.CAP_PROP_GAIN, gain)

    @property
    def exposure_time(self) -> float:
        return self.cap.get(cv.CAP_PROP_EXPOSURE)

    @exposure_time.setter
    def exposure_time(self, exposure_time: float):
        self.cap.set(cv.CAP_PROP_EXPOSURE, exposure_time)

    @property
    def frame_size(self) -> tuple[int, int]:
        x = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        y = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        return (int(x), int(y))

    @frame_size.setter
    def frame_size(self, frame_size: tuple[int, int]):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    @property
    def frame_rate(self) -> float:
        return self.cap.get(cv.CAP_PROP_FPS)

    @frame_rate.setter
    def frame_rate(self, frame_rate: float):
        self.cap.set(cv.CAP_PROP_FPS, frame_rate)

    @property
    def max_value(self) -> int:
        return self._max_value

    @property
    def shutter_cal_values(self) -> np.ndarray:
        raise ValueError(
            'exposure_time cannot be adjusted with MSMF camera; adjust screen brightness instead.'
        )

    def close(self):
        self.cap.release()
