import cv2 as cv
import numpy as np

from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
import opencsp.common.lib.tool.exception_tools as et


class ImageAcquisition(ImageAcquisitionAbstract):
    def __init__(self, instance: int = 0):
        # Connect to camera using MicroSoft Media Foundation API
        self.cap = cv.VideoCapture(instance, cv.CAP_MSMF)

        # Set auto exposure off
        self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.0)

        # Save max saturation value
        self._max_value = int(220)

        # Call super().__init__() once we have enough information for instance_matches().
        super().__init__()

        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            raise IOError("Error opening webcam")

    def instance_matches(self, possible_matches: list[ImageAcquisitionAbstract]) -> bool:
        """
        Determine whether this camera instance matches any instance in the provided list.

        This method checks if there is another instance of the `ImageAcquisition` class
        in the `possible_matches` list. Since only one MSMF camera is supported,
        the method returns True if any instance of `ImageAcquisition` is found; otherwise,
        it returns False.

        Parameters
        ----------
        possible_matches : list[ImageAcquisitionAbstract]
            A list of camera instances to check against. Each instance should be of
            type `ImageAcquisitionAbstract`.

        Returns
        -------
        bool
            True if a matching instance of `ImageAcquisition` is found; False otherwise.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        for camera in possible_matches:
            if isinstance(camera, ImageAcquisition):
                # only one MSMF camera is supported
                return True
        return False

    def get_frame(self) -> np.ndarray:
        """
        Captures a single frame from the connected camera.

        This method reads a frame from the camera and returns it as a NumPy array.
        If the captured frame is in color (3-dimensional), it is converted to a
        grayscale image by averaging the color channels. The method raises an
        exception if the frame capture is unsuccessful.

        Returns
        -------
        np.ndarray
            The captured image as a NumPy array. The shape of the array will be:
            - (height, width) for grayscale images.
            - If the input frame is in color, it will be converted to grayscale
            by averaging the channels.

        Raises
        ------
        Exception
            If the frame was not captured successfully, an exception is raised
            indicating the failure to capture the frame.

        ValueError
            If the output frame does not have 2 or 3 dimensions, a ValueError
            is raised indicating the incorrect number of dimensions.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Capture image
        ret, frame = self.cap.read()

        # Check frame was captured successfully
        if not ret:
            raise Exception("Frame was not captured successfully.")

        # Format image
        if np.ndim(frame) == 3:
            frame = frame.mean(axis=2)
        elif np.ndim(frame) != 2:
            raise ValueError(f"Output frame must have 2 or 3 dimensions, not {np.ndim(frame):d}.")

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
        raise ValueError("exposure_time cannot be adjusted with MSMF camera; adjust screen brightness instead.")

    def close(self):
        """Closes the camera connection"""
        with et.ignored(Exception):
            super().close()
        with et.ignored(Exception):
            self.cap.release()
