import numpy as np
from pypylon import pylon

from opencsp.common.lib.camera.image_processing import encode_RG_to_RGB
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition as MonoIA
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.log_tools as lt


class ImageAcquisition(ImageAcquisitionAbstract):
    def __init__(self, instance: int = 0, pixel_format: str = "BayerRG12"):
        """
        Class to control a Basler DCAM color camera. Grabs one frame
        at a time. Assumes the color camera is in a Bayer pattern. The
        gain is initially set to the lowest possible value.

        Parameters
        ----------
        instance : int
            The Nth instance of cameras (sorted by serial number) to instantiate.
        pixel_format : str
            Available formats include:
                - BayerRG12 (default)
                - BayerRG12Packed
                - Other RGB based formats as defined by Basler

        """
        MonoIA._check_pypylon_version()

        # Find all instances of DCAM cameras
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        # Check that at least one camera is available
        if len(devices) == 0:
            raise pylon.RuntimeException("No cameras found.")
        else:
            print(f"{len(devices):d} devices found.")
            for idx, device in enumerate(devices):
                if idx == instance:
                    print("--> ", end="")
                else:
                    print("    ", end="")
                print(device.GetFriendlyName())

        # Check number of instances
        if instance >= len(devices):
            raise ValueError(f"Cannot load instance {instance:d}. Only {len(devices):d} devices found.")

        # Connect to camera
        self.basler_index = instance
        try:
            self.cap = pylon.InstantCamera(tlFactory.CreateDevice(devices[instance]))
            self.cap.Open()
        except Exception as ex:
            lt.error("Failed to connect to camera")
            raise RuntimeError("Failed to connect to camera") from ex

        # Call super().__init__() once we have enough information for instance_matches() and we know that connecting to
        # the camera isn't going to fail.
        super().__init__()

        # Set up device to single frame acquisition
        self.cap.AcquisitionMode.SetValue("SingleFrame")

        # Set pixel format
        self.cap.PixelFormat.SetValue(pixel_format)

        # Set gain to minimum value
        self.cap.GainRaw.SetValue(self.cap.GainRaw.Min)

        # Set exposure values to be stepped over when performing exposure calibration
        shutter_min = self.cap.ExposureTimeRaw.Min
        shutter_max = self.cap.ExposureTimeRaw.Max
        self._shutter_cal_values = np.linspace(shutter_min, shutter_max, 2**13).astype(int)

    def instance_matches(self, possible_matches: list[ImageAcquisitionAbstract]) -> bool:
        """
        Determine whether this camera instance matches any instance in the provided list.

        This method checks if there is another camera instance in the `possible_matches`
        list that has the same `basler_index` as the current instance.

        Parameters
        ----------
        possible_matches : list[ImageAcquisitionAbstract]
            A list of camera instances to check against. Each instance should be of
            type `ImageAcquisitionAbstract` and may have a `basler_index` attribute.

        Returns
        -------
        bool
            True when the first matching instance is found; False otherwise.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        for camera in possible_matches:
            if not hasattr(camera, "basler_index"):
                continue
            if getattr(camera, "basler_index") == self.basler_index:
                return True
        return False

    def get_frame(self, encode: bool = True) -> np.ndarray:
        """
        Captures a single frame from the camera.

        This method initiates the frame capture process and retrieves the image data
        from the camera. The captured image can be returned in its raw Bayer format
        or converted to an RGB format based on the `encode` parameter.

        Parameters
        ----------
        encode : bool, optional
            If True (default), the captured Bayer-encoded image will be converted to
            a 3D RGB image using the `encode_RG_to_RGB` function. If False, the raw
            Bayer-encoded image will be returned.

        Returns
        -------
        np.ndarray
            The captured image as a NumPy array. The shape of the array will depend
            on the encoding:
            - If `encode` is True, the shape will be (height, width, 3) for RGB images.
            - If `encode` is False, the shape will be (height, width) for Bayer images.

        Raises
        ------
        pylon.RuntimeException
            If the frame grab is unsuccessful, an exception is raised indicating the
            failure to capture the frame.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Start frame capture
        self.cap.StartGrabbingMax(1)
        grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Access image data
        if grabResult.GrabSucceeded():
            img = grabResult.Array
        else:
            raise pylon.RuntimeException("Frame grab unsuccessful.")

        # Wait for capturing to finish
        grabResult.Release()

        # Convert from 2D Bayer-encoded image to 3D RGB image
        if encode:
            return encode_RG_to_RGB(img)
        else:
            return img

    @property
    def gain(self) -> float:
        return self.cap.GainRaw.GetValue()

    @gain.setter
    def gain(self, gain: float):
        self.cap.GainRaw.SetValue(gain)

    @property
    def exposure_time(self) -> int:
        return self.cap.ExposureTimeRaw.GetValue()

    @exposure_time.setter
    def exposure_time(self, exposure_time: int):
        self.cap.ExposureTimeRaw.SetValue(int(exposure_time))

    @property
    def frame_size(self) -> tuple[int, int]:
        x = self.cap.Width() / 2
        y = self.cap.Height() / 2
        return (int(x), int(y))

    @frame_size.setter
    def frame_size(self, frame_size: tuple[int, int]):
        self.cap.Width.SetValue(frame_size[0])
        self.cap.Height.SetValue(frame_size[1])

    @property
    def frame_rate(self) -> float:
        return self.cap.AcquisitionFrameRateAbs.GetValue()

    @frame_rate.setter
    def frame_rate(self, frame_rate: float):
        self.cap.AcquisitionFrameRateAbs.SetValue(frame_rate)

    @property
    def max_value(self) -> int:
        return self.cap.PixelDynamicRangeMax()

    @property
    def shutter_cal_values(self) -> np.ndarray:
        return self._shutter_cal_values

    def close(self):
        """Closes the camera connection"""
        with et.ignored(Exception):
            super().close()
        with et.ignored(Exception):
            self.cap.Close()
