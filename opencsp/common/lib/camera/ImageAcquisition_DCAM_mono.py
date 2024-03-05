import numpy as np
from   pypylon import pylon

from   opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract


class ImageAcquisition(ImageAcquisitionAbstract):
    def __init__(self, instance: int = 0, pixel_format: str = 'Mono8'):
        """
        Class to control a Basler DCAM monochromatic camera. Grabs one frame
        at a time. The gain is initially set to the lowest possible value.

        Parameters
        ----------
        instance : int
            The Nth instance of cameras (sorted by serial number) to instantiate.
        pixel_format : str
            Available formats include:
                - Mono8 (default)
                - Mono12
                - Mono12Packed
                - Others as defined by Basler

        """
        # Find all instances of DCAM cameras
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        # Check that at least one camera is available
        if len(devices) == 0:
            raise pylon.RuntimeException('No cameras found.')
        else:
            print(f'{len(devices):d} devices found.')
            for idx, device in enumerate(devices):
                if idx == instance:
                    print('--> ', end='')
                else:
                    print('    ', end='')
                print(device.GetFriendlyName())

        # Check number of instances
        if instance >= len(devices):
            raise ValueError(f'Cannot load instance {instance:d}. Only {len(devices):d} devices found.')

        # Connect to camera
        self.cap = pylon.InstantCamera(tlFactory.CreateDevice(devices[instance]))
        self.cap.Open()

        # Set up device to single frame acquisition
        self.cap.AcquisitionMode.SetValue('SingleFrame')

        # Set pixel format
        self.cap.PixelFormat.SetValue(pixel_format)

        # Set gain to minimum value
        self.cap.GainRaw.SetValue(self.cap.GainRaw.Min)

        # Set exposure values to be stepped over when performing exposure calibration
        shutter_min = self.cap.ExposureTimeRaw.Min
        shutter_max = self.cap.ExposureTimeRaw.Max
        self._shutter_cal_values = np.linspace(shutter_min, shutter_max, 2**13).astype(int)

    def get_frame(self) -> np.ndarray:
        # Start frame capture
        self.cap.StartGrabbingMax(1)
        grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Access image data
        if grabResult.GrabSucceeded():
            img = grabResult.Array
        else:
            raise pylon.RuntimeException('Frame grab unsuccessful.')

        # Wait for capturing to finish
        grabResult.Release()

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
        x = self.cap.Width()
        y = self.cap.Height()
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
        self.cap.Close()
