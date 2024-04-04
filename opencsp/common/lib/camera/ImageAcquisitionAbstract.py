from abc import abstractmethod, ABC, abstractproperty
import numpy as np


class ImageAcquisitionAbstract(ABC):
    def calibrate_exposure(self):
        """
        Sets the camera's exposure so that only 1% of pixels are above the set
        saturation threshold. Uses a binary search algorithm.

        """

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
        im = self.get_frame()
        if _check_saturated(im):
            raise ValueError('Minimum exposure value is too high; image still saturated.')

        # Checks that the maximum value is over-exposed
        self.exposure_time = exposure_values[-1]
        im = self.get_frame()
        if not _check_saturated(im):
            raise ValueError('Maximum exposure value is too low; image not saturated.')

        # Check if exposure is set
        max_iters = int(np.ceil(np.log2(exposure_values.size)) + 1)
        for i in range(max_iters):
            # Get next exposure index to test
            idx = _get_next_exposure_idx()

            # Set exposure
            self.exposure_time = exposure_values[idx]
            print('Trying: {}'.format(exposure_values[idx]))

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
            raise ValueError('Error with setting exposure.')

        # Set final exposure and print results
        exposure_value_set = exposure_values[_get_exposure_idxs()[0]]
        self.exposure_time = exposure_value_set
        print(f'Exposure set to: {exposure_value_set}')

    @abstractmethod
    def get_frame(self):
        """Gets a single frame from the camera"""
        pass

    @abstractproperty
    def gain(self):
        """Camera gain value"""
        pass

    @abstractproperty
    def exposure_time(self) -> int:
        """Camera exposure_time value (microseconds)"""
        pass

    @property
    def exposure_time_seconds(self) -> float:
        return self.exposure_time / 1_000_000

    @exposure_time_seconds.setter
    def exposure_time_seconds(self, exposure_time_seconds: float):
        self.exposure_time = int(exposure_time_seconds / 1_000_000)

    @abstractproperty
    def frame_size(self):
        """camera frame size (X pixels by Y pixels)"""
        pass

    @abstractproperty
    def frame_rate(self):
        """Camera frame rate (units of FPS)"""
        pass

    @abstractproperty
    def max_value(self):
        """
        Camera's maximum saturation value.
        Example: If camera outputs 8-bit images, max_value = 255

        """
        pass

    @abstractproperty
    def shutter_cal_values(self):
        """
        Returns camera exposure_time values to use when calibrating the exposure_time.

        """
        pass

    @abstractmethod
    def close(self):
        """Closes the camera connection"""
        pass
