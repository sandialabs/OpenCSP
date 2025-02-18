import numpy as np


class Fringes:
    def __init__(self, periods_x: list, periods_y: list):
        """
        Representation of projected fringe images.

        Parameters
        ----------
        periods_x : list
            Fringe X periods to display, units of fractional screens.
        periods_y : list
            Fringe Y periods to display, units of fractional screens.

        """
        # Save fringe data
        self.periods_x = periods_x
        self.periods_y = periods_y

        # Calculations/constants
        self.phase_shifts_x = int(4)
        self.phase_shifts_y = int(4)
        self.num_x_images = np.size(periods_x) * self.phase_shifts_x
        self.num_y_images = np.size(periods_y) * self.phase_shifts_y
        self.num_images = self.num_y_images + self.num_x_images

    @classmethod
    def from_num_periods(cls, fringe_periods_x=4, fringe_periods_y=4) -> "Fringes":
        """Creates fringes to be displayed during run_measurement().

        The fringes are displayed as sinusoidal grayscale images. A value of 1 means that only a single large sinusoidal
        will be used. A higher value will display more images with a faster sinusoidal. Therefore, a higher value will
        result in finer slope resolution, up to the resolving power of the optical setup.

        Note that a different number of periods may be required for x and for y, if the x and y resolutions differ
        enough.

        Params:
        -------
        fringe_periods_x: int, optional
            Granularity for fringe periods in the x direction. Defaults to 4.
        fringe_periods_y: int, optional
            Granularity for fringe periods in the y direction. Defaults to 4.

        Returns:
            fringes: The fringes object, to be used with run_measurement().
        """
        # Get fringe periods
        periods_x = [4**idx for idx in range(fringe_periods_x)]
        periods_y = [4**idx for idx in range(fringe_periods_y)]
        periods_x[0] -= 0.1
        periods_y[0] -= 0.1

        # Create fringe object
        fringes = cls(periods_x, periods_y)

        return fringes

    def get_frames(self, x: int, y: int, dtype: str, range_: list[float, float]) -> np.ndarray:
        """
        Returns 3D ndarray of scaled, monochrome fringe images.

        Parameters
        ----------
        x/y : int
            Size of image in x/y.
        dtype : str
            Data type accepted by image display.
        range_ : list[float, float]
            The range of the output sinusoids. [min, max].

        Returns
        -------
        images : ndarray
            3D numpy array of scaled images.

        """
        # Create numpy array container
        images = np.ones((y, x, self.num_images))  # float

        # Create sinusoids [-1, 1]
        y_sinusoids = self.get_sinusoids(y, self.periods_y, self.phase_shifts_y)  # float
        x_sinusoids = self.get_sinusoids(x, self.periods_x, self.phase_shifts_x)  # float

        # Create y fringes [-1, 1]
        for idx, sinusoid in enumerate(y_sinusoids):
            images[..., idx] *= sinusoid[:, np.newaxis]  # float

        # Create x fringes [-1, 1]
        for idx, sinusoid in enumerate(x_sinusoids):
            images[..., idx + self.num_y_images] *= sinusoid[np.newaxis, :]  # float

        # Modify value range to [0, 1]
        images = images / 2 + 0.5  # float

        # Apply scaling function [value_min, value_max]
        images *= range_[1] - range_[0]  # float
        images += range_[0]  # float

        # Convert to output data_type
        images = images.astype(dtype)  # output dtype

        return images

    @staticmethod
    def get_sinusoids(length: int, periods: list[float], phase_shifts: int) -> list[np.ndarray]:
        """
        Creates list of phase shifted sinusoids for given periods ranging from
        -1 to 1.

        Parameters
        ----------
        length : int
            Number of samples to generate over width.
        periods : list[float, ...]
            List of periods to make sinusoids for, fractional widths.
        phase_shifts : int
            Number of phase shifts.

        Returns
        -------
        sinusoids : list[np.ndarray, ...]
            List of 1d nparrays. Float, magnitude ranges from -1 to 1

        """
        # Create fringe sinusoids
        sinusoids = []
        for period in periods:
            # Create vector
            v_rad = np.linspace(1, 0, length) * period * 2 * np.pi  # radian
            for shift_idx in range(phase_shifts):
                # Calculate shift
                shift = shift_idx * np.pi / 2  # radians
                # Shift vector
                v_rad_shift = v_rad - shift  # radians
                # Create sinusoid
                v_mag = np.cos(v_rad_shift)  # magnitude
                sinusoids.append(v_mag)

        return sinusoids
