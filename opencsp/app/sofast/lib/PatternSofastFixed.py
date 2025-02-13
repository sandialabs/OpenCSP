from typing import Literal

import numpy as np
from numpy import ndarray
from scipy.signal import convolve2d


class PatternSofastFixed:
    """Class that holds parameters for displaying a Fixed Pattern for use
    in fixed pattern deflectometry.
    """

    def __init__(self, size_x: int, size_y: int, width_pattern: int, spacing_pattern: int) -> "PatternSofastFixed":
        """Instantiates PatternSofastFixed class from screen geometry parameters

        Parameters
        ----------
        size_x/size_y : int
            Size of image x/y dimension
        width_pattern : int
            Width of each Fixed Pattern marker in the image, pixels
        spacing_pattern : int
            Spacing between (not center-to-center) pattern markers, pixels

        Attributes
        ----------
        - size_x
        - size_y
        - width_pattern
        - spacing_pattern
        - x_locs_pixel
        - y_locs_pixel
        - nx
        - ny
        - x_locs_frac
        - y_locs_frac
        - x_indices
        - y_indices
        """
        # Store data
        self.size_x = size_x
        self.size_y = size_y
        self.width_pattern = width_pattern
        self.spacing_pattern = spacing_pattern

        # Create location vectors
        x_locs_pixel = np.arange(0, size_x, width_pattern + spacing_pattern)
        y_locs_pixel = np.arange(0, size_y, width_pattern + spacing_pattern)

        # Make vectors odd
        if x_locs_pixel.size % 2 == 0:
            x_locs_pixel = x_locs_pixel[:-1]
        if y_locs_pixel.size % 2 == 0:
            y_locs_pixel = y_locs_pixel[:-1]

        # Center in image
        dx = int(float((size_x - 1) - x_locs_pixel[-1]) / 2)
        dy = int(float((size_y - 1) - y_locs_pixel[-1]) / 2)
        x_locs_pixel += dx
        y_locs_pixel += dy

        # Save calculations
        self.x_locs_pixel = x_locs_pixel
        self.y_locs_pixel = y_locs_pixel
        self.nx = x_locs_pixel.size
        self.ny = y_locs_pixel.size

        # Calculate point indices
        self.x_indices = np.arange(self.nx) - int(np.floor(float(self.nx) / 2))
        self.y_indices = np.arange(self.ny) - int(np.floor(float(self.ny) / 2))

        # Calculate fractional screen points
        self.x_locs_frac = x_locs_pixel.astype(float) / float(size_x)
        self.y_locs_frac = y_locs_pixel.astype(float) / float(size_y)

    def _get_dot_image(self, dot_shape: str) -> ndarray[float]:
        """Returns 2d image of individual pattern element. Active area is 1
        and inactive area is 0, dtype float.
        """
        if dot_shape not in ["circle", "square"]:
            raise ValueError(f'pattern_type must be one of ["circle", "square"], not {dot_shape:s}')

        if dot_shape == "square":
            return np.ones((self.width_pattern, self.width_pattern), dtype=float)
        elif dot_shape == "circle":
            x, y = np.meshgrid(np.arange(self.width_pattern, dtype=float), np.arange(self.width_pattern, dtype=float))
            x -= x.mean()
            y -= y.mean()
            r = np.sqrt(x**2 + y**2)
            return (r < float(self.width_pattern) / 2).astype(float)

    def get_image(self, dtype: str, max_int: int, dot_shape: Literal["circle", "square"] = "circle") -> ndarray:
        """Creates a NxMx3 fixed pattern image

        Parameters
        ----------
        dtype : str
            Output datatype of image
        max_int : int
            Integer value corresponding to "white." Zero corresponds to black.
        dot_shape : str
            'circle' or 'square'

        Returns
        -------
        ndarray
            (size_y, size_x, 3) ndarray
        """
        # Create image with point locations
        image = np.zeros((self.size_y, self.size_x), dtype=dtype)
        locs_x, locs_y = np.meshgrid(self.x_locs_pixel, self.y_locs_pixel)
        image[locs_y, locs_x] = 1

        # Add patterns (active=1)
        pattern = self._get_dot_image(dot_shape).astype(dtype)
        image = convolve2d(image, pattern, mode="same")

        # Convert image to white background and black patterns
        image = (1 - image) * max_int

        # Add RGB channels
        return np.concatenate([image[..., None]] * 3, axis=2)
