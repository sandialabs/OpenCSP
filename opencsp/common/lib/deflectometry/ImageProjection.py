"""Class handling the projection of images on a monitor/projector
"""

import tkinter
from typing import Callable

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools
import opencsp.common.lib.tool.tk_tools as tkt


class CalParams:
    """Class holding parameters for ImageProjection calibration image

    Attributes
    ----------
    x/y_screen : ndarray[float]
        x/y fiducial coordinates in fractional screens
    x/y_pixel : ndarray[int]
        x/y fiducial coordinates in screen pixels
    """

    def __init__(self, size_x: int, size_y: int) -> 'CalParams':
        """Instantiates CalParams given size of active screen area.
        Measured from upper left corner of screen. Screen fractions
        measured from edges of screen, pixels measured from centers
        of image pixels. Fiducials should be measured from centers
        of image pixels.

        Parameters
        ----------
        size_x/y : int
            Size of active area of screen in x/y directions in pixels
        """
        # Distance from edges of image
        dx = int(0.01 * float(size_x))  # pixels
        dy = int(0.01 * float(size_y))  # pixels
        # Number of fiducials in x/y
        nx = 3
        ny = 3
        # Fiducial locations along axes
        x_ax = np.linspace(dx, size_x - dx, nx).astype(int)  # pixels
        y_ax = np.linspace(dy, size_y - dy, ny).astype(int)  # pixels
        self.x_pixel_axis: np.ndarray[int] = x_ax  # pixels
        self.y_pixel_axis: np.ndarray[int] = y_ax  # pixels
        self.x_screen_axis: np.ndarray[float] = 1 - (self.x_pixel_axis.astype(float) + 0.5) / float(size_x)
        self.y_screen_axis: np.ndarray[float] = 1 - (self.y_pixel_axis.astype(float) + 0.5) / float(size_y)
        # Fiducial x/y locations in image, pixels
        x_mat_pixel, y_mat_pixel = np.meshgrid(self.x_pixel_axis, self.y_pixel_axis)
        self.x_pixel: np.ndarray[int] = x_mat_pixel.flatten()
        self.y_pixel: np.ndarray[int] = y_mat_pixel.flatten()
        # Fiducial x/y locations in image, screen fractions
        x_mat_screen, y_mat_screen = np.meshgrid(self.x_screen_axis, self.y_screen_axis)
        self.x_screen: np.ndarray[float] = x_mat_screen.flatten()
        self.y_screen: np.ndarray[float] = y_mat_screen.flatten()
        # Point index
        self.index: np.ndarray[int] = np.arange(self.x_pixel.size, dtype=int)


class ImageProjection:
    _instance: 'ImageProjection' = None

    def __init__(self, root: tkinter.Tk, display_data: dict):
        """
        Image projection control.

        Parameters
        ----------
        root : tkinter.Tk
            Tk window root.
        display_data : dict
            Display geometry parameters

        """
        # Register this instance as the global instance
        if ImageProjection._instance is None:
            ImageProjection._instance = self

        # Declare instance variables
        self.is_closed = False
        self.on_close: list[Callable] = []

        # Save root
        self.root = root

        # Create drawing canvas
        self.canvas = tkinter.Canvas(self.root)
        self.canvas.pack()
        self.canvas.configure(background='black', highlightthickness=0)

        # Save active area data
        self.upate_window(display_data)

        # Make window frameless
        self.root.overrideredirect(1)

        # Make window always on top
        self.root.wm_attributes("-topmost", 1)

        # Set escape to exit window
        self.root.bind("<Escape>", lambda e: self.close())

        # Create black image
        image = self._format_image(
            np.zeros((self.win_size_y, self.win_size_x, 3), dtype=self.display_data['projector_data_type'])
        )
        self.canvas_image = self.canvas.create_image(0, 0, image=image, anchor='nw')

        # Show crosshairs
        self.show_crosshairs()

    def __del__(self):
        with et.ignored(Exception):
            self.close()

    @classmethod
    def instance(cls) -> "ImageProjection" | None:
        """Get the global ImageProjection instance, if one is available.

        We use the singleton design pattern (single global instance) for this class because we don't expect there to be
        more than one projector in the system.
        """
        return cls._instance

    def run(self):
        """Runs the Tkinter instance"""
        self.root.mainloop()

    @classmethod
    def in_new_window(cls, display_data: dict):
        """
        Instantiates ImageProjection object in new window.

        Parameters
        ----------
        display_data : dict
            Display data used to create window.

        Returns
        -------
        ImageProjection object.

        """
        # Create new tkinter window
        root = tkt.window()
        # Instantiate class
        return cls(root, display_data)

    def upate_window(self, display_data: dict) -> None:
        """
        Updates window display data.

        Parameters
        ----------
        display_data : dict
            Input data for position/shift/timing/etc.

        """
        # Save display_data
        self.display_data = display_data

        # Save window position data
        self.win_size_x = display_data['win_size_x']
        self.win_size_y = display_data['win_size_y']
        self.win_position_x = display_data['win_position_x']
        self.win_position_y = display_data['win_position_y']

        # Save active area position data
        self.size_x = display_data['size_x']
        self.size_y = display_data['size_y']
        self.position_x = display_data['position_x']
        self.position_y = display_data['position_y']

        # Save maximum projector data integer
        self.max_int = display_data['projector_max_int']
        self.dtype = display_data['projector_data_type']

        # Calculate active area extents
        self.x_active_1 = self.position_x
        self.x_active_2 = self.position_x + self.size_x
        self.y_active_1 = self.position_y
        self.y_active_2 = self.position_y + self.size_y

        # Calculate center of active area
        self.x_active_mid = int(self.size_x / 2)
        self.y_active_mid = int(self.size_y / 2)

        # Resize window
        self.root.geometry(
            '{:d}x{:d}+{:d}+{:d}'.format(self.win_size_x, self.win_size_y, self.win_position_x, self.win_position_y)
        )

        # Resize canvas size
        self.canvas.configure(width=self.win_size_x, height=self.win_size_y)

    def show_crosshairs(self) -> None:
        """
        Shows crosshairs on display.

        """
        # Add white active region
        array = np.ones((self.size_y, self.size_x, 3), dtype=self.display_data['projector_data_type']) * self.max_int

        # Add crosshairs vertical
        array[:, self.x_active_mid, :] = 0
        array[self.y_active_mid, :, :] = 0

        # Add crosshairs diagonal
        width = np.min([self.size_x, self.size_y])
        xd1 = int(self.x_active_mid - width / 4)
        xd2 = int(self.x_active_mid + width / 4)
        yd1 = int(self.y_active_mid - width / 4)
        yd2 = int(self.y_active_mid + width / 4)
        xds = np.arange(xd1, xd2, dtype=int)
        yds = np.arange(yd1, yd2, dtype=int)
        array[yds, xds, :] = 0
        array[yds, np.flip(xds) + 1, :] = 0

        # Display image
        self.display_image_in_active_area(array)

    def show_axes(self) -> None:
        """
        Shows the X and Y axes directions on display.

        """
        # Add white active region
        array = np.ones((self.size_y, self.size_x, 3), dtype=self.display_data['projector_data_type']) * self.max_int

        # Add arrows
        width = int(np.min([self.size_x, self.size_y]) / 4)
        thickness = 5

        # Add green X axis arrow
        start_point = (self.x_active_mid, self.y_active_mid)
        end_point = (self.x_active_mid - width, self.y_active_mid)
        color = (int(self.max_int), 0, 0)  # RGB
        array = cv.arrowedLine(array, start_point, end_point, color, thickness)

        # Add red Y axis arrow
        start_point = (self.x_active_mid, self.y_active_mid)
        end_point = (self.x_active_mid, self.y_active_mid + width)
        color = (0, int(self.max_int), 0)  # RGB
        array = cv.arrowedLine(array, start_point, end_point, color, thickness)

        # Add X text
        font = cv.FONT_HERSHEY_PLAIN
        array = cv.putText(
            array,
            'X',
            (self.x_active_mid - width - 20, self.y_active_mid + 20),
            font,
            6,
            (int(self.max_int), 0, 0),
            2,
            bottomLeftOrigin=True,
        )

        # Add Y text
        font = cv.FONT_HERSHEY_PLAIN
        array = cv.putText(
            array, 'Y', (self.x_active_mid + 20, self.y_active_mid + width + 20), font, 6, (0, int(self.max_int), 0), 2
        )

        # Display image
        self.display_image_in_active_area(array)

    def show_calibration_image(self):
        """Shows a calibration image with N fiducials. Fiducials are black dots
        on a white background. Fiducial locations measured from center of dots.
        """
        # Create base image to show
        array = np.ones((self.size_y, self.size_x, 3), dtype=self.dtype)

        # Get calibration pattern parameters
        pattern_params = CalParams(self.size_x, self.size_y)

        # Add fiducials
        for x_loc, y_loc, idx in zip(pattern_params.x_pixel, pattern_params.y_pixel, pattern_params.index):
            # Place fiducial
            array[y_loc, x_loc, 1] = self.max_int
            # Place label (offset so label is in view)
            x_pt_to_center = float(self.size_x) / 2 - x_loc
            y_pt_to_center = float(self.size_y) / 2 - y_loc
            if x_pt_to_center >= 0:
                dx = 15
            else:
                dx = -35
            if y_pt_to_center >= 0:
                dy = 20
            else:
                dy = -10
            # Draw text
            cv.putText(array, f'{idx:d}', (x_loc + dx, y_loc + dy), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        # Display with black border
        self.display_image_in_active_area(array)

    def zeros(self) -> None:
        """
        Creates a black image to fill the entire projected area.

        Returns:
        --------
        image : np.ndarray
            A 2D image with shape (self.size_x, self.size_y, 3), filled with zeros
        """
        # Create black image
        black_image = np.zeros(
            (self.size_y, self.size_x, 3),
            dtype=self.display_data['projector_data_type'],
        )

        return black_image

    def display_image(self, array: np.ndarray) -> None:
        """
        Formats and displays input numpy array in entire window.

        Parameters
        ----------
        array : ndarray
            NxMx3 image array. Data must be int ranging from 0 to self.max_int.
            Array XY shape must match window size in pixels.

        """
        # Check image is RGB
        if np.ndim(array) != 3 or array.shape[2] != 3:
            raise ValueError('Input array must have 3 dimensions and dimension 2 must be length 3.')

        # Check array is correct xy shape
        if array.shape[0] != self.win_size_y or array.shape[1] != self.win_size_x:
            raise ValueError(
                'Input image incorrect size. Input image size is {}, but frame size is {}.'.format(
                    array.shape[:2], (self.win_size_y, self.win_size_x)
                )
            )

        # Format image
        image = self._format_image(array)

        # Display image
        self.canvas.imgref = image
        self.canvas.itemconfig(self.canvas_image, image=image)

    def display_image_in_active_area(self, array: np.ndarray) -> None:
        """Formats and displays input numpy array in active area only. Input
        array must have size (self.size_y, self.size_x, 3) and is displayed
        with a black border to fill entire window area.

        Parameters
        ----------
        array : ndarray
            NxMx3 image array. Data must be int ranging from 0 to self.max_int.
            Array XY shape must match active window area size in pixels.

        """
        # Check image is RGB
        if np.ndim(array) != 3 or array.shape[2] != 3:
            raise ValueError('Input array must have 3 dimensions and dimension 2 must be length 3.')

        # Check array is correct xy shape
        if array.shape[0] != self.size_y or array.shape[1] != self.size_x:
            raise ValueError(
                'Input image incorrect size. Input image size is {}, but frame size is {}.'.format(
                    array.shape[:2], (self.size_y, self.size_x)
                )
            )

        # Create black image and place array in correct position
        array_out = np.zeros((self.win_size_y, self.win_size_x, 3), dtype=self.display_data['projector_data_type'])
        array_out[self.y_active_1 : self.y_active_2, self.x_active_1 : self.x_active_2, :] = array

        # Display image
        self.display_image(array_out)

    def _format_image(self, array: np.ndarray) -> ImageTk.PhotoImage:
        """
        Blue/Red shifts and formats input image to Tkinter format

        Parameters
        ----------
        array : MxNx3 ndarray
            Input RGB numpy array.

        Returns
        -------
        ImageTk.PhotoImage
            Formatted image.

        """
        # Shift red channel
        array[..., 0] = np.roll(array[..., 0], self.display_data['shift_red_x'], 1)
        array[..., 0] = np.roll(array[..., 0], self.display_data['shift_red_y'], 0)

        # Shift blue channel
        array[..., 2] = np.roll(array[..., 2], self.display_data['shift_blue_x'], 1)
        array[..., 2] = np.roll(array[..., 2], self.display_data['shift_blue_y'], 0)

        # Format array into tkinter format
        image = Image.fromarray(array, 'RGB')
        return ImageTk.PhotoImage(image)

    @staticmethod
    def load_from_hdf(file: str) -> dict:
        """
        Loads and returns ImageProjection data from HDF file.

        Parameters
        ----------
        file : str
            HDF file path.

        Returns
        -------
        dict
            Display data.

        """
        # Load data
        datasets = [
            'ImageProjection/name',
            'ImageProjection/win_size_x',
            'ImageProjection/win_size_y',
            'ImageProjection/win_position_x',
            'ImageProjection/win_position_y',
            'ImageProjection/size_x',
            'ImageProjection/size_y',
            'ImageProjection/position_x',
            'ImageProjection/position_y',
            'ImageProjection/projector_data_type',
            'ImageProjection/projector_max_int',
            'ImageProjection/image_delay',
            'ImageProjection/shift_red_x',
            'ImageProjection/shift_red_y',
            'ImageProjection/shift_blue_x',
            'ImageProjection/shift_blue_y',
            'ImageProjection/ui_position_x',
        ]
        return hdf5_tools.load_hdf5_datasets(datasets, file)

    @classmethod
    def load_from_hdf_and_display(cls, file: str):
        """
        Loads data from HDF and opens window.

        Parameters
        ----------
        file : str
            HDF file path.

        """
        # Load data
        display_data = cls.load_from_hdf(file)

        # Open window
        return cls.in_new_window(display_data)

    @staticmethod
    def save_to_hdf(display_data: dict, file: str):
        """Saves ImageProjection parameters to HDF file

        Parameters
        ----------
        display_data : dict
            ImageProjection parameters in dictionary form
        file : str
            HDF file to save
        """
        # Extract data entries
        datasets = []
        data = []
        for field in display_data.keys():
            datasets.append('ImageProjection/' + field)
            data.append(display_data[field])

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)

    def close(self):
        """Closes all windows"""
        # don't double-close in order to avoid warnings
        if self.is_closed:
            return
        self.is_closed = True

        # unset as the global instance
        if ImageProjection._instance == self:
            ImageProjection._instance = None

        # callbacks
        for callback in self.on_close:
            with et.ignored(Exception):
                callback()

        with et.ignored(Exception):
            self.root.destroy()
