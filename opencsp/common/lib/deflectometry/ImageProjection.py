"""Class handling the projection of images on a monitor/projector
"""

from dataclasses import dataclass
import tkinter
from typing import Callable, Optional

import cv2 as cv
import cv2.aruco as aruco
import numpy as np
from PIL import Image, ImageTk

import opencsp.common.lib.tool.exception_tools as et
from opencsp.common.lib.tool import hdf5_tools
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

    def __init__(self, size_x: int, size_y: int) -> "CalParams":
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
        # Width of Aruco markers
        self.marker_width: int = int(min(size_x, size_y) * 0.2)


@dataclass
class ImageProjectionData(hdf5_tools.HDF5_IO_Abstract):
    """Dataclass containting ImageProjection parameters used to define the image/screen
    geometry when projecting an image on a display.
    * All position/size/shift units are screen pixels.
    * The *active_area* is the region which is actively used to project images, as opposed to the
      black background.
    * The *main_window* is the tkinter window which holds the *active_area* region. Typically, the
      main_window is the size of the screen being projected onto. Any area of the main_window
      that is not filled by the active_area is filled with black. This is useful with reducing
      light.
    * When defining window positions, the reference point is the upper-left corner of the
      window/screen.
    """

    name: str
    """The name of the ImageProjection configuration"""
    main_window_size_x: int
    """The width of the main window. (For more information on main_window, see class docstring)"""
    main_window_size_y: int
    """The height of the main window. (For more information on main_window, see class docstring)"""
    main_window_position_x: int
    """The x position of the upper-left corner of the main window relative to the upper-left corner
    of the projection screen. Right is positive. (For more information on main_window, see class docstring)"""
    main_window_position_y: int
    """The y position of the upper-left corner of the main window relative to the upper-left corner
    of the projection screen. Down is positive. (For more information on main_window, see class docstring)"""
    active_area_size_x: int
    """The width of the active area within the main window. (For more information on active_area, see class docstring)"""
    active_area_size_y: int
    """The height of the active area within the main window. (For more information on active_area, see class docstring)"""
    active_area_position_x: int
    """The x position of the upper-left corner of the active area relative to the upper-left corner
    of the main window. Right is positive (For more information on main_window and active_area, see class docstring)"""
    active_area_position_y: int
    """The y position of the upper-left corner of the active area relative to the upper-left corner
    of the main window. Down is positive. (For more information on main_window and active_area, see class docstring)"""
    projector_data_type: str
    """The data type string to be sent to the projector. In most cases, this will be an unsigned 8-bit integer ('uint8')"""
    projector_max_int: int
    """The integer value that corresponds to a perfectly white image on the screen. In most cases
    this will be 255."""
    image_delay_ms: float
    """The delay between the display being sent an image to display, and when the camera should start recording. This
    is specific to each display/camera/computer setup."""
    shift_red_x: int
    """The red channel x shift in an RGB display relative to green in pixels. Right is positive."""
    shift_red_y: int
    """The red channel y shift in an RGB display relative to green in pixels. Down is positive."""
    shift_blue_x: int
    """The blue channel x shift in an RGB display relative to green in pixels. Right is positive."""
    shift_blue_y: int
    """The blue channel y shift in an RGB display relative to green in pixels. Down is positive."""

    def save_to_hdf(self, file: str, prefix: str = "") -> None:
        datasets = []
        data = []
        for name, value in self.__dict__.items():
            datasets.append(prefix + "ImageProjection/" + name)
            data.append(value)

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ""):
        # Load data
        datasets = [
            "ImageProjection/name",
            "ImageProjection/main_window_size_x",
            "ImageProjection/main_window_size_y",
            "ImageProjection/main_window_position_x",
            "ImageProjection/main_window_position_y",
            "ImageProjection/active_area_size_x",
            "ImageProjection/active_area_size_y",
            "ImageProjection/active_area_position_x",
            "ImageProjection/active_area_position_y",
            "ImageProjection/projector_data_type",
            "ImageProjection/projector_max_int",
            "ImageProjection/image_delay_ms",
            "ImageProjection/shift_red_x",
            "ImageProjection/shift_red_y",
            "ImageProjection/shift_blue_x",
            "ImageProjection/shift_blue_y",
        ]
        for dataset in datasets:
            dataset = prefix + dataset

        kwargs = hdf5_tools.load_hdf5_datasets(datasets, file)

        return cls(**kwargs)


class ImageProjection(hdf5_tools.HDF5_IO_Abstract):
    """Controls projecting an image on a computer display (projector, monitor, etc.)"""

    _instance: "ImageProjection" = None

    def __init__(self, root: tkinter.Tk, display_data: ImageProjectionData) -> "ImageProjection":
        """Instantiates class

        Parameters
        ----------
        root : tkinter.Tk
            Tk window root.
        display_data : ImageProjectionData
            Display geometry parameters

        """
        # Register this instance as the global instance
        if ImageProjection._instance is None:
            ImageProjection._instance = self

        # Declare instance variables
        self.is_closed = False
        self.on_close: list[Callable[[ImageProjection], None]] = []

        # Save root
        self.root = root

        # Create drawing canvas
        self.canvas = tkinter.Canvas(self.root)
        self.canvas.pack()
        self.canvas.configure(background="black", highlightthickness=0)

        # Save aruco marker dictionary for calibration image generation
        self.aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

        # Save active area data
        self._x_active_1: int
        self._x_active_2: int
        self._y_active_1: int
        self._y_active_2: int
        self._x_active_mid: int
        self._y_active_mid: int

        self.display_data = display_data
        self.update_window()

        # Make window frameless
        self.root.overrideredirect(1)

        # Make window always on top
        self.root.wm_attributes("-topmost", 1)

        # Set escape to exit window
        self.root.bind("<Escape>", lambda e: self.close())

        # Create black image
        image = self._format_image(
            np.zeros(
                (self.display_data.main_window_size_y, self.display_data.main_window_size_x, 3),
                dtype=self.display_data.projector_data_type,
            )
        )
        self.canvas_image = self.canvas.create_image(0, 0, image=image, anchor="nw")

        # Show crosshairs
        self.show_crosshairs()

    def __del__(self):
        with et.ignored(Exception):
            self.close()

    @classmethod
    def instance(cls) -> Optional["ImageProjection"]:
        """Get the global ImageProjection instance, if one is available.

        We use the singleton design pattern (single global instance) for this class because we don't expect there to be
        more than one projector in the system.
        """
        return cls._instance

    def run(self) -> None:
        """Runs the Tkinter instance"""
        self.root.mainloop()

    @classmethod
    def in_new_window(cls, display_data: ImageProjectionData) -> "ImageProjection":
        """
        Instantiates ImageProjection object in new window.

        Parameters
        ----------
        display_data : ImageProjectionData
            Display data used to create window.

        Returns
        -------
        ImageProjection object.

        """
        # Create new tkinter window
        root = tkt.window()
        # Instantiate class
        return cls(root, display_data)

    def update_window(self) -> None:
        """Updates window display data."""
        # Calculate active area extents
        self._x_active_1 = self.display_data.active_area_position_x
        self._x_active_2 = self.display_data.active_area_position_x + self.display_data.active_area_size_x
        self._y_active_1 = self.display_data.active_area_position_y
        self._y_active_2 = self.display_data.active_area_position_y + self.display_data.active_area_size_y

        # Calculate center of active area
        self._x_active_mid = int(self.display_data.active_area_size_x / 2)
        self._y_active_mid = int(self.display_data.active_area_size_y / 2)

        # Resize window
        self.root.geometry(
            f"{self.display_data.main_window_size_x:d}x{self.display_data.main_window_size_y:d}"
            + f"+{self.display_data.main_window_position_x:d}+{self.display_data.main_window_position_y:d}"
        )

        # Resize canvas size
        self.canvas.configure(width=self.display_data.main_window_size_x, height=self.display_data.main_window_size_y)

    def show_crosshairs(self) -> None:
        """
        Shows crosshairs on display.

        """
        # Add white active region
        array = (
            np.ones(
                (self.display_data.active_area_size_y, self.display_data.active_area_size_x, 3),
                dtype=self.display_data.projector_data_type,
            )
            * self.display_data.projector_max_int
        )

        # Add crosshairs vertical
        array[:, self._x_active_mid, :] = 0
        array[self._y_active_mid, :, :] = 0

        # Add crosshairs diagonal
        width = np.min([self.display_data.active_area_size_x, self.display_data.active_area_size_y])
        xd1 = int(self._x_active_mid - width / 4)
        xd2 = int(self._x_active_mid + width / 4)
        yd1 = int(self._y_active_mid - width / 4)
        yd2 = int(self._y_active_mid + width / 4)
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
        array = (
            np.ones(
                (self.display_data.active_area_size_y, self.display_data.active_area_size_x, 3),
                dtype=self.display_data.projector_data_type,
            )
            * self.display_data.projector_max_int
        )

        # Add arrows
        width = int(np.min([self.display_data.active_area_size_x, self.display_data.active_area_size_y]) / 4)
        thickness = 5

        # Add green X axis arrow
        start_point = (self._x_active_mid, self._y_active_mid)
        end_point = (self._x_active_mid - width, self._y_active_mid)
        color = (int(self.display_data.projector_max_int), 0, 0)  # RGB
        array = cv.arrowedLine(array, start_point, end_point, color, thickness)

        # Add red Y axis arrow
        start_point = (self._x_active_mid, self._y_active_mid)
        end_point = (self._x_active_mid, self._y_active_mid + width)
        color = (0, int(self.display_data.projector_max_int), 0)  # RGB
        array = cv.arrowedLine(array, start_point, end_point, color, thickness)

        # Add X text
        font = cv.FONT_HERSHEY_PLAIN
        array = cv.putText(
            array,
            "X",
            (self._x_active_mid - width - 20, self._y_active_mid + 20),
            font,
            6,
            (int(self.display_data.projector_max_int), 0, 0),
            2,
            bottomLeftOrigin=True,
        )

        # Add Y text
        font = cv.FONT_HERSHEY_PLAIN
        array = cv.putText(
            array,
            "Y",
            (self._x_active_mid + 20, self._y_active_mid + width + 20),
            font,
            6,
            (0, int(self.display_data.projector_max_int), 0),
            2,
        )

        # Display image
        self.display_image_in_active_area(array)

    def show_calibration_fiducial_image(self):
        """Shows a calibration image with N fiducials. Fiducials are green dots
        on a white background. Fiducial locations measured from center of dots.
        """
        # Create base image to show
        array = np.zeros(
            (self.display_data.active_area_size_y, self.display_data.active_area_size_x, 3),
            dtype=self.display_data.projector_data_type,
        )

        # Get calibration pattern parameters
        pattern_params = CalParams(self.display_data.active_area_size_x, self.display_data.active_area_size_y)

        # Add fiducials
        for x_loc, y_loc, idx in zip(pattern_params.x_pixel, pattern_params.y_pixel, pattern_params.index):
            # Place fiducial
            array[y_loc, x_loc, 1] = self.display_data.projector_max_int
            # Place label (offset so label is in view)
            x_pt_to_center = float(self.display_data.active_area_size_x) / 2 - x_loc
            y_pt_to_center = float(self.display_data.active_area_size_y) / 2 - y_loc
            if x_pt_to_center >= 0:
                dx = 15
            else:
                dx = -35
            if y_pt_to_center >= 0:
                dy = 20
            else:
                dy = -10
            # Draw text
            cv.putText(array, f"{idx:d}", (x_loc + dx, y_loc + dy), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        # Display with black border
        self.display_image_in_active_area(array)

    def show_calibration_marker_image(self):
        """Shows a calibration image with N Aruco markers. Markers are black
        on a white background.
        """
        # Create base image to show
        array = (
            np.ones(
                (self.display_data.active_area_size_y, self.display_data.active_area_size_x, 3),
                dtype=self.display_data.projector_data_type,
            )
            * self.display_data.projector_max_int
        )

        # Get calibration pattern parameters
        pattern_params = CalParams(self.display_data.active_area_size_x, self.display_data.active_area_size_y)
        w = pattern_params.marker_width

        # Add markers
        for x_loc, y_loc, idx in zip(pattern_params.x_pixel, pattern_params.y_pixel, pattern_params.index):
            # Make marker
            img_mkr = self._make_aruco_marker(w, idx)
            # Place marker and label
            x_pt_to_center = float(self.display_data.active_area_size_x) / 2 - x_loc
            y_pt_to_center = float(self.display_data.active_area_size_y) / 2 - y_loc

            if (x_pt_to_center >= 0) and (y_pt_to_center >= 0):  # upper left quadrant
                array[y_loc : y_loc + w, x_loc : x_loc + w, :] = img_mkr
                dy = int(w / 2)
                dx = w + 5
            elif x_pt_to_center >= 0:  # lower left quadrant
                array[y_loc - w : y_loc, x_loc : x_loc + w, :] = np.rot90(img_mkr, 1)
                dy = -int(w / 2)
                dx = w + 5
            elif y_pt_to_center >= 0:  # top right quadrant
                array[y_loc : y_loc + w, x_loc - w : x_loc, :] = np.rot90(img_mkr, 3)
                dy = int(w / 2)
                dx = -w - 15
            else:  # bottom right quadrant
                array[y_loc - w : y_loc, x_loc - w : x_loc, :] = np.rot90(img_mkr, 2)
                dy = -int(w / 2)
                dx = -w - 15
            # Draw text
            cv.putText(array, f"{idx:d}", (x_loc + dx, y_loc + dy), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

        # Display with black border
        self.display_image_in_active_area(array)

    def get_black_array_active_area(self) -> np.ndarray:
        """
        Creates a black image to fill the active area of self.display_data.active_area_size_y by self.display_data.active_area_size_x pixels.

        Returns:
        --------
        image : np.ndarray
            A 2D image with shape (self.display_data.active_area_size_y, self.display_data.active_area_size_x, 3), filled with zeros
        """
        # Create black image
        black_image = np.zeros(
            (self.display_data.active_area_size_y, self.display_data.active_area_size_x, 3),
            dtype=self.display_data.projector_data_type,
        )

        return black_image

    def display_image(self, array: np.ndarray) -> None:
        """
        Formats and displays input numpy array in entire window.

        Parameters
        ----------
        array : ndarray
            NxMx3 image array. Data must be int ranging from 0 to self.display_data.projector_max_int.
            Array XY shape must match window size in pixels.

        """
        # Check image is RGB
        if np.ndim(array) != 3 or array.shape[2] != 3:
            raise ValueError("Input array must have 3 dimensions and dimension 2 must be length 3.")

        # Check array is correct xy shape
        if (
            array.shape[0] != self.display_data.main_window_size_y
            or array.shape[1] != self.display_data.main_window_size_x
        ):
            raise ValueError(
                f"Input image incorrect size. Input image size is {array.shape[:2]:d},"
                + f" but frame size is {self.display_data.main_window_size_y:d}x"
                + f"{self.display_data.main_window_size_x:d}."
            )

        # Format image
        image = self._format_image(array)

        # Display image
        self.canvas.imgref = image  # So garbage collector doesn't collect `image`
        self.canvas.itemconfig(self.canvas_image, image=image)

    def display_image_in_active_area(self, array: np.ndarray) -> None:
        """Formats and displays input numpy array in active area only. Input
        array must have size (self.display_data.active_area_size_y, self.display_data.active_area_size_x, 3) and is displayed
        with a black border to fill entire window area.

        Parameters
        ----------
        array : ndarray
            NxMx3 image array. Data must be int ranging from 0 to self.display_data.projector_max_int.
            Array XY shape must match active window area size in pixels.

        """
        # Check image is RGB
        if np.ndim(array) != 3 or array.shape[2] != 3:
            raise ValueError("Input array must have 3 dimensions and dimension 2 must be length 3.")

        # Check array is correct xy shape
        if (
            array.shape[0] != self.display_data.active_area_size_y
            or array.shape[1] != self.display_data.active_area_size_x
        ):
            raise ValueError(
                f"Input image incorrect size. Input image size is {array.shape[:2]:d},"
                + f" but frame size is {self.display_data.active_area_size_y:d}x"
                + f"{self.display_data.active_area_size_x:d}."
            )

        # Create black image and place array in correct position
        array_out = np.zeros(
            (self.display_data.main_window_size_y, self.display_data.main_window_size_x, 3),
            dtype=self.display_data.projector_data_type,
        )
        array_out[self._y_active_1 : self._y_active_2, self._x_active_1 : self._x_active_2, :] = array

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
        array[..., 0] = np.roll(array[..., 0], self.display_data.shift_red_x, 1)
        array[..., 0] = np.roll(array[..., 0], self.display_data.shift_red_y, 0)

        # Shift blue channel
        array[..., 2] = np.roll(array[..., 2], self.display_data.shift_blue_x, 1)
        array[..., 2] = np.roll(array[..., 2], self.display_data.shift_blue_y, 0)

        # Format array into tkinter format
        image = Image.fromarray(array, "RGB")
        return ImageTk.PhotoImage(image)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = "") -> "ImageProjection":
        """Loads display_data from the given file into a new image_projection window. Assumes data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.

        Returns
        -------
        projector: ImageProjection
            The new ImageProjection instance, opened in a new tkinter window.
        """
        # Load data
        display_data = ImageProjectionData.load_from_hdf(file, prefix)

        # Open window
        return cls.in_new_window(display_data)

    def save_to_hdf(self, file: str, prefix: str = "") -> None:
        """Saves image projection display_data parameters to given file. Data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        self.display_data.save_to_hdf(file, prefix)

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
                callback(self)

        with et.ignored(Exception):
            self.root.destroy()

    def _make_aruco_marker(self, width: int, id_: int) -> np.ndarray:
        """Returns NxMx3 aruco marker array"""
        # Define marker image
        img_2d = np.ones((width, width), dtype="uint8") * self.display_data.projector_max_int

        # Create marker image
        aruco.generateImageMarker(self.aruco_dictionary, id_, width, img_2d)
        return np.concatenate([img_2d[:, :, None]] * 3, axis=2)
