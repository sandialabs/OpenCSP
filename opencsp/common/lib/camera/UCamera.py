"""


Model of machine vision camera.

    ***  NOTE: THIS IS CURRENTLY TWO SEPARATE MODELS, DUE TO EVOLVING NEEDS DURING CODE DEVELOPMENT.  ***
    ***        WE NEED TO MERGE THESE, RECONCILING CALLERS.                                           ***
    ***        WE ALSO SHOULD INTEGRATE THIS SEAMLESSLY WITH CAMERA CALIBRATON ANALYSIS.              ***
    ***  NOTE: THIS CODE NEEDS TO BE MERGED WITH THE CAMERA CLASS FROM THE SOFAST CODE, ALSO LOCATED  ***
    ***        IN THIS DIRECTORY.                                                                     ***

"""

import inspect
import math
import numpy as np
import sys

import opencsp.common.lib.file.CsvInterface as csvi


# -------------------------------------------------------------------------------------------------------
# SIMPLIFIED CAMERA MODEL
#
# This was implemented first, and is used by the flight planning code.
# It primarily delivers field of view information.
#


class Camera(csvi.CsvInterface):
    """
    Model of a camera and its optical properties.
    """

    def __init__(
        self,
        name,  # String describing camera and lens.
        sensor_x_mm,  # mm.  Width of sensor image area.
        sensor_y_mm,  # mm.  Height of sensor image area.
        pixels_x,  # For still images.  Video may be different.
        pixels_y,  # For still images.  Video may be different.
        focal_lengths_mm,  # mm.  [min, max] range.  For fixed focal length min == max.
    ):
        super(Camera, self).__init__()

        # Input parameters.
        self.name = name  #     String describing camera and lens.
        self.sensor_x = sensor_x_mm / 1000.0  # m.  Size of sensor in horizontal direction.
        self.sensor_y = sensor_y_mm / 1000.0  # m.  Size of sensor in vertical direction.
        self.pixels_x = pixels_x  #     Number of pizels in horizontal direction for still images. Video may differ.
        self.pixels_y = pixels_y  #     Number of pizels in vertical direction for still images. Video may differ.
        self.focal_length_min = (
            focal_lengths_mm[0] / 1000.0
        )  # m.  Minimum focal length.  For fixed focal length min == max.
        self.focal_length_max = (
            focal_lengths_mm[1] / 1000.0
        )  # m.  Maximum focal length.  For fixed focal length min == max.
        # Dependent parameters.
        self.fov_vertical_min = 2.0 * math.atan(
            (self.sensor_y / 2.0) / self.focal_length_max
        )  # Angular field of view of the camera, in the vertical direction.
        self.fov_vertical_max = 2.0 * math.atan((self.sensor_y / 2.0) / self.focal_length_min)  #
        self.fov_horizontal_min = 2.0 * math.atan(
            (self.sensor_x / 2.0) / self.focal_length_max
        )  # Angular field of view of the camera, in the horizontal direction.
        self.fov_horizontal_max = 2.0 * math.atan((self.sensor_x / 2.0) / self.focal_length_min)  #

    # ACCESS

    def max_focal_length(self):
        """
        Returns the maximum focal length of the camera.

        Returns
        -------
        float
            The maximum focal length in meters.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return self.focal_length_max

    @staticmethod
    def csv_header(delimeter=","):
        """
        Returns the CSV column headings for the camera data.

        Parameters
        ----------
        delimeter : str, optional
            The delimiter to use in the CSV header (default is ',').

        Returns
        -------
        str
            A string containing the CSV column headings.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return delimeter.join(
            [
                "name",
                "sensor_w",
                "sensor_h",
                "pixels_x",
                "pixels_y",
                "focal_length_min",
                "focal_length_max",
                "fov_vert_min",
                "fov_vert_max",
                "fov_hor_min",
                "fov_hor_max",
            ]
        )

    @classmethod
    def from_csv_line(cls, data_row: list[str]):
        """
        Creates a Camera object from a CSV data row.

        Parameters
        ----------
        data_row : list of str
            A list containing the camera data in CSV format.

        Returns
        -------
        tuple
            A tuple containing the Camera object and any remaining data.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        ret = cls("", 1, 1, 1, 1, 1)
        ret.name: str = data_row[0].replace("commmmma", ",")
        ret.sensor_x: float = np.float64(data_row[1])
        ret.sensor_y: float = np.float64(data_row[2])
        ret.pixels_x: int = int(data_row[3])
        ret.pixels_y: int = int(data_row[4])
        ret.focal_length_min: float = np.float64(data_row[5])
        ret.focal_length_max: float = np.float64(data_row[6])
        ret.fov_vertical_min: float = np.float64(data_row[7])
        ret.fov_vertical_max: float = np.float64(data_row[8])
        ret.fov_horizontal_min: float = np.float64(data_row[9])
        ret.fov_horizontal_max: float = np.float64(data_row[10])
        return ret, data_row[11:]

    def to_csv_line(self, delimeter=","):
        """
        Converts the Camera object to a CSV line.

        Parameters
        ----------
        delimeter : str, optional
            The delimiter to use in the CSV line (default is ',').

        Returns
        -------
        str
            A string representing the Camera object in CSV format.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return delimeter.join(
            [
                str(v)
                for v in [
                    self.name.replace(",", "commmmma"),
                    self.sensor_x,
                    self.sensor_y,
                    self.pixels_x,
                    self.pixels_y,
                    self.focal_length_min,
                    self.focal_length_max,
                    self.fov_vertical_min,
                    self.fov_vertical_max,
                    self.fov_horizontal_min,
                    self.fov_horizontal_max,
                ]
            ]
        )


# CAMERAS WE USE


def mavic_zoom():
    """
    Creates a Camera object for the Mavic Zoom.

    Returns
    -------
    Camera
        A Camera object configured for the Mavic Zoom.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return Camera(
        name='Mavic Zoom',
        sensor_x_mm=6.17,  # mm.
        sensor_y_mm=4.54,  # mm.  sqrt(7.66^2 - 6.17^2)
        pixels_x=4000,  # For still images.
        pixels_y=3000,  # For still images.
        focal_lengths_mm=[4.33, 8.60],  # [min, max] range, in mm.  For fixed focal length min == max.
    )


def sony_alpha_20mm_landscape():
    """
    Creates a Camera object for the Sony Alpha 20mm (landscape orientation).

    Returns
    -------
    Camera
        A Camera object configured for the Sony Alpha 20mm (landscape).
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return Camera(
        name='Sony Alpha, 20mm',
        sensor_x_mm=35.9,  # mm.
        sensor_y_mm=24.0,  # mm.
        pixels_x=8760,  # For still images.
        pixels_y=4864,  # For still images.
        focal_lengths_mm=[20, 20],  # [min, max] range, in mm.  For fixed focal length min == max.
    )


def sony_alpha_20mm_portrait():
    """
    Creates a Camera object for the Sony Alpha 20mm (portrait orientation).

    Returns
    -------
    Camera
        A Camera object configured for the Sony Alpha 20mm (portrait).
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return Camera(
        name='Sony Alpha, 20mm',
        sensor_x_mm=24.0,  # mm.
        sensor_y_mm=35.9,  # mm.
        pixels_x=4864,  # For still images.
        pixels_y=8760,  # For still images.
        focal_lengths_mm=[20, 20],  # [min, max] range, in mm.  For fixed focal length min == max.
    )


def ultra_wide_angle():
    """
    Creates a Camera object for an ultra wide angle camera.

    Returns
    -------
    Camera
        A Camera object configured for an ultra wide angle lens.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return Camera(
        name='Ultra Wide Angle',
        sensor_x_mm=24.0,  # mm.
        sensor_y_mm=35.9,  # mm.
        pixels_x=4864,  # For still images.
        pixels_y=8760,  # For still images.
        focal_lengths_mm=[5, 5],  # [min, max] range, in mm.  For fixed focal length min == max.
    )


# -------------------------------------------------------------------------------------------------------
# CAMERA MODEL INCLUDING DISTORTION
#
# This was implemented second, and includes the representation employed by OpenCV.
#


class RealCamera(csvi.CsvInterface):
    """
    Model of a camera and its intrinsic parameters, including distortion.
    """

    def __init__(
        self,
        # These values are taken from the utils.py file as it existed in the original repository on July 2, 2021.
        # I believe they correspond to the Mavic Zoom.
        name='Mavic Zoom',  # String describing camera and lens.
        # Image size.
        n_x=3840,  # Pixels.
        n_y=2160,  # Pixels.
        # Focal length.
        f_x=2868.1,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        f_y=2875.9,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Optical center.
        c_x=1920.0,  # Pixels.  Before camera calibration, assume c_x = w/2 = 3840/2
        c_y=1080.0,  # Pixels.  Before camera calibration, assume c_y = h/2 = 2160/2
        # Radial distortion.
        k_1=-0.024778,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        k_2=0.012383,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Tangential distortion.
        p_1=-0.00032978,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        p_2=-0.0001401,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
    ):
        """
        Initializes the RealCamera object with specified parameters.

        Parameters
        ----------
        name : str, optional
            String describing the camera and lens (default is 'Mavic Zoom').
        n_x : int, optional
            Number of pixels in the horizontal direction (default is 3840).
        n_y : int, optional
            Number of pixels in the vertical direction (default is 2160).
        f_x : float, optional
            Focal length in pixels in the x-direction (default is 2868.1).
        f_y : float, optional
            Focal length in pixels in the y-direction (default is 2875.9).
        c_x : float, optional
            Optical center x-coordinate in pixels (default is 1920.0).
        c_y : float, optional
            Optical center y-coordinate in pixels (default is 1080.0).
        k_1 : float, optional
            Radial distortion coefficient (default is -0.024778).
        k_2 : float, optional
            Radial distortion coefficient (default is 0.012383).
        p_1 : float, optional
            Tangential distortion coefficient (default is -0.00032978).
        p_2 : float, optional
            Tangential distortion coefficient (default is -0.0001401).
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        super(RealCamera, self).__init__()

        # Input parameters.
        self.name = name
        self.n_x = n_x
        self.n_y = n_y
        self._f_x = f_x  # Do not access directly.  Fetch via camera_matrix.
        self._f_y = f_y  # Do not access directly.  Fetch via camera_matrix.
        self._c_x = c_x  # Do not access directly.  Fetch via camera_matrix.
        self._c_y = c_y  # Do not access directly.  Fetch via camera_matrix.
        self._k_1 = k_1  # Do not access directly.  Fetch via distortion_coeffs.
        self._k_2 = k_2  # Do not access directly.  Fetch via distortion_coeffs.
        self._p_1 = p_1  # Do not access directly.  Fetch via distortion_coeffs.
        self._p_2 = p_2  # Do not access directly.  Fetch via distortion_coeffs.
        # Dependent parameters.
        self.frame_box_pq = self.construct_frame_box_pq()
        self.camera_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]).reshape(3, 3)
        self.distortion_coeffs = np.array([[k_1, k_2, p_1, p_2]])

    # CONSTRUCTION

    @staticmethod
    def csv_header(delimeter=","):
        """
        Returns the CSV column headings for the RealCamera data.

        Parameters
        ----------
        delimeter : str, optional
            The delimiter to use in the CSV header (default is ',').

        Returns
        -------
        str
            A string containing the CSV column headings.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return delimeter.join(
            [
                "name",
                "pix_x",
                "pix_y",
                "focal_length_x",
                "focal_length_y",
                "center_pix_x",
                "center_pix_y",
                "radial_1",
                "radial_2",
                "tangential_1",
                "tangential_2",
            ]
        )

    @classmethod
    def from_csv_line(cls, data_row: list[str]):
        """
        Creates a RealCamera object from a CSV data row.

        Parameters
        ----------
        data_row : list of str
            A list containing the camera data in CSV format.

        Returns
        -------
        tuple
            A tuple containing the RealCamera object and any remaining data.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return (
            cls(
                name=data_row[0].replace("commmmma", ","),
                n_x=np.float64(data_row[1]),
                n_y=np.float64(data_row[2]),
                f_x=np.float64(data_row[3]),
                f_y=np.float64(data_row[4]),
                c_x=np.float64(data_row[5]),
                c_y=np.float64(data_row[6]),
                k_1=np.float64(data_row[7]),
                k_2=np.float64(data_row[8]),
                p_1=np.float64(data_row[9]),
                p_2=np.float64(data_row[10]),
            ),
            data_row[11:],
        )

    def to_csv_line(self, delimeter=","):
        """
        Converts the RealCamera object to a CSV line.

        Parameters
        ----------
        delimeter : str, optional
            The delimiter to use in the CSV line (default is ',').

        Returns
        -------
        str
            A string representing the RealCamera object in CSV format.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return delimeter.join(
            [
                str(v)
                for v in [
                    self.name.replace(",", "commmmma"),
                    self.n_x,
                    self.n_y,
                    self._f_x,
                    self._f_y,
                    self._c_x,
                    self._c_y,
                    self._k_1,
                    self._k_2,
                    self._p_1,
                    self._p_2,
                ]
            ]
        )

    def construct_frame_box_pq(self):
        """
        Constructs the frame box coordinates for the camera image.

        Returns
        -------
        list of list of float
            A list containing the minimum and maximum coordinates of the frame box.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        p_min = 0
        p_max = self.n_x
        q_min = -self.n_y  # Negate y because image is flipped.
        q_max = 0
        pq_min = [p_min, q_min]
        pq_max = [p_max, q_max]
        return [pq_min, pq_max]

    # ACCESS

    def max_focal_length(self):
        """
        Returns the maximum focal length of the camera.

        Returns
        -------
        float
            The maximum focal length in pixels.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return max(self._f_x, self._f_y)

    def image_frame_corners(self):
        """
        Returns the coordinates of the corners of the image frame.

        Returns
        -------
        list of list of float
            A list containing the coordinates of the four corners of the image frame.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return [[0, 0], [self.n_x, 0], [self.n_x, self.n_y], [0, self.n_y]]


# CAMERAS WE USE


def ideal_camera_wide_angle():
    """
    Creates a RealCamera object for an ideal wide angle camera.

    Returns
    -------
    RealCamera
        A RealCamera object configured for an ideal wide angle camera.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return RealCamera(
        name='Ideal Camera, Wide Angle',
        # Image size.
        n_x=4,  # Pixels.
        n_y=3,  # Pixels.
        # Focal length.
        f_x=1,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        f_y=1,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Optical center.
        c_x=2.0,  # Pixels.  Before camera calibration, assume c_x = w/2 = 4/2
        c_y=1.5,  # Pixels.  Before camera calibration, assume c_y = h/2 = 2/2
        # Radial distortion.
        k_1=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        k_2=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Tangential distortion.
        p_1=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        p_2=0.0,
    )  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?


def ideal_camera_normal():
    """
    Creates a RealCamera object for an ideal normal camera.

    Returns
    -------
    RealCamera
        A RealCamera object configured for an ideal normal camera.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # For a 35 mm film format, a 50mm lens is considered to provide a "normal" view.
    # The 35 mm film format is (36 mm x 24 mm), which amounts to a diagonal of 43.3 mm.
    # Thus a "normal" view results from a focal length that is 50mm/43.3mm = 1.156 times the image diagonal.
    #
    # For a 4x3 image sensor, the diagonal distance is 5.
    # Thus a "normal" focal length for our 4x3 ideal sensor is 5 * 1.156 = 5.78, or approximately 6.0.
    #
    return RealCamera(
        name='Ideal Camera, Normal',
        # Image size.
        n_x=4,  # Pixels.
        n_y=3,  # Pixels.
        # Focal length.
        f_x=6,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        f_y=6,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Optical center.
        c_x=2.0,  # Pixels.  Before camera calibration, assume c_x = w/2 = 4/2
        c_y=1.5,  # Pixels.  Before camera calibration, assume c_y = h/2 = 2/2
        # Radial distortion.
        k_1=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        k_2=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Tangential distortion.
        p_1=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        p_2=0.0,
    )  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?


def ideal_camera_telephoto():
    """
    Creates a RealCamera object for an ideal telephoto camera.

    Returns
    -------
    RealCamera
        A RealCamera object configured for an ideal telephoto camera.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return RealCamera(
        name='Ideal Camera, Telephoto',
        # Image size.
        n_x=4,  # Pixels.
        n_y=3,  # Pixels.
        # Focal length.
        f_x=12,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        f_y=12,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Optical center.
        c_x=2.0,  # Pixels.  Before camera calibration, assume c_x = w/2 = 4/2
        c_y=1.5,  # Pixels.  Before camera calibration, assume c_y = h/2 = 2/2
        # Radial distortion.
        k_1=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        k_2=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Tangential distortion.
        p_1=0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        p_2=0.0,
    )  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?


def real_mavic_zoom():
    """
    Creates a RealCamera object for the Real Mavic Zoom.

    Returns
    -------
    RealCamera
        A RealCamera object configured for the Real Mavic Zoom with specified calibration values.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Values from calibration.
    return RealCamera(
        name='Real Mavic Zoom',
        # Image size.
        n_x=3840,  # Pixels.
        n_y=2160,  # Pixels.
        # Focal length.
        f_x=2868.1,  # Pixels.
        f_y=2875.9,  # Pixels.
        # Optical center.
        c_x=1920.0,  # Pixels.  Before camera calibration, assume c_x = w/2 = 3840/2
        c_y=1080.0,  # Pixels.  Before camera calibration, assume c_y = h/2 = 2160/2
        # Radial distortion.
        k_1=-0.024778,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        k_2=0.012383,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Tangential distortion.
        p_1=-0.00032978,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        p_2=-0.0001401,
    )  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?


def real_sony_alpha_20mm_still():
    """
    Creates a RealCamera object for the Real Sony Alpha 20mm (still images).

    Returns
    -------
    RealCamera
        A RealCamera object configured for the Real Sony Alpha 20mm (still) with specified calibration values.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Values derived from measurements in <Experiments_dir>\2022-12-21_SonyCalibration\1_Data\2023-03-22
    # Matlab computed intrisic matrix:
    #    [[4675.73, 0,        4343.252]
    #     [0,       4672.100, 2883.190]
    #     [0,       0,        1       ]]
    # Matlab distortion coefficients:
    #     [-0.0568132871107041, 0.0395572945993200, 0, 0]
    return RealCamera(
        name='Real Sony Alpha, 20mm (still)',
        # Image size.
        n_x=8640,  # Pixels.
        n_y=5760,  # Pixels.
        # Focal length.
        f_x=4675.73,  # Pixels.
        f_y=4672.100,  # Pixels.
        # Optical center.
        c_x=4343.252,  # Pixels.  Before camera calibration, assume c_x = w/2 = 3840/2
        c_y=2883.190,  # Pixels.  Before camera calibration, assume c_y = h/2 = 2160/2
        # Radial distortion.
        k_1=-0.0568132871107041,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        k_2=0.0395572945993200,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Tangential distortion.
        p_1=-0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        p_2=-0.0,
    )  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?

    # # Values derived from published specifications.
    # # See https://www.sony.com/ug/electronics/interchangeable-lens-cameras/ilce-1/specifications
    # # Assumes zero distortion.
    # return RealCamera(name='Real Sony Alpha, 20mm (still)',
    #                    # Image size.
    #                    n_x=8640,           # Pixels.
    #                    n_y=5760,           # Pixels.
    #                    # Focal length.
    #                    f_x=4801.9,         # Pixels.
    #                    f_y=4801.9,         # Pixels.
    #                    # Optical center.
    #                    c_x=4320.0,         # Pixels.  Before camera calibration, assume c_x = w/2 = 3840/2
    #                    c_y=2880.0,         # Pixels.  Before camera calibration, assume c_y = h/2 = 2160/2
    #                    # Radial distortion.
    #                    k_1=-0.0,      # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
    #                    k_2= 0.0,      # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
    #                    # Tangential distortion.
    #                    p_1=-0.0,    # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
    #                    p_2=-0.0)     # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?


def real_sony_alpha_20mm_video():
    """
    Creates a RealCamera object for the Real Sony Alpha 20mm (video).

    Returns
    -------
    RealCamera
        A RealCamera object configured for the Real Sony Alpha 20mm (video) with extrapolated calibration values.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    # Values extrapolated to smaller image size from measurements in real_sony_alpha_20mm_still()
    return RealCamera(
        name='Real Sony Alpha, 20mm (still)',
        # Image size.
        n_x=7680,  # 8640,           # Pixels.
        n_y=4320,  # 5760,           # Pixels.
        # Focal length.
        f_x=4675.73,  # Pixels.
        f_y=4672.100,  # Pixels.
        # Optical center.
        c_x=4343.252 / 8640 * 7680,  # Pixels.  Before camera calibration, assume c_x = w/2 = 3840/2
        c_y=2883.190 / 5760 * 4320,  # Pixels.  Before camera calibration, assume c_y = h/2 = 2160/2
        # Radial distortion.
        k_1=-0.0568132871107041,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        k_2=0.0395572945993200,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        # Tangential distortion.
        p_1=-0.0,  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
        p_2=-0.0,
    )  # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?

    # # Values derived from published specifications.
    # # See https://www.sony.com/ug/electronics/interchangeable-lens-cameras/ilce-1/specifications
    # # Assumes zero distortion.
    # return RealCamera(name='Real Sony Alpha, 20mm (video)',
    #                    # Image size.
    #                    n_x=7680,           # Pixels.
    #                    n_y=4320,           # Pixels.
    #                    # Focal length.
    #                    f_x=4801.9,         # Pixels.
    #                    f_y=4801.9,         # Pixels.
    #                    # # Focal length.
    #                    # f_x=4273.5,         # Pixels.
    #                    # f_y=4273.5,         # Pixels.
    #                    # Optical center.
    #                    c_x=3840.0,         # Pixels.  Before camera calibration, assume c_x = w/2 = 3840/2
    #                    c_y=2160.0,         # Pixels.  Before camera calibration, assume c_y = h/2 = 2160/2
    #                    # Radial distortion.
    #                    k_1=-0.0,      # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
    #                    k_2= 0.0,      # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
    #                    # Tangential distortion.
    #                    p_1=-0.0,    # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
    #                    p_2=-0.0)     # ?? SCAFFOLDING RCB -- WHAT ARE UNITS?  ARE VALUES VALID FOR EXPECTED UNITS?
