import inspect
from itertools import compress
import math
from typing import Callable
from warnings import warn

import matplotlib.tri as mtri
import numpy as np
import sympy.vector as vec
from matplotlib.tri import Triangulation
from scipy.spatial.transform import Rotation
from sympy import Symbol, diff

import opencsp.common.lib.render.Color as cl  # ?? SCAFFOLDING RCB - FIX FILENAME TO CAPITALIZED
import opencsp.common.lib.target.TargetColor as tc
import opencsp.common.lib.target.target_color_2d_rgb as tc2r
import opencsp.common.lib.target.target_color_convert as tcc
import opencsp.common.lib.target.target_color_1d_gradient as tc1g
from opencsp.common.lib.target.TargetAbstract import TargetAbstract
import opencsp.common.lib.tool.time_date_tools as tdt


class TargetColor(TargetAbstract):
    """
    Target implementation for managing color patterns on a canvas.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    # CONSTRUCTION
    def __init__(
        self,
        image_width: float,  # Meters
        image_height: float,  # Meters
        dpm: float,  # dots per meter
        initial_color: cl.Color,  # Color to fill canvas before adding patterns.
    ) -> None:
        """
        Initializes the TargetColor instance with specified dimensions and initial color.

        Parameters
        ----------
        image_width : float
            The width of the image in meters.
        image_height : float
            The height of the image in meters.
        dpm : float
            Dots per meter for the image resolution.
        initial_color : cl.Color
            The color to fill the canvas before adding patterns.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super().__init__(image_width, image_height, dpm)  # initalizes the attributes universal to all mirrors
        # Set initial pattern.
        self.initial_color = initial_color
        self.pattern_description = initial_color.name
        initial_rgb = (
            self.initial_color.rgb_255()
        )  # ?? SCAFFOLDING RCB -- I GOT TRIPPED UP BY CONFUSION RE: IMAGES IN [0,1.0] AND IMAGES IN [0,255].  HOW BEST RESOLVE/PREVENT?
        n_rows, n_cols = self.rows_cols()
        for row in range(0, n_rows):
            for col in range(0, n_cols):
                # Set pixel color
                self.image[row, col, 0] = initial_rgb[0]
                self.image[row, col, 1] = initial_rgb[1]
                self.image[row, col, 2] = initial_rgb[2]
        self.pattern_description = initial_color.name

    # ACCESS

    # See TargetAbstract for generic accessor functions.

    def rows_cols(
        self,
    ):  # ?? SCAFFOLDING RCB -- MODIFY TO USE BASE CLASS ROWS_COLS_BANDS() ACCESSOR, AND THEN APPLY COLOR-SPECIFIC TEST.
        """
        Returns the number of rows and columns in the image.

        Returns
        -------
        tuple
            A tuple containing the number of rows and columns in the image.

        Raises
        ------
        AssertionError
            If the number of image bands is not equal to 3.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows = self.image.shape[0]
        n_cols = self.image.shape[1]
        n_bands = self.image.shape[2]
        if n_bands != 3:
            print("ERROR: In TargetAbstract.row_cols(), number of input image bands is not 3.")
            assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
        return n_rows, n_cols

    # MODIFICATION

    def set_pattern_description(self, description: str) -> None:
        """
        Sets the pattern description for the target color.

        Parameters
        ----------
        description : str
            The description of the color pattern.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.pattern_description = description

    # Linear color bar, x direction
    def set_image_to_linear_color_bar_x(
        self, color_below_min: cl, color_bar, color_above_max: cl, discrete_or_continuous: str
    ) -> None:
        """
        Sets the image to a linear color bar in the x direction.

        Parameters
        ----------
        color_below_min : cl
            The color to use for values below the minimum.
        color_bar :
            The color bar to use for mapping values.
        color_above_max : cl
            The color to use for values above the maximum.
        discrete_or_continuous : str
            Specifies whether the color bar is discrete or continuous.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        for row in range(0, n_rows):
            for col in range(0, n_cols):
                # Lookup color bar entry.
                val = col
                val_min = 0
                val_max = n_cols
                color = tcc.color_given_value(
                    val, val_min, val_max, color_below_min, color_bar, color_above_max, discrete_or_continuous
                )
                # Set pixel color
                # ?? SCAFFOLDING RCB -- FIXUP ALL THIS CONFUSION REGARDING WHETHER COLORS ARE OVER [0,1] OR [0,255].
                self.image[row, col, 0] = color[
                    0
                ]  # /255.0  # ?? SCAFFOLDING RCB -- CONVERT COLOR BAR TO INTERVAL [0,1]
                self.image[row, col, 1] = color[1]  # /255.0
                self.image[row, col, 2] = color[2]  # /255.0

    # Linear color bar, y direction
    def set_image_to_linear_color_bar_y(
        self,
        color_below_min: cl,
        color_bar,
        color_above_max: cl,
        discrete_or_continuous: str,
        # Defines scheme for modulating saturation from left to right.  Values: None, 'saturated_to_white' or 'light_to_saturated'
        lateral_gradient_type: str = "saturated_to_white",
        # Dimensionless.  Applies if lateral_gradient_type == 'saturated_to_white'.
        saturated_to_white_exponent: float = 1.5,
        # Dimensionless.  Applies if lateral_gradient_type == 'light_to_saturated'.
        light_to_saturated_min: float = 0.2,
        # Dimensionless.  Applies if lateral_gradient_type == 'light_to_saturated'.
        light_to_saturated_max: float = 1.0,
    ) -> None:
        """
        Sets the image to a linear color bar in the y direction.

        Parameters
        ----------
        color_below_min : cl
            The color to use for values below the minimum.
        color_bar :
            The color bar to use for mapping values.
        color_above_max : cl
            The color to use for values above the maximum.
        discrete_or_continuous : str
            Specifies whether the color bar is discrete or continuous.
        lateral_gradient_type : str, optional
            Defines the scheme for modulating saturation from left to right.
            Values: None, 'saturated_to_white', or 'light_to_saturated'.
        saturated_to_white_exponent : float, optional
            Dimensionless exponent applied if lateral_gradient_type is 'saturated_to_white'.
        light_to_saturated_min : float, optional
            Dimensionless minimum value applied if lateral_gradient_type is 'light_to_saturated'.
        light_to_saturated_max : float, optional
            Dimensionless maximum value applied if lateral_gradient_type is 'light_to_saturated'.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        for row in range(0, n_rows):
            for col in range(0, n_cols):
                # Lookup color bar entry.
                val = row
                val_min = 0
                val_max = n_rows  # Last row in color bar is the final color; there is not a color beyond.
                color = tcc.color_given_value(
                    val, val_min, val_max, color_below_min, color_bar, color_above_max, discrete_or_continuous
                )  # ?? SCAFFOLDING -- USE "SPLIT" CONTROL PARAMETER.

                # Adjust saturation.
                lateral_fraction = col / n_cols
                # Color components.
                this_red = color[0]  # /255.0  # ?? SCAFFOLDING RCB -- CONVERT COLOR BAR TO INTERVAL [0,1]
                this_green = color[1]  # /255.0
                this_blue = color[2]  # /255.0

                if lateral_gradient_type == None:
                    pass

                elif lateral_gradient_type == "saturated_to_white":
                    # Transition from saturated at left to white.
                    saturation_factor = 1.0 - pow(lateral_fraction, saturated_to_white_exponent)
                    if saturation_factor < 0.0:
                        saturation_factor = 0.0
                    if saturation_factor > 1.0:
                        saturation_factor = 0.0
                    this_red_from_white = 255 - this_red
                    this_green_from_white = 255 - this_green
                    this_blue_from_white = 255 - this_blue
                    this_red_from_white *= saturation_factor
                    this_green_from_white *= saturation_factor
                    this_blue_from_white *= saturation_factor
                    this_red = 255 - this_red_from_white
                    this_green = 255 - this_green_from_white
                    this_blue = 255 - this_blue_from_white

                elif lateral_gradient_type == "light_to_saturated":
                    # Transition from partially saturated at left to fully saturated at boundary.
                    saturation_range = light_to_saturated_max - light_to_saturated_min
                    saturation_factor = light_to_saturated_min + (lateral_fraction * saturation_range)
                    if saturation_factor < 0.0:
                        saturation_factor = 0.0
                    if saturation_factor > 1.0:
                        saturation_factor = 0.0
                    if lateral_fraction <= 1.0:
                        ref_red = 255
                        ref_green = ref_red
                        ref_blue = ref_red
                        this_red_from_white = ref_red - this_red
                        this_green_from_white = ref_green - this_green
                        this_blue_from_white = ref_blue - this_blue
                        this_red_from_white *= saturation_factor
                        this_green_from_white *= saturation_factor
                        this_blue_from_white *= saturation_factor
                        this_red = ref_red - this_red_from_white
                        this_green = ref_green - this_green_from_white
                        this_blue = ref_blue - this_blue_from_white
                    else:
                        this_red = 255
                        this_green = 255
                        this_blue = 255
                else:
                    print(
                        "ERROR: In TargetColor.set_image_to_linear_color_bar_y(), encountered unexpected lateral_gradient_type = "
                        + str(lateral_gradient_type)
                    )
                    assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION

                # Set pixel color
                self.image[row, col, 0] = this_red
                self.image[row, col, 1] = this_green
                self.image[row, col, 2] = this_blue

    # Polar color bar
    def set_image_to_polar_color_bar(
        self,
        # Control color by angle.
        color_below_min: cl = cl.black(),  # Color to use for values below min limit of color bar.
        # Color bar for angular values in [0,2pi).  # ?? SCAFFOLDING RCB -- TYPE TIP NEEDED.
        color_bar=tcc.O_color_bar(),
        color_above_max: cl = cl.white(),  # Color to use for values of max limit of color bar.
        # 'discrete' or 'continuous'  # ?? SCAFFOLDING RCB -- RENAME TO "color_interpolation_type" ?
        discrete_or_continuous: str = "continuous",
        # Modulating saturation with radius.
        # Defines range for saturation modulation as a function of radius.  Values: 'circle' or 'image_boundary'
        pattern_boundary: str = "image_boundary",
        # Defines scheme for modulating saturation as a function of radius.  Values: 'saturated_center_to_white' or 'light_center_to_saturated'
        radial_gradient_type: str = "saturated_center_to_white",
        # Dimensionless.  Applies if radial_gradient_type == 'saturated_center_to_white'.
        saturated_center_to_white_exponent: float = 1.5,
        # Dimensionless.  Applies if radial_gradient_type == 'light_center_to_saturated'.
        light_center_to_saturated_saturation_min: float = 0.2,
        # Dimensionless.  Applies if radial_gradient_type == 'light_center_to_saturated'.
        light_center_to_saturated_saturation_max: float = 1.0,
        # Fiducial marks.
        draw_center_fiducial=True,  # Boolean.   Whether to draw a mark at the target center.
        center_fiducial_width_pix=3,  # Pixels.    Width of center fiducial; should be odd number.
        center_fiducial_color: cl = cl.black(),  # Color.     Color of center fiducial.
        draw_edge_fiducials=True,  # Boolean.   Whether to draw fiducial tick marks along target edges.
        n_ticks_x=7,  # No units.  Number of tick marks to draw along top/bottom horizontal target edges.
        n_ticks_y=7,  # No units.  Number of tick marks to draw along left/right vertical target edges.
        tick_length=0.025,  # Meters.    Length to draw edge tick marks.
        tick_width_pix=3,  # Pixels.    Width to draw edge tick marks; should be odd number.
        tick_color: cl = cl.black(),  # Color.     Color of edge tick marks.
    ) -> None:
        """
        Sets the image to a polar color bar based on angular values.

        This method generates a polar color bar where the color is controlled by the angle,
        and the saturation is modulated by the radius from the center of the image.

        Parameters
        ----------
        color_below_min : cl, optional
            The color to use for values below the minimum limit of the color bar. Default is black.
        color_bar :
            The color bar for angular values in the range [0, 2π). Default is a predefined color bar.
        color_above_max : cl, optional
            The color to use for values above the maximum limit of the color bar. Default is white.
        discrete_or_continuous : str, optional
            Specifies whether the color interpolation is 'discrete' or 'continuous'. Default is 'continuous'.
        pattern_boundary : str, optional
            Defines the range for saturation modulation as a function of radius.
            Values can be 'circle' or 'image_boundary'. Default is 'image_boundary'.
        radial_gradient_type : str, optional
            Defines the scheme for modulating saturation as a function of radius.
            Values can be 'saturated_center_to_white' or 'light_center_to_saturated'. Default is 'saturated_center_to_white'.
        saturated_center_to_white_exponent : float, optional
            Dimensionless exponent applied if radial_gradient_type is 'saturated_center_to_white'. Default is 1.5.
        light_center_to_saturated_saturation_min : float, optional
            Dimensionless minimum saturation value applied if radial_gradient_type is 'light_center_to_saturated'. Default is 0.2.
        light_center_to_saturated_saturation_max : float, optional
            Dimensionless maximum saturation value applied if radial_gradient_type is 'light_center_to_saturated'. Default is 1.0.
        draw_center_fiducial : bool, optional
            Whether to draw a mark at the target center. Default is True.
        center_fiducial_width_pix : int, optional
            Width of the center fiducial in pixels; should be an odd number. Default is 3.
        center_fiducial_color : cl, optional
            Color of the center fiducial. Default is black.
        draw_edge_fiducials : bool, optional
            Whether to draw fiducial tick marks along the target edges. Default is True.
        n_ticks_x : int, optional
            Number of tick marks to draw along the top and bottom horizontal target edges. Default is 7.
        n_ticks_y : int, optional
            Number of tick marks to draw along the left and right vertical target edges. Default is 7.
        tick_length : float, optional
            Length to draw edge tick marks in meters. Default is 0.025.
        tick_width_pix : int, optional
            Width to draw edge tick marks in pixels; should be an odd number. Default is 3.
        tick_color : cl, optional
            Color of edge tick marks. Default is black.

        Raises
        ------
        AssertionError
            If an unexpected value is encountered for pattern_boundary or radial_gradient_type.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        center_row = n_rows / 2.0
        center_col = n_cols / 2.0
        width = n_cols
        height = n_rows
        half_width = width / 2.0
        half_height = height / 2.0
        diameter = min(width, height)
        radius = diameter / 2.0
        for row in range(0, n_rows):
            # # Progress report when generating large images.
            # if (row % 10) == 0:
            #     print('In set_image_to_polar_color_bar(), time = ', tdt.current_time_string(), '  row = ', str(row))
            for col in range(0, n_cols):
                # Lookup color bar entry.
                delta_x = col - center_col
                delta_y = -(row - center_row)  # Row 0 is at the top of the image.
                this_angle = math.atan2(delta_y, delta_x)
                this_radius = math.sqrt((delta_x * delta_x) + (delta_y * delta_y))
                # Lookup color given angle.# (Saturation not adjusted yet.)
                color = tcc.color_given_value(
                    this_angle, -math.pi, math.pi, color_below_min, color_bar, color_above_max, discrete_or_continuous
                )
                # Compute saturation adjustment.
                # Determine the radius to use for scaling the saturation.
                if pattern_boundary == "circle":
                    # Circle
                    radius_for_this_angle = radius
                elif pattern_boundary == "image_boundary":
                    # Rectangle
                    if math.sin(this_angle) == 0:
                        radius_for_this_angle = abs(half_width / math.cos(this_angle))
                    elif math.cos(this_angle) == 0:
                        radius_for_this_angle = abs(half_height / math.sin(this_angle))
                    else:
                        radius_for_this_angle = min(
                            abs(half_width / math.cos(this_angle)), abs(half_height / math.sin(this_angle))
                        )
                else:
                    print(
                        'ERROR:  In TargetColor.set_image_to_polar_color_bar(), unexpected pattern_boundary = "'
                        + str(pattern_boundary)
                    )
                    assert False  # ?? SCAFFOLDING RCB -- CHANGE THIS TO EXCEPTION.
                # Add a margin to avoid border points that are above-max color due to numerical roundoff error.
                radius_tolerance = 2  # Units are pixels
                radius_for_this_angle += radius_tolerance
                # Compute radius fraction, the percentage distance of this pixel from the image center to the image boundary.
                radius_fraction = this_radius / radius_for_this_angle

                # Adjust saturation.
                # Color components.
                this_red = color[0]  # /255.0  # ?? SCAFFOLDING RCB -- CONVERT COLOR BAR TO INTERVAL [0,1]
                this_green = color[1]  # /255.0
                this_blue = color[2]  # /255.0

                if radial_gradient_type == "saturated_center_to_white":
                    # Transition from saturated at center to white.
                    saturation_factor = 1.0 - pow(radius_fraction, saturated_center_to_white_exponent)
                    if saturation_factor < 0.0:
                        saturation_factor = 0.0
                    if saturation_factor > 1.0:
                        saturation_factor = 0.0
                    this_red_from_white = 255 - this_red
                    this_green_from_white = 255 - this_green
                    this_blue_from_white = 255 - this_blue
                    this_red_from_white *= saturation_factor
                    this_green_from_white *= saturation_factor
                    this_blue_from_white *= saturation_factor
                    this_red = 255 - this_red_from_white
                    this_green = 255 - this_green_from_white
                    this_blue = 255 - this_blue_from_white

                elif radial_gradient_type == "light_center_to_saturated":
                    # Transition from partially saturated at center to fully saturated at boundary.
                    saturation_range = (
                        light_center_to_saturated_saturation_max - light_center_to_saturated_saturation_min
                    )
                    saturation_factor = light_center_to_saturated_saturation_min + (radius_fraction * saturation_range)
                    if saturation_factor < 0.0:
                        saturation_factor = 0.0
                    if saturation_factor > 1.0:
                        saturation_factor = 0.0
                    if radius_fraction <= 1.0:
                        ref_red = 255
                        ref_green = ref_red
                        ref_blue = ref_red
                        this_red_from_white = ref_red - this_red
                        this_green_from_white = ref_green - this_green
                        this_blue_from_white = ref_blue - this_blue
                        this_red_from_white *= saturation_factor
                        this_green_from_white *= saturation_factor
                        this_blue_from_white *= saturation_factor
                        this_red = ref_red - this_red_from_white
                        this_green = ref_green - this_green_from_white
                        this_blue = ref_blue - this_blue_from_white
                    else:
                        this_red = 255
                        this_green = 255
                        this_blue = 255
                else:
                    print(
                        "ERROR: In TargetColor.set_image_to_polar_color_bar(), encountered unexpected radial_gradient_type = "
                        + str(radial_gradient_type)
                    )
                    assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION

                # Set pixel color
                # ?? SCAFFOLDING RCB -- FIXUP ALL THIS CONFUSION REGARDING WHETHER COLORS ARE OVER [0,1] OR [0,255].
                self.image[row, col, 0] = (
                    this_red  # /255.0  # ?? SCAFFOLDING RCB -- CONVERT COLOR BAR TO INTERVAL [0,1]?
                )
                self.image[row, col, 1] = this_green  # /255.0
                self.image[row, col, 2] = this_blue  # /255.0

        # Add fiducial marks.
        if draw_center_fiducial:
            self.set_center_fiducial(center_fiducial_width_pix, center_fiducial_color)
        if draw_edge_fiducials:
            self.set_ticks_along_top_and_bottom_edges(n_ticks_x, tick_length, tick_width_pix, tick_color)
            self.set_ticks_along_left_and_right_edges(n_ticks_y, tick_length, tick_width_pix, tick_color)

    # Fiducial tick marks.
    def set_center_fiducial(self, center_fiducial_width_pix, center_fiducial_color):
        """
        Draws a center fiducial mark on the image.

        Parameters
        ----------
        center_fiducial_width_pix : int
            The width of the center fiducial in pixels; should be an odd number.
        center_fiducial_color : cl
            The color of the center fiducial.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        width = n_cols
        height = n_rows
        center_x = width / 2.0
        center_y = height / 2.0
        center_row = round(center_x)
        center_col = round(center_y)
        # Number of pixels in center_fiducial either side of center point.  Use int() to intentionally truncate.
        center_fiducial_half_margin_pix = int(center_fiducial_width_pix / 2.0)
        if center_fiducial_half_margin_pix == 0:
            if ((center_row >= 0) and (center_row < n_rows)) and ((center_col >= 0) and (center_col < n_cols)):
                self.set_fiducial_pixel(center_row, center_col, center_fiducial_color)
        elif center_fiducial_half_margin_pix > 0:
            for this_row in range(
                center_row - center_fiducial_half_margin_pix, (center_row + center_fiducial_half_margin_pix) + 1
            ):
                for this_col in range(
                    center_col - center_fiducial_half_margin_pix, (center_col + center_fiducial_half_margin_pix) + 1
                ):
                    if ((this_row >= 0) and (this_row < n_rows)) and ((this_col >= 0) and (this_col < n_cols)):
                        self.set_fiducial_pixel(this_row, this_col, center_fiducial_color)
                        self.set_fiducial_pixel(this_row, this_col, center_fiducial_color)
        else:
            print(
                "ERROR: In TargetColor.set_center_fiducial(), unexpected negative center_fiducial_half_margin_pix = "
                + str(center_fiducial_half_margin_pix)
            )
            assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION.

    def set_ticks_along_top_and_bottom_edges(self, n_ticks_x, tick_length, tick_width_pix, tick_color):
        """
        Draws tick marks along the top and bottom edges of the image.

        Parameters
        ----------
        n_ticks_x : int
            Number of tick marks to draw along the top and bottom horizontal target edges.
        tick_length : float
            Length to draw edge tick marks in meters.
        tick_width_pix : int
            Width to draw edge tick marks in pixels; should be an odd number.
        tick_color : cl
            Color of edge tick marks.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        width = n_cols
        dx_tick = width / (n_ticks_x - 1)
        # Number of pixels in tick width either side of centerline.  Use int() to intentionally truncate.
        tick_half_margin_pix = int(tick_width_pix / 2.0)
        tick_length_pix = round(tick_length * self.dpm)  # Pixels
        for tick_idx_x in range(n_ticks_x):
            x_tick = tick_idx_x * dx_tick
            col_tick = round(x_tick)
            for tick_idx_y in range(tick_length_pix):
                this_row_from_top = tick_idx_y
                this_row_from_bottom = (n_rows - this_row_from_top) - 1
                if tick_half_margin_pix == 0:
                    if col_tick < 0:
                        col_tick = 0
                    if col_tick == n_cols:  # Use "==" because if tick is past end of image, don't draw.
                        col_tick = n_cols - 1
                    if (col_tick >= 0) and (col_tick < n_cols):
                        self.set_fiducial_pixel(this_row_from_top, col_tick, tick_color)
                        self.set_fiducial_pixel(this_row_from_bottom, col_tick, tick_color)
                elif tick_half_margin_pix > 0:
                    for this_col in range(col_tick - tick_half_margin_pix, (col_tick + tick_half_margin_pix) + 1):
                        if (this_col >= 0) and (this_col < n_cols):
                            self.set_fiducial_pixel(this_row_from_top, this_col, tick_color)
                            self.set_fiducial_pixel(this_row_from_bottom, this_col, tick_color)
                else:
                    print(
                        "ERROR: In TargetColor.set_ticks_along_top_and_bottom_edges(), unexpected negative tick_half_margin_pix = "
                        + str(tick_half_margin_pix)
                    )
                    assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION.

    def set_ticks_along_left_and_right_edges(self, n_ticks_y, tick_length, tick_width_pix, tick_color):
        """
        Draws tick marks along the left and right edges of the image.

        Parameters
        ----------
        n_ticks_y : int
            Number of tick marks to draw along the left and right vertical target edges.
        tick_length : float
            Length to draw edge tick marks in meters.
        tick_width_pix : int
            Width to draw edge tick marks in pixels; should be an odd number.
        tick_color : cl
            Color of edge tick marks.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        height = n_rows
        dy_tick = height / (n_ticks_y - 1)
        # Number of pixels in tick width either side of centerline.  Use int() to intentionally truncate.
        tick_half_margin_pix = int(tick_width_pix / 2.0)
        tick_length_pix = round(tick_length * self.dpm)  # Pixels
        for tick_idx_y in range(n_ticks_y):
            y_tick = tick_idx_y * dy_tick
            row_tick = round(y_tick)
            for tick_idx_x in range(tick_length_pix):
                this_col_from_left = tick_idx_x
                this_col_from_right = (n_cols - this_col_from_left) - 1
                if tick_half_margin_pix == 0:
                    if row_tick < 0:
                        row_tick = 0
                    if row_tick == n_rows:  # Use "==" because if tick is past end of image, don't draw.
                        row_tick = n_rows - 1
                    if (row_tick >= 0) and (row_tick < n_rows):
                        self.set_fiducial_pixel(row_tick, this_col_from_left, tick_color)
                        self.set_fiducial_pixel(row_tick, this_col_from_right, tick_color)
                elif tick_half_margin_pix > 0:
                    for this_row in range(row_tick - tick_half_margin_pix, (row_tick + tick_half_margin_pix) + 1):
                        if (this_row >= 0) and (this_row < n_rows):
                            self.set_fiducial_pixel(this_row, this_col_from_left, tick_color)
                            self.set_fiducial_pixel(this_row, this_col_from_right, tick_color)
                else:
                    print(
                        "ERROR: In TargetColor.set_ticks_along_left_and_right_edges(), unexpected negative tick_half_margin_pix = "
                        + str(tick_half_margin_pix)
                    )
                    assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION.

    def set_fiducial_pixel(self, this_row: int, this_col: int, color: cl) -> None:
        """
        Sets a pixel in the image to the specified color.

        Parameters
        ----------
        this_row : int
            The row index of the pixel to set.
        this_col : int
            The column index of the pixel to set.
        color : cl
            The color to set the pixel to.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        if (this_row >= 0) and (this_row < n_rows):
            if (this_col >= 0) and (this_col < n_cols):
                self.image[this_row, this_col, 0] = color.rgb_255()[0]
                self.image[this_row, this_col, 1] = color.rgb_255()[1]
                self.image[this_row, this_col, 2] = color.rgb_255()[2]

    # Blue underlying red cross green.
    def set_image_to_blue_under_red_cross_green(self):
        """
        Sets the image to a blue color under a red cross on a green background.

        This method modifies the image to display a blue color beneath a red cross
        on a green background, effectively creating a specific color pattern.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        self.image = tc2r.construct_blue_under_red_cross_green(n_cols, n_rows)

    # Square inscribed in the [R,G,B] space basis vector hexagon.
    def set_image_to_rgb_cube_inscribed_square(self, project_to_cube):
        """
        Sets the image to a square inscribed in the RGB color cube.

        Parameters
        ----------
        project_to_cube : bool
            If True, the square will be projected onto the RGB cube; otherwise, it will be unprojected.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        n_rows, n_cols = self.rows_cols()
        self.image = tc2r.construct_rgb_cube_inscribed_square_image(n_cols, n_rows, project_to_cube)

    # Compute color saturation adjustment.
    # ?? SCAFFOLDING RCB -- ADD TYPE TIPS.
    def adjust_rgb_color_saturation(self, rgb, saturation_fraction, max_rgb):
        """
        Adjusts the saturation of a given RGB color.

        Parameters
        ----------
        rgb : tuple
            A tuple containing the RGB values of the color (red, green, blue).
        saturation_fraction : float
            The fraction by which to adjust the saturation (0.0 to 1.0).
        max_rgb : int
            The maximum RGB value (typically 255).

        Returns
        -------
        tuple
            A tuple containing the adjusted RGB values.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Lookup color.
        original_red = rgb[0]
        original_green = rgb[1]
        original_blue = rgb[2]
        # Compute new color.
        # ?? SCAFFOLDING RCB -- DOCUMENT ABANDONED MID-IMPLEMENTATION
        new_red = original_red * saturation_fraction  # ?? SCAFFOLDING RCB -- TEMPORARY
        new_green = original_green * saturation_fraction  # ?? SCAFFOLDING RCB -- TEMPORARY
        new_blue = original_blue * saturation_fraction  # ?? SCAFFOLDING RCB -- TEMPORARY
        # Return.
        return (new_red, new_green, new_blue)

    # Adjust color saturation.
    def adjust_color_saturation(self, saturation_fraction: float) -> None:
        """
        Adjusts the color saturation of each pixel in the image.

        Each pixel's vector from the origin in RGB space is adjusted to have a length
        equal to (saturation_fraction * max_length), where max_length is the distance
        from the origin to the bounding cube in RGB space defined by the maximum pixel value.

        Parameters
        ----------
        saturation_fraction : float
            The fraction to adjust the saturation by (0.0 to 1.0).
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        # Modify image content.
        n_rows, n_cols = self.rows_cols()
        for row in range(0, n_rows):
            for col in range(0, n_cols):
                # Lookup color.
                original_red = self.image[row, col, 0]
                original_green = self.image[row, col, 1]
                original_blue = self.image[row, col, 2]
                original_rgb = (original_red, original_green, original_blue)
                # Compute new color.
                max_rgb = 255  # ?? SCAFFOLDING RCB -- GET THIS VALUE RATIONALLY.
                new_red, new_green, new_blue = self.adjust_rgb_color_saturation(
                    original_rgb, saturation_fraction, max_rgb
                )
                # Set pixel color.
                self.image[row, col, 0] = new_red
                self.image[row, col, 1] = new_green
                self.image[row, col, 2] = new_blue
        # Update pattern description.
        original_description = self.pattern_description
        new_description = original_description + "_sat" + str(saturation_fraction)
        self.pattern_description = new_description


# GENERATORS


def construct_target_linear_color_bar(
    image_width: float,  # Meter
    image_height: float,  # Meter
    dpm: float,  # Dots per meter
    color_below_min: cl,  # Color for values below min of color bar.
    color_bar,  # Color bar mapping values to colors.  # ?? SCAFFOLDING RCB -- ADD COLOR_BAR TYPE TIP
    color_bar_name: str,  # ?? SCAFFOLDING RCB -- REPLACE "color_bar_name" WITH CLASS FETCH
    color_above_max: cl,  # Color for values above max of color bar.
    x_or_y: str,  # Direction of color variation.
    discrete_or_continuous: str,
    # Defines scheme for modulating saturation from left to right.  Values: None, 'saturated_to_white' or 'light_to_saturated'
    lateral_gradient_type: str = None,
    # Dimensionless.  Applies if lateral_gradient_type == 'saturated_to_white'.
    saturated_to_white_exponent: float = 1.5,
    light_to_saturated_min: float = 0.2,  # Dimensionless.  Applies if lateral_gradient_type == 'light_to_saturated'.
    light_to_saturated_max: float = 1.0,  # Dimensionless.  Applies if lateral_gradient_type == 'light_to_saturated'.
) -> TargetColor:
    """
    Constructs a linear color bar target.

    Parameters
    ----------
    image_width : float
        The width of the image in meters.
    image_height : float
        The height of the image in meters.
    dpm : float
        Dots per meter for the image resolution.
    color_below_min : cl
        The color to use for values below the minimum of the color bar.
    color_bar :
        The color bar mapping values to colors.
    color_bar_name : str
        The name of the color bar for output purposes.
    color_above_max : cl
        The color to use for values above the maximum of the color bar.
    x_or_y : str
        The direction of color variation ('x' or 'y').
    discrete_or_continuous : str
        Specifies whether the color interpolation is 'discrete' or 'continuous'.
    lateral_gradient_type : str, optional
        Defines the scheme for modulating saturation from left to right. Values: None, 'saturated_to_white', or 'light_to_saturated'.
    saturated_to_white_exponent : float, optional
        Dimensionless exponent applied if lateral_gradient_type is 'saturated_to_white'.
    light_to_saturated_min : float, optional
        Dimensionless minimum saturation value applied if lateral_gradient_type is 'light_to_saturated'.
    light_to_saturated_max : float, optional
        Dimensionless maximum saturation value applied if lateral_gradient_type is 'light_to_saturated'.

    Returns
    -------
    TargetColor
        An instance of TargetColor configured with the specified linear color bar.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Blank target.
    target = tc.TargetColor(image_width, image_height, dpm, color_below_min)
    # Set colors.
    if x_or_y == "x":
        target.set_image_to_linear_color_bar_x(
            color_below_min, color_bar, color_above_max, discrete_or_continuous
        )  # ?? SCAFFOLDING RCB -- UPDATE COLOR BAR X TO MATCH Y.
    elif x_or_y == "y":
        target.set_image_to_linear_color_bar_y(
            color_below_min,
            color_bar,
            color_above_max,
            discrete_or_continuous,
            lateral_gradient_type=lateral_gradient_type,
            saturated_to_white_exponent=saturated_to_white_exponent,
            light_to_saturated_min=light_to_saturated_min,
            light_to_saturated_max=light_to_saturated_max,
        )

    else:
        print('ERROR: In construct_target_linear_color_bar(), x_or_y has unexpected value "' + str(x_or_y) + '"')
        assert False
    # Set pattern description.
    include_above_below_in_pattern_name = True
    if include_above_below_in_pattern_name:
        color_pattern_name = (
            color_below_min.short_name + "." + color_bar_name + "_linear." + color_above_max.short_name
        )  # ?? SCAFFOLDING RCB -- REPLACE "color_bar_name" WITH CLASS FETCH
    else:
        color_pattern_name = color_bar_name  # ?? SCAFFOLDING RCB -- REPLACE "color_bar_name" WITH CLASS FETCH
    target.set_pattern_description(color_pattern_name + "_" + x_or_y + "_" + discrete_or_continuous)
    # Return.
    return target


def construct_target_polar_color_bar(  # Target dimensions.
    image_width: float,  # Meter
    image_height: float,  # Meter
    dpm: float,  # Dots per meter
    # Control color by angle.
    color_below_min: cl = cl.black(),  # Color to use for values below min limit of color bar.
    color_bar=tcc.O_color_bar(),  # Color bar for angular values in [0,2pi).  # ?? SCAFFOLDING RCB -- TYPE TIP NEEDED.
    color_bar_name: str = "O",  # Terse name of color bar, for output filename.  # ?? SCAFFOLDING RCB -- REPLACE "color_bar_name" WITH CLASS FETCH
    color_above_max: cl = cl.white(),  # Color to use for values of max limit of color bar.
    # Indicates whether to interpolate colors between segments. Values 'discrete' or 'continuous'  # ?? SCAFFOLDING RCB -- RENAME TO "color_interpolation_type" ?
    discrete_or_continuous: str = "continuous",
    # Modulating saturation with radius.
    # Defines range for saturation modulation as a function of radius. Values 'circle' or 'image_boundary'
    pattern_boundary: str = "image_boundary",
    # Defines scheme for modulatiing saturation as a function of radius. Values 'saturated_center_to_white' or 'light_center_to_saturated'
    radial_gradient_type: str = "saturated_center_to_white",
    radial_gradient_name: str = "s2w",  # Terse name of radial intensity function, for output filename. Values 's2w' or 'l2s'
    # Dimensionless.  Applies if radial_gradient_type == 'saturated_center_to_white'.
    saturated_center_to_white_exponent: float = 1.5,
    # Dimensionless.  Applies if radial_gradient_type == 'light_center_to_saturated'.
    light_center_to_saturated_saturation_min: float = 0.2,
    # Dimensionless.  Applies if radial_gradient_type == 'light_center_to_saturated'.
    light_center_to_saturated_saturation_max: float = 1.0,
    # Fiducial marks.
    draw_center_fiducial=True,  # Boolean.   Whether to draw a mark at the target center.
    center_fiducial_width_pix=3,  # Pixels.    Width of center fiducial; should be odd number.
    center_fiducial_color: cl = cl.white(),  # Color.     Color of center fiducial.
    draw_edge_fiducials=True,  # Boolean.   Whether to draw fiducial tick marks along target edges.
    n_ticks_x=7,  # No units.  Number of tick marks to draw along top/bottom horizontal target edges.
    n_ticks_y=7,  # No units.  Number of tick marks to draw along left/right vertical target edges.
    tick_length=0.010,  # Meters.    Length to draw edge tick marks.
    tick_width_pix=3,  # Pixels.    Width to draw edge tick marks; should be odd number.
    tick_color: cl = cl.black(),  # Color.     Color of edge tick marks.
) -> TargetColor:
    """
    Constructs a polar color bar target.

    Parameters
    ----------
    image_width : float
        The width of the image in meters.
    image_height : float
        The height of the image in meters.
    dpm : float
        Dots per meter for the image resolution.
    color_below_min : cl, optional
        The color to use for values below the minimum limit of the color bar. Default is black.
    color_bar :
        The color bar for angular values in the range [0, 2π).
    color_bar_name : str, optional
        A terse name of the color bar for output filename. Default is 'O'.
    color_above_max : cl, optional
        The color to use for values above the maximum limit of the color bar. Default is white.
    discrete_or_continuous : str, optional
        Indicates whether to interpolate colors between segments. Values: 'discrete' or 'continuous'. Default is 'continuous'.
    pattern_boundary : str, optional
        Defines the range for saturation modulation as a function of radius. Values: 'circle' or 'image_boundary'. Default is 'image_boundary'.
    radial_gradient_type : str, optional
        Defines the scheme for modulating saturation as a function of radius. Values: 'saturated_center_to_white' or 'light_center_to_saturated'. Default is 'saturated_center_to_white'.
    radial_gradient_name : str, optional
        A terse name of the radial intensity function for output filename. Default is 's2w'.
    saturated_center_to_white_exponent : float, optional
        Dimensionless exponent applied if radial_gradient_type is 'saturated_center_to_white'. Default is 1.5.
    light_center_to_saturated_saturation_min : float, optional
        Dimensionless minimum saturation value applied if radial_gradient_type is 'light_center_to_saturated'. Default is 0.2.
    light_center_to_saturated_saturation_max : float, optional
        Dimensionless maximum saturation value applied if radial_gradient_type is 'light_center_to_saturated'. Default is 1.0.
    draw_center_fiducial : bool, optional
        Whether to draw a mark at the target center. Default is True.
    center_fiducial_width_pix : int, optional
        Width of the center fiducial in pixels; should be an odd number. Default is 3.
    center_fiducial_color : cl, optional
        Color of the center fiducial. Default is white.
    draw_edge_fiducials : bool, optional
        Whether to draw fiducial tick marks along the target edges. Default is True.
    n_ticks_x : int, optional
        Number of tick marks to draw along the top and bottom horizontal target edges. Default is 7.
    n_ticks_y : int, optional
        Number of tick marks to draw along the left and right vertical target edges. Default is 7.
    tick_length : float, optional
        Length to draw edge tick marks in meters. Default is 0.010.
    tick_width_pix : int, optional
        Width to draw edge tick marks in pixels; should be an odd number. Default is 3.
    tick_color : cl, optional
        Color of edge tick marks. Default is black.

    Returns
    -------
    TargetColor
        An instance of TargetColor configured with the specified polar color bar.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Blank target.
    target = tc.TargetColor(image_width, image_height, dpm, color_below_min)
    # Set colors.
    target.set_image_to_polar_color_bar(  # Control color by angle.
        color_below_min=color_below_min,
        color_bar=color_bar,
        color_above_max=color_above_max,
        discrete_or_continuous=discrete_or_continuous,
        # Modulating saturation with radius.
        pattern_boundary=pattern_boundary,
        radial_gradient_type=radial_gradient_type,
        saturated_center_to_white_exponent=saturated_center_to_white_exponent,
        light_center_to_saturated_saturation_min=light_center_to_saturated_saturation_min,
        light_center_to_saturated_saturation_max=light_center_to_saturated_saturation_max,
        # Fiducial marks.
        draw_center_fiducial=draw_center_fiducial,
        center_fiducial_width_pix=center_fiducial_width_pix,
        center_fiducial_color=center_fiducial_color,
        draw_edge_fiducials=draw_edge_fiducials,
        n_ticks_x=n_ticks_x,
        n_ticks_y=n_ticks_y,
        tick_length=tick_length,
        tick_width_pix=tick_width_pix,
        tick_color=tick_color,
    )
    # Set pattern description.
    # Color bar.
    pattern_description = color_below_min.short_name + "." + color_bar_name + "." + color_above_max.short_name
    # Linear vs. polar.
    pattern_description += "_polar_" + radial_gradient_name
    # Color interpolation.
    if discrete_or_continuous == "discrete":
        pattern_description += "_disc"
    elif discrete_or_continuous == "continuous":
        pattern_description += "_cont"
    else:
        print(
            "ERROR: In TargetColor.construct_target_polar_color_bar(), encountered unexpected discrete_or_continuous = "
            + str(discrete_or_continuous)
        )
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION
    # Radial gradient.
    if radial_gradient_type == "saturated_center_to_white":
        pattern_description += "_exp" + "{0:.2f}".format(saturated_center_to_white_exponent)
    elif radial_gradient_type == "light_center_to_saturated":
        pattern_description += (
            "_sat"
            + "{0:.2f}".format(light_center_to_saturated_saturation_min)
            + "to"
            + "{0:.2f}".format(light_center_to_saturated_saturation_max)
        )
    else:
        print(
            "ERROR: In TargetColor.construct_target_polar_color_bar(), encountered unexpected radial_gradient_type = "
            + str(radial_gradient_type)
        )
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION
    # Radial boundary.
    if pattern_boundary == "circle":
        pattern_description += "_circ"
    elif pattern_boundary == "image_boundary":
        pattern_description += "_box"
    else:
        print(
            "ERROR: In TargetColor.construct_target_polar_color_bar(), encountered unexpected pattern_boundary = "
            + str(pattern_boundary)
        )
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION
    # Fiducials.
    if draw_center_fiducial:
        pattern_description += "_cf." + center_fiducial_color.short_name
    if draw_edge_fiducials:
        pattern_description += "_ef." + tick_color.short_name
    # Set result.
    target.set_pattern_description(pattern_description)

    # Return.
    return target


def construct_target_blue_under_red_cross_green(
    image_width: float, image_height: float, dpm: float  # Meter  # Meter  # Dots per meter
) -> TargetColor:
    """
    Constructs a target with a blue color under a red cross on a green background.

    Parameters
    ----------
    image_width : float
        The width of the image in meters.
    image_height : float
        The height of the image in meters.
    dpm : float
        Dots per meter for the image resolution.

    Returns
    -------
    TargetColor
        An instance of TargetColor configured with the specified blue under red cross green pattern.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Blank target.
    target = tc.TargetColor(
        image_width, image_height, dpm, cl.black()
    )  # We will cover the entire image with new colors.
    # Set colors.
    target.set_image_to_blue_under_red_cross_green()
    # Set pattern description.
    color_pattern_name_root = (
        "blue_under_red_cross_green"  # ?? SCAFFOLDING RCB -- REPLACE "color_pattern_name_root" WITH CLASS FETCH?
    )
    color_pattern_name = color_pattern_name_root
    target.set_pattern_description(color_pattern_name)
    # Return.
    return target


def construct_target_rgb_cube_inscribed_square(
    image_width: float, image_height: float, dpm: float, project_to_cube: bool  # Meter  # Meter  # Dots per meter
) -> TargetColor:
    """
    Constructs a target with a square inscribed in the RGB color cube.

    Parameters
    ----------
    image_width : float
        The width of the image in meters.
    image_height : float
        The height of the image in meters.
    dpm : float
        Dots per meter for the image resolution.
    project_to_cube : bool
        If True, the square will be projected onto the RGB cube; otherwise, it will be unprojected.

    Returns
    -------
    TargetColor
        An instance of TargetColor configured with the specified RGB cube inscribed square pattern.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Blank target.
    target = tc.TargetColor(
        image_width, image_height, dpm, cl.black()
    )  # We will cover the entire image with new colors.
    # Set colors.
    target.set_image_to_rgb_cube_inscribed_square(project_to_cube)
    # Set pattern description.
    color_pattern_name_root = (
        "rgb_cube_inscribed_square"  # ?? SCAFFOLDING RCB -- REPLACE "color_pattern_name_root" WITH CLASS FETCH?
    )
    if project_to_cube:
        color_pattern_name_root += "_projected"
    else:
        color_pattern_name_root += "_unprojected"
    color_pattern_name = color_pattern_name_root
    target.set_pattern_description(color_pattern_name)
    # Return.
    return target


# TARGET EXTENSION

# ?? SCAFFOLDING RCB -- THE EXTEND LEFT/RIGHT/TOP/BOTTOM ROUTINES HAVE A LOT OF CODE COPYING.  DOES IT MAKE SENSE TO MERGE THEM?

# ?? SCAFFOLDING RCB -- THIS SHOULD BE WITHIN TARGET_ABSTRACT.  HOWEVER, WE NEED TO FIGURE OUT HOW TO MANAGE THE NEW TARGET CONSTRUCTOR,
# ?? SCAFFOLDING RCB    SO THAT IT IS A DERIVED CLASS, NOT BASE CLASS.  MAYBE WE NEED TO CREATE A "SHELL" TARGET_COLOR, AND THEN PASS IT
# ?? SCAFFOLDING RCB    INTO THE GENERIC FUNCTION WIHTIN THE TARGET_ABSTRACT FILE.  FOR LATER.

# ?? SCAFFOLDING RCB -- SHOULD THIS BE DONE BY COMBINING IMAGES WITH AN IMAGE LIBRARY, OR BY ARRAY COMBINATION IN NUMPY?


def extend_target_left(
    target: TargetColor,  # Target to extend.
    new_pixels: int,  # Pixels.  Number of additional pixels to extend.
    new_color: cl.Color,  # Color to fill new pixels.
    new_target_name: str = None,  # If none, new target name will extend input target name.
) -> TargetColor:
    """
    Constructs a new target by extending the input target with additional pixels on the left side.

    Parameters
    ----------
    target : TargetColor
        The target to extend.
    new_pixels : int
        The number of additional pixels to extend on the left side.
    new_color : cl.Color
        The color to fill the new pixels.
    new_target_name : str, optional
        If provided, this will be the name of the new target; otherwise, the name will be derived from the input target.

    Returns
    -------
    TargetColor
        A new instance of TargetColor with the specified extension on the left side.

    Raises
    ------
    AssertionError
        If the new target dimensions do not match the expected dimensions after extension.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Fetch sizes.
    n_rows, n_cols, n_bands = target.rows_cols_bands()
    dpm = (
        target.dpm
    )  # ?? SCAFFOLDING RCB -- SHOULD THESE BE ACCESSOR FUNCTIONS?  WHAT IS OPENCSP POLICY/PATTERN ON THIS?

    # Compute new size.
    new_n_rows = n_rows
    new_n_cols = n_cols + new_pixels
    new_n_bands = n_bands

    # Construct new target.
    new_image_width = new_n_cols / dpm
    new_image_height = target.image_height
    new_image_dpm = dpm
    new_target = TargetColor(new_image_width, new_image_height, new_image_dpm, new_color)

    # Check code consistency.
    check_n_rows, check_n_cols, check_n_bands = new_target.rows_cols_bands()
    if new_n_rows != check_n_rows:
        print(
            "ERROR: In extend_target_left(), unequal new_n_rows="
            + str(new_n_rows)
            + " and check_n_rows="
            + str(check_n_rows)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_cols != check_n_cols:
        print(
            "ERROR: In extend_target_left(), unequal new_n_cols="
            + str(new_n_cols)
            + " and check_n_cols="
            + str(check_n_cols)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_bands != check_n_bands:
        print(
            "ERROR: In extend_target_left(), unequal new_n_bands="
            + str(new_n_bands)
            + " and check_n_bands="
            + str(check_n_bands)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION

    # Copy image.
    image = target.image
    new_image = new_target.image
    start_col = new_pixels
    for row in range(new_n_rows):
        for col in range(n_cols):
            for band in range(new_n_bands):
                new_image[row][start_col + col][band] = image[row][col][band]

    # Set description.
    if new_target_name == None:
        new_target.set_pattern_description(
            "l" + str(new_pixels) + new_color.short_name + "px_" + target.pattern_description
        )
    else:
        new_target.set_pattern_description(new_target_name)

    # Return.
    return new_target


def extend_target_right(
    target: TargetColor,  # Target to extend.
    new_pixels: int,  # Pixels.  Number of additional pixels to extend.
    new_color: cl.Color,  # Color to fill new pixels.
    new_target_name: str = None,  # If none, new target name will extend input target name.
) -> TargetColor:
    """
    Constructs a new target by extending the input target with additional pixels on the right side.

    Parameters
    ----------
    target : TargetColor
        The target to extend.
    new_pixels : int
        The number of additional pixels to extend on the right side.
    new_color : cl.Color
        The color to fill the new pixels.
    new_target_name : str, optional
        If provided, this will be the name of the new target; otherwise, the name will be derived from the input target.

    Returns
    -------
    TargetColor
        A new instance of TargetColor with the specified extension on the right side.

    Raises
    ------
    AssertionError
        If the new target dimensions do not match the expected dimensions after extension.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Fetch sizes.
    n_rows, n_cols, n_bands = target.rows_cols_bands()
    dpm = (
        target.dpm
    )  # ?? SCAFFOLDING RCB -- SHOULD THESE BE ACCESSOR FUNCTIONS?  WHAT IS OPENCSP POLICY/PATTERN ON THIS?

    # Compute new size.
    new_n_rows = n_rows
    new_n_cols = n_cols + new_pixels
    new_n_bands = n_bands

    # Construct new target.
    new_image_width = new_n_cols / dpm
    new_image_height = target.image_height
    new_image_dpm = dpm
    new_target = TargetColor(new_image_width, new_image_height, new_image_dpm, new_color)

    # Check code consistency.
    check_n_rows, check_n_cols, check_n_bands = new_target.rows_cols_bands()
    if new_n_rows != check_n_rows:
        print(
            "ERROR: In extend_target_right(), unequal new_n_rows="
            + str(new_n_rows)
            + " and check_n_rows="
            + str(check_n_rows)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_cols != check_n_cols:
        print(
            "ERROR: In extend_target_right(), unequal new_n_cols="
            + str(new_n_cols)
            + " and check_n_cols="
            + str(check_n_cols)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_bands != check_n_bands:
        print(
            "ERROR: In extend_target_right(), unequal new_n_bands="
            + str(new_n_bands)
            + " and check_n_bands="
            + str(check_n_bands)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION

    # Copy image.
    image = target.image
    new_image = new_target.image
    start_col = new_pixels
    for row in range(new_n_rows):
        for col in range(n_cols):
            for band in range(new_n_bands):
                new_image[row][col][band] = image[row][col][band]

    # Set description.
    if new_target_name == None:
        new_target.set_pattern_description(
            target.pattern_description + "_r" + str(new_pixels) + new_color.short_name + "px"
        )
    else:
        new_target.set_pattern_description(new_target_name)

    # Return.
    return new_target


# ?? SCAFFOLDING RCB -- THIS SHOULD BE WITHIN TARGET_ABSTRACT.  HOWEVER, WE NEED TO FIGURE OUT HOW TO MANAGE THE NEW TARGET CONSTRUCTOR,
# ?? SCAFFOLDING RCB    SO THAT IT IS A DERIVED CLASS, NOT BASE CLASS.  MAYBE WE NEED TO CREATE A "SHELL" TARGET_COLOR, AND THEN PASS IT
# ?? SCAFFOLDING RCB    INTO THE GENERIC FUNCTION WIHTIN THE TARGET_ABSTRACT FILE.  FOR LATER.

# ?? SCAFFOLDING RCB -- SHOULD THIS BE DONE BY COMBINING IMAGES WITH AN IMAGE LIBRARY, OR BY ARRAY COMBINATION IN NUMPY?


def extend_target_top(
    target: TargetColor,  # Target to extend.
    new_pixels: int,  # Pixels.  Number of additional pixels to extend.
    new_color: cl.Color,  # Color to fill new pixels.
    new_target_name: str = None,  # If none, new target name will extend input target name.
) -> TargetColor:
    """
    Constructs a new target by extending the input target with additional pixels on the top side.

    Parameters
    ----------
    target : TargetColor
        The target to extend.
    new_pixels : int
        The number of additional pixels to extend on the top side.
    new_color : cl.Color
        The color to fill the new pixels.
    new_target_name : str, optional
        If provided, this will be the name of the new target; otherwise, the name will be derived from the input target.

    Returns
    -------
    TargetColor
        A new instance of TargetColor with the specified extension on the top side.

    Raises
    ------
    AssertionError
        If the new target dimensions do not match the expected dimensions after extension.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Fetch sizes.
    n_rows, n_cols, n_bands = target.rows_cols_bands()
    dpm = (
        target.dpm
    )  # ?? SCAFFOLDING RCB -- SHOULD THESE BE ACCESSOR FUNCTIONS?  WHAT IS OPENCSP POLICY/PATTERN ON THIS?

    # Compute new size.
    new_n_rows = n_rows + new_pixels
    new_n_cols = n_cols
    new_n_bands = n_bands

    # Construct new target.
    new_image_width = target.image_width
    new_image_height = new_n_rows / dpm
    new_image_dpm = dpm
    new_target = TargetColor(new_image_width, new_image_height, new_image_dpm, new_color)

    # Check code consistency.
    check_n_rows, check_n_cols, check_n_bands = new_target.rows_cols_bands()
    if new_n_rows != check_n_rows:
        print(
            "ERROR: In extend_target_top(), unequal new_n_rows="
            + str(new_n_rows)
            + " and check_n_rows="
            + str(check_n_rows)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_cols != check_n_cols:
        print(
            "ERROR: In extend_target_top(), unequal new_n_cols="
            + str(new_n_cols)
            + " and check_n_cols="
            + str(check_n_cols)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_bands != check_n_bands:
        print(
            "ERROR: In extend_target_top(), unequal new_n_bands="
            + str(new_n_bands)
            + " and check_n_bands="
            + str(check_n_bands)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION

    # Copy image.
    image = target.image
    new_image = new_target.image
    start_row = new_pixels
    for row in range(n_rows):
        for col in range(new_n_cols):
            for band in range(new_n_bands):
                new_image[start_row + row][col][band] = image[row][col][band]

    # Set description.
    if new_target_name == None:
        new_target.set_pattern_description(
            "t" + str(new_pixels) + new_color.short_name + "px_" + target.pattern_description
        )
    else:
        new_target.set_pattern_description(new_target_name)

    # Return.
    return new_target


# ?? SCAFFOLDING RCB -- THIS SHOULD BE WITHIN TARGET_ABSTRACT.  HOWEVER, WE NEED TO FIGURE OUT HOW TO MANAGE THE NEW TARGET CONSTRUCTOR,
# ?? SCAFFOLDING RCB    SO THAT IT IS A DERIVED CLASS, NOT BASE CLASS.  MAYBE WE NEED TO CREATE A "SHELL" TARGET_COLOR, AND THEN PASS IT
# ?? SCAFFOLDING RCB    INTO THE GENERIC FUNCTION WIHTIN THE TARGET_ABSTRACT FILE.  FOR LATER.

# ?? SCAFFOLDING RCB -- SHOULD THIS BE DONE BY COMBINING IMAGES WITH AN IMAGE LIBRARY, OR BY ARRAY COMBINATION IN NUMPY?


def extend_target_bottom(
    target: TargetColor,  # Target to extend.
    new_pixels: int,  # Pixels.  Number of additional pixels to extend.
    new_color: cl.Color,  # Color to fill new pixels.
    new_target_name: str = None,  # If none, new target name will extend input target name.
) -> TargetColor:
    """
    Constructs a new target by extending the input target with additional pixels on the bottom side.

    Parameters
    ----------
    target : TargetColor
        The target to extend.
    new_pixels : int
        The number of additional pixels to extend on the bottom side.
    new_color : cl.Color
        The color to fill the new pixels.
    new_target_name : str, optional
        If provided, this will be the name of the new target; otherwise, the name will be derived from the input target.

    Returns
    -------
    TargetColor
        A new instance of TargetColor with the specified extension on the bottom side.

    Raises
    ------
    AssertionError
        If the new target dimensions do not match the expected dimensions after extension.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Fetch sizes.
    n_rows, n_cols, n_bands = target.rows_cols_bands()
    dpm = (
        target.dpm
    )  # ?? SCAFFOLDING RCB -- SHOULD THESE BE ACCESSOR FUNCTIONS?  WHAT IS OPENCSP POLICY/PATTERN ON THIS?

    # Compute new size.
    new_n_rows = n_rows + new_pixels
    new_n_cols = n_cols
    new_n_bands = n_bands

    # Construct new target.
    new_image_width = target.image_width
    new_image_height = new_n_rows / dpm
    new_image_dpm = dpm
    new_target = TargetColor(new_image_width, new_image_height, new_image_dpm, new_color)

    # Check code consistency.
    check_n_rows, check_n_cols, check_n_bands = new_target.rows_cols_bands()
    if new_n_rows != check_n_rows:
        print(
            "ERROR: In extend_target_bottom(), unequal new_n_rows="
            + str(new_n_rows)
            + " and check_n_rows="
            + str(check_n_rows)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_cols != check_n_cols:
        print(
            "ERROR: In extend_target_bottom(), unequal new_n_cols="
            + str(new_n_cols)
            + " and check_n_cols="
            + str(check_n_cols)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_bands != check_n_bands:
        print(
            "ERROR: In extend_target_bottom(), unequal new_n_bands="
            + str(new_n_bands)
            + " and check_n_bands="
            + str(check_n_bands)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION

    # Copy image.
    image = target.image
    new_image = new_target.image
    for row in range(n_rows):
        for col in range(new_n_cols):
            for band in range(new_n_bands):
                new_image[row][col][band] = image[row][col][band]

    # Set description.
    if new_target_name == None:
        new_target.set_pattern_description(
            target.pattern_description + "_b" + str(new_pixels) + new_color.short_name + "px"
        )
    else:
        new_target.set_pattern_description(new_target_name)

    # Return.
    return new_target


# ?? SCAFFOLDING RCB -- THIS SHOULD BE WITHIN TARGET_ABSTRACT.  HOWEVER, WE NEED TO FIGURE OUT HOW TO MANAGE THE NEW TARGET CONSTRUCTOR,
# ?? SCAFFOLDING RCB    SO THAT IT IS A DERIVED CLASS, NOT BASE CLASS.  MAYBE WE NEED TO CREATE A "SHELL" TARGET_COLOR, AND THEN PASS IT
# ?? SCAFFOLDING RCB    INTO THE GENERIC FUNCTION WIHTIN THE TARGET_ABSTRACT FILE.  FOR LATER.

# ?? SCAFFOLDING RCB -- SHOULD THIS BE DONE BY COMBINING IMAGES WITH AN IMAGE LIBRARY, OR BY ARRAY COMBINATION IN NUMPY?


def extend_target_all(
    target: TargetColor,  # Target to extend.
    new_pixels: int,  # Pixels.  Number of additional pixels to extend.
    new_color: cl.Color,  # Color to fill new pixels.
    new_target_name: str = None,  # If none, new target name will extend input target name.
) -> TargetColor:
    """
    Constructs a new target by extending the input target with additional pixels on all sides.

    Parameters
    ----------
    target : TargetColor
        The target to extend.
    new_pixels : int
        The number of additional pixels to extend on all sides.
    new_color : cl.Color
        The color to fill the new pixels.
    new_target_name : str, optional
        If provided, this will be the name of the new target; otherwise, the name will be derived from the input target.

    Returns
    -------
    TargetColor
        A new instance of TargetColor with the specified extension on all sides.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Extend.
    extended_target_l = tc.extend_target_left(target, new_pixels, new_color)
    extended_target_lr = tc.extend_target_right(extended_target_l, new_pixels, new_color)
    extended_target_lrt = tc.extend_target_top(extended_target_lr, new_pixels, new_color)
    extended_target_lrtb = tc.extend_target_bottom(extended_target_lrt, new_pixels, new_color)
    new_target = extended_target_lrtb

    # Set description.
    if new_target_name == None:
        # Add to target pattern_description to discard per-side description changes.
        new_target.set_pattern_description(
            "bord" + str(new_pixels) + new_color.short_name + "px_" + target.pattern_description
        )
    else:
        new_target.set_pattern_description(new_target_name)

    # Return.
    return new_target


# TARGET COMBINATION

# ?? SCAFFOLDING RCB -- THIS SHOULD BE WITHIN TARGET_ABSTRACT.  HOWEVER, WE NEED TO FIGURE OUT HOW TO MANAGE THE NEW TARGET CONSTRUCTOR,
# ?? SCAFFOLDING RCB    SO THAT IT IS A DERIVED CLASS, NOT BASE CLASS.  MAYBE WE NEED TO CREATE A "SHELL" TARGET_COLOR, AND THEN PASS IT
# ?? SCAFFOLDING RCB    INTO THE GENERIC FUNCTION WIHTIN THE TARGET_ABSTRACT FILE.  FOR LATER.


def extend_target_for_splice_left_right(
    target: TargetColor, n_extend: int, fill_color: cl.Color, auto_expand: str
) -> TargetColor:
    """
    Extends the input target by adding pixels either above or below based on the specified auto-expand option.

    Parameters
    ----------
    target : TargetColor
        The target to extend.
    n_extend : int
        The number of additional pixels to add.
    fill_color : cl.Color
        The color to fill the new pixels.
    auto_expand : str
        Specifies how to expand the target if the heights differ. Options are:
        - 'fill_top': Extend the top.
        - 'fill_bottom': Extend the bottom.
        - 'fill_even': Extend both top and bottom evenly.

    Returns
    -------
    TargetColor
        A new instance of TargetColor with the specified extension.

    Raises
    ------
    AssertionError
        If the auto-expand option is invalid or if both n_extend_top and n_extend_bottom are zero.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    if auto_expand == 'fill_top':
        new_target = extend_target_top(target, n_extend, fill_color, new_target_name=target.pattern_description)
        return new_target
    elif auto_expand == "fill_bottom":
        new_target = extend_target_bottom(target, n_extend, fill_color, new_target_name=target.pattern_description)
        return new_target
    elif auto_expand == "fill_even":
        n_extend_top = int(n_extend / 2)
        n_extend_bottom = n_extend - n_extend_top
        if (n_extend_top == 0) and (n_extend_bottom == 0):
            print(
                "ERROR: In extend_target_for_splice_left_right(), encountered unexpected situation where n_extend_top and n_extend_bottom are both zero."
            )
            assert False  # ?? SCAFFOLDING -- CONVERT TO EXCEPTION.
        if n_extend_top > 0:
            new_target_top = extend_target_top(
                target, n_extend_top, fill_color, new_target_name=target.pattern_description
            )
        else:
            new_target_top = target
        if n_extend_bottom > 0:
            new_target_top_bottom = extend_target_bottom(
                new_target_top, n_extend_bottom, fill_color, new_target_name=target.pattern_description
            )
        else:
            new_target_top_bottom = new_target_top
        return new_target_top_bottom
    else:
        print(
            'ERROR: In extend_target_for_splice_left_right(), encountered unexpected auto_expand = "'
            + str(auto_expand)
            + '".'
        )
        assert False  # ?? SCAFFOLDING -- CONVERT TO EXCEPTION.


# ?? SCAFFOLDING RCB -- THE ABOVE/BELOW AND LEFT/RIGHT ROUTINES HAVE A LOT OF CODE COPYING.  DOES IT MAKE SENSE TO MERGE THEM?

# ?? SCAFFOLDING RCB -- THIS SHOULD BE WITHIN TARGET_ABSTRACT.  HOWEVER, WE NEED TO FIGURE OUT HOW TO MANAGE THE NEW TARGET CONSTRUCTOR,
# ?? SCAFFOLDING RCB    SO THAT IT IS A DERIVED CLASS, NOT BASE CLASS.  MAYBE WE NEED TO CREATE A "SHELL" TARGET_COLOR, AND THEN PASS IT
# ?? SCAFFOLDING RCB    INTO THE GENERIC FUNCTION WIHTIN THE TARGET_ABSTRACT FILE.  FOR LATER.

# ?? SCAFFOLDING RCB -- SHOULD THIS BE DONE BY COMBINING IMAGES WITH AN IMAGE LIBRARY, OR BY ARRAY COMBINATION IN NUMPY?


def splice_targets_left_right(
    left_target: TargetColor,  # Target to place at left of new target.
    right_target: TargetColor,  # Target to place at right of new target.
    gap: int,  # Pixels.  Gap to leave between targets.
    initial_color: cl.Color,  # Color to fill canvas before adding patterns.
    auto_expand: str = "fill_even",  # Whether to expand smaller target if sizes don't match.
    # Values: None, 'fill_top', 'fill_bottom', 'fill_even'.
    new_target_name: str = None,  # If none, new target name will combine left/right names.
) -> TargetColor:
    """
    Constructs a new target by placing one target to the left of another with a specified gap.

    Parameters
    ----------
    left_target : TargetColor
        The target to place on the left side of the new target.
    right_target : TargetColor
        The target to place on the right side of the new target.
    gap : int
        The number of pixels to leave as a gap between the two targets.
    initial_color : cl.Color
        The color to fill the canvas before adding patterns.
    auto_expand : str, optional
        Specifies how to expand the smaller target if sizes don't match. Options are:
        - None: No expansion.
        - 'fill_top': Extend the top of the smaller target.
        - 'fill_bottom': Extend the bottom of the smaller target.
        - 'fill_even': Extend both top and bottom evenly. Default is 'fill_even'.
    new_target_name : str, optional
        If provided, this will be the name of the new target; otherwise, the name will be derived from the input targets.

    Returns
    -------
    TargetColor
        A new instance of TargetColor with the left and right targets spliced together.

    Raises
    ------
    AssertionError
        If the targets have different heights or bands, or if the auto-expand option is invalid.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Fetch sizes.
    left_n_rows, left_n_cols, left_n_bands = left_target.rows_cols_bands()
    right_n_rows, right_n_cols, right_n_bands = right_target.rows_cols_bands()
    left_dpm = (
        left_target.dpm
    )  # ?? SCAFFOLDING RCB -- SHOULD THESE BE ACCESSOR FUNCTIONS?  WHAT IS OPENCSP POLICY/PATTERN ON THIS?
    right_dpm = (
        right_target.dpm
    )  # ?? SCAFFOLDING RCB -- SHOULD THESE BE ACCESSOR FUNCTIONS?  WHAT IS OPENCSP POLICY/PATTERN ON THIS?

    # Auto-expand if needed.
    if (left_n_rows != right_n_rows) and (auto_expand != None):
        if left_n_rows < right_n_rows:
            n_extend = right_n_rows - left_n_rows
            left_target = extend_target_for_splice_left_right(left_target, n_extend, initial_color, auto_expand)
            left_n_rows, left_n_cols, left_n_bands = left_target.rows_cols_bands()
        elif left_n_rows > right_n_rows:
            n_extend = left_n_rows - right_n_rows
            right_target = extend_target_for_splice_left_right(right_target, n_extend, initial_color, auto_expand)
            right_n_rows, right_n_cols, right_n_bands = right_target.rows_cols_bands()
        else:
            print("ERROR: In splice_targets_left_right(), unexpected situation encountered.")
            assert False  # ?? SCAFFOLDING -- CONVERT TO EXCEPTION.

    # Check input.
    if left_n_rows != right_n_rows:
        print(
            "ERROR: In splice_targets_left_right(), unequal left_n_rows="
            + str(left_n_rows)
            + " and right_n_rows="
            + str(right_n_rows)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if left_n_bands != right_n_bands:
        print(
            "ERROR: In splice_targets_left_right(), unequal left_n_bands="
            + str(left_n_bands)
            + " and right_n_bands="
            + str(right_n_bands)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if left_n_bands != 3:
        print("ERROR: In splice_targets_left_right(), left_n_bands=" + str(left_n_bands) + " is not equal to 3.")
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if left_dpm != right_dpm:
        print(
            "ERROR: In splice_targets_left_right(), unequal left_dpm="
            + str(left_dpm)
            + " and right_dpm="
            + str(right_dpm)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION

    # Compute new size.
    new_n_rows = left_n_rows
    new_n_cols = left_n_cols + gap + right_n_cols
    new_n_bands = left_n_bands

    # Construct new target.
    new_image_width = new_n_cols / left_dpm
    new_image_height = left_target.image_height
    new_image_dpm = left_dpm
    new_target = TargetColor(new_image_width, new_image_height, new_image_dpm, initial_color)

    # Check code consistency.
    check_n_rows, check_n_cols, check_n_bands = new_target.rows_cols_bands()
    if new_n_rows != check_n_rows:
        print(
            "ERROR: In splice_targets_left_right(), unequal new_n_rows="
            + str(new_n_rows)
            + " and check_n_rows="
            + str(check_n_rows)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_cols != check_n_cols:
        print(
            "ERROR: In splice_targets_left_right(), unequal new_n_cols="
            + str(new_n_cols)
            + " and check_n_cols="
            + str(check_n_cols)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_bands != check_n_bands:
        print(
            "ERROR: In splice_targets_left_right(), unequal new_n_bands="
            + str(new_n_bands)
            + " and check_n_bands="
            + str(check_n_bands)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION

    # Copy left image.
    left_image = left_target.image
    new_image = new_target.image
    for row in range(new_n_rows):
        for col in range(left_n_cols):
            for band in range(new_n_bands):
                new_image[row][col][band] = left_image[row][col][band]

    # Copy right image.
    right_image = right_target.image
    new_image = new_target.image
    start_col = left_n_cols + gap
    for row in range(new_n_rows):
        for col in range(right_n_cols):
            for band in range(new_n_bands):
                new_image[row][start_col + col][band] = right_image[row][col][band]

    # Set description.
    if new_target_name == None:
        new_target.set_pattern_description(
            left_target.pattern_description + "__left__" + right_target.pattern_description
        )
    else:
        new_target.set_pattern_description(new_target_name)

    # Return.
    return new_target


# ?? SCAFFOLDING RCB -- THIS SHOULD BE WITHIN TARGET_ABSTRACT.  HOWEVER, WE NEED TO FIGURE OUT HOW TO MANAGE THE NEW TARGET CONSTRUCTOR,
# ?? SCAFFOLDING RCB    SO THAT IT IS A DERIVED CLASS, NOT BASE CLASS.  MAYBE WE NEED TO CREATE A "SHELL" TARGET_COLOR, AND THEN PASS IT
# ?? SCAFFOLDING RCB    INTO THE GENERIC FUNCTION WIHTIN THE TARGET_ABSTRACT FILE.  FOR LATER.

# ?? SCAFFOLDING RCB -- SHOULD THIS BE DONE BY COMBINING IMAGES WITH AN IMAGE LIBRARY, OR BY ARRAY COMBINATION IN NUMPY?


def splice_targets_above_below(
    above_target: TargetColor,  # Target to place at top of new target.
    below_target: TargetColor,  # Target to place at bottom of new target.
    gap: int,  # Pixels.  Gap to leave between targets.
    initial_color: cl.Color,  # Color to fill canvas before adding patterns.
    new_target_name: str = None,  # If none, new target name will combine above/below names.
) -> TargetColor:
    """
    Constructs a new target by placing one target above another with a specified gap.

    Parameters
    ----------
    above_target : TargetColor
        The target to place on the top side of the new target.
    below_target : TargetColor
        The target to place on the bottom side of the new target.
    gap : int
        The number of pixels to leave as a gap between the two targets.
    initial_color : cl.Color
        The color to fill the canvas before adding patterns.
    new_target_name : str, optional
        If provided, this will be the name of the new target; otherwise, the name will be derived from the input targets.

    Returns
    -------
    TargetColor
        A new instance of TargetColor with the above and below targets spliced together.

    Raises
    ------
    AssertionError
        If the targets have different widths or bands.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Fetch sizes.
    above_n_rows, above_n_cols, above_n_bands = above_target.rows_cols_bands()
    below_n_rows, below_n_cols, below_n_bands = below_target.rows_cols_bands()
    above_dpm = (
        above_target.dpm
    )  # ?? SCAFFOLDING RCB -- SHOULD THESE BE ACCESSOR FUNCTIONS?  WHAT IS OPENCSP POLICY/PATTERN ON THIS?
    below_dpm = (
        below_target.dpm
    )  # ?? SCAFFOLDING RCB -- SHOULD THESE BE ACCESSOR FUNCTIONS?  WHAT IS OPENCSP POLICY/PATTERN ON THIS?

    # Check input.
    if above_n_cols != below_n_cols:  # ?? SCAFFOLDING RCB -- EXTEND ROUTINE TO ALLOW UNEQUAL COLUMNS?
        print(
            "ERROR: In splice_targets_above_below(), unequal above_n_cols="
            + str(above_n_cols)
            + " and below_n_cols="
            + str(below_n_cols)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if above_n_bands != below_n_bands:
        print(
            "ERROR: In splice_targets_above_below(), unequal above_n_bands="
            + str(above_n_bands)
            + " and below_n_bands="
            + str(below_n_bands)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if above_n_bands != 3:
        print("ERROR: In splice_targets_above_below(), above_n_bands=" + str(above_n_bands) + " is not equal to 3.")
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if above_dpm != below_dpm:
        print(
            "ERROR: In splice_targets_above_below(), unequal above_dpm="
            + str(above_dpm)
            + " and below_dpm="
            + str(below_dpm)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION

    # Compute new size.
    new_n_rows = above_n_rows + gap + below_n_rows
    new_n_cols = above_n_cols
    new_n_bands = above_n_bands

    # Construct new target.
    new_image_width = above_target.image_width
    new_image_height = new_n_rows / above_dpm
    new_image_dpm = above_dpm
    new_target = TargetColor(new_image_width, new_image_height, new_image_dpm, initial_color)

    # Check code consistency.
    check_n_rows, check_n_cols, check_n_bands = new_target.rows_cols_bands()
    if new_n_rows != check_n_rows:
        print(
            "ERROR: In splice_targets_above_below(), unequal new_n_rows="
            + str(new_n_rows)
            + " and check_n_rows="
            + str(check_n_rows)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_cols != check_n_cols:
        print(
            "ERROR: In splice_targets_above_below(), unequal new_n_cols="
            + str(new_n_cols)
            + " and check_n_cols="
            + str(check_n_cols)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION
    if new_n_bands != check_n_bands:
        print(
            "ERROR: In splice_targets_above_below(), unequal new_n_bands="
            + str(new_n_bands)
            + " and check_n_bands="
            + str(check_n_bands)
        )
        assert False  # ?? SCAFFOLDING RCB -- CONVERT TO EXCEPTION

    # Copy above image.
    above_image = above_target.image
    new_image = new_target.image
    for row in range(above_n_rows):
        for col in range(new_n_cols):
            for band in range(new_n_bands):
                new_image[row][col][band] = above_image[row][col][band]

    # Copy below image.
    below_image = below_target.image
    new_image = new_target.image
    start_row = above_n_rows + gap
    for row in range(below_n_rows):
        for col in range(new_n_cols):
            for band in range(new_n_bands):
                new_image[start_row + row][col][band] = below_image[row][col][band]

    # Set description.
    if new_target_name == None:
        new_target.set_pattern_description(
            above_target.pattern_description + "__above__" + below_target.pattern_description
        )
    else:
        new_target.set_pattern_description(new_target_name)

    # Return.
    return new_target


# ?? SCAFFOLDING RCB -- ADD TOOL TIPS TO THIS ROUTINE.
def construct_stacked_linear_color_bar(
    n_stack: int,
    color_bar_width,
    color_total_height,
    composite_dpm,
    color_below_min,
    color_bar,
    color_bar_name,
    color_above_max,
    x_or_y,
    discrete_or_continuous_list,  # List of 'discrete' or 'continuous' strings indicating whether to interpolate colors for each stack entry.  # ?? SCAFFOLDING RCB -- DOCUMENT BETTER
    saturation_spec_list,  # ?? SCAFFOLDING RCB -- DOCUMENT THIS.
    gap_color,
) -> tc.TargetColor:
    """
    Constructs a stacked linear color bar composed of multiple stacked color bars.

    Parameters
    ----------
    n_stack : int
        The number of color bars to stack vertically or horizontally.
    color_bar_width : float
        The width of each stacked color bar in meters.
    color_total_height : float
        The total height of the stacked color bars in meters.
    composite_dpm : float
        The dots per meter for the image resolution.
    color_below_min : cl.Color
        The color to use below the minimum end of the color bar.
    color_bar : list
        A list of colors representing the color sequence for the color bar.
    color_bar_name : str
        A terse descriptive name of the color sequence for output purposes.
    color_above_max : cl.Color
        The color to use above the maximum end of the color bar.
    x_or_y : str
        The direction of the color bar stacking ('x' for horizontal, 'y' for vertical).
    discrete_or_continuous_list : list
        A list of strings indicating whether to interpolate colors for each stack entry ('discrete' or 'continuous').
    saturation_spec_list : list
        A list of specifications for saturation adjustments for each stack entry.
    gap_color : cl.Color
        The color to fill the gaps between stacked color bars.

    Returns
    -------
    tc.TargetColor
        A new instance of TargetColor representing the stacked linear color bar.

    Raises
    ------
    AssertionError
        If the number of stacks is less than 1 or if the lengths of the lists do not match the number of stacks.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Check input.
    if n_stack < 1:
        print("ERROR: In stack_linear_color_bar(), encountered non-positive n_stack = " + str(n_stack))
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION
    if n_stack != len(discrete_or_continuous_list):
        print(
            "ERROR: In stack_linear_color_bar(), encountered mismatched n_stack="
            + str(n_stack)
            + " and len(discrete_or_continuous_list)="
            + len(discrete_or_continuous_list)
        )
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION
    if n_stack != len(saturation_spec_list):
        print(
            "ERROR: In stack_linear_color_bar(), encountered mismatched n_stack="
            + str(n_stack)
            + " and len(saturation_spec_list)="
            + len(saturation_spec_list)
        )
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION

    # Construct final color bar name.
    target_name = color_bar_name + "_" + str(n_stack) + "x"

    # First saturation spec.
    first_saturation_spec = saturation_spec_list[0]
    first_lateral_gradient_type = first_saturation_spec[0]
    first_saturated_to_white_exponent = first_saturation_spec[1]
    first_light_to_saturated_min = first_saturation_spec[2]
    first_light_to_saturated_max = first_saturation_spec[3]

    # Construct needed targets.
    single_bar_target_discrete = tc.construct_target_linear_color_bar(
        color_bar_width,
        color_total_height / n_stack,
        composite_dpm,
        color_below_min,
        color_bar,
        color_bar_name,
        color_above_max,
        x_or_y,
        "discrete",
        lateral_gradient_type=first_lateral_gradient_type,
        saturated_to_white_exponent=first_saturated_to_white_exponent,
        light_to_saturated_min=first_light_to_saturated_min,
        light_to_saturated_max=first_light_to_saturated_max,
    )

    single_bar_target_continuous = tc.construct_target_linear_color_bar(
        color_bar_width,
        color_total_height / n_stack,
        composite_dpm,
        color_below_min,
        color_bar,
        color_bar_name,
        color_above_max,
        x_or_y,
        "continuous",
        lateral_gradient_type=first_lateral_gradient_type,
        saturated_to_white_exponent=first_saturated_to_white_exponent,
        light_to_saturated_min=first_light_to_saturated_min,
        light_to_saturated_max=first_light_to_saturated_max,
    )

    # Initial target.
    if discrete_or_continuous_list[0] == "discrete":
        single_bar_target = single_bar_target_discrete
    elif discrete_or_continuous_list[0] == "continuous":
        single_bar_target = single_bar_target_continuous
    else:
        print(
            'ERROR: In construct_stacked_linear_color_bar(1), encountered unexpected discrete_or_continuous_list[0] = "'
            + str(discrete_or_continuous_list[0])
            + '".'
        )
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION

    # Add color bars.
    if n_stack == 1:
        return single_bar_target
    else:
        # n_stack > 1.
        stacked_target = single_bar_target
        gap = 0
        for idx in range(1, n_stack):
            # This saturation spec.
            this_saturation_spec = saturation_spec_list[idx]
            this_lateral_gradient_type = this_saturation_spec[0]
            this_saturated_to_white_exponent = this_saturation_spec[1]
            this_light_to_saturated_min = this_saturation_spec[2]
            this_light_to_saturated_max = this_saturation_spec[3]
            if discrete_or_continuous_list[idx] == "discrete":
                this_bar_target_discrete = tc.construct_target_linear_color_bar(
                    color_bar_width,
                    color_total_height / n_stack,
                    composite_dpm,
                    color_below_min,
                    color_bar,
                    color_bar_name,
                    color_above_max,
                    x_or_y,
                    "discrete",
                    lateral_gradient_type=this_lateral_gradient_type,
                    saturated_to_white_exponent=this_saturated_to_white_exponent,
                    light_to_saturated_min=this_light_to_saturated_min,
                    light_to_saturated_max=this_light_to_saturated_max,
                )
                stacked_target = tc.splice_targets_above_below(
                    this_bar_target_discrete, stacked_target, gap, initial_color=gap_color, new_target_name=target_name
                )
            elif discrete_or_continuous_list[idx] == "continuous":
                this_bar_target_continuous = tc.construct_target_linear_color_bar(
                    color_bar_width,
                    color_total_height / n_stack,
                    composite_dpm,
                    color_below_min,
                    color_bar,
                    color_bar_name,
                    color_above_max,
                    x_or_y,
                    "continuous",
                    lateral_gradient_type=this_lateral_gradient_type,
                    saturated_to_white_exponent=this_saturated_to_white_exponent,
                    light_to_saturated_min=this_light_to_saturated_min,
                    light_to_saturated_max=this_light_to_saturated_max,
                )
                stacked_target = tc.splice_targets_above_below(
                    this_bar_target_continuous,
                    stacked_target,
                    gap,
                    initial_color=gap_color,
                    new_target_name=target_name,
                )
            else:
                print(
                    "ERROR: In construct_stacked_linear_color_bar(2), encountered unexpected discrete_or_continuous_list["
                    + str(idx)
                    + '] = "'
                    + str(discrete_or_continuous_list[idx])
                    + '".'
                )
                assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION
        return stacked_target


def stacked_color_bar_name(n_stack: int, color_bar_name: str) -> str:
    """
    Constructs a name for a stacked color bar based on the number of stacks and the base color bar name.

    Parameters
    ----------
    n_stack : int
        The number of stacked color bars.
    color_bar_name : str
        The base name of the color bar.

    Returns
    -------
    str
        The constructed name for the stacked color bar.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return color_bar_name + '_stacked' + str(n_stack) + 'x'


# ?? SCAFFOLDING RCB -- ADD TOOL TIPS TO THIS ROUTINE.
def construct_linear_color_bar_cascade(  # Dimensions.
    color_bar_width,  # meters.      Width of each stacked color bar.
    color_total_height,  # meters.      Height of each stacked color bar, and also grey bars.
    composite_dpm,  # dots/meter.  Pixel resolution.
    # Main color bar.
    color_below_min,  # Color.       Color to use below minimum end of color bar.
    color_bar,  # Color sequence for color bar.
    color_bar_name,  # Terse descriptive name of color sequence, for output filename.
    color_above_max,  # Color.       Color to use above maximum end of color bar.
    # Reference color bar.
    ref_color_below_min,  # Color.       Color to use below minimum end of reference color bar.
    ref_color_bar,  # Color sequence for reference color bar.
    ref_color_bar_name,  # Terse descriptive name of reference color sequence, for output filename.
    ref_color_above_max,  # Color.       Color to use above maximum end of reference color bar.
    # Direction.
    x_or_y,  # ?? SCAFFOLDING RCB -- DOESN'T CHANGE STACK OR CASCADE DIRECTION WRT X_OR_Y PARAMETER.
    # Color stack specification.
    stack_sequence,  # List of stack heights desired.  Example:  [1, 2, 4, 6, 8, 10]
    # Corresponding list of lists of 'discrete' or 'continuous' strings indicating whether to interpolate colors for each stack entry.  # ?? SCAFFOLDING RCB -- DOCUMENT BETTER
    list_of_discrete_or_continuous_lists,
    list_of_saturation_spec_lists,  # ?? SCAFFOLDING RCB -- DOCUMENT THIS
    # Grey context bar specification.
    # Boolean.  Whether to include a discrete grey scale for disambiguation on either side of each stack.
    include_grey_neighbors,
    grey_bar_width: float,  # meters.  Width of grey context bars, if included.
    # Space between bars.
    gap_between_bars_pix: int = 10,  # pixels  # ?? SCAFFOLDING RCB -- SHOULD THIS BE IN INCHES?
    ref_gap_pix: int = 10,  # pixels  # ?? SCAFFOLDING RCB -- SHOULD THIS BE IN INCHES?
    # Color for both gap between color bar stacks, and also spacing required at end of stack if needed.
    gap_color=cl.white(),
) -> tc.TargetColor:
    """
    Constructs a cascade of linear color bars, including reference and main color bars.

    Parameters
    ----------
    color_bar_width : float
        The width of each stacked color bar in meters.
    color_total_height : float
        The total height of the stacked color bars in meters.
    composite_dpm : float
        The dots per meter for the image resolution.
    color_below_min : cl.Color
        The color to use below the minimum end of the color bar.
    color_bar : list
        A list of colors representing the color sequence for the color bar.
    color_bar_name : str
        A terse descriptive name of the color sequence for output purposes.
    color_above_max : cl.Color
        The color to use above the maximum end of the color bar.
    ref_color_below_min : cl.Color
        The color to use below the minimum end of the reference color bar.
    ref_color_bar : list
        A list of colors representing the color sequence for the reference color bar.
    ref_color_bar_name : str
        A terse descriptive name of the reference color sequence for output purposes.
    ref_color_above_max : cl.Color
        The color to use above the maximum end of the reference color bar.
    x_or_y : str
        The direction of the color bar stacking ('x' for horizontal, 'y' for vertical).
    stack_sequence : list
        A list of stack heights desired for the main color bars.
    list_of_discrete_or_continuous_lists : list
        A list of lists of strings indicating whether to interpolate colors for each stack entry ('discrete' or 'continuous').
    list_of_saturation_spec_lists : list
        A list of specifications for saturation adjustments for each stack entry.
    include_grey_neighbors : bool
        Whether to include a discrete grey scale for disambiguation on either side of each stack.
    grey_bar_width : float
        The width of grey context bars in meters, if included.
    gap_between_bars_pix : int, optional
        The number of pixels to leave as a gap between color bar stacks. Default is 10.
    ref_gap_pix : int, optional
        The number of pixels to leave as a gap for the reference color bar. Default is 10.
    gap_color : cl.Color, optional
        The color to fill the gaps between color bar stacks. Default is white.

    Returns
    -------
    tc.TargetColor
        A new instance of TargetColor representing the cascade of linear color bars.

    Raises
    ------
    AssertionError
        If the stack sequence is empty or if the lengths of the lists do not match the number of stacks.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Check input.
    if len(stack_sequence) == 0:
        print("ERROR: In construct_linear_color_bar_cascade(), encountered len(stack_sequence) == 0.")
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION
    if len(list_of_discrete_or_continuous_lists) != len(stack_sequence):
        print(
            "ERROR: In construct_linear_color_bar_cascade(), mismatched len(stack_sequence)="
            + str(len(stack_sequence))
            + " and len(list_of_discrete_or_continuous_lists)="
            + str(len(list_of_discrete_or_continuous_lists))
        )
        assert False  # ?? SCAFFOLDING RCB -- USE EXCEPTION

    # Construct target name.
    cascade_target_name = color_bar_name + "_cascade"

    # Reference linear color bar.
    print("In construct_linear_color_bar_cascade(), generating reference linear bar...")
    ref_target = tc.construct_target_linear_color_bar(
        color_bar_width,
        color_total_height,
        composite_dpm,
        ref_color_below_min,
        ref_color_bar,
        ref_color_bar_name,
        ref_color_above_max,
        x_or_y,
        "discrete",
    )
    cascade_target = ref_target
    cascade_target.set_pattern_description = tc.stacked_color_bar_name(1, cascade_target_name)

    # Adjacent main linear color bar.
    print('In construct_linear_color_bar_cascade(), generating "' + color_bar_name + '" linear bar...')
    main_color_target = tc.construct_target_linear_color_bar(
        color_bar_width,
        color_total_height,
        composite_dpm,
        color_below_min,
        color_bar,
        color_bar_name,
        color_above_max,
        x_or_y,
        "discrete",
    )
    cascade_target = tc.splice_targets_left_right(
        cascade_target, main_color_target, gap=ref_gap_pix, initial_color=gap_color, new_target_name=cascade_target_name
    )

    # Generate cascade.
    for n_bars_in_stack, discrete_or_continuous_list, saturation_spec_list in zip(
        stack_sequence, list_of_discrete_or_continuous_lists, list_of_saturation_spec_lists
    ):
        # Status update.
        print("In construct_linear_color_bar_cascade(), generating stacked bar " + str(n_bars_in_stack) + "...")

        # Generate color bar stack and its neighbors.
        stacked_color_target = tc.construct_stacked_linear_color_bar(
            n_bars_in_stack,
            color_bar_width,
            color_total_height,
            composite_dpm,
            color_below_min,
            color_bar,
            color_bar_name,
            color_above_max,
            x_or_y,
            discrete_or_continuous_list,
            saturation_spec_list,
            gap_color,
        )
        # Add grey-scale neighbors, if desired.
        if include_grey_neighbors:
            grey_step = 1.0 / n_bars_in_stack
            grey_scale_list = []
            for grey_idx in range(0, n_bars_in_stack):
                grey_level = 1.0 - (grey_step * grey_idx)
                grey_scale_list.append(grey_level)
            grey_scale_list.append(grey_level)  # Repeat initial value.
            grey_rgb_list = [
                (255 * x, 255 * x, 255 * x) for x in grey_scale_list
            ]  # ?? SCAFFOLDING RCB -- IF COLOR_BAR BECOMES A CLASS, UPDATE THIS.
            grey_below_min = cl.black()
            grey_above_max = cl.white()
            grey_bar_name = "grey_" + str(len(grey_scale_list))
            grey_target = tc.construct_target_linear_color_bar(
                grey_bar_width,
                color_total_height,
                composite_dpm,
                grey_below_min,
                grey_rgb_list,
                grey_bar_name,
                grey_above_max,
                x_or_y,
                "discrete",
            )  # Always discrete.
            stacked_target = tc.splice_targets_left_right(
                grey_target, stacked_color_target, gap=0, initial_color=gap_color, new_target_name=cascade_target_name
            )
            stacked_target = tc.splice_targets_left_right(
                stacked_target, grey_target, gap=0, initial_color=gap_color, new_target_name=cascade_target_name
            )
        else:
            stacked_target = stacked_color_target

        # Update cascade.
        cascade_target = tc.splice_targets_left_right(
            cascade_target, stacked_target, gap_between_bars_pix, gap_color, new_target_name=cascade_target_name
        )

    # Return.
    return cascade_target
