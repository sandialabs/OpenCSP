"""
Demonstrate Solar Field Plotting Routines



"""

import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.common.lib.target.target_color_convert
import opencsp.common.lib.target.target_color_1d_gradient
import opencsp.common.lib.target.target_image


if __name__ == "__main__":
    plt.close('all')

    base_dir = opencsp_dir() + '\\common\\lib\\test\\output\\TestTargetColor\\actual_output'
    output_dir = os.path.join(
        'app', 'target', 'target_color', 'test', 'data', 'output', source_file_body
    )  # ?? SCAFFOLDING RCB -- ADD CODE TO CREATE DIRECTORY IF NECESSARY.
    output_ext = '.png'

    print("Hello, world!")

    # Define image size and resolution.
    dpi = 30
    #    dpi = 100
    # img_height_in = 89.0  # Plascor boards are 90" long; assume a half-inch margin.
    # img_width_in  = 60.0  # Printer is 60" wide
    img_height_in = 10.0  # Small test.
    img_width_in = 7.0  # Small test.

    # Construct image object.
    img = target_image.construct_target_image(dpi, img_height_in, img_width_in)

    # Fetch image dimensions.
    n_rows, n_cols = target_image.rows_cols(img)

    # Size in (x,y) coordinates.
    x_max = n_cols
    y_max = n_rows

    # Report statistics.
    print('Min value = ', img.min())
    print('Max value = ', img.max())

    # Max intensity.
    max_intensity = 1.0

    # Matplotlib color bar
    color_bar = target_color_convert.matplotlib_color_bar()
    n_colors = len(color_bar)
    color_below_min = [0, 0, 0]  # Black below bottom of color bar.
    color_above_max = [255, 255, 255]  # White background for "saturated data."

    # DRAW IMAGES

    # Discrete linear color bar
    target_color_1d_gradient.linear_color_bar(
        'discrete', color_bar, color_below_min, color_above_max, img, dpi, output_dir, output_ext
    )

    # Continuous linear color bar
    target_color_1d_gradient.linear_color_bar(
        'continuous', color_bar, color_below_min, color_above_max, img, dpi, output_dir, output_ext
    )

    assert False

    # Discrete bullseye color bar
    # Target definition parameters.
    target_design_focal_length_m = (
        100  # meters.  Standard focal length for a multi-purpose target.  To measure a mirror with a different
    )
    #          focal length, adjust scale on color bar in final commmunication.
    target_design_err_max_mrad = 3.0  # mrad.  To match Braden's SOFAST plot, which has a color bar from 0 to 3 mrad.
    target_design_err_min_mrad = 0.005 * target_design_err_max_mrad  # Size of center circle for placing pilot drill.
    color_below_min = [255, 255, 255]  # White center circle for placing pilot drill.
    color_above_max = [255, 255, 255]  # White background for "saturated data."
    color_frame = [240, 240, 240]  # Overall image frame color
    frame_width_mm = 1.0  # millimeters.
    #    color_frame     = [255,  0,255]  # Overall image frame color
    #    frame_width_mm  = 10.0  # millimeters.
    frame_width_in = frame_width_mm / 25.4
    frame_width_pix = max(1, round(frame_width_in * dpi))

    # Image size calculation.
    target_design_err_max_rad = target_design_err_max_mrad / 1000.0  # radians
    target_diameter_m = (
        8.0 * target_design_focal_length_m * target_design_err_max_rad
    )  # Required target diameter is d=8*f*err_max.
    target_diameter_in = target_diameter_m * (1000 / 25.4)  # Convert to inches.
    # Image height and width determined by bullseye size.
    target_img_width_in = target_diameter_in  # Margin is controlled by the cx offset parameters.
    target_img_height_in = target_img_width_in  # For designing a square image
    print('target_img_width_in  = ', target_img_width_in)
    print('target_img_height_in = ', target_img_height_in)
    # Convert to pixels, for later use.
    target_img_width_pix = target_img_width_in * dpi
    target_img_height_pix = target_img_height_in * dpi
    # Clip width to printer limit.
    #    img_max_width_in = 1e6   # See full-size target.  Not printable.
    img_max_width_in = 60.0  # Printer maximum width.
    img_width_in = min(target_img_width_in, img_max_width_in)
    # Define board and top roller.
    #    board_height_in = 90.0  # inch.  Current plascore is cut to 90" length.
    board_height_in = 96.0  # inch.  Standard sheets are 96" long.
    board_half_height_in = board_height_in / 2.0
    top_roller_diameter_in = 0.75  # inch.  PVC tubing to give a smooth surface for the target to roll/slide over.
    top_roller_radius_in = top_roller_diameter_in / 2.0
    top_roller_half_arc_in = (top_roller_diameter_in * np.pi) / 2.0
    up_over_board_in = (
        board_half_height_in
        + top_roller_radius_in
        + top_roller_half_arc_in
        + top_roller_radius_in
        + board_half_height_in
    )  # Distance to go from board center up over the roller, back to board center.
    leader_up_over_board_in = up_over_board_in - (target_img_height_in / 2.0)
    # Add leader and trailer.
    spool_diameter_in = 3.5  # PVC tube outer diameter.
    num_turns_when_unfurled = 1.75  # Number of wraps still around the spool when the target is installed.
    wrap_distance_when_unfurled_in = num_turns_when_unfurled * (
        np.pi * spool_diameter_in
    )  # Length around spool when target installed.
    leader_margin_in = up_over_board_in - (
        target_img_height_in / 2.0
    )  # Leader required to go up over board and back to board center.
    trailer_margin_in = 6.0  # inch. Maximum distance from target bottom edge to spool when installed.
    leader_in = leader_margin_in + wrap_distance_when_unfurled_in
    trailer_in = trailer_margin_in + wrap_distance_when_unfurled_in
    img_height_in = target_img_height_in + leader_in + trailer_in
    y_offset_in = trailer_in + (target_img_height_in / 2)  # Defined from bottom of image.
    y_offset_pix = y_offset_in * dpi  #
    # Image size in pixels.
    img_rows = int(img_height_in * dpi)
    img_cols = int(img_width_in * dpi)

    # Image setup and size.
    img = np.zeros([img_rows, img_cols, 3])
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    n_bands = img.shape[2]
    x_max = n_cols
    y_max = n_rows

    # Define image offsets.
    cx_offset_1_pix = round(0.30 * x_max)  # Defined from center of image.  Controls total image width.
    cx_offset_2_pix = round(-0.55 * x_max)  # Defined from center of image.  Does not affect total image width.

    # Generate image #1.
    alignment_line_start_x, dummy = bullseye_color_bar(
        'discrete',
        cx_offset_1_pix,
        y_offset_pix,
        target_design_focal_length_m,
        target_design_err_min_mrad,
        target_design_err_max_mrad,
        color_below_min,
        color_above_max,
        color_frame,
        frame_width_pix,
        True,
        cx_offset_2_pix,
        False,
        -999,
        -999,
    )  # Dummy value, ignored if not printing trim line.

    # Generate image #2.
    dummy, trim_line_start_x = bullseye_color_bar(
        'discrete',
        cx_offset_2_pix,
        y_offset_pix,
        target_design_focal_length_m,
        target_design_err_min_mrad,
        target_design_err_max_mrad,
        color_below_min,
        color_above_max,
        color_frame,
        frame_width_pix,
        False,
        -999,  # Dummy value, ignored if not printing alignment line.
        True,
        cx_offset_1_pix,
        target_img_width_pix,
    )

    # Compute total width after assembly.
    total_width_x = alignment_line_start_x + trim_line_start_x
    total_width_x_in = total_width_x / dpi

    # Report.
    print('dpi = ', dpi)
    print('x_max = ', x_max)
    print('alignment_line_start_x = ', alignment_line_start_x)
    print('trim_line_start_x = ', trim_line_start_x)
    print('total_width_x = ', total_width_x)
    print('total_width_x_in = ', total_width_x_in)

    # Continuous bullseye color bar
#    bullseye_color_bar('continuous')

#    # Change to a square image
#    img_width = 60  # Printer is 60" wide
#    img_cols = int(img_width * dpi)
#    img_rows = int(img_width * dpi)  # Make a square image
#    img = np.zeros([img_rows,img_cols,3])
#    print('Image shape =', img.shape)
#    n_rows  = img.shape[0]
#    n_cols  = img.shape[1]
#    n_bands = img.shape[2]
#    # Size in (x,y) coordinates.
#    x_max = n_cols
#    y_max = n_rows
#
#    # Blue underlying red cross green (square)
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            x = col
#            y = n_rows - row
#            x_frac = (x/x_max)
#            y_frac = (y/y_max)
#            diagonal_frac = np.sqrt(x*x + y*y) / np.sqrt(x_max*x_max + y_max*y_max)
#            img[row,col,0] = x_frac * max_intensity
#            img[row,col,1] = y_frac * max_intensity
#            img[row,col,2] = (1-diagonal_frac) * max_intensity
#    output_file_body = 'blue_under_red_cross_green_square'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
