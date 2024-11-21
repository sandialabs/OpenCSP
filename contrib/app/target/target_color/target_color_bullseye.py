"""
Demonstrate Solar Field Plotting Routines



"""

import matplotlib.pyplot as plt
import os

import opencsp.common.lib.target.target_color_convert
import opencsp.app.target.target_color.target_color_bullseye_error


#    # Bullseye color bar (Gen 1 version)
#    def bullseye_color_bar_gen_1(discrete_or_continuous):
#        for row in range(0,n_rows):
#            for col in range(0,n_cols):
#                x = col
#                y = n_rows - row
#                cx = x_max/2
#                cy = y_max/2
#                dx = x-cx
#                dy = y-cy
#                r_pixel = math.sqrt((dx*dx) + (dy*dy))
#                theta = math.atan2(dy,dx)
#                # Lookup color bar entry.
#                r_meter = meters_given_pixels(r_pixel)
#                r_mrad = surface_normal_error_magnitude_given_radius_in_meters(r_meter)
#                color = color_given_value(r_mrad, max_mrad, color_bar, discrete_or_continuous)
#                # Set pixel color
#                img[row,col,0] = color[0]/255.0
#                img[row,col,1] = color[1]/255.0
#                img[row,col,2] = color[2]/255.0
#        output_file_body = 'matplotlib_' + discrete_or_continuous + '_bullseye_color_bar'
#        output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#        print('Saving file:', output_file_dir_body_ext)
#        plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#        print('Done.')


# Bullseye color bar


def bullseye_color_bar(
    discrete_or_continuous,
    cx_offset_pix,
    y_offset_pix,
    focal_length_meter,
    r_min_mrad,
    r_max_mrad,
    color_below_min,
    color_above_max,
    color_frame,
    frame_width_pix,
    draw_alignment_line,
    cx_offset_alignment_pix,
    draw_trim_line,
    cx_offset_trim_pix,
    target_img_width_pix,
):
    """
    Placeholder
    """
    for row in range(0, n_rows):
        for col in range(0, n_cols):
            r_mrad = bullseye_error.radius_in_mrad_given_row_col(
                row, col, x_max, cx_offset_pix, y_offset_pix, focal_length_meter
            )
            color = color_convert.color_given_value(
                r_mrad, r_min_mrad, r_max_mrad, color_bar, discrete_or_continuous, color_below_min, color_above_max
            )
            # Set pixel color
            img[row, col, 0] = color[0] / 255.0
            img[row, col, 1] = color[1] / 255.0
            img[row, col, 2] = color[2] / 255.0
    # Frame.
    for row in range(0, n_rows):
        for col in range(0, frame_width_pix):
            # Set pixel color
            img[row, col, 0] = color_frame[0] / 255.0
            img[row, col, 1] = color_frame[1] / 255.0
            img[row, col, 2] = color_frame[2] / 255.0
        for col in range(n_cols - frame_width_pix, n_cols):
            # Set pixel color
            img[row, col, 0] = color_frame[0] / 255.0
            img[row, col, 1] = color_frame[1] / 255.0
            img[row, col, 2] = color_frame[2] / 255.0
    for row in range(0, frame_width_pix):
        for col in range(0, n_cols):
            # Set pixel color
            img[row, col, 0] = color_frame[0] / 255.0
            img[row, col, 1] = color_frame[1] / 255.0
            img[row, col, 2] = color_frame[2] / 255.0
    for row in range(n_rows - frame_width_pix, n_rows):
        for col in range(0, n_cols):
            # Set pixel color
            img[row, col, 0] = color_frame[0] / 255.0
            img[row, col, 1] = color_frame[1] / 255.0
            img[row, col, 2] = color_frame[2] / 255.0
    # Alignment line.
    if draw_alignment_line:
        cx = (x_max / 2) + cx_offset_pix
        this_hole_center_x = round(cx)  # x position of target center, in this image.
        other_hole_center_x = round(cx_offset_alignment_pix)  # x position of target center, in other image.
        alignment_line_start_x = this_hole_center_x - (other_hole_center_x + round(x_max / 2))
        for row in range(0, n_rows):
            for col in range(alignment_line_start_x, (alignment_line_start_x + frame_width_pix)):
                # Don't draw within the target.
                r_mrad = bullseye_error.radius_in_mrad_given_row_col(
                    row, col, x_max, cx_offset_pix, y_offset_pix, focal_length_meter
                )
                if (r_mrad > r_max_mrad) and (0 <= col) and (col < n_cols):
                    # Set pixel color
                    img[row, col, 0] = color_frame[0] / 255.0
                    img[row, col, 1] = color_frame[1] / 255.0
                    img[row, col, 2] = color_frame[2] / 255.0
        print('alignment_line_start_x = ', alignment_line_start_x)
    else:
        alignment_line_start_x = -999
    # Trim line.
    if draw_trim_line:
        cx = (x_max / 2) + cx_offset_pix
        target_edge_x = cx + (target_img_width_pix / 2)
        other_cx = (x_max / 2) + cx_offset_trim_pix
        other_edge_x = other_cx - (target_img_width_pix / 2)
        other_margin_x = other_edge_x - 0
        trim_line_start_x = round(target_edge_x + other_margin_x)
        for row in range(0, n_rows):
            for col in range(trim_line_start_x, (trim_line_start_x + frame_width_pix)):
                # Don't draw within the target.
                r_mrad = bullseye_error.radius_in_mrad_given_row_col(
                    row, col, x_max, cx_offset_pix, y_offset_pix, focal_length_meter
                )
                if (r_mrad > r_max_mrad) and (0 <= col) and (col < n_cols):
                    # Set pixel color
                    img[row, col, 0] = color_frame[0] / 255.0
                    img[row, col, 1] = color_frame[1] / 255.0
                    img[row, col, 2] = color_frame[2] / 255.0
        print('trim_line_start_x = ', trim_line_start_x)
    else:
        trim_line_start_x = -999
    # Save.
    output_file_body = 'matplotlib_' + discrete_or_continuous + '_bullseye_color_bar' + '_cx' + str(cx_offset_pix)
    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
    print('Saving file:', output_file_dir_body_ext)
    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
    print('Done.')
    # Return
    return alignment_line_start_x, trim_line_start_x
