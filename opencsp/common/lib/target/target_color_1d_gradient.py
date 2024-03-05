"""
Demonstrate Solar Field Plotting Routines



"""

import matplotlib.pyplot as plt
import os

import opencsp.common.lib.target.target_color_convert as tcc
import opencsp.common.lib.target.target_image as ti


#    # Red spectrum
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            x = col
#            x_frac = (x/x_max)
#            img[row,col,0] = x_frac * max_intensity
#            img[row,col,1] = 0
#            img[row,col,2] = 0
#    output_file_body = 'red'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
#
#    # Green spectrum
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            y = n_rows - row
#            y_frac = (y/y_max)
#            img[row,col,0] = 0
#            img[row,col,1] = y_frac * max_intensity
#            img[row,col,2] = 0
#    output_file_body = 'green'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
#
#    # Green to red
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            x = col
#            x_frac = (x/x_max)
#            img[row,col,0] = x_frac * max_intensity
#            img[row,col,1] = (1-x_frac) * max_intensity
#            img[row,col,2] = 0
#    output_file_body = 'green_to_red'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
#
#    # Green to blue
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            y = n_rows - row
#            y_frac = (y/y_max)
#            img[row,col,0] = 0
#            img[row,col,1] = (1-y_frac) * max_intensity
#            img[row,col,2] = y_frac * max_intensity
#    output_file_body = 'green_to_blue'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
#
#    # Red to green to blue
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            x = col
#            y = n_rows - row
#            half_x = x_max/2
#            if x < half_x:
#                red_frac = 1 - (x/half_x)
#            else:
#                red_frac = 0
#            if x < half_x:
#                green_frac = x/half_x
#            else:
#                green_frac = 1 - ((x-half_x)/half_x)
#            if x > half_x:
#                blue_frac = (x-half_x)/half_x
#            else:
#                blue_frac = 0
#            y_frac = (y/y_max)
#            # img[row,col,0] = 0
#            # img[row,col,1] = 0
#            # img[row,col,2] = 0
#            img[row,col,0] = red_frac * max_intensity
#            img[row,col,1] = green_frac * max_intensity
#            img[row,col,2] = blue_frac * max_intensity
#    output_file_body = 'red_to_green_to_blue'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
#
#    # Red to green to blue to red
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            x = col
#            y = n_rows - row
#            third_x = x_max/3
#            # Red
#            if x < third_x:
#                red_frac = 1 - (x/third_x)
#            elif x < (2 * third_x):
#                red_frac = 0
#            else:
#                red_frac = (x-(2*third_x))/third_x
#            # Green
#            if x < third_x:
#                green_frac = x/third_x
#            elif x < (2 * third_x):
#                green_frac = 1 - ((x-third_x)/third_x)
#            else:
#                green_frac = 0
#            # Blue
#            if x < third_x:
#                blue_frac = 0
#            elif x < (2*third_x):
#                blue_frac = (x-third_x)/third_x
#            else:
#                blue_frac = 1 - ((x-(2*third_x))/third_x)
#            # Set pixel color
#            img[row,col,0] = red_frac * max_intensity
#            img[row,col,1] = green_frac * max_intensity
#            img[row,col,2] = blue_frac * max_intensity
#    output_file_body = 'red_to_green_to_blue_to_red'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
#
