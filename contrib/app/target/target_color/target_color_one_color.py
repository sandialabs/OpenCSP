"""
Demonstrate Solar Field Plotting Routines



"""

import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.app.target.target_color.target_color_bullseye_error
import opencsp.common.lib.target.target_color_convert


#    # Set to black
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            x = col
#            y = n_rows - row
#            img[row,col,0] = 0
#            img[row,col,1] = 0
#            img[row,col,2] = 0
#            if ( ((190 < x) and (x <= 210)) and
#                 (( 90 < y) and (y <= 110)) ):
#                img[row,col,0] = max_intensity
#                img[row,col,1] = max_intensity
#                img[row,col,2] = max_intensity
#    output_file_body = 'black'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
