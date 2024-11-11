"""
Demonstrate Solar Field Plotting Routines



"""

import matplotlib.pyplot as plt


#    # Red cross green
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            x = col
#            y = n_rows - row
#            x_frac = (x/x_max)
#            y_frac = (y/y_max)
#            img[row,col,0] = x_frac * max_intensity
#            img[row,col,1] = y_frac * max_intensity
#            img[row,col,2] = 0
#    output_file_body = 'red_cross_green'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
#
#    # Blue underlying red cross green
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
#
#    output_file_body = 'blue_under_red_cross_green_100dpi'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=100)
#
#    output_file_body = 'blue_under_red_cross_green_10dpi'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=10)
#
#    print('Done.')
#
#    # (Green to red) cross (Green to blue)
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            half_max_intensity = max_intensity/2.0
#            x = col
#            y = n_rows - row
#            x_frac = (x/x_max)
#            y_frac = (y/y_max)
#            img[row,col,0] = x_frac * half_max_intensity
#            img[row,col,1] = (1-x_frac) * half_max_intensity + (1-y_frac) * half_max_intensity
#            img[row,col,2] = y_frac * half_max_intensity
#    output_file_body = 'green_to_red_cross_green_to_blue'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
