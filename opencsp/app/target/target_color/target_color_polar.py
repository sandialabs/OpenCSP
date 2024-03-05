"""
Demonstrate Solar Field Plotting Routines



"""

#    # Polar red to green to blue to red
#    for row in range(0,n_rows):
#        for col in range(0,n_cols):
#            x = col
#            y = n_rows - row
#            cx = x_max/2
#            cy = y_max/2
#            dx = x-cx
#            dy = y-cy
#            r = math.sqrt((dx*dx) + (dy*dy))
#            theta = math.atan2(dy,dx)
#            if theta < 0:
#                theta += (2*np.pi)
#            third_theta = (2*np.pi)/3
#            # Red
#            if theta < third_theta:
#                red_frac = 1 - (theta/third_theta)
#            elif theta < (2 * third_theta):
#                red_frac = 0
#            else:
#                red_frac = (theta-(2*third_theta))/third_theta
#            # Green
#            if theta < third_theta:
#                green_frac = theta/third_theta
#            elif theta < (2 * third_theta):
#                green_frac = 1 - ((theta-third_theta)/third_theta)
#            else:
#                green_frac = 0
#            # Blue
#            if theta < third_theta:
#                blue_frac = 0
#            elif theta < (2*third_theta):
#                blue_frac = (theta-third_theta)/third_theta
#            else:
#                blue_frac = 1 - ((theta-(2*third_theta))/third_theta)
#            # Set pixel color
#            img[row,col,0] = red_frac * max_intensity
#            img[row,col,1] = green_frac * max_intensity
#            img[row,col,2] = blue_frac * max_intensity
#    output_file_body = 'polar_red_to_green_to_blue_to_red'
#    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#    print('Saving file:', output_file_dir_body_ext)
#    plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#    print('Done.')
# 
#     # Color by angle
#     for row in range(0,n_rows):
#         for col in range(0,n_cols):
#             row_vs_center = row - (n_rows/2)
#             col_vs_center = col - (n_cols/2)
#             row_frac = (row/n_rows)
#             col_frac = (col/n_cols)
#             img[row,col,0] = int(row_frac * max_intensity)
#             img[row,col,1] = int((1-row_frac) * max_intensity) + int(col_frac * max_intensity)
#             img[row,col,2] = int((1-col_frac) * max_intensity)
#     output_file_body = 'green_to_red_cross_green_to_blue'
#     output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
#     print('Saving file:', output_file_dir_body_ext)
#     plt.imsave(output_file_dir_body_ext, img, dpi=dpi)
#     print('Done.')
