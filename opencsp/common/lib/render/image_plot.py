"""
Image plotting, with annotations.



"""

import cv2 as cv
import matplotlib.pyplot as plt
import os

global_figure_idx = 1


def plot_image_figure(
    image,
    draw_image=True,  # Whether to display the image.
    rgb=True,  # True ==> image is RGB.  False ==> image is BGR.
    title=None,  # String to put at top of figure
    annotation_list=None,  # Annotations to draw on the plot.
    crop_box=None,  # Crop box is [[x_min, y_min], [x_max, y_max]] or None.
    context_str=None,  # Explanatory string to include in status output line.
    save=True,  # Whether to write to disk.
    output_dir=None,  # Where to write.
    output_body='image',  # Filename base.
    dpi=200,  # Resolution to write.
    include_figure_idx_in_filename=True,
):  # Whether to auto-index the filenames.
    # Create the figure.
    plt.figure()
    plt.title(title)

    # Add the image.
    if draw_image:
        if rgb == True:
            plt.imshow(image)
        else:
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    # Add annotations.
    if (annotation_list != None) and (len(annotation_list) > 0):
        for annotation in annotation_list:
            annotation.plot(crop_box)

    # Save.
    if save:
        output_body_ext = output_body + '.png'
        if include_figure_idx_in_filename:
            global global_figure_idx
            output_body_ext = '{0:03d}'.format(global_figure_idx) + '_' + output_body_ext
            global_figure_idx += 1
        output_dir_body_ext = os.path.join(output_dir, output_body_ext)
        print('In plot_image_figure(), called from ' + context_str + ', writing ' + output_dir_body_ext)
        plt.savefig(output_dir_body_ext, dpi=dpi)
    # Close plot to free up resources.
    plt.close()
