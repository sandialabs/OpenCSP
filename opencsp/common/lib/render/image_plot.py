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
    output_body="image",  # Filename base.
    dpi=200,  # Resolution to write.
    include_figure_idx_in_filename=True,
):  # Whether to auto-index the filenames.
    """
    Plots an image with optional annotations and saves it to disk.

    This is meant to be a simplistic version of a plot. For a more standard OpenCSP plot,
    please use something similar to the example in :py:func:`.figure_management.setup_figure`.

    This function creates a figure to display an image, optionally drawing annotations
    and saving the figure to a specified directory. The image can be displayed in either
    RGB or BGR format.

    Parameters
    ----------
    image : np.ndarray
        The image to be plotted, represented as a NumPy array.
    draw_image : bool, optional
        If True, the image will be displayed in the figure. Defaults to True.
    rgb : bool, optional
        If True, the image is assumed to be in RGB format. If False, it is assumed to be in BGR format. Defaults to True.
    title : str | None, optional
        The title to display at the top of the figure. Defaults to None.
    annotation_list : list, optional
        A list of annotations to draw on the plot. Each annotation should have a `plot` method. Defaults to None.
    crop_box : list[list[int]] | None, optional
        A list defining the crop box as [[x_min, y_min], [x_max, y_max]]. If None, no cropping is applied. Defaults to None.
    context_str : str | None, optional
        An explanatory string to include in the status output line. Defaults to None.
    save : bool, optional
        If True, the figure will be saved to disk. Defaults to True.
    output_dir : str | None, optional
        The directory where the figure will be saved. Defaults to None.
    output_body : str, optional
        The base filename for the saved figure. Defaults to 'image'.
    dpi : int, optional
        The resolution (dots per inch) for the saved figure. Defaults to 200.
    include_figure_idx_in_filename : bool, optional
        If True, the figure index will be included in the filename. Defaults to True.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the output directory is not specified when saving the figure.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
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
        output_body_ext = output_body + ".png"
        if include_figure_idx_in_filename:
            global global_figure_idx
            output_body_ext = "{0:03d}".format(global_figure_idx) + "_" + output_body_ext
            global_figure_idx += 1
        output_dir_body_ext = os.path.join(output_dir, output_body_ext)
        print("In plot_image_figure(), called from " + context_str + ", writing " + output_dir_body_ext)
        plt.savefig(output_dir_body_ext, dpi=dpi)
    # Close plot to free up resources.
    plt.close()
