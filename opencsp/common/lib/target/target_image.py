"""
Demonstrate Solar Field Plotting Routines



"""

import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.unit_conversion as uc


def construct_target_image(image_width, image_height, dpm):  # Meters  # Meters  # Dots per meter
    image_cols = round(image_width * dpm)
    image_rows = round(image_height * dpm)
    img = np.uint8(
        np.zeros([image_rows, image_cols, 3])
    )  # ?? SCAFFOLDING RCB -- IS THIS THE CORRECT PLACE TO DO THIS?  WILL WE LOSE IMAGE PRECISION INTERMEDIATE CALCULATIONS?
    # print('In construct_target_image(), image shape =', img.shape)  # ?? SCAFFOLDING RCB -- TEMPORARY
    return img


def rows_cols(img):
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    n_bands = img.shape[2]
    if n_bands != 3:
        print("ERROR: Number of input image bands is not 3.")
        assert False
    return n_rows, n_cols


def save_image(img, output_dpm, output_dir, output_file_body, output_ext):
    ft.create_directories_if_necessary(output_dir)
    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
    print("Saving file:", output_file_dir_body_ext)
    output_dpi = round(uc.dpm_to_dpi(output_dpm))
    plt.imsave(output_file_dir_body_ext, img, dpi=output_dpi)
    print("Done.")
    return output_file_dir_body_ext
