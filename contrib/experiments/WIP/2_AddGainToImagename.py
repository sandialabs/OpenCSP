import shutil
import multiprocessing

import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract

from opencsp import opencsp_settings
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt

from contrib.experiments.WIP.AssignImageGains import get_matching_bcs_images


def get_parent_dir(image_path_name_ext: str):
    image_path, image_name, image_ext = ft.path_components(image_path_name_ext)
    parent_path, image_subdir, _ = ft.path_components(image_path)
    return parent_path


if __name__ == "__main__":
    data_dir = ft.join(opencsp_settings["opencsp_root_path"]["experiment_dir"], "2_Data")
    image_path_name_exts = it.image_files_in_directory(data_dir, recursive=True)

    gains = it.get_exif_value(data_dir, image_path_name_exts)
    for gain, image_path_name_ext in zip(gains, image_path_name_exts):
        if gain is None:
            lt.error(f"Image {image_path_name_ext} has no gain value set")
        else:
            p, n, e = ft.path_components(image_path_name_ext)
            ft.rename_file(ft.join(data_dir, image_path_name_ext), ft.join(data_dir, p, n + f"_g{gain}" + e))
