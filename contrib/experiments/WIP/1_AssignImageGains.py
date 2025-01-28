import shutil
import multiprocessing

import exiftool
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


def get_matching_bcs_images(image_path_name_exts: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Find the matching Raw and PixelData images to the processed images.

    Performs the following steps:
        1. Find all "(.*) Processed.JPG" images
           (for example "20241004_140500.32 hourly_1400 Processed.JPG")
        2. Find all images in the same directory as the processed image, or a
           sister directory of the processed image, that match the first part of
           the processed image name
        3. Collect these images together in a tuple
        4. Return a list of all gathered tuples

    Parameters
    ----------
    image_path_name_exts: list[str]
        The images path/name.ext for all the images to be matched, such as from
        image_tools.image_files_in_directory().

    Returns
    -------
    matched_images: list[tuple[str, str|None, str|None]]
        The processed images path/name.ext, and potentially the matching Raw and
        PixelData images.
    """
    # Index image base directories
    base_dirs_to_images: dict[str, list[tuple[str, str, str]]] = {}
    for image_path_name_ext in image_path_name_exts:
        pne = ft.path_components(image_path_name_ext)
        path, name, ext = pne
        base_dir, _, _ = ft.path_components(path)

        # Add the base directory
        if base_dir not in base_dirs_to_images:
            base_dirs_to_images[base_dir] = []
        base_dirs_to_images[base_dir].append(pne)

        # Add the leaf directory
        if path not in base_dirs_to_images:
            base_dirs_to_images[path] = []
        base_dirs_to_images[path].append(pne)

    # Find matches
    matched_images = []
    for image_path_name_ext in image_path_name_exts:
        path, name, ext = ft.path_components(image_path_name_ext)

        # Check if the image is a processed image
        if name.endswith(' Processed') and ext.upper() == '.JPG':
            base_name = name.split(' Processed')[0]

            # Get the images that are in the same directory or a sister directory
            base_dir, _, _ = ft.path_components(path)
            sister_image_path_name_exts = list(set(base_dirs_to_images[path] + base_dirs_to_images[base_dir]))

            # Find matching Raw image
            raw_image = None
            for other_path, other_name, other_ext in sister_image_path_name_exts:
                if other_name == base_name + " Raw":
                    assert raw_image is None
                    raw_image = ft.join(other_path, other_name + other_ext)

            # Find matching PixelData image
            pixel_data_image = None
            for other_path, other_name, other_ext in sister_image_path_name_exts:
                if (other_name == base_name + " PixelData") or (other_name == base_name + "_nosun PixelData"):
                    assert pixel_data_image is None
                    pixel_data_image = ft.join(other_path, other_name + other_ext)

            # Collect the images together in a tuple
            matched_images.append((image_path_name_ext, raw_image, pixel_data_image))

    # Sanity check: no image gets matched multiple times
    all_matching_images: list[str] = []
    for processed, raw, pixeldata in matched_images:
        assert processed not in all_matching_images
        all_matching_images.append(processed)
        assert raw not in all_matching_images
        all_matching_images.append(raw)
        assert pixeldata not in all_matching_images
        all_matching_images.append(pixeldata)

    return matched_images


def prepare_for_tesseract(data_dir: str, processed_image_path_name_ext: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the region of image for the given processed image, and just the red
    channel for the same region of interest."""
    roi = {'left': 514, 'top': 98, 'right': 514 + 84, 'bottom': 98 + 41}

    processed_image = np.array(Image.open(ft.join(data_dir, processed_image_path_name_ext)))
    processed_image_roi = processed_image[roi['top'] : roi['bottom'], roi['left'] : roi['right']]
    processed_image_red = processed_image_roi[:, :, 0]

    return processed_image_roi, processed_image_red


def load_gain_values(
    data_dir: str, processed_raw_pixeldata: list[tuple[str, str, str]]
) -> list[tuple[str, str, str, int | None]]:
    """Returns the ISO Exif information on the given images, if they have such
    information. If not, then None is returned for that set of images."""
    processed_images = [ft.join(data_dir, processed) for processed, raw, pixeldata in processed_raw_pixeldata]
    raw_images = [ft.join(data_dir, raw) for processed, raw, pixeldata in processed_raw_pixeldata]
    pixeldata_images = [ft.join(data_dir, pixeldata) for processed, raw, pixeldata in processed_raw_pixeldata]

    with exiftool.ExifToolHelper() as et:
        processed_tags = et.get_tags(processed_images, tags=["EXIF:ISO"])
        raw_tags = et.get_tags(raw_images, tags=["EXIF:ISO"])
        pixeldata_tags = et.get_tags(pixeldata_images, tags=["EXIF:ISO"])

    has_processed_gain = ['EXIF:ISO' in tags for tags in processed_tags]
    has_raw_gain = ['EXIF:ISO' in tags for tags in raw_tags]
    has_pixeldata_gain = ['EXIF:ISO' in tags for tags in pixeldata_tags]

    gain_exif_values: list[int | None] = []
    for i in range(len(has_processed_gain)):
        if has_processed_gain[i] and has_raw_gain[i] and has_pixeldata_gain[i]:
            g1 = int(processed_tags[i]['EXIF:ISO'])
            g2 = int(raw_tags[i]['EXIF:ISO'])
            g3 = int(pixeldata_tags[i]['EXIF:ISO'])
            if g1 == g2 and g1 == g3:
                gain_exif_values.append(g1)
            else:
                gain_exif_values.append(None)
        else:
            gain_exif_values.append(None)

    ret: list[tuple[str, str, str, int | None]] = []
    for i in range(len(has_processed_gain)):
        processed, raw, pixeldata = processed_raw_pixeldata[i]
        ret.append((processed, raw, pixeldata, gain_exif_values[i]))

    return ret


def tesseract_read_gain_values(
    data_dir: str, processed: str, raw: str, pixeldata: str
) -> tuple[str, str, str, int | None]:
    """
    Use OCR to read the gain value from the processed version of the given image

    Parameters
    ----------
    data_dir : str
        The top level directory containing images. The image paths will be
        relative to this directory.
    processed : str
        The relative path/name.ext of the processed image to read the gain value from.
    raw : str
        The relative path/name.ext of the raw image that matches the processed image.
    pixeldata : str
        The relative path/name.ext of the pixeldata image that matches the processed image.

    Returns
    -------
    tuple[str, str, str, int | None]
        The [processed, raw, pixeldata, gain] tuple for the image.
    """
    gain: int = None

    # prepare the image
    processed_image_roi, processed_image_red = prepare_for_tesseract(data_dir, processed)

    # Try from the standard RGB image
    gain_str = pytesseract.image_to_string(processed_image_roi, lang='eng')
    try:
        gain = int(gain_str)
    except Exception:
        gain_str = ""

    # Try to read the gain value from just the red channel
    if gain is None:
        gain_str = pytesseract.image_to_string(processed_image_red, lang='eng')
        try:
            gain = int(gain_str)
        except Exception:
            gain_str = ""

    # Increase contrast and try again
    if gain is None:
        processed_image_red = cv.convertScaleAbs(processed_image_red, alpha=2.0, beta=0)
        gain_str = pytesseract.image_to_string(processed_image_red, lang='eng')
        try:
            gain = int(gain_str)
        except Exception:
            gain_str = ""

    return processed, raw, pixeldata, gain


def tesseract_read_gain_values_as_necessary(
    data_dir: str, processed: str, raw: str, pixeldata: str, gain: int | None
) -> tuple[str, str, str, int | None]:
    if gain is not None:
        return processed, raw, pixeldata, gain
    return tesseract_read_gain_values(data_dir, processed, raw, pixeldata)


def ask_user_for_gain(data_dir: str, processed: str, raw: str, pixeldata: str) -> int:
    # prepare the image
    processed_image_roi, processed_image_red = prepare_for_tesseract(data_dir, processed)

    # draw the image
    axis_control = rca.image(grid=False)
    figure_control = rcfg.RenderControlFigure()
    view_spec_2d = vs.view_spec_im()
    fig_record1 = fm.setup_figure(
        figure_control, axis_control, view_spec_2d, title=processed + " ROI", code_tag=f"{__file__}", equal=False
    )
    fig_record2 = fm.setup_figure(
        figure_control, axis_control, view_spec_2d, title=processed + " Red", code_tag=f"{__file__}", equal=False
    )
    fig_record1.view.imshow(processed_image_roi)
    fig_record1.view.show(block=False)
    fig_record2.view.imshow(processed_image_red)
    fig_record2.view.show(block=False)

    # ask the user for input
    gain_str = input("What is the gain for this image? ")
    gain = int(gain_str)

    fig_record1.close()
    fig_record2.close()

    return gain


def ask_user_for_gain_as_necessary(data_dir: str, processed: str, raw: str, pixeldata: str, gain: int | None) -> int:
    if gain is not None:
        return gain
    return ask_user_for_gain(data_dir, processed, raw, pixeldata)


def write_gain_value(image_path_name_exts: list[str], gain: int):
    with exiftool.ExifToolHelper() as et:
        try:
            et.set_tags(image_path_name_exts, tags={"EXIF:ISO": str(gain)}, params=["-P", "-overwrite_original"])
        except exiftool.exceptions.ExifToolExecuteError:

            # The image may have been poorly formatted the first time around
            for image_pne in image_path_name_exts:

                # Save the image to a new file with a trusted program (Pillow)
                p, n, e = ft.path_components(image_pne)
                rewrite = ft.join(p, n + " - rewrite" + e)
                Image.open(image_pne).save(rewrite)
                shutil.copystat(image_pne, rewrite)
                ft.delete_file(image_pne)
                ft.rename_file(rewrite, image_pne)

                # Try to set the gain EXIF information again
                et.set_tags(image_pne, tags={"EXIF:ISO": str(gain)}, params=["-P", "-overwrite_original"])


if __name__ == "__main__":
    data_dir = ft.join(opencsp_settings["opencsp_root_path"]["experiment_dir"], "2_Data\\BCS_Data")
    image_path_name_exts = it.image_files_in_directory(data_dir, recursive=True)
    matching_bcs_images = get_matching_bcs_images(image_path_name_exts)

    # Sanity check: there are no images that aren't matched
    all_matching_images: list[str] = []
    unmatched_images: list[str] = []
    for processed, raw, pixeldata in matching_bcs_images:
        all_matching_images.append(processed)
        all_matching_images.append(raw)
        all_matching_images.append(pixeldata)
    for image_path_name_ext in image_path_name_exts:
        if image_path_name_ext not in all_matching_images:
            unmatched_images.append(image_path_name_ext)
    assert len(unmatched_images) == 0

    # Read the already assigned gain values, in case some images already have this value
    print("Loading gain values...", end="")
    bcs_images_gains = load_gain_values(data_dir, matching_bcs_images)
    print("[done]")

    # How many images need gains?
    missing_bcs_images_gains: list[tuple[str, str, str, int | None]] = []
    for i in range(len(bcs_images_gains)):
        processed, raw, pixeldata, gain = bcs_images_gains[i]
        if gain is None:
            missing_bcs_images_gains.append(bcs_images_gains[i])
    print(f"Missing {len(missing_bcs_images_gains)} gains")

    # Use OCR to read gain values
    print("Reading gain values...")
    tesseract_bcs_images_gains: list[tuple[str, str, str, int | None]] = []
    with multiprocessing.Pool() as pool:
        chunk_size = pool._processes * 10
        for chunk_start in range(0, len(missing_bcs_images_gains), chunk_size):
            chunk_stop = min(len(missing_bcs_images_gains), chunk_start + chunk_size)
            chunk = missing_bcs_images_gains[chunk_start:chunk_stop]
            print(f"processing {chunk_start}:{chunk_stop}/{len(missing_bcs_images_gains)}")
            tesseract_bcs_images_gains += pool.starmap(
                tesseract_read_gain_values_as_necessary, [(data_dir, *matches) for matches in chunk]
            )

    # How many more gains did we identify?
    new_tesseract_bcs_images_gains: list[tuple[str, str, str, int | None]] = []
    for i in range(len(tesseract_bcs_images_gains)):
        processed, raw, pixeldata, gain = missing_bcs_images_gains[i]
        _, _, _, updated_gain = tesseract_bcs_images_gains[i]
        if updated_gain is not None:
            if (gain is None) or (updated_gain != gain):
                new_tesseract_bcs_images_gains.append(tesseract_bcs_images_gains[i])
    print(f"Found {len(new_tesseract_bcs_images_gains)} new gain values")

    # Write gain values
    print("Write gain values...", end="")
    for i in range(len(new_tesseract_bcs_images_gains)):
        print(f"\rWrite gain values...{i}", end="")
        processed, raw, pixeldata, gain = new_tesseract_bcs_images_gains[i]
        to_set_images = [ft.join(data_dir, pne) for pne in [processed, raw, pixeldata]]
        write_gain_value(to_set_images, gain)
    print(" [done]")

    # Populate unknown gain values from the user
    nvalues_from_user = len(filter(lambda prpg: prpg[3] is None, tesseract_bcs_images_gains))
    print(f"Gathering {nvalues_from_user} gain values from the user")
    manual_bcs_images_gains: list[tuple[str, str, str, int | None]] = []
    for i in range(len(tesseract_bcs_images_gains)):
        processed, raw, pixeldata, gain = tesseract_bcs_images_gains[i]
        if gain is None:
            gain = ask_user_for_gain_as_necessary(data_dir, processed, raw, pixeldata, gain)
            if gain is not None:
                manual_bcs_images_gains += [(processed, raw, pixeldata, gain)]

    # Write gain values
    print("Write gain values...", end="")
    for i in range(len(manual_bcs_images_gains)):
        print(f"\rWrite gain values...{i}", end="")
        processed, raw, pixeldata, gain = manual_bcs_images_gains[i]
        to_set_images = [ft.join(data_dir, pne) for pne in [processed, raw, pixeldata]]
        write_gain_value(to_set_images, gain)
    print(" [done]")
