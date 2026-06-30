import os
import gc
from logging import INFO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import imageio.v2 as imageio
from PIL import Image
import rawpy  # Import rawpy for processing RAW image files
from tqdm import tqdm

# import opencsp.app.lookback.lookback_tools as lbt
import contrib.app.LookFast.lookback_tools as lbt

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_coverage_map_mp_debug.txt",
    log_type=INFO,
)


def process_image(image_file, threshold_fractions, max_intensity, is_raw):
    """
    Processes a single image to create binary maps for the given thresholds.

    Parameters:
        image_file (str): Path to the image file.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        max_intensity (float): Maximum possible intensity value for the image format.
        is_raw (bool): Whether the image is a RAW file.

    Returns:
        dict: A dictionary where keys are threshold fractions and values are binary maps for the image.
    """
    try:
        # Read the image
        if is_raw:
            with rawpy.imread(image_file) as raw:
                image = raw.postprocess()
        else:
            image = imageio.imread(image_file)

        # Handle multi-channel images (e.g., RGB)
        if len(image.shape) == 3:  # Multi-channel image
            # Calculate pixel intensity as the average across channels
            pixel_intensity = np.mean(image, axis=-1)
        else:  # Grayscale image
            pixel_intensity = image

        # Create binary maps for each threshold
        binary_maps = {}
        for threshold_fraction in threshold_fractions:
            intensity_threshold = threshold_fraction * max_intensity
            binary_maps[threshold_fraction] = (pixel_intensity > intensity_threshold).astype(bool)

        return binary_maps

    except Exception as e:
        logger.error("Error processing %s : %s", image_file, e, exc_info=True)
        return None


def compile_binary_maps(output_folder, threshold_fractions, prefix):
    """
    Compiles all binary maps from batches into a single final output for each threshold.

    Parameters:
        output_folder (str): Path to the folder containing the binary map outputs.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        prefix (str): Prefix for identifying binary maps (e.g., "traditional" or "raw").

    Returns:
        None
    """
    # Initialize a dictionary to store the compiled binary maps
    compiled_binary_maps = {}

    # Iterate over each threshold fraction
    for threshold_fraction in threshold_fractions:
        compiled_map = None
        # Find all batch files for the current threshold
        batch_files = [
            os.path.join(output_folder, f)
            for f in os.listdir(output_folder)
            if f.startswith(f"{prefix}_binary_map_{int(threshold_fraction * 100):02d}")
        ]

        # Combine all batch files
        for batch_file in batch_files:
            binary_map = imageio.imread(batch_file) // 255  # Convert back to binary (0 or 1)
            if compiled_map is None:
                compiled_map = binary_map
            else:
                compiled_map = np.maximum(compiled_map, binary_map)  # Combine using logical OR

        # Store the compiled map
        compiled_binary_maps[threshold_fraction] = compiled_map

        # Save the compiled map to disk
        output_path = os.path.join(
            output_folder, f"{prefix}_compiled_binary_map_{int(threshold_fraction * 100):02d}.png"
        )
        imageio.imwrite(output_path, compiled_map * 255)  # Scale binary map to 0-255 for saving as an image
        print(f"Compiled binary map for threshold {threshold_fraction} saved to {output_path}")


def _close_memmap(arr):
    # arr is a numpy.memmap
    try:
        arr.flush()
    except Exception:
        pass
    # Close underlying mmap handle if present (important on Windows)
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except Exception:
            pass


def process_images_in_batches(
    image_files, threshold_fractions, max_intensity, is_raw, batch_num, output_folder, prefix
):
    """
    Processes images in batches to create binary maps for the given thresholds.

    Parameters:
        image_files (list of str): List of image file paths to process.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        max_intensity (float): Maximum possible intensity value for the image format.
        is_raw (bool): Whether the images are RAW files.
        batch_num (int): Batch Number.
        output_folder (str): Path to location for binary map output.
        prefix (str): Prefix for output filenames.

    Returns:
        None
    """
    # Initialize binary maps for each threshold
    first_image = imageio.imread(image_files[0]) if not is_raw else rawpy.imread(image_files[0]).postprocess()
    H, W = first_image.shape[0], first_image.shape[1]

    # IMPORTANT: back memmaps with a non-PNG file
    binary_maps = {}
    memmap_paths = {}
    for threshold in threshold_fractions:
        mm_path = os.path.join(output_folder, f"{prefix}_binary_map_{int(threshold * 100):02d}_{batch_num}.dat")
        memmap_paths[threshold] = mm_path
        binary_maps[threshold] = np.memmap(mm_path, dtype=np.bool_, mode="w+", shape=(H, W))
        binary_maps[threshold].fill(False)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_image, os.path.normpath(f), threshold_fractions, max_intensity, is_raw)
            for f in image_files
        ]

        with tqdm(total=len(futures), desc=f"Coverage Map Batch {batch_num}", unit="images") as pbar:
            for fut in as_completed(futures):
                result = None
                try:
                    result = fut.result()
                    if result:
                        for t in threshold_fractions:
                            # In-place OR to avoid temporary arrays
                            np.maximum(binary_maps[t], result[t], out=binary_maps[t])
                except Exception as e:
                    logger.error("Error processing image: %s", e, exc_info=True)
                finally:
                    # ensure ref dropped even on exceptions
                    result = None
                    pbar.update(1)

    # Write PNG outputs exactly as before
    for t, binary_map in binary_maps.items():
        output_path = os.path.join(output_folder, f"{prefix}_binary_map_{int(t * 100):02d}_{batch_num}.png")
        bin_image = np.asarray(binary_map, dtype=np.uint8) * 255
        Image.fromarray(bin_image).save(output_path)
        print(f"Binary map for threshold {t} saved to {output_path}")

    # Explicitly flush/close and remove memmap backing files
    for t, mm in binary_maps.items():
        _close_memmap(mm)
        # Remove the .dat file to keep behavior similar (only PNG outputs remain)
        try:
            os.remove(memmap_paths[t])
        except OSError:
            pass

    binary_maps.clear()
    gc.collect()


def construct_binary_maps_parallel(
    image_folder,
    output_folder,
    checkpoint_folder,
    threshold_fractions,
    batch_size=10,
    max_workers=4,
    checkpoint_file="coverage_map_checkpoint.json",
):
    """
    Constructs a series of binary maps of pixels that exceeded intensity thresholds across all images,
    including traditional formats using parallel processing and batching.
    Supports restarting from a checkpoint.

    Checkpoint behavior update:
      - If new threshold fractions are requested that are NOT in checkpoint_data["completed_thresholds"],
        the function resets per-image progress ("processed_images", "current_batch") and re-processes
        the whole dataset for the union of (old completed + new requested) thresholds.
      - After completion, checkpoint_data["completed_thresholds"] is extended (union) and
        "processed_images" is repopulated to reflect the completed processing pass.

    Notes/assumptions:
      - This is the "minimal change" approach: when new thresholds are requested, we re-run the
        batch processing for *all* images (not just the new thresholds), because the per-image
        products are threshold-dependent.
      - completed_thresholds is treated as a *set* of fraction values (with float tolerance via rounding).

    Parameters:
        image_folder (str): Path to the folder containing the sequence of images.
        output_folder (str): Path to location for binary map output.
        checkpoint_folder (str): Path to the checkpoint folder.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        batch_size (int): Number of images to process in each batch.
        max_workers (int): Maximum number of parallel threads to use.
        checkpoint_file (str): Checkpoint file name.

    Returns:
        None
    """

    # ----------------------------
    # helpers
    # ----------------------------
    def _flatten(seq):
        """Flatten arbitrarily nested lists/tuples/sets (but not strings/bytes)."""
        if seq is None:
            return
        if isinstance(seq, (list, tuple, set)):
            for item in seq:
                yield from _flatten(item)
        else:
            yield seq

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)

    if checkpoint_data is None:
        checkpoint_data = {"processed_images": [], "current_batch": 0, "completed_thresholds": [], "prefix": None}
    else:
        checkpoint_data["processed_images"] = [os.path.basename(p) for p in checkpoint_data.get("processed_images", [])]

    requested_thr = _flatten(threshold_fractions)
    completed_thr = _flatten(checkpoint_data.get("completed_thresholds", []))

    requested_set = set(requested_thr)
    completed_set = set(completed_thr)

    new_thresholds = sorted(requested_set - completed_set)
    needs_rerun = len(new_thresholds) > 0

    # If anything new was requested, wipe progress and rerun using union thresholds
    if needs_rerun:
        checkpoint_data["processed_images"] = []
        checkpoint_data["current_batch"] = 0
        # keep completed_thresholds as-is for now; only update after successful compile
        lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)
    else:
        union_thresholds = requested_thr  # nothing new; run normally / continue if interrupted

    # List all image files in the folder
    image_files_all = os.listdir(image_folder)
    image_files = []
    image_files_raw = []
    for f in image_files_all:
        if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP")):
            image_files.append(os.path.join(image_folder, f))
        elif f.endswith((".nef", ".cr2", ".arw", ".dng", ".NEF", ".CR2", ".ARW", ".DNG")):
            image_files_raw.append(os.path.join(image_folder, f))

    if not image_files and not image_files_raw:
        raise ValueError("No image files (traditional or RAW) found in the specified folder.")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process traditional images
    if image_files:
        # Determine the maximum possible intensity value for traditional images
        first_image = imageio.imread(image_files[0])
        max_intensity = np.iinfo(first_image.dtype).max if np.issubdtype(first_image.dtype, np.integer) else 1.0

        # If we did not wipe progress, continue from checkpoint; otherwise run from scratch
        unprocessed_images = [
            img for img in image_files if os.path.basename(img) not in checkpoint_data["processed_images"]
        ]

        # Process images in batches
        for batch_start in range(checkpoint_data["current_batch"], len(unprocessed_images), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_images))
            batch_files = unprocessed_images[batch_start:batch_end]
            # Extract file names
            # file_names = [os.path.basename(file) for file in batch_files]

            process_images_in_batches(
                batch_files,
                new_thresholds,
                max_intensity,
                is_raw=False,
                batch_num=batch_start // batch_size + 1,
                output_folder=output_folder,
                prefix="traditional",
            )

            # Update checkpoint
            checkpoint_data["processed_images"].extend(os.path.basename(f) for f in batch_files)
            checkpoint_data["current_batch"] = batch_start + batch_size
            checkpoint_data["prefix"] = "traditional"
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        # Compile binary maps for traditional images
        if new_thresholds not in checkpoint_data["completed_thresholds"]:
            compile_binary_maps(output_folder, new_thresholds, prefix="traditional")
            checkpoint_data["completed_thresholds"].extend(new_thresholds)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)


'''
    # Process RAW images
    if image_files_raw:
        # Determine the maximum possible intensity value for RAW images
        with rawpy.imread(image_files_raw[0]) as raw:
            first_image_raw = raw.postprocess()
        max_intensity_raw = (
            np.iinfo(first_image_raw.dtype).max if np.issubdtype(first_image_raw.dtype, np.integer) else 1.0
        )

        # Filter out already processed images
        unprocessed_images_raw = [img for img in image_files_raw if img not in checkpoint_data["processed_images"]]

        # Process images in batches
        for batch_start in range(checkpoint_data["current_batch"], len(unprocessed_images_raw), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_images_raw))
            batch_files = unprocessed_images_raw[batch_start:batch_end]
            # Extract file names
            file_names = [os.path.basename(file) for file in batch_files]

            process_images_in_batches(
                batch_files,
                threshold_fractions,
                max_intensity_raw,
                is_raw=True,
                batch_num=batch_start // batch_size + 1,
                output_folder=output_folder,
                prefix="raw",
            )

            # Update checkpoint
            checkpoint_data["processed_images"].extend(batch_files)
            checkpoint_data["current_batch"] = batch_start + batch_size
            checkpoint_data["prefix"] = "raw"
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        # Compile binary maps for RAW images
        if "raw" not in checkpoint_data["completed_thresholds"]:
            compile_binary_maps(output_folder, threshold_fractions, prefix="raw")
            checkpoint_data["completed_thresholds"].append("raw")
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

'''
