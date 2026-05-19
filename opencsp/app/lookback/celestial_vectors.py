import os
from logging import DEBUG, ERROR
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from multiprocessing import Pool, Manager

import skyfield.api as skf
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import opencsp.app.lookback.lookback_tools as lbt

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_celestial_vectors_debug.txt",
    log_type=ERROR,
)


def calculate_vectors_celestial_observer(celestial_object_name, target_location, observer_location, observation_time):
    """
    Calculates vectors from a celestial object to a target point and from the target point to an observer.

    Parameters:
        celestial_object_name (str): Name of the celestial object (e.g., "moon", "sun").
        target_location (tuple): Latitude, longitude, and elevation of the target point (degrees, meters).
        observation_time (tuple): Observation time as (year, month, day, hour, minute, second).

    Returns:
        dict: A dictionary containing the vectors:
            - "celestial_to_target": Vector from the celestial object to the target point (normalized).
            - "target_to_observer": Vector from the target point to the observer (normalized).
    """
    # Load ephemeris data (DE430 dataset)
    eph = skf.load("de430t.bsp")  # DE430 ephemeris file

    # Define celestial object
    celestial_object = eph[celestial_object_name]

    # Define target location
    target = skf.Topos(
        latitude_degrees=target_location[0], longitude_degrees=target_location[1], elevation_m=target_location[2]
    )
    # Define observer location
    observer = skf.Topos(
        latitude_degrees=observer_location[0], longitude_degrees=observer_location[1], elevation_m=observer_location[2]
    )

    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, skf.Time):
        pass
    elif isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)

    # Calculate position of the celestial object relative to the target
    earth = eph["earth"]
    target_position_vec = earth + target
    target_position_bary = target_position_vec.at(observation_time)
    target_obsv_celest = target_position_bary.observe(celestial_object)
    target_obsv_celest_apparent = target_obsv_celest.apparent()

    celestial_to_target = target_position_vec.at(observation_time).observe(celestial_object).position.km
    celestial_to_target_normalized = celestial_to_target / np.linalg.norm(celestial_to_target)

    # Calculate position of the target relative to the observer
    observer_position = earth + observer
    target_to_observer = observer_position.at(observation_time).observe(target_position_vec)

    earth_target_bary = earth.at(observation_time)  # Barycentric Coordinate System
    earth_obsv_target = earth_target_bary.observe(target_position_vec)  # Astrometric Position
    earth_obsv_target_apparent = earth_obsv_target.apparent()  # Apparent Observer reference frame

    NSTTF_coord_toca = [
        target_obsv_celest_apparent.frame_xyz(target).km[1],
        target_obsv_celest_apparent.frame_xyz(target).km[0],
        target_obsv_celest_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_eota = [
        earth_obsv_target_apparent.frame_xyz(target).km[1],
        earth_obsv_target_apparent.frame_xyz(target).km[0],
        earth_obsv_target_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_otta = [
        target_to_observer.frame_xyz(target).km[1],
        target_to_observer.frame_xyz(target).km[0] * -1,
        target_to_observer.frame_xyz(target).km[2] * -1,
    ]
    # Calculate angular size in radians
    distance_to_celestial_object = target_obsv_celest_apparent.distance().km  # Distance in kilometers

    if celestial_object_name.lower() == 'moon':
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon
    elif celestial_object_name.lower() == 'sun':
        radius_of_celestial_object = 1391400 / 2  # Radius in kilometers of the Sun
    else:
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon

    angular_size_radians = 2 * np.arctan(radius_of_celestial_object / distance_to_celestial_object)

    # Apparent function returns a tuple with (Altitude, Azimuth, and Distance)
    # https://rhodesmill.org/skyfield/positions.html#reference-frames
    # Altitude measures the angle above or below the horizon. The zenith is at +90°, an object on the horizon’s great circle is at 0°, and the nadir beneath your feet is at −90°.
    # Azimuth measures the angle around the sky from the north pole: 0° means exactly north, 90° is east, 180° is south, and 270° is west.
    # "earth_to_target_spherical": earth_obsv_target_apparent.altaz()
    # Cartesian for horizonal skyfield coordinates is Left Handed {x points north, y points east, z points to zenith}
    # NSTTF coordinates is Right Handed {x points east, y points north, z points to zenith}
    return {
        "celestial_to_target": celestial_to_target_normalized,
        "cel_to_target_cartesian": NSTTF_coord_toca / np.linalg.norm(NSTTF_coord_toca),
        "earth_to_target_cartesian": NSTTF_coord_eota / np.linalg.norm(NSTTF_coord_eota),
        "target_to_observer": NSTTF_coord_otta / np.linalg.norm(NSTTF_coord_otta),
        "angular_size_radians": angular_size_radians,
    }


def process_pixel(args):
    (
        pixel,
        transitions,
        video_metadata,
        video_start_time,
        camera_time_shift,
        celestial_object_name,
        target_location,
        observer_location,
        tzone,
    ) = args

    checkpoint_data = {"processed_pixels": [], "pixels_with_data": [], "pixels_without_data": []}

    frame_diff = 0
    if len(transitions) == 0 or len(transitions) < 2:
        checkpoint_data["processed_pixels"].append(pixel)
        checkpoint_data["pixels_without_data"].append(pixel)
        # lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        return pixel, None, checkpoint_data

    if len(transitions) == 2:
        frame_range = []
        for transition in transitions:
            if transition["transition"] == "bright":
                frame_range.append(tuple((1, lbt.frame_number_from_img_name(transition['to_frame']))))
            elif transition["transition"] == "dark":
                frame_range.append(tuple((0, lbt.frame_number_from_img_name(transition['to_frame']))))
        frame_diff = frame_range[1][1] - frame_range[0][1]
    elif len(transitions) > 2:
        frame_range_all = []
        for transition in transitions:
            if transition["transition"] == "bright":
                frame_range_all.append(tuple((1, lbt.frame_number_from_img_name(transition['to_frame']))))
            elif transition["transition"] == "dark":
                frame_range_all.append(tuple((0, lbt.frame_number_from_img_name(transition['to_frame']))))
        frame_range, frame_diff = maximum_frame_range(frame_ranges=frame_range_all)

    elapsed_time_bright = frame_diff / video_metadata['frame_rate']
    elapsed_time_start = frame_range[0][1] / video_metadata['frame_rate']

    bright_start_time = video_start_time + timedelta(seconds=elapsed_time_start)
    bright_end_time = video_start_time + timedelta(seconds=elapsed_time_start) + timedelta(seconds=elapsed_time_bright)

    obsv_time_utc_start = lbt.define_observation_time_skyfield(bright_start_time, camera_time_shift)
    obsv_time_utc_end = lbt.define_observation_time_skyfield(bright_end_time, camera_time_shift)

    start_vector = calculate_vectors_celestial_observer(
        celestial_object_name=celestial_object_name,
        target_location=target_location,
        observer_location=observer_location,
        observation_time=obsv_time_utc_start,
    )
    end_vector = calculate_vectors_celestial_observer(
        celestial_object_name=celestial_object_name,
        target_location=target_location,
        observer_location=observer_location,
        observation_time=obsv_time_utc_end,
    )
    points = np.array([[0, 0, 0], start_vector['cel_to_target_cartesian'], end_vector['cel_to_target_cartesian']])
    radii = np.array([1, start_vector['angular_size_radians'] / 2, end_vector['angular_size_radians'] / 2])

    pixel_data = {
        "start_time_Local": obsv_time_utc_start.astimezone(tzone),
        "start_time_UTC": obsv_time_utc_start.utc_datetime(),
        "start_vector": start_vector,
        "end_time_Local": obsv_time_utc_end.astimezone(tzone),
        "end_time_UTC": obsv_time_utc_end.utc_datetime(),
        "end_vector": end_vector,
        "intersection_1": [],
        "intersection_2": [],
        "slope_1": [],
        "slope_2": [],
        "observer_vector": (start_vector['target_to_observer'] + end_vector['target_to_observer']) / 2,
    }

    try:
        intersection_1, intersection_2 = trilaterate(points, radii, raise_on_no_solution=False)

        observer_vec, slope_1, slope_2 = safe_calculate_slope(
            start_vector['target_to_observer'], end_vector['target_to_observer'], intersection_1, intersection_2
        )

        pixel_data = {
            "start_time_Local": obsv_time_utc_start.astimezone(tzone),
            "start_time_UTC": obsv_time_utc_start.utc_datetime(),
            "start_vector": start_vector,
            "end_time_Local": obsv_time_utc_end.astimezone(tzone),
            "end_time_UTC": obsv_time_utc_end.utc_datetime(),
            "end_vector": end_vector,
            "intersection_1": intersection_1,
            "intersection_2": intersection_2,
            "slope_1": slope_1,
            "slope_2": slope_2,
            "observer_vector": observer_vec,
        }

        checkpoint_data["processed_pixels"].append(pixel)
        checkpoint_data["pixels_with_data"].append(pixel)
        # lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
    # except ValueError:
    except Exception:
        logger.debug("process_pixel Error pixel %s", pixel, exc_info=True)

        try:
            observer_vec, slope_1, slope_2 = safe_calculate_slope(
                start_vector['target_to_observer'], end_vector['target_to_observer'], intersection_1, intersection_2
            )

            pixel_data = {
                "start_time_Local": obsv_time_utc_start.astimezone(tzone),
                "start_time_UTC": obsv_time_utc_start.utc_datetime(),
                "start_vector": start_vector,
                "end_time_Local": obsv_time_utc_end.astimezone(tzone),
                "end_time_UTC": obsv_time_utc_end.utc_datetime(),
                "end_vector": end_vector,
                "intersection_1": intersection_1,
                "intersection_2": intersection_2,
                "slope_1": slope_1,
                "slope_2": slope_2,
                "observer_vector": observer_vec,
            }
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_with_data"].append(pixel)
        except Exception:
            logger.debug("process_pixel Error pixel %s", pixel, exc_info=True)
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_without_data"].append(pixel)
            # lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
            return pixel, pixel_data, checkpoint_data

    return pixel, pixel_data, checkpoint_data


def extract_pixel_timing_and_celestial_vectors_parallel(
    celestial_object_name,
    target_location,
    observer_location,
    camera_time_shift,
    data_time_zone,
    data_location,
    video_metadata,
    output_folder,
    output_json_name,
    checkpoint_folder,
    checkpoint_file,
    batch_size=5000,  # Number of pixels to process in each batch
):
    if os.path.exists(os.path.join(output_folder, output_json_name)):
        logger.info("Compiled output file already exists, skipping code block")
        return

    video_start_time = datetime.strptime(video_metadata['create_date'], "%Y:%m:%d %H:%M:%S")
    timezone = ZoneInfo(data_time_zone)
    video_start_time = video_start_time.replace(tzinfo=timezone)

    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_pixels": [], "pixels_with_data": [], "pixels_without_data": []}

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read in compressed json with lookback tools function
    data = lbt.read_compressed_json(data_location)

    # Convert data to a list of items for batching
    data_items = list(data.items())
    total_pixels = len(data_items)

    # Process data in batches
    for batch_start in range(0, total_pixels, batch_size):
        batch_end = min(batch_start + batch_size, total_pixels)
        batch_data = data_items[batch_start:batch_end]
        # Filter batch_data to exclude already processed pixels
        filtered_batch_data = [
            (pixel, transitions)
            for pixel, transitions in batch_data
            if pixel not in checkpoint_data["processed_pixels"]
        ]
        # Use multiprocessing to parallelize the pixel processing
        with Manager() as manager:
            args_list = [
                (
                    pixel,
                    transitions,
                    video_metadata,
                    video_start_time,
                    camera_time_shift,
                    celestial_object_name,
                    target_location,
                    observer_location,
                    timezone,
                )
                for pixel, transitions in filtered_batch_data
            ]

            # with Pool(processes=1) as pool:  # Maybe look into hyperthreading
            with Pool(processes=os.cpu_count()) as pool:  # Use all available CPU cores
                results = list(
                    tqdm(
                        pool.imap(process_pixel, args_list),
                        total=len(filtered_batch_data),
                        desc=f"Processing Batch {batch_start}-{batch_end}",
                    )
                )

            # Aggregate checkpoint data
            for _, _, pixel_checkpoint_data in results:
                ppixel = pixel_checkpoint_data["processed_pixels"]
                dpixel = pixel_checkpoint_data["pixels_with_data"]
                wpixel = pixel_checkpoint_data["pixels_without_data"]
                checkpoint_data["processed_pixels"].append(ppixel[0])
                if dpixel:
                    checkpoint_data["pixels_with_data"].append(dpixel[0])
                if wpixel:
                    checkpoint_data["pixels_without_data"].append(wpixel[0])

            # Save checkpoint data serially after each batch
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)

            # Update the data with processed results
            for pixel, pixel_data, _ in results:
                if pixel_data is not None:
                    data[pixel] = pixel_data

            # Save intermediate results to avoid data loss
            if results:
                base_name = os.path.splitext(os.path.basename(output_json_name))[0]
                lbt.write_compressed_json(
                    data=data,
                    file_path=os.path.join(output_folder, f"{base_name}_batch_{batch_start}_{batch_end}.json.gz"),
                    print_path=False,
                )

    # Save final results
    if os.path.exists(os.path.join(output_folder, output_json_name)):
        pass
    else:
        lbt.write_compressed_json(data=data, file_path=os.path.join(output_folder, output_json_name), print_path=True)


def calculate_slope(observer_vec_start, observer_vec_end, inter_1, inter_2):
    observer_vec = (observer_vec_start + observer_vec_end) / 2
    slope_1 = (observer_vec + inter_1) / np.linalg.norm(observer_vec + inter_1)
    slope_2 = (observer_vec + inter_2) / np.linalg.norm(observer_vec + inter_2)
    return observer_vec, slope_1, slope_2


def safe_calculate_slope(start_vector, end_vector, intersection_1, intersection_2):
    """
    Wrapper for calculate_slope with error handling and logging.
    """
    try:
        # Call the calculate_slope function
        observer_vec, slope_1, slope_2 = calculate_slope(start_vector, end_vector, intersection_1, intersection_2)
        return observer_vec, slope_1, slope_2
    except KeyError as e:
        # Handle missing keys in dictionaries
        logger.debug("KeyError in calculate_slope: %s", e, exc_info=True)
        return None, None, None
    except TypeError as e:
        # Handle type-related issues (e.g., NoneType or invalid types)
        logger.debug("TypeError in calculate_slope: %s", e, exc_info=True)
        return None, None, None
    except Exception as e:
        # Catch any other unexpected errors
        logger.debug("Unexpected error in calculate_slope: %s", e, exc_info=True)
        return None, None, None


def plot_pixel_batch(batch_pixels, data, checkpoint_data, output_folder_img, celestial_object):
    """
    Process a batch of pixels and return the list of successfully plotted pixels.
    """
    plotted_pixels = []

    for pixel in batch_pixels:
        height, _ = eval(pixel)
        plot_file = os.path.normpath(os.path.join(output_folder_img, str(height), f"pixel_{pixel}_sky_plot_slope.png"))

        if pixel in checkpoint_data["pixels_without_data"]:
            logger.info("Skipping 3D plot for pixel %s without data.", pixel)
            continue

        if isinstance(data[pixel], list):
            logger.info("Pixel %s incorrectly saved as having data", pixel)
            continue

        if os.path.exists(plot_file):
            logger.info("Pixel %s 3D plot already exists, skipping", pixel)
            continue

        start_vector = data[pixel]['start_vector']
        end_vector = data[pixel]['end_vector']
        local_time_start = data[pixel]['start_time_Local']
        local_time_end = data[pixel]['end_time_Local']
        inter_1 = data[pixel]['intersection_1']
        inter_2 = data[pixel]['intersection_2']
        slope_1 = data[pixel]['slope_1']
        slope_2 = data[pixel]['slope_2']
        observer_vec = data[pixel]['observer_vector']

        # Create the Positive Zenith side of the south half of a unit sphere for visualization of the horizon
        u = np.linspace(np.pi, 2 * np.pi, 100)
        v = np.linspace(0, np.pi / 2, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Initialize 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the unit sphere
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)

        # Initialize inset 3D plot
        inset_ax = fig.add_axes([0.02, 0.05, 0.3, 0.3], projection='3d')

        # Plot the unit sphere on the inset plot
        inset_ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)

        ins_x_lim = []
        ins_y_lim = []
        ins_z_lim = []

        # Calculate azimuthal angle (azim)
        azim = np.degrees(np.arctan2(slope_1[1], slope_1[0]))

        # Calculate elevation angle (elev)
        if slope_1[0] == 0 and slope_1[1] == 0:
            elev = 90 if slope_1[2] > 0 else -90
        else:
            elev = np.degrees(np.arctan2(slope_1[2], np.sqrt(slope_1[0] ** 2 + slope_1[1] ** 2)))

        inset_ax.view_init(elev=elev, azim=azim)

        # Plot the projections for each interpolated time
        for i, vector in enumerate([start_vector, end_vector]):
            angular_radius = vector['angular_size_radians'] / 2
            num_points = 360
            theta = np.linspace(0, 2 * np.pi, num_points)

            arbitrary_vector = (
                np.array([1, 0, 0])
                if not np.allclose(vector['cel_to_target_cartesian'], [1, 0, 0])
                else np.array([0, 1, 0])
            )
            basis1 = np.cross(vector['cel_to_target_cartesian'], arbitrary_vector)
            basis1 /= np.linalg.norm(basis1)

            basis2 = np.cross(vector['cel_to_target_cartesian'], basis1)
            basis2 /= np.linalg.norm(basis2)

            circle_points = []
            for angle in theta:
                point = np.cos(angular_radius) * vector['cel_to_target_cartesian'] + np.sin(angular_radius) * (
                    np.cos(angle) * basis1 + np.sin(angle) * basis2
                )
                circle_points.append(point)
            circle_points = np.array(circle_points)

            ins_x_lim.append([np.min(circle_points[:, 0]), np.max(circle_points[:, 0])])
            ins_y_lim.append([np.min(circle_points[:, 1]), np.max(circle_points[:, 1])])
            ins_z_lim.append([np.min(circle_points[:, 2]), np.max(circle_points[:, 2])])

            ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red')
            inset_ax.plot(
                circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red'
            )

            if i == 0:
                ax.quiver(
                    0,
                    0,
                    0,
                    vector['cel_to_target_cartesian'][0],
                    vector['cel_to_target_cartesian'][1],
                    vector['cel_to_target_cartesian'][2],
                    color='green',
                    label=f"Start Time Local: {local_time_start}",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    observer_vec[0],
                    observer_vec[1],
                    observer_vec[2],
                    color='black',
                    label="Observer to Target",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    slope_1[0],
                    slope_1[1],
                    slope_1[2],
                    color='royalblue',
                    label="Slope Option 1",
                    arrow_length_ratio=0.1,
                )
                inset_ax.scatter(inter_1[0], inter_1[1], inter_1[2], c='royalblue', label="Intersection 1")
            else:
                ax.quiver(
                    0,
                    0,
                    0,
                    vector['cel_to_target_cartesian'][0],
                    vector['cel_to_target_cartesian'][1],
                    vector['cel_to_target_cartesian'][2],
                    color='red',
                    label=f"End Time Local: {local_time_end}",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    slope_2[0],
                    slope_2[1],
                    slope_2[2],
                    color='darkorange',
                    label="Slope Option 2",
                    arrow_length_ratio=0.1,
                )
                inset_ax.scatter(inter_2[0], inter_2[1], inter_2[2], c='darkorange', label="Intersection 2")

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 0])
        ax.set_zlim([0, 1])
        ax.set_xlabel('X - East is Positive')
        ax.set_ylabel('Y - North is Positive')
        ax.set_zlabel('Z - Zenith is Positive')
        ax.set_title(f'Mixel {pixel} Bright to Dark on Unit Sphere with {celestial_object.capitalize()}')
        ax.set_aspect('equal')
        ax.legend(loc='lower center')

        inset_ax.set_xlim([np.min(ins_x_lim) - 0.01, np.max(ins_x_lim) + 0.01])
        inset_ax.set_ylim([np.min(ins_y_lim) - 0.01, np.max(ins_y_lim) + 0.01])
        inset_ax.set_zlim([np.min(ins_z_lim) - 0.01, np.max(ins_z_lim) + 0.01])
        inset_ax.set_xticks([np.round(np.min(ins_x_lim), 2), np.round(np.max(ins_x_lim), 2)])
        inset_ax.set_yticks([np.round(np.min(ins_y_lim), 2), np.round(np.max(ins_y_lim), 2)])
        inset_ax.set_zticks([np.round(np.min(ins_z_lim), 2), np.round(np.max(ins_z_lim), 2)])
        inset_ax.set_title("Zoomed Projection")
        inset_ax.set_xlabel("X")
        inset_ax.set_ylabel("Y")
        inset_ax.set_zlabel("Z")

        if os.path.isdir(os.path.join(output_folder_img, str(height))):
            plt.savefig(plot_file)
            plt.close()
        else:
            os.makedirs(os.path.join(output_folder_img, str(height)), exist_ok=True)
            plt.savefig(plot_file)
            plt.close()

        plotted_pixels.append(pixel)

    return plotted_pixels


def plotting_pixel_transition_vectors_decoupled_mp(
    data_location,
    checkpoint_folder,
    checkpoint_data_file,
    checkpoint_plot_file,
    celestial_object,
    output_folder_img,
    batch_size=5000,
):
    """
    Function to process pixels in batches, skipping already processed pixels.
    """
    os.makedirs(output_folder_img, exist_ok=True)

    # Load checkpoint data and plots
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_data_file)
    checkpoint_plots = lbt.load_checkpoint(checkpoint_folder, checkpoint_plot_file)
    if checkpoint_plots is None:
        checkpoint_plots = {"plotted_pixels": []}
    data = lbt.read_compressed_json(data_location)

    # Get the list of all pixels and filter out already processed ones
    all_pixels = list(data.keys())
    processed_pixels = checkpoint_plots["plotted_pixels"]
    unprocessed_pixels = [pixel for pixel in all_pixels if pixel not in processed_pixels]
    total_pixels = len(all_pixels)

    # Initialize progress bar
    progress_bar = tqdm(total=total_pixels, desc="Creating 3D Pixel Vector Plots")
    progress_bar.update(len(processed_pixels))  # Update progress bar for already processed pixels

    # Process pixels in batches
    for i in range(0, len(unprocessed_pixels), batch_size):
        batch_pixels = unprocessed_pixels[i : i + batch_size]

        with Pool() as pool:
            results = pool.starmap(
                plot_pixel_batch, [(batch_pixels, data, checkpoint_data, output_folder_img, celestial_object)]
            )

            for plotted_pixels in results:
                checkpoint_plots["plotted_pixels"].extend(plotted_pixels)
                lbt.save_checkpoint(checkpoint_folder, checkpoint_plot_file, checkpoint_plots)
                progress_bar.update(len(plotted_pixels))

    progress_bar.close()


def trilaterate(positions: np.ndarray, radii: np.ndarray, raise_on_no_solution: bool = False) -> np.ndarray:
    """
    Trilateration algorithm to find the intersection points of three spheres in 3D space.

    :param positions: A 3x3 array where each row represents the (x, y, z) coordinates of a sphere's center.
    :param radii: A 3-element array representing the radius of each sphere.
    :param raise_on_no_solution: If True, raises an error when no exact solution exists. If False, returns a close-enough solution.
    :return: A 2x3 array of intersection points (two solutions), or a 1x3 array if there is only one solution or no exact solution.

    Notes:
    - This implementation is based on the mathematical derivation of trilateration.
    - Modified from https://stackoverflow.com/a/18654302/313768
    """
    # Extract radii and positions
    radius1, radius2, radius3 = radii
    center1, center2, center3 = positions

    # Step 1: Compute inter-point vectors
    vector21 = center2 - center1  # Vector from center1 to center2
    vector31 = center3 - center1  # Vector from center1 to center3
    distance12 = np.linalg.norm(vector21)  # Distance between center1 and center2
    distance13 = np.linalg.norm(vector31)  # Distance between center1 and center3
    distance23 = np.linalg.norm(center3 - center2)  # Distance between center2 and center3

    # Step 2: Check for degenerate cases (e.g., overlapping spheres)
    if np.allclose(center1, center2) and np.isclose(radius1, radius2):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")
    if np.allclose(center1, center3) and np.isclose(radius1, radius3):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")
    if np.allclose(center2, center3) and np.isclose(radius2, radius3):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")

    # Check for non-overlapping spheres
    if distance12 > radius1 + radius2 or distance12 < abs(radius1 - radius2):
        raise ValueError("Spheres 1 and 2 do not overlap, no solution exists.")
    if distance13 > radius1 + radius3 or distance13 < abs(radius1 - radius3):
        raise ValueError("Spheres 1 and 3 do not overlap, no solution exists.")
    if distance23 > radius2 + radius3 or distance23 < abs(radius2 - radius3):
        # Compute the closest point on the unit sphere to the line segment connecting centers of spheres 2 and 3
        vector23 = center3 - center2
        unit_vector23 = vector23 / np.linalg.norm(vector23)  # Unit vector along vector23
        midpoint = center2 + unit_vector23 * (distance23 / 2)  # Midpoint of the line segment
        closest_point_on_unit_sphere = midpoint / np.linalg.norm(midpoint)  # Project midpoint onto the unit sphere
        logger.info(
            "Spheres 2 and 3 do not overlap. A single solution between the two spheres is returned for both solutions"
        )
        return np.stack(
            (closest_point_on_unit_sphere, closest_point_on_unit_sphere)
        )  # Return as a single point for both intersections
        # raise ValueError("Spheres 2 and 3 do not overlap, no solution exists.")

    # Step 3: Compute basis vectors for the coordinate system
    unit_vector_u = vector21 / distance12  # Unit vector along vector21
    projection_i = unit_vector_u.dot(vector31)  # Projection of vector31 onto unit_vector_u
    vector_v = vector31 - unit_vector_u * projection_i  # Orthogonal component of vector31
    vector_v /= np.linalg.norm(vector_v)  # Normalize vector_v
    projection_j = vector_v.dot(vector31)  # Projection of vector31 onto vector_v
    unit_vector_w = np.cross(unit_vector_u, vector_v)  # Unit vector orthogonal to both unit_vector_u and vector_v

    # Step 4: Solve for the x and y coordinates in the projected space
    x = 0.5 / distance12 * (radius1**2 - radius2**2 + distance12**2)
    y = 0.5 / projection_j * (radius1**2 - radius3**2 - 2 * projection_i * x + projection_i**2 + projection_j**2)
    radicand = radius1**2 - x**2 - y**2  # Compute the radicand for the z-coordinate

    # Step 5: Handle cases where the radicand is negative (no exact solution)
    if radicand < 0:
        logger.debug("Negative radicand %f: no exact solution exists", radicand)
        if raise_on_no_solution:
            raise ValueError(f"Negative radicand {radicand}: no exact solutions exist.")
        return (center1 + unit_vector_u * x + vector_v * y)[np.newaxis, :]  # Return a close-enough solution

    # Step 6: Compute the z-coordinate and intersection points
    z = np.sqrt(radicand)  # Compute the z-coordinate
    offset_z = unit_vector_w * z  # Offset in the z-direction
    solution_a = center1 + unit_vector_u * x + vector_v * y + offset_z  # First intersection point
    solution_b = center1 + unit_vector_u * x + vector_v * y - offset_z  # Second intersection point

    # Step 7: Return the solutions
    return np.stack((solution_a, solution_b))  # Return both intersection points as a 2x3 array


def maximum_frame_range(frame_ranges):
    """
    Calculates the maximum range (duration) between sequential frames
    where the transition type alternates between bright (1) and dark (0).

    Parameters:
        frame_ranges (list of tuples): Each tuple contains (transition_type, frame_number).
                                       transition_type is 1 for bright and 0 for dark.
                                       frame_number is an integer representing the frame number.

    Returns:
        tuple: A tuple containing:
            - max_range_frames (list): The two tuples representing the start and end of the maximum range.
            - max_duration (int): The maximum duration between sequential frames.
    """
    # Ensure the input list is sorted by frame_number
    frame_ranges = sorted(frame_ranges, key=lambda x: x[1])

    # Initialize variables to track the maximum duration and corresponding frame range
    max_duration = 0
    max_range_frames = None

    # Iterate through the sorted list to calculate differences between sequential frames
    for i in range(len(frame_ranges) - 1):
        current_frame = frame_ranges[i]
        next_frame = frame_ranges[i + 1]

        # Check if the transition types alternate (bright -> dark or dark -> bright)
        if current_frame[0] != next_frame[0]:
            # Calculate the duration between the current and next frame
            duration = abs(next_frame[1] - current_frame[1])

            # Update the maximum duration and corresponding frame range if needed
            if duration > max_duration:
                max_duration = duration
                max_range_frames = [current_frame, next_frame]

    return max_range_frames, max_duration
