import os
import time
import numpy as np
from logging import DEBUG, ERROR
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import ast


import opencsp.app.sofast.lib.image_processing as imgp
import opencsp.app.lookback.lookback_tools as lbt

from opencsp.common.lib.camera.Camera import Camera

from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render.view_spec as vs
import opencsp.app.sofast.lib.spatial_processing as sp
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_lookfast_camera_adjust.txt",
    log_type=DEBUG,
)


def rotation_matrix_scipy(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2 using SciPy.
    """
    # SciPy works with 1D arrays for single vectors
    a = vec1.reshape(-1)
    b = vec2.reshape(-1)

    # The align_vectors method returns a Rotation object and a rmsd value
    rotation_object, rssd = Rotation.align_vectors(b, a)  # Note: order may need adjustment based on specific use case

    # Convert the Rotation object to a 3x3 matrix
    # rotation_matrix = rotation.as_matrix()

    return rotation_object, rssd


def rotate_vector(vector, axis, degrees):
    """
    Rotates a vector about a specified axis by a given number of degrees.

    Parameters:
        vector (numpy.ndarray): The vector to rotate (1D array of shape (3,)).
        axis (numpy.ndarray): The axis to rotate about (1D array of shape (3,)).
        degrees (float): The angle in degrees to rotate the vector.

    Returns:
        numpy.ndarray: The rotated vector (1D array of shape (3,)).
    """
    try:
        # Validate inputs
        if not isinstance(vector, np.ndarray) or vector.shape != (3,):
            logger.error("Invalid vector: Must be a numpy array of shape (3,).")
            raise ValueError("Vector must be a numpy array of shape (3,).")

        if not isinstance(axis, np.ndarray) or axis.shape != (3,):
            logger.error("Invalid axis: Must be a numpy array of shape (3,).")
            raise ValueError("Axis must be a numpy array of shape (3,).")

        if not isinstance(degrees, (int, float)):
            logger.error("Invalid degrees: Must be a numeric value.")
            raise ValueError("Degrees must be a numeric value.")

        # Normalize the axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            logger.error("Invalid axis: Cannot be a zero vector.")
            raise ValueError("Axis cannot be a zero vector.")
        axis = axis / axis_norm

        # Convert degrees to radians
        radians = np.deg2rad(degrees)

        # Compute rotation matrix using Rodrigues' rotation formula
        cos_theta = np.cos(radians)
        sin_theta = np.sin(radians)
        one_minus_cos = 1 - cos_theta

        # Outer product of axis with itself
        axis_outer = np.outer(axis, axis)

        # Cross-product matrix of axis
        axis_cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        # Rotation matrix
        rotation_matrix = cos_theta * np.eye(3) + one_minus_cos * axis_outer + sin_theta * axis_cross

        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector

    except Exception as e:
        logger.exception("An error occurred while rotating the vector.")
        raise e


def create_rotation_object(axis, angle_degrees):
    """
    Create a scipy Rotation object for a rotation about a specified axis.

    Parameters:
        axis (array-like): A 3D vector specifying the axis of rotation (e.g., [1, 0, 0]).
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        scipy.spatial.transform.Rotation: A Rotation object representing the rotation.
    """
    # Normalize the axis to ensure it is a unit vector
    axis = np.array(axis)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")
    axis_normalized = axis / np.linalg.norm(axis)

    # Create the rotation object using the axis-angle representation
    rotation = Rotation.from_rotvec(np.radians(angle_degrees) * axis_normalized)

    return rotation


'''
def binary_search_angle(vector, axis, function, tolerance=1e-6, max_iterations=1000):
    """
    Performs a binary search to find the angle that minimizes the z-component of the rotated vector.

    Parameters:
        vector (numpy.ndarray): The vector to rotate (1D array of shape (3,)).
        axis (numpy.ndarray): The axis to rotate about (1D array of shape (3,)).
        rotate_vector (function): Function to rotate the vector.
        tolerance (float): The tolerance for the z-component to be considered zero.
        max_iterations (int): Maximum number of iterations for the binary search.

    Returns:
        float: The angle in degrees that minimizes the z-component of the rotated vector.
    """
    # Define the search range for the angle (0 to 360 degrees)
    low = 0.0
    high = 360.0

    for iteration in range(max_iterations):
        # Calculate the midpoint angle
        mid = (low + high) / 2.0

        # Rotate the vector at the midpoint angle
        rotated_vector = function(vector, axis, mid)

        # Check the z-component of the rotated vector
        z_component = rotated_vector[2]

        if abs(z_component) < tolerance:
            # If the z-component is close to zero, return the angle
            return mid

        # Update the search range based on the sign of the z-component
        if z_component > 0:
            high = mid
        else:
            low = mid

    # If the search did not converge, return the midpoint of the final range
    return (low + high) / 2.0

'''


def binary_search_angle_select(
    vector, axis, function, component_index=2, tolerance=1e-6, max_iterations=1000, num_initial_samples=36
):
    """
    Performs a binary search to find the angle that minimizes the user-selected component of the rotated vector.

    Parameters:
        vector (numpy.ndarray): The vector to rotate (1D array of shape (3,)).
        axis (numpy.ndarray): The axis to rotate about (1D array of shape (3,)).
        function (callable): Function to rotate the vector.
        component_index (int): The index of the component to minimize (0 for x, 1 for y, 2 for z).
        tolerance (float): The tolerance for the selected component to be considered zero.
        max_iterations (int): Maximum number of iterations for the binary search.
        num_initial_samples (int): Number of initial angles to sample for the linear search.

    Returns:
        float: The angle in degrees that minimizes the specified component of the rotated vector.
    """
    # Initial linear search from -180 to 180 degrees
    angles = np.linspace(-180, 180, num_initial_samples)
    best_angle = None
    best_value = float('inf')

    for angle in angles:
        rotated_vector = function(vector, axis, angle)
        component_value = rotated_vector[component_index]

        # Update best angle and value
        if abs(component_value) < abs(best_value):
            best_value = component_value
            best_angle = angle

    # Determine the two angles around the best angle for binary search
    low = best_angle - 1  # Start just below the best angle
    high = best_angle + 1  # Start just above the best angle

    for iteration in range(max_iterations):
        # Calculate the midpoint angle
        mid = (low + high) / 2.0

        # Rotate the vector at the midpoint angle
        rotated_vector = function(vector, axis, mid)

        # Check the specified component of the rotated vector
        component_value = rotated_vector[component_index]

        if abs(component_value) < tolerance:
            # If the component is close to zero, return the angle
            return mid

        # Update the search range based on the sign of the component value
        if component_value > 0:
            high = mid
        else:
            low = mid

    # If the search did not converge, return the midpoint of the final range
    return (low + high) / 2.0


def find_zero_component_angle_bisect_strict(
    vector, axis, function, component_index=2, tolerance=1e-6, max_iterations=5000, num_initial_samples=721
):
    angles = np.linspace(-180.0, 180.0, num_initial_samples)
    vals = np.array([function(vector, axis, a)[component_index] for a in angles])

    # Find sign changes (root brackets)
    s = np.sign(vals)
    sign_changes = np.where(s[:-1] * s[1:] < 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError("Selected component never crosses zero; no solution exists.")

    # Pick bracket that contains the smallest |value| sample (good heuristic)
    k_best = np.argmin(np.abs(vals))
    idx = sign_changes[np.argmin(np.abs(sign_changes - k_best))]

    low, high = angles[idx], angles[idx + 1]
    f_low, f_high = vals[idx], vals[idx + 1]

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        f_mid = function(vector, axis, mid)[component_index]

        if abs(f_mid) < tolerance:
            return mid

        if np.sign(f_low) * np.sign(f_mid) < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    # If we got here, we still have a bracket but didn't hit tolerance fast enough
    mid = 0.5 * (low + high)
    f_mid = function(vector, axis, mid)[component_index]
    raise RuntimeError(f"Bisection did not reach tolerance. Final |component|={abs(f_mid)} at angle {mid} deg.")


def _rotate_with_scipy(vec, axis, angle_degrees):
    """Helper rotate function compatible with your binary_search_angle_select signature."""
    r = create_rotation_object(axis, angle_degrees)
    return r.apply(np.asarray(vec))


def select_rotation_with_direction_check(
    vector_to_minimize,
    axis,
    component_index,
    reference_vector,
    reference_component_index=None,
    desired_reference_sign=+1,
    *,
    tolerance=1e-6,
    max_iterations=10000,
    num_initial_samples=721,
    function=_rotate_with_scipy,
):
    """
    Wrapper that resolves the 'two-solution' ambiguity by checking a rotated reference vector's component sign.

    Steps:
      1) Find angle that makes vector_to_minimize[component_index] ~ 0 (your binary search).
      2) Consider the two equivalent solutions: theta and theta+180 (about same axis).
      3) Build Rotation objects for both; rotate reference_vector; pick the one whose chosen component has desired sign.

    Parameters
    ----------
    vector_to_minimize : (3,) array-like
        Vector whose selected component you want minimized to ~0.
    axis : (3,) array-like
        Rotation axis.
    component_index : int
        Component index (0/1/2) to drive to ~0 for vector_to_minimize.
    reference_vector : (3,) array-like
        A second vector used to disambiguate which of the two rotations you want.
    reference_component_index : int or None
        Which component of the rotated reference_vector to test. If None, uses component_index.
    desired_reference_sign : {+1, -1}
        Desired sign for the selected component of the rotated reference vector.
    function : callable
        Rotation function used by your binary_search_angle_select. Defaults to scipy-based helper.

    Returns
    -------
    angle_degrees : float
        Selected angle in degrees.
    rotation : scipy.spatial.transform.Rotation
        Rotation object for the selected angle.
    debug : dict
        Useful diagnostics (candidate angles and rotated vectors/components).
    """
    axis = np.asarray(axis, dtype=float)
    vmin = np.asarray(vector_to_minimize, dtype=float)
    vref = np.asarray(reference_vector, dtype=float)

    if reference_component_index is None:
        reference_component_index = component_index
    if desired_reference_sign not in (+1, -1):
        raise ValueError("desired_reference_sign must be +1 or -1.")

    # 1) Base solution from your minimization search
    theta = find_zero_component_angle_bisect_strict(
        vmin,
        axis,
        function=function,
        component_index=component_index,
        tolerance=tolerance,
        max_iterations=max_iterations,
        num_initial_samples=num_initial_samples,
    )

    # 2) Two candidate solutions (commonly separated by 180° for this kind of constraint)
    candidates = [theta, theta + 180.0]

    # 3) Evaluate candidates using the reference-vector sign rule
    results = []
    for ang in candidates:
        rot = create_rotation_object(axis, ang)
        vmin_rot = rot.apply(vmin)
        vref_rot = rot.apply(vref)

        ref_comp = vref_rot[reference_component_index]
        # Score: prefer correct sign; tie-break by magnitude (more positive if desired +1, more negative if desired -1)
        sign_ok = (ref_comp >= 0) if desired_reference_sign == +1 else (ref_comp <= 0)
        score = (1 if sign_ok else 0, desired_reference_sign * ref_comp)

        results.append(
            dict(
                angle=ang,
                rotation=rot,
                vmin_rot=vmin_rot,
                vref_rot=vref_rot,
                ref_component=ref_comp,
                sign_ok=sign_ok,
                score=score,
            )
        )

    # Pick best by (sign_ok first, then component preference)
    best = max(results, key=lambda d: d["score"])

    debug = {
        "component_index_minimized": component_index,
        "reference_component_index": reference_component_index,
        "desired_reference_sign": desired_reference_sign,
        "candidates": [
            {
                "angle": r["angle"],
                "vmin_rot": r["vmin_rot"],
                "vref_rot": r["vref_rot"],
                "ref_component": r["ref_component"],
                "sign_ok": r["sign_ok"],
            }
            for r in results
        ],
    }

    return best["angle"], best["rotation"], debug


def ransac_average_direction(vectors, num_iterations=100, tolerance=0.00025):
    """
    Compute a RANSAC-style average direction from a set of 3D vectors.

    Parameters:
        vectors (list or np.ndarray): Array of shape (N, 3) containing [x, y, z] direction components.
        num_iterations (int): Number of RANSAC iterations to perform.
        tolerance (float): Angular tolerance (in radians) for consensus evaluation.

    Returns:
        np.ndarray: The "average" direction vector with the highest consensus.
    """
    # Ensure input is a NumPy array
    vectors = np.array(vectors)

    # Normalize all vectors to unit length
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    best_consensus_count = 0
    best_average_direction = None

    for _ in range(num_iterations):
        # Randomly sample a subset of vectors
        sample_indices = np.random.choice(
            len(normalized_vectors), size=round(len(normalized_vectors) / 3), replace=False
        )
        sample_vectors = normalized_vectors[sample_indices]

        # Compute the average direction of the sample
        average_direction = np.mean(sample_vectors, axis=0)
        average_direction /= np.linalg.norm(average_direction)  # Normalize to unit length

        # Compute angular distance between average direction and all vectors
        dot_products = np.dot(normalized_vectors, average_direction)
        angular_distances = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Clip to avoid numerical issues

        # Count how many vectors are within the tolerance
        consensus_count = np.sum(angular_distances < tolerance)

        # Update the best result if this iteration has higher consensus
        if consensus_count > best_consensus_count:
            best_consensus_count = consensus_count
            best_average_direction = average_direction

    return best_average_direction


def calculate_angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculates the angle (in radians) between two unit vectors in 3D space.

    Parameters:
        vector1 (np.ndarray): A 3D unit vector [x, y, z].
        vector2 (np.ndarray): A 3D unit vector [x, y, z].

    Returns:
        float: The angle between the two vectors in radians.
    """
    # Ensure the input vectors are unit vectors
    if not np.isclose(np.linalg.norm(vector1), 1.0):
        raise ValueError("vector1 is not a unit vector.")
    if not np.isclose(np.linalg.norm(vector2), 1.0):
        raise ValueError("vector2 is not a unit vector.")

    # Compute the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)

    # Clamp the dot product to the range [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle using arccos
    angle = np.arccos(dot_product)

    return angle


def calculate_slope(observer_vec_start, observer_vec_end, inter_1, inter_2):
    observer_vec = (observer_vec_start + observer_vec_end) / 2
    slope_1 = (observer_vec + inter_1) / np.linalg.norm(observer_vec + inter_1)
    slope_2 = (observer_vec + inter_2) / np.linalg.norm(observer_vec + inter_2)
    return observer_vec, slope_1, slope_2


def calculate_slope_differences(reference_vector, sample_vectors):
    """
    Compare a reference vector with multiple sample vectors and return the differences and errors in all 3 dimensions.

    Parameters:
        reference_vector (list or np.ndarray): A 3D vector [x, y, z] representing the reference.
        sample_vectors (list or np.ndarray): A list or array of 3D vectors [[x1, y1, z1], [x2, y2, z2], ...].

    Returns:
        lists: A list containing the differences and errors for each sample vector.
                  "differences": [[dx1, dy1, dz1], [dx2, dy2, dz2], ...],
                  "errors": [error1, error2, ...]
    """
    # Ensure inputs are numpy arrays for easier manipulation
    reference_vector = np.array(reference_vector)
    sample_vectors = np.array(sample_vectors)

    # Validate dimensions
    if reference_vector.shape != (3,):
        raise ValueError("Reference vector must be a 3D vector [x, y, z].")
    if sample_vectors.ndim != 2 or sample_vectors.shape[1] != 3:
        raise ValueError("Sample vectors must be a list or array of 3D vectors [[x, y, z], ...].")

    # Calculate differences
    differences = sample_vectors - reference_vector

    # Calculate errors (Euclidean distance)
    errors_mag = np.linalg.norm(differences, axis=1)

    return differences.tolist(), errors_mag.tolist()


def safe_calculate_slope(observer_vector_start, observer_vector_end, intersection_1, intersection_2):
    """
    Wrapper for calculate_slope with error handling and logging.
    """
    try:
        # Call the calculate_slope function
        observer_vec, slope_1, slope_2 = calculate_slope(
            observer_vector_start, observer_vector_end, intersection_1, intersection_2
        )
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


def get_pixel_pointing_vector(pixel_directions, row, col, imagewidth):
    # pixel_pointing[row * light_image.shape[1] + col
    temp = pixel_directions[int(row * imagewidth + col)]
    return temp.data


def plot_slope_heat_maps(data_dict):
    """
    Iterates through a dictionary with pixel coordinates as keys, extracts up to two sets of data,
    and plots contour plots for each set separately.

    Parameters:
        pixel_dict (dict): A dictionary where:
            - Keys are pixel coordinates (tuples) like (x, y).
            - Values are either:
                - Sub-dictionaries containing:
                    - "slope_1" or "slope_2": List of up to two sets of [nx, ny, nz].
                    - "angle_between": Float of angle between start and end vector.
                - Empty ndarray for pixels without data.

    Returns:
        None: Displays the contour plots for both sets of data.
    """
    # Initialize lists to store data for the first and second sets
    x_coords_set, y_coords_set, slope_set1, slope_set2, angle_between = [], [], [], [], []

    # Iterate through the dictionary
    for pixel, data in data_dict.items():
        if isinstance(data, dict):  # Check if the value is a sub-dictionary
            # Extract the first set of data
            if data["slope_1"].size < 3 or data["slope_2"].size < 3:
                continue
            elif len(data["slope_1"]) > 0:
                row, col = ast.literal_eval(pixel)
                x_coords_set.append(col)
                y_coords_set.append(row)
                slope_set1.append(data["slope_1"])
                slope_set2.append(data["slope_2"])
                angle_between.append(data["angle_between"])
        elif isinstance(data, np.ndarray) and len(data) == 0:  # Skip empty lists
            continue
        else:
            raise ValueError(f"Unexpected data format for pixel {pixel}: {data}")

    best_slope_1 = ransac_average_direction(np.array(slope_set1))
    best_slope_2 = ransac_average_direction(np.array(slope_set2))

    plot_angle_between_vectors(x_coords_set, y_coords_set, angle_between)

    plot_heat_maps_no_comparison(x_coords_set, y_coords_set, slope_set1, "Set 1")

    plot_heat_maps_no_comparison(x_coords_set, y_coords_set, slope_set2, "Set 2")

    plot_heat_maps(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1")

    plot_heat_maps(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2")

    plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1 Radians")

    plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2 Radians")


def plot_slope_heat_maps_horizonal(data_dict, output_dir):
    """
    Iterates through a dictionary with pixel coordinates as keys, extracts up to two sets of data,
    and plots contour plots for each set separately.

    Parameters:
        pixel_dict (dict): A dictionary where:
            - Keys are pixel coordinates (tuples) like (x, y).
            - Values are either:
                - Sub-dictionaries containing:
                    - "slope_1" or "slope_2": List of up to two sets of [nx, ny, nz].
                    - "angle_between": Float of angle between start and end vector.
                - Empty ndarray for pixels without data.

    Returns:
        None: Displays the contour plots for both sets of data.
    """
    # Initialize lists to store data for the first and second sets
    x_coords_set, y_coords_set, slope_set1, slope_set2, angle_between = [], [], [], [], []

    # Iterate through the dictionary
    for pixel, data in data_dict.items():
        if isinstance(data, dict):  # Check if the value is a sub-dictionary
            if data["intersection_1"].size > 0:
                # Extract the first set of data
                if data["H_slope_1_camera_corrected"].size < 3 or data["H_slope_2_camera_corrected"].size < 3:
                    continue
                elif len(data["H_slope_1_camera_corrected"]) > 0:
                    row, col = ast.literal_eval(pixel)
                    x_coords_set.append(col)
                    y_coords_set.append(row)
                    slope_set1.append(data["H_slope_1_camera_corrected"])
                    slope_set2.append(data["H_slope_2_camera_corrected"])
                    angle_between.append(data["angle_between"])
            else:
                continue
        elif isinstance(data, np.ndarray) and len(data) == 0:  # Skip empty lists
            continue
        else:
            raise ValueError(f"Unexpected data format for pixel {pixel}: {data}")

    best_slope_1 = ransac_average_direction(np.array(slope_set1))
    best_slope_2 = ransac_average_direction(np.array(slope_set2))

    plot_angle_between_vectors(
        x_coords_set,
        y_coords_set,
        angle_between,
        set_label="Horizonal_Original_Data",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps_no_comparison(
        x_coords_set,
        y_coords_set,
        slope_set1,
        set_label="Set_1_Horizonal_Camera_Vector_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps_no_comparison(
        x_coords_set,
        y_coords_set,
        slope_set2,
        set_label="Set_2_Horizonal_Camera_Vector_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps(
        x_coords_set,
        y_coords_set,
        slope_set1,
        best_slope_1,
        set_label="Set_1_Horizonal_Difference_Camera_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps(
        x_coords_set,
        y_coords_set,
        slope_set2,
        best_slope_2,
        set_label="Set_2_Horizonal_Difference_Camera_Corrected",
        output_dir=output_dir,
        render=False,
    )

    # plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1 Horizonal Radians")

    # plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2 Horizonal Radians")


def plot_slope_heat_maps_camera(data_dict, output_dir):
    """
    Iterates through a dictionary with pixel coordinates as keys, extracts up to two sets of data,
    and plots contour plots for each set separately.

    Parameters:
        pixel_dict (dict): A dictionary where:
            - Keys are pixel coordinates (tuples) like (x, y).
            - Values are either:
                - Sub-dictionaries containing:
                    - "slope_1" or "slope_2": List of up to two sets of [nx, ny, nz].
                    - "angle_between": Float of angle between start and end vector.
                - Empty ndarray for pixels without data.

    Returns:
        None: Displays the contour plots for both sets of data.
    """
    # Initialize lists to store data for the first and second sets
    x_coords_set, y_coords_set, slope_set1, slope_set2, angle_between = [], [], [], [], []

    # Iterate through the dictionary
    for pixel, data in data_dict.items():
        if isinstance(data, dict):  # Check if the value is a sub-dictionary
            if data["intersection_1"].size > 0:
                # Extract the first set of data
                if data["C_slope_1_camera_corrected"].size < 3 or data["C_slope_2_camera_corrected"].size < 3:
                    continue
                elif len(data["C_slope_1_camera_corrected"]) > 0:
                    row, col = ast.literal_eval(pixel)
                    x_coords_set.append(col)
                    y_coords_set.append(row)
                    slope_set1.append(data["C_slope_1_camera_corrected"])
                    slope_set2.append(data["C_slope_2_camera_corrected"])
                    angle_between.append(data["angle_between"])
            else:
                continue
        elif isinstance(data, np.ndarray) and len(data) == 0:  # Skip empty lists
            continue
        else:
            raise ValueError(f"Unexpected data format for pixel {pixel}: {data}")

    best_slope_1 = ransac_average_direction(np.array(slope_set1))
    best_slope_2 = ransac_average_direction(np.array(slope_set2))

    plot_angle_between_vectors(
        x_coords_set, y_coords_set, angle_between, set_label="Camera_Original_Data", output_dir=output_dir, render=False
    )

    plot_heat_maps_no_comparison(
        x_coords_set,
        y_coords_set,
        slope_set1,
        set_label="Set_1_CamCoords_Camera_Vector_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps_no_comparison(
        x_coords_set,
        y_coords_set,
        slope_set2,
        set_label="Set_2_CamCoords_Camera_Vector_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps(
        x_coords_set,
        y_coords_set,
        slope_set1,
        best_slope_1,
        set_label="Set_1_CamCoords_Difference_Camera_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps(
        x_coords_set,
        y_coords_set,
        slope_set2,
        best_slope_2,
        set_label="Set_2_CamCoords_Difference_Camera_Corrected",
        output_dir=output_dir,
        render=False,
    )

    # plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1 Camera Radians")

    # plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2 Camera Radians")


def plot_slope_heat_maps_mirror(data_dict, output_dir):
    """
    Iterates through a dictionary with pixel coordinates as keys, extracts up to two sets of data,
    and plots contour plots for each set separately.

    Parameters:
        pixel_dict (dict): A dictionary where:
            - Keys are pixel coordinates (tuples) like (x, y).
            - Values are either:
                - Sub-dictionaries containing:
                    - "slope_1" or "slope_2": List of up to two sets of [nx, ny, nz].
                    - "angle_between": Float of angle between start and end vector.
                - Empty ndarray for pixels without data.

    Returns:
        None: Displays the contour plots for both sets of data.
    """
    # Initialize lists to store data for the first and second sets
    x_coords_set, y_coords_set, slope_set1, slope_set2, angle_between = [], [], [], [], []

    # Iterate through the dictionary
    for pixel, data in data_dict.items():
        if isinstance(data, dict):  # Check if the value is a sub-dictionary
            if data["intersection_1"].size > 0:
                # Extract the first set of data
                if data["M_slope_1_camera_corrected"].size < 3 or data["M_slope_2_camera_corrected"].size < 3:
                    continue
                elif len(data["M_slope_1_camera_corrected"]) > 0:
                    row, col = ast.literal_eval(pixel)
                    x_coords_set.append(col)
                    y_coords_set.append(row)
                    slope_set1.append(data["M_slope_1_camera_corrected"])
                    slope_set2.append(data["M_slope_2_camera_corrected"])
                    angle_between.append(data["angle_between"])
            else:
                continue
        elif isinstance(data, np.ndarray) and len(data) == 0:  # Skip empty lists
            continue
        else:
            raise ValueError(f"Unexpected data format for pixel {pixel}: {data}")

    best_slope_1 = ransac_average_direction(np.array(slope_set1))
    best_slope_2 = ransac_average_direction(np.array(slope_set2))

    plot_angle_between_vectors(
        x_coords_set, y_coords_set, angle_between, set_label="Mirror_Original_Data", output_dir=output_dir, render=False
    )

    plot_heat_maps_no_comparison(
        x_coords_set,
        y_coords_set,
        slope_set1,
        set_label="Set_1_Mirror_Camera_Vector_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps_no_comparison(
        x_coords_set,
        y_coords_set,
        slope_set2,
        set_label="Set_2_Mirror_Camera_Vector_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps(
        x_coords_set,
        y_coords_set,
        slope_set1,
        best_slope_1,
        set_label="Set_1_Mirror_Difference_Camera_Corrected",
        output_dir=output_dir,
        render=False,
    )

    plot_heat_maps(
        x_coords_set,
        y_coords_set,
        slope_set2,
        best_slope_2,
        set_label="Set_2_Mirror_Difference_Camera_Corrected",
        output_dir=output_dir,
        render=False,
    )

    # plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1 Mirror Radians")

    # plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2 Mirror Radians")


def plot_angle_between_vectors(
    x_coords, y_coords, angles_between, set_label="angle_between_vectors", output_dir=None, render=False
):
    if output_dir:
        ft.create_directories_if_necessary(output_dir)
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
    )

    angles = griddata((x_coords, y_coords), angles_between, (grid_x, grid_y), method="linear")

    fig, ax = plt.subplots()
    contour = ax.contourf(grid_x, grid_y, angles, cmap="jet", levels=250)  # vmin=scale_min, vmax=scale_max
    cbar = plt.colorbar(contour, ax=ax)
    ax.set_xlabel("X Pixel Location")
    ax.set_ylabel("Y Pixel Location")
    ax.set_title("Coverage Map Type Plot Heat Map")
    cbar.set_label("Angle Between Vectors [Radians]")
    ax.invert_yaxis()
    if output_dir:
        file_path = ft.join(output_dir, set_label + ".png")
        plt.savefig(file_path)
    if not render:
        plt.close()


def plot_heat_maps_no_comparison(x_coords, y_coords, slopes, set_label, view_tup=None, output_dir=None, render=False):
    if output_dir:
        ft.create_directories_if_necessary(output_dir)
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    slopes = np.array(slopes)

    # scale_max = np.max([slopes])
    # scale_min = np.min([slopes])
    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
    )

    # Interpolate errors onto the grid
    error_grid = {
        "X Value": griddata((x_coords, y_coords), slopes[:, 0], (grid_x, grid_y), method="linear"),
        "Y Value": griddata((x_coords, y_coords), slopes[:, 1], (grid_x, grid_y), method="linear"),
        "Z Value": griddata((x_coords, y_coords), slopes[:, 2], (grid_x, grid_y), method="linear"),
    }

    # Plot each error type as a contour plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    error_types = ["X Value", "Y Value", "Z Value"]
    for ax, error_type in zip(axes.flat, error_types):
        contour = ax.contourf(
            grid_x, grid_y, error_grid[error_type], cmap="jet", levels=250  # , vmin=scale_min, vmax=scale_max
        )
        # ax.scatter(x_coords, y_coords, s=0.1, c='k', marker='.')  # Actual locations where we have data
        cbar = plt.colorbar(contour, ax=ax)
        ax.set_title(f"{error_type} Heat Map ({set_label})", wrap=True)
        ax.set_xlabel("X Pixel Location")
        ax.set_ylabel("Y Pixel Location")
        cbar.set_label(f"{error_type}")
        ax.invert_yaxis()

        if view_tup:
            ax.view_init(elev=view_tup[0], azim=view_tup[1])

    plt.tight_layout()
    if output_dir:
        if view_tup:
            file_path = ft.join(output_dir, set_label + f"_el{view_tup[0]}_az{view_tup[1]}.png")
        else:
            file_path = ft.join(output_dir, set_label + ".png")
        plt.savefig(file_path)
    if not render:
        plt.close()
    # plt.show()


def plot_heat_maps(x_coords, y_coords, slopes, best_slope, set_label, view_tup=None, output_dir=None, render=False):
    if output_dir:
        ft.create_directories_if_necessary(output_dir)
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    slopes = np.array(slopes)

    slope_differences, slope_diff_norms = calculate_slope_differences(best_slope, slopes)
    slope_differences = np.array(slope_differences)
    slope_diff_norms = np.array(slope_diff_norms)

    # scale_max = np.max([slope_diff_norms])
    # scale_min = np.min([slope_diff_norms])
    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
    )

    # Interpolate errors onto the grid
    error_grid = {
        "Norm Difference": griddata((x_coords, y_coords), slope_diff_norms, (grid_x, grid_y), method="linear"),
        "X Difference": griddata((x_coords, y_coords), slope_differences[:, 0], (grid_x, grid_y), method="linear"),
        "Y Difference": griddata((x_coords, y_coords), slope_differences[:, 1], (grid_x, grid_y), method="linear"),
        "Z Difference": griddata((x_coords, y_coords), slope_differences[:, 2], (grid_x, grid_y), method="linear"),
    }

    # Plot each error type as a contour plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    error_types = ["Norm Difference", "X Difference", "Y Difference", "Z Difference"]
    for ax, error_type in zip(axes.flat, error_types):
        contour = ax.contourf(
            grid_x, grid_y, error_grid[error_type], cmap="jet", levels=500  # , vmin=scale_min, vmax=scale_max
        )
        cbar = plt.colorbar(contour, ax=ax)
        ax.set_title(f"{error_type} Heat Map ({set_label})", wrap=True)
        ax.set_xlabel("X Pixel Location")
        ax.set_ylabel("Y Pixel Location")
        cbar.set_label(f"{error_type}")
        ax.invert_yaxis()

        if view_tup:
            ax.view_init(elev=view_tup[0], azim=view_tup[1])

    plt.tight_layout()
    if output_dir:
        if view_tup:
            file_path = ft.join(output_dir, set_label + f"_el{view_tup[0]}_az{view_tup[1]}.png")
        else:
            file_path = ft.join(output_dir, set_label + ".png")
        plt.savefig(file_path)
    if not render:
        plt.close()
    # plt.show()


def plot_heat_maps_radians(x_coords, y_coords, slopes, best_slope, set_label):
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    slopes = np.array(slopes)

    slope_deviation_radians = []
    for slope in slopes:
        slope_deviation_radians.append(calculate_angle_between_vectors(best_slope, slope))

    slope_deviation_radians = np.array(slope_deviation_radians)

    # scale_max = np.max([slope_deviation_radians])
    # scale_min = np.min([slope_deviation_radians])
    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
    )

    # Interpolate errors onto the grid
    error_grid = {
        "Radian Difference": griddata((x_coords, y_coords), slope_deviation_radians, (grid_x, grid_y), method="linear")
    }

    # Plot each error type as a contour plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    contour = ax.contourf(
        grid_x, grid_y, error_grid["Radian Difference"], cmap="jet", levels=500  # , vmin=scale_min, vmax=scale_max
    )
    cbar = plt.colorbar(contour, ax=ax)
    ax.set_title(f"Difference Heat Map ({set_label})")
    ax.set_xlabel("X Pixel Location")
    ax.set_ylabel("Y Pixel Location")
    cbar.set_label("Radian Difference")
    ax.invert_yaxis()

    plt.tight_layout()
    # plt.show()


def plot_scatter_heat_map(coordinates, slopes, set_label, view_tup=None, output_dir=None, render=False, invert_y=None):

    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    z_coords = coordinates[:, 2]
    x_vec = slopes[:, 0]
    y_vec = slopes[:, 1]
    z_vec = slopes[:, 2]

    # Create subplots
    fig = plt.figure(figsize=(18, 6))

    # Scatter plot for x_vec
    ax1 = fig.add_subplot(131, projection='3d')
    scatter_x = ax1.scatter(x_coords, y_coords, z_coords, c=x_vec, cmap='jet', s=1)
    ax1.set_title("X " + set_label)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate')
    fig.colorbar(scatter_x, ax=ax1)
    if invert_y is not None:
        ax1.invert_yaxis()

    # Scatter plot for y_vec
    ax2 = fig.add_subplot(132, projection='3d')
    scatter_y = ax2.scatter(x_coords, y_coords, z_coords, c=y_vec, cmap='jet', s=1)
    ax2.set_title("Y " + set_label)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('Z Coordinate')
    fig.colorbar(scatter_y, ax=ax2)
    if invert_y is not None:
        ax2.invert_yaxis()

    # Scatter plot for z_vec
    ax3 = fig.add_subplot(133, projection='3d')
    scatter_z = ax3.scatter(x_coords, y_coords, z_coords, c=z_vec, cmap='jet', s=1)
    ax3.set_title("Z " + set_label)
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_zlabel('Z Coordinate')
    fig.colorbar(scatter_z, ax=ax3)
    if invert_y is not None:
        ax3.invert_yaxis()

    if view_tup:
        ax1.view_init(elev=view_tup[0], azim=view_tup[1])
        ax2.view_init(elev=view_tup[0], azim=view_tup[1])
        ax3.view_init(elev=view_tup[0], azim=view_tup[1])

    # Adjust layout
    plt.tight_layout()
    if output_dir:
        if view_tup:
            file_path = ft.join(output_dir, set_label + f"_el{view_tup[0]}_az{view_tup[1]}.png")
        else:
            file_path = ft.join(output_dir, set_label + ".png")
        plt.savefig(file_path)
    if not render:
        plt.close()


def plot_scatter3d(coordinates, color, set_label, add_pts=False, view_tup=None, output_dir=None, render=False):
    """
    Plot (or add to) a 3D scatter plot.

    Parameters
    ----------
    coordinates : (N,3) array-like
    color : matplotlib color or array-like
    set_label : str
    add_pts : bool
        If True, add points to an existing axes (ax or current figure's 3D axes).
    view_tup : (elev, azim) or None
    output_dir : str or None
    render : bool
        If False, close the figure after saving (or after plotting).
    """
    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must be an (N, 3) array-like")

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # One-figure-at-a-time convention:
    # - if add_pts: reuse current figure/axes
    # - else: make a fresh figure/axes
    if add_pts:
        fig = plt.gcf()
        # If there's no axes yet, create one; otherwise use the first axes.
        if fig.get_axes():
            ax = fig.get_axes()[0]
        else:
            ax = fig.add_subplot(111, projection="3d")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, c=color, s=1)
    ax.set_title(set_label)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    if view_tup is not None:
        ax.view_init(elev=view_tup[0], azim=view_tup[1])

    # Adjust layout
    plt.tight_layout()
    if output_dir:
        if view_tup:
            file_path = ft.join(output_dir, set_label + f"_el{view_tup[0]}_az{view_tup[1]}.png")
        else:
            file_path = ft.join(output_dir, set_label + ".png")
        plt.savefig(file_path, bbox_inches="tight")
    if not render:
        plt.close()


def plot_heat_maps_horizonal_looking_up(vector_data, output_dir):
    x_coords_set, y_coords_set, slope_set1, slope_set2 = [], [], [], []
    for pixel, details in vector_data.items():
        if isinstance(details, dict):
            if details["intersection_1"].size > 0:
                row, col = ast.literal_eval(pixel)

                x_coords_set.append(col)
                y_coords_set.append(row)
                slope_set1.append(vector_data[pixel]["H_slope_1_camera_corrected"])
                slope_set2.append(vector_data[pixel]["H_slope_2_camera_corrected"])

            else:
                continue
        else:
            pass

    mean_direction_1 = np.mean(np.array(slope_set1), axis=0)
    mean_direction_2 = np.mean(np.array(slope_set2), axis=0)
    H_look_up_rot_1, _ = rotation_matrix_scipy(mean_direction_1, np.array([0, 0, 1]))
    H_look_up_rot_2, _ = rotation_matrix_scipy(mean_direction_2, np.array([0, 0, 1]))
    H_slope_look_up_1, H_slope_look_up_2 = [], []

    H_slope_look_up_1 = H_look_up_rot_1.apply(np.array(slope_set1))
    H_slope_look_up_2 = H_look_up_rot_2.apply(np.array(slope_set2))

    plot_heat_maps_no_comparison(
        x_coords_set,
        y_coords_set,
        H_slope_look_up_1,
        set_label="Set_1_Horz_Mean_Vec_Rot_to_UP",
        output_dir=output_dir,
        render=False,
    )
    plot_heat_maps_no_comparison(
        x_coords_set,
        y_coords_set,
        H_slope_look_up_2,
        set_label="Set_2_Horz_Mean_Vec_Rot_to_UP",
        output_dir=output_dir,
        render=False,
    )


def plot_heat_maps_camera_looking_up_coords(vector_data, control, output_dir, debug_plots=True):
    x_pix_set, y_pix_set, C_coords, slope_set1, slope_set2 = [], [], [], [], []
    for pixel, details in vector_data.items():
        if isinstance(details, dict):
            if details["intersection_1"].size > 0:
                row, col = ast.literal_eval(pixel)

                x_pix_set.append(col)
                y_pix_set.append(row)
                C_coords.append(vector_data[pixel]["C_point_location"])
                slope_set1.append(vector_data[pixel]["C_slope_1_camera_corrected"])
                slope_set2.append(vector_data[pixel]["C_slope_2_camera_corrected"])

            else:
                continue
        else:
            pass

    C_coords = np.array(C_coords).squeeze(axis=1)
    C_xyz_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mean_direction_1 = np.mean(np.array(slope_set1), axis=0)
    mean_direction_1 = mean_direction_1 / np.linalg.norm(mean_direction_1)
    mean_direction_2 = np.mean(np.array(slope_set2), axis=0)
    mean_direction_2 = mean_direction_2 / np.linalg.norm(mean_direction_2)
    C_look_axis_rot_1, _ = rotation_matrix_scipy(mean_direction_1, np.array([0, 0, -1]))  # Rotation
    C_look_axis_rot_2, _ = rotation_matrix_scipy(mean_direction_2, np.array([0, 0, -1]))  # Rotation
    mean_direction_r_1 = C_look_axis_rot_1.apply(mean_direction_1)
    mean_direction_r_2 = C_look_axis_rot_2.apply(mean_direction_2)
    C_xyz_axis_r_1 = C_look_axis_rot_1.apply(C_xyz_axis)
    C_xyz_axis_r_2 = C_look_axis_rot_2.apply(C_xyz_axis)
    ctrl_r_1 = apply_rt_to_control(control, C_look_axis_rot_1, None)
    ctrl_r_2 = apply_rt_to_control(control, C_look_axis_rot_2, None)
    C_slope_look_axis_1, C_slope_look_axis_2, C_coords_look_axis_1, C_coords_look_axis_2 = [], [], [], []

    C_slope_look_axis_1 = C_look_axis_rot_1.apply(np.array(slope_set1))
    C_coords_look_axis_1 = C_look_axis_rot_1.apply(C_coords)
    C_slope_look_axis_2 = C_look_axis_rot_2.apply(np.array(slope_set2))
    C_coords_look_axis_2 = C_look_axis_rot_2.apply(C_coords)

    if debug_plots:
        plot_scatter_heat_map(
            C_coords,
            np.array(slope_set1),
            set_label="Set_1_Cam_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            C_coords[0],
            C_xyz_axis,
            color_sequence=None,
            arrow_length=2,
            set_label="Set_1_Cam_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            C_coords[round(len(C_coords) / 2)],
            mean_direction_1,
            color='orange',
            arrow_length=2,
            set_label="Set_1_Cam_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            C_coords[round(len(C_coords) / 2)],
            mean_direction_r_1,
            color='black',
            arrow_length=2,
            set_label="Set_1_Cam_Vec_Coords",
            # view_tup=(30, 135),
            output_dir=output_dir,
            render=False,
        )

        plot_scatter_heat_map(
            C_coords,
            np.array(slope_set2),
            set_label="Set_2_Cam_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            C_coords[0],
            C_xyz_axis,
            color_sequence=None,
            arrow_length=2,
            set_label="Set_2_Cam_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            C_coords[round(len(C_coords) / 2)],
            mean_direction_2,
            color='orange',
            arrow_length=2,
            set_label="Set_2_Cam_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            C_coords[round(len(C_coords) / 2)],
            mean_direction_r_2,
            color='black',
            arrow_length=2,
            set_label="Set_2_Cam_Vec_Coords",
            # view_tup=(30, 135),
            output_dir=output_dir,
            render=False,
        )

        plot_scatter_heat_map(
            C_coords_look_axis_1,
            C_slope_look_axis_1,
            set_label="Set_1_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            C_coords_look_axis_1[0],
            C_xyz_axis,
            color_sequence=None,
            arrow_length=2,
            set_label="Set_1_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            C_coords_look_axis_1[0],
            C_xyz_axis_r_1,
            color_sequence=["y", "m", "c"],
            arrow_length=2,
            set_label="Set_1_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            C_coords_look_axis_1[round(len(C_coords_look_axis_1) / 2)],
            mean_direction_1,
            color='orange',
            arrow_length=2,
            set_label="Set_1_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            C_coords_look_axis_1[round(len(C_coords_look_axis_1) / 2)],
            mean_direction_r_1,
            color='black',
            arrow_length=2,
            set_label="Set_1_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            output_dir=output_dir,
            render=True,
        )

        plot_scatter_heat_map(
            C_coords_look_axis_2,
            C_slope_look_axis_2,
            set_label="Set_2_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            C_coords_look_axis_2[0],
            C_xyz_axis,
            color_sequence=None,
            arrow_length=2,
            set_label="Set_2_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            C_coords_look_axis_2[0],
            C_xyz_axis_r_2,
            color_sequence=["y", "m", "c"],
            arrow_length=2,
            set_label="Set_2_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            C_coords_look_axis_2[round(len(C_coords_look_axis_2) / 2)],
            mean_direction_2,
            color='orange',
            arrow_length=2,
            set_label="Set_2_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            C_coords_look_axis_2[round(len(C_coords_look_axis_2) / 2)],
            mean_direction_r_2,
            color='black',
            arrow_length=2,
            set_label="Set_2_Cam_Look_Opt_Axis",
            # view_tup=(30, 135),
            output_dir=output_dir,
            render=False,
        )

    C_coords_look_axis_1_translate = C_coords_look_axis_1 - np.mean(C_coords_look_axis_1, axis=0)  # Translation
    C_coords_look_axis_2_translate = C_coords_look_axis_2 - np.mean(C_coords_look_axis_2, axis=0)  # Translation

    ctrl_rt_1 = apply_rt_to_control(ctrl_r_1, Rotation.identity(), -1 * np.mean(C_coords_look_axis_1, axis=0))
    ctrl_rt_2 = apply_rt_to_control(ctrl_r_2, Rotation.identity(), -1 * np.mean(C_coords_look_axis_2, axis=0))

    C_coords_xy_rot_1 = transform_to_xy_plane(
        [
            C_coords_look_axis_1_translate[0, :],
            C_coords_look_axis_1_translate[-1, :],
            C_coords_look_axis_1_translate[1000, :],
        ],
        z_orientation=-1,
        reference_normal=ctrl_rt_1["normal"],
    )  # Rotation for Coordinates Only
    C_coords_xy_rot_2 = transform_to_xy_plane(
        [
            C_coords_look_axis_2_translate[0, :],
            C_coords_look_axis_2_translate[-1, :],
            C_coords_look_axis_2_translate[1000, :],
        ],
        z_orientation=-1,
        reference_normal=ctrl_rt_2["normal"],
    )  # Rotation for Coordinates Only

    ctrl_xy_1 = apply_rt_to_control(ctrl_rt_1, C_coords_xy_rot_1, None)
    ctrl_xy_2 = apply_rt_to_control(ctrl_rt_2, C_coords_xy_rot_2, None)

    C_xyz_axis_rr_1 = C_coords_xy_rot_1.apply(C_xyz_axis_r_1)
    C_xyz_axis_rr_2 = C_coords_xy_rot_2.apply(C_xyz_axis_r_2)

    C_coords_xy_1 = C_coords_xy_rot_1.apply(C_coords_look_axis_1)
    # C_slopes_xy_1 = C_coords_xy_rot_1.apply(C_slope_look_axis_1)
    C_coords_xy_2 = C_coords_xy_rot_2.apply(C_coords_look_axis_2)
    # C_slopes_xy_2 = C_coords_xy_rot_2.apply(C_slope_look_axis_2)
    '''
    plot_scatter_heat_map(C_coords_xy_1, C_slopes_xy_1, "Set 1 Camera Looking Anti Optical Axis XY Plane")
    add_3d_axes_to_plot(C_coords_xy_1[0], C_xyz_axis)
    add_3d_axes_to_plot(C_coords_xy_1[0], C_xyz_axis_rr_1, color_sequence=["y", "m", "c"])
    

    x_align_angle_1 = binary_search_angle_select(
        C_xyz_axis_rr_1[0],
        C_xyz_axis[2],
        rotate_vector,
        component_index=1,
        tolerance=1e-6,
        max_iterations=5000,
        num_initial_samples=36,
    )
    x_align_rot_obj_1 = create_rotation_object(axis=C_xyz_axis[2], angle_degrees=x_align_angle_1)  # Rotation

    ctrl_xy_align_1 = apply_rt_to_control(ctrl_xy_1, x_align_rot_obj_1, None)
    C_xyz_axis_rrr_1 = x_align_rot_obj_1.apply(C_xyz_axis_rr_1)

    # add_3d_axes_to_plot(C_coords_xy_1[0], C_xyz_axis_rrr_1, color_sequence=["black", "gray", "purple"], arrow_length=3)
    
    plot_scatter_heat_map(C_coords_xy_2, C_slopes_xy_2, "Set 2 Camera Looking Anti Optical Axis XY Plane")
    add_3d_axes_to_plot(C_coords_xy_2[0], C_xyz_axis)
    add_3d_axes_to_plot(C_coords_xy_2[0], C_xyz_axis_rr_2, color_sequence=["y", "m", "c"])
    

    x_align_angle_2 = binary_search_angle_select(
        C_xyz_axis_rr_2[0],
        C_xyz_axis[2],
        rotate_vector,
        component_index=1,
        tolerance=1e-6,
        max_iterations=5000,
        num_initial_samples=36,
    )
    x_align_rot_obj_2 = create_rotation_object(axis=C_xyz_axis[2], angle_degrees=x_align_angle_2)  # Rotation

    ctrl_xy_align_2 = apply_rt_to_control(ctrl_xy_2, x_align_rot_obj_2, None)
    C_xyz_axis_rrr_2 = x_align_rot_obj_2.apply(C_xyz_axis_rr_1)
    # add_3d_axes_to_plot(C_coords_xy_1[0], C_xyz_axis_rrr_1, color_sequence=["black", "gray", "purple"], arrow_length=3)

    
    C_coords_xy_align_1 = x_align_rot_obj_1.apply(C_coords_xy_1)
    # C_slopes_xy_align_1 = x_align_rot_obj_1.apply(C_slopes_xy_1)
    C_slopes_xy_align_1 = x_align_rot_obj_1.apply(C_slope_look_axis_1)

    C_coords_xy_align_2 = x_align_rot_obj_2.apply(C_coords_xy_2)
    # C_slopes_xy_align_2 = x_align_rot_obj_2.apply(C_slopes_xy_2)
    C_slopes_xy_align_2 = x_align_rot_obj_2.apply(C_slope_look_axis_2)
    
    if debug_plots:
        plot_scatter_heat_map(
            C_coords_xy_align_1, C_slopes_xy_align_1, set_label="Set_1_Cam_XY_Aligned", view_tup=(90, 270), render=True
        )
        add_3d_axes_to_plot(
            C_coords_xy_align_1[0],
            C_xyz_axis,
            color_sequence=None,
            set_label="Set_1_Cam_XY_Aligned",
            view_tup=(90, 270),
            render=True,
        )
        add_3d_axes_to_plot(
            C_coords_xy_align_1[0],
            C_xyz_axis_rrr_1,
            color_sequence=["black", "gray", "purple"],
            arrow_length=3,
            set_label="Set_1_Cam_XY_Aligned",
            output_dir=output_dir,
            render=False,
        )

        plot_scatter_heat_map(
            C_coords_xy_align_2, C_slopes_xy_align_2, set_label="Set_2_Cam_XY_Aligned", view_tup=(90, 270), render=True
        )
        add_3d_axes_to_plot(
            C_coords_xy_align_2[0],
            C_xyz_axis,
            color_sequence=None,
            set_label="Set_2_Cam_XY_Aligned",
            view_tup=(90, 270),
            render=True,
        )
        add_3d_axes_to_plot(
            C_coords_xy_align_2[0],
            C_xyz_axis_rrr_2,
            color_sequence=["black", "gray", "purple"],
            arrow_length=3,
            set_label="Set_2_Cam_XY_Aligned",
            output_dir=output_dir,
            render=False,
        )
    '''

    final_xy_align_rot_obj_1 = axis_aligned_mirror_points(
        C_coords_xy_1, angular_tol=0.05, refine=True, refine_span_deg=1, refine_step_deg=0.01
    )
    final_xy_align_rot_obj_2 = axis_aligned_mirror_points(
        C_coords_xy_2, angular_tol=0.05, refine=True, refine_span_deg=1, refine_step_deg=0.01
    )

    ctrl_xy_align_final_1 = apply_rt_to_control(ctrl_xy_1, final_xy_align_rot_obj_1, None)
    C_coords_xy_final_1 = final_xy_align_rot_obj_1.apply(C_coords_xy_1)
    C_slopes_xy_final_1 = final_xy_align_rot_obj_1.apply(C_slope_look_axis_1)

    ctrl_xy_align_final_2 = apply_rt_to_control(ctrl_xy_2, final_xy_align_rot_obj_2, None)
    C_coords_xy_final_2 = final_xy_align_rot_obj_2.apply(C_coords_xy_2)
    C_slopes_xy_final_2 = final_xy_align_rot_obj_2.apply(C_slope_look_axis_2)

    # check final alignment with control points and normal vector and ensure that it lines up with expectations.

    if debug_plots:
        plot_scatter_heat_map(
            C_coords_xy_final_1,
            C_slopes_xy_final_1,
            "Set_1_Cam_XY_Align_Final",
            view_tup=(90, 270),
            output_dir=output_dir,
            render=False,
            invert_y=True,
        )
        plot_scatter_heat_map(
            C_coords_xy_final_2,
            C_slopes_xy_final_2,
            "Set_2_Cam_XY_Align_Final",
            view_tup=(90, 270),
            output_dir=output_dir,
            render=False,
            invert_y=True,
        )

    set_1_results = {
        "slopes_rotation_combined": final_xy_align_rot_obj_1 * C_look_axis_rot_1,
        "coords_rotation_combined": final_xy_align_rot_obj_1 * C_coords_xy_rot_1 * C_look_axis_rot_1,
        "coords_translation_combined": -1 * np.mean(C_coords_look_axis_1, axis=0),
        "coords": C_coords_xy_final_1,
        "slopes": C_slopes_xy_final_1,
    }

    set_2_results = {
        "slopes_rotation_combined": final_xy_align_rot_obj_2 * C_look_axis_rot_2,
        "coords_rotation_combined": final_xy_align_rot_obj_2 * C_coords_xy_rot_2 * C_look_axis_rot_2,
        "coords_translation_combined": -1 * np.mean(C_coords_look_axis_2, axis=0),
        "coords": C_coords_xy_final_2,
        "slopes": C_slopes_xy_final_2,
    }

    # sofast_plotting(output_directory=os.path.join(output_dir, "set_1"), solution_set=set_1_results)
    # sofast_plotting(output_directory=os.path.join(output_dir, "set_2"), solution_set=set_2_results)

    print("cam_plotting.....")

    return set_1_results, set_2_results


def plot_heat_maps_mirror_looking_up_coords(vector_data, output_dir, debug_plots=True):
    x_pix_set, y_pix_set, M_coords, slope_set1, slope_set2 = [], [], [], [], []
    for pixel, details in vector_data.items():
        if isinstance(details, dict):
            if details["intersection_1"].size > 0:
                row, col = ast.literal_eval(pixel)

                x_pix_set.append(col)
                y_pix_set.append(row)
                M_coords.append(vector_data[pixel]["M_point_location"])
                slope_set1.append(vector_data[pixel]["M_slope_1_camera_corrected"])
                slope_set2.append(vector_data[pixel]["M_slope_2_camera_corrected"])

            else:
                continue
        else:
            pass

    M_coords = np.array(M_coords).squeeze(axis=1)
    mean_direction_1 = np.mean(np.array(slope_set1), axis=0)
    mean_direction_1 = mean_direction_1 / np.linalg.norm(mean_direction_1)
    mean_direction_2 = np.mean(np.array(slope_set2), axis=0)
    mean_direction_2 = mean_direction_2 / np.linalg.norm(mean_direction_2)

    m_pt_min = np.min(np.array(M_coords), axis=0).reshape(3)
    m_pt_max = np.max(np.array(M_coords), axis=0).reshape(3)
    m_pt_avg = np.mean(np.array(M_coords), axis=0).reshape(3)

    control_pts = np.array(
        [
            [m_pt_min[0], m_pt_min[1], m_pt_avg[2] / 2],
            [m_pt_max[0], m_pt_max[1], m_pt_avg[2] / 2],
            [m_pt_min[0], m_pt_max[1], m_pt_avg[2] / 2],
        ]
    )
    ctrl_n = _triangle_normal(control_pts)

    if np.dot(ctrl_n, mean_direction_1) < 0:
        ctrl_n = -ctrl_n

    control = {"points": control_pts, "normal": ctrl_n}

    M_xyz_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    M_look_axis_rot_1, _ = rotation_matrix_scipy(mean_direction_1, np.array([0, 0, 1]))  # Rotation
    M_look_axis_rot_2, _ = rotation_matrix_scipy(mean_direction_2, np.array([0, 0, 1]))  # Rotation
    mean_direction_r_1 = M_look_axis_rot_1.apply(mean_direction_1)
    mean_direction_r_2 = M_look_axis_rot_2.apply(mean_direction_2)
    M_xyz_axis_r_1 = M_look_axis_rot_1.apply(M_xyz_axis)
    M_xyz_axis_r_2 = M_look_axis_rot_2.apply(M_xyz_axis)
    ctrl_r_1 = apply_rt_to_control(control, M_look_axis_rot_1, None)
    ctrl_r_2 = apply_rt_to_control(control, M_look_axis_rot_2, None)
    M_slope_look_axis_1, M_slope_look_axis_2, M_coords_look_axis_1, M_coords_look_axis_2 = [], [], [], []

    M_slope_look_axis_1 = M_look_axis_rot_1.apply(np.array(slope_set1))
    M_coords_look_axis_1 = M_look_axis_rot_1.apply(M_coords)
    M_slope_look_axis_2 = M_look_axis_rot_2.apply(np.array(slope_set2))
    M_coords_look_axis_2 = M_look_axis_rot_2.apply(M_coords)

    if debug_plots:
        plot_scatter_heat_map(
            M_coords,
            np.array(slope_set1),
            set_label="Set_1_Mir_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            M_coords[0],
            M_xyz_axis,
            color_sequence=None,
            arrow_length=2,
            set_label="Set_1_Mir_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            M_coords[round(len(M_coords) / 2)],
            mean_direction_1,
            color='orange',
            arrow_length=2,
            set_label="Set_1_Mir_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            M_coords[round(len(M_coords) / 2)],
            mean_direction_r_1,
            color='black',
            arrow_length=2,
            set_label="Set_1_Mir_Vec_Coords",
            output_dir=output_dir,
            # view_tup=(30, 135),
            render=False,
        )

        plot_scatter_heat_map(
            M_coords,
            np.array(slope_set2),
            set_label="Set_2_Mir_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            M_coords[0],
            M_xyz_axis,
            color_sequence=None,
            arrow_length=2,
            set_label="Set_2_Mir_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            M_coords[round(len(M_coords) / 2)],
            mean_direction_2,
            color='orange',
            arrow_length=2,
            set_label="Set_2_Mir_Vec_Coords",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            M_coords[round(len(M_coords) / 2)],
            mean_direction_r_2,
            color='black',
            arrow_length=2,
            set_label="Set_2_Mir_Vec_Coords",
            output_dir=output_dir,
            # view_tup=(30, 135),
            render=False,
        )

        plot_scatter_heat_map(
            M_coords_look_axis_1,
            M_slope_look_axis_1,
            set_label="Set_1_Mir_Look_Up",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            M_coords_look_axis_1[0],
            M_xyz_axis,
            color_sequence=None,
            arrow_length=2,
            set_label="Set_1_Mir_Look_Up",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            M_coords_look_axis_1[round(len(M_coords_look_axis_1) / 2)],
            mean_direction_1,
            color='orange',
            arrow_length=2,
            set_label="Set_1_Mir_Look_Up",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            M_coords_look_axis_1[round(len(M_coords_look_axis_1) / 2)],
            mean_direction_r_1,
            color='black',
            arrow_length=2,
            set_label="Set_1_Mir_Look_Up",
            # view_tup=(30, 135),
            output_dir=output_dir,
            render=False,
        )

        plot_scatter_heat_map(
            M_coords_look_axis_2,
            M_slope_look_axis_2,
            set_label="Set_2_Mir_Look_Up",
            # view_tup=(30, 135),
            render=True,
        )
        add_3d_axes_to_plot(
            M_coords_look_axis_2[0],
            M_xyz_axis,
            color_sequence=None,
            arrow_length=2,
            set_label="Set_2_Mir_Look_Up",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            M_coords_look_axis_2[round(len(M_coords_look_axis_2) / 2)],
            mean_direction_2,
            color='orange',
            arrow_length=2,
            set_label="Set_2_Mir_Look_Up",
            # view_tup=(30, 135),
            render=True,
        )
        add_vector_to_plot(
            M_coords_look_axis_2[round(len(M_coords_look_axis_2) / 2)],
            mean_direction_r_2,
            color='black',
            arrow_length=2,
            set_label="Set_2_Mir_Look_Up",
            # view_tup=(30, 135),
            output_dir=output_dir,
            render=False,
        )

    M_coords_look_axis_1_translate = M_coords_look_axis_1 - np.mean(M_coords_look_axis_1, axis=0)  # Translation
    M_coords_look_axis_2_translate = M_coords_look_axis_2 - np.mean(M_coords_look_axis_2, axis=0)  # Translation

    ctrl_rt_1 = apply_rt_to_control(ctrl_r_1, Rotation.identity(), -1 * np.mean(M_coords_look_axis_1, axis=0))
    ctrl_rt_2 = apply_rt_to_control(ctrl_r_2, Rotation.identity(), -1 * np.mean(M_coords_look_axis_2, axis=0))

    M_coords_xy_rot_1 = transform_to_xy_plane(
        [
            M_coords_look_axis_1_translate[0, :],
            M_coords_look_axis_1_translate[-1, :],
            M_coords_look_axis_1_translate[1000, :],
        ],
        z_orientation=1,
        reference_normal=ctrl_rt_1["normal"],
    )  # Rotation for Coordinates Only
    M_coords_xy_rot_2 = transform_to_xy_plane(
        [
            M_coords_look_axis_2_translate[0, :],
            M_coords_look_axis_2_translate[-1, :],
            M_coords_look_axis_2_translate[1000, :],
        ],
        z_orientation=1,
        reference_normal=ctrl_rt_2["normal"],
    )  # Rotation for Coordinates Only

    ctrl_xy_1 = apply_rt_to_control(ctrl_rt_1, M_coords_xy_rot_1, None)
    ctrl_xy_2 = apply_rt_to_control(ctrl_rt_2, M_coords_xy_rot_2, None)

    M_xyz_axis_rr_1 = M_coords_xy_rot_1.apply(M_xyz_axis_r_1)
    M_xyz_axis_rr_2 = M_coords_xy_rot_2.apply(M_xyz_axis_r_2)

    M_coords_xy_1 = M_coords_xy_rot_1.apply(M_coords_look_axis_1)
    # M_slopes_xy_1 = M_coords_xy_rot_1.apply(M_slope_look_axis_1)
    M_coords_xy_2 = M_coords_xy_rot_2.apply(M_coords_look_axis_2)
    # M_slopes_xy_2 = M_coords_xy_rot_2.apply(M_slope_look_axis_2)

    M_coords_xy_centered_1 = M_coords_xy_1 - np.mean(M_coords_xy_1, axis=0)  # Translation
    M_coords_xy_centered_2 = M_coords_xy_2 - np.mean(M_coords_xy_2, axis=0)  # Translation

    '''
    plot_scatter_heat_map(M_coords_xy_1, M_slopes_xy_1, "Set 1 Mirror Looking Up XY Plane")
    add_3d_axes_to_plot(M_coords_xy_1[0], M_xyz_axis)
    add_3d_axes_to_plot(M_coords_xy_1[0], M_xyz_axis_rr_1, color_sequence=["y", "m", "c"])
    

    x_align_angle_1 = binary_search_angle_select(
        M_xyz_axis_rr_1[0],
        M_xyz_axis[2],
        rotate_vector,
        component_index=1,
        tolerance=1e-6,
        max_iterations=5000,
        num_initial_samples=36,
    )
    x_align_rot_obj_1 = create_rotation_object(axis=M_xyz_axis[2], angle_degrees=x_align_angle_1)  # Rotation

    M_xyz_axis_rrr_1 = x_align_rot_obj_1.apply(M_xyz_axis_rr_1)
    # add_3d_axes_to_plot(M_coords_xy_1[0], M_xyz_axis_rrr_1, color_sequence=["black", "gray", "purple"], arrow_length=3)
    
    plot_scatter_heat_map(M_coords_xy_2, M_slopes_xy_2, "Set 2 Mirror Looking Up XY Plane")
    add_3d_axes_to_plot(M_coords_xy_2[0], M_xyz_axis)
    add_3d_axes_to_plot(M_coords_xy_2[0], M_xyz_axis_rr_2, color_sequence=["y", "m", "c"])
    

    x_align_angle_2 = binary_search_angle_select(
        M_xyz_axis_rr_2[0],
        M_xyz_axis[2],
        rotate_vector,
        component_index=1,
        tolerance=1e-6,
        max_iterations=5000,
        num_initial_samples=36,
    )
    x_align_rot_obj_2 = create_rotation_object(axis=M_xyz_axis[2], angle_degrees=x_align_angle_2)  # Rotation

    M_xyz_axis_rrr_2 = x_align_rot_obj_2.apply(M_xyz_axis_rr_1)
    # add_3d_axes_to_plot(M_coords_xy_1[0], M_xyz_axis_rrr_1, color_sequence=["black", "gray", "purple"], arrow_length=3)

    M_coords_xy_align_1 = x_align_rot_obj_1.apply(M_coords_xy_1)
    # M_slopes_xy_align_1 = x_align_rot_obj_1.apply(M_slopes_xy_1)
    M_slopes_xy_align_1 = x_align_rot_obj_1.apply(M_slope_look_axis_1)

    M_coords_xy_align_2 = x_align_rot_obj_2.apply(M_coords_xy_2)
    # M_slopes_xy_align_2 = x_align_rot_obj_2.apply(M_slopes_xy_2)
    M_slopes_xy_align_2 = x_align_rot_obj_2.apply(M_slope_look_axis_2)

    if debug_plots:
        plot_scatter_heat_map(
            M_coords_xy_align_1, M_slopes_xy_align_1, set_label="Set_1_Mir_XY_Aligned", view_tup=(90, 270), render=True
        )
        add_3d_axes_to_plot(
            M_coords_xy_align_1[0], M_xyz_axis, set_label="Set_1_Mir_XY_Aligned", view_tup=(90, 270), render=True
        )
        add_3d_axes_to_plot(
            M_coords_xy_align_1[0],
            M_xyz_axis_rrr_1,
            color_sequence=["black", "gray", "purple"],
            arrow_length=3,
            set_label="Set_1_Mir_XY_Aligned",
            view_tup=(90, 270),
            output_dir=output_dir,
            render=False,
        )

        plot_scatter_heat_map(
            M_coords_xy_align_2, M_slopes_xy_align_2, set_label="Set_2_Mir_XY_Aligned", view_tup=(90, 270), render=True
        )
        add_3d_axes_to_plot(
            M_coords_xy_align_2[0], M_xyz_axis, set_label="Set_2_Mir_XY_Aligned", view_tup=(90, 270), render=True
        )
        add_3d_axes_to_plot(
            M_coords_xy_align_2[0],
            M_xyz_axis_rrr_2,
            color_sequence=["black", "gray", "purple"],
            arrow_length=3,
            set_label="Set_2_Mir_XY_Aligned",
            view_tup=(90, 270),
            output_dir=output_dir,
            render=False,
        )

    '''

    final_xy_align_rot_obj_1 = axis_aligned_mirror_points(
        M_coords_xy_centered_1, angular_tol=0.05, refine=True, refine_span_deg=1, refine_step_deg=0.01
    )
    final_xy_align_rot_obj_2 = axis_aligned_mirror_points(
        M_coords_xy_centered_2, angular_tol=0.05, refine=True, refine_span_deg=1, refine_step_deg=0.01
    )

    ctrl_xy_align_final_1 = apply_rt_to_control(ctrl_xy_1, final_xy_align_rot_obj_1, None)
    M_coords_xy_final_1 = final_xy_align_rot_obj_1.apply(M_coords_xy_centered_1)
    M_slopes_xy_final_1 = final_xy_align_rot_obj_1.apply(M_slope_look_axis_1)

    ctrl_xy_align_final_2 = apply_rt_to_control(ctrl_xy_2, final_xy_align_rot_obj_2, None)
    M_coords_xy_final_2 = final_xy_align_rot_obj_2.apply(M_coords_xy_centered_2)
    M_slopes_xy_final_2 = final_xy_align_rot_obj_2.apply(M_slope_look_axis_2)

    if debug_plots:
        plot_scatter_heat_map(
            M_coords_xy_final_1,
            M_slopes_xy_final_1,
            set_label="Set_1_Mir_XY_Align_Final",
            view_tup=(90, 270),
            render=False,
            output_dir=output_dir,
        )
        plot_scatter_heat_map(
            M_coords_xy_final_2,
            M_slopes_xy_final_2,
            set_label="Set_2_Mir_XY_Align_Final",
            view_tup=(90, 270),
            render=False,
            output_dir=output_dir,
        )

    set_1_results = {
        "slopes_rotation_combined": final_xy_align_rot_obj_1 * M_look_axis_rot_1,
        "coords_rotation_combined": final_xy_align_rot_obj_1 * M_coords_xy_rot_1 * M_look_axis_rot_1,
        "coords_translation_combined": -1 * np.mean(M_coords_look_axis_1, axis=0)
        + -1 * -np.mean(M_coords_xy_1, axis=0),
        "coords": M_coords_xy_final_1,
        "slopes": M_slopes_xy_final_1,
    }
    lbt.write_compressed_json(set_1_results, ft.join(output_dir, "set_1_final_RT.json.gz"))
    # foo = lbt.read_compressed_json(ft.join(output_dir, "set_1_final_RT.json.gz"))
    set_2_results = {
        "slopes_rotation_combined": final_xy_align_rot_obj_2 * M_look_axis_rot_2,
        "coords_rotation_combined": final_xy_align_rot_obj_2 * M_coords_xy_rot_2 * M_look_axis_rot_2,
        "coords_translation_combined": -1 * np.mean(M_coords_look_axis_2, axis=0)
        + -1 * -np.mean(M_coords_xy_2, axis=0),
        "coords": M_coords_xy_final_2,
        "slopes": M_slopes_xy_final_2,
    }
    lbt.write_compressed_json(set_2_results, ft.join(output_dir, "set_2_final_RT.json.gz"))

    sofast_plotting(output_directory=ft.join(output_dir, "set_1"), solution_set=set_1_results)
    sofast_plotting(output_directory=ft.join(output_dir, "set_2"), solution_set=set_2_results)

    print("mirror plotting....")

    return set_1_results, set_2_results


def _triangle_normal(pts3):
    """Return unit normal of triangle (p0,p1,p2) using right-hand rule."""
    p0, p1, p2 = np.asarray(pts3, float)
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise ValueError("Degenerate control triangle (zero area).")
    return n / nn


def _ensure_normal_points_toward_camera(tri_pts, normal, camera_origin=np.zeros(3)):
    """
    Ensure normal points from triangle toward camera_origin.
    If not, swap p1/p2 (flips winding) and invert normal.
    """
    tri_pts = np.asarray(tri_pts, float)
    normal = np.asarray(normal, float)

    centroid = tri_pts.mean(axis=0)
    to_cam = camera_origin - centroid  # vector from triangle to camera
    if np.dot(normal, to_cam) < 0:  # normal points away; flip it
        tri_pts = tri_pts.copy()
        tri_pts[[1, 2]] = tri_pts[[2, 1]]
        normal = -normal
    return tri_pts, normal


def estimate_pixel_to_camera_coords_3D(camera, data_dict, ref_distance, rot_obj, t_vec):
    cam_coords = []
    K_inv = np.linalg.inv(camera.intrinsic_mat)
    for pixel, details in data_dict.items():
        if isinstance(details, dict):
            if details["intersection_1"].size > 0:
                row, col = ast.literal_eval(pixel)
                normalized_point = K_inv @ np.array([col, row, 1])
                point_camera = normalized_point * ref_distance
                C_point = rot_obj.as_matrix().T @ (point_camera - t_vec.data.T).T
                data_dict[pixel]["C_point_location"] = C_point.T
                cam_coords.append(data_dict[pixel]["C_point_location"])
            else:
                continue
        else:
            pass

    c_pt_min = np.min(np.array(cam_coords), axis=0).reshape(3)
    c_pt_max = np.max(np.array(cam_coords), axis=0).reshape(3)
    c_pt_avg = np.mean(np.array(cam_coords), axis=0).reshape(3)

    control_pts = np.array(
        [
            [c_pt_min[0], c_pt_min[1], c_pt_avg[2] / 2],
            [c_pt_max[0], c_pt_max[1], c_pt_avg[2] / 2],
            [c_pt_min[0], c_pt_max[1], c_pt_avg[2] / 2],
        ]
    )
    ctrl_n = _triangle_normal(control_pts)
    control_pts, ctrl_n = _ensure_normal_points_toward_camera(control_pts, ctrl_n)

    control = {"points": control_pts, "normal": ctrl_n}

    return data_dict, control


def apply_rt_to_control(control, rot=None, trans=None):
    """
    Apply rotation (and optional translation) to control triangle.
    trans: (3,) translation applied as p' = rot.apply(p + trans) or p' = rot.apply(p) + trans,
    depending on your convention. Here we use p' = rot.apply(p) + trans (standard).
    """
    pts = np.asarray(control["points"], float)
    n = np.asarray(control["normal"], float)
    n_r = None

    if rot is not None:
        pts_r = rot.apply(pts)
        n_r = rot.apply(n)

    if trans is not None:
        trans = np.asarray(trans, float).reshape(3)
        pts_r = pts_r + trans  # normals do NOT translate

    return {"points": pts_r, "normal": n_r if n_r is not None else n}


def add_3d_axes_to_plot(
    point,
    xyz_vec_array,
    color_sequence=None,
    arrow_length=2,
    set_label="added_3d_axes",
    view_tup=None,
    output_dir=None,
    render=False,
):
    if output_dir:
        ft.create_directories_if_necessary(output_dir)
    # Access the current figure and cycle through each axis
    fig = plt.gcf()
    for ax in fig.get_axes():
        if isinstance(ax, Axes3D):
            # Plot arrows using quiver
            if color_sequence:
                for index, color in enumerate(color_sequence):
                    ax.quiver(
                        point[0],
                        point[1],
                        point[2],
                        xyz_vec_array[index, 0],
                        xyz_vec_array[index, 1],
                        xyz_vec_array[index, 2],
                        color=color,
                        length=arrow_length,
                        normalize=True,
                    )
            else:
                for index, color in enumerate(["r", "g", "b"]):
                    ax.quiver(
                        point[0],
                        point[1],
                        point[2],
                        xyz_vec_array[index, 0],
                        xyz_vec_array[index, 1],
                        xyz_vec_array[index, 2],
                        color=color,
                        length=arrow_length,
                        normalize=True,
                    )
    if output_dir:
        if view_tup:
            file_path = ft.join(output_dir, set_label + f"_el{view_tup[0]}_az{view_tup[1]}.png")
        else:
            file_path = ft.join(output_dir, set_label + ".png")
        plt.savefig(file_path)
    if not render:
        plt.close()


def add_vector_to_plot(
    point,
    xyz_vec_array,
    color,
    arrow_length=2,
    set_label="added_3d_vector",
    view_tup=None,
    output_dir=None,
    render=False,
):
    if output_dir:
        ft.create_directories_if_necessary(output_dir)
    # Access the current figure and cycle through each axis
    fig = plt.gcf()
    for ax in fig.get_axes():
        if isinstance(ax, Axes3D):
            # Plot arrows using quiver
            ax.quiver(
                point[0],
                point[1],
                point[2],
                xyz_vec_array[0],
                xyz_vec_array[1],
                xyz_vec_array[2],
                color=color,
                length=arrow_length,
                normalize=True,
            )

    if output_dir:
        if view_tup:
            file_path = ft.join(output_dir, set_label + f"_el{view_tup[0]}_az{view_tup[1]}.png")
        else:
            file_path = ft.join(output_dir, set_label + ".png")
        plt.savefig(file_path)
    if not render:
        plt.close()


def plot_pixel_camera_and_mirror_coords(data_dict, skip_num=50, view_tup=None, output_dir=None, render=False):
    if output_dir:
        ft.create_directories_if_necessary(output_dir)
    cam_coords, mir_coords = [], []
    for pixel, details in data_dict.items():
        if isinstance(details, dict):
            if details["intersection_1"].size > 0:
                cam_coords.append(data_dict[pixel]["C_point_location"])
                mir_coords.append(data_dict[pixel]["M_point_location"])
            else:
                continue
        else:
            pass

    cam_coords = np.array(cam_coords).squeeze(axis=1)
    mir_coords = np.array(mir_coords).squeeze(axis=1)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # skip_num = 50
    ax1.scatter(cam_coords[::skip_num, 0], cam_coords[::skip_num, 1], cam_coords[::skip_num, 2], c='k', s=1)
    ax1.set_xlabel('X axis Camera')
    ax1.set_ylabel('Y axis Camera')
    ax1.set_zlabel('Z axis Camera')
    ax1.set_title("Camera Coordinates")
    if view_tup:
        ax1.view_init(elev=view_tup[0], azim=view_tup[1])

    ax2.scatter(mir_coords[::skip_num, 0], mir_coords[::skip_num, 1], mir_coords[::skip_num, 2], c='g', s=1)
    ax2.set_xlabel('X axis Mirror')
    ax2.set_ylabel('Y axis Mirror')
    ax2.set_zlabel('Z axis Mirror')
    ax2.set_title("Mirror Coordinates")
    if view_tup:
        ax2.view_init(elev=view_tup[0], azim=view_tup[1])

    if output_dir:
        if view_tup:
            file_path = ft.join(output_dir, "projected_coordinates" + f"_el{view_tup[0]}_az{view_tup[1]}.png")
        else:
            file_path = ft.join(output_dir, "projected_coordinates" + ".png")
        plt.savefig(file_path)
    if not render:
        plt.close()


def transform_to_xy_plane(points, z_orientation=1, reference_normal=None):
    points = np.asarray(points)
    if points.shape[0] < 3:
        raise ValueError("At least three points are required to define a plane.")

    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    z_axis = np.array([0, 0, z_orientation])

    # If provided, ensure chosen normal has desired relationship to reference_normal.
    if reference_normal is not None:
        reference_normal = np.asarray(reference_normal, float)
        # Choose between normal and -normal
        if np.dot(normal, reference_normal) < 0:
            normal = -normal

    rotation, _ = rotation_matrix_scipy(normal, z_axis)
    return rotation


'''
def axis_aligned_mirror_points(points_xyz, angular_tol, step_size):
    points_og = np.array(points_xyz)
    points_og_xy = np.array(points_og[:, 0:2])

    rect = cv2.minAreaRect(np.float32(points_og_xy))
    box = cv2.boxPoints(rect)
    # In test, points of box came out in CCW starting with the top left

    vec_v_1 = box[0] - box[1]
    vec_v_2 = box[3] - box[2]

    vec_h_1 = box[2] - box[1]
    vec_h_2 = box[3] - box[0]

    rot_obj_v_1, _ = rotation_matrix_scipy(np.append(vec_v_1, 0), np.array([0, 1, 0]))
    rot_obj_v_2, _ = rotation_matrix_scipy(np.append(vec_v_2, 0), np.array([0, 1, 0]))
    rot_obj_h_1, _ = rotation_matrix_scipy(np.append(vec_h_1, 0), np.array([1, 0, 0]))
    rot_obj_h_2, _ = rotation_matrix_scipy(np.append(vec_h_2, 0), np.array([1, 0, 0]))

    obj_list = [rot_obj_v_1, rot_obj_v_2, rot_obj_h_1, rot_obj_h_2]
    min_rot = None
    test_pts = np.empty_like(points_og)
    for index, thing in enumerate(obj_list):
        test_pts = thing.apply(points_og)
        rect_test = cv2.minAreaRect(np.float32(test_pts[:, 0:2]))

        if min_rot is None:
            min_rot = (rect_test[2], index)
        else:
            if rect_test[2] < min_rot[0]:
                min_rot = (rect_test[2], index)

    rot_angle = min_rot[0]
    refine_obj = obj_list[min_rot[1]]
    foo = refine_obj.as_rotvec()
    while abs(rot_angle) > angular_tol:

        foo_up = np.array([foo[0], foo[1], foo[2] + step_size])
        foo_down = np.array([foo[0], foo[1], foo[2] - step_size])

        foo_up = Rotation.from_rotvec(foo_up)
        foo_down = Rotation.from_rotvec(foo_down)

        refine_up = foo_up.apply(points_og)
        refine_down = foo_down.apply(points_og)

        refine_up_rect = cv2.minAreaRect(np.float32(refine_up[:, 0:2]))
        refine_down_rect = cv2.minAreaRect(np.float32(refine_down[:, 0:2]))

        min_index = np.argmin(np.array([abs(refine_up_rect[2]), abs(refine_down_rect[2])]))

        if min_index == 0:
            rot_angle = refine_up_rect[2]
            foo = foo_up
        elif min_index == 1:
            rot_angle = refine_down_rect[2]
            foo = foo_down
        else:
            break

    if isinstance(foo, np.ndarray):
        rot_obj = Rotation.from_rotvec(foo)
        return rot_obj
    elif isinstance(foo, Rotation):
        return foo
    else:
        raise ValueError("Output was not the correct data type... Need to Debug")
'''


def _wrap_to_90(deg):
    """Map angle (deg) to [-45, 45] by exploiting 90-deg symmetry."""
    return ((deg + 45.0) % 90.0) - 45.0


def _rect_yaw_deg(points_xy):
    """
    Compute a robust yaw (deg) that would axis-align the minAreaRect of points_xy.
    Returns yaw in degrees to ROTATE points by (about +Z) to align.
    """
    rect = cv2.minAreaRect(points_xy.astype(np.float32))
    (cx, cy), (w, h), a = rect  # a is in degrees, OpenCV convention

    # OpenCV convention: angle is the rotation of the rectangle's width side
    # but varies by version; a is typically in [-90, 0).
    # If width < height, the "width side" is actually the short side; adjust by 90.
    if w < h:
        a = a + 90.0

    # We want to rotate by -a to align the long side with +X (or nearest axis)
    yaw = -a

    # Because aligning to X or Y is equivalent up to 90 degrees, reduce ambiguity:
    yaw = _wrap_to_90(yaw)
    return yaw


def _objective_abs_angle(points_xy):
    """Objective: |minAreaRect angle| after wrapping to nearest axis."""
    rect = cv2.minAreaRect(points_xy.astype(np.float32))
    (_, _), (w, h), a = rect
    if w < h:
        a = a + 90.0
    # a is angle of long side relative to +X (after adjustment)
    # We want that angle near 0 (axis-aligned), modulo 90.
    return abs(_wrap_to_90(a))


def axis_aligned_mirror_points(points_xyz, angular_tol=0.25, refine=True, refine_span_deg=2.0, refine_step_deg=0.05):
    """
    Robustly compute a Rotation that (approximately) yaw-rotates points so their
    min-area bounding rectangle becomes axis-aligned in XY.

    Parameters
    ----------
    points_xyz : (N,3) array-like
    angular_tol : float
        Tolerance in degrees for final rectangle misalignment (after wrap-to-90).
    refine : bool
        If True, do a local 1D search around the closed-form yaw for extra robustness.
    refine_span_deg : float
        Half-span of local search window in degrees.
    refine_step_deg : float
        Step size (deg) for local search.

    Returns
    -------
    scipy.spatial.transform.Rotation
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be an (N,3) array")

    pts_xy = pts[:, :2]

    # Closed-form yaw from minAreaRect
    yaw0 = _rect_yaw_deg(pts_xy)
    rot0 = Rotation.from_euler('z', yaw0, degrees=True)

    if not refine:
        return rot0

    # Local search around yaw0 to minimize axis-misalignment objective
    # (still avoids corner-order assumptions)
    best_yaw = yaw0
    best_val = _objective_abs_angle(rot0.apply(pts)[:, :2])

    # If already good enough, stop
    if best_val <= angular_tol:
        return rot0

    # 1D brute-force local search (simple + robust)
    yaws = np.arange(yaw0 - refine_span_deg, yaw0 + refine_span_deg + 1e-12, refine_step_deg)
    for yaw in yaws:
        r = Rotation.from_euler('z', yaw, degrees=True)
        val = _objective_abs_angle(r.apply(pts)[:, :2])
        if val < best_val:
            best_val = val
            best_yaw = yaw
            if best_val <= angular_tol:
                break

    return Rotation.from_euler('z', best_yaw, degrees=True)


def sofast_plotting(output_directory, solution_set):
    # dir_save_cur = os.path.join(output_directory, "lookfast_processed_data")
    ft.create_directories_if_necessary(output_directory)

    coords_centroid = np.mean(np.array(solution_set["coords"]), axis=0)
    centered_coords = np.array(solution_set["coords"]) - coords_centroid
    coords_centroid = np.mean(centered_coords, axis=0)

    rect = cv2.minAreaRect(np.float32(centered_coords[:, 0:2]))
    box = cv2.boxPoints(rect)  # In test, points of box came out in CCW starting with the top left

    expected_facet_corners = Vxy(
        list(zip(box[3], box[2], box[1], box[0])), dtype=float
    )  # Counterclockwise Starting from Top Right Corner in [row, column] SOFAST Example uses this convention
    '''
    coords_max = np.max(centered_coords, axis=0)
    coords_min = np.min(centered_coords, axis=0)

    expected_facet_corners = Vxy(
        list(
            zip(
                [coords_max[0], coords_max[1]],
                [coords_min[0], coords_max[1]],
                [coords_min[0], coords_min[1]],
                [coords_max[1], coords_min[1]],
            )
        ),
        dtype=float,
    )  # Counterclockwise Starting from Top Right Corner in [row, column] SOFAST Example uses this convention
    '''

    loop = LoopXY.from_vertices(expected_facet_corners)
    region = RegionXY(loop)

    # Get measured and reference optics
    # mirror_measured = sofast.get_optic().mirror.no_parent_copy()
    mirror_measured = MirrorPoint(
        surface_points=Pxyz(centered_coords.T),
        normal_vectors=Uxyz(solution_set["slopes"].T),
        shape=region,
        interpolation_type="nearest",
    )

    mirror_reference = MirrorParametric.generate_symmetric_paraboloid(100, mirror_measured.region)

    # Save optic objects
    plots = StandardPlotOutput()
    plots.optic_measured = mirror_measured  # MirrorPoint object
    plots.optic_reference = mirror_reference

    plots.options_file_output.to_save = True
    plots.options_file_output.number_in_name = False
    plots.options_file_output.output_dir = output_directory
    plots.options_file_output.save_dpi = 200
    plots.options_file_output.save_format = "png"
    plots.options_file_output.close_after_save = True

    # Update visualization parameters
    plots.options_slope_vis.to_plot = True
    plots.options_slope_vis.clim = 20
    plots.options_slope_vis.resolution = 0.01
    plots.options_slope_vis.quiver_density = None  # default 0.1
    plots.options_slope_vis.quiver_scale = 3
    plots.options_slope_vis.quiver_color = "white"

    plots.options_slope_deviation_vis.to_plot = True
    plots.options_slope_deviation_vis.clim = 1.5
    plots.options_slope_deviation_vis.resolution = 0.01
    plots.options_slope_deviation_vis.quiver_density = None  # default 0.1
    plots.options_slope_deviation_vis.quiver_scale = 3
    plots.options_slope_deviation_vis.quiver_color = "white"

    plots.options_curvature_vis.to_plot = True
    plots.options_curvature_vis.clim = 50
    plots.options_curvature_vis.resolution = 0.001
    # plots.options_curvature_vis.processing = # Leave Default for now
    plots.options_curvature_vis.smooth_kernel_width = 1

    plots.options_ray_trace_vis.to_plot = True
    plots.options_ray_trace_vis.ray_trace_optic_res = 0.05
    plots.options_ray_trace_vis.hist_bin_res = 0.07
    plots.options_ray_trace_vis.hist_extent = 3
    plots.options_ray_trace_vis.enclosed_energy_max_semi_width = 1

    # Define viewing/illumination geometry
    v_target_center = Vxyz((0, 0, 100))
    v_target_normal = Vxyz((0, 0, -1))
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=40)

    # Define ray trace parameters
    plots.params_ray_trace.source = source
    plots.params_ray_trace.v_target_center = v_target_center
    plots.params_ray_trace.v_target_normal = v_target_normal

    # Create standard output plots
    plots.plot()


def camera_and_pixel_pointing(cam_obj, light_mask_path):

    ##### reading in masks for light, dark, and all pixels
    light_image = cv2.imread(light_mask_path, cv2.IMREAD_GRAYSCALE)
    dark_image = np.zeros(shape=light_image.shape, dtype=np.uint8)
    all_pixels = np.ones(shape=light_image.shape, dtype=bool)

    ##### calculate pixel pointing vectors for all pixels
    pixel_pointing = imgp.calculate_active_pixels_vectors(mask=all_pixels, camera=cam_obj)

    ##### assigning 9 pixel pointing vectors as the corners, edge midpoints and center to show camera model alignment
    pyramid_pixel_vectors = [
        pixel_pointing[int(0 * light_image.shape[1] + 0)],
        pixel_pointing[int(0 * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int(0 * light_image.shape[1] + light_image.shape[1]) - 1],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + 0)],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + light_image.shape[1]) - 1],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + 0)],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + light_image.shape[1]) - 1],
    ]

    ##### setting initial mask location
    mask_raw = imgp.calc_mask_raw(
        np.concatenate((dark_image[:, :, np.newaxis], light_image[:, :, np.newaxis]), axis=2),
        hist_thresh=0.5,
        filt_width=9,
        filt_thresh=4,
        thresh_active_pixels=0.01,
    )
    mask = imgp.keep_largest_mask_area(mask_raw)
    v_mask_centroid_image = imgp.centroid_mask(mask)
    v_edges_image = imgp.edges_from_mask(mask)

    # mask_image = mask.astype(np.uint8) * 255

    mask_pts = np.array(np.where(mask), dtype=np.float32)

    mask_rect = cv2.minAreaRect(mask_pts.T)
    mask_corners = cv2.boxPoints(mask_rect)  # provided in (row, column pairs) ≈ (y, x) CCW from top left

    ##### setting expected corners in mirror coordinates
    expected_corners_facet_coords_manual = Vxyz(
        list(zip([0.606, 0.606, 0], [-0.606, 0.606, 0], [-0.606, -0.606, 0], [0.606, -0.606, 0])), dtype=float
    )  # Counterclockwise Starting from Top Right Corner in [row, column] SOFAST Example uses this convention

    ##### setting expected corners
    '''
    expected_corners_manual = Vxy(
        list(zip([860, 444], [840, 643], [1047, 657], [1062, 455])), dtype=int
    )  # Counterclockwise Starting from Bottom Left Corner in [row, column]
    '''
    expected_corners_manual = Vxy(
        list(zip(mask_corners[1][::-1], mask_corners[2][::-1], mask_corners[3][::-1], mask_corners[0][::-1])),
        dtype=float,
    )  # Counterclockwise Starting from Bottom Left Corner in [row, column]

    ##### refine corners
    v_corners_image = imgp.refine_facet_corners(
        Puv_facet_corns_exp=expected_corners_manual,
        Puv_cent=v_mask_centroid_image,
        Puv_edges=v_edges_image,
        step=20,
        d_perp=20,
        frac_keep=1,
    )

    ##### estimate camera pose from refined pixel corners and expected mirror coordinate corners
    r_optic_cam_refine_1, v_cam_optic_cam_refine_1 = sp.calc_rt_from_img_pts(
        pts_image=v_corners_image.vertices, pts_object=expected_corners_facet_coords_manual, camera=cam_obj
    )

    return pixel_pointing, r_optic_cam_refine_1, v_cam_optic_cam_refine_1, mask


def main(
    camera_obj,
    light_mask_path,
    vec_data_path,
    reference_distance_m,
    reference_pixel_key,
    output_directory,
    checkpoint_directory,
    checkpoint_file,
):

    # Load checkpoint data and plots
    checkpoint_data = lbt.load_checkpoint(checkpoint_directory, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"Lookfast_RT_Alignment": []}

    pixel_pointing, r_optic_cam_refine_1, v_cam_optic_cam_refine_1, mask = camera_and_pixel_pointing(
        camera_obj, light_mask_path
    )
    vector_data = lbt.read_compressed_json(vec_data_path)

    ##### estimate mirror coordinates in 3D space based on pose estimation
    vector_data, control_pts_cam = estimate_pixel_to_camera_coords_3D(
        camera_obj,
        vector_data,
        ref_distance=reference_distance_m,
        rot_obj=r_optic_cam_refine_1,
        t_vec=v_cam_optic_cam_refine_1,
    )

    ##### pick arbitraty pixel for a horizon reference vector
    reference_vector_horizon = Uxyz(vector_data[reference_pixel_key]["observer_vector"] * -1)

    oa_row, oa_col = ast.literal_eval(reference_pixel_key)

    temp_vec = get_pixel_pointing_vector(
        pixel_directions=pixel_pointing, row=oa_row, col=oa_col, imagewidth=mask.shape[1]
    )
    cam_vec_reference = Uxyz(temp_vec)

    ##### first rotation from "optical axis pointing vector" and reference observer_to_optic_h vector
    rot_obj_no_roll, rssd = rotation_matrix_scipy(
        np.array([cam_vec_reference.x[0], cam_vec_reference.y[0], cam_vec_reference.z[0]]),
        np.array([reference_vector_horizon.x[0], reference_vector_horizon.y[0], reference_vector_horizon.z[0]]),
    )

    ##### apply first rotation, resulting in an arbitrary rotation of x and y axis about cam optical axis / horizonal reference vector
    cam_x_axis = Uxyz(np.array([1, 0, 0]))
    cam_y_axis = Uxyz(np.array([0, 1, 0]))
    cam_xyz_t = rot_obj_no_roll.apply(
        np.array([cam_x_axis.data, cam_y_axis.data, cam_vec_reference.data]).reshape(3, 3)
    )  # The original camera vector that corresponded to the horizonal reference vectors are aligned.

    ##### binary search to find which additional rotation about rotated camera optical axis (now in horizonal coordinates)
    ##### results in the x axis of transformed camera coordinates to have a minimal z-component. i.e. x axis of new camera coordinates in the horizonal XY plane
    ##### this assumes that the orientation of the data collection is landscape mode. Function needs to be updated if changing to portrait orientation.
    '''
    roll_control_angle = binary_search_angle(
        vector=cam_xyz_t[0], axis=cam_xyz_t[2], function=rotate_vector, tolerance=1e-6, max_iterations=1000
    )
    roll_control_angle = (180 - roll_control_angle) * -1

    roll_control_obj = create_rotation_object(axis=cam_xyz_t[2], angle_degrees=roll_control_angle)
    '''

    roll_control_angle, roll_control_obj, dbg_info = select_rotation_with_direction_check(
        vector_to_minimize=cam_xyz_t[0],
        axis=cam_xyz_t[2],
        component_index=2,
        reference_vector=cam_xyz_t[1],
        reference_component_index=2,
        desired_reference_sign=-1,
    )

    ##### apply the roll control roation
    # The rotation to align the transformed camera-to-horizonal axis vectors such that direction of the x component transformed camera-to-horizonal set has a minimized z component.
    # i.e. The x component of that vector (now in the horizonal coordinate system) must lie in the plane created by the X and Y horizonal vectors (East-West and North South)
    cam_xyz_tr = roll_control_obj.apply(cam_xyz_t)

    ##### combine rotation objects for camera to horizonal coordinates and camera to mirror coordinates transform
    cam_horizon_transform = roll_control_obj * rot_obj_no_roll

    cam_horizon_pose_transform = roll_control_obj * rot_obj_no_roll * r_optic_cam_refine_1

    ##### extract data, apply sets of rotations (inverse) to convert horizonal data to camera coordinates and camera coordinates to mirror coordinates
    for pixel, details in vector_data.items():
        if isinstance(details, dict):
            if details["intersection_1"].size > 0:
                row, col = ast.literal_eval(pixel)
                cam_vec_original = get_pixel_pointing_vector(
                    pixel_directions=pixel_pointing, row=row, col=col, imagewidth=mask.shape[1]
                )
                cam_vec_horizon = cam_horizon_transform.apply(cam_vec_original.reshape(3))
                _, slope_1_corr, slope_2_corr = safe_calculate_slope(
                    cam_vec_horizon.reshape(3) * -1,
                    cam_vec_horizon.reshape(3) * -1,
                    details["intersection_1"],
                    details["intersection_2"],
                )
                angle_between = calculate_angle_between_vectors(
                    details['start_vector']['celestial_to_target'], details['end_vector']['celestial_to_target']
                )

                vector_data[pixel]["angle_between"] = angle_between
                vector_data[pixel]["observer_vector_camera_corrected"] = cam_vec_horizon
                vector_data[pixel]["H_slope_1_camera_corrected"] = slope_1_corr
                vector_data[pixel]["H_slope_2_camera_corrected"] = slope_2_corr
                vector_data[pixel]["C_slope_1_camera_corrected"] = cam_horizon_transform.inv().apply(slope_1_corr)
                vector_data[pixel]["C_slope_2_camera_corrected"] = cam_horizon_transform.inv().apply(slope_2_corr)
                vector_data[pixel]["M_point_location"] = (
                    r_optic_cam_refine_1.inv().apply(vector_data[pixel]["C_point_location"])
                    - v_cam_optic_cam_refine_1.data.T
                )
                vector_data[pixel]["M_slope_1_camera_corrected"] = r_optic_cam_refine_1.inv().apply(
                    vector_data[pixel]["C_slope_1_camera_corrected"]
                )
                vector_data[pixel]["M_slope_2_camera_corrected"] = r_optic_cam_refine_1.inv().apply(
                    vector_data[pixel]["C_slope_2_camera_corrected"]
                )
            else:
                continue
        else:
            pass

    ##### take extracted coords and slopes, rotate and align with mirror coordinate system and feed to sofast plotting
    if "horizonal_og_cam_corrected" not in checkpoint_data["Lookfast_RT_Alignment"]:
        plot_slope_heat_maps_horizonal(data_dict=vector_data, output_dir=ft.join(output_directory, "horz"))
        checkpoint_data["Lookfast_RT_Alignment"].append("horizonal_og_cam_corrected")
        lbt.save_checkpoint(checkpoint_directory, checkpoint_file, checkpoint_data, print_path=False)

    if "camera_og_cam_corrected" not in checkpoint_data["Lookfast_RT_Alignment"]:
        plot_slope_heat_maps_camera(data_dict=vector_data, output_dir=ft.join(output_directory, "cam"))
        checkpoint_data["Lookfast_RT_Alignment"].append("camera_og_cam_corrected")
        lbt.save_checkpoint(checkpoint_directory, checkpoint_file, checkpoint_data, print_path=False)

    if "mirror_og_cam_corrected" not in checkpoint_data["Lookfast_RT_Alignment"]:
        plot_slope_heat_maps_mirror(data_dict=vector_data, output_dir=ft.join(output_directory, "mir"))
        checkpoint_data["Lookfast_RT_Alignment"].append("mirror_og_cam_corrected")
        lbt.save_checkpoint(checkpoint_directory, checkpoint_file, checkpoint_data, print_path=False)

    if "horizonal_cam_corrected_zenith_adjust" not in checkpoint_data["Lookfast_RT_Alignment"]:
        plot_heat_maps_horizonal_looking_up(vector_data, output_dir=ft.join(output_directory, "horz_zen_adj"))
        checkpoint_data["Lookfast_RT_Alignment"].append("horizonal_cam_corrected_zenith_adjust")
        lbt.save_checkpoint(checkpoint_directory, checkpoint_file, checkpoint_data, print_path=False)

    if "projected_cam_mirror_coordinates" not in checkpoint_data["Lookfast_RT_Alignment"]:
        plot_pixel_camera_and_mirror_coords(
            vector_data, skip_num=50, output_dir=ft.join(output_directory, "proj_coords")
        )
        checkpoint_data["Lookfast_RT_Alignment"].append("projected_cam_mirror_coordinates")
        lbt.save_checkpoint(checkpoint_directory, checkpoint_file, checkpoint_data, print_path=False)

    if "camera_RT_align" not in checkpoint_data["Lookfast_RT_Alignment"]:
        cam_solution_set_1, cam_solution_set_2 = plot_heat_maps_camera_looking_up_coords(
            vector_data, control_pts_cam, output_dir=ft.join(output_directory, "camera_RT_align"), debug_plots=True
        )
        checkpoint_data["Lookfast_RT_Alignment"].append("camera_RT_align")
        lbt.save_checkpoint(checkpoint_directory, checkpoint_file, checkpoint_data, print_path=False)
        # What to do with the solution set data?

    if "mirror_RT_align" not in checkpoint_data["Lookfast_RT_Alignment"]:
        mir_solution_set_1, mir_solution_set_2 = plot_heat_maps_mirror_looking_up_coords(
            vector_data, output_dir=ft.join(output_directory, "mirror_RT_align"), debug_plots=True
        )
        checkpoint_data["Lookfast_RT_Alignment"].append("mirror_RT_align")
        lbt.save_checkpoint(checkpoint_directory, checkpoint_file, checkpoint_data, print_path=False)
        # What to do with the solution set data?


def main_original_script():
    primary_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025_lookfast"
    video_name = "DSC_0025.MOV"
    checkpoint_folder = os.path.join(primary_folder, "0_checkpoints")
    checkpoint_main_name = "lookback_main_checkpoint.json"

    plotting = True
    function_testing = True

    ##### Setting camera object
    # Sofast Example Camera Intrinsic matrix
    K_intrin = np.array([[5492.064314084441, 0, 1920 / 2], [0, 5486.2706013814895, 1080 / 2], [0, 0, 1]])
    # Sofast Example Camera Distortion coefficients
    D_coeff = np.array([-0.144160742602367, 1.609744377391114, 2.503498158416561e-5, -0.001899042260179])

    cam = Camera(
        intrinsic_mat=K_intrin, distortion_coef=D_coeff, image_shape_xy=tuple[1920, 1080], name="Arbitrary_Example"
    )

    ##### reading in vector data
    original_data_location = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025_Final_ExEx/8_pixel_vector_information/debug/pixel_vector_information_wslope_debug.json.gz"
    vector_data = lbt.read_compressed_json(original_data_location)

    ##### pick arbitraty pixel for a horizon reference vector
    reference_vector_horizon = Uxyz(vector_data["(500, 900)"]["observer_vector"] * -1)
    reference_vector_horizon_camera = Uxyz(
        [reference_vector_horizon.x[0], reference_vector_horizon.z[0] * -1, reference_vector_horizon.y[0]]
    )

    ##### reading in masks for light, dark, and all pixels
    light_image = cv2.imread(os.path.join(primary_folder, "light_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    dark_image = cv2.imread(os.path.join(primary_folder, "dark_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    all_pixels = np.ones(shape=light_image.shape, dtype=bool)

    ##### calculate pixel pointing vectors for all pixels
    pixel_pointing = imgp.calculate_active_pixels_vectors(mask=all_pixels, camera=cam)
    # pixel_pointing_2 = imgp.calculate_active_pixels_vectors(mask=all_pixels, camera=cam2)

    ##### central pixel vector (with better camera model needs to pick optical axis not just the center)
    central_pixel_vector = pixel_pointing[
        int((light_image.shape[0] / 2) * light_image.shape[1]) + int(light_image.shape[1] / 2)
    ]

    ##### assigning 9 pixel pointing vectors as the corners, edge midpoints and center to show camera model alignment
    pyramid_pixel_vectors = [
        pixel_pointing[int(0 * light_image.shape[1] + 0)],
        pixel_pointing[int(0 * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int(0 * light_image.shape[1] + light_image.shape[1]) - 1],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + 0)],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + light_image.shape[1]) - 1],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + 0)],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + light_image.shape[1]) - 1],
    ]

    ##### setting initial mask location
    mask_raw = imgp.calc_mask_raw(
        np.concatenate((dark_image[:, :, np.newaxis], light_image[:, :, np.newaxis]), axis=2),
        hist_thresh=0.5,
        filt_width=9,
        filt_thresh=4,
        thresh_active_pixels=0.01,
    )
    mask = imgp.keep_largest_mask_area(mask_raw)
    v_mask_centroid_image = imgp.centroid_mask(mask)
    v_edges_image = imgp.edges_from_mask(mask)

    mask_image = mask.astype(np.uint8) * 255

    ##### setting expected corners
    expected_corners_manual = Vxy(
        list(zip([860, 444], [840, 643], [1047, 657], [1062, 455])), dtype=int
    )  # Counterclockwise Starting from Bottom Left Corner in [row, column]

    ##### setting expected corners in mirror coordinates
    expected_corners_facet_coords_manual = Vxyz(
        list(zip([0.606, 0.606, 0], [-0.606, 0.606, 0], [-0.606, -0.606, 0], [0.606, -0.606, 0])), dtype=float
    )  # Counterclockwise Starting from Top Right Corner in [row, column] SOFAST Example uses this convention

    ##### refine corners
    v_corners_image = imgp.refine_facet_corners(
        Puv_facet_corns_exp=expected_corners_manual,
        Puv_cent=v_mask_centroid_image,
        Puv_edges=v_edges_image,
        step=20,
        d_perp=20,
        frac_keep=1,
    )

    ##### estimate camera pose from refined pixel corners and expected mirror coordinate corners
    r_optic_cam_refine_1, v_cam_optic_cam_refine_1 = sp.calc_rt_from_img_pts(
        pts_image=v_corners_image.vertices, pts_object=expected_corners_facet_coords_manual, camera=cam
    )

    ##### estimate mirror coordinates in 3D space based on pose estimation
    vector_data = estimate_pixel_to_camera_coords_3D(
        cam, vector_data, ref_distance=99.94392, rot_obj=r_optic_cam_refine_1, t_vec=v_cam_optic_cam_refine_1
    )

    ##### first rotation from "optical axis pointing vector" and reference observer_to_optic_h vector
    rot_obj_no_roll, rssd = rotation_matrix_scipy(
        np.array([central_pixel_vector.x[0], central_pixel_vector.y[0], central_pixel_vector.z[0]]),
        np.array([reference_vector_horizon.x[0], reference_vector_horizon.y[0], reference_vector_horizon.z[0]]),
    )

    ##### apply first rotation, resulting in an arbitrary rotation of x and y axis about cam optical axis / horizonal reference vector
    cam_x_axis = Uxyz(np.array([1, 0, 0]))
    cam_y_axis = Uxyz(np.array([0, 1, 0]))
    cam_xyz_t = rot_obj_no_roll.apply(
        np.array([cam_x_axis.data, cam_y_axis.data, central_pixel_vector.data]).reshape(3, 3)
    )  # The original camera vector that corresponded to the horizonal reference vectors are aligned.

    ##### binary search to find which additional rotation about rotated camera optical axis (now in horizonal coordinates)
    ##### results in the x axis of transformed camera coordinates to have a minimal z-component. i.e. x axis of new camera coordinates in the horizonal XY plane
    roll_control_angle = binary_search_angle(
        vector=cam_xyz_t[0], axis=cam_xyz_t[2], function=rotate_vector, tolerance=1e-6, max_iterations=1000
    )
    roll_control_angle = (180 - roll_control_angle) * -1

    roll_control_obj = create_rotation_object(axis=cam_xyz_t[2], angle_degrees=roll_control_angle)

    ##### test first rotation for sanity check with function and output
    if function_testing:

        test_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        test_output_x = []
        test_output_y = []
        test_output_z = []
        for item in test_vectors:
            test_output_x.append(rotate_vector(item, np.array([1, 0, 0]), degrees=90))
            test_output_y.append(rotate_vector(item, np.array([0, 1, 0]), degrees=90))
            test_output_z.append(rotate_vector(item, np.array([0, 0, 1]), degrees=90))
            test_output_x.append(rotate_vector(item, np.array([1, 0, 0]), degrees=45))
            test_output_y.append(rotate_vector(item, np.array([0, 1, 0]), degrees=45))
            test_output_z.append(rotate_vector(item, np.array([0, 0, 1]), degrees=45))

    ##### apply the roll control roation
    # The rotation to align the transformed camera-to-horizonal axis vectors such that direction of the x component transformed camera-to-horizonal set has a minimized z component.
    # i.e. The x component of that vector (now in the horizonal coordinate system) must lie in the plane created by the X and Y horizonal vectors (East-West and North South)
    cam_xyz_tr = roll_control_obj.apply(cam_xyz_t)

    ##### combine rotation objects for camera to horizonal coordinates and camera to mirror coordinates transform
    cam_horizon_transform = roll_control_obj * rot_obj_no_roll

    cam_horizon_pose_transform = roll_control_obj * rot_obj_no_roll * r_optic_cam_refine_1

    ##### vector plot of original camera coords, first rotation applied, and roll control rotation applied
    if plotting:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vectors_data = [
            [0, 0, 0, cam_x_axis.x[0], cam_x_axis.y[0], cam_x_axis.z[0], "Cam X Cam1", "r"],
            [0, 0, 0, cam_y_axis.x[0], cam_y_axis.y[0], cam_y_axis.z[0], "Cam Y Cam1", "g"],
            [
                0,
                0,
                0,
                central_pixel_vector.x[0],
                central_pixel_vector.y[0],
                central_pixel_vector.z[0],
                "Cam Z (Central Pixel Pointing) Cam1",
                "b",
            ],
            [
                0,
                0,
                0,
                reference_vector_horizon.x[0],
                reference_vector_horizon.y[0],
                reference_vector_horizon.z[0],
                "Horizon Reference Horz",
                "darkorange",
            ],
            [
                0,
                0,
                0,
                reference_vector_horizon_camera.x[0],
                reference_vector_horizon_camera.y[0],
                reference_vector_horizon_camera.z[0],
                "Horizon Reference Camera Cam1",
                "tan",
            ],
            [
                0,
                0,
                0,
                cam_xyz_t[0][0] * 0.75,
                cam_xyz_t[0][1] * 0.75,
                cam_xyz_t[0][2] * 0.75,
                "Cam X Transformed Horz",
                "maroon",
            ],
            [
                0,
                0,
                0,
                cam_xyz_t[1][0] * 0.75,
                cam_xyz_t[1][1] * 0.75,
                cam_xyz_t[1][2] * 0.75,
                "Cam Y Transformed Horz",
                "lime",
            ],
            [
                0,
                0,
                0,
                cam_xyz_t[2][0] * 0.75,
                cam_xyz_t[2][1] * 0.75,
                cam_xyz_t[2][2] * 0.75,
                "Cam Z Transformed Horz",
                "midnightblue",
            ],
            [
                0,
                0,
                0,
                cam_xyz_tr[0][0] * 0.5,
                cam_xyz_tr[0][1] * 0.5,
                cam_xyz_tr[0][2] * 0.5,
                "Cam X Transformed Roll Horz",
                "aqua",
            ],
            [
                0,
                0,
                0,
                cam_xyz_tr[1][0] * 0.5,
                cam_xyz_tr[1][1] * 0.5,
                cam_xyz_tr[1][2] * 0.5,
                "Cam Y Transformed Roll Horz",
                "blueviolet",
            ],
            [
                0,
                0,
                0,
                cam_xyz_tr[2][0] * 0.5,
                cam_xyz_tr[2][1] * 0.5,
                cam_xyz_tr[2][2] * 0.5,
                "Cam Z Transformed Roll Horz",
                "deeppink",
            ],
        ]

        for ox, oy, oz, dx, dy, dz, label, color in vectors_data:
            # Plot the vector using quiver
            ax.quiver(ox, oy, oz, dx, dy, dz, color=color, arrow_length_ratio=0.1, label=label)

        # fig_pyr = plt.figure()
        # ax_pyr = fig_pyr.add_subplot(111, projection='3d')
        pix_pyr_t = []
        for index, vec in enumerate(pyramid_pixel_vectors):
            if index == 0:
                ax.quiver(
                    0,
                    0,
                    0,
                    vec.x * 1.15,
                    vec.y * 1.15,
                    vec.z * 1.15,
                    arrow_length_ratio=0.1,
                    color="olivedrab",
                    label="Untransformed Camera Vec Sample",
                )
                pix_pyr_t.append(cam_horizon_pose_transform.apply(vec.data.reshape(3)))
                ax.quiver(
                    0,
                    0,
                    0,
                    pix_pyr_t[index][0] * 1.15,
                    pix_pyr_t[index][1] * 1.15,
                    pix_pyr_t[index][2] * 1.15,
                    arrow_length_ratio=0.1,
                    color="black",
                    label="Transformed Camera Vec Sample",
                )
            else:
                ax.quiver(
                    0,
                    0,
                    0,
                    vec.x * 1.15,
                    vec.y * 1.15,
                    vec.z * 1.15,
                    arrow_length_ratio=0.1,
                    color="olivedrab",
                    label=None,
                )
                pix_pyr_t.append(cam_horizon_pose_transform.apply(vec.data.reshape(3)))
                ax.quiver(
                    0,
                    0,
                    0,
                    pix_pyr_t[index][0] * 1.15,
                    pix_pyr_t[index][1] * 1.15,
                    pix_pyr_t[index][2] * 1.15,
                    arrow_length_ratio=0.1,
                    color="black",
                    label=None,
                )

        '''
        for index, vec in enumerate(pyramid_pixel_vectors_2):
            if index == 0:
                ax.quiver(
                    0,
                    0,
                    0,
                    vec.x * 1.5,
                    vec.y * 1.5,
                    vec.z * 1.5,
                    arrow_length_ratio=0.1,
                    color="midnightblue",
                    label="New Cam Matrix",
                )
            else:
                ax.quiver(
                    0,
                    0,
                    0,
                    vec.x * 1.5,
                    vec.y * 1.5,
                    vec.z * 1.5,
                    arrow_length_ratio=0.1,
                    color="midnightblue",
                    label=None,
                )
        '''

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim([-1.25, 1.25])
        ax.set_ylim([-1.25, 1.25])
        ax.set_zlim([-1.25, 1.25])
        ax.set_aspect('equal')
        ax.legend()

    ##### extract data, apply sets of rotations (inverse) to convert horizonal data to camera coordinates and camera coordinates to mirror coordinates
    for pixel, details in vector_data.items():
        if isinstance(details, dict):
            if details["intersection_1"].size > 0:
                row, col = ast.literal_eval(pixel)
                cam_vec_original = get_pixel_pointing_vector(
                    pixel_directions=pixel_pointing, row=row, col=col, imagewidth=mask.shape[1]
                )
                cam_vec_horizon = cam_horizon_transform.apply(cam_vec_original.reshape(3))
                _, slope_1_corr, slope_2_corr = safe_calculate_slope(
                    cam_vec_horizon.reshape(3) * -1,
                    cam_vec_horizon.reshape(3) * -1,
                    details["intersection_1"],
                    details["intersection_2"],
                )
                angle_between = calculate_angle_between_vectors(
                    details['start_vector']['celestial_to_target'], details['end_vector']['celestial_to_target']
                )

                vector_data[pixel]["angle_between"] = angle_between
                vector_data[pixel]["observer_vector_camera_corrected"] = cam_vec_horizon
                vector_data[pixel]["H_slope_1_camera_corrected"] = slope_1_corr
                vector_data[pixel]["H_slope_2_camera_corrected"] = slope_2_corr
                vector_data[pixel]["C_slope_1_camera_corrected"] = cam_horizon_transform.inv().apply(slope_1_corr)
                vector_data[pixel]["C_slope_2_camera_corrected"] = cam_horizon_transform.inv().apply(slope_2_corr)
                vector_data[pixel]["M_point_location"] = (
                    r_optic_cam_refine_1.inv().apply(vector_data[pixel]["C_point_location"])
                    - v_cam_optic_cam_refine_1.data.T
                )
                vector_data[pixel]["M_slope_1_camera_corrected"] = r_optic_cam_refine_1.inv().apply(
                    vector_data[pixel]["C_slope_1_camera_corrected"]
                )
                vector_data[pixel]["M_slope_2_camera_corrected"] = r_optic_cam_refine_1.inv().apply(
                    vector_data[pixel]["C_slope_2_camera_corrected"]
                )
            else:
                continue
        else:
            pass

    ##### take extracted coords and slopes, rotate and align with mirror coordinate system and feed to sofast plotting
    # plot_heat_maps_horizonal_looking_up(vector_data)
    # plot_pixel_camera_and_mirror_coords(vector_data)
    # plot_heat_maps_camera_looking_up_coords(vector_data)
    plot_heat_maps_mirror_looking_up_coords(
        vector_data, output_dir=os.path.join(primary_folder, "9_sofast_data_compare"), debug_plots=False
    )
    # plot_slope_heat_maps_horizonal(data_dict=vector_data)
    # plot_slope_heat_maps_camera(data_dict=vector_data)
    # plot_slope_heat_maps_mirror(data_dict=vector_data)
    print("done")


if __name__ == "__main__":
    print("not intended to be run as a script any more. See lookfast_camera_adjust_V2.py instead.")
    main_original_script()
