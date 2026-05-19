import os
import time
import numpy as np
from logging import DEBUG, ERROR
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


import opencsp.app.sofast.lib.image_processing as imgp
import opencsp.app.lookback.lookback_tools as lbt

from opencsp.common.lib.camera.Camera import Camera

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render.view_spec as vs
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Uxyz import Uxyz


def compute_rotation_matrix(v_global, v_camera):
    """
    Compute the rotation matrix that aligns v_camera with v_global.
    Uses Rodrigues' rotation formula.
    """
    v_camera = v_camera.normalize()
    v_global = v_global.normalize()

    # Compute the axis of rotation (cross product)
    r = v_global.cross(v_camera)
    r_norm = np.linalg.norm([r.x, r.y, r.z])

    # If the vectors are parallel, no rotation is needed
    if r_norm == 0:
        return np.eye(3)

    r = r.normalize()  # Normalize the rotation axis

    # Compute the angle of rotation (dot product)
    cos_theta = v_global.dot(v_camera)
    theta = np.arccos(cos_theta)

    # Compute the skew-symmetric matrix of r
    r_x = np.array([[0, -r.z[0], r.y[0]], [r.z[0], 0, -r.x[0]], [-r.y[0], r.x[0], 0]])

    # Compute the rotation matrix using Rodrigues' formula
    R = np.eye(3) + np.sin(theta) * r_x + (1 - cos_theta) * np.dot(r_x, r_x)
    return R


def compute_rotation_matrix_sandia_ai(global_ref_vector: Uxyz, camera_ref_vector: Uxyz, camera_up_vector: Uxyz):
    """
    Computes the rotation matrix to transform vectors from the camera coordinate system
    to the global coordinate system.

    Parameters:
    global_ref_vector (numpy array): Reference vector in the global coordinate system.
    camera_ref_vector (numpy array): Reference vector in the camera coordinate system.
    camera_up_vector (numpy array): "Up" vector in the camera coordinate system (to resolve roll).

    Returns:
    numpy array: Rotation matrix to transform vectors from the camera coordinate system
                 to the global coordinate system.
    """
    # Normalize the input vectors to ensure they are unit vectors
    global_ref_vector = global_ref_vector.normalize()
    camera_ref_vector = camera_ref_vector.normalize()
    camera_up_vector = camera_up_vector / np.linalg.norm(camera_up_vector)

    # Step 1: Compute the rotation matrix to align the reference vectors
    # Compute the axis of rotation (cross product)
    r = global_ref_vector.cross(camera_ref_vector)
    r_norm = np.linalg.norm([r.x, r.y, r.z])
    if r_norm == 0:
        return np.eye(3)

    r = r.normalize()  # Normalize the rotation axis

    # Compute the angle of rotation (dot product)
    cos_theta = global_ref_vector.dot(camera_ref_vector)
    theta = np.arccos(cos_theta)

    # Compute the skew-symmetric matrix of r
    r_skew = np.array([[0, -r.z[0], r.y[0]], [r.z[0], 0, -r.x[0]], [-r.y[0], r.x[0], 0]])

    # Compute the rotation matrix using Rodrigues' formula
    Rotation_prime = np.eye(3) + np.sin(theta) * r_skew + (1 - cos_theta) * np.dot(r_skew, r_skew)

    # Step 2: Resolve the roll degree of freedom
    transformed_camera_up = np.dot(Rotation_prime, camera_up_vector)
    global_up_vector = np.array([0, 0, 1])  # Assuming global "up" is along the z-axis
    roll_axis = global_ref_vector
    roll_angle = np.arccos(np.dot(transformed_camera_up, global_up_vector))
    roll_direction = np.cross(transformed_camera_up, global_up_vector)
    if roll_axis.dot(Uxyz(roll_direction)) < 0:
        roll_angle = -roll_angle

    K_roll = np.array(
        [
            [0, -roll_axis.z[0], roll_axis.y[0]],
            [roll_axis.z[0], 0, -roll_axis.x[0]],
            [-roll_axis.y[0], roll_axis.x[0], 0],
        ]
    )
    R_roll = np.eye(3) + np.sin(roll_angle) * K_roll + (1 - np.cos(roll_angle)) * np.dot(K_roll, K_roll)

    R_final = np.dot(R_roll, Rotation_prime)
    return R_final


def error_function(camera_up_guess, global_ref_vector, camera_ref_vector):
    """
    Computes the error between the global reference vector and the transformed camera reference vector.

    Parameters:
    camera_up_guess (numpy array): Current guess for the camera "up" vector.
    global_ref_vector (numpy array): Reference vector in the global coordinate system.
    camera_ref_vector (numpy array): Reference vector in the camera coordinate system.

    Returns:
    float: Error metric (angular difference in radians).
    """
    camera_up_guess = camera_up_guess / np.linalg.norm(camera_up_guess)  # Normalize the guess
    rotation_matrix = compute_rotation_matrix_sandia_ai(global_ref_vector, camera_ref_vector, camera_up_guess)
    transformed_camera_ref = np.dot(
        rotation_matrix, np.array([camera_ref_vector.x, camera_ref_vector.y, camera_ref_vector.z])
    )
    error = np.arccos(global_ref_vector.dot(Uxyz(transformed_camera_ref)))
    return float(error[0])


def error_function_no_roll(global_ref_vector, camera_ref_vector):
    """
    Computes the error between the global reference vector and the transformed camera reference vector.

    Parameters:
    global_ref_vector (numpy array): Reference vector in the global coordinate system.
    camera_ref_vector (numpy array): Reference vector in the camera coordinate system.

    Returns:
    float: Error metric (angular difference in radians).
    """
    rotation_matrix = compute_rotation_matrix(global_ref_vector, camera_ref_vector)
    transformed_camera_ref = np.dot(
        rotation_matrix, np.array([camera_ref_vector.x, camera_ref_vector.y, camera_ref_vector.z])
    )
    error = np.arccos(global_ref_vector.dot(Uxyz(transformed_camera_ref)))
    return float(error[0])


def error_function_compute_transformed(rotation_object, vector_to_transform, reference_vector):
    """
    Returns:
    float: Error metric (angular difference in radians).
    """
    transformed_vec = rotation_object.apply(
        np.array([vector_to_transform.x[0], vector_to_transform.y[0], vector_to_transform.z[0]])
    )
    transformed_vec = transformed_vec / np.linalg.norm(transformed_vec)
    reference_vector.normalize()
    cos_theta = reference_vector.dot(Uxyz(transformed_vec))
    error = np.arccos(np.clip(cos_theta, -1, 1))
    return float(error[0])


def refine_camera_up_vector(global_ref_vector, camera_ref_vector, initial_camera_up_guess):
    """
    Refines the camera "up" vector to minimize the error between the global reference vector
    and the transformed camera reference vector.

    Parameters:
    global_ref_vector (numpy array): Reference vector in the global coordinate system.
    camera_ref_vector (numpy array): Reference vector in the camera coordinate system.
    initial_camera_up_guess (numpy array): Initial guess for the camera "up" vector.

    Returns:
    numpy array: Refined camera "up" vector.
    """
    result = minimize(
        error_function,
        np.array([initial_camera_up_guess.x, initial_camera_up_guess.y, initial_camera_up_guess.z]).flatten(),
        args=(global_ref_vector, camera_ref_vector),
        method='Nelder-Mead',
        tol=0.05,
        options={"disp": True, "xatol": 0.05, "fatol": 0.05},
    )
    refined_camera_up = result.x / np.linalg.norm(result.x)  # Normalize the result
    return refined_camera_up, result


def refine_camera_up_vector_manual_search(global_ref_vector, camera_ref_vector, camera_up_guess):
    starting_error = error_function(camera_up_guess, global_ref_vector, camera_ref_vector)

    current_error = starting_error
    current_up_guess = camera_up_guess.copy()
    iteration_count = 0
    while current_error >= 0.05:  # radians
        new_up_guesses = generate_cone_vectors(current_up_guess, current_error / 5, num_vectors=8)
        iteration_errors = []
        for guess in new_up_guesses:
            iteration_errors.append(error_function(guess, global_ref_vector, camera_ref_vector))

        min_index = iteration_errors.index(min(iteration_errors))
        error_change = current_error - iteration_errors[min_index]
        current_error = iteration_errors[min_index]
        current_up_guess = new_up_guesses[min_index]
        iteration_count += 1
        print(f"\nNum Iterations {iteration_count}")
        print(f"\nCurrent Guess {current_up_guess}")
        print(f"\nCurrent Error {current_error} [Radians]")
        print(f"\nChange in Error {error_change} [Radians]")

    return current_up_guess, current_error, starting_error


def generate_cone_vectors(axis_vector, cone_angle, num_vectors=8):
    """
    Generates unit vectors evenly spaced around a cone defined by the axis vector and cone angle.

    Parameters:
    axis_vector (numpy array): The axis vector defining the direction of the cone.
    cone_angle (float): The internal angle of the cone in radians.
    num_vectors (int): The number of unit vectors to generate around the cone.

    Returns:
    numpy array: Array of unit vectors evenly spaced around the cone.
    """
    # Normalize the axis vector to ensure it's a unit vector
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Generate an orthogonal basis to the axis vector
    # Find a vector not parallel to the axis vector
    if np.allclose(axis_vector, [1, 0, 0]):
        orthogonal_vector = np.array([0, 1, 0])
    else:
        orthogonal_vector = np.array([1, 0, 0])

    # Create two orthogonal vectors to the axis vector
    v1 = np.cross(axis_vector, orthogonal_vector)
    v1 = v1 / np.linalg.norm(v1)  # Normalize
    v2 = np.cross(axis_vector, v1)
    v2 = v2 / np.linalg.norm(v2)  # Normalize

    # Generate vectors around the cone
    cone_vectors = []
    for i in range(num_vectors):
        # Compute the angle for this vector around the cone
        theta = 2 * np.pi * i / num_vectors

        # Compute the vector direction
        cone_vector = np.cos(cone_angle) * axis_vector + np.sin(cone_angle) * (np.cos(theta) * v1 + np.sin(theta) * v2)
        cone_vector = cone_vector / np.linalg.norm(cone_vector)  # Normalize
        cone_vectors.append(cone_vector)

    return np.array(cone_vectors)
