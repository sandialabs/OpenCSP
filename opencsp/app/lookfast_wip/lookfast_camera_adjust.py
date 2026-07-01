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


def compute_perspective_transform(source_points, destination_points):
    perp_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    return perp_matrix


def compute_homography_with_length_scale(pixel_coords, destination_points):
    """
    rect_width, rect_height
    Compute the homography matrix for perspective correction using real-world dimensions
    and pixel coordinates to minimize distortion.

    Parameters:
        pixel_coords (numpy.ndarray): A 4x2 array of pixel coordinates in the image
                                      corresponding to the corners of the subject.
                                      Clockwise definition of points starting from the bottom left corner
                                      The order should be bottom-left, top-left, top-right, bottom-right.
        rect_width (float): The width of the subject in real-world dimensions.
        rect_height (float): The height of the subject in real-world dimensions.

    Returns:
        numpy.ndarray: The 3x3 homography matrix.
    """
    if pixel_coords.shape != (4, 2):
        raise ValueError("pixel_coords must be a 4x2 array of corner coordinates.")
    '''
    # Compute pixel distances between corners
    top_edge_length = np.linalg.norm(pixel_coords[2] - pixel_coords[1])  # Top-left to top-right
    bottom_edge_length = np.linalg.norm(pixel_coords[3] - pixel_coords[0])  # Bottom-right to bottom-left
    left_edge_length = np.linalg.norm(pixel_coords[1] - pixel_coords[0])  # Bottom-left to top-left
    right_edge_length = np.linalg.norm(pixel_coords[3] - pixel_coords[2])  # Bottom-right to top-right

    # Compute scaling factors based on real-world dimensions
    scale_x_top = rect_width / top_edge_length
    scale_x_bottom = rect_width / bottom_edge_length
    scale_y_left = rect_height / left_edge_length
    scale_y_right = rect_height / right_edge_length

     # Define destination points as a perfect rectangle with real-world dimensions
    rect_coords = np.array(
        [
            [0, 0],  # Bottom-left corner
            [0, rect_height],  # Top-left corner
            [rect_width, rect_height],  # Top-right corner
            [rect_width, 0],  # Bottom-right corner
        ],
        dtype=np.float32,
    )
    '''

    # Compute the homography matrix
    homography_matrix, _ = cv2.findHomography(pixel_coords.astype(np.float32), destination_points)

    return homography_matrix


def warp_image(opencv_transform_matrix, image_to_warp, show_plot=False, plot_name="This Image was Warped"):
    warped_image = cv2.warpPerspective(
        image_to_warp, opencv_transform_matrix, (image_to_warp.shape[1], image_to_warp.shape[0])
    )
    if show_plot:
        cv2.imshow(plot_name, warped_image)
    return warped_image


def main():
    primary_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025"
    video_name = "DSC_0025.MOV"
    checkpoint_folder = os.path.join(primary_folder, "0_checkpoints")
    checkpoint_main_name = "lookback_main_checkpoint.json"

    original_data_location = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025_Final_ExEx/8_pixel_vector_information/debug/pixel_vector_information_wslope_debug.json.gz"
    vector_data = lbt.read_compressed_json(original_data_location)

    video_metadata = lbt.extract_detailed_video_metadata(os.path.join(primary_folder, video_name))

    light_image = cv2.imread(os.path.join(primary_folder, "light_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    dark_image = cv2.imread(os.path.join(primary_folder, "dark_mask_test.png"), cv2.IMREAD_GRAYSCALE)

    all_pixels = np.ones(shape=light_image.shape, dtype=bool)
    # expected_corners_manual = Vxy(list(zip([643, 840], [444, 860], [455, 1062], [657, 1047])), dtype=int)
    expected_corners_manual = Vxy(list(zip([860, 444], [840, 643], [1047, 657], [1062, 455])), dtype=int)
    # expected_corners_manual = Vxy(list(zip([449, 865], [460, 1057], [652, 1042], [638, 845])), dtype=int)

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

    v_corners_image = imgp.refine_facet_corners(
        Puv_facet_corns_exp=expected_corners_manual,
        Puv_cent=v_mask_centroid_image,
        Puv_edges=v_edges_image,
        step=20,
        d_perp=20,
        frac_keep=1,
    )

    mirror_bl = v_corners_image.vertices[2].data
    mirror_tl = v_corners_image.vertices[1].data
    mirror_tr = v_corners_image.vertices[0].data
    mirror_br = v_corners_image.vertices[3].data

    square_points = np.array(
        [
            [860, 440],  # Bottom-left corner
            [860, 640],  # Top-left corner
            [1060, 640],  # Top-right corner
            [1060, 440],  # Bottom-right corner
        ],
        dtype=np.float32,
    )

    homography_matrix = compute_homography_with_length_scale(
        np.array([mirror_bl, mirror_tl, mirror_tr, mirror_br], dtype=np.float32).reshape(4, 2),
        destination_points=square_points,
    )

    perspective_matrix = compute_perspective_transform(
        source_points=np.array([mirror_bl, mirror_tl, mirror_tr, mirror_br], dtype=np.float32).reshape(4, 2),
        destination_points=square_points,
    )

    cv2.imshow("Dark Image", dark_image)
    cv2.imshow("Light Image", light_image)
    cv2.imshow("Mask", mask_image)
    warp_image(homography_matrix, mask_image, show_plot=True, plot_name="Homography Warp")
    warp_image(perspective_matrix, mask_image, show_plot=True, plot_name="Perspective Warp")

    # Arbitrary Camera Intrinsic matrix
    K = np.array([[1, 0, 1920 / 2], [0, 1, 1080 / 2], [0, 0, 1]])

    # Distortion coefficients
    D = np.array([0.1, -0.05, 0.001, 0.001])

    cam = Camera(intrinsic_mat=K, distortion_coef=D, image_shape_xy=tuple[1920, 1080], name="Arbitrary_Example")

    pixel_pointing = imgp.calculate_active_pixels_vectors(mask=all_pixels, camera=cam)

    # reference_pixel = (670, 950)
    row_range = (600, 900)
    col_range = (750, 1100)
    pixel_locations = [
        (row, col) for row in range(row_range[0], row_range[1]) for col in range(col_range[0], col_range[1])
    ]
    # reference_vector_camera = pixel_pointing[reference_pixel[0] * light_image.shape[1] + reference_pixel[1]]
    reference_vector_camera = []
    for row, col in pixel_locations:
        reference_vector_camera.append(pixel_pointing[row * light_image.shape[1] + col])

    reference_vector_horizon = Uxyz(vector_data["(500, 900)"]["observer_vector"] * -1)

    initial_error_estimate = []
    for cam_vec in reference_vector_camera:
        _, rssd = rotation_matrix_scipy(
            np.array([cam_vec.x[0], cam_vec.y[0], cam_vec.z[0]]),
            np.array([reference_vector_horizon.x[0], reference_vector_horizon.y[0], reference_vector_horizon.z[0]]),
        )
        initial_error_estimate.append(rssd)
        # rotated_cam_vec = rot_obj.apply(np.array([cam_vec.x[0], cam_vec.y[0], cam_vec.z[0]]))
        # initial_error_estimate.append(error_function_compute_transformed(rot_obj, cam_vec, reference_vector_horizon))
        # initial_error_estimate.append(error_function_no_roll(reference_vector_horizon, cam_vec))

    min_ref_cam_vec_index = initial_error_estimate.index(min(initial_error_estimate))
    working_cam_reference_vector = reference_vector_camera[min_ref_cam_vec_index]
    rotation_object, _ = rotation_matrix_scipy(
        np.array(
            [working_cam_reference_vector.x[0], working_cam_reference_vector.y[0], working_cam_reference_vector.z[0]]
        ),
        np.array([reference_vector_horizon.x[0], reference_vector_horizon.y[0], reference_vector_horizon.z[0]]),
    )

    transformed_cam_reference_vec = rotation_object.apply(
        np.array(
            [working_cam_reference_vector.x[0], working_cam_reference_vector.y[0], working_cam_reference_vector.z[0]]
        )
    )
    reference_pixel = pixel_locations[min_ref_cam_vec_index]

    cam_point_north_up_start = np.array([0, -1, 0])
    cam_elevation = rotate_vector(cam_point_north_up_start, np.array([1, 0, 0]), degrees=28)
    cam_up_initial_guess = rotate_vector(cam_elevation, np.array([0, 1, 0]), degrees=56)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vectors_data = [
        [0, 0, 0, 1, 0, 0, "Cam X", "r"],
        [0, 0, 0, 0, 1, 0, "Cam Y", "g"],
        [0, 0, 0, 0, 0, 1, "Cam Z", "b"],
        [0, 0, 0, 0, -1, 0, "Global Start Up", "y"],
        [0, 0, 0, cam_elevation[0], cam_elevation[1], cam_elevation[2], "Global Up Elevation Rotation", "k"],
        [
            0,
            0,
            0,
            cam_up_initial_guess[0],
            cam_up_initial_guess[1],
            cam_up_initial_guess[2],
            "Global Up Initial Guess",
            "m",
        ],
        [
            0,
            0,
            0,
            working_cam_reference_vector.x[0],
            working_cam_reference_vector.y[0],
            working_cam_reference_vector.z[0],
            "Working Cam Reference",
            "darkorange",
        ],
        [
            0,
            0,
            0,
            transformed_cam_reference_vec[0],
            transformed_cam_reference_vec[1],
            transformed_cam_reference_vec[2],
            "Transformed Cam Reference",
            "gold",
        ],
    ]
    for ox, oy, oz, dx, dy, dz, label, color in vectors_data:
        # Plot the vector using quiver
        ax.quiver(ox, oy, oz, dx, dy, dz, color=color, arrow_length_ratio=0.1, label=label)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([-1.25, 1.25])
    ax.set_zlim([-1.25, 1.25])
    ax.set_aspect('equal')
    ax.legend()

    plt.show()

    '''

    

    camera_up_refined_manual, ending_search_error, starting_search_error = refine_camera_up_vector_manual_search(
        reference_vector_horizon, working_cam_reference_vector, cam_up_initial_guess
    )

    rotation_matrix = compute_rotation_matrix_sandia_ai(
        reference_vector_horizon, reference_vector_camera, camera_up_refined_manual
    )

    converted_camera_vector = np.dot(
        rotation_matrix, np.array([reference_vector_camera.x, reference_vector_camera.y, reference_vector_camera.z])
    )
    
    refined_camera_up, mimization_results = refine_camera_up_vector(
        reference_vector_horizon, reference_vector_camera, Uxyz(cam_up_initial_guess)
    )
    ax.quiver(
        0,
        0,
        0,
        refined_camera_up[0],
        refined_camera_up[1],
        refined_camera_up[2],
        color="c",
        arrow_length_ratio=0.1,
        label="Refined Global Up Cam Coords",
    )


    R_matrix = compute_rotation_matrix(reference_vector_horizon, reference_vector_camera)

    new_cam_vec = np.dot(
        R_matrix, [reference_vector_camera.x[0], reference_vector_camera.y[0], reference_vector_camera.z[0]]
    )
    '''

    '''
    fig_control = rcfg.RenderControlFigure()
    axs_control = rca.RenderControlAxis()
    style = rcps.RenderControlPointSeq(color=None, marker=None)
    fig_rec = fm.setup_figure_for_3d_data(
        figure_control=fig_control, axis_control=axs_control, name="test", view_spec=vs.view_spec_3d()
    )
    # fig = v3d.View3d(fig_rec, axis=axs_control)
    pixel_pointing.draw_points(fig_rec, style=style)  # why is no figure actually generated?
    '''
    print(f"Homography Matrix\n {homography_matrix}")
    print(f"Perspective Matrix\n {perspective_matrix}")
    print(f"Vector Alignment Matrix\n {rotation_object.as_matrix()}")
    print("done")


if __name__ == "__main__":
    main()
