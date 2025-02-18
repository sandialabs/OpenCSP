"""
Creates a 2D color target

"""

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def construct_blue_under_red_cross_green(
    nx, ny
):  # ?? SCAFFOLDING RCB -- MAKE PARAMETER NAMES CONSISTENT WITH CONTEXT CODE
    """
    # ?? SCAFFOLDING RCB -- IMPROVE THIS COMMENT

    Blue underlying red cross green.
    """
    print("In construct_blue_under_red_cross_green()...")  # ?? SCAFFOLDING RCB -- TEMPORARY

    # Create an empty image.
    n_rows = ny
    n_cols = nx
    image = np.zeros([n_rows, n_cols, 3])

    # Size in (x,y) coordinates.
    x_max = n_cols
    y_max = n_rows

    # Max intensity.
    max_intensity = 1.0

    # FIll in blue underlying red cross green
    for row in range(0, n_rows):
        for col in range(0, n_cols):
            x = col
            y = n_rows - row
            x_frac = x / x_max
            y_frac = y / y_max
            diagonal_frac = np.sqrt(x * x + y * y) / np.sqrt(x_max * x_max + y_max * y_max)
            image[row, col, 0] = x_frac * max_intensity
            image[row, col, 1] = y_frac * max_intensity
            image[row, col, 2] = (1 - diagonal_frac) * max_intensity

    # Convert to uint8
    image_uint8 = np.uint8(image * 255)

    return image_uint8


def construct_rgb_cube_inscribed_square_image(
    nx, ny, project_to_cube
):  # ?? SCAFFOLDING RCB -- MAKE PARAMETER NAMES CONSISTENT WITH CONTEXT CODE
    """
    # ?? SCAFFOLDING RCB -- IMPROVE THIS COMMENT

    Image defined by a square inscribed in the hexagon in [R,G,B] space formed by
    the Red, Green, and Blue basis vectors, and their pairwise combinations to
    form Cyan, Magenta, and Yellow secondary vectors.
    """
    print("In construct_rgb_cube_inscribed_square_image()...")  # ?? SCAFFOLDING RCB -- TEMPORARY

    # Create container for color vectors
    l = 0.4
    X, Y = np.meshgrid(np.linspace(-l, l, nx), np.linspace(-l, l, ny))
    vecs_centerd = np.array([X.flatten(), Y.flatten(), np.zeros(X.size)]).T  # Nx3

    #  Rotate points to be normal to white (1, 1, 1)
    rot = Rotation.from_euler(seq="zyz", angles=[90, 45, 45], degrees=True)
    vecs = rot.apply(vecs_centerd)  # Nx3

    # Add offset along white (1, 1, 1) direction
    vecs += np.array([[0.5, 0.5, 0.5]])

    # Create mask for non-valid points
    mask = (
        (vecs[:, 0] > 1) + (vecs[:, 0] < 0) + (vecs[:, 1] > 1) + (vecs[:, 1] < 0) + (vecs[:, 2] > 1) + (vecs[:, 2] < 0)
    )

    # Apply mask
    vecs[mask] = np.nan

    # Reshape vector into 3D array
    R = vecs[:, 0].reshape((ny, nx, 1))
    G = vecs[:, 1].reshape((ny, nx, 1))
    B = vecs[:, 2].reshape((ny, nx, 1))

    image = np.concatenate((R, G, B), 2)

    # Scale bands to saturate colors
    if project_to_cube:
        image /= np.nanmax(image, 2)[:, :, None]

    # Convert to uint8
    image_uint8 = np.uint8(image * 255)

    return image_uint8


if __name__ == "__main__":
    print("Running Braden's original code! (target_color_2d_rgb.py)")

    # Define number of sample points of greater image
    nx = 500
    ny = 500

    # Create container for color vectors
    l = 0.4
    X, Y = np.meshgrid(np.linspace(-l, l, nx), np.linspace(-l, l, ny))
    vecs_centerd = np.array([X.flatten(), Y.flatten(), np.zeros(X.size)]).T  # Nx3

    #  Rotate points to be normal to white (1, 1, 1)
    rot = Rotation.from_euler(seq="zyz", angles=[90, 45, 45], degrees=True)
    vecs = rot.apply(vecs_centerd)  # Nx3

    # Add offset along white (1, 1, 1) direction
    vecs += np.array([[0.5, 0.5, 0.5]])

    # Create mask for non-valid points
    mask = (
        (vecs[:, 0] > 1) + (vecs[:, 0] < 0) + (vecs[:, 1] > 1) + (vecs[:, 1] < 0) + (vecs[:, 2] > 1) + (vecs[:, 2] < 0)
    )

    # Apply mask
    vecs[mask] = np.nan

    # Reshape vector into 3D array
    R = vecs[:, 0].reshape((ny, nx, 1))
    G = vecs[:, 1].reshape((ny, nx, 1))
    B = vecs[:, 2].reshape((ny, nx, 1))

    image = np.concatenate((R, G, B), 2)

    # Scale bands to saturate colors
    # image /= np.nanmax(image, 2)[:, :, None]

    # Convert to uint8
    image_uint8 = np.uint8(image * 255)

    # Save target
    imageio.imwrite("target_color_2d_rgb.png", image_uint8)

    # Plot points in 3D color space
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(*vecs.T, marker='.')
    # ax.plot([0, 1], [0, 0], [0, 0], color='r')
    # ax.plot([0, 0], [0, 1], [0, 0], color='g')
    # ax.plot([0, 0], [0, 0], [0, 1], color='b')
    # ax.scatter(0.5, 0.5, 0.5)
    # plt.show()

    # Plot color image
    # plt.imshow(image_uint8)
    # plt.show()
