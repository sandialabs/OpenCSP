import csv
import os
from opencsp.app.ufacets.helio_scan.lib.DEPRECATED_utils import *


def save_connected_components(components=None, filename=None, path=None):
    csv_path = os.path.join(path, "csv_files")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    output_file = os.path.join(csv_path, filename)
    with open(output_file, "w", newline="") as output:
        writer = csv.writer(output)
        for component in components:
            row_output = []
            row_output.append("color")
            row_output.append(str(list(component["color"])))
            row_output.append("original_pixels")
            row_output.append(str(list(component["original_pixels"])))
            row_output.append("boundary_type")
            row_output.append(str(component["boundary_type"]))
            writer.writerow(row_output)


def read_connected_components(filename=None, path=None):
    def csv_row_to_connected_component(input_row):
        n_claus = len(input_row)
        if (n_claus % 2) != 0:
            # print('ERROR: In csv_row_to_connected_component(), input row has odd number of clauses.  n_claus = ', n_claus)
            assert False
        component = {}
        i = 0
        while i < n_claus:
            key = input_row[i]
            value = input_row[i + 1]
            if value != "left" and value != "right" and value != "top" and value != "bottom":
                value = eval(input_row[i + 1])
            component[key] = value
            i += 2
        return component

    # Construct the full input path.
    csv_path = os.path.join(path, "csv_files")
    input_file = os.path.join(csv_path, filename)
    # Check if the input file exists.
    if not os.path.exists(input_file):
        raise OSError("In read_connected_components(), file does not exist: " + str(input_file))
    # Open and read the file.
    components = []
    with open(input_file, newline="") as input_stream:
        # print('Opened successfully.')
        reader = csv.reader(input_stream)
        for input_row in reader:
            component = csv_row_to_connected_component(input_row)
            components.append(component)
    return components


def save_fitted_lines_connected_components(components=None, filename=None, path=None):
    csv_path = os.path.join(path, "csv_files")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    output_file = os.path.join(csv_path, filename)
    with open(output_file, "w", newline="") as output:
        writer = csv.writer(output)
        for component in components:
            row_output = []
            row_output.append("color")
            row_output.append(str(list(component["color"])))
            row_output.append("original_pixels")
            row_output.append(str(list(component["original_pixels"])))
            row_output.append("boundary_type")
            row_output.append(str(component["boundary_type"]))
            row_output.append("original_line_hom_coef")
            row_output.append(str(list(component["original_line_hom_coef"])))
            row_output.append("original_line_residual")
            row_output.append(str(component["original_line_residual"]))
            row_output.append("original_line_points")
            row_output.append(str(list(component["original_line_points"])))
            writer.writerow(row_output)


def save_fitted_lines_inliers_connected_components(components=None, filename=None, path=None):
    csv_path = os.path.join(path, "csv_files")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    output_file = os.path.join(csv_path, filename)
    with open(output_file, "w", newline="") as output:
        writer = csv.writer(output)
        for component in components:
            row_output = []
            row_output.append("color")
            row_output.append(str(list(component["color"])))
            row_output.append("boundary_type")
            row_output.append(str(component["boundary_type"]))

            row_output.append("original_pixels")
            row_output.append(str(list(component["original_pixels"])))

            row_output.append("inliers_pixels")
            row_output.append(str(list(component["inliers_pixels"])))

            row_output.append("outliers_pixels")
            row_output.append(str(list(component["outliers_pixels"])))

            row_output.append("original_line_hom_coef")
            row_output.append(str(list(component["original_line_hom_coef"])))

            row_output.append("inliers_line_hom_coef")
            row_output.append(str(list(component["inliers_line_hom_coef"])))

            row_output.append("original_line_residual")
            row_output.append(str(component["original_line_residual"]))

            row_output.append("inliers_line_residual")
            row_output.append(str(component["inliers_line_residual"]))

            row_output.append("original_line_points")
            row_output.append(str(list(component["original_line_points"])))

            row_output.append("inliers_line_points")
            row_output.append(str(list(component["inliers_line_points"])))

            row_output.append("tolerance")
            row_output.append(str(component["tolerance"]))
            writer.writerow(row_output)


def save_corners_facets(corners=None, filename=None, path=None, corners_type="top_left", facets=None):
    csv_path = os.path.join(path, "csv_files")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    output_file = os.path.join(csv_path, filename)
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        if corners is None and facets is not None:
            corners = []
            for facet in facets:
                corners.append(facet["top_left"])
                corners.append(facet["top_right"])
                corners.append(facet["bottom_right"])
                corners.append(facet["bottom_left"])
                corners.append(facet["center"])
                if "pairs" in facet:
                    corners.append(tuple(facet["pairs"]))
        for corner in corners:
            row_output = []
            if isinstance(corner, tuple):
                row_output.append("pairs")
                row_output.append(str(list(corner)))
                writer.writerow(row_output)
                continue

            if isinstance(corner, list):
                row_output.append("center")
                row_output.append(str(list(corner)))
                writer.writerow(row_output)
                continue

            corners_type = corner["corner_type"]
            row_output.append("corner_type")
            row_output.append(str(corner["corner_type"]))
            row_output.append("point")
            row_output.append(str(list(corner["point"])))
            key1 = "edge_coeff"
            key2 = "edge_pixels"
            key3 = "edge_points"
            if corners_type == "top_left":
                prefix1 = "left_"
                prefix2 = "top_"
            elif corners_type == "top_right":
                prefix1 = "top_"
                prefix2 = "right_"
            elif corners_type == "bottom_right":
                prefix1 = "right_"
                prefix2 = "bottom_"
            elif corners_type == "bottom_left":
                prefix1 = "bottom_"
                prefix2 = "left_"

            row_output.append(prefix1 + key1)
            row_output.append(str(list(corner[prefix1 + key1])))
            row_output.append(prefix1 + key2)
            row_output.append(str(list(corner[prefix1 + key2])))
            row_output.append(prefix1 + key3)
            row_output.append(str(list(corner[prefix1 + key3])))

            row_output.append(prefix2 + key1)
            row_output.append(str(list(corner[prefix2 + key1])))
            row_output.append(prefix2 + key2)
            row_output.append(str(list(corner[prefix2 + key2])))
            row_output.append(prefix2 + key3)
            row_output.append(str(list(corner[prefix2 + key3])))
            writer.writerow(row_output)


def read_corners_facets(filename=None, path=None, facets_flag=False):
    def csv_row_to_corner(input_row):
        n_claus = len(input_row)
        if (n_claus % 2) != 0:
            # print('ERROR: In csv_row_to_connected_component(), input row has odd number of clauses.  n_claus = ', n_claus)
            assert False
        corner = {}
        i = 0
        while i < n_claus:
            key = input_row[i]
            value = input_row[i + 1]
            if key == "corner_type":
                corner[key] = value
            else:
                corner[key] = eval(value)
            i += 2
        return corner

    # Construct the full input path.
    csv_path = os.path.join(path, "csv_files")
    input_file = os.path.join(csv_path, filename)
    # Check if the input file exists.
    if not os.path.exists(input_file):
        raise OSError('In read_corners_facets(), file "' + str(input_file) + '" does not exist.')
    # Open and read the file.
    corners = []
    with open(input_file, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for input_row in reader:
            corner = csv_row_to_corner(input_row)
            corners.append(corner)

    if facets_flag:
        facets = []
        facet = {}
        for corner_indx in range(0, len(corners)):
            corner = corners[corner_indx]
            if "pairs" in corner:
                facet["pairs"] = corner["pairs"]
                facets.append(facet)
                facet = {}
            elif "center" in corner:
                facet["center"] = corner["center"]
                if corner_indx + 1 < len(corners):
                    next_corner = corners[corner_indx + 1]
                    if "pairs" not in next_corner:
                        facets.append(facet)
                        facet = {}
            else:
                corner_type = corner["corner_type"]
                # print(corner_type)
                if corner_type == "top_left":
                    facet["top_left"] = corner
                elif corner_type == "top_right":
                    facet["top_right"] = corner
                elif corner_type == "bottom_right":
                    facet["bottom_right"] = corner
                elif corner_type == "bottom_left":
                    facet["bottom_left"] = corner
        if facet:
            facets.append(facet)
        return facets
    return corners


def save_facets(facets=None, filename=None, path=None):
    csv_path = os.path.join(path, "csv_files")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    output_file = os.path.join(csv_path, filename)
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        for facet in facets:
            top_left_corner = facet["top_left"]
            top_right_corner = facet["top_right"]
            bottom_right_corner = facet["bottom_right"]
            bottom_left_corner = facet["bottom_left"]
            center = facet["center"]
            corners = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
            for corner in corners:
                row_output = []
                row_output.append("corner_type")
                row_output.append(str(corner["corner_type"]))
                row_output.append("point")
                row_output.append(str(list(corner["point"])))
                key1 = "edge_coeff"
                key2 = "edge_pixels"
                key3 = "edge_points"
                corners_type = corner[
                    "corner_type"
                ]  # ?? SCAFFOLDING RCB -- INSERTED THIS LINE BLINDLY WITHOUT THOROUGH STUDY, TO ELIMINATE COMPILER WARNING.
                if corners_type == "top_left":
                    prefix1 = "left_"
                    prefix2 = "top_"
                elif corners_type == "top_right":
                    prefix1 = "top_"
                    prefix2 = "right_"
                elif corners_type == "bottom_right":
                    prefix1 = "right_"
                    prefix2 = "bottom_"
                elif corners_type == "bottom_left":
                    prefix1 = "bottom_"
                    prefix2 = "left_"

                row_output.append(prefix1 + key1)
                row_output.append(str(list(corner[prefix1 + key1])))
                row_output.append(prefix1 + key2)
                row_output.append(str(list(corner[prefix1 + key2])))
                row_output.append(prefix1 + key3)
                row_output.append(str(list(corner[prefix1 + key3])))

                row_output.append(prefix2 + key1)
                row_output.append(str(list(corner[prefix2 + key1])))
                row_output.append(prefix2 + key2)
                row_output.append(str(list(corner[prefix2 + key2])))
                row_output.append(prefix2 + key3)
                row_output.append(str(list(corner[prefix2 + key3])))
                writer.writerow(row_output)

    return 0


def centers3d_to_corners3d(facet_centers, facet_width, facet_height):
    width = facet_width  # facet width
    height = facet_height  # facet height

    corner_offsets = np.array(
        [
            [-width / 2, height / 2, 0],
            [width / 2, height / 2, 0],
            [width / 2, -height / 2, 0],
            [-width / 2, -height / 2, 0],
        ]
    )

    facet_centers = np.array(facet_centers)
    corners = []
    for row in range(0, facet_centers.shape[0]):
        facet_center = facet_centers[row, :]
        facet_corners = (facet_center + corner_offsets).tolist()
        for facet_corner in facet_corners:
            corners.append(facet_corner)

    return corners


def read_centers3d(input_file):
    # Check if the input file exists.
    if not os.path.exists(input_file):
        raise OSError('In read_centers3d(), file "' + str(input_file) + '" does not exist.')
    # Open and read the file.
    facets_coords = []
    with open(input_file, newline="") as csvfile:
        # print('Opened sucessfully')
        reader = csv.reader(csvfile)
        count = 0
        for input_row in reader:
            if not count:
                count += 1
                continue  # get rid of header
            _, x, y, z = (input_row[0], float(input_row[1]), float(input_row[2]), float(input_row[3]))
            facets_coords.append([x, y, z])

    return facets_coords


def save_corners(corners, filename, path):
    csv_path = os.path.join(path, "csv_files")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    output_file = os.path.join(csv_path, filename)
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(corners)


def read_projected_corners(filename=None, corners_per_heliostat=None, path=None):
    # Construct the full input path.
    input_file = os.path.join(path, "csv_files", filename)
    # Check if the input file exists.
    if not os.path.exists(input_file):
        raise OSError("In read_projected_corners(), file does not exist: " + str(input_file))
    # Open and read the file.
    with open(input_file, newline="") as csvfile:
        reader = csv.reader(csvfile)
        corners = list(reader)

    proj_corners = []
    hel_corners = [[]]
    for corner in corners:
        c = [eval(corner[0]), eval(corner[1])]
        if len(hel_corners[-1]) < corners_per_heliostat:
            hel_corners[-1].append(c)
        else:
            hel_corners.append([c])
    return hel_corners
