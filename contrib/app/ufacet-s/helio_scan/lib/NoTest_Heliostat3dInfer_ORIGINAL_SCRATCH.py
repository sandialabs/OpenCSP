"""
Inferring heliostat 3-d geometry from tracked corners.



"""

import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

import opencsp.common.lib.tool.file_tools as ft
from opencsp.app.ufacets.helio_scan.lib.DEPRECATED_utils import *  # ?? SCAFFOLDING RCB -- ELIMINATE THIS
from opencsp.app.ufacets.helio_scan.lib.DEPRECATED_save_read import *  # ?? SCAFFOLDING RCB -- ELIMINATE THIS
import opencsp.app.ufacets.helio_scan.lib.FrameNameXyList as fnxl
import opencsp.app.ufacets.helio_scan.lib.ufacet_pipeline_frame as upf


class Heliostat3dInfer:
    """
    Infers heliostat 3-d geometry from tracked corners.
    """

    def __init__(
        self,
        # Execution control.
        iterations,
        cam_matrix,
        dist_coeff,
        # Input/output sources.
        input_video_body,  # Used for output figure file names.
        key_frame_projected_corners_fnxl,  # A FrameNameXyList object containing exactly two frames:  The key frame, and the frame that follows it.  Both with associated found projected corners.
        key_frame_confirmed_corners_fnxl,  # A FrameNameXyList object containing exactly two frames:  The key frame, and the frame that follows it.  Both with associated found confirmed corners.
        specifications,
        input_frame_dir,
        input_frame_id_format,
        all_frame_body_ext_list,
        output_construction_dir,
        # Render control.
        draw_track_images,
    ):
        # Execution control.
        self.iterations = iterations
        self.canny_levels = canny_levels
        self.solvePnPtype = solvePnPtype
        self.cam_matrix = cam_matrix
        self.dist_coeff = dist_coeff
        self.confirm_type = confirm_type
        # Input/outpout sources.
        self.input_video_body = input_video_body
        self.key_frame_projected_corners_fnxl = key_frame_projected_corners_fnxl
        self.specifications = specifications
        self.input_frame_dir = input_frame_dir
        self.input_frame_id_format = input_frame_id_format
        self.all_frame_body_ext_list = all_frame_body_ext_list
        self.output_construction_dir = output_construction_dir
        # Render control.
        self.draw_track_images = draw_track_images

        # Tracking exit control.
        self.minimum_fraction_of_confirmed_corners = MINIMUM_FRACTION_OF_CONFIRMED_CORNERS
        self.minimum_corners_required_inside_frame = math.ceil(
            MINIMUM_CORNERS_REQUIRED_INSIDE_FRAME * self.specifications.corners_per_heliostat
        )

        # Heliostat shape.
        self.corners3d = self.specifications.facets_corners

        # Extact key frame information.
        self.sorted_key_frame_ids = key_frame_projected_corners_fnxl.sorted_frame_id_list()
        self.key_frame_id_1 = self.sorted_key_frame_ids[0]
        self.key_frame_id_2 = self.sorted_key_frame_ids[1]
        self.key_frame_id_str_1 = upf.frame_id_str_given_frame_id(self.key_frame_id_1, self.input_frame_id_format)
        self.key_frame_id_str_2 = upf.frame_id_str_given_frame_id(self.key_frame_id_2, self.input_frame_id_format)
        self.key_frame_1_projected_list_of_name_xy_lists = key_frame_projected_corners_fnxl.list_of_name_xy_lists(
            self.key_frame_id_1
        )
        self.key_frame_2_projected_list_of_name_xy_lists = key_frame_projected_corners_fnxl.list_of_name_xy_lists(
            self.key_frame_id_2
        )
        self.key_frame_1_confirmed_list_of_name_xy_lists = key_frame_confirmed_corners_fnxl.list_of_name_xy_lists(
            self.key_frame_id_1
        )
        self.key_frame_2_confirmed_list_of_name_xy_lists = key_frame_confirmed_corners_fnxl.list_of_name_xy_lists(
            self.key_frame_id_2
        )
        self.heliostat_names = [name_xy_list[0] for name_xy_list in self.key_frame_1_projected_list_of_name_xy_lists]

        # Create output directory for frame figures.
        # Projected.
        self.output_projected_corners_dir = os.path.join(self.output_construction_dir, "projected")
        self.output_frame_projected_corners_dir = os.path.join(
            self.output_projected_corners_dir, self.key_frame_id_str_1
        )
        print(
            "In KeyFrameTracksSearch.__init__(), self.output_frame_projected_corners_dir =",
            self.output_frame_projected_corners_dir,
        )
        if self.draw_track_images:
            ft.create_directories_if_necessary(self.output_frame_projected_corners_dir)
        # Confirmed.
        self.output_confirmed_corners_dir = os.path.join(self.output_construction_dir, "confirmed")
        self.output_frame_confirmed_corners_dir = os.path.join(
            self.output_confirmed_corners_dir, self.key_frame_id_str_1
        )
        print(
            "In KeyFrameTracksSearch.__init__(), self.output_frame_confirmed_corners_dir =",
            self.output_frame_confirmed_corners_dir,
        )
        if self.draw_track_images:
            ft.create_directories_if_necessary(self.output_frame_confirmed_corners_dir)

        # Output FrameNameXyList objects for collecting tracks.
        self.key_frame_projected_track_fnxl = fnxl.FrameNameXyList()
        self.key_frame_confirmed_track_fnxl = fnxl.FrameNameXyList()

        # Perform the tracking in both directions.
        self.PredictConfirm(tracking_direction="forward")
        self.PredictConfirm(tracking_direction="backward")

    def PredictConfirm(self, tracking_direction):
        print('In Heliostat3dInfer.PredictConfirm(tracking_direction="' + tracking_direction + '")...')

        # Build image name list from image file list.
        all_frame_id_str_list = [
            upf.frame_id_str_given_frame_file_body_ext(body_ext) for body_ext in self.all_frame_body_ext_list
        ]
        if tracking_direction == "forward":
            # Velocity is from 1 --> 2.
            velocity, latest_projected_corners = self.velocity_and_initial_corners(
                self.key_frame_1_projected_list_of_name_xy_lists, self.key_frame_2_projected_list_of_name_xy_lists
            )
            # The image name list proceeds forward from key_frame 1.
            key_frame_id_str_1_idx = all_frame_id_str_list.index(self.key_frame_id_str_1)
            frame_id_str_sequence = all_frame_id_str_list[key_frame_id_str_1_idx:]
        elif tracking_direction == "backward":
            # Velocity is from 2 --> 1.
            velocity, latest_projected_corners = self.velocity_and_initial_corners(
                self.key_frame_2_projected_list_of_name_xy_lists, self.key_frame_1_projected_list_of_name_xy_lists
            )
            # The image name list proceeds backward from key_frame 2.
            all_frame_id_str_list_reverse = copy.copy(all_frame_id_str_list)
            all_frame_id_str_list_reverse.reverse()
            key_frame_id_str_2_idx = all_frame_id_str_list_reverse.index(self.key_frame_id_str_2)
            frame_id_str_sequence = all_frame_id_str_list_reverse[key_frame_id_str_2_idx:]
        else:
            raise ValueError(
                'ERROR: In Heliostat3dInfer.PredictConfirm(), unexpected tracking_direction="'
                + str(tracking_direction)
                + '" encountered(1).  Allowable values are "forward" and "backward" only (2).'
            )

        # Initialize track collection variables.
        num_hel = len(self.key_frame_1_projected_list_of_name_xy_lists)
        stop_track_flags = [False for _ in range(0, num_hel)]  # flags for tracking
        skip_flag = []

        # If this is a forward track, add and draw the seed images and corners.
        if tracking_direction == "forward":
            # Key frame 1.
            self.key_frame_projected_track_fnxl.add_list_of_name_xy_lists(
                self.key_frame_id_1, self.key_frame_1_projected_list_of_name_xy_lists
            )
            self.key_frame_confirmed_track_fnxl.add_list_of_name_xy_lists(
                self.key_frame_id_1, self.key_frame_1_confirmed_list_of_name_xy_lists
            )
            self.draw_frame_with_points_if_desired(
                self.key_frame_id_str_1, self.key_frame_1_projected_list_of_name_xy_lists, "Projected", point_color="g"
            )
            self.draw_frame_with_points_if_desired(
                self.key_frame_id_str_1, self.key_frame_1_confirmed_list_of_name_xy_lists, "Confirmed", point_color="b"
            )
            # Key frame 2.
            self.key_frame_projected_track_fnxl.add_list_of_name_xy_lists(
                self.key_frame_id_2, self.key_frame_2_projected_list_of_name_xy_lists
            )
            self.key_frame_confirmed_track_fnxl.add_list_of_name_xy_lists(
                self.key_frame_id_2, self.key_frame_2_confirmed_list_of_name_xy_lists
            )
            self.draw_frame_with_points_if_desired(
                self.key_frame_id_str_2,
                self.key_frame_2_projected_list_of_name_xy_lists,
                "Projected",
                point_color="orange",
            )
            self.draw_frame_with_points_if_desired(
                self.key_frame_id_str_2,
                self.key_frame_2_confirmed_list_of_name_xy_lists,
                "Confirmed",
                point_color="yellow",
            )

        """Tracking"""
        for frame_id_str in frame_id_str_sequence[2:]:
            # print('In Heliostat3dInfer.PredictConfirm(), for key_frame_id='+self.key_frame_id_str_1+', '+str(tracking_direction)+' tracking corners into image: '+frame_id_str)
            img = None
            frame_body_ext = upf.frame_file_body_ext_given_frame_id_str(self.input_video_body, frame_id_str)
            frame_dir_body_ext = os.path.join(self.input_frame_dir, frame_body_ext)
            if os.path.exists(frame_dir_body_ext):
                img = cv.imread(frame_dir_body_ext)
            if img is None:  # you skip that image
                skip_flag.append(True)
                continue
            else:
                skip_flag.append(False)
                if len(skip_flag) >= 2 and skip_flag[-2]:
                    cnt = 1
                    for i in range(len(skip_flag) - 2, 0, -1):
                        if skip_flag[i]:
                            cnt += 1
                        else:
                            break
                    for hel_indx in range(0, num_hel):
                        for vel_indx in range(0, len(velocity[hel_indx])):
                            new_vel = [velocity[hel_indx][vel_indx][0] * cnt, velocity[hel_indx][vel_indx][1] * cnt]
                            velocity[hel_indx][vel_indx] = new_vel

            """Edge Detection based on Image"""
            img = cv.GaussianBlur(img, (5, 5), 0)
            cnt = 0

            projected_list_of_name_xy_lists = []  # For adding to the FrameNameXyList object.
            confirmed_list_of_name_xy_lists = []  # For adding to the FrameNameXyList object.
            for hel_indx in range(0, num_hel):
                if stop_track_flags[hel_indx]:
                    continue
                """Predict Corners"""
                predicted_corners = self.predict_corners(velocity, latest_projected_corners, hel_indx)
                n_inside = self.number_of_predicted_corners_inside_frame(img, predicted_corners)
                # print('In Heliostat3dInfer.PredictConfirm(), n_inside =', n_inside, ';  self.minimum_corners_required_inside_frame =', self.minimum_corners_required_inside_frame)
                if n_inside < self.minimum_corners_required_inside_frame:
                    """ending criterion"""
                    stop_track_flags[hel_indx] = True
                    continue
                """Confirm Corners"""
                (projected_corners, confirmed_corners, num_non_None_confirmed_corners) = self.confirm_corners(
                    img,
                    predicted_corners,
                    iterations=self.iterations,
                    canny_levels=self.canny_levels,
                    confirm_type=self.confirm_type,
                )
                confirmed_ratio = num_non_None_confirmed_corners / float(
                    n_inside
                )  # Don't penalize for not confirming corners outside frame.
                # print('In Heliostat3dInfer.PredictConfirm(), confirmed_ratio =', confirmed_ratio, ';  self.minimum_fraction_of_confirmed_corners =', self.minimum_fraction_of_confirmed_corners)
                if confirmed_ratio < self.minimum_fraction_of_confirmed_corners:
                    """ending criterion"""
                    stop_track_flags[hel_indx] = True
                    continue
                # store the tracks
                projected_list_of_name_xy_lists.append(
                    [self.heliostat_names[hel_indx], projected_corners]
                )  # For adding to the FrameNameXyList object.
                confirmed_list_of_name_xy_lists.append(
                    [self.heliostat_names[hel_indx], confirmed_corners]
                )  # For adding to the FrameNameXyList object.
                # update
                velocity[hel_indx] = self.update_velocity(
                    velocity, predicted_corners, projected_corners, hel_indx, skip_flag
                )

                latest_projected_corners[hel_indx] = projected_corners

            if sum(stop_track_flags) == num_hel:
                break

            # Add to FrameNameXyList objects.
            self.key_frame_projected_track_fnxl.add_list_of_name_xy_lists(
                upf.frame_id_given_frame_id_str(frame_id_str), projected_list_of_name_xy_lists
            )
            self.key_frame_confirmed_track_fnxl.add_list_of_name_xy_lists(
                upf.frame_id_given_frame_id_str(frame_id_str), confirmed_list_of_name_xy_lists
            )
            # Draw.
            self.draw_frame_with_points_if_desired_aux(
                img, frame_id_str, projected_list_of_name_xy_lists, "Projected", point_color="m"
            )
            self.draw_frame_with_points_if_desired_aux(
                img, frame_id_str, confirmed_list_of_name_xy_lists, "Confirmed", point_color="c"
            )

    def velocity_and_initial_corners(self, key_frame_A_list_of_name_xy_lists, key_frame_B_list_of_name_xy_lists):
        """
        Motion is from A --> B.
        Reurns point-by-point velocity vector in image coordinates, and also the set of points
        after the A--> B transition, which forms the starting point for the tracking.
        """
        velocity = []
        latest_projected_corners = []
        for name_xy_list_A, name_xy_list_B in zip(key_frame_A_list_of_name_xy_lists, key_frame_B_list_of_name_xy_lists):
            name_A = name_xy_list_A[0]
            xy_list_A = name_xy_list_A[1]
            name_B = name_xy_list_B[0]
            xy_list_B = name_xy_list_B[1]
            if name_A != name_B:
                msg = (
                    'In KeyFrameTrackAndSearch.velocity_and_initial_corners(), encountered mismatched name_A="'
                    + str(name_A)
                    + '" and name_B="'
                    + str(name_B)
                    + '".'
                )
                print("ERROR: " + msg)
                raise ValueError(msg)
            vxy_list = []
            for xy_A, xy_B in zip(xy_list_A, xy_list_B):
                vel_x = xy_B[0] - xy_A[0]
                vel_y = xy_B[1] - xy_A[1]
                vxy_list.append([vel_x, vel_y])
            velocity.append(vxy_list)
            latest_projected_corners.append(xy_list_B)
        return velocity, latest_projected_corners

    def draw_frame_with_points_if_desired(
        self, frame_id_str, list_of_name_xy_lists, Confirmed_or_Projected_str, point_color
    ):
        if self.draw_track_images:
            frame_body_ext = upf.frame_file_body_ext_given_frame_id_str(self.input_video_body, frame_id_str)
            img = cv.imread(os.path.join(self.input_frame_dir, frame_body_ext))
            self.draw_frame_with_points_if_desired_aux(
                img, frame_id_str, list_of_name_xy_lists, Confirmed_or_Projected_str, point_color
            )

    def draw_frame_with_points_if_desired_aux(
        self, img, frame_id_str, list_of_name_xy_lists, Confirmed_or_Projected_str, point_color
    ):
        if self.draw_track_images:
            plt.figure()
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            for name_xy_list in list_of_name_xy_lists:
                for corner in name_xy_list[1]:
                    if corner is not None:
                        plt.scatter(corner[0], corner[1], facecolor=point_color, s=1)
            plt.title(
                Confirmed_or_Projected_str
                + " Points for Key Frame "
                + self.key_frame_id_str_1
                + ", Frame: "
                + frame_id_str
            )
            fig_file_body_ext = (
                self.input_video_body + "_" + frame_id_str + "_" + Confirmed_or_Projected_str.lower() + ".png"
            )
            if Confirmed_or_Projected_str == "Projected":
                figure_output_dir = self.output_frame_projected_corners_dir
            elif Confirmed_or_Projected_str == "Confirmed":
                figure_output_dir = self.output_frame_confirmed_corners_dir
            else:
                msg = (
                    "In Heliostat3dInfer.draw_frame_with_points(), encountered unexpected Confirmed_or_Projected_str = "
                    + str(Confirmed_or_Projected_str)
                )
                print(msg)
                raise ValueError(msg)
            fig_file_dir_body_ext = os.path.join(figure_output_dir, fig_file_body_ext)
            # print('In Heliostat3dInfer(), saving figure file: ', fig_file_dir_body_ext)
            plt.savefig(fig_file_dir_body_ext, dpi=250)
            plt.close()

    def predict_corners(self, velocity, latest_projected_corners, hel_indx):
        predicted_corners = []
        velocity = velocity[hel_indx]
        latest_projected_corners = latest_projected_corners[hel_indx]
        for corner_indx in range(0, len(latest_projected_corners)):
            pred_col = latest_projected_corners[corner_indx][0] + velocity[corner_indx][0]
            pred_row = latest_projected_corners[corner_indx][1] + velocity[corner_indx][1]
            predicted_corners.append([pred_col, pred_row])
        return predicted_corners

    def update_velocity(self, velocity, predicted_corners, confirmed_corners, hel_indx, skip_flag):
        previous_velocity = velocity[hel_indx]
        deltas_adjust = []
        for conf_corner, pred_corner in zip(confirmed_corners, predicted_corners):
            dx_adjust = conf_corner[0] - pred_corner[0]
            dy_adjust = conf_corner[1] - pred_corner[1]
            deltas_adjust.append([dx_adjust, dy_adjust])
            pass

        new_velocity = []
        for velocity, delta_adjust in zip(previous_velocity, deltas_adjust):
            dx, dy = velocity
            dx_adjust, dy_adjust = delta_adjust
            dx_new = dx + dx_adjust
            dy_new = dy + dy_adjust
            new_velocity.append([dx_new, dy_new])

        if len(skip_flag) >= 2 and skip_flag[-2]:  # skipped
            cnt = 1
            for i in range(len(skip_flag) - 2, 0, -1):
                if skip_flag[i]:
                    cnt += 1
                else:
                    break
            for vel_indx in range(0, len(new_velocity)):
                new_velocity[vel_indx] = [new_velocity[vel_indx][0] / cnt, new_velocity[vel_indx][1] / cnt]

        return new_velocity

    def number_of_predicted_corners_inside_frame(self, img, predicted_corners):
        max_row = img.shape[0]
        max_col = img.shape[1]
        cnt = 0
        for corner in predicted_corners:
            col, row = corner
            if ((0 <= col) and (col < max_col)) and ((0 <= row) and (row < max_row)):
                cnt += 1
        return cnt

    def confirm_corners(
        self, img, predicted_corners, canny_levels, iterations, confirm_type="", tolerance=3, pixels=100
    ):
        def confirm_facets(expected_corners, edges, tolerance, pixels):
            confirmed_facets = {}
            for indx in range(0, len(expected_corners), self.specifications.corners_per_facet):
                facet_id = indx // self.specifications.corners_per_facet
                corners = [expected_corners[indx + i] for i in range(0, self.specifications.corners_per_facet)]
                for corner_indx in range(0, len(corners)):
                    corner = corners[corner_indx]
                    if corner[0] >= max_col or corner[0] < 0 or corner[1] >= max_row or corner[1] < 0:
                        corners[corner_indx] = None
                confirmed_facets[facet_id] = {"edges": confirm_facet_edges(corners, edges, tolerance, pixels)}
            return confirmed_facets

        def confirm_facet_edges(corners, edges, tolerance, pixels):
            confirmed_edges = []
            corners.append(corners[0])  # cyclic
            for indx in range(0, len(corners) - 1):
                corner1 = corners[indx]
                corner2 = corners[indx + 1]
                if corner1 is None or corner2 is None:
                    confirmed_edges.append(None)
                    continue
                # edge coefficients
                A, B, C = find_hom_line_2points(corner1, corner2)
                if A is None:
                    continue
                min_col, max_col, min_row, max_row = min_max_col_row(edges, corner1, corner2)
                edge_pixels = []
                # confirming
                if indx % 2 == 0:
                    for row in range(min_row, max_row):
                        for col in range(min_col, max_col):
                            dist = abs(A * col + B * row + C)
                            if edges[row][col] and dist <= tolerance:
                                edge_pixels.append([col, row])
                else:
                    for col in range(min_col, max_col):
                        for row in range(min_row, max_row):
                            dist = abs(A * col + B * row + C)
                            if edges[row][col] and dist <= tolerance:
                                edge_pixels.append([col, row])
                if len(edge_pixels) < pixels:
                    confirmed_edges.append(None)  # edge was not confirmed
                    continue

                # confirmed edge
                edge_coeff = fit_line_pixels(edge_pixels)
                edge_inliers_coeff = fit_line_inliers_pixels(edge_pixels, edge_coeff)
                confirmed_edges.append(edge_inliers_coeff)
            return confirmed_edges

        def find_corners(confirmed_facets):
            hel_corners = [None for _ in range(0, self.specifications.corners_per_heliostat)]
            for facet_indx, facet in confirmed_facets.items():
                corners = []
                edges = facet["edges"]
                edges.append(edges[0])  # cyclic
                for edge_indx in range(0, len(edges) - 1):
                    edge0 = edges[edge_indx]
                    edge1 = edges[edge_indx + 1]
                    if edge0 is not None and edge1 is not None:
                        corners.append(findIntersectionLines(edge0, edge1))
                    else:
                        corners.append(None)
                corners.insert(0, corners.pop())
                indx = facet_indx * self.specifications.corners_per_facet
                for i, j in zip(
                    range(indx, indx + self.specifications.corners_per_facet),
                    range(0, self.specifications.corners_per_facet),
                ):
                    hel_corners[i] = corners[j]
            return hel_corners

        def construct_points(confirmed_corners, corners3d):
            imgcorners = []
            objcorners = []
            for indx in range(0, len(confirmed_corners)):
                if confirmed_corners[indx] is not None:
                    imgcorners.append(confirmed_corners[indx])
                    objcorners.append(corners3d[indx])

            points3d = np.array(objcorners).astype("float32")
            points2d = np.array(imgcorners).astype("float32")

            return points3d, points2d

        h, w = img.shape[:2]
        max_row = img.shape[0]
        max_col = img.shape[1]
        expected_corners = predicted_corners
        canny_types = canny_levels
        corners3d = self.corners3d
        previous_confirmed_corners = []
        previous_expected_corners = []

        if confirm_type == "iterative":
            upper_downlimit, upper_uplimit = 50, 100
            lower_downlimit, lower_uplimit = 25, 50
            canny_steps = 5
            # edge images
            upper_step = int((upper_uplimit - upper_downlimit) / canny_steps)
            lower_step = int((lower_uplimit - lower_downlimit) / canny_steps)
            edges_list = []
            lower, upper = lower_uplimit, upper_uplimit
            for step in range(0, canny_steps + 1):
                upper = upper - step * upper_step
                lower = lower - step * lower_step
                edges_list.append(CannyImg(img, lower=lower, upper=upper))
        for ite in range(0, iterations):
            if confirm_type == "iterative":
                inner_loop_range = len(edges_list)
            else:
                inner_loop_range = len(canny_types)
            for i in range(0, inner_loop_range):  # levels of canny
                if confirm_type == "iterative":
                    edges = edges_list[i]
                else:
                    edges = CannyImg(img, canny_type=canny_types[i])

                confirmed_facets = confirm_facets(expected_corners, edges, tolerance, pixels)
                confirmed_corners = find_corners(confirmed_facets)
                flag_break = True
                flag_less_than_6 = False
                for corner in confirmed_corners:  # not confirmed corner
                    flag_break *= corner is None
                if flag_break:
                    expected_corners = []
                    break
                points3d, points2d = construct_points(confirmed_corners, corners3d)
                if len(points2d) < 6:
                    if previous_confirmed_corners:
                        confirmed_corners = previous_confirmed_corners
                    if previous_expected_corners:
                        expected_corners = previous_expected_corners
                    flag_less_than_6 = True
                    break

                # NOTE:  This code passage is very confusing, because it looks like solvePNP() is refining the camera matrix
                # and distortion coefficients where in fact under current calling conditions it is receiving "None" as input
                # for both these parameters, and then appears to set them to the default values provided in utils.py, and
                # not modifying them after that.  Whether that it true or not depends on whether OpenCV's function is
                # modifying its arguments a a side effect, which I need to test to determine.  Caveat!
                mtx, dist, rvec, tvec, error = solvePNP(
                    points3d,
                    points2d,
                    h,
                    w,
                    pnptype=self.solvePnPtype,
                    cam_matrix=self.cam_matrix,
                    dist_coeff=self.dist_coeff,
                )

                expected_corners, _ = cv.projectPoints(np.array(corners3d).astype("float32"), rvec, tvec, mtx, dist)
                expected_corners = expected_corners.reshape(-1, 2)
                expected_corners = expected_corners.tolist()
                previous_expected_corners = expected_corners.copy()
                previous_confirmed_corners = confirmed_corners.copy()

            if self.solvePnPtype == "calib" and ite == 0:
                self.solvePnPtype = "pnp"  # change of mode
                self.cam_matrix, self.dist_coeff = (
                    mtx,
                    dist,
                )  # save the current camera model and use that for the rest of the tracking

            if flag_break or flag_less_than_6:
                break

        # Prepare return values.
        projected_corners = previous_expected_corners  # Return the corners from the previous iteration, since we may have bailed from this iteration.
        confirmed_corners = previous_confirmed_corners  # Return the corners from the previous iteration, since we may have bailed from this iteration.
        num_non_None_confirmed_corners = len(
            [x for x in previous_confirmed_corners if x is not None]
        )  # Don't count "None" corners.

        # print('In Heliostat3dInfer.confirm_corners(), upon exit, num_non_None_confirmed_corners =', num_non_None_confirmed_corners)  # ?? SCAFFOLDING RCB -- TEMPORARY
        return projected_corners, confirmed_corners, num_non_None_confirmed_corners
