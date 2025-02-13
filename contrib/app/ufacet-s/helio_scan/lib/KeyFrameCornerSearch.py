"""
Searching for facet corners in a key frame of a UFACET scan video.



"""

from typing import Any, Union, NewType
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from cv2 import cv2 as cv
from numpy.lib.type_check import imag

import opencsp.common.lib.geometry.geometry_2d as g2d
from DEPRECATED_utils import *  # ?? SCAFFOLDING RCB -- ELIMINATE THIS
from DEPRECATED_save_read import *  # ?? SCAFFOLDING RCB -- ELIMINATE THIS
import FrameNameXyList as fnxl
from opencsp.common.lib.render_control.RenderControlKeyCorners import RenderControlKeyCorners

Component = NewType("Component", dict[str, Union[str, list[int], list[float], list[list[int]]]])


class KeyFrameCornerSearch:
    """
    Class executing search for facet corners in a key frame of a UFACET scan video.
    """

    # CONSTRUCTION

    def __init__(
        self,
        # Problem definition.
        key_frame_id,  # Numerical key frame index.  Uniquely determines the frame within the video.
        key_frame_id_str,  # Not the same as str(key_frame_id), because this includes the proper number of leading zeros, etc.
        key_frame_img: np.ndarray,  # The key frame image, already loaded.
        list_of_name_polygons: list[tuple[str, list[list[int]]]],  # List of expected [hel_name, polygon] pairs.
        specifications,  # Solar field specifications.  # ?? SCAFFOLDING RCB -- REPLACE THIS WITH MASTER INFORMATION LOADED FROM DISK FILES.
        # Input/output sources.
        output_construction_dir,  # Where to save the detailed image processing step-by-step plots.
        solvePnPtype,  # how to solve PnP.  Values are 'pnp' and 'calib'
        # Render control.
        render_control: RenderControlKeyCorners,
    ):  # Flags to control rendering on this run.
        """Search the given key_frame_img for one heliostat per list_of_name_polygons.
        The results can be accessed by:
            First check successful(), then
            Retrieve results with projected_fnxl()
        """

        print("In KeyFrameCornerSearch.__init__()...")  # ?? SCAFFOLDING RCB -- TEMPORARY

        # Store input.
        self.key_frame_id = key_frame_id
        self.key_frame_id_str = key_frame_id_str
        self.key_frame_img = key_frame_img
        self.list_of_name_polygons = list_of_name_polygons
        self.specifications = specifications
        self.output_construction_dir = output_construction_dir
        self.solvePnPtype = solvePnPtype
        self.render_control = render_control

        self.frame = {  # ?? SCAFFOLDING RCB -- DO WE STILL NEED THIS FRAME DATA STRUCTURE?  SHOULD WE STORE IN SELF INSTEAD?
            "key_frame_img": key_frame_img,  # ?? SCAFFOLDING RCB -- DO WE STILL NEED THIS FRAME DATA STRUCTURE?  SHOULD WE STORE IN SELF INSTEAD?
            "output_construction_dir": output_construction_dir,  # ?? SCAFFOLDING RCB -- DO WE STILL NEED THIS FRAME DATA STRUCTURE?  SHOULD WE STORE IN SELF INSTEAD?
        }  # ?? SCAFFOLDING RCB -- DO WE STILL NEED THIS FRAME DATA STRUCTURE?  SHOULD WE STORE IN SELF INSTEAD?

        print(
            "In KeyFrameCornerSearch.__init__(), performing full image analysis..."
        )  # ?? SCAFFOLDING RCB -- TEMPORARY

        # Input polygons
        self.draw_img_polygons()
        # Edge detection
        self.frame["edges"], self.frame["edges_img"] = self.canny()
        # Sky detection
        # self.frame['sky'],     \
        # self.frame['sky_img']                          = self.skyhsv()  # ?? SCAFFOLDING RCB -- PREVIOUS VERSION
        self.frame["sky"], self.frame["sky_img"] = self.sky()
        # Facet boundaries
        self.frame["boundaries"], self.frame["boundaries_img"] = self.facet_boundaries()
        # Connected_components
        (self.frame["components"], self.frame["components_img"]) = self.connected_components()
        # Filtered connected_components
        (self.frame["filt_components"], self.frame["filt_components_img"]) = self.filter_connected_components()
        # TODO BGB make sure none of the components bridge the gap between mirrors
        # Fitted lines connected components
        self.frame["fitted_lines_components"] = self.fitted_lines_connected_components()
        # Line inliers
        self.frame["fitted_lines_inliers_components"] = self.fitted_lines_inliers_components()
        # Corners
        self.frame["corners"] = self.find_corners()
        # Facets
        self.frame["facets"] = self.facets()
        # Filter facets based on polygons
        (self.frame["filtered_facets"], self.frame["heliostats"]) = (
            self.filter_facets_polygons()
        )  # Initial setting of self.frame['heliostats']
        # Top row
        self.top_row_facets()  # Updates self.frame['heliostats']
        # Register top row
        self.classify_top_row_facets()  # Updates self.frame['heliostats']
        # Projected Corners
        self.project_and_confirm(
            iterations=5,  # Updates: self.frame['heliostats'], self.frame['all_projected_corners']
            canny_levels=["auto", "light", "lighter"],
        )  #

    # ACCESS

    def all_confirmed_corners(self):
        return self.frame["all_confirmed_corners"]

    def all_projected_corners(self):
        return self.frame["all_projected_corners"]

    def confirmed_fnxl(self) -> fnxl.FrameNameXyList:
        return self.frame["confirmed_fnxl"]

    def projected_fnxl(self) -> fnxl.FrameNameXyList:
        return self.frame["projected_fnxl"]

    def successful(self):
        """
        Returns true if the image processing successfully produced final corners.
        """
        return ("all_projected_corners" in self.frame) and (len(self.frame["all_projected_corners"]) > 0)

    # IMAGE PROCESSING

    def draw_img_polygons(self):
        print("In KeyFrameCornerSearch.draw_img_polygons(), entering routine...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        if self.render_control.draw_img_box:
            img = self.frame["key_frame_img"]
            plt.figure()
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

            for (
                name_polygon
            ) in self.list_of_name_polygons:  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                name = name_polygon[0]  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                polygon = name_polygon[1]  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                color = "g"  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                # Draw the polygon.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                closed_xy_list = polygon.copy()
                closed_xy_list.append(polygon[0])
                x_list = [
                    pt[0] for pt in closed_xy_list
                ]  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                y_list = [
                    pt[1] for pt in closed_xy_list
                ]  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                plt.plot(
                    x_list, y_list, color=color
                )  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                # Draw the heliostat name.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                if (
                    len(polygon) > 0
                ):  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                    label_xy = g2d.label_point(
                        polygon
                    )  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                    plt.text(
                        label_xy[
                            0
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                        label_xy[
                            1
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                        name,  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                        color=color,  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                        horizontalalignment="center",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                        verticalalignment="center",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                        fontsize=10,  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                        fontstyle="normal",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.
                        fontweight="bold",
                    )  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.  # ?? SCAFFOLDING RCB -- INTEGRATRE THIS WITH STANDARD FNXL RENDERING.

            plt.savefig(
                os.path.join(self.frame["output_construction_dir"], self.key_frame_id_str + "_00_img_box.png"), dpi=500
            )
            plt.close()

    def canny(self, img=None):
        print("In KeyFrameCornerSearch.canny()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        if img is None:
            img = self.frame[
                "key_frame_img"
            ]  # ?? SCAFFOLDING RCB -- SEE CANNY() AND CONFIRM() ROUTINES FOR DUPLICATE PLACES WHERE THIS CODE IS PLACED.  MUST KEEP CONSISTENT.
            img = cv.GaussianBlur(
                img, (5, 5), 0
            )  # ?? SCAFFOLDING RCB -- SEE CANNY() AND CONFIRM() ROUTINES FOR DUPLICATE PLACES WHERE THIS CODE IS PLACED.  MUST KEEP CONSISTENT.

        edges = CannyImg(
            img=img, canny_type="auto"
        )  # ! auto  # ?? SCAFFOLDING RCB -- ORIGINAL CODE was 'light'  # ?? SCAFFOLDING RCB -- SEE CANNY() AND CONFIRM() ROUTINES FOR DUPLICATE PLACES WHERE THIS CODE IS PLACED.  MUST KEEP CONSISTENT.
        # edges = CannyImg(img=img, canny_type='light') # ! auto  # ?? SCAFFOLDING RCB -- ORIGINAL CODE
        row, col = np.nonzero(edges)
        edges = np.zeros((img.shape[0], img.shape[1])).astype("int")
        edges[row, col] = 1
        edge_img = 0 * self.frame["key_frame_img"]
        edge_img[row, col, :] = EDGE_COLOR

        if self.render_control.draw_edge:
            save_image(
                img=(edges * 255),
                imgname=self.key_frame_id_str + "_01_edge.png",
                path=self.frame["output_construction_dir"],
            )  # ?? SCAFFOLDING RCB -- MULTIPLIED BY 255 TO MAKE IMAGE VISIBLE IN PREVIEW.
        if self.render_control.draw_edge_fig:
            save_fig(
                img=edges,
                imgname=self.key_frame_id_str + "_01_edge_fig.png",
                path=self.frame["output_construction_dir"],
                rgb=True,
                dpi=1000,
            )  # ?? SCAFFOLDING RCB -- SETTING RGB=TRUE, TO SUPPRESS CV CALL WITHIN SAVE_FIG() TO CONVERT FROM BGR TO RGB.  THIS CRASHES ON THIS INPUT.
        if self.render_control.draw_edge_img:
            save_image(
                img=edge_img,
                imgname=self.key_frame_id_str + "_01_edge_img.png",
                path=self.frame["output_construction_dir"],
            )
        if self.render_control.draw_edge_img_fig:
            save_fig(
                img=edge_img,
                imgname=self.key_frame_id_str + "_01_edge_img_fig.png",
                path=self.frame["output_construction_dir"],
                rgb=True,
                dpi=1000,
            )

        return edges, edge_img

    def skyhsv(self):
        print("In KeyFrameCornerSearch.skyhsv()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        img = self.frame["key_frame_img"]
        sky, sky_img = sky_with_hsv(img=img, rgb=False)

        # img_rgb         = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # hsv_img         = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
        # light_sky       = (100, 30, 100) #(100, 30, 1)
        # dark_sky        = (150, 140, 255) #(150, 140, 255)
        # sky             = cv.inRange(hsv_img, light_sky, dark_sky)
        # sky_img         = cv.bitwise_and(img_rgb, img_rgb, mask=sky)
        if self.render_control.draw_skyhsv:
            save_image(
                img=sky, imgname=self.key_frame_id_str + "_02_skyhsv.png", path=self.frame["output_construction_dir"]
            )
        if self.render_control.draw_skyhsv_fig:
            save_fig(
                img=sky,
                imgname=self.key_frame_id_str + "_02_skyhsv_fig.png",
                path=self.frame["output_construction_dir"],
                rgb=True,
                dpi=1000,
            )
        if self.render_control.draw_skyhsv_img:
            save_image(
                img=sky_img,
                imgname=self.key_frame_id_str + "_02_skyhsv_img.png",
                path=self.frame["output_construction_dir"],
            )
        if self.render_control.draw_skyhsv_img_fig:
            save_fig(
                img=sky_img,
                imgname=self.key_frame_id_str + "_02_skyhsv_img_fig.png",
                path=self.frame["output_construction_dir"],
                rgb=True,
                dpi=1000,
            )
        return sky, sky_img

    def sky(self):
        print("In KeyFrameCornerSearch.sky()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        img = self.frame["key_frame_img"]
        sky_img = img.copy()

        b = sky_img[:, :, 0] / 255.0
        g = sky_img[:, :, 1] / 255.0
        r = sky_img[:, :, 2] / 255.0
        # Identify sky
        sky_x, sky_y = np.nonzero(b + g + r > SKY_THRESHOLD)
        sky = np.zeros((img.shape[0], img.shape[1])).astype("int")
        sky[sky_x, sky_y] = 1

        sky_img[sky_x, sky_y, 0] = SKY_COLOR[0]
        sky_img[sky_x, sky_y, 1] = SKY_COLOR[1]
        sky_img[sky_x, sky_y, 2] = SKY_COLOR[2]

        if self.render_control.draw_sky:
            save_image(
                img=(sky * 255),
                imgname=self.key_frame_id_str + "_02_sky.png",
                path=self.frame["output_construction_dir"],
            )  # ?? SCAFFOLDING RCB -- MULTIPLIED BY 255 TO MAKE IMAGE VISIBLE IN PREVIEW.
        if self.render_control.draw_sky_fig:
            save_fig(
                img=sky,
                imgname=self.key_frame_id_str + "_02_sky_fig.png",
                path=self.frame["output_construction_dir"],
                rgb=True,
                dpi=1000,
            )  # ?? SCAFFOLDING RCB -- SETTING RGB=TRUE, TO SUPPRESS CV CALL WITHIN SAVE_FIG() TO CONVERT FROM BGR TO RGB.  THIS CRASHES ON THIS INPUT.
        if self.render_control.draw_sky_img:
            save_image(
                img=sky_img,
                imgname=self.key_frame_id_str + "_02_sky_img.png",
                path=self.frame["output_construction_dir"],
            )
        if self.render_control.draw_sky_img_fig:
            save_fig(
                img=sky_img,
                imgname=self.key_frame_id_str + "_02_sky_img_fig.png",
                path=self.frame["output_construction_dir"],
                rgb=True,
                dpi=1000,
            )

        return sky, sky_img

    def facet_boundaries(self):
        """Colors pixels based on if they match is_boundary_pixel(...).

        Returns
        -------
        boundaries: a 0 (not a boundary pixel) or 1 (boundary pixel) ndarray that is the same size as self.frame['key_frame_img']
        boundaries_img: an ndarray with with boundary pixels colored based on whether they are a top/left/right/bottom edge pixel
        """
        print("In KeyFrameCornerSearch.facet_boundaries()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        img = self.frame["key_frame_img"]
        edges = self.frame["edges"]

        row_edges, col_edges = np.nonzero(edges)
        print(
            "In KeyFrameCornerSearch.facet_boundaries(), number of edge pixels len(row_edges) =", len(row_edges)
        )  # ?? SCAFFOLDING RCB -- TEMPORARY
        boundaries_img = 0 * img
        boundaries_rows = []
        boundaries_cols = []
        for row, col in zip(row_edges, col_edges):
            # TODO this for loop could be easily optimized by creating an intermediary image that is colored based on how many adjacent vertical or horizontal sky pixels there are
            # determine if such an optimization is worth doing
            # ~BGB20221003
            is_boundary = False
            # Left
            if (
                self.is_boundary_pixel(row - 1, col, "left")
                and self.is_boundary_pixel(row, col, "left")
                and self.is_boundary_pixel(row + 1, col, "left")
            ):
                boundaries_img[row, col, :] = LEFT_BOUNDARY_COLOR
                is_boundary = True
            # Right
            elif (
                self.is_boundary_pixel(row - 1, col, "right")
                and self.is_boundary_pixel(row, col, "right")
                and self.is_boundary_pixel(row + 1, col, "right")
            ):
                boundaries_img[row, col, :] = RIGHT_BOUNDARY_COLOR
                is_boundary = True
            # Top
            elif (
                self.is_boundary_pixel(row, col - 1, "top")
                and self.is_boundary_pixel(row, col, "top")
                and self.is_boundary_pixel(row, col + 1, "top")
            ):
                boundaries_img[row, col, :] = TOP_BOUNDARY_COLOR
                is_boundary = True
            # Bottom
            elif (
                self.is_boundary_pixel(row, col - 1, "bottom")
                and self.is_boundary_pixel(row, col, "bottom")
                and self.is_boundary_pixel(row, col + 1, "bottom")
            ):
                boundaries_img[row, col, :] = BOTTOM_BOUNDARY_COLOR
                is_boundary = True

            if is_boundary:
                boundaries_rows.append(row)
                boundaries_cols.append(col)

        if self.render_control.draw_boundaries:
            save_image(
                img=boundaries_img,
                imgname=self.key_frame_id_str + "_03_boundaries.png",
                path=self.frame["output_construction_dir"],
            )
        if self.render_control.draw_boundaries_fig:
            save_fig(
                img=boundaries_img,
                imgname=self.key_frame_id_str + "_03_boundaries_fig.png",
                path=self.frame["output_construction_dir"],
                dpi=1000,
            )

        boundaries = np.zeros((img.shape[0], img.shape[1])).astype("int")
        boundaries[boundaries_rows, boundaries_cols] = 1
        return boundaries, boundaries_img

    def is_boundary_pixel(
        self, row: int, col: int, btype: str, required_sky_width: int = None, ignore_margin: int = None
    ) -> bool:
        """Checks if the pixel at the given row/col is a mirror edge boundary pixel (it is assumed to be an edge pixel).

        Parameters
        ----------
            btype: Which side of the mirror this pixel is on, one of 'left', 'top', 'right', or 'bottom'
            required_sky_width: How many pixels of sky must be adjacent to this pixel
            ignore_margin: How many pixels of edge are assumed to be next to this pixel
        """
        edges = self.frame["edges"]
        sky = self.frame["sky"]
        img = self.frame["key_frame_img"]

        if required_sky_width is None:
            required_sky_width = REQUIRED_SKY_WIDTH
        if ignore_margin is None:
            ignore_margin = IGNORE_MARGIN
        # if required_non_sky_width is None:
        #     required_non_sky_width  = REQUIRED_NON_SKY_WIDTH

        max_row = img.shape[0]
        max_col = img.shape[1]

        if row < 0 or row >= max_row or col < 0 or col >= max_col:
            return False
        if btype == "left":
            low = col + ignore_margin
            high = col + ignore_margin + required_sky_width
            max_indx = max_col
        elif btype == "top":
            low = row + ignore_margin
            high = row + ignore_margin + required_sky_width
            max_indx = max_row
        elif btype == "right":
            low = col - (ignore_margin + required_sky_width)
            high = col - ignore_margin
            max_indx = max_col
        elif btype == "bottom":
            low = row - (ignore_margin + required_sky_width)
            high = row - ignore_margin
            max_indx = max_row

        for indx in range(low, high):
            if (indx < 0) or (indx >= max_indx):
                return False
            if btype == "left" or btype == "right":
                is_sky = sky[row, indx]
                is_edge = edges[row, indx]
            else:
                is_sky = sky[indx, col]
                is_edge = edges[indx, col]

            if is_edge or not is_sky:
                return False
        return True

    def connected_components(self) -> tuple[list[Component], np.ndarray]:
        """Interpret the facet edges as "components" (groups of same-colored pixels).

        Returns
        -------
            components: the dict['original_pixels'] entries contains the list of component pixels.
            component_img: the image with the components drawn on top of it."""

        def construct_component(row: int, col: int, btype: str, color: list[int], img: np.ndarray) -> Component:
            """Builds out a list of adjacent pixels that all have the same color (including diagonals).

            Parameters
            ----------
                btype: one of 'left', 'right', 'top', or 'bottom'
                color: the rgb color that corresponds to the given btype

            Returns
            -------
                component: dict with key 'original_pixels': a list of xy pairs"""
            component = {"color": color, "boundary_type": btype, "original_pixels": []}
            horizon = [[row, col]]
            max_row = img.shape[0]
            max_col = img.shape[1]
            while len(horizon) > 0:
                void_color = [0, 0, 0]
                pixel_to_add = horizon.pop()
                component["original_pixels"].append(pixel_to_add)
                r, c = pixel_to_add[0], pixel_to_add[1]
                img[r, c, :] = void_color
                # add neighbors
                if (
                    ((r - 1) >= 0)
                    and ((c - 1) >= 0)
                    and (img[r - 1, c - 1, :] == color).all(axis=-1)
                    and [r - 1, c - 1] not in horizon
                ):
                    horizon.append([r - 1, c - 1])
                if ((r - 1) >= 0) and (img[r - 1, c, :] == color).all(axis=-1) and [r - 1, c] not in horizon:
                    horizon.append([r - 1, c])
                if (
                    ((r - 1) >= 0)
                    and ((c + 1) < max_col)
                    and (img[r - 1, c + 1, :] == color).all(axis=-1)
                    and [r - 1, c + 1] not in horizon
                ):
                    horizon.append([r - 1, c + 1])
                if ((c - 1) >= 0) and (img[r, c - 1, :] == color).all(axis=-1) and [r, c - 1] not in horizon:
                    horizon.append([r, c - 1])
                if ((c + 1) < max_col) and (img[r, c + 1, :] == color).all(axis=-1) and [r, c + 1] not in horizon:
                    horizon.append([r, c + 1])
                if (
                    ((r + 1) < max_row)
                    and ((c - 1) >= 0)
                    and (img[r + 1, c - 1, :] == color).all(axis=-1)
                    and [r + 1, c - 1] not in horizon
                ):
                    horizon.append([r + 1, c - 1])
                if ((r + 1) < max_row) and (img[r + 1, c, :] == color).all(axis=-1) and [r + 1, c] not in horizon:
                    horizon.append([r + 1, c])
                if (
                    ((r + 1) < max_row)
                    and ((c + 1) < max_col)
                    and (img[r + 1, c + 1, :] == color).all(axis=-1)
                    and [r + 1, c + 1] not in horizon
                ):
                    horizon.append([r + 1, c + 1])

            return component

        def construct_component_img(components, img):
            """Draws the components on top of the given img"""
            components_img = 0 * img
            for component in components:
                pixels = component["original_pixels"]
                color = component["color"]
                for pixel in pixels:
                    components_img[pixel[0], pixel[1], :] = color
            return components_img

        print("In KeyFrameCornerSearch.connected_components()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        img = self.frame["key_frame_img"]
        boundaries = self.frame["boundaries"]
        boundaries_img = self.frame["boundaries_img"]

        # print('Estimating Connected Components ...')
        components = []
        rows, cols = np.nonzero(boundaries)
        copied_img = boundaries_img.copy()
        for row, col in zip(rows, cols):
            if (boundaries_img[row, col, :] == LEFT_BOUNDARY_COLOR).all(axis=-1):
                btype = "left"
                color = LEFT_BOUNDARY_COLOR
            elif (boundaries_img[row, col, :] == RIGHT_BOUNDARY_COLOR).all(axis=-1):
                btype = "right"
                color = RIGHT_BOUNDARY_COLOR
            elif (boundaries_img[row, col, :] == TOP_BOUNDARY_COLOR).all(axis=-1):
                btype = "top"
                color = TOP_BOUNDARY_COLOR
            elif (boundaries_img[row, col, :] == BOTTOM_BOUNDARY_COLOR).all(axis=-1):
                btype = "bottom"
                color = BOTTOM_BOUNDARY_COLOR
            # else: #TODO I (BGB) think we need this catch-all case, because the copied_img is getting updated to be all black in construct_component
            #     continue

            component = construct_component(row, col, btype, color, copied_img)
            components.append(component)

        # construct image
        components_img = construct_component_img(components, img)
        if self.render_control.draw_components:
            save_image(
                img=components_img,
                imgname=self.key_frame_id_str + "_04_components.png",
                path=self.frame["output_construction_dir"],
            )
        if self.render_control.draw_components_fig:
            save_fig(
                img=components_img,
                imgname=self.key_frame_id_str + "_04_components_fig.png",
                path=self.frame["output_construction_dir"],
            )

        if self.render_control.write_components:
            save_connected_components(
                components=components, filename="components.csv", path=self.frame["output_construction_dir"]
            )

        return components, components_img

    def filter_connected_components(self) -> tuple[list[Component], np.ndarray]:
        """Filters self.frame['components'] to only include those that have at least COMPONENT_THRESHOLD pixels."""

        def construct_component_img(components, img):
            components_img = 0 * img
            for component in components:
                pixels = component["original_pixels"]
                color = component["color"]
                for pixel in pixels:
                    components_img[pixel[0], pixel[1], :] = color
            return components_img

        print("In KeyFrameCornerSearch.filter_connected_components()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        img = self.frame["key_frame_img"]
        components = self.frame["components"]

        filtered_components = []
        for component in components:
            if len(component["original_pixels"]) >= COMPONENT_THRESHOLD:
                filtered_components.append(component)

        if self.render_control.draw_filt_components or self.render_control.draw_filt_components_fig:
            filt_connected_comp_img = construct_component_img(filtered_components, img)
            if self.render_control.draw_filt_components:
                save_image(
                    img=filt_connected_comp_img,
                    imgname=self.key_frame_id_str + "_05_filt_components.png",
                    path=self.frame["output_construction_dir"],
                )
            if self.render_control.draw_filt_components_fig:
                save_fig(
                    img=filt_connected_comp_img,
                    imgname=self.key_frame_id_str + "_05_filt_components_fig.png",
                    path=self.frame["output_construction_dir"],
                )

        if self.render_control.write_filt_components:
            save_connected_components(
                filtered_components, filename="filt_components.csv", path=self.frame["output_construction_dir"]
            )

        return filtered_components, filt_connected_comp_img

    def fitted_lines_connected_components(self, type_fit="regression"):
        """Does an initial line fit on the pixels in the components."""
        print("In KeyFrameCornerSearch.fitted_lines_connected_components()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        line_components = []
        components = self.frame["filt_components"]
        for component in components:
            new_component = fit_line_component(component=component, type_fit=type_fit)
            line_components.append(new_component)

        if self.render_control.write_fitted_lines_components:
            save_fitted_lines_connected_components(
                components=line_components,
                filename="fitted_lines_components.csv",
                path=self.frame["output_construction_dir"],
            )

        return line_components

    def fitted_lines_inliers_components(self):
        """Updates the component to fit to the pixels that are closest to the original line fit.

        New Component Keys
        ------------------
            tolerance: the maximum distance that any pixel in inliers is from the fit line, float value from MIN_TOLERANCE to MAX_TOLERANCE (or 99)
            inliers_pixels: pixels within the tolerance of the inliers_line_hom_coef fit line
            outliers_pixels: pixels that are outside the tolerance
            inliers_line_hom_coef: new A, B, and C for the new fit line Ax + By + C = 0
            inliers_line_residual: numpy.polyfit residual
            inliers_line_points: a pair of points [x1, y1, x2, y2] defining the line at the extremis of the component pixels
        """

        def find_inliers_component(component: Component):
            A, B, C = component["original_line_hom_coef"]  # from Ax + By + C = 0
            btype: str = component["boundary_type"]  # 'left', 'right', 'top', or 'bottom'
            original_pixels: list[list[int]] = component["original_pixels"]
            required_inliers = int(round(INLIERS_THRESHOLD * len(original_pixels)))

            tolerance = MIN_TOLERANCE
            max_tolerance = MAX_TOLERANCE
            tol_step = TOL_STEP
            while tolerance <= max_tolerance:
                inliers = []
                inlier_cnt = 0
                for pixel in original_pixels:
                    row, col = pixel[0], pixel[1]
                    if abs(A * col + B * row + C) <= tolerance:
                        inlier_cnt += 1
                        inliers.append(pixel)
                if inlier_cnt >= required_inliers:
                    break
                tolerance += tol_step

            if tolerance > max_tolerance:
                # not sufficient inliers found
                # populate original information to inliers
                component["tolerance"] = 99
                component["outliers_pixels"] = []
                component["inliers_pixels"] = component["original_pixels"]
                component["inliers_line_hom_coef"] = component["original_line_hom_coef"]
                component["inliers_line_residual"] = component["original_line_residual"]
                component["inliers_line_points"] = component["original_line_points"]
            else:
                row, col = np.array([a[0] for a in inliers]), np.array([a[1] for a in inliers])
                if btype == "left" or btype == "right":
                    # expected horizontal line in terms of row
                    x, y = row, col
                else:
                    # expected horizontal line in terms of col
                    x, y = col, row

                reg_fit = np.polyfit(x, y, deg=1, full=True)
                m, b = reg_fit[0]
                residual = reg_fit[1][0]
                A_inl, B_inl, C_inl = -m, 1, -b
                norm = np.linalg.norm(np.array([A_inl, B_inl]))
                A_inl, B_inl, C_inl = A_inl / norm, B_inl / norm, C_inl / norm
                x1 = np.min(x)
                y1 = (-A_inl * x1 - C_inl) / B_inl
                x2 = np.max(x)
                y2 = (-A_inl * x2 - C_inl) / B_inl
                if btype == "left" or btype == "right":
                    # swap x, y   <-- why do we do this? ~BGB
                    A_inl, B_inl = B_inl, A_inl
                    x1, y1 = y1, x1
                    x2, y2 = y2, x2
                start_point = [
                    x1,
                    y1,
                ]  # point at first x pixel [first y for left/right], with the y [x] adjusted to lie on the fit line
                A_inl, B_inl, C_inl = set_proper_hom_coef_sign(start_point, btype, A_inl, B_inl, C_inl)
                outliers = [[pixel[0], pixel[1]] for pixel in original_pixels if pixel not in inliers]
                component["tolerance"] = tolerance
                component["outliers_pixels"] = outliers
                component["inliers_pixels"] = inliers
                component["inliers_line_hom_coef"] = [A_inl, B_inl, C_inl]
                component["inliers_line_residual"] = residual
                component["inliers_line_points"] = [x1, y1, x2, y2]

            return component

        print("In KeyFrameCornerSearch.fitted_lines_inliers_components()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        components = self.frame["fitted_lines_components"]
        inliers_components: list[Component] = []
        for component in components:
            new_component = find_inliers_component(component)
            inliers_components.append(new_component)

        if self.render_control.write_fitted_lines_inliers_components:
            save_fitted_lines_inliers_connected_components(
                components=inliers_components,
                filename="fitted_lines_inliers_components.csv",
                path=self.frame["output_construction_dir"],
            )

        return inliers_components

    def find_corners(self, corners_type=None):
        """Finds the corners based on the intersection points of the components in the image.

        Arguments
        ---------
            corners_type: one of 'top_left'/'top_right'/'bottom_right'/'bottom_left' (for debugging), or None for all four

        Returns
        -------
            corners: a list of corner dict lists [[TL],[TR],[BR],[BL]], where each dict contains the xy 'point' for the corner.
        """
        print("In KeyFrameCornerSearch.find_corners()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        top_left_corners: dict[str, Any] = []
        top_right_corners: dict[str, Any] = []
        bottom_right_corners: dict[str, Any] = []
        bottom_left_corners: dict[str, Any] = []
        output_construction_dir = self.frame["output_construction_dir"]
        corners_types = (
            [corners_type] if (corners_type != None) else ["top_left", "top_right", "bottom_right", "bottom_left"]
        )

        max_row = self.frame["key_frame_img"].shape[0]
        max_col = self.frame["key_frame_img"].shape[1]
        all_corners = {}
        components: list[Component] = self.frame["fitted_lines_inliers_components"]
        for corners_type in corners_types:
            # Get lists of all the counter-clockwise (tomatched) and clockwise (candidates) edges
            # for the given corner. Example, left and top edges for top_left.
            components_tomatched: list[Component] = []
            components_candidates: list[Component] = []
            for component in components:
                btype = component["boundary_type"]
                if corners_type == "top_left":
                    if btype == "left":
                        components_tomatched.append(component)
                    elif btype == "top":
                        components_candidates.append(component)
                elif corners_type == "top_right":
                    if btype == "top":
                        components_tomatched.append(component)
                    elif btype == "right":
                        components_candidates.append(component)
                elif corners_type == "bottom_right":
                    if btype == "right":
                        components_tomatched.append(component)
                    elif btype == "bottom":
                        components_candidates.append(component)
                elif corners_type == "bottom_left":
                    if btype == "bottom":
                        components_tomatched.append(component)
                    elif btype == "left":
                        components_candidates.append(component)

            # Find the corners by searching through the intersecting lines that are
            # closest to each other.
            components_candidates_initial = components_candidates.copy()
            for component in components_tomatched:
                # Filter matching candidates for this component to those on the
                # mirror side of the line.
                mirrorside_candidates: list[Component] = []
                A, B, C = component["inliers_line_hom_coef"]
                components_candidates = components_candidates_initial
                for candidate in components_candidates:
                    pixels = candidate["inliers_pixels"]
                    avg_row = sum([p[0] for p in pixels]) / len(pixels)
                    avg_col = sum([p[1] for p in pixels]) / len(pixels)
                    if (A * avg_col + B * avg_row + C) < 0:
                        mirrorside_candidates.append(candidate)

                components_candidates = mirrorside_candidates
                mirrorside_candidates = []
                pixels = component["inliers_pixels"]
                avg_row = sum([p[0] for p in pixels]) / len(pixels)
                avg_col = sum([p[1] for p in pixels]) / len(pixels)
                for candidate in components_candidates:
                    A, B, C = candidate["inliers_line_hom_coef"]
                    if (A * avg_col + B * avg_row + C) < 0:
                        mirrorside_candidates.append(candidate)

                components_candidates = mirrorside_candidates
                if not len(components_candidates):
                    continue

                # keep the component's representative
                pixels = component["inliers_pixels"]
                if corners_type == "top_left":  # left edge
                    representative = sorted(pixels, key=lambda pix: pix[0], reverse=False)[0]  # ascending
                elif corners_type == "top_right":  # top edge
                    representative = sorted(pixels, key=lambda pix: pix[1], reverse=True)[0]  # descending
                elif corners_type == "bottom_right":  # right edge
                    representative = sorted(pixels, key=lambda pix: pix[0], reverse=True)[0]
                elif corners_type == "bottom_left":  # bottom edge
                    representative = sorted(pixels, key=lambda pix: pix[1], reverse=False)[0]

                representative_tomatched = representative
                #  keep the closest ones
                representatives_candidates = []
                for candidate in components_candidates:
                    pixels = candidate["inliers_pixels"]
                    if corners_type == "top_left":  # top edge
                        representative = sorted(pixels, key=lambda pix: pix[1], reverse=False)[0]
                    elif corners_type == "top_right":  # right edge
                        representative = sorted(pixels, key=lambda pix: pix[0], reverse=False)[0]
                    elif corners_type == "bottom_right":  # bottom edge
                        representative = sorted(pixels, key=lambda pix: pix[1], reverse=True)[0]
                    elif corners_type == "bottom_left":  # left edge
                        representative = sorted(pixels, key=lambda pix: pix[0], reverse=True)[0]

                    representatives_candidates.append(representative)

                distances = [
                    euclidean_distance(representative_tomatched, candidate) for candidate in representatives_candidates
                ]
                found_candidate_indx = np.argsort(np.array(distances))[0]
                found_candidate = components_candidates[found_candidate_indx]

                component_points = component["inliers_line_points"]
                component_start_point = [component_points[0], component_points[1]]  # col, row
                component_end_point = [component_points[2], component_points[3]]
                found_candidate_points = found_candidate["inliers_line_points"]
                found_candidate_start_point = [found_candidate_points[0], found_candidate_points[1]]
                found_candidate_end_point = [found_candidate_points[2], found_candidate_points[3]]

                corner = intersection_point(
                    component_start_point[0],
                    component_start_point[1],
                    component_end_point[0],
                    component_end_point[1],
                    found_candidate_start_point[0],
                    found_candidate_start_point[1],
                    found_candidate_end_point[0],
                    found_candidate_end_point[1],
                )

                # Filter corners
                # Looking for distance conditions
                pixels = component["inliers_pixels"]
                avg_row = sum([p[0] for p in pixels]) / len(pixels)
                avg_col = sum([p[1] for p in pixels]) / len(pixels)

                distance1 = euclidean_distance([corner[1], corner[0]], [avg_row, avg_col])

                pixels = found_candidate["inliers_pixels"]
                avg_row = sum([p[0] for p in pixels]) / len(pixels)
                avg_col = sum([p[1] for p in pixels]) / len(pixels)
                distance2 = euclidean_distance([corner[1], corner[0]], [avg_row, avg_col])

                distance = max([distance1, distance2])

                if distance >= SIDE_FACET_DISTANCE / 2:
                    continue

                # if (euclidean_distance([corner[1], corner[0]], representative_tomatched)
                #     > euclidean_distance(representative_tomatched, [avg_row, avg_col])
                #     or (euclidean_distance([corner[1], corner[0]], [avg_row, avg_col])
                #     < euclidean_distance(representative_tomatched, [avg_row, avg_col]))):
                #     continue

                if corner[1] >= 0 and corner[1] < max_row and corner[0] >= 0 and corner[0] < max_col:
                    corner_structure = {}
                    key1 = "edge_coeff"
                    key2 = "edge_pixels"
                    key3 = "edge_points"
                    if corners_type == "top_left":
                        prefix1 = "left_"
                        prefix2 = "top_"
                        corners = top_left_corners
                    elif corners_type == "top_right":
                        prefix1 = "top_"
                        prefix2 = "right_"
                        corners = top_right_corners
                    elif corners_type == "bottom_right":
                        prefix1 = "right_"
                        prefix2 = "bottom_"
                        corners = bottom_right_corners
                    elif corners_type == "bottom_left":
                        prefix1 = "bottom_"
                        prefix2 = "left_"
                        corners = bottom_left_corners

                    corner_structure = {
                        "corner_type": corners_type,
                        "point": corner,
                        prefix1 + key1: component["inliers_line_hom_coef"],
                        prefix1 + key2: component["inliers_pixels"],
                        prefix1
                        + key3: [
                            component_start_point[0],
                            component_start_point[1],
                            component_end_point[0],
                            component_end_point[1],
                        ],
                        prefix2 + key1: found_candidate["inliers_line_hom_coef"],
                        prefix2 + key2: found_candidate["inliers_pixels"],
                        prefix2
                        + key3: [
                            found_candidate_start_point[0],
                            found_candidate_start_point[1],
                            found_candidate_end_point[0],
                            found_candidate_end_point[1],
                        ],
                    }
                    corners.append(corner_structure)
                    all_corners[tuple(corner)] = corners_type

        top_left = []
        top_right = []
        bottom_right = []
        bottom_left = []
        for key, val in all_corners.items():
            if val == "top_left":
                top_left.append(list(key))
            elif val == "top_right":
                top_right.append(list(key))
            elif val == "bottom_right":
                bottom_right.append(list(key))
            elif val == "bottom_left":
                bottom_left.append(list(key))

        if self.render_control.draw_corners:
            plt.figure()
            plt.imshow(self.frame["edges_img"])
            plt.scatter(
                [x[0] for x in top_left], [x[1] for x in top_left], marker="o", facecolor=PLT_TOP_LEFT_COLOR, s=5
            )
            plt.scatter(
                [x[0] for x in top_right], [x[1] for x in top_right], marker="o", facecolor=PLT_TOP_RIGHT_COLOR, s=5
            )
            plt.scatter(
                [x[0] for x in bottom_right],
                [x[1] for x in bottom_right],
                marker="o",
                facecolor=PLT_BOTTOM_RIGHT_COLOR,
                s=5,
            )
            plt.scatter(
                [x[0] for x in bottom_left],
                [x[1] for x in bottom_left],
                marker="o",
                facecolor=PLT_BOTTOM_LEFT_COLOR,
                s=5,
            )
            plt.savefig(
                os.path.join(self.frame["output_construction_dir"], self.key_frame_id_str + "_08_corners.png"), dpi=200
            )
            plt.close()

        if self.render_control.write_top_left_corners:
            save_corners_facets(
                corners=top_left_corners,
                filename="top_left_corners.csv",
                path=self.frame["output_construction_dir"],
                corners_type="top_left",
            )
        if self.render_control.write_top_right_corners:
            save_corners_facets(
                corners=top_right_corners,
                filename="top_right_corners.csv",
                path=self.frame["output_construction_dir"],
                corners_type="top_right",
            )
        if self.render_control.write_bottom_right_corners:
            save_corners_facets(
                corners=bottom_right_corners,
                filename="bottom_right_corners.csv",
                path=self.frame["output_construction_dir"],
                corners_type="bottom_right",
            )
        if self.render_control.write_bottom_left_corners:
            save_corners_facets(
                corners=bottom_left_corners,
                filename="bottom_left_corners.csv",
                path=self.frame["output_construction_dir"],
                corners_type="bottom_left",
            )

        return [top_left_corners, top_right_corners, bottom_right_corners, bottom_left_corners]

    def facets(self):
        print("In KeyFrameCornerSearch.facets()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        (top_left_corners, top_right_corners, bottom_right_corners, bottom_left_corners) = self.frame["corners"]
        already_matched_corners = []
        facets = []
        # For each Top Left corner
        for top_left_corner in top_left_corners:
            # Finding Top Right Corner
            top_left_point = top_left_corner["point"]
            left_edge_points = top_left_corner["left_edge_points"]
            A_left, B_left, C_left = top_left_corner["left_edge_coeff"]
            # distances with top right corners
            top_right_distances = [
                euclidean_distance(top_left_point, top_right_corner["point"]) for top_right_corner in top_right_corners
            ]
            indices = np.argsort(np.array(top_right_distances))
            flag = True
            for indice in indices:
                top_right_corner = top_right_corners[indice]
                top_right_point = top_right_corner["point"]
                top_edge_points = top_right_corner["top_edge_points"]
                # confirm intersection point is top left corner
                interpoint = intersection_point(
                    left_edge_points[0],
                    left_edge_points[1],
                    left_edge_points[2],
                    left_edge_points[3],
                    top_edge_points[0],
                    top_edge_points[1],
                    top_edge_points[2],
                    top_edge_points[3],
                )

                distance = euclidean_distance(top_left_point, interpoint)
                if (A_left * top_right_point[0] + B_left * top_right_point[1] + C_left < 0) and (  # on the right side
                    distance <= INTER_POINT_DISTANCE
                ):  # ?? MAGIC NUMBER
                    flag = False
                    break

            if flag or top_left_corner in already_matched_corners or top_right_corner in already_matched_corners:
                continue

            # Finding Bottom Right corner
            top_right_point = top_right_corner["point"]
            top_edge_points = top_right_corner["top_edge_points"]
            A_top, B_top, C_top = top_right_corner["top_edge_coeff"]
            # distances with bottom right corners
            bottom_right_distances = [
                euclidean_distance(top_right_point, bottom_right_corner["point"])
                for bottom_right_corner in bottom_right_corners
            ]
            indices = np.argsort(np.array(bottom_right_distances))
            flag = True
            for indice in indices:
                bottom_right_corner = bottom_right_corners[indice]
                bottom_right_point = bottom_right_corner["point"]
                right_edge_points = bottom_right_corner["right_edge_points"]
                # confirm intersection point is top right corner
                interpoint = intersection_point(
                    top_edge_points[0],
                    top_edge_points[1],
                    top_edge_points[2],
                    top_edge_points[3],
                    right_edge_points[0],
                    right_edge_points[1],
                    right_edge_points[2],
                    right_edge_points[3],
                )

                distance = euclidean_distance(top_right_point, interpoint)
                if (
                    (A_top * bottom_right_point[0] + B_top * bottom_right_point[1] + C_top < 0)  # on the right side
                    and (
                        A_left * bottom_right_point[0] + B_left * bottom_right_point[1] + C_left < 0
                    )  # on the right side
                    and (distance <= INTER_POINT_DISTANCE)
                ):  # ?? MAGIC NUMBER
                    flag = False
                    break

            if flag or bottom_right_corner in already_matched_corners:
                continue

            # Finding Bottom Left corner
            bottom_right_point = bottom_right_corner["point"]
            right_edge_points = bottom_right_corner["right_edge_points"]
            A_right, B_right, C_right = bottom_right_corner["right_edge_coeff"]
            # distances with bottom left corners
            bottom_left_distances = [
                euclidean_distance(bottom_right_point, bottom_left_corner["point"])
                for bottom_left_corner in bottom_left_corners
            ]
            indices = np.argsort(bottom_left_distances)

            flag = True
            for indice in indices:
                bottom_left_corner = bottom_left_corners[indice]
                bottom_left_point = bottom_left_corner["point"]
                bottom_edge_points = bottom_left_corner["bottom_edge_points"]
                # confirm intersection point is bottom right corner
                interpoint = intersection_point(
                    right_edge_points[0],
                    right_edge_points[1],
                    right_edge_points[2],
                    right_edge_points[3],
                    bottom_edge_points[0],
                    bottom_edge_points[1],
                    bottom_edge_points[2],
                    bottom_edge_points[3],
                )
                # confirm intersection point is top left corner
                interpoint_alt = intersection_point(
                    left_edge_points[0],
                    left_edge_points[1],
                    left_edge_points[2],
                    left_edge_points[3],
                    bottom_edge_points[0],
                    bottom_edge_points[1],
                    bottom_edge_points[2],
                    bottom_edge_points[3],
                )

                distance = euclidean_distance(bottom_right_point, interpoint)
                distance_alt = euclidean_distance(interpoint_alt, bottom_left_point)
                if (
                    (A_right * bottom_left_point[0] + B_right * bottom_left_point[1] + C_right < 0)  # on the right side
                    and (A_top * bottom_left_point[0] + B_top * bottom_left_point[1] + C_top < 0)  # on the below side
                    and (distance <= INTER_POINT_DISTANCE)  # ?? MAGIC NUMBER
                    and (distance_alt <= INTER_POINT_DISTANCE)
                ):  # ?? MAGIC NUMBER
                    flag = False
                    break

            if flag or bottom_left_corner in already_matched_corners:
                continue

            center = intersection_point(
                top_left_point[0],
                top_left_point[1],
                bottom_right_point[0],
                bottom_right_point[1],
                top_right_point[0],
                top_right_point[1],
                bottom_left_point[0],
                bottom_left_point[1],
            )

            # center should satisfy boundary constraints
            hom_coeff = [
                top_left_corner["left_edge_coeff"],
                top_right_corner["top_edge_coeff"],
                bottom_right_corner["right_edge_coeff"],
                bottom_left_corner["left_edge_coeff"],
            ]

            flag = True
            for A, B, C in hom_coeff:
                if not (A * center[0] + B * center[1] + C < 0):
                    flag = False
            if not flag:
                continue

            c = int(center[0])
            r = int(center[1])
            flag = False

            required_sky_width = int(REQUIRED_SKY_WIDTH / 4)
            if (
                self.is_boundary_pixel(r, c, "left", required_sky_width=required_sky_width)
                and self.is_boundary_pixel(r, c, "top", required_sky_width=required_sky_width)
                and self.is_boundary_pixel(r, c, "right", required_sky_width=required_sky_width)
                and self.is_boundary_pixel(r, c, "bottom", required_sky_width=required_sky_width)
            ):
                flag = True
            if not flag:
                continue
            facet = {
                "top_left": top_left_corner,
                "top_right": top_right_corner,
                "bottom_right": bottom_right_corner,
                "bottom_left": bottom_left_corner,
                "center": center,
            }
            already_matched_corners.append(top_left_corner)
            already_matched_corners.append(top_right_corner)
            already_matched_corners.append(bottom_right_corner)
            already_matched_corners.append(bottom_left_corner)
            facets.append(facet)

        if self.render_control.draw_facets:
            plt.figure()
            plt.imshow(self.frame["edges_img"])
            for facet in facets:
                top_left_corner = facet["top_left"]["point"]
                top_right_corner = facet["top_right"]["point"]
                bottom_right_corner = facet["bottom_right"]["point"]
                bottom_left_corner = facet["bottom_left"]["point"]
                center = facet["center"]
                plt.scatter(top_left_corner[0], top_left_corner[1], facecolor=PLT_TOP_LEFT_COLOR, s=1)
                plt.scatter(top_right_corner[0], top_right_corner[1], facecolor=PLT_TOP_RIGHT_COLOR, s=1)
                plt.scatter(bottom_right_corner[0], bottom_right_corner[1], facecolor=PLT_BOTTOM_RIGHT_COLOR, s=1)
                plt.scatter(bottom_left_corner[0], bottom_left_corner[1], facecolor=PLT_BOTTOM_LEFT_COLOR, s=1)
                plt.scatter(center[0], center[1], facecolor=PLT_CENTER_COLOR, s=1)
            plt.savefig(
                os.path.join(self.frame["output_construction_dir"], self.key_frame_id_str + "_09_facets.png"), dpi=200
            )
            plt.close()

        if self.render_control.write_facets:
            save_corners_facets(facets=facets, filename="facets.csv", path=self.frame["output_construction_dir"])

        return facets

    def filter_facets_polygons(self):
        print("In KeyFrameCornerSearch.filter_facets_polygons()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        all_facets = self.frame["facets"]

        # filter the facets
        filtered_facets = []
        for facet in all_facets:
            center = facet["center"]
            in_polygon = False
            for name_polygon in self.list_of_name_polygons:
                name = name_polygon[0]
                polygon = name_polygon[1]
                x_list = [pt[0] for pt in polygon]
                y_list = [pt[1] for pt in polygon]
                x_min = min(x_list)
                x_max = max(x_list)
                y_min = min(y_list)
                y_max = max(y_list)
                if (x_min < center[0]) and (center[0] < x_max) and (y_min < center[1]) and (center[1] < y_max):
                    in_polygon = True
                    break
            if in_polygon:
                filtered_facets.append(facet)

        # heliostats as many as the boxes
        heliostats = []
        assigned_centers = []
        for name_polygon in self.list_of_name_polygons:
            name = name_polygon[0]
            polygon = name_polygon[1]
            x_list = [
                pt[0] for pt in polygon
            ]  # ?? SCAFFOLDING RCB -- REDUNDANT AND INEFFIICENT; IMPLEMENT POINT IN POLYGON TEST, OR SIMILAR.
            y_list = [
                pt[1] for pt in polygon
            ]  # ?? SCAFFOLDING RCB -- REDUNDANT AND INEFFIICENT; IMPLEMENT POINT IN POLYGON TEST, OR SIMILAR.
            x_min = min(
                x_list
            )  # ?? SCAFFOLDING RCB -- REDUNDANT AND INEFFIICENT; IMPLEMENT POINT IN POLYGON TEST, OR SIMILAR.
            x_max = max(
                x_list
            )  # ?? SCAFFOLDING RCB -- REDUNDANT AND INEFFIICENT; IMPLEMENT POINT IN POLYGON TEST, OR SIMILAR.
            y_min = min(
                y_list
            )  # ?? SCAFFOLDING RCB -- REDUNDANT AND INEFFIICENT; IMPLEMENT POINT IN POLYGON TEST, OR SIMILAR.
            y_max = max(
                y_list
            )  # ?? SCAFFOLDING RCB -- REDUNDANT AND INEFFIICENT; IMPLEMENT POINT IN POLYGON TEST, OR SIMILAR.
            heliostat = {}
            heliostat["name"] = name
            heliostat["facets"] = []
            for facet in filtered_facets:
                center = facet["center"]
                if (center not in assigned_centers) and (
                    (x_min < center[0]) and (center[0] < x_max) and (y_min < center[1]) and (center[1] < y_max)
                ):
                    assigned_centers.append(center)
                    heliostat["facets"].append(facet)
            heliostats.append(heliostat)

        if self.render_control.draw_filtered_facets:
            plt.figure()
            plt.imshow(self.frame["edges_img"])
            for facet in filtered_facets:
                top_left_corner = facet["top_left"]["point"]
                top_right_corner = facet["top_right"]["point"]
                bottom_right_corner = facet["bottom_right"]["point"]
                bottom_left_corner = facet["bottom_left"]["point"]
                center = facet["center"]
                plt.scatter(top_left_corner[0], top_left_corner[1], facecolor=PLT_TOP_LEFT_COLOR, s=1)
                plt.scatter(top_right_corner[0], top_right_corner[1], facecolor=PLT_TOP_RIGHT_COLOR, s=1)
                plt.scatter(bottom_right_corner[0], bottom_right_corner[1], facecolor=PLT_BOTTOM_RIGHT_COLOR, s=1)
                plt.scatter(bottom_left_corner[0], bottom_left_corner[1], facecolor=PLT_BOTTOM_LEFT_COLOR, s=1)
                plt.scatter(center[0], center[1], facecolor=PLT_CENTER_COLOR, s=1)
            plt.savefig(
                os.path.join(self.frame["output_construction_dir"], self.key_frame_id_str + "_10_filtered_facets.png"),
                dpi=200,
            )
            plt.close()

        if self.render_control.draw_filtered_heliostats:
            colors = ["c", "r", "g", "y", "p"]
            plt.figure()
            plt.imshow(self.frame["edges_img"])
            for hel_indx in range(0, len(heliostats)):
                color = colors[hel_indx]
                heliostat = heliostats[hel_indx]
                for facet in heliostat["facets"]:
                    center = facet["center"]
                    plt.scatter(center[0], center[1], facecolor=color, s=1)
            plt.savefig(
                os.path.join(
                    self.frame["output_construction_dir"], self.key_frame_id_str + "_11_filtered_heliostats.png"
                ),
                dpi=200,
            )
            plt.close()

        return filtered_facets, heliostats

    def top_row_facets(self):
        """
        Assumption: We trust first row in terms of correct found centers
        """
        print("In KeyFrameCornerSearch.top_row_facets()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        for heliostat in self.frame["heliostats"]:
            facets = heliostat["facets"]  # this is a list
            facets = sorted(facets, key=lambda f: f["center"][1])  # sort in terms of rows
            top_row_facets = facets[: self.specifications.facets_per_row]  # top row facets
            keys = [
                [["bottom_right", "bottom_edge_coeff"], ["bottom_left", "bottom_edge_coeff"]],
                [["top_left", "top_edge_coeff"], ["top_right", "top_edge_coeff"]],
            ]
            for type_of_keys in keys:
                coeff = []
                for key_list in type_of_keys:
                    key1, key2 = key_list
                    for facet in top_row_facets:
                        coeff.append(facet[key1][key2])

                facet_indx = len(top_row_facets) - 1
                cnt = len(top_row_facets)
                while facet_indx >= 0:
                    facet = top_row_facets[facet_indx]
                    center = facet["center"]
                    flag = True
                    for c in coeff:
                        flag *= center[0] * c[0] + center[1] * c[1] + c[2] < 0

                    if not flag:
                        cnt -= 1
                    facet_indx -= 1

                top_row_facets = top_row_facets[:cnt]

            heliostat["top_row_facets"] = top_row_facets

        if self.render_control.draw_top_row_facets:
            edge_img = self.frame["edges_img"]
            plt.figure()
            plt.imshow(edge_img)
            for heliostat in self.frame["heliostats"]:
                top_row_facets = heliostat["top_row_facets"]
                for facet in top_row_facets:
                    plt.scatter(facet["center"][0], facet["center"][1], s=1, facecolor="m")
            plt.savefig(
                os.path.join(self.frame["output_construction_dir"], self.key_frame_id_str + "_14_top_row_facets.png"),
                dpi=200,
            )
            plt.close()

    def classify_top_row_facets(self):
        def find_combinations(inp, out):
            if len(inp) == 0:
                if len(out) != 0:
                    all_combinations.append(out)
                return

            find_combinations(inp[1:], out[:])
            if len(out) == 0:
                find_combinations(inp[1:], inp[:1])
            elif inp[0] > out[-1]:
                out.append(inp[0])
                find_combinations(inp[1:], out[:])

        print("In KeyFrameCornerSearch.classify_top_row_facets()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        for heliostat in self.frame["heliostats"]:
            top_row_facets = heliostat["top_row_facets"]
            top_row_facets = sorted(top_row_facets, key=lambda f: f["center"][0])  # sort in terms of column
            ids = [i for i in range(0, self.specifications.facets_per_row)]
            if len(top_row_facets) == self.specifications.facets_per_row:  # all facets have been identified
                for facet_indx in range(0, len(top_row_facets)):
                    facet = top_row_facets[facet_indx]
                    facet["id"] = facet_indx
                    top_row_facets[facet_indx] = facet
            else:
                # all diferent combinations, brute-force - Complexity O(self.specifications.facets_per_row!)
                all_combinations = []
                find_combinations(ids, [])
                combinations = [x for x in all_combinations if len(x) == len(top_row_facets)]

                # image points
                img_centers2d = []
                img_corners2d = []
                for facet in top_row_facets:
                    img_centers2d.append(facet["center"])
                    for key in CLOCKWISE_DIR:
                        img_corners2d.append(facet[key]["point"])
                points2d = np.array(img_corners2d + img_centers2d).astype("float32")

                centers3d = self.specifications.facets_centroids
                corners3d = self.specifications.facets_corners
                # centers3d = read_centers3d(facet_centoids_csv)
                # corners3d = centers3d_to_corners3d(centers3d)
                rank = []
                proj_errors = []
                for combination in combinations:
                    # object points - flat heliostat
                    obj_centers3d = []
                    obj_corners3d = []
                    for i in combination:
                        obj_centers3d.append(centers3d[i])
                        corners_indx = i * self.specifications.corners_per_facet
                        for indx in range(corners_indx, corners_indx + self.specifications.corners_per_facet):
                            obj_corners3d.append(corners3d[indx])
                    ## Projection
                    points3d = np.array(obj_corners3d + obj_centers3d).astype("float32")
                    h, w = self.frame["key_frame_img"].shape[:2]
                    _, _, _, _, error = solvePNP(points3d, points2d, h, w, pnptype=self.solvePnPtype)
                    proj_errors.append(error)

                # select based on projected error
                best_indx = np.argsort(np.array(proj_errors))[0]
                selected_combination = combinations[best_indx]
                for facet_indx, i in zip(range(0, len(top_row_facets)), selected_combination):
                    facet = top_row_facets[facet_indx]
                    facet["id"] = i
                    top_row_facets[facet_indx] = facet

        if self.render_control.draw_top_row_facets_labels:
            edge_img = self.frame["edges_img"]
            plt.figure()
            plt.imshow(edge_img)
            for heliostat in self.frame["heliostats"]:
                top_row_facets = heliostat["top_row_facets"]
                for facet in top_row_facets:
                    center = facet["center"]
                    label = facet["id"]
                    plt.scatter(center[0], center[1], s=1, facecolor="m")
                    plt.annotate(str(label), (center[0], center[1]), color="c", fontsize=5)

            plt.savefig(
                os.path.join(
                    self.frame["output_construction_dir"], self.key_frame_id_str + "_15_top_row_facets_labels.png"
                ),
                dpi=200,
            )
            plt.close()

    def project_and_confirm(self, canny_levels=["tight", "normal", "light"], iterations=5):
        print("In KeyFrameCornerSearch.project_and_confirm()...")  # ?? SCAFFOLDING RCB -- TEMPORARY
        edge_img = self.frame["edges_img"]  # demonstration
        h, w = self.frame["key_frame_img"].shape[:2]
        # facet_centoids_csv  = self.facet_centroids_dir_body_ext + 'csv_files/' + 'Facets_Centroids.csv'
        centers3d = self.specifications.facets_centroids
        corners3d = self.specifications.facets_corners
        # centers3d           = read_centers3d(facet_centoids_csv)
        # corners3d           = centers3d_to_corners3d(centers3d)
        for heliostat in self.frame["heliostats"]:
            top_row_facets = heliostat["top_row_facets"]
            top_row_facets = sorted(top_row_facets, key=lambda x: x["id"])
            # Image Points
            imgcorners = []
            imgcenters = []
            for facet in top_row_facets:
                imgcenters.append(facet["center"])
                for key in CLOCKWISE_DIR:
                    imgcorners.append(facet[key]["point"])

            points2d = np.array(imgcorners + imgcenters).astype("float32")
            # Object Points
            labels = [f["id"] for f in top_row_facets]
            objcorners = []
            objcenters = []
            for label in labels:
                objcenters.append(centers3d[label])
                corner_indx = label * self.specifications.corners_per_facet
                for indx in range(corner_indx, corner_indx + self.specifications.corners_per_facet):
                    objcorners.append(corners3d[indx])

            points3d = np.array(objcorners + objcenters).astype("float32")
            if len(points3d) != len(points2d):
                msg = (
                    "In KeyFrameCornerSearch.project_and_confirm(), len(points3d)="
                    + str(len(points3d))
                    + " does not equal len(points3d)="
                    + str(len(points3d))
                )
                print("ERROR: " + msg)
                raise ValueError(msg)
            if len(points3d) < 4:
                msg = (
                    "In KeyFrameCornerSearch.project_and_confirm(), len(points3d)="
                    + str(len(points3d))
                    + " is not at least 4.  Not expected here."
                )
                print("ERROR: " + msg)
                raise ValueError(msg)
            print(
                "In KeyFrameCornerSearch.project_and_confirm(), calling solvePNP(); len(points3d) =",
                len(points3d),
                "; len(points2d) =",
                len(points2d),
            )  # ?? SCAFFOLDING RCB -- TEMPORARY
            mtx, dist, rvec, tvec, pnp_error = solvePNP(points3d, points2d, h, w, pnptype=self.solvePnPtype)

            proj_corners, _ = cv.projectPoints(np.array(corners3d).astype("float32"), rvec, tvec, mtx, dist)
            proj_corners = proj_corners.reshape(-1, 2)
            proj_corners = proj_corners.tolist()

            confirmed_corners, projected_corners = self.confirm(
                proj_corners, corners3d, canny_levels=canny_levels, iterations=iterations
            )

            if (
                not projected_corners
            ):  # no corner was confirmed, discard this heliostat  # ?? SCAFFOLDING RCB -- DOES THIS EVER WORK, GIVEN THAT WE ARE LOOKING AT OUTPUT OF EXPECTED CORNERS ROUTINE?
                continue
            heliostat["confirmed_corners"] = self.convert_None_corners(
                confirmed_corners
            )  # Replace None entries with [-1,-1]
            heliostat["projected_corners"] = projected_corners

        final_heliostats = []
        for heliostat in self.frame["heliostats"]:
            if "projected_corners" in heliostat:
                final_heliostats.append(heliostat)

        # Assemble and store final results.
        # Lists.
        all_confirmed_corners = []
        all_projected_corners = []
        list_of_name_confirmed_corners = []
        list_of_name_projected_corners = []
        for heliostat in final_heliostats:
            all_confirmed_corners += heliostat[
                "confirmed_corners"
            ]  # ?? SCAFFOLDING RCB -- AFTER THIS REFACTOR IS COMPLETE, MAYBE THESE ARE NO LONGER NEEDED.
            all_projected_corners += heliostat[
                "projected_corners"
            ]  # ?? SCAFFOLDING RCB -- AFTER THIS REFACTOR IS COMPLETE, MAYBE THESE ARE NO LONGER NEEDED.
            list_of_name_confirmed_corners.append([heliostat["name"], heliostat["confirmed_corners"]])
            list_of_name_projected_corners.append([heliostat["name"], heliostat["projected_corners"]])
        # FrameNameXyList objects.
        confirmed_fnxl = fnxl.FrameNameXyList()
        projected_fnxl = fnxl.FrameNameXyList()
        confirmed_fnxl.add_list_of_name_xy_lists(self.key_frame_id, list_of_name_confirmed_corners)
        projected_fnxl.add_list_of_name_xy_lists(self.key_frame_id, list_of_name_projected_corners)
        # Store in this class object.
        self.frame["all_confirmed_corners"] = all_confirmed_corners
        self.frame["all_projected_corners"] = all_projected_corners
        self.frame["confirmed_fnxl"] = confirmed_fnxl
        self.frame["projected_fnxl"] = projected_fnxl
        # Write to disk.
        if self.render_control.write_all_confirmed_corners:
            save_corners(
                corners=all_confirmed_corners,
                filename="all_confirmed_corners.csv",
                path=self.frame["output_construction_dir"],
            )  # ?? SCAFFOLDING RCB -- AFTER THIS REFACTOR IS COMPLETE, MAYBE THESE ARE NO LONGER NEEDED.
        if self.render_control.write_all_projected_corners:
            save_corners(
                corners=all_projected_corners,
                filename="all_projected_corners.csv",
                path=self.frame["output_construction_dir"],
            )  # ?? SCAFFOLDING RCB -- AFTER THIS REFACTOR IS COMPLETE, MAYBE THESE ARE NO LONGER NEEDED.
        if self.render_control.write_confirmed_fnxl:
            confirmed_fnxl.save(
                os.path.join(
                    self.frame["output_construction_dir"], "csv_files", (self.key_frame_id_str + "_confirmed_fnxl.csv")
                )
            )  # ?? SCAFFOLDING RCB -- INSTEAD OF ADDING "CSV_FILES" HERE, SHOULD BE PASSED IN THAT WAY.  SAVE TO ANSWER DIRECTORY?  -- PROBABLY NOT; INSEAD SAVE FROM CALLER?
        if self.render_control.write_projected_fnxl:
            projected_fnxl.save(
                os.path.join(
                    self.frame["output_construction_dir"], "csv_files", (self.key_frame_id_str + "_projected_fnxl.csv")
                )
            )  # ?? SCAFFOLDING RCB -- INSTEAD OF ADDING "CSV_FILES" HERE, SHOULD BE PASSED IN THAT WAY.  SAVE TO ANSWER DIRECTORY?  -- PROBABLY NOT; INSEAD SAVE FROM CALLER?

        if self.render_control.draw_confirmed_corners:
            # Confirmed corners.
            # Initialize the figure.
            plt.figure()
            plt.imshow(edge_img)
            for final_heliostat in final_heliostats:
                found_confirmed_corners = self.filter_not_found_corners(final_heliostat["confirmed_corners"])
                # Draw the heliostat name.
                if (
                    len(found_confirmed_corners) > 0
                ):  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                    label_xy = g2d.label_point(
                        found_confirmed_corners
                    )  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                    plt.text(
                        label_xy[
                            0
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        label_xy[
                            1
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        final_heliostat[
                            "name"
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        color="c",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        horizontalalignment="center",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        verticalalignment="center",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontsize=5,  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontstyle="normal",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontweight="bold",
                    )  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                # Draw confirmed corners.
                for corner in found_confirmed_corners:
                    if corner is not None:
                        plt.scatter(corner[0], corner[1], s=1, facecolor="c")
            # Save the figure.
            plt.savefig(
                os.path.join(
                    self.frame["output_construction_dir"], self.key_frame_id_str + "_16_confirmed_corners.png"
                ),
                dpi=200,
            )
            plt.close()

        if self.render_control.draw_projected_corners:
            # Projected corners.
            # Initialize the figure.
            plt.figure()
            plt.imshow(edge_img)
            for final_heliostat in final_heliostats:
                projected_corners = final_heliostat["projected_corners"]
                # Draw the heliostat name.
                if (
                    len(projected_corners) > 0
                ):  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                    label_xy = g2d.label_point(
                        projected_corners
                    )  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                    plt.text(
                        label_xy[
                            0
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        label_xy[
                            1
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        final_heliostat[
                            "name"
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        color="m",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        horizontalalignment="center",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        verticalalignment="center",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontsize=5,  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontstyle="normal",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontweight="bold",
                    )  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                # Draw projected corners.
                for corner in projected_corners:
                    if corner is None:
                        msg = "In KeyFrameCornerSearch.project_and_confirm(), encountered null projected corner."
                        print("ERROR: ", msg)
                        raise ValueError(msg)
                    plt.scatter(corner[0], corner[1], s=1, facecolor="m")
            # Save the figure.
            plt.savefig(
                os.path.join(
                    self.frame["output_construction_dir"], self.key_frame_id_str + "_17_projected_corners.png"
                ),
                dpi=200,
            )
            plt.close()

        if self.render_control.draw_projected_and_confirmed_corners:
            # Both projected and confirmed corners.
            # Initialize the figure.
            plt.figure()
            plt.imshow(edge_img)
            for final_heliostat in final_heliostats:
                projected_corners = final_heliostat["projected_corners"]
                found_confirmed_corners = self.filter_not_found_corners(final_heliostat["confirmed_corners"])
                # Draw the heliostat name.
                if (
                    len(projected_corners) > 0
                ):  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                    label_xy = g2d.label_point(
                        projected_corners
                    )  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                    plt.text(
                        label_xy[
                            0
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        label_xy[
                            1
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        final_heliostat[
                            "name"
                        ],  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        color="m",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        horizontalalignment="center",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        verticalalignment="center",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontsize=5,  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontstyle="normal",  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                        fontweight="bold",
                    )  # ?? SCAFFOLDING RCB -- MAKE THIS INTEGRATED WITH STANDARD PLOTTING AND RENDER CONTROL ROUTINES.
                # Draw projected corners.
                for corner in projected_corners:
                    if corner is None:
                        msg = "In KeyFrameCornerSearch.project_and_confirm(), encountered null projected corner."
                        print("ERROR: ", msg)
                        raise ValueError(msg)
                    plt.scatter(corner[0], corner[1], s=1, facecolor="m")
                # Draw confirmed corners.
                for corner in found_confirmed_corners:
                    if corner is not None:
                        plt.scatter(corner[0], corner[1], s=1, facecolor="c")
            # Save the figure.
            plt.savefig(
                os.path.join(
                    self.frame["output_construction_dir"],
                    self.key_frame_id_str + "_18_projected_and_confirmed_corners.png",
                ),
                dpi=200,
            )
            plt.close()

        # Sort heliostats left to right  # ?? SCAFFOLDING RCB -- MAY NOT DO ANYTHING, IN WHICH CASE WE SHOULD DELETE.  WILL BE INCORRECT IF BOXES ARE NOT SORTED LEFT TO RIGHT?
        final_heliostats = sorted(
            final_heliostats, key=lambda x: np.mean(np.array(x["projected_corners"]), axis=0)[0]
        )  # ?? SCAFFOLDING RCB -- MAY NOT DO ANYTHING, IN WHICH CASE WE SHOULD DELETE.  WILL BE INCORRECT IF BOXES ARE NOT SORTED LEFT TO RIGHT?

    def confirm(
        self,
        expected_corners,
        corners3d,
        canny_levels=["tight", "normal", "light"],
        tolerance=3,
        pixels=100,
        iterations=5,
    ):
        h, w = self.frame["key_frame_img"].shape[:2]
        max_row = self.frame["key_frame_img"].shape[0]
        max_col = self.frame["key_frame_img"].shape[1]

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
                                cnt = 0
                                edge_pixels.append([col, row])
                else:
                    for col in range(min_col, max_col):
                        for row in range(min_row, max_row):
                            dist = abs(A * col + B * row + C)
                            if edges[row][col] and dist <= tolerance:
                                cnt = 0
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

        canny_types = canny_levels
        for i in range(0, iterations):
            flag_break = False
            for canny_type in canny_types:
                img = self.frame[
                    "key_frame_img"
                ]  # ?? SCAFFOLDING RCB -- SEE CANNY() AND CONFIRM() ROUTINES FOR DUPLICATE PLACES WHERE THIS CODE IS PLACED.  MUST KEEP CONSISTENT.  # ?? SCAFFOLDING RCB -- REDUNDANT IMAGE LOAD, BLUR, AND EDGE FINDING.  COMPUTE ONCE, CACHE AND COMMUNICATE.
                img = cv.GaussianBlur(
                    img, (5, 5), 0
                )  # ?? SCAFFOLDING RCB -- SEE CANNY() AND CONFIRM() ROUTINES FOR DUPLICATE PLACES WHERE THIS CODE IS PLACED.  MUST KEEP CONSISTENT.  # ?? SCAFFOLDING RCB -- REDUNDANT IMAGE LOAD, BLUR, AND EDGE FINDING.  COMPUTE ONCE, CACHE AND COMMUNICATE.
                edges = CannyImg(
                    img, canny_type=canny_type
                )  # ?? SCAFFOLDING RCB -- SEE CANNY() AND CONFIRM() ROUTINES FOR DUPLICATE PLACES WHERE THIS CODE IS PLACED.  MUST KEEP CONSISTENT.  # ?? SCAFFOLDING RCB -- REDUNDANT IMAGE LOAD, BLUR, AND EDGE FINDING.  COMPUTE ONCE, CACHE AND COMMUNICATE.
                # edges                           = CannyImg(self.frame['sky'], canny_type=canny_type)  # ?? SCAFFOLDING RCB -- ORIGINAL CODE, MULTIPLE FAILURE IMPLICATIONS:  (1) USING SKY, WHEN SKY WAS NOT USERED PREVIOUSLY.  (2) CAUSES OPENCV TO CRASH.  (THANKFULLY; OTHERWISE I WOULDN'T HAVE FOUND THE OTHER BUG.)
                confirmed_facets = confirm_facets(expected_corners, edges, tolerance, pixels)
                confirmed_corners = find_corners(confirmed_facets)
                flag_break = True
                for corner in confirmed_corners:
                    flag_break *= corner is None
                if flag_break:
                    expected_corners = []
                    break
                # if not enough corners were confirmed
                points3d, points2d = construct_points(confirmed_corners, corners3d)
                if len(points3d) != len(points2d):
                    msg = (
                        "In KeyFrameCornerSearch.confirm(), len(points3d)="
                        + str(len(points3d))
                        + " does not equal len(points2d)="
                        + str(len(points2d))
                    )
                    print("ERROR: " + msg)
                    raise ValueError(msg)
                if len(points3d) < 4:  # Four points needed for solvePNP().
                    expected_corners = []
                    break
                mtx, dist, rvec, tvec, pnp_error = solvePNP(points3d, points2d, h, w, pnptype=self.solvePnPtype)
                expected_corners, _ = cv.projectPoints(np.array(corners3d).astype("float32"), rvec, tvec, mtx, dist)
                expected_corners = expected_corners.reshape(-1, 2)
                expected_corners = expected_corners.tolist()
            if flag_break:
                break

        projected_corners = expected_corners
        return confirmed_corners, projected_corners

    def convert_None_corners(self, input_corners):
        corners_with_None_entries_converted = []
        for corner_entry in input_corners:
            if corner_entry is None:
                corners_with_None_entries_converted.append([-1, -1])
            else:
                corners_with_None_entries_converted.append(corner_entry)
        return corners_with_None_entries_converted

    def filter_not_found_corners(self, input_corners):
        found_corners = []
        for corner_entry in input_corners:
            if corner_entry != [-1, -1]:
                found_corners.append(corner_entry)
        return found_corners
