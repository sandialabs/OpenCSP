import cv2 as cv
from PIL import Image
import numpy as np

import opencsp.common.lib.cv.image_reshapers as ir
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.geometry.LineXY as l2
import opencsp.common.lib.geometry.Pxy as p2
import contrib.common.lib.geometry.RectXY as r2
import opencsp.common.lib.geometry.RegionXY as reg2
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.PowerpointSlide as pps
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlPowerpointPresentation as rcpp
import opencsp.common.lib.render_control.RenderControlPowerpointSlide as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class RegionDetector:
    """
    A class used to detect regions in images using various image processing
    techniques including Canny edge detection, blob analysis, ray intersection,
    edge identification, and edge intersection.

    Attributes
    ----------
    edge_coarse_width : int
        The maximum encapsulating width of the edges for :py:meth:`step5_find_edge_groups`.
    canny_edges_gradient : int
        The threshold gradient value for edges in Canny edge detection.
    canny_non_edges_gradient : int
        The threshold gradient value for non-edges in Canny edge detection.
    canny_test_gradients : list[tuple[int, int]]
        A list of tuples containing (edge, non-edge) gradient pairs to test for
        Canny edge detection. Useful for setting up edge detection with new
        experiments.
    edge_colors : dict
        What colors to use to indicate the "top", "right", "bottom", and "left" edges.
    corner_colors : dict
        What colors to use to indicate the "tl", "tr", "br", and "bl" corners.
    images_to_show : list[tuple[str, np.ndarray]]
        A list of images to be displayed for use with canny_test_gradients.
    summary_visualizations : list[tuple[str, np.ndarray]]
        A list of summary visualizations that explain the steps used in locating
        the region.
    ppt_deck : RenderControlPowerpointPresentation
        The PowerPoint presentation control.
    slide_control : RenderControlPowerpointSlide
        The PowerPoint slide control.
    """

    def __init__(
        self,
        edge_coarse_width: int,
        canny_edges_gradient=10,
        canny_non_edges_gradient=5,
        canny_test_gradients: list[tuple[int, int]] = None,
    ):
        """
        Parameters
        ----------
        edge_coarse_width : int
            The maximum encapsulating width of the edges for
            :py:meth:`step5_find_edge_groups`. This should at least as large as
            width of the largest vertical edge, or the height of the largest
            horizontal edge.
        canny_edges_gradient : int, optional
            The threshold gradient value for edges in Canny edge detection.
        canny_non_edges_gradient : int, optional
            The threshold gradient value for non-edges in Canny edge detection.
        canny_test_gradients : list[tuple[int, int]], optional
            A list of tuples containing (edge, non-edge) gradient pairs to test
            for Canny edge detection. Useful for setting up edge detection with
            new experiments.  For example: [(10,5), (10,10), (10,15), (20,5),
            (20,10), (20,15)].  Default None.
        """
        self.edge_coarse_width = edge_coarse_width
        self.canny_edges_gradient = canny_edges_gradient
        self.canny_non_edges_gradient = canny_non_edges_gradient
        self.canny_test_gradients = canny_test_gradients

        # visualization values
        self.edge_colors = {
            "top": color.red(),
            "right": color.cyan(),
            "bottom": color.magenta(),
            "left": color.yellow(),
        }
        self.corner_colors = {"tl": color.red(), "tr": color.cyan(), "br": color.magenta(), "bl": color.yellow()}
        self.images_to_show: list[tuple[str, np.ndarray]] = []
        self.summary_visualizations: list[tuple[str, np.ndarray]] = []

        # powerpoint values
        self.ppt_deck = rcpp.RenderControlPowerpointPresentation()
        self.slide_control = rcps.RenderControlPowerpointSlide()

    def draw_image(self, title: str, image: np.ndarray, show=False, block=False):
        """
        Draws an image and optionally displays it.

        Parameters
        ----------
        title : str
            The title of the image.
        image : np.ndarray
            The image to be drawn.
        show : bool, optional
            Whether to show the image immediately, by default False.
        block : bool, optional
            Whether to block the execution until the image window is closed, by default False.
        """
        self.images_to_show.append((title, image.copy()))

        if show:
            rows = 1
            if len(self.images_to_show) > 3:
                rows = 2
            if len(self.images_to_show) > 8:
                rows = 3
            cols = int(np.ceil(len(self.images_to_show) / rows))

            axis_control = rca.image(grid=False)
            figure_control = rcfg.RenderControlFigure(tile=True, tile_array=(cols, rows))
            view_spec_2d = vs.view_spec_im()
            fig_record = None
            for title, img in self.images_to_show:
                if fig_record is not None:
                    fig_record.view.show(block=False)
                fig_record = fm.setup_figure(
                    figure_control, axis_control, view_spec_2d, title=title, code_tag=f"{__file__}", equal=False
                )
                fig_record.view.imshow(Image.fromarray(img))
            fig_record.view.show(block=block)
            if block:
                fig_record.close()
            self.images_to_show.clear()

    def draw_images(self, images: list[tuple[str, np.ndarray]]):
        """
        Draws multiple images and displays them.

        Parameters
        ----------
        images : list[tuple[str, np.ndarray]]
            A list of tuples containing the title and image to be drawn.
        """
        for i, title_image in enumerate(images):
            show = i == len(images) - 1
            self.draw_image(title_image[0], title_image[1], show, show)

    def step1_canny_edge_detection(self, debug_canny_settings: bool, image: np.ndarray) -> np.ndarray:
        """
        Performs Canny edge detection on the given image.

        Parameters
        ----------
        debug_canny_settings : bool
            If True, or if canny_test_gradients is set, then debug images will be shown.
        image : np.ndarray
            The input image on which to perform edge detection.

        Returns
        -------
        np.ndarray
            A boolean array with the same width/height as the input image, where
            1s represent pixels that are classified as edges and 0s represent
            non-edges.
        """
        slide = pps.PowerpointSlide.template_content_grid(nrows=2, ncols=3, slide_control=self.slide_control)
        slide.set_title("step1_canny_edge_detection")

        captions_images: list[tuple[str, np.ndarray]] = []
        captions_images.append(("input image", image))

        if self.canny_test_gradients is not None:
            for i, gradients in enumerate(self.canny_test_gradients):
                # Exploring parameters for canny edge detection
                edges_gradient, non_edges_gradient = gradients

                canny: np.ndarray = cv.Canny(image, non_edges_gradient, edges_gradient)

                image_title = f"Canny (edges gradient {edges_gradient}, non-edges gradient {non_edges_gradient})"
                if edges_gradient == self.canny_edges_gradient and non_edges_gradient == self.canny_non_edges_gradient:
                    self.summary_visualizations.append((image_title, canny))
                    image_title += "**"
                captions_images.append((image_title, canny))

        # chosen canny parameters
        canny_edges: np.ndarray = cv.Canny(image, self.canny_non_edges_gradient, self.canny_edges_gradient)

        for caption, image in captions_images:
            image = ir.false_color_reshaper(image).astype(np.uint8)
            slide.add_image(pps.PowerpointImage(image, caption=caption))
        self.ppt_deck.add_slide(slide)
        if debug_canny_settings or self.canny_test_gradients:
            self.draw_images(captions_images)

        return canny_edges

    def step2_blob_analysis(
        self, canny_edges: np.ndarray, debug_blob_analysis=False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies horizontal and vertical blob growth/filtering to the canny edges.

        Parameters
        ----------
        canny_edges : np.ndarray[uint8]
            The edge pixels as found with :py:meth:`step1_canny_edge_detection`.
        debug_blob_analysis : bool, optional
            True to show debug images, by default False.

        Returns
        -------
        horizontal_mask : np.ndarray[bool]
            Numpy array with the same width and height as the given canny_edges
            image. Non-zero where the horizontal edge pixels are detected.
        vertical_mask : np.ndarray[bool]
            Numpy array with the same width and height as the given canny_edges
            image. Non-zero where the vertical edge pixels are detected.
        negative_mask : np.ndarray[bool]
            Numpy array with the same width and height as the given canny_edges
            image. Non-zero where there aren't any edges.
        """
        slide = pps.PowerpointSlide.template_content_grid(nrows=2, ncols=3, slide_control=self.slide_control)
        slide.set_title("step2_blob_analysis")
        slice_l, slice_s = 13, 3  # long, short
        slice_area = slice_l * slice_s  # 13*3 = 39
        slice_thresh = 10 / slice_area  # 10/39 = 0.26
        horizontal_kernel = np.ones((slice_s, slice_l), np.float32) / slice_area
        vertical_kernel = np.ones((slice_l, slice_s), np.float32) / slice_area

        # convolve with the horizontal kernel
        binary_edges = np.clip(canny_edges, 0, 1).astype(np.uint8)
        negative_horizontal = binary_edges.copy().astype(np.float64)
        negative_horizontal: np.ndarray = cv.filter2D(src=negative_horizontal, ddepth=-1, kernel=horizontal_kernel)
        negative_horizontal[np.where(negative_horizontal < slice_thresh)] = 0
        negative_horizontal[np.where(negative_horizontal > 0)] = 255
        horizontal_mask = np.full(negative_horizontal.shape, False)
        horizontal_mask[np.where(negative_horizontal != 0)] = True

        # convolve with the vertical kernel
        negative_vertical = binary_edges.copy().astype(np.float64)
        negative_vertical: np.ndarray = cv.filter2D(src=negative_vertical, ddepth=-1, kernel=vertical_kernel)
        negative_vertical[np.where(negative_vertical < slice_thresh)] = 0
        negative_vertical[np.where(negative_vertical > 0)] = 255
        vertical_mask = np.full(negative_vertical.shape, False)
        vertical_mask[np.where(negative_vertical != 0)] = True

        negative_mask = np.logical_not(np.logical_or(horizontal_mask, vertical_mask))
        thin_blob = canny_edges.copy()
        thin_blob[negative_mask] = 0
        self.summary_visualizations.append(("Horizontal+Vertical Blob", thin_blob))

        fat_blob = thin_blob.copy().astype(np.float64) / 255
        fat_blob: np.ndarray = cv.filter2D(fat_blob, -1, np.ones((7, 7), np.float32))
        fat_blob[np.where(fat_blob > 0.5)] = 255

        captions_images = [
            ("canny_edges", canny_edges),
            ("horizontal_mask", horizontal_mask),
            ("vertical_mask", vertical_mask),
            ("thin_blob", thin_blob),
            ("fat_blob", fat_blob),
        ]
        for caption, image in captions_images:
            image = ir.false_color_reshaper(image).astype(np.uint8)
            slide.add_image(pps.PowerpointImage(image, caption=caption))
        self.ppt_deck.add_slide(slide)
        if debug_blob_analysis:
            self.draw_images(captions_images)

        return horizontal_mask, vertical_mask, negative_mask

    def step3_ray_project_edge_intercept(
        self, thin_boundaries: np.ndarray, approx_area_center_pixel: p2.Pxy, debug_ray_projection=False
    ) -> tuple[p2.Pxy, p2.Pxy]:
        """
        Project rays out from the given approximate center to find the first
        intercepts with the boundaries image, if any. This process is iterative,
        and updates the center approximation at each step.

        Parameters
        ----------
        thin_boundaries : np.ndarray[uint8]
            The boundaries image to test against. Should be 0 where there aren't
            any boundaries, or non-0 where there are boundaries.
        approx_area_center_pixel : p2.Pxy
            Where to start ray projecting from.
        debug_ray_projection : bool, optional
            True to show debug image and block, by default False

        Returns
        -------
        area_center: p2.Pxy
            The updated approximation of the area's center.
        boundary_samples: p2.Pxy
            The set of sampled boundary intercepts, minus those that fall outside the typical range.
        """
        niterations = 10
        nrays = 50
        slide = pps.PowerpointSlide.template_content_grid(nrows=3, ncols=4, slide_control=self.slide_control)
        slide.set_title("step3_ray_project_edge_intercept")

        # initialize/declare variables
        all_boundary_samples: p2.Pxy = None
        """ All the sampled points """
        boundary_samples: p2.Pxy = None
        """ All the sampled points that are within the typical distance """
        area_center = p2.Pxy(approx_area_center_pixel)
        """ Approximate center of the area. Updated each iteration. """
        all_area_centers: list[p2.Pxy] = []
        """ All area_center values, for debugging """
        captions_images: list[tuple[str, np.ndarray]] = []

        # get the bounding box, to make calculating rays easier
        image_height, image_width = thin_boundaries.shape[0], thin_boundaries.shape[1]
        image_vertices = [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]]
        image_vertices = [p2.Pxy(xy_pair) for xy_pair in image_vertices]
        image_area = reg2.RegionXY.from_vertices(p2.Pxy.from_list(image_vertices))

        # get a base image that we can additively draw to
        sampling_image = thin_boundaries.copy()
        if sampling_image.ndim > 2:
            sampling_image = sampling_image[:, :, 0]
        sampling_image[np.where(sampling_image != 0)] = 125
        debug_all_image = sampling_image.copy()

        # For finding the closes point in the given list to the given reference
        def get_closest_point(reference: p2.Pxy, points_list: p2.Pxy):
            deltas = p2.Pxy([points_list.x - reference.x[0], points_list.y - reference.y[0]])
            distances = deltas.magnitude()
            min_idx = np.argmin(distances)
            min_pnt = p2.Pxy((points_list.x[min_idx], points_list.y[min_idx]))
            return min_pnt

        # project some rays
        for iteration in range(niterations):
            iteration_samples: list[p2.Pxy] = []
            debug_image = sampling_image.copy()

            for ray_idx in range(nrays):
                angle = np.random.random() * np.pi * 2  # radians
                ray = l2.LineXY.from_location_angle(area_center, angle)
                pnt1, pnt2 = None, None

                # draw the line
                image_edge_intersect = image_area.loops[0].intersect_line(ray)
                iei1 = (int(image_edge_intersect.x[0]), int(image_edge_intersect.y[0]))
                iei2 = (int(image_edge_intersect.x[1]), int(image_edge_intersect.y[1]))
                ray_image = cv.line(np.zeros_like(sampling_image), iei1, iei2, color=(125), thickness=2)

                # add the line's values to the sampling_image's values
                # get the intersection locations
                ray_image += sampling_image
                y_intersects, x_intersects = np.where(ray_image == 250)
                if len(x_intersects) > 0:
                    xy_intersects = p2.Pxy([x_intersects, y_intersects])

                    # find the closest point to the center
                    pnt1 = get_closest_point(area_center, xy_intersects)
                    iteration_samples.append(pnt1)

                    # remove all points on this side of the line
                    bisect = l2.LineXY.from_location_angle(area_center, angle + (np.pi / 2))
                    if not (bisect.dist_from_signed(pnt1)[0] < 0):
                        bisect = bisect.flip()
                    intersect_is_other_side = bisect.dist_from_signed(xy_intersects) > 0
                    other_half_xy_intersects = xy_intersects[np.where(intersect_is_other_side)]

                    # find the other closest point to the center, on the other side
                    if len(other_half_xy_intersects) > 0:
                        pnt2 = get_closest_point(area_center, other_half_xy_intersects)
                        iteration_samples.append(pnt2)

                # debugging
                center_pixel = (int(area_center.x[0]), int(area_center.y[0]))
                pnt1_pixel, pnt2_pixel = iei1, iei2
                if pnt1 is not None:
                    pnt1_pixel = (int(pnt1.x[0]), int(pnt1.y[0]))
                if pnt2 is not None:
                    pnt2_pixel = (int(pnt2.x[0]), int(pnt2.y[0]))
                ray_image = cv.circle(ray_image, center_pixel, 7, color=(255))
                debug_image = cv.line(debug_image, pnt1_pixel, pnt2_pixel, color=(50), thickness=2)
                debug_image = cv.circle(debug_image, center_pixel, 4, color=(255))
                debug_all_image = cv.line(debug_all_image, pnt1_pixel, pnt2_pixel, color=(50), lineType=cv.LINE_AA)
                if pnt1 is not None:
                    ray_image = cv.circle(ray_image, pnt1_pixel, 7, color=(255))
                    debug_image = cv.circle(debug_image, pnt1_pixel, 4, color=(255))
                    debug_all_image = cv.drawMarker(debug_all_image, pnt1_pixel, color=(255), markerSize=1)
                if pnt2 is not None:
                    ray_image = cv.circle(ray_image, pnt2_pixel, 7, color=(255))
                    debug_image = cv.circle(debug_image, pnt2_pixel, 4, color=(255))
                    debug_all_image = cv.drawMarker(debug_all_image, pnt2_pixel, color=(255), markerSize=1)
                # self.draw_image(f"Ray {ray_idx} ({center_pixel=}, deg={np.rad2deg(angle)})", debug_image, True, True)

            # record information from this iteration
            captions_images.append((f"Ray projection, iteration {iteration}", debug_image))
            if all_boundary_samples is None:
                all_boundary_samples = p2.Pxy.from_list(iteration_samples)
                boundary_samples = p2.Pxy(all_boundary_samples.data)
            else:
                all_boundary_samples = p2.Pxy.from_list([all_boundary_samples] + iteration_samples)

            # discard duplicate points
            all_boundary_samples_xy = list(zip(all_boundary_samples.x.tolist(), all_boundary_samples.y.tolist()))
            unique_boundary_samples = list(set(all_boundary_samples_xy))
            unique_samples_x = [v[0] for v in unique_boundary_samples]
            unique_samples_y = [v[1] for v in unique_boundary_samples]
            all_boundary_samples = p2.Pxy([unique_samples_x, unique_samples_y])

            # choose the samples that aren't outliers
            deltas = p2.Pxy([all_boundary_samples.x - area_center.x[0], all_boundary_samples.y - area_center.y[0]])
            distances = deltas.magnitude()
            distances_list = sorted(distances.tolist())
            mid50 = distances_list[int(len(distances) * 0.25) : int(len(distances) * 0.75)]
            average_distance = np.average(mid50)
            in_range = np.logical_and(distances > average_distance / 2, distances < average_distance * 1.5)
            boundary_samples = all_boundary_samples[np.where(in_range)]
            lt.info(f"Sampled {len(boundary_samples)} points")

            # update center pixel approximation
            all_area_centers.append(area_center)
            area_center = p2.Pxy(np.average(boundary_samples.data, axis=1))

        # draw the debug_all_image with all rays and all sampled points
        for ac in all_area_centers:
            center_pixel = (int(ac.x[0]), int(ac.y[0]))
            debug_all_image = cv.drawMarker(debug_all_image, center_pixel, color=(255), markerSize=3)
        captions_images.append((f"Ray projection, sum", debug_all_image))
        self.summary_visualizations.append((f"Ray Projection", ir.false_color_reshaper(debug_all_image)))

        for caption, image in captions_images:
            image = ir.false_color_reshaper(image).astype(np.uint8)
            slide.add_image(pps.PowerpointImage(image, caption=caption))
        self.ppt_deck.add_slide(slide)
        if debug_ray_projection:
            self.draw_images(captions_images)

        return area_center, boundary_samples

    def step4_boundary_locations_to_edges(
        self, boundary_locations: p2.Pxy, canny_edges: np.ndarray, diameter=7, debug_ray_projection=False
    ) -> np.ndarray:
        """
        Get all edge pixels within a small distance of the boundary locations.

        Parameters
        ----------
        boundary_locations : p2.Pxy
            The list of boundary locations to match against.
        canny_edges : np.ndarray[uint8]
            All edge pixels, as from the canny edge detector.
        diameter : int, optional
            The distance that an edge pixel can be from the boundary locations.
            Applies a box filter of this width and height.

        Returns
        -------
        boundary_edges: np.ndarray[uint8]
            The edge pixels close to the boundary locations. Matched edge pixels
            will be 255, all others will be 0.
        """
        slide = pps.PowerpointSlide.template_content_grid(nrows=1, ncols=3, slide_control=self.slide_control)
        slide.set_title("step4_boundary_locations_to_edges")
        diameter = 7

        boundary_pixels = np.zeros_like(canny_edges)
        boundary_pixels[boundary_locations.asindex("yx")] = 255

        boundary_blob = boundary_pixels.copy().astype(np.float64)
        boundary_blob: np.ndarray = cv.filter2D(boundary_blob, -1, np.ones((diameter, diameter), np.float32))

        matched_edges = canny_edges.copy()
        matched_edges[np.where(boundary_blob == 0)] = 0

        captions_images = [
            (f"Boundary pixels", boundary_pixels),
            (f"Boundary blob", boundary_blob),
            (f"Boundary edges", matched_edges),
        ]
        for caption, image in captions_images:
            image = ir.false_color_reshaper(image).astype(np.uint8)
            slide.add_image(pps.PowerpointImage(image, caption=caption))
        self.ppt_deck.add_slide(slide)
        if debug_ray_projection:
            self.draw_images(captions_images)

        return matched_edges

    def step5_find_edge_groups(
        self, boundary_edges: np.ndarray, debug_edge_groups=False
    ) -> tuple[list[r2.RectXY], np.ndarray]:
        """
        Finds the top/bottom and left/right groups of edge pixels in the given
        edges image. We do this by looking for large groups of pixels in the x
        and y axes.

        Two iterations are done. In the first iteration the rough location of
        the edges is determined by scanning all edge pixels. In the second
        iteration, the search region is constrained to only include the region
        where we expect the edges to be, and to exclude the corners.

        Parameters
        ----------
        boundary_edges : np.ndarray[uint8]
            A black and white image to search for edges within, with 0 pixels
            (non-edges) and non-zero pixels (edges).

        Returns
        -------
        edge_groups : tuple[list[r2.RectXY], np.ndarray]
            The four edge groups
        """
        slide = pps.PowerpointSlide.template_content_grid(nrows=1, ncols=2, slide_control=self.slide_control)
        slide.set_title("step5_find_edge_groups")

        refined_edge_groups: list[r2.RectXY] = []
        width = boundary_edges.shape[1]
        height = boundary_edges.shape[0]

        # find initial edge groups, two per 'x' and 'y' search dir2.rections
        scan_dirs = ("y", "x")
        scansteps_xy = (np.array([0, 1]), np.array([1, 0]))
        windows_wh = (np.array([width, self.edge_coarse_width]), np.array([self.edge_coarse_width, height]))
        initial_edge_groups: dict[str, tuple[r2.RectXY, r2.RectXY]] = {}
        for scan_dir, scanstep_xy, window_wh in zip(scan_dirs, scansteps_xy, windows_wh):
            initial_edge_groups[scan_dir] = self._find_edge_groups_xory(
                boundary_edges, scan_dir, scanstep_xy, window_wh
            )

        # find the edge groups again, excluding pixels from the opposite groups
        for scan_dir, scanstep_xy, window_wh in zip(scan_dirs, scansteps_xy, windows_wh):
            # get the subregion to scan within
            if scan_dir == "y":
                groups = initial_edge_groups["x"]
                left, right = sorted(groups, key=lambda g: g.left)
                region = r2.RectXY(p2.Pxy((left.right, 0)), p2.Pxy((right.left, height)))
                window_wh = np.array([region.right - region.left, self.edge_coarse_width])
            elif scan_dir == "x":
                groups = initial_edge_groups["y"]
                top, bottom = sorted(groups, key=lambda g: g.top)
                region = r2.RectXY(p2.Pxy((0, top.bottom)), p2.Pxy((width, bottom.top)))
                window_wh = np.array([self.edge_coarse_width, region.bottom - region.top])
            boundary_edges_region = boundary_edges[
                int(region.top) : int(region.bottom), int(region.left) : int(region.right)
            ]

            # get a more accurate scan, now that the confusing pixels at the corners are excluded
            offset_edge_groups = self._find_edge_groups_xory(boundary_edges_region, scan_dir, scanstep_xy, window_wh)
            for edge_group in offset_edge_groups:
                refined_edge_groups.append(edge_group + region.tl)

        # visualize the found edge groups
        vis_image = np.expand_dims(boundary_edges, axis=2)
        vis_image = np.broadcast_to(vis_image, (vis_image.shape[0], vis_image.shape[1], 3))
        captions_groups = [
            ("Initial edge groups", initial_edge_groups["x"] + initial_edge_groups["y"]),
            ("Refined edge groups", refined_edge_groups),
        ]
        captions_images: list[tuple[str, np.ndarray]] = []
        for caption, edge_groups in captions_groups:
            edge_group_image = vis_image.copy()

            for edge_group in edge_groups:
                tl, br = (int(edge_group.left), int(edge_group.top)), (int(edge_group.right), int(edge_group.bottom))
                edge_group_image = cv.rectangle(edge_group_image, tl, br, color.magenta().rgb_255(), thickness=2)

            captions_images.append((caption, edge_group_image))
        self.summary_visualizations.append(("Edge Groups", captions_images[-1][1]))

        for caption, image in captions_images:
            slide.add_image(pps.PowerpointImage(image.astype(np.uint8), caption=caption))
        self.ppt_deck.add_slide(slide)
        if debug_edge_groups:
            self.draw_images(captions_images)

        return refined_edge_groups

    def _find_edge_groups_xory(
        self, canny: np.ndarray, scan_dir: str, scanstep_xy: np.ndarray, window_wh: np.ndarray
    ) -> tuple[r2.RectXY, r2.RectXY]:
        """
        Finds clumps of pixels along the x or y axis. A sliding window approach
        is used with size window_wh, and has a resolution of scanstep_xy. Two
        mutually exclusive windows are returned for the windows with the largest
        number of edge pixels.

        Parameters
        ----------
        canny : np.ndarray[number]
            Image with non-edge (0) and edge (non-zero) pixels.
        scan_dir : str
            The dir2.rection to scan in. One of 'x' or 'y'.
        scanstep_xy : np.ndarray[int]
            The resolution of the sliding window. Should be a length-2 numpy array.
        window_wh : np.ndarray[int]
            The size of the window. Should be a length-2 numpy array. Either the
            width should match the width of the given canny image, or the height
            should match.

        Returns
        -------
        tuple[r2.RectXY, r2.RectXY]
            The two largest windows.
        """
        ret: list[r2.RectXY] = []
        width = canny.shape[1]
        height = canny.shape[0]
        previous_window: r2.RectXY = None
        scanstep_xy = p2.Pxy(scanstep_xy)
        window_wh = p2.Pxy(window_wh)

        # validate input
        if window_wh.x[0] != width:
            if window_wh.y[0] != height:
                lt.error_and_raise(
                    ValueError,
                    "Error in TargetBoardLocatorImageProcessor._find_edge_groups_xory(): "
                    + f"either the width or height of window_wh should match the given image, "
                    + "but {window_wh.x[0]=} != {width=} and {window_wh.y[0]=} != {height=}",
                )

        # search for the two largest windows
        for i in range(2):
            # position of the window to search within
            curr_pos = r2.RectXY(p2.Pxy((0, 0)), window_wh) - scanstep_xy
            largest_pos: r2.RectXY = None
            largest_count = -1

            while True:
                curr_pos += scanstep_xy
                if (curr_pos.bottom >= height) and (curr_pos.right >= width):
                    break

                # We want two well-separated lines for the top/bottom and
                # left/right sides. Don't check within the overlapping
                # region of the previously found window.
                if i == 1:
                    if scan_dir == "y":
                        if (curr_pos.bottom > previous_window.top) and (curr_pos.top < previous_window.bottom):
                            continue
                    elif scan_dir == "x":
                        if (curr_pos.right > previous_window.left) and (curr_pos.left < previous_window.right):
                            continue
                    else:
                        raise RuntimeError

                # get the current window
                window = canny[int(curr_pos.top) : int(curr_pos.bottom), int(curr_pos.left) : int(curr_pos.right)]

                # check if this is the largest window
                count = np.sum(window)
                if count > largest_count:
                    largest_pos = curr_pos
                    largest_count = count

            # register the location of the largest window as one of our edge groups
            ret.append(largest_pos)

            # get ready for the next iteration in the same scan direction
            previous_window = largest_pos

        return ret

    def step6_assign_edges_and_corners(
        self, edge_groups: list[r2.RectXY], canny: np.ndarray, debug_edge_assignment=False
    ) -> tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]:
        """
        Finds the edges in the canny image for each edge group, and assign the
        edges to be the "left", "top", "right" or "bottom" of a rectangle. Uses
        the RANSAC algorithm to find the edges.

        Parameters
        ----------
        edge_groups : list[XY]
            List of regions to search for edges within using RANSAC. Should be
            length 4, with one group for each of the top, right, bottom, and
            left sides. Order doesn't matter.
        canny : np.ndarray
            The edges image from the canny algorithm.
        debug_edge_assignment : bool, optional
            True to draw debug images, by default False

        Returns
        -------
        edges: dict[str, l2.LineXY]
            One edge per "left", "top", "right", and "bottom"
        corners: dict[str, p2.Pxy]
            The intersection points of the edges. Includes "tl", "tr", "br", and "bl".
        region: reg2.RegionXY
            The region described the by the corners.
        """
        slide = pps.PowerpointSlide.template_content_grid(nrows=1, ncols=2, slide_control=self.slide_control)
        slide.set_title("step6_assign_edges_and_corners")

        edges: dict[str, l2.LineXY] = {"top": None, "bottom": None, "left": None, "right": None}
        corners: dict[str, p2.Pxy] = {"tl": None, "tr": None, "br": None, "bl": None}
        region: reg2.RegionXY = None
        width, height = canny.shape[1], canny.shape[0]

        # use RANSAC to assign lines to the edge groups
        lines: list[l2.LineXY] = []
        for eg in edge_groups:
            edge_region = canny[int(eg.top) : int(eg.bottom), int(eg.left) : int(eg.right)]
            edge_idxs = np.where(edge_region == 255)
            # get the correct x/y offsets
            edge_idxs = (edge_idxs[0] + int(eg.top), edge_idxs[1] + int(eg.left))
            # Pxy expects x/y value, but images/np.ndarrays are y-major and so
            # edge_idxs are in y/x order. Flip to x/y order.
            edge_idxs = (edge_idxs[1], edge_idxs[0])
            points = p2.Pxy(edge_idxs)
            line = l2.LineXY.fit_from_points(points)
            lines.append(line)

        # Assign edges to be top, bottom, left, or right
        for line in lines:
            # Lines with slope == 1 are at 45 degrees. Use this halfway point
            # between vertical/horizontal to determine the whether a line should
            # be classified as top/bottom or left/right.
            if np.abs(line.slope) > 1:
                # vertical(ish) line
                if edges["left"] is None:
                    edges["left"] = line
                else:
                    line_x_intercept = line.x_from_y(height / 2)
                    left_x_intercept = edges["left"].x_from_y(height / 2)
                    if line_x_intercept < left_x_intercept:
                        edges["right"] = edges["left"]
                        edges["left"] = line
                    else:
                        edges["right"] = line
            else:
                # horizontal(ish) line
                if edges["top"] is None:
                    edges["top"] = line
                else:
                    line_y_intercept = line.y_from_x(width / 2)
                    left_y_intercept = edges["top"].y_from_x(width / 2)
                    if line_y_intercept < left_y_intercept:
                        edges["bottom"] = edges["top"]
                        edges["top"] = line
                    else:
                        edges["bottom"] = line

        # Assign corners from the edges
        corners["tl"] = p2.Pxy(edges["top"].intersect_with(edges["left"]))
        corners["tr"] = p2.Pxy(edges["top"].intersect_with(edges["right"]))
        corners["br"] = p2.Pxy(edges["bottom"].intersect_with(edges["right"]))
        corners["bl"] = p2.Pxy(edges["bottom"].intersect_with(edges["left"]))

        # build a region from the corners
        vertices = p2.Pxy.from_list(corners.values())
        region = reg2.RegionXY.from_vertices(vertices)

        edge_corners_image = self.visualize_edges_corners(canny, edges, corners)
        captions_images = [
            ("assigned_edges_and_corners", edge_corners_image),
            ("Encompassing region", self.visualize_region(canny, region)),
        ]
        self.summary_visualizations.append(("Edges & Corners", edge_corners_image))
        for caption, image in captions_images:
            slide.add_image(pps.PowerpointImage(image.astype(np.uint8), caption=caption))
        self.ppt_deck.add_slide(slide)
        if debug_edge_assignment:
            self.draw_images(captions_images)

        return edges, corners, region

    def visualize_edges_corners(
        self, base_image: np.ndarray, edges: dict[str, l2.LineXY], corners: dict[str, p2.Pxy], thickness=2
    ):
        """
        Visualize edges and corners on a base image.

        Parameters
        ----------
        base_image: np.ndarray
            The base image on which to visualize edges and corners.
        edges: dict[str, l2.LineXY]
            A dictionary of edge labels ("top"/"bottom"/"left"/"right") to edge line objects.
        corners: dict[str, p2.Pxy]
            A dictionary of corner labels ("tl"/"tr"/"bl"/"br") to corner point objects.
        thickness: int, optional
            The thickness of the lines and circles to draw. Default is 2.

        Returns
        -------
        vis_image: np.ndarray
            The image with visualized edges and corners.
        """
        vis_image = ir.false_color_reshaper(base_image)
        width, height = vis_image.shape[1], vis_image.shape[0]

        # Visualize the edges
        for edge_label in edges:
            edge: l2.LineXY = edges[edge_label]
            if edge_label in ["left", "right"]:
                pt1 = (int(edge.x_from_y(0)), 0)
                pt2 = (int(edge.x_from_y(height)), height)
            elif edge_label in ["top", "bottom"]:
                pt1 = (0, int(edge.y_from_x(0)))
                pt2 = (width, int(edge.y_from_x(width)))
            else:
                raise RuntimeError
            edge_color = self.edge_colors[edge_label].rgb_255()
            vis_image = cv.line(vis_image, pt1, pt2, edge_color, thickness=thickness, lineType=cv.LINE_AA)

        # Visualize the corners
        for corner_label in corners:
            corner: p2.Pxy = corners[corner_label]
            center = (int(corner.x[0]), int(corner.y[0]))
            corner_color = self.corner_colors[corner_label].rgb_255()
            vis_image = cv.circle(vis_image, center, 7, corner_color, thickness=thickness)

        return vis_image

    def visualize_region(self, base_image: np.ndarray, region: reg2.RegionXY) -> np.ndarray:
        """
        Visualize a region on a base image. Sets all pixels to 255 within the
        given region.

        Parameters
        ----------
        base_image: np.ndarray
            The base image on which to visualize the region.
        region: reg2.RegionXY
            The region to visualize.

        Returns
        -------
        vis_image: np.ndarray
            The image with the visualized region.
        """
        vis_image = base_image.copy()
        width, height = base_image.shape[1], base_image.shape[0]

        # Build the pixels index list
        all_x = list(range(width)) * height
        all_y = np.zeros((width * height))
        for y in range(height):
            all_y[y * width : (y + 1) * width] = y
        all_xy = p2.Pxy([all_x, all_y])

        # Visualize the region
        matching_xy = all_xy[np.where(region.is_inside(all_xy))]
        vis_image[matching_xy.asindex("yx")] = 255

        return vis_image

    def find_boundary_pixels_in_image(
        self,
        image: np.ndarray,
        approx_center_pixel: p2.Pxy,
        debug_canny_settings=False,
        debug_blob_analysis=False,
        debug_ray_projection=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the pixels that are good candidates for the boundaries of a the
        region within the given image.

        Parameters
        ----------
        image: np.ndarray
            The input image.
        approx_center_pixel: p2.Pxy
            The approximate center pixel.
        debug_canny_settings: bool, optional
            Flag to enable debugging for Canny edge detection. Default is False.
        debug_blob_analysis: bool, optional
            Flag to enable debugging for blob analysis. Default is False.
        debug_ray_projection: bool, optional
            Flag to enable debugging for ray projection. Default is False.

        Returns
        -------
        canny_edges_boundary_pixels: tuple[np.ndarray, np.ndarray]
            A tuple containing numpy arrays, with the Canny edge pixels from the
            Canny edge detector and the resulting boundary edge pixels from
            steps 1-4.
        """
        # Reset the debug images
        self.ppt_deck = rcpp.RenderControlPowerpointPresentation()
        self.images_to_show.clear()
        self.summary_visualizations.clear()

        # Apply canny edge detection
        canny_edges = self.step1_canny_edge_detection(debug_canny_settings, image)
        # Image.fromarray(canny_edges).save(orp.opencsp_temporary_dir() + "/tmp/canny_edges.png")
        # canny_edges = np.array(Image.open(orp.opencsp_temporary_dir() + "/tmp/canny_edges.png"))

        # Use blobs to create a "wall" that can be easiliy detected.
        horizontal_mask, vertical_mask, negative_mask = self.step2_blob_analysis(canny_edges, debug_blob_analysis)
        thin_boundaries = np.ones_like(canny_edges, dtype=np.uint8)
        thin_boundaries[negative_mask] = 0
        # Image.fromarray(thin_boundaries).save(orp.opencsp_temporary_dir() + "/tmp/thin_boundaries.png")
        # thin_boundaries = np.array(Image.open(orp.opencsp_temporary_dir() + "/tmp/thin_boundaries.png"))

        # Use rays to find the boundaries of the wall from within the target area
        rectangle_center, boundary_samples = self.step3_ray_project_edge_intercept(
            thin_boundaries, approx_center_pixel, debug_ray_projection
        )
        # np.save(orp.opencsp_temporary_dir() + "/tmp/rectangle_center", rectangle_center.data)
        # np.save(orp.opencsp_temporary_dir() + "/tmp/boundary_samples", boundary_samples.data)
        # rectangle_center = p2.Pxy(np.load(orp.opencsp_temporary_dir() + "/tmp/rectangle_center.npy"))
        # boundary_samples = p2.Pxy(np.load(orp.opencsp_temporary_dir() + "/tmp/boundary_samples.npy"))

        # Get the edge pixels from the boundary locations
        boundary_edges = self.step4_boundary_locations_to_edges(
            boundary_samples, canny_edges, debug_ray_projection=debug_ray_projection
        )

        return canny_edges, boundary_edges

    def find_rectangular_region(
        self, boundary_edges: p2.Pxy, canny_edges: np.ndarray, debug_edge_groups=False, debug_edge_assignment=False
    ) -> tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]:
        """
        Find a rectangular region from the given boundary and canny edge images.

        Parameters
        ----------
        boundary_edges: p2.Pxy
            The boundary edges, as a numpy array with zeros for non-matching
            pixels and non-zeros for potential boundary pixels.
        canny_edges: np.ndarray
            The Canny edges, as a numpy array.
        debug_edge_groups: bool, optional
            Flag to enable debugging for edge groups. Default is False.
        debug_edge_assignment: bool, optional
            Flag to enable debugging for edge groups assignment. Default is False.

        Returns
        -------
        edges: dict[str, l2.LineXY]
            The edges of the rectangular region ("top", "bottom", "left", and "right").
        corners: dict[str, p2.Pxy]
            The corners of the rectangular region ("tl", "tr", "bl", and "br").
        region: reg2.RegionXY
            The located rectangular region.
        """
        # Scan for the left, right, top, and bottom edge groups
        edge_groups = self.step5_find_edge_groups(boundary_edges, debug_edge_groups)

        # Find the target region from the edge groups
        edges, corners, region = self.step6_assign_edges_and_corners(edge_groups, canny_edges, debug_edge_assignment)

        return edges, corners, region

    def save_debug_images(self, powerpoint_save_file: str, overwrite=False):
        """
        Saves the debug images to a powerpoint file. This is useful for
        debugging the region detection steps.

        Parameters
        ----------
        powerpoint_save_file: str
            The path/name.ext to save to.
        overwrite: bool, optional
            Flag to allow overwriting the file if it exists. Default is False.

        Raises
        ------
        FileNotFoundError:
            If the path to save to does not exist.
        FileExistsError:
            If the PowerPoint file already exists and overwrite is not allowed.
        """
        # Validate input
        ppt_path, ppt_name, ppt_ext = ft.path_components(powerpoint_save_file)
        if not ft.directory_exists(ppt_path):
            lt.error_and_raise(
                FileNotFoundError,
                "Error in RegionDetector.find_region(): "
                + f"can't find the directory {ppt_path} to save the powerpoint to",
            )
        if ft.file_exists(powerpoint_save_file):
            if not overwrite:
                lt.error_and_raise(
                    FileExistsError,
                    "Error in RegionDetector.find_region(): " + f"file {powerpoint_save_file} already exists",
                )

        # Save the visualizations to a powerpoint
        if powerpoint_save_file is not None:
            self.ppt_deck.save(powerpoint_save_file, overwrite)
