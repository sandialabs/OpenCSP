import matplotlib.axes
import matplotlib.collections
import matplotlib.patches
import numpy as np
import scipy.spatial.transform

from opencsp.common.lib.cv.annotations.AbstractAnnotations import AbstractAnnotations
import opencsp.common.lib.geometry.LoopXY as l2
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.log_tools as lt


class RectangleAnnotations(AbstractAnnotations):
    """
    A collection of pixel bounding boxes where points of interest are located in an image.
    """

    def __init__(
        self,
        style: rcps.RenderControlPointSeq = None,
        upperleft_lowerright_corners: tuple[p2.Pxy, p2.Pxy] = None,
        pixels_to_meters: float = None,
    ):
        """
        Parameters
        ----------
        style : RenderControlBcs, optional
            The rendering style, by default {magenta, no corner markers}
        upperleft_lowerright_corners : Pxy
            The upper-left and lower-right corners of the bounding box for this rectangle, in pixels
        pixels_to_meters : float, optional
            A simple conversion method for how many meters a pixel represents,
            for use in scale(). By default None.
        """
        if style is None:
            style = rcps.default(marker=None, color=color.magenta())
        super().__init__(style)

        self.points = upperleft_lowerright_corners
        self.pixels_to_meters = pixels_to_meters

    def get_bounding_box(self, index=0) -> reg.RegionXY:
        x1 = self.points[0].x[index]
        y1 = self.points[0].y[index]
        x2 = self.points[1].x[index]
        y2 = self.points[1].y[index]
        w = x2 - x1
        h = y2 - y1

        return reg.RegionXY(l2.LoopXY.from_rectangle(x1, y1, w, h))

    @property
    def origin(self) -> p2.Pxy:
        return self.points[0]

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        raise NotImplementedError("Orientation is not yet implemented for RectangleAnnotations")

    @property
    def size(self) -> list[float]:
        width_height = self.points[1] - self.points[0]
        max_size: np.ndarray = np.max(width_height.data, axis=0)
        return max_size.tolist()

    @property
    def scale(self) -> list[float]:
        if self.pixels_to_meters is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in RectangeAnnotations.scale(): "
                + "no pixels_to_meters conversion ratio is set, so scale can't be estimated",
            )
        return [self.size * self.pixels_to_meters]

    def render_to_figure(self, fig: rcfr.RenderControlFigureRecord, image: np.ndarray = None, include_label=False):
        label = self.get_label(include_label)

        # get the corner vertices for each bounding box
        draw_loops: list[list[tuple[int, int]]] = []
        for index in range(len(self.points[0])):
            bbox = self.get_bounding_box(index)
            for loop in bbox.loops:
                loop_verts = list(zip(loop.vertices.x, loop.vertices.y))
                loop_verts = [(int(x), int(y)) for x, y in loop_verts]
                draw_loops.append(loop_verts)

        # draw the bounding boxes
        for i, draw_loop in enumerate(draw_loops):
            fig.view.draw_pq_list(draw_loop, close=True, style=self.style, label=label)
            label = None


if __name__ == "__main__":
    from PIL import Image

    import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
    from opencsp.common.lib.opencsp_path import opencsp_settings
    import opencsp.common.lib.tool.file_tools as ft
    import opencsp.common.lib.render.Color as color
    import opencsp.common.lib.render.figure_management as fm
    import opencsp.common.lib.render.view_spec as vs
    import opencsp.common.lib.render_control.RenderControlAxis as rca
    import opencsp.common.lib.render_control.RenderControlFigure as rcfg
    import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
    import opencsp.common.lib.render_control.RenderControlPointSeq as rcps

    src_dir = ft.join(
        opencsp_settings["opencsp_root_path"]["collaborative_dir"],
        "NSTTF_Optics/Experiments/2024-05-22_SolarNoonTowerTest_4/2_Data/BCS_data/20240522_115941 TowerTest4_NoSun/Raw Images",
    )
    src_file = "20240522_115941.65 TowerTest4_NoSun Raw.JPG"

    # Prepare the rectangles
    ul1, lr1 = p2.Pxy((447, 898)), p2.Pxy((519, 953))
    ul2, lr2 = p2.Pxy((1158, 877)), p2.Pxy((1241, 935))
    ul_lr_corners = (ul1.concatenate(ul2), lr1.concatenate(lr2))
    rectangles = RectangleAnnotations(upperleft_lowerright_corners=ul_lr_corners)

    # Load and update the image
    image = np.array(Image.open(ft.join(src_dir, src_file)))
    image2 = rectangles.render_to_image(image)

    # Draw the image using figure_management
    axis_control = rca.image(grid=False)
    figure_control = rcfg.RenderControlFigure()
    view_spec_2d = vs.view_spec_im()
    fig_record = fm.setup_figure(
        figure_control, axis_control, view_spec_2d, title="Flashlight ROIs", code_tag=f"{__file__}", equal=False
    )
    fig_record.view.imshow(image2)
    fig_record.view.show(block=True)
    fig_record.close()
