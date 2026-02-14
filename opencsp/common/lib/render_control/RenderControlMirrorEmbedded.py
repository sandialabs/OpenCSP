import opencsp.common.lib.render_control.RenderControlMirrorProjected as rcmp


class RenderControlMirrorEmbedded:
    """
    A class for controlling the construction and rendering of a mirror representing
    the embedding surface of a parametric mirror.
    """

    def __init__(
        self,
        margin: float = 0.1,
        round_to: float = 0.25,
        n_slices_x: int = 11,
        project_slices_x: bool = False,
        draw_special_origin_slice_x: bool = True,
        project_origin_slice_x: bool = True,
        n_slices_y: int = 11,
        project_slices_y: bool = False,
        draw_special_origin_slice_y: bool = True,
        project_origin_slice_y: bool = True,
        slice_style: rcmp.RenderControlMirrorProjected = None,
        origin_slice_style: rcmp.RenderControlMirrorProjected = None,
    ) -> None:
        """
        Initializes a RenderControlMirrorEmbedded object with the specified parameters.

        Parameters
        ----------
        margin : float, optional
            Minimum amount embedding surface should extend beyond mirror boundary, Default 0.1
        round_to : float, optional
            Even increment to extend margin to match plot axis tick mark.  Default 0.25
        n_slices_x : int, optional
            Number of constant-x contour slides to draw.  Default 11.
        project_slices_x : bool, optional
            Whether to show projection of constant-x slices to the (x,y) plane.  Default False.
        draw_special_origin_slice_x : bool, optional
            Whether to draw a special constant-x slice through x=0.  Default True.
        project_origin_slice_x : bool, optional
            Whether to show projection for the special constant-x slice at x=0.  Default True.
        n_slices_y : int, optional
            Number of constant-y contour slides to draw.  Default 11.
        project_slices_y : bool, optional
            Whether to show projection of constant-y slices to the (x,y) plane.  Default False.
        draw_special_origin_slice_y : bool, optional
            Whether to draw a special constant-y slice through y=0.  Default True.
        project_origin_slice_y : bool, optional
            Whether to show projection for the special constant-x slice at x=0.  Default True.
        slice_style : RenderControlMirrorProjected, optional
            Drawing attribute options for the contour slices.
            None implies use default contour style.
        origin_slice_style : RenderControlMirrorProjected, optional
            Drawing attribute options for the special origin slices.
            None implies use default origin contour style.
        """
        self.round_to = round_to
        self.margin = margin
        self.n_slices_x = n_slices_x
        self.project_slices_x = project_slices_x
        self.draw_special_origin_slice_x = draw_special_origin_slice_x
        self.project_origin_slice_x = project_origin_slice_x
        self.n_slices_y = n_slices_y
        self.project_slices_y = project_slices_y
        self.draw_special_origin_slice_y = draw_special_origin_slice_y
        self.project_origin_slice_y = project_origin_slice_y
        if slice_style is None:
            self.slice_style = slice_style
        else:
            self.slice_style = rcmp.mirror_contour()
        if origin_slice_style is None:
            self.origin_slice_style = origin_slice_style
        else:
            self.origin_slice_style = rcmp.mirror_origin_contour()


# Common Configurations


def standard_embedding_mirror() -> RenderControlMirrorEmbedded:
    return RenderControlMirrorEmbedded(
        margin=0.1,
        round_to=0.25,
        n_slices_x=9,
        project_slices_x=False,
        draw_special_origin_slice_x=True,
        project_origin_slice_x=True,
        n_slices_y=9,
        project_slices_y=False,
        draw_special_origin_slice_y=True,
        project_origin_slice_y=True,
        slice_style=rcmp.mirror_contour(),
        origin_slice_style=rcmp.mirror_origin_contour(),
    )
