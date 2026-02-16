import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface

# from opencsp.common.lib.geometry.Resolution import Resolution


class RenderControlMirrorProjected:
    """
    A class for controlling the rendering of a mirror projection onto the (x,y) plane
    in a graphical environment.

    This class allows for the configuration of various visual aspects of a projection,
    including its lifted, projected, and connection line features.
    """

    def __init__(
        self,
        lifted_line_style: RenderControlPointSeq = None,
        lifted_vertex_style: RenderControlPointSeq = None,
        projected_style: RenderControlPointSeq = None,
        connection_style: RenderControlPointSeq = None,
        slice_n_vertices: int = 9,
        slice_fine_n_vertices: int = 41,
    ) -> None:
        """
        Initializes a RenderControlMirrorProjected object with the specified parameters.

        Parameters
        ----------
        lifted_line_style : RenderControlPointSeq | None, optional
            Style to draw the contour of the lifted slice.
            Default None.
        lifted_vertex_style : RenderControlPointSeq | None, optional
            Style to draw the coarse vertices of the lifted slice.
            Default None.
        projected_style: RenderControlPointSeq | None, optional
            Style to draw the projected slice.
            Default None.
        connection_style: RenderControlPointSeq | None, optional
            Style to draw the connection line between the projected and lifted slices.
            Default None.
        slice_n_vertices : int, optional
            Default 9.
        slice_fine_n_vertices:int, optional
            Default 41.
        """
        self.lifted_line_style = lifted_line_style
        self.lifted_vertex_style = lifted_vertex_style
        self.projected_style = projected_style
        self.connection_style = connection_style
        self.slice_n_vertices = slice_n_vertices
        self.slice_fine_n_vertices = slice_fine_n_vertices


# Common Configurations


def mirror_contour(slice_n_vertices: int = 5) -> RenderControlMirrorProjected:
    """Style for drawing a contour illustrating a mirror surface"""
    return RenderControlMirrorProjected(
        lifted_line_style=rcps.outline(color='grey', linewidth=0.5), slice_n_vertices=slice_n_vertices
    )


def mirror_origin_contour(slice_n_vertices: int = 5) -> RenderControlMirrorProjected:
    """Style for drawing a contour illustrating a mirror surface slice passing through the (0,0) origin."""
    return RenderControlMirrorProjected(
        lifted_line_style=rcps.outline(color='grey', linewidth=1.0),
        lifted_vertex_style=rcps.marker(color='grey', markersize=1.3),
        projected_style=rcps.outline(color='grey', linewidth=0.75),
        connection_style=rcps.outline(color='grey', linewidth=0.75),
        slice_n_vertices=slice_n_vertices,
    )


def mirror_boundary() -> RenderControlMirrorProjected:
    """Style for drawing the projection of a mirror XyRegion up to the embedding surface."""
    return RenderControlMirrorProjected(
        lifted_line_style=rcps.outline(color='red'),
        lifted_vertex_style=rcps.marker(color='red', markersize=2),
        projected_style=rcps.outline(color='blue'),
        connection_style=rcps.outline(color='blue', linewidth=0.6),
    )
