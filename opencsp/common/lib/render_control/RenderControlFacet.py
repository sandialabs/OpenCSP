import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.render_control.RenderControlMirror as rcm


# Constants
DEFAULT_SURFACE_NORMAL_LENGTH = 2  # m
DEFAULT_CORNER_NORMAL_LENGTH = 1  # m


class RenderControlFacet:
    """
    Render control for heliostat facets.
    """

    def __init__(
        self,
        draw_centroid=False,
        centroid_style=rcps.marker(),
        draw_outline=True,
        outline_style=rcps.outline(),
        draw_surface_normal=False,
        surface_normal_length=4,
        surface_normal_style=rcps.outline(),
        surface_normal_base_style=rcps.marker(),
        # draw_surface_normal_at_corners=False, # Unimplemented
        # corner_normal_length=2,               # Unimplemented
        # corner_normal_style=rcps.outline(),   # Unimplemented
        # corner_normal_base_style=rcps.marker(), # Unimplemented
        draw_name=False,
        name_style=rctxt.default(color="k"),
        draw_mirror_curvature=False,
        mirror_styles=rcm.RenderControlMirror(),
    ):
        """
        Render control for heliostat facets.

        This class manages the rendering settings for heliostat facets, allowing customization
        of various visual elements such as centroids, outlines, surface normals, and mirror curvature.

        Parameters
        ----------
        draw_centroid : bool, optional
            Whether to draw the centroid of the facet. By default, False.
        centroid_style : object, optional
            Style for the centroid marker. By default, rcps.marker().
        draw_outline : bool, optional
            Whether to draw the outline of the facet. By default, True.
        outline_style : object, optional
            Style for the outline. By default, rcps.outline().
        draw_surface_normal : bool, optional
            Whether to draw the surface normal. By default, False.
        surface_normal_length : float, optional
            Length of the surface normal. By default, 4.
        surface_normal_style : object, optional
            Style for the surface normal. By default, rcps.outline().
        surface_normal_base_style : object, optional
            Style for the base of the surface normal. By default, rcps.marker().
        draw_name : bool, optional
            Whether to draw the name of the facet. By default, False.
        name_style : object, optional
            Style for the name text. By default, rctxt.default(color='k').
        draw_mirror_curvature : bool, optional
            Whether to draw the curvature of the mirror. By default, False.
        mirror_styles : object, optional
            Styles for the mirror. By default, rcm.RenderControlMirror().
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlFacet, self).__init__()

        self.draw_centroid = draw_centroid
        self.centroid_style = centroid_style
        self.draw_outline = draw_outline
        self.outline_style = outline_style
        self.draw_surface_normal = draw_surface_normal
        self.surface_normal_length = surface_normal_length
        self.surface_normal_style = surface_normal_style
        self.surface_normal_base_style = surface_normal_base_style
        # self.draw_surface_normal_at_corners = draw_surface_normal_at_corners
        # self.corner_normal_length = corner_normal_length
        # self.corner_normal_style = corner_normal_style
        # self.corner_normal_base_style = corner_normal_base_style
        self.draw_name = draw_name
        self.name_style = name_style
        self.draw_mirror_curvature = draw_mirror_curvature
        self.mirror_styles = mirror_styles

    def style(self, imput_name: str):
        return self


# COMMON CASES


def default():
    """
    Create a default render control for heliostat facets.

    This function returns a `RenderControlFacet` instance with default settings.

    Returns
    -------
    RenderControlFacet
        An instance of `RenderControlFacet` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacet()


def outline(color='k'):
    """
    Create a render control that displays only the outline of the facet.

    This function returns a `RenderControlFacet` instance configured to draw the outline
    without the centroid or surface normal.

    Parameters
    ----------
    color : str, optional
        Color of the outline. By default, 'k' (black).

    Returns
    -------
    RenderControlFacet
        An instance of `RenderControlFacet` configured to display only the outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_name=False,
    )


def outline_thin(color="k", linewidth=0.5):
    """
    Create a render control that displays a thin outline of the facet.

    This function returns a `RenderControlFacet` instance configured to draw a thin outline
    of the facet without the centroid or surface normal.

    Parameters
    ----------
    color : str, optional
        Color of the outline. By default, 'k' (black).
    linewidth : float, optional
        Width of the outline line. By default, 0.5.

    Returns
    -------
    RenderControlFacet
        An instance of `RenderControlFacet` configured to display a thin outline.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color, linewidth=linewidth),
        draw_surface_normal=False,
        draw_name=False,
    )


def outline_name(color='k'):
    """
    Create a render control that displays the outline and the name of the facet.

    This function returns a `RenderControlFacet` instance configured to draw the outline
    and the name of the facet.

    Parameters
    ----------
    color : str, optional
        Color of the outline and name. By default, 'k' (black).

    Returns
    -------
    RenderControlFacet
        An instance of `RenderControlFacet` configured to display the outline and name.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_name=True,
        name_style=rctxt.default(color=color),
    )


def normal_mirror_surface(color='k'):
    """
    Create a render control that displays the normal mirror surface.

    This function returns a `RenderControlFacet` instance configured to draw the mirror curvature
    without the centroid, outline, or surface normal.

    Parameters
    ----------
    color : str, optional
        Color of the outline. By default, 'k' (black).

    Returns
    -------
    RenderControlFacet
        An instance of `RenderControlFacet` configured to display the normal mirror surface.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=False,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_name=False,
        name_style=rctxt.default(color=color),
        draw_mirror_curvature=True,
        mirror_styles=rcm.normal_mirror(),
    )


def normal_outline(color='k'):
    """
    Create a render control that displays the normal and outline of the facet.

    This function returns a `RenderControlFacet` instance configured to draw the outline
    and the surface normal.

    Parameters
    ----------
    color : str, optional
        Color of the outline and surface normal. By default, 'k' (black).

    Returns
    -------
    RenderControlFacet
        An instance of `RenderControlFacet` configured to display the normal and outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=True,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_name=False,
    )


def corner_normals_outline_NOTWORKING(color="k"):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=True,
        corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        draw_name=False,
    )


def corner_normals_outline_name_NOTWORKING(color="k"):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=True,
        corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        draw_name=True,
        name_style=rctxt.default(color=color),
    )


def highlight_NOTWORKING(color="b"):
    return RenderControlFacet(
        centroid_style=rcps.marker(color=color),
        outline_style=rcps.outline(color=color),
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        name_style=rctxt.default(color=color),
    )
