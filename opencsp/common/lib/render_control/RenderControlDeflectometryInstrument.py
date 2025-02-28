import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt


# Constants
DEFAULT_SURFACE_NORMAL_LENGTH = 4  # m
DEFAULT_CORNER_NORMAL_LENGTH = 2  # m


class RenderControlDeflectometryInstrument:
    """
    Render control for deflectometry instruments.
    """

    def __init__(
        self,
        draw_centroid=True,
        centroid_style=rcps.marker(),
        draw_outline=True,
        outline_style=rcps.outline(),
        draw_surface_normal=True,
        surface_normal_length=4,
        surface_normal_style=rcps.outline(),
        surface_normal_base_style=rcps.marker(),
        draw_surface_normal_at_corners=True,
        corner_normal_length=2,
        corner_normal_style=rcps.outline(),
        corner_normal_base_style=rcps.marker(),
        draw_facets=False,
        facet_styles=rce.RenderControlEnsemble(rcf.outline()),
        draw_name=False,
        name_style=rctxt.default(color="k"),
    ):
        """
        Render control for deflectometry instruments.

        This class manages the rendering settings for deflectometry instruments, allowing customization
        of various visual elements such as centroids, outlines, surface normals, and facets.

        Parameters
        ----------
        draw_centroid : bool, optional
            Whether to draw the centroid. By default, True.
        centroid_style : object, optional
            Style for the centroid marker. By default, rcps.marker().
        draw_outline : bool, optional
            Whether to draw the outline. By default, True.
        outline_style : object, optional
            Style for the outline. By default, rcps.outline().
        draw_surface_normal : bool, optional
            Whether to draw the surface normal. By default, True.
        surface_normal_length : float, optional
            Length of the surface normal. By default, 4.
        surface_normal_style : object, optional
            Style for the surface normal. By default, rcps.outline().
        surface_normal_base_style : object, optional
            Style for the base of the surface normal. By default, rcps.marker().
        draw_surface_normal_at_corners : bool, optional
            Whether to draw surface normals at corners. By default, True.
        corner_normal_length : float, optional
            Length of the corner normal. By default, 2.
        corner_normal_style : object, optional
            Style for the corner normal. By default, rcps.outline().
        corner_normal_base_style : object, optional
            Style for the base of the corner normal. By default, rcps.marker().
        draw_facets : bool, optional
            Whether to draw facets. By default, False.
        facet_styles : object, optional
            Styles for the facets. By default, rce.RenderControlEnsemble(rcf.outline()).
        draw_name : bool, optional
            Whether to draw the name. By default, False.
        name_style : object, optional
            Style for the name text. By default, rctxt.default(color='k').
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlDeflectometryInstrument, self).__init__()

        self.draw_centroid = draw_centroid
        self.centroid_style = centroid_style
        self.draw_outline = draw_outline
        self.outline_style = outline_style
        self.draw_surface_normal = draw_surface_normal
        self.surface_normal_length = surface_normal_length
        self.surface_normal_style = surface_normal_style
        self.surface_normal_base_style = surface_normal_base_style
        self.draw_surface_normal_at_corners = draw_surface_normal_at_corners
        self.corner_normal_length = corner_normal_length
        self.corner_normal_style = corner_normal_style
        self.corner_normal_base_style = corner_normal_base_style
        self.draw_facets = draw_facets
        self.facet_styles = facet_styles
        self.draw_name = draw_name
        self.name_style = name_style


# COMMON CASES


def default():
    """
    Create a default render control for deflectometry instruments.

    This function returns a `RenderControlDeflectometryInstrument` instance with standard settings.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlDeflectometryInstrument()


def blank():
    """
    Create a blank render control for deflectometry instruments.

    This function returns a `RenderControlDeflectometryInstrument` instance with all drawing options disabled.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` with no visual elements drawn.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Nothing.  (Used for cases where heliostats are added as special cases.)
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def name(color='k', fontsize='medium'):
    """
    Create a render control that displays only the name.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    only the name of the instrument.

    Parameters
    ----------
    color : str, optional
        Color of the name text. By default, 'k'.
    fontsize : str, optional
        Font size of the name text. By default, 'medium'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display only the name.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Name only.
    return RenderControlDeflectometryInstrument(
        draw_centroid=True,  # Draw a tiny point to ensure that things will draw.
        centroid_style=rcps.marker(color=color, marker=".", markersize=0.1),
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(color=color, fontsize=fontsize),
    )


def centroid(color='k'):
    """
    Create a render control that displays only the centroid.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    only the centroid of the instrument.

    Parameters
    ----------
    color : str, optional
        Color of the centroid marker. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display only the centroid.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Centroid only.
    return RenderControlDeflectometryInstrument(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def centroid_name(color='k'):
    """
    Create a render control that displays both the centroid and the name.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    both the centroid and the name of the instrument.

    Parameters
    ----------
    color : str, optional
        Color of the centroid marker. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display both the centroid and the name.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Centroid and name.
    return RenderControlDeflectometryInstrument(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(color=color, horizontalalignment="left"),
    )


def centroid_name_outline(
    color="k", horizontalalignment="center", verticalalignment="center"  # center, right, left
):  # center, top, bottom, baseline, center_baseline
    """
    Create a render control that displays the centroid, name, and overall outline.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    the centroid, the name, and the outline of the instrument.

    Parameters
    ----------
    color : str, optional
        Color of the centroid marker and outline. By default, 'k'.
    horizontalalignment : str, optional
        Horizontal alignment of the name text. By default, 'center'.
    verticalalignment : str, optional
        Vertical alignment of the name text. By default, 'center'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display the centroid, name, and outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Name and overall outline.
    return RenderControlDeflectometryInstrument(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(
            color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
        ),
    )


def outline(color='k'):
    """
    Create a render control that displays only the overall outline.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    only the outline of the instrument.

    Parameters
    ----------
    color : str, optional
        Color of the outline. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display only the outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall outline only.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def name_outline(
    color="k", horizontalalignment="center", verticalalignment="center"  # center, right, left
):  # center, top, bottom, baseline, center_baseline
    """
    Create a render control that displays the name and overall outline.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    the name and the outline of the instrument.

    Parameters
    ----------
    color : str, optional
        Color of the outline. By default, 'k'.
    horizontalalignment : str, optional
        Horizontal alignment of the name text. By default, 'center'.
    verticalalignment : str, optional
        Vertical alignment of the name text. By default, 'center'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display the name and outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Name and overall outline.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(
            color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
        ),
    )


def facet_outlines(color='k'):
    """
    Create a render control that displays only the facet outlines.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    only the outlines of the facets.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display only the facet outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Facet outline only.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline(color=color)),
        draw_name=False,
    )


def facet_outlines_names(color='k'):
    """
    Create a render control that displays facet outlines and facet name labels.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    the outlines of the facets along with their corresponding names.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines and names. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display facet outlines and names.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Facet outlines and facet name labels.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline_name(color=color)),
        draw_name=False,
    )


def normal(color='k', surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH):
    """
    Create a render control that displays the overall surface normal.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    the surface normal of the instrument.

    Parameters
    ----------
    color : str, optional
        Color of the surface normal and its base. By default, 'k'.
    surface_normal_length : float, optional
        Length of the surface normal. By default, 4.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display the surface normal.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall surface normal only.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=True,
        surface_normal_length=surface_normal_length,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def normal_outline(color='k', surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH):
    """
    Create a render control that displays the overall surface normal and overall outline.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    both the surface normal and the outline of the instrument.

    Parameters
    ----------
    color : str, optional
        Color of the surface normal and outline. By default, 'k'.
    surface_normal_length : float, optional
        Length of the surface normal. By default, 4.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display the surface normal and outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall surface normal and overall outline.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=True,
        surface_normal_length=surface_normal_length,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def normal_facet_outlines_names(color='k'):
    """
    Create a render control that displays facet outlines and facet name labels along with the surface normal.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    the surface normal and the outlines of the facets along with their corresponding names.

    Parameters
    ----------
    color : str, optional
        Color of the surface normal and facet outlines. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display facet outlines, names, and surface normal.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Facet outlines and facet name labels.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=True,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline_name(color=color)),
        draw_name=False,
    )


def corner_normals_outline(color='k'):
    """
    Create a render control that displays the overall outline and surface normals at each corner.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    the outline of the instrument and surface normals at each corner.

    Parameters
    ----------
    color : str, optional
        Color of the outline and corner normals. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display the outline and corner normals.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall outline, and overall surface normal drawn at each overall corner.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=True,
        corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        draw_facets=False,
        draw_name=False,
    )


def normal_facet_outlines(color='k'):
    """
    Create a render control that displays the surface normal and facet outlines.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    the surface normal and the outlines of the facets.

    Parameters
    ----------
    color : str, optional
        Color of the surface normal and facet outlines. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display the surface normal and facet outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=True,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline(color=color)),
        draw_name=False,
    )


def facet_outlines_normals(color='k'):
    """
    Create a render control that displays facet outlines without surface normals.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    only the outlines of the facets.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display facet outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.normal_outline(color=color)),
        draw_name=False,
    )


def facet_outlines_corner_normals(color='k'):
    """
    Create a render control that displays facet outlines and surface normals at corners.

    This function returns a `RenderControlDeflectometryInstrument` instance configured to draw
    the outlines of the facets and surface normals at the corners.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines and corner normals. By default, 'k'.

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured to display facet outlines and corner normals.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlDeflectometryInstrument(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.corner_normals_outline(color=color)),
        draw_name=False,
    )


def highlight(color='b'):
    """
    Create a render control with highlighted styles for deflectometry instruments.

    This function returns a `RenderControlDeflectometryInstrument` instance configured with
    styles that emphasize the centroid, outline, and surface normals, making them visually distinct.

    Parameters
    ----------
    color : str, optional
        Color for the highlighted elements (centroid, outline, surface normals, and corner normals).
        By default, 'b' (blue).

    Returns
    -------
    RenderControlDeflectometryInstrument
        An instance of `RenderControlDeflectometryInstrument` configured with highlighted styles.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlDeflectometryInstrument(
        centroid_style=rcps.marker(color=color),
        outline_style=rcps.outline(color=color),
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        draw_facets=False,
        facet_styles=rce.RenderControlEnsemble(rcf.outline()),
        name_style=rctxt.default(color=color),
    )


# name
# centroid
# centroid_name
## outline
## normal_outline
# normal_corners_outline
# facet_outlines
## normal_facet_outlines
# facet_outlines_normals
