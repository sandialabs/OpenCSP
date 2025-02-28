import opencsp.common.lib.render_control.RenderControlFacetEnsemble as rcfe
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt

# Constants
DEFAULT_SURFACE_NORMAL_LENGTH = 4  # m
DEFAULT_CORNER_NORMAL_LENGTH = 2  # m


class RenderControlHeliostat:
    """
    Render control for heliostats.
    """

    # TODO: decide if some of these should be in heliostat and ensemble or just ensemble
    def __init__(
        self,
        draw_centroid=False,
        centroid_style=rcps.marker(),
        #  draw_outline=False,
        #  outline_style=rcps.outline(),
        #  draw_surface_normal=True, # unimplmeneted
        #  surface_normal_length=4, # unimplmeneted
        #  surface_normal_style=rcps.outline(),  # unimplmeneted
        #  surface_normal_base_style=rcps.marker(),  # unimplmeneted
        #  draw_surface_normal_at_corners=False,  # unimplmeneted
        #  corner_normal_length=2,  # unimplmeneted
        #  corner_normal_style=rcps.outline(),  # unimplmeneted
        #  corner_normal_base_style=rcps.marker(),  # unimplmeneted
        draw_facet_ensemble=True,
        facet_ensemble_style=rcfe.RenderControlFacetEnsemble(rcf.outline()),
        draw_name=False,  # unimplmeneted
        name_style=rctxt.default(color="k"),  # unimplmeneted
        post=0,  # by default there is no post
    ):
        """
        Render control for heliostats.

        This class manages the rendering settings for heliostats, allowing customization of various visual
        elements such as centroids, facet ensembles, and names.

        Parameters
        ----------
        draw_centroid : bool, optional
            Whether to draw the centroid of the heliostat. By default, False.
        centroid_style : object, optional
            Style for the centroid marker. By default, `rcps.marker()`.
        draw_facet_ensemble : bool, optional
            Whether to draw the facet ensemble. By default, True.
        facet_ensemble_style : object, optional
            Style for the facet ensemble. By default, `rcfe.RenderControlFacetEnsemble(rcf.outline())`.
        draw_name : bool, optional
            Whether to draw the name of the heliostat. By default, False (unimplemented).
        name_style : object, optional
            Style for the name text. By default, `rctxt.default(color='k')`.
        post : int, optional
            Identifier for the post associated with the heliostat. By default, 0 (no post).
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlHeliostat, self).__init__()

        self.draw_centroid = draw_centroid
        self.centroid_style = centroid_style
        # self.draw_outline = draw_outline
        # self.outline_style = outline_style
        # self.draw_surface_normal = draw_surface_normal
        # self.surface_normal_length = surface_normal_length
        # self.surface_normal_style = surface_normal_style
        # self.surface_normal_base_style = surface_normal_base_style
        # self.draw_surface_normal_at_corners = draw_surface_normal_at_corners
        # self.corner_normal_length = corner_normal_length
        # self.corner_normal_style = corner_normal_style
        # self.corner_normal_base_style = corner_normal_base_style

        self.draw_facet_ensemble = draw_facet_ensemble
        self.facet_ensemble_style = facet_ensemble_style
        self.draw_name = draw_name
        self.name_style = name_style
        self.post = post


# SPECIAL CASES


# COMMON CASES


def default():
    """
    Create a default render control for heliostats.

    This function returns a `RenderControlHeliostat` instance with default settings.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlHeliostat()


def blank():
    """
    Create a blank render control for heliostats.

    This function returns a `RenderControlHeliostat` instance with no visual elements drawn,
    suitable for cases where heliostats are added as special cases.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` with no visual elements.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Nothing.  (Used for cases where heliostats are added as special cases.)
    return RenderControlHeliostat(draw_centroid=False, draw_facet_ensemble=False)


def name(color='k', fontsize='medium'):
    """
    Create a render control that displays only the name of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw the name
    of the heliostat.

    Parameters
    ----------
    color : str, optional
        Color of the name text. By default, 'k' (black).
    fontsize : str, optional
        Font size of the name text. By default, 'medium'.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display only the name.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Name only.
    return RenderControlHeliostat(
        draw_centroid=True,  # Draw a tiny point to ensure that things will draw.
        centroid_style=rcps.marker(color=color, marker=".", markersize=0.1),
        draw_facet_ensemble=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(color=color, fontsize=fontsize),
    )


def centroid(color='k'):
    """
    Create a render control that displays only the centroid of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw only the centroid.

    Parameters
    ----------
    color : str, optional
        Color of the centroid marker. By default, 'k' (black).

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display only the centroid.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Centroid only.
    return RenderControlHeliostat(
        draw_centroid=True, centroid_style=rcps.marker(color=color), draw_facet_ensemble=False
    )


def centroid_name(color='k'):
    """
    Create a render control that displays both the centroid and the name of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw both the centroid
    and the name.

    Parameters
    ----------
    color : str, optional
        Color of the centroid marker. By default, 'k' (black).

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display both the centroid and the name.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Centroid and name.
    return RenderControlHeliostat(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_facet_ensemble=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(color=color, horizontalalignment="left"),
    )


def centroid_name_outline(
    color="k", horizontalalignment="center", verticalalignment="center"  # center, right, left
):  # center, top, bottom, baseline, center_baseline
    """
    Create a render control that displays the centroid, name, and overall outline of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw the centroid,
    the name, and the outline.

    Parameters
    ----------
    color : str, optional
        Color of the centroid marker and outline. By default, 'k' (black).
    horizontalalignment : str, optional
        Horizontal alignment of the name text. By default, 'center'.
    verticalalignment : str, optional
        Vertical alignment of the name text. By default, 'center'.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display the centroid, name, and outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Name and overall outline.
    return RenderControlHeliostat(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_facet_ensemble=True,
        facet_ensemble_style=rcfe.only_outline(color=color),
        draw_name=True,
        name_style=rctxt.RenderControlText(
            color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
        ),
    )


def outline(color='k'):
    """
    Create a render control that displays only the overall outline of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw only the outline.

    Parameters
    ----------
    color : str, optional
        Color of the outline. By default, 'k' (black).

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display only the outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall outline only.
    return RenderControlHeliostat(
        draw_centroid=False, draw_facet_ensemble=True, facet_ensemble_style=rcfe.only_outline(color=color)
    )


def name_outline(
    color="k", horizontalalignment="center", verticalalignment="center"  # center, right, left
):  # center, top, bottom, baseline, center_baseline
    """
    Create a render control that displays the name and overall outline of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw the name and the outline.

    Parameters
    ----------
    color : str, optional
        Color of the outline. By default, 'k' (black).
    horizontalalignment : str, optional
        Horizontal alignment of the name text. By default, 'center'.
    verticalalignment : str, optional
        Vertical alignment of the name text. By default, 'center'.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display the name and outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Name and overall outline.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facet_ensemble=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(
            color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
        ),
    )


def facet_outlines_names(color='k'):
    """
    Create a render control that displays facet outlines and facet name labels.

    This function returns a `RenderControlHeliostat` instance configured to draw the outlines of the facets
    along with their corresponding names.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines and names. By default, 'k' (black).

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display facet outlines and names.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Facet outlines and facet name labels.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_facet_ensemble=True,
        facet_ensemble_style=rcfe.RenderControlFacetEnsemble(rcf.outline_name(color=color)),
        draw_name=False,
    )


def normal(color='k', normal_vector_length=DEFAULT_SURFACE_NORMAL_LENGTH):
    """
    Create a render control that displays the overall surface normal of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw the surface normal.

    Parameters
    ----------
    color : str, optional
        Color of the surface normal. By default, 'k' (black).
    normal_vector_length : float, optional
        Length of the surface normal. By default, 4.0.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display the surface normal.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall surface normal only.
    return RenderControlHeliostat(
        draw_centroid=True,
        #   draw_normal_vector=True,
        #   normal_vector_length=normal_vector_length,
        #   normal_vector_style=rcps.outline(color=color),
        #   normal_vector_base_style=rcps.marker(color=color),
        draw_facet_ensemble=True,
        facet_ensemble_style=rcfe.normal_only(color=color, normal_vector_length=normal_vector_length),
    )


def normal_outline(color='k', normal_vector_length=DEFAULT_SURFACE_NORMAL_LENGTH, **kwargs):
    """
    Create a render control that displays both the overall surface normal and the outline of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw the surface normal and the outline.

    Parameters
    ----------
    color : str, optional
        Color of the surface normal and outline. By default, 'k' (black).
    normal_vector_length : float, optional
        Length of the surface normal. By default, 4.0.
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlHeliostat`.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display the surface normal and outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall surface normal and overall outline.
    fe_style = rcfe.facet_ensemble_outline(color=color, normal_vector_length=normal_vector_length)
    return RenderControlHeliostat(
        draw_centroid=False, facet_ensemble_style=fe_style, draw_facet_ensemble=True, draw_name=False, **kwargs
    )


def normal_facet_outlines_names(color='k'):
    """
    Create a render control that displays facet outlines and facet name labels with normal vectors.

    This function returns a `RenderControlHeliostat` instance configured to draw the outlines of the facets
    along with their corresponding names and normal vectors.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines and names. By default, 'k' (black).

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display facet outlines, names, and normal vectors.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Facet outlines and facet name labels.
    fe_style = rcfe.RenderControlFacetEnsemble(
        draw_normal_vector=True,
        normal_vector_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        default_style=rcf.outline_name(color=color),
    )
    return RenderControlHeliostat(
        draw_centroid=False, draw_facet_ensemble=True, facet_ensemble_style=fe_style, draw_name=False
    )


def mirror_surfaces(color='k', **kwargs):
    """
    Create a render control that displays the mirror surfaces of the heliostat.

    This function returns a `RenderControlHeliostat` instance configured to draw the mirror surfaces.

    Parameters
    ----------
    color : str, optional
        Color of the mirror surfaces. By default, 'k' (black).
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlHeliostat`.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display the mirror surfaces.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    fe_style = rcfe.RenderControlFacetEnsemble(rcf.normal_mirror_surface(color=color))
    return RenderControlHeliostat(
        draw_centroid=False, draw_facet_ensemble=True, facet_ensemble_style=fe_style, **kwargs
    )


# TODO implement the normals at corners
# def corner_normals_outline(color='k'):
#     # Overall outline, and overall surface normal drawn at each overall corner.
#     return RenderControlHeliostat(draw_centroid=False,
#                                 #   draw_outline=True,
#                                 #   outline_style=rcps.outline(color=color),
#                                 #   draw_surface_normal_at_corners=True # unimplemented
#                                   corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
#                                   corner_normal_style=rcps.outline(color=color),
#                                   corner_normal_base_style=rcps.marker(color=color),
#                                   draw_facet_ensemble=True,
#                                   draw_name=False)


def normal_facet_outlines(color='k'):
    """
    Create a render control that displays normal facet outlines.

    This function returns a `RenderControlHeliostat` instance configured to draw the outlines of the facets
    with normal vectors.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines. By default, 'k' (black).

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display normal facet outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlHeliostat(facet_ensemble_style=rcfe.normal_facet_outlines(color=color))


def facet_outlines(color='k', **kwargs):
    """
    Create a render control that displays facet outlines only.

    This function returns a `RenderControlHeliostat` instance configured to draw only the outlines of the facets.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines. By default, 'k' (black).
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlHeliostat`.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display facet outlines only.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Facet outline only.
    return RenderControlHeliostat(facet_ensemble_style=rcfe.facet_outlines(color=color))


def facet_outlines_normals(color='k'):
    """
    Create a render control that displays facet outlines with normal vectors.

    This function returns a `RenderControlHeliostat` instance configured to draw the outlines of the facets
    along with their normal vectors.

    Parameters
    ----------
    color : str, optional
        Color of the facet outlines and normal vectors. By default, 'k' (black).

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured to display facet outlines with normal vectors.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    fe_style = rcfe.RenderControlFacetEnsemble(default_style=rcf.normal_outline(color=color))
    return RenderControlHeliostat(draw_facet_ensemble=True, facet_ensemble_style=fe_style)


# TODO implement the normals at corners
# def facet_outlines_corner_normals(color='k'):
#     return RenderControlHeliostat(draw_centroid=False,
#                                   draw_outline=False,
#                                   draw_surface_normal=False,
#                                   surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
#                                   surface_normal_style=rcps.outline(color=color),
#                                   surface_normal_base_style=rcps.marker(color=color),
#                                   draw_surface_normal_at_corners=False,
#                                   draw_facet_ensemble=True,
#                                   facet_ensemble_style=rcfe.RenderControlFacetEnsemble(rcf.corner_normals_outline(color=color)),
#                                   draw_name=False)


# TODO this is not working, check if it should be fixed
# def highlight(color='b'):
#     return RenderControlHeliostat(centroid_style=rcps.marker(color=color),
#                                   outline_style=rcps.outline(color=color),
#                                   surface_normal_style=rcps.outline(color=color),
#                                   surface_normal_base_style=rcps.marker(color=color),
#                                   corner_normal_style=rcps.outline(color=color),
#                                   corner_normal_base_style=rcps.marker(color=color),
#                                   draw_facet_ensemble=False,
#                                   facet_ensemble_style=rcfe.RenderControlFacetEnsemble(rcf.outline()),
#                                   name_style=rctxt.default(color=color))


def low_res_heliostat():
    """
    Create a low-resolution render control for heliostats.

    This function returns a `RenderControlHeliostat` instance configured for low-resolution rendering
    of the heliostat, with specific styles for facets.

    Returns
    -------
    RenderControlHeliostat
        An instance of `RenderControlHeliostat` configured for low-resolution rendering.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlHeliostat(
        facet_ensemble_style=rcfe.RenderControlFacetEnsemble(
            default_style=rcf.RenderControlFacet(
                draw_outline=False,
                draw_centroid=False,
                draw_name=False,
                draw_surface_normal_at_corners=False,
                draw_mirror_curvature=True,
                surface_normal_length=60,
                mirror_styles=rcm.low_res_mirror(),
            )
        ),
        draw_facet_ensemble=True,
        draw_centroid=False,
    )
