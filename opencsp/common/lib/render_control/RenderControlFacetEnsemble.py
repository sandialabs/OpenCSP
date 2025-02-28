from typing import Iterable
from warnings import warn

import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.tool.typing_tools import strict_types


class RenderControlFacetEnsemble:
    """
    Render control for collections of named facets.

    Provides a default render copntrol, with exceptions for objets with specific names.

    Multiple classes of exceptions can be defineid, each with its own specialized render style.

    Render styles may be of arbitrary type:  RenderControlFacet, RenderControlHeliostat, etc.

    """

    def __init__(
        self,
        default_style: rcf.RenderControlFacet = rcf.RenderControlFacet(),
        draw_facets=True,
        special_styles: dict[str, rcf.RenderControlFacet] = None,
        draw_centroid=False,
        draw_normal_vector=False,
        normal_vector_length=4.0,
        normal_vector_style=rcps.outline(),
        normal_vector_base_style=rcps.marker(),
        draw_outline=False,
        outline_style=rcps.outline(),
        draw_surface_normal_at_corners=False,  # unimplmeneted
        corner_normal_length=2,  # unimplmeneted
        corner_normal_style=rcps.outline(),  # unimplmeneted
        corner_normal_base_style=rcps.marker(),  # unimplmeneted
    ):
        """
        Render control for collections of named facets.

        Provides a default render control, with exceptions for objects with specific names.
        Multiple classes of exceptions can be defined, each with its own specialized render style.
        Render styles may be of arbitrary type: RenderControlFacet, RenderControlHeliostat, etc.

        Parameters
        ----------
        default_style : RenderControlFacet, optional
            The default rendering style for facets. By default, an instance of `RenderControlFacet`.
        draw_facets : bool, optional
            Whether to draw the facets. By default, True.
        special_styles : dict[str, RenderControlFacet], optional
            A dictionary mapping facet names to their specialized rendering styles. By default, None.
        draw_centroid : bool, optional
            Whether to draw the centroid of the facets. By default, False.
        draw_normal_vector : bool, optional
            Whether to draw the normal vector for the facets. By default, False.
        normal_vector_length : float, optional
            Length of the normal vector. By default, 4.0.
        normal_vector_style : object, optional
            Style for the normal vector. By default, `rcps.outline()`.
        normal_vector_base_style : object, optional
            Style for the base of the normal vector. By default, `rcps.marker()`.
        draw_outline : bool, optional
            Whether to draw the outline of the facets. By default, False.
        outline_style : object, optional
            Style for the outline. By default, `rcps.outline()`.
        draw_surface_normal_at_corners : bool, optional
            Whether to draw surface normals at corners. By default, False (unimplemented).
        corner_normal_length : float, optional
            Length of the corner normal. By default, 2.0 (unimplemented).
        corner_normal_style : object, optional
            Style for the corner normal. By default, `rcps.outline()` (unimplemented).
        corner_normal_base_style : object, optional
            Style for the base of the corner normal. By default, `rcps.marker()` (unimplemented).
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.draw_facets = draw_facets
        self.default_style = default_style
        self.draw_normal_vector = draw_normal_vector
        self.normal_vector_style = normal_vector_style

        self.draw_outline = draw_outline
        self.outline_style = outline_style
        self.draw_normal_vector = draw_normal_vector
        self.normal_vector_length = normal_vector_length
        self.normal_vector_style = normal_vector_style
        self.normal_vector_base_style = normal_vector_base_style
        self.draw_surface_normal_at_corners = draw_surface_normal_at_corners
        self.corner_normal_length = corner_normal_length
        self.corner_normal_style = corner_normal_style
        self.corner_normal_base_style = corner_normal_base_style

        if special_styles is None:
            special_styles = {}
        self.special_styles = special_styles

        self.draw_centroid = draw_centroid

    def get_facet_style(self, facet_name: str):
        """
        Retrieve the rendering style for a specific facet.

        This method checks if a special style exists for the given facet name and returns it.
        If no special style is found, the default style is returned.

        Parameters
        ----------
        facet_name : str
            The name of the facet for which to retrieve the rendering style.

        Returns
        -------
        RenderControlFacet
            The rendering style associated with the specified facet name.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        style = self.default_style
        if facet_name in self.special_styles:
            style = self.special_styles[facet_name]
        return style

    # @strict_types
    def add_special_style(self, facet_name: str | list, facet_style: rcf.RenderControlFacet):
        """
        Add a special rendering style for a specific facet or a list of facets.

        This method allows the user to associate a custom rendering style with a facet name or
        multiple facet names. If a facet name is None, a warning is issued.

        Parameters
        ----------
        facet_name : str or list
            The name of the facet or a list of facet names to which the special style will be applied.
        facet_style : RenderControlFacet
            The rendering style to associate with the specified facet name(s).

        Raises
        ------
        UserWarning
            If `facet_name` is None, a warning is issued indicating that special styles should not be applied.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if type(facet_name) is list and isinstance(facet_name[0], str):
            for name in facet_name:
                self.add_special_style(name, facet_style)
            return
        if facet_name is None:
            warn("Facets with name=None should not have special styles in RenderControlFacetEnsemble.")
        self.special_styles[facet_name] = facet_style


# GENERATORS


def normal_facet_outlines(color='k', **kwargs):
    """
    Create a render control ensemble with normal facet outlines.

    This function returns a `RenderControlFacetEnsemble` instance configured to draw normal
    facet outlines with specified styles.

    Parameters
    ----------
    color : str, optional
        Color for the outlines and normal vectors. By default, 'k' (black).
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFacetEnsemble`.

    Returns
    -------
    RenderControlFacetEnsemble
        An instance of `RenderControlFacetEnsemble` configured for normal facet outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacetEnsemble(
        draw_normal_vector=True,
        default_style=rcf.outline(color=color),
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        draw_centroid=True,
        **kwargs
    )


def facet_outlines(color='k', **kwargs):
    """
    Create a render control ensemble with facet outlines.

    This function returns a `RenderControlFacetEnsemble` instance configured to draw facet
    outlines without normal vectors or centroids.

    Parameters
    ----------
    color : str, optional
        Color for the outlines. By default, 'k' (black).
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFacetEnsemble`.

    Returns
    -------
    RenderControlFacetEnsemble
        An instance of `RenderControlFacetEnsemble` configured for facet outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacetEnsemble(
        draw_normal_vector=False,
        default_style=rcf.outline(color=color),
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        draw_centroid=False,
        **kwargs
    )


def facet_outlines_thin(color="k", linewidth=0.25, **kwargs):
    """
    Create a render control ensemble with thin facet outlines.

    This function returns a `RenderControlFacetEnsemble` instance configured to draw thin
    outlines of the facets without normal vectors or centroids.

    Parameters
    ----------
    color : str, optional
        Color for the outlines. By default, 'k' (black).
    linewidth : float, optional
        Width of the outline line. By default, 0.25.
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFacetEnsemble`.

    Returns
    -------
    RenderControlFacetEnsemble
        An instance of `RenderControlFacetEnsemble` configured for thin facet outlines.
    """
    # "ChatGPT 4o-mini" assisted with generating this docstring.
    return RenderControlFacetEnsemble(
        draw_normal_vector=False,
        default_style=rcf.outline_thin(color=color, linewidth=linewidth),
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        draw_centroid=False,
        **kwargs
    )


def facet_ensemble_outline(color='k', normal_vector_length=4.0, **kwargs):
    """
    Create a render control ensemble with outlines and normal vectors for facets.

    This function returns a `RenderControlFacetEnsemble` instance configured to draw outlines
    and normal vectors for facets.

    Parameters
    ----------
    color : str, optional
        Color for the outlines and normal vectors. By default, 'k' (black).
    normal_vector_length : float, optional
        Length of the normal vectors. By default, 4.0.
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFacetEnsemble`.

    Returns
    -------
    RenderControlFacetEnsemble
        An instance of `RenderControlFacetEnsemble` configured for outlines and normal vectors.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacetEnsemble(
        draw_normal_vector=True,
        normal_vector_length=normal_vector_length,
        default_style=rcf.outline(color=color),
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        draw_centroid=True,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_facets=False,
        **kwargs
    )


def only_outline(color='k'):
    """
    Create a render control ensemble that displays only the outline of facets.

    This function returns a `RenderControlFacetEnsemble` instance configured to draw only
    the outlines of the facets without any other visual elements.

    Parameters
    ----------
    color : str, optional
        Color for the outlines. By default, 'k' (black).

    Returns
    -------
    RenderControlFacetEnsemble
        An instance of `RenderControlFacetEnsemble` configured to display only the outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacetEnsemble(draw_outline=True, outline_style=rcps.outline(color=color), draw_facets=False)


def normal_only(color='k', normal_vector_length=4.0, **kwargs):
    """
    Create a render control ensemble that displays only the normal vectors of facets.

    This function returns a `RenderControlFacetEnsemble` instance configured to draw only
    the normal vectors without any outlines or centroids.

    Parameters
    ----------
    color : str, optional
        Color for the normal vectors. By default, 'k' (black).
    normal_vector_length : float, optional
        Length of the normal vectors. By default, 4.0.
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the `RenderControlFacetEnsemble`.

    Returns
    -------
    RenderControlFacetEnsemble
        An instance of `RenderControlFacetEnsemble` configured to display only the normal vectors.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlFacetEnsemble(
        draw_normal_vector=True,
        default_style=rcf.outline(color=color),
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        normal_vector_length=normal_vector_length,
        draw_centroid=True,
        draw_facets=False,
        **kwargs
    )
