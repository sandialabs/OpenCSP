"""


"""

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
        style = self.default_style
        if facet_name in self.special_styles:
            style = self.special_styles[facet_name]
        return style

    # @strict_types
    def add_special_style(self, facet_name: str | list, facet_style: rcf.RenderControlFacet):
        if type(facet_name) is list and isinstance(facet_name[0], str):
            for name in facet_name:
                self.add_special_style(name, facet_style)
            return
        if facet_name is None:
            warn("Facets with name=None should not have special styles in RenderControlFacetEnsemble.")
        self.special_styles[facet_name] = facet_style


# GENERATORS


def normal_facet_outlines(color='k', **kwargs):
    return RenderControlFacetEnsemble(
        draw_normal_vector=True,
        default_style=rcf.outline(color=color),
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        draw_centroid=True,
        **kwargs
    )


def facet_outlines(color='k', **kwargs):
    return RenderControlFacetEnsemble(
        draw_normal_vector=False,
        default_style=rcf.outline(color=color),
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        draw_centroid=False,
        **kwargs
    )


def facet_outlines_thin(color='k', linewidth=0.25, **kwargs):
    return RenderControlFacetEnsemble(
        draw_normal_vector=False,
        default_style=rcf.outline_thin(color=color, linewidth=linewidth),
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        draw_centroid=False,
        **kwargs
    )


def facet_ensemble_outline(color='k', normal_vector_length=4.0, **kwargs):
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
    return RenderControlFacetEnsemble(draw_outline=True, outline_style=rcps.outline(color=color), draw_facets=False)


def normal_only(color='k', normal_vector_length=4.0, **kwargs):
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
