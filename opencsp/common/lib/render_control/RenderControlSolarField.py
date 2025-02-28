from warnings import warn
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
from opencsp.common.lib.tool.typing_tools import strict_types


class RenderControlSolarField:
    """
    Render control for solar fields.

    This class manages the rendering settings for solar fields, allowing customization of various
    visual elements related to heliostats and their configurations.
    """

    # "ChatGPT 4o" assisted with generating this docstring.

    def __init__(
        self,
        #  draw_outline=True, # unimplemented
        #  outline_style=rcps.outline(), # umimplemented
        draw_origin=False,
        draw_heliostats=True,
        heliostat_styles=rch.RenderControlHeliostat(),
        draw_name=False,
        name_style=rctxt.default(color="k"),
        special_styles: dict[str, rch.RenderControlHeliostat] = None,
    ):
        """
        Render control for solar fields.

        This class manages the rendering settings for solar fields, allowing customization of various
        visual elements related to heliostats and their configurations.

        Parameters
        ----------
        clear_previous : bool, optional
            Whether to remove any existing files in the designated output directory. By default, True.
        draw_origin : bool, optional
            Whether to draw the origin point of the solar field. By default, False.
        draw_heliostats : bool, optional
            Whether to draw the heliostats in the solar field. By default, True.
        heliostat_styles : RenderControlHeliostat, optional
            Styles for rendering the heliostats. By default, an instance of `RenderControlHeliostat()`.
        draw_name : bool, optional
            Whether to draw the names of the heliostats. By default, False (unimplemented).
        name_style : RenderControlText, optional
            Style for the names of the heliostats. By default, `rctxt.default(color='k')`.
        special_styles : dict[str, RenderControlHeliostat], optional
            A dictionary mapping heliostat names to their specialized rendering styles. By default, None.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        super(RenderControlSolarField, self).__init__()

        # self.draw_outline = draw_outline
        # self.outline_style = outline_style
        self.draw_origin = draw_origin
        self.draw_heliostats = draw_heliostats
        self.heliostat_styles = heliostat_styles
        self.draw_name = draw_name
        self.name_style = name_style
        if special_styles is None:
            special_styles = {}
        self.special_styles = special_styles

    # def get_heliostat_style(self, heliostat_name: str):
    #     style = self.heliostat_styles
    #     if heliostat_name in self.special_styles:
    #         style = self.special_styles[heliostat_name]
    #     return style

    # def add_special_names(self, heliostat_name: str, heliostat_style: rch.RenderControlHeliostat):
    #     if heliostat_name is None:
    #         warn("heliostats with name=None should not have special styles in RenderControlheliostatEnsemble.")
    #     self.special_styles[heliostat_name] = heliostat_style

    def get_special_style(self, heliostat_name: str):
        style = self.heliostat_styles
        if heliostat_name in self.special_styles:
            style = self.special_styles[heliostat_name]
        return style

    # @strict_types
    def add_special_names(self, heliostat_names: str | list, heliostat_style: rch.RenderControlHeliostat):
        if type(heliostat_names) is list and isinstance(heliostat_names[0], str):
            for name in heliostat_names:
                self.add_special_names(name, heliostat_style)
            return
        if heliostat_names is None:
            warn("Heliostats with name=None should not have special styles in RenderControlSolarField.")
        self.special_styles[heliostat_names] = heliostat_style


# COMMON CASES


def default():
    """
    Create a default render control for solar fields.

    This function returns a `RenderControlSolarField` instance with default settings.

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured with default parameters.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField()


def outline(color='k'):
    """
    Create a render control that displays only the overall outline of the solar field.

    This function returns a `RenderControlSolarField` instance configured to draw only the outline.

    Parameters
    ----------
    color : str, optional
        Color for the outline. By default, 'k' (black).

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display only the outline.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Overall field outline only.
    return RenderControlSolarField(
        draw_outline=True, outline_style=rcps.outline(color=color), draw_heliostats=False, draw_name=False
    )


def heliostat_blanks(color='k'):
    """
    Create a render control that draws nothing for heliostats.

    This function returns a `RenderControlSolarField` instance configured to not draw any heliostats,
    allowing for special rendering categories to be added later.

    Parameters
    ----------
    color : str, optional
        Color for the heliostat outlines. By default, 'k' (black).

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` with no heliostats drawn.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Draw nothing.  Heliostats will be added as special rendering categories.
    return RenderControlSolarField(draw_heliostats=True, heliostat_styles=(rch.blank()), draw_name=False)


def heliostat_names(color='k'):
    """
    Create a render control that displays the names of the heliostats.

    This function returns a `RenderControlSolarField` instance configured to draw the heliostats
    with their corresponding names.

    Parameters
    ----------
    color : str, optional
        Color for the heliostat names. By default, 'k' (black).

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display heliostat names.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(draw_heliostats=True, heliostat_styles=rch.name(color=color), draw_name=False)


def heliostat_centroids(color='k'):
    """
    Create a render control that displays the centroids of the heliostats.

    This function returns a `RenderControlSolarField` instance configured to draw the centroids
    of the heliostats.

    Parameters
    ----------
    color : str, optional
        Color for the centroid markers. By default, 'k' (black).

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display heliostat centroids.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(draw_heliostats=True, heliostat_styles=(rch.centroid(color=color)), draw_name=False)


def heliostat_centroids_names(color='k'):
    """
    Create a render control that displays both the centroids and names of the heliostats.

    This function returns a `RenderControlSolarField` instance configured to draw the centroids
    and names of the heliostats.

    Parameters
    ----------
    color : str, optional
        Color for the centroid markers and names. By default, 'k' (black).

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display both centroids and names.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(
        draw_heliostats=True, heliostat_styles=(rch.centroid_name(color=color)), draw_name=False
    )


def heliostat_outlines(color='k'):
    """
    Create a render control that displays outlines of the heliostats.

    This function returns a `RenderControlSolarField` instance configured to draw the outlines
    of the heliostats.

    Parameters
    ----------
    color : str, optional
        Color for the heliostat outlines. By default, 'k' (black).

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display heliostat outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(draw_heliostats=True, heliostat_styles=(rch.outline(color=color)), draw_name=False)


def heliostat_normals_outlines(color='k'):
    """
    Create a render control that displays heliostat outlines with normal vectors.

    This function returns a `RenderControlSolarField` instance configured to draw the outlines
    of the heliostats along with their normal vectors.

    Parameters
    ----------
    color : str, optional
        Color for the heliostat outlines and normal vectors. By default, 'k' (black).

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display heliostat outlines with normals.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(
        draw_heliostats=True, heliostat_styles=(rch.normal_outline(color=color)), draw_name=False
    )


def heliostat_outlines_names(color='k'):
    """
    Create a render control that displays heliostat outlines and names.

    This function returns a `RenderControlSolarField` instance configured to draw the outlines
    of the heliostats along with their corresponding names.

    Parameters
    ----------
    color : str, optional
        Color for the heliostat outlines and names. By default, 'k' (black).

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display heliostat outlines and names.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(
        draw_heliostats=True, heliostat_styles=(rch.name_outline(color=color)), draw_name=False
    )


def heliostat_centroids_outlines_names(
    color="k", horizontalalignment="center", verticalalignment="center"  # center, right, left
):  # center, top, bottom, baseline, center_baseline
    """
    Create a render control that displays heliostat centroids, outlines, and names.

    This function returns a `RenderControlSolarField` instance configured to draw the centroids,
    outlines, and names of the heliostats.

    Parameters
    ----------
    color : str, optional
        Color for the centroid markers and outlines. By default, 'k' (black).
    horizontalalignment : str, optional
        Horizontal alignment of the name text. By default, 'center'.
    verticalalignment : str, optional
        Vertical alignment of the name text. By default, 'center'.

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display centroids, outlines, and names.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(
        draw_heliostats=True,
        heliostat_styles=(
            rch.centroid_name_outline(
                color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
            )
        ),
        draw_name=False,
    )


def heliostat_vector_field(color='k', vector_length=9):
    """
    Create a render control that displays a vector field for heliostats.

    This function returns a `RenderControlSolarField` instance configured to draw the heliostats
    with normal vectors representing their orientations.

    Parameters
    ----------
    color : str, optional
        Color for the vector field. By default, 'k' (black).
    vector_length : float, optional
        Length of the vectors. By default, 9.

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display a vector field for heliostats.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(
        draw_heliostats=True,
        heliostat_styles=(rch.normal(color=color, normal_vector_length=vector_length)),
        draw_name=False,
    )


def heliostat_vector_field_outlines(color='k', vector_length=9):
    """
    Create a render control that displays a vector field with outlines for heliostats.

    This function returns a `RenderControlSolarField` instance configured to draw the heliostats
    with outlines and normal vectors representing their orientations.

    Parameters
    ----------
    color : str, optional
        Color for the outlines and vectors. By default, 'k' (black).
    vector_length : float, optional
        Length of the vectors. By default, 9.

    Returns
    -------
    RenderControlSolarField
        An instance of `RenderControlSolarField` configured to display a vector field with outlines.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return RenderControlSolarField(
        # outline_style=rcps.outline(color=color),
        draw_heliostats=True,
        heliostat_styles=(rch.normal_outline(color=color, normal_vector_length=vector_length)),
        draw_name=False,
    )
