"""Parametric mirror representing a single reflective surface defined
by an algebraic function.
"""

import numpy as np

import opencsp.common.lib.tool.log_tools as lt

from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
import opencsp.common.lib.render.figure_management as fm
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ, identity_transform
from opencsp.common.lib.render.View3d import View3d
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlMirrorEmbedded as rcme
import opencsp.common.lib.render_control.RenderControlMirrorProjected as rcmp
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure


def embedding_mirror_half_width(
    parametric_mirror: MirrorParametric, margin: float = 0.1, round_to: float = 0.5
) -> float:
    """
    Computes the min/max extent of a square centered on the origin that encloses
    the input mirror.  This includes an input margin, rounded up to the nearest
    increment indicated by the round_to parameter.

    Parameters
    ----------
    parametric_mirror : MirrorParametric
        Mirror to derive the embedding surface from.
    margin : float, optional
        Lower bound of the margin to leave all around the mirror.
        Default 0.1.
    round_to : float, optional
        Increment to round the margin, so that it falls on an even plot axis tick mark.
        Default 0.5.

    Returns
    -------
    float
        Half-width of the square enclosing the mirror, including the margin.
        Example:
        Mirror vertices:
            [1.0, 2.0]
            [1.5, 3.0]
            [3.2, 2.9]
            [4.0, 2.3]

        Mirror bounding box:
            [x_min, x_max, y_min, y_max] = [1.0, 4.0, 2.0, 3.0].
        Bounding square centered at the origin:
            [x_min, x_max, y_min, y_max] = [-4.0, 4.0, -3.0, 3.0].
        Adding margin=0.1:
            [x_min, x_max, y_min, y_max] = [-4.1, 4.1, -3.1, 3.1].
        Rounding to nearest enclosing 0.5:
            [x_min, x_max, y_min, y_max] = [-4.5, 4.5, -3.5, 3.5].
        Mirror extent:
            4.5
    """
    # Verify this is a parametric mirror.
    if not isinstance(parametric_mirror, MirrorParametric):
        lt.error(
            f"In embedding_mirror_half_width(), non-parametric mirror encountered.  Mirror type = {type(parametric_mirror).__name__}"
        )

    # Compute extent of embedding surface.
    mirror_x_min, mirror_x_max, mirror_y_min, mirror_y_max = parametric_mirror.axis_aligned_bounding_box
    mirror_half_width = max(abs(mirror_x_min), abs(mirror_x_max), abs(mirror_y_min), abs(mirror_y_max))
    return np.ceil((mirror_half_width + margin) / round_to) * round_to


def construct_embedding_mirror(
    parametric_mirror: MirrorParametric, margin: float = 0.1, round_to: float = 0.5
) -> MirrorParametricRectangular:
    """
    Constructs a mirror with a square boundary with the same embedding surface
    as the input parametric mirror.  Useful for illustrating the geometric
    construction of a specific mirror.

    Parameters
    ----------
    parametric_mirror : MirrorParametric
        Mirror to derive the embedding surface from.
    margin : float, optional
        Lower bound of the margin to leave all around the mirror.
        Default 0.1.
    round_to : float, optional
        Increment to round the margin, so that it falls on an even plot axis tick mark.
        Default 0.5.

    Returns
    -------
    MirrorParametric
        A mirror object with the same embedding surface as the input mirror,
        but with a square boundary that is larger than the input mirror by
        the given margin plus potentially more to meet an even plot tick mark.
    """
    # Verify this is a parametric mirror.
    if not isinstance(parametric_mirror, MirrorParametric):
        lt.error(
            f"In embedding_mirror_half_width(), non-parametric mirror encountered.  Mirror type = {type(parametric_mirror).__name__}"
        )

    # Compute extent of embedding surface.
    embedding_extent = embedding_mirror_half_width(parametric_mirror, margin, round_to)
    # Construct embedding mirror.
    len_xy = 2.0 * embedding_extent
    rectangle_xy = (len_xy, len_xy)
    return MirrorParametricRectangular(parametric_mirror.surface_function(), rectangle_xy)


def draw_mirror_and_embedding_mirror(
    input_mirror: MirrorParametric,
    view: View3d,
    mirror_style: rcm.RenderControlMirror = None,
    draw_projection: bool = True,
    projected_style: rcmp.RenderControlMirrorProjected = None,
    embedding_style: rcme.RenderControlMirrorEmbedded = None,
    transform: TransformXYZ | None = None,
) -> None:
    """
    Draws a mirror in the given view, and also a slightly larger embedding mirror.
    There are options to draw slices and projections onto the (x,y) plane.

    Parameters
    ----------
    input_mirror : MirrorParametric
        Mirror to draw.
    view : View3d
        Vie to display the mirorr.
    mirror_style : RenderControlMirror, optional
        Attributes for drawing mirror features.  Default None.
    draw_projection : bool, optional
        Whether to draw the projection of the mirror boundary onto the (x,y) plane.
        Default True.
    projected_style : RenderControlMirrorProjected, optional
        Attributes for drawing the projected mirror boundary.  Default None.
    embedding_style : RenderControlMirrorEmbedded, optional
        Attributes for drawing the embedding mirror surface.  Default None.
    transform : TransformXYZ | None, optional
        Transform to position the miror in space.  Default None.
    """
    # Construct a second mirror, with larger boundary, enabling us to show the embedding surface in a plot.
    embedding_mirror = construct_embedding_mirror(
        input_mirror, margin=embedding_style.margin, round_to=embedding_style.round_to
    )

    # Set view axes to match the extent of the embedding mirror, which is square and larger than the input mirror.
    # Also set equal axes to prevent z exaggeration.
    embedding_x_min, embedding_x_max, embedding_y_min, embedding_y_max = embedding_mirror.axis_aligned_bounding_box
    limit_xy = max(abs(embedding_x_min), abs(embedding_x_max), abs(embedding_y_min), abs(embedding_y_max))
    view.show(x_limits=[-limit_xy, limit_xy], y_limits=[-limit_xy, limit_xy], z_limits=[0, 2 * limit_xy])

    # Draw second mirror showing embedding surface.
    embedding_mirror.draw(view=view, mirror_style=mirror_style, draw_projection=False, transform=transform)

    # Draw slices of embedding surface.
    embedding_mirror.draw_surface_mesh(
        view,
        n_slices_x=embedding_style.n_slices_x,
        project_slices_x=embedding_style.project_slices_x,
        draw_special_origin_slice_x=embedding_style.draw_special_origin_slice_x,
        project_origin_slice_x=embedding_style.project_origin_slice_x,
        n_slices_y=embedding_style.n_slices_y,
        project_slices_y=embedding_style.project_slices_y,
        draw_special_origin_slice_y=embedding_style.draw_special_origin_slice_y,
        project_origin_slice_y=embedding_style.project_origin_slice_y,
        slice_style=embedding_style.slice_style,
        origin_slice_style=embedding_style.origin_slice_style,
        transform=transform,
    )

    # Draw primary mirror.
    input_mirror.draw(
        view=view,
        mirror_style=mirror_style,
        draw_projection=draw_projection,
        projected_style=projected_style,
        transform=transform,
    )


def setup_draw_and_save_mirror_and_embedding_mirror(
    # Required parameters.
    figure_control: RenderControlFigure,
    parametric_mirror: MirrorParametric,
    title: str,
    output_dir: str,
    # Options.
    axis_control: rca.RenderControlAxis = None,
    view_spec: dict = None,
    number_in_name: bool = False,
    input_prefix: str = None,
    caption: str = None,
    comments: list[str] = [],
    code_tag: str = None,
    mirror_style: rcm.RenderControlMirror = rcm.RenderControlMirror(),
    draw_projection: bool = True,
    projected_style: rcmp.RenderControlMirrorProjected = rcmp.mirror_boundary(),
    embedding_style: rcme.RenderControlMirrorEmbedded = rcme.standard_embedding_mirror(),
    transform: TransformXYZ | None = None,
    # Save figure parameters.
    dpi: int = 200,
    output_format: str = 'png',
    close_after_save: bool = True,
    include_view_suffix: bool = True,
    include_limit_suffix: bool = False,
):
    """
    This function wraps function draw_mirror_and_embedding_mirror() above,
    creating and saving a figure for testing and other use.
    """
    # Ensure all parameters are set.
    if axis_control is None:
        axis_control = rca.meters(grid=False)  # Drawing axis grid and surface grid is confusing.
    if view_spec is None:
        view_spec = vs.view_spec_3d()
    if transform is None:
        transform = identity_transform()

    # Setup figure.
    fig_record = fm.setup_figure_for_3d_data(
        figure_control=figure_control,
        axis_control=axis_control,
        view_spec=view_spec,
        number_in_name=number_in_name,
        input_prefix=input_prefix,
        title=title,
        caption=caption,
        comments=comments,
        code_tag=code_tag,
    )

    draw_mirror_and_embedding_mirror(
        parametric_mirror,
        view=fig_record.view,
        mirror_style=mirror_style,
        draw_projection=draw_projection,
        projected_style=projected_style,
        embedding_style=embedding_style,
        transform=transform,
    )

    # Save.
    fig_record.save(
        output_dir=output_dir,
        dpi=dpi,
        format=output_format,
        close_after_save=close_after_save,
        include_view_suffix=include_view_suffix,
        include_limit_suffix=include_limit_suffix,
    )
