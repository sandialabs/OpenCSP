"""
Figure Management

Manages figure creation and display.

Features include:
  Tiling figures across a window, so they do not appear one on top of the other.
  Providing Figure names for window names and later fetching
  Control of figure creation with a figure_control object.



"""

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.tool.log_tools as lt


# Global index used for tile control.
figure_tile_idx = 0
show_figures = True


def reset_figure_tiles():
    global figure_tile_idx
    figure_tile_idx = 0


def do_show_figures(flag: bool = True):
    global show_figures
    show_figures = flag


# Global variables used for documentation.
figure_num = 0
fig_record_list: list[RenderControlFigureRecord] = []

figure_tile_idx = 0  # Used for tile control.


def reset_figure_management():
    reset_figure_tiles()
    global figure_num
    figure_num = 0
    global fig_record_list
    fig_record_list = []


def mpl_pyplot_figure(*vargs, **kwargs):
    """Initializes and returns a matplotlib.pyplot.figure() instance.

    If creating the figure fails, try again (up to two more times).

    Sometimes initializing a matplotlib figure fails. But if you try to
    initialize the figure() class again, it seems to always succeed. The
    measured failure rate for first time initialization is between 7% and 15%.
    When it fails, it is often with an error about not being able to find a
    file. Something like::

        _tkinter.TclError: Can't find a usable init.tcl in the following directories
    """
    try:
        # try to create a figure
        return plt.figure(*vargs, **kwargs)
    except Exception:
        try:
            lt.warn("Failed to create a matplotlib.pyplot.figure instance. Trying again (2nd attempt).")
            # first attempt failed, try again
            return plt.figure(*vargs, **kwargs)
        except Exception:
            # second attempt failed, give the system a second to stabalize and
            # try a third time
            lt.warn("Failed to create a matplotlib.pyplot.figure instance. Trying again (3rd attempt).")
            import time

            time.sleep(1)
            return plt.figure(*vargs, **kwargs)


def _tile_figure(
    name=None,  # Handle and title of figure window.
    tile_array: tuple[int, int] = (3, 2),  # (n_y, n_x) ~ (columns, rows)
    tile_square: bool = False,  # Force figure to have equal x:y aspect ratio.
    screen_size: tuple[float, float] = (19.0, 10.0),  # Screen (width, height) in "inches."  Set by experimentation.
    header_height: float = 0.8,  # Height of window title and display tool header, in "inches."
    screen_pixels: tuple[float, float] = (1920, 1080),  # (n_x, n_y).  Subtract task bar pixels from y.
    task_bar_pixels: float = 40,
):  # Height of task bar in pixels.
    """
    Places figures in a regular tile pattern on screen.
    """
    # Compute figure size.
    n_x = tile_array[0]
    n_y = tile_array[1]
    n = n_x * n_y
    size_x = screen_size[0] / n_x  # inch
    size_y = screen_size[1] / n_y  # inch
    if tile_square:
        if size_x > size_y:
            size_x = size_y
        elif size_x < size_y:
            size_y = size_x
        else:
            pass
    plot_size_y = size_y - header_height
    # Compute figure position.
    global figure_tile_idx
    figure_pos_idx = figure_tile_idx % n
    figure_tile_idx += 1
    y = figure_pos_idx / n_x
    y_idx = int(y)
    x = (y - y_idx) * n_x
    x_idx = round(x)
    y_idx = int(y)  # Don't use round()
    size_x_pixels = int(screen_pixels[0] / n_x)
    size_y_pixels = int((screen_pixels[1] - task_bar_pixels) / n_y)
    ul_x = size_x_pixels * x_idx
    ul_y = size_y_pixels * y_idx

    # Create figure.
    # fig = figure(constrained_layout=True).subplots(5, 5)
    fig = mpl_pyplot_figure(name, figsize=(size_x, plot_size_y))
    # Turn off the axis around the plot drawing area.  This leads to confusing duplicate, mismatched,
    # axis information.  Why this suddenly appeared is beyond me. - RCB
    # The command below does not suppress the actual plot axes.
    plt.axis('off')

    # Set the x,y offset of the figure
    # matplotlib.use('tkagg') # BGB I don't think we should depend on this
    mnger = plt.get_current_fig_manager()
    if hasattr(mnger, 'window'):
        if hasattr(mnger.window, 'wm_geometry'):
            mnger.window.wm_geometry(f"+{ul_x}+{ul_y}")
        elif hasattr(mnger.window, 'geometry'):
            curr_dims = mnger.window.geometry().getRect()  # x, y, w, h
            mnger.window.setGeometry(curr_dims[0], curr_dims[1], ul_x, ul_y)

    # matplotlib.use("wx")
    # fig.canvas.manager.window.move(ul_x, ul_y) # Used to make the plots display in a productive setup

    # TODO TJL: find a native matplolib method to make the plots display properly
    return fig


def _display_image(
    image: np.ndarray | str,
    name: str = None,  # Figure handle and title of figure window.
    title: str = None,  # Title of plot. Used for name if name is None.
    figsize: tuple[float, float] = (6.4, 4.8),  # inch.
    tile: bool = True,  # True => Lay out figures in grid.  False => Place at upper_left or default screen center.
    tile_array: tuple[int, int] = (3, 2),  # (n_x, n_y)
    upper_left_xy: tuple[float, float] = None,  # pixel.  (0,0) --> Upper left corner of screen.
    cmap=None,  # Color scheme to use.
    block=False,
) -> plt.Figure:
    """If all you want to do is draw an image to the screen, then this is the method for you."""
    # set up the figure
    axis_control = rca.image(grid=False)
    figure_control = RenderControlFigure(tile=tile, tile_array=tile_array, figsize=figsize, upper_left_xy=upper_left_xy)
    view_spec_2d = vs.view_spec_im()
    fig_record = setup_figure(
        figure_control,
        axis_control,
        view_spec_2d,
        name=name,
        title=title,
        code_tag=f"{__file__}.display_image",
        equal=False,
    )

    # draw the image
    view = fig_record.view
    view.imshow(image, cmap=cmap)
    if show_figures:
        view.show(block=block)


def _setup_figure(
    figure_control: RenderControlFigure,
    axis_control: rca.RenderControlAxis = None,
    equal: bool = True,
    number_in_name: bool = True,  # Include figure index in the figure name.
    input_prefix: str = None,  # Prefix to include at beginning of figure name, before number is added.
    name: str = None,  # Figure handle and title of figure window.  If none, use title.
    title: str = None,  # Title of plot (before number is added, if applicable).
    caption: str = None,  # Caption providing concise descrption plot.  Optional details may be added via comments.
    comments: list[str] = None,  # List of strings including comments to associate with the figure.
    # String of form "code_file.function_name()" showing where to look in code for call that generated this figure.
    code_tag: str = None,
) -> RenderControlFigureRecord:
    """Common figure setup for 2D and 3D data."""
    # defaults
    axis_control = axis_control if axis_control != None else rca.RenderControlAxis()
    comments = comments if comments != None else []

    # Figure number.
    global figure_num
    # Starts at zero, so increment now.
    figure_num += 1

    # Figure name.
    if name == None:
        name = title
    if input_prefix == None:
        input_prefix = ""
    prefix = input_prefix
    if number_in_name:
        # Add a figure number, so that figure name is a unique key even if the input figure name is re-used.
        # May be suppressed by an input parameter.
        prefix += '{0:03d}'.format(figure_num)
    prefix = (prefix + '_') if (prefix != "") else ""
    name = prefix + name
    figure_control.figure_names.append(name)

    # Create figure.
    if figure_control.tile:
        fig = _tile_figure(name, tile_array=figure_control.tile_array, tile_square=figure_control.tile_square)
    else:
        fig = mpl_pyplot_figure(name, figsize=figure_control.figsize)
        if figure_control.upper_left_xy:
            upper_left_xy = figure_control.upper_left_xy
            x = upper_left_xy[0]
            y = upper_left_xy[1]
            window = fig.canvas.manager.window
            if hasattr(window, "move"):
                window.move(x, y)  # qt
            else:
                window.geometry(f"+{x}+{y}")  # tkinter
        if figure_control.maximize:
            window = fig.canvas.manager.window
            if hasattr(window, "showMaximized"):
                window.showMaximized()  # qt
            else:
                window.state("zoomed")  # tkinter
        # Copying this command, as from Randy, which suppresses duplicate axes in tile_figure(). ~ BGB
        plt.axis('off')

    # Add title and grid
    if title and len(title) != 0:
        plt.title(title)
    if axis_control.grid:
        plt.grid()

    # Update figure collection variables.
    fig_record = rcfr.RenderControlFigureRecord(name, title, caption, figure_num, fig, axis_control)
    global fig_record_list
    fig_record_list.append(fig_record)

    # Basic figure properties
    fig_record.equal = equal

    # Initialize comments.
    # Standard comments.
    fig_record.add_metadata_line('Figure number: ' + str(fig_record.figure_num))
    fig_record.add_metadata_line('Name: ' + str(name))
    fig_record.add_metadata_line('Title: ' + str(title))
    fig_record.add_metadata_line('Code tag: ' + str(code_tag))
    # Input comments.
    for comment_line in comments:
        fig_record.add_comment_line(comment_line)

    return fig_record


def setup_figure(
    figure_control: RenderControlFigure,
    axis_control=None,
    view_spec=None,
    equal: bool = True,
    number_in_name: bool = True,
    input_prefix: str = None,
    name: str = None,
    title: str = None,
    caption: str = None,
    comments: list[str] = None,
    code_tag: str = None,
) -> RenderControlFigureRecord:
    """Create and setup a new RenderControlFigureRecord for rendering on a 2D graph.

    Example::

        # draw an image using figure_management
        img = cv.imread(os.path.join(img_dir, img_name))
        axis_control = rca.image(grid=False)
        figure_control = rcfg.RenderControlFigure()
        view_spec_2d = vs.view_spec_im()
        fig_record = fm.setup_figure(figure_control, axis_control, view_spec_2d, title=img_name, code_tag=f"{__file__}", equal=False)
        fig_record.view.imshow(img)
        fig_record.view.show(block=True)
        # ...
        fig_record.close()

    Note that even through the returned figure_record will ensure that the associated plot is closed when the associated
    view object is destructed, it is almost always better to close the figure as soon as it's not needed any more via
    the figure_record.close() method.

    Arguments:
    ----------
        - view_spec (view_spec dict): Defines how to draw the plot (which axis is horizontal and vertical).  Defaults to view_spec_xy.
        - See setup_figure_for_3d_data() for a description of the other arguments.

    See Also:
    ---------
        A similar function for setting up figures for 3D rendering is `setup_figure_for_3d_data`
    """
    # defaults
    view_spec = view_spec if view_spec != None else vs.view_spec_xy()

    # Setup the figure.
    fig_record = _setup_figure(
        figure_control, axis_control, equal, number_in_name, input_prefix, name, title, caption, comments, code_tag
    )
    axis_control = fig_record.axis_control

    # Setup the axes.
    ax = plt.axes()
    if view_spec['type'] == 'xy':
        ax.set_xlabel(axis_control.x_label)
        ax.set_ylabel(axis_control.y_label)
    elif view_spec['type'] == 'xz':
        ax.set_xlabel(axis_control.x_label)
        ax.set_ylabel(axis_control.z_label)
    elif view_spec['type'] == 'yz':
        ax.set_xlabel(axis_control.y_label)
        ax.set_ylabel(axis_control.z_label)
    elif view_spec['type'] == 'vplane':
        ax.set_xlabel(axis_control.p_label)
        ax.set_ylabel(axis_control.q_label)
    elif view_spec['type'] == 'camera':
        ax.set_xlabel(axis_control.p_label)
        ax.set_ylabel(axis_control.q_label)
    elif view_spec['type'] == 'image':
        ax.set_xlabel(axis_control.p_label)
        ax.set_ylabel(axis_control.q_label)
    else:
        lt.error_and_raise(
            RuntimeError,
            "ERROR: In setup_figure_for_3d_data(), unrecognized view_spec['type'] = '"
            + str(view_spec['type'])
            + "' encountered.",
        )

    # Create the view object.
    view = v3d.View3d(fig_record.figure, ax, view_spec=view_spec, equal=equal, parent=fig_record)
    # Add view to log data.
    fig_record.axis = ax
    fig_record.view = view
    fig_record.add_metadata_line('View spec: ' + str(view_spec['type']))

    return fig_record


def setup_figure_for_3d_data(
    figure_control: RenderControlFigure,
    axis_control=None,
    view_spec=None,
    equal: bool = True,
    number_in_name: bool = True,
    input_prefix: str = None,
    name: str = None,
    title: str = None,
    caption: str = None,
    comments: list[str] = None,
    code_tag: str = None,
) -> RenderControlFigureRecord:
    """Create and setup a new RenderControlFigureRecord for rendering on a 3D graph.

    Note that even through the returned figure_record will ensure that the associated plot is closed when the associated
    view object is destructed, it is almost always better to close the figure as soon as it's not needed any more via
    the figure_record.close() method.

    Arguments:
    ----------
        - figure_control (RenderControlFigure): Controls how multiple figures get plotted at the same time in multiple windows.
        - axis_control (RenderControlAxis): Defines the axis labels and whether to draw the grid.  Defaults to RenderControlAxis().
        - view_spec (view_spec dict): Defines how to draw the plot (which axis is horizontal and vertical, or XYZ for 3D data).  Defaults to view_spec_xy.
        - equal (bool): Equal axis ticks.  Passed to Lib.Render.View3d.View3D()
        - number_in_name (bool): Each figure record has a unique identifier.  If True, add this identifier to the plot name, after the input_prefix.  Defaults to True.
        - input_prefix (str): Prefix to include at beginning of figure name, before number is added.
        - name (str): The name for the figure.  Used as the file name when saved.  Defaults to title.
        - title (str): The title for the plot.  Used as the name if the name is None.
        - caption (str): Caption providing concise description plot.  Optional details may be added via comments.
        - comments (list[str]): List of strings including comments to associate with the figure.
        - code_tag (str): String of form "code_file.function_name()" showing where to look in code for call that generated this figure.

    See Also:
    ---------
        A similar function for setting up figures for 2D rendering is `setup_figure`"""
    # defaults
    view_spec = view_spec if view_spec != None else vs.view_spec_3d()

    # Setup the figure.
    fig_record = _setup_figure(
        figure_control, axis_control, equal, number_in_name, input_prefix, name, title, caption, comments, code_tag
    )
    axis_control = fig_record.axis_control

    # Setup the axes.
    if view_spec['type'] == '3d':
        ax = plt.axes(projection='3d')
        ax.set_xlabel(axis_control.x_label)
        ax.set_ylabel(axis_control.y_label)
        ax.set_zlabel(axis_control.z_label)
    elif view_spec['type'] == 'xy':
        ax = plt.axes()
        ax.set_xlabel(axis_control.x_label)
        ax.set_ylabel(axis_control.y_label)
    elif view_spec['type'] == 'xz':
        ax = plt.axes()
        ax.set_xlabel(axis_control.x_label)
        ax.set_ylabel(axis_control.z_label)
    elif view_spec['type'] == 'yz':
        ax = plt.axes()
        ax.set_xlabel(axis_control.y_label)
        ax.set_ylabel(axis_control.z_label)
    elif view_spec['type'] == 'vplane':
        ax = plt.axes()
        ax.set_xlabel(axis_control.p_label)
        ax.set_ylabel(axis_control.q_label)
    elif view_spec['type'] == 'camera':
        ax = plt.axes()
        ax.set_xlabel(axis_control.p_label)
        ax.set_ylabel(axis_control.q_label)
    elif view_spec['type'] == 'image':
        ax = plt.axes()
        ax.set_xlabel(axis_control.p_label)
        ax.set_ylabel(axis_control.q_label)
    else:
        lt.error_and_raise(
            RuntimeError,
            "ERROR: In setup_figure_for_3d_data(), unrecognized view_spec['type'] = '"
            + str(view_spec['type'])
            + "' encountered.",
        )

    # Create the view object.
    view = v3d.View3d(fig_record.figure, ax, view_spec=view_spec, equal=equal, parent=fig_record)
    # Add view to log data.
    fig_record.axis = ax
    fig_record.view = view
    fig_record.add_metadata_line('View spec: ' + str(view_spec['type']))

    # Return.
    return fig_record


def _display_plot(
    x: float,
    # x_labels,
    y: float,
    label=None,  # Legend label for this data series.
    name: str = None,  # Figure handle and title of figure window.
    title: str = None,  # Title of plot.
    figsize: tuple[float, float] = (6.4, 4.8),  # inch.
    tile: bool = True,  # True => Lay out figures in grid.  False => Place at upper_left or default screen center.
    tile_array: tuple[float, float] = (3, 2),  # (n_x, n_y)
    upper_left_xy: tuple[float, float] = None,  # pixel.  (0,0) --> Upper left corner of screen.
    legend: bool = True,  # Whether to draw a legend.
    color='k',
    linewidth: float = 1,
    marker='.',
    markersize: float = 2,
) -> plt.Figure:
    if tile:
        fig = _tile_figure(name, tile_array=tile_array)
    else:
        fig = mpl_pyplot_figure(name, figsize=figsize)
        if upper_left_xy:
            x = upper_left_xy[0]
            y = upper_left_xy[1]
            fig.canvas.manager.window.move(x, y)
    if title and len(title) != 0:
        plt.title(title)
    (line,) = plt.plot(x, y, color=color, linewidth=linewidth, marker=marker, markersize=markersize)
    if label:
        line.set_label(label)
    # # Rotate x-axis tick marks.
    # plt.xticks(rotation=90, ha='right')
    # plt.xticks(x, x_labels)
    if show_figures:
        plt.show(block=False)
    return fig


def _display_bar(
    x_labels,
    y_values,
    name: str = None,  # Figure handle and title of figure window.
    title: str = None,  # Title of plot.
    figsize: tuple[float, float] = (6.4, 4.8),  # inch.
    tile: bool = True,  # True => Lay out figures in grid.  False => Place at upper_left or default screen center.
    tile_array: tuple[int, int] = (3, 2),  # (n_x, n_y)
    upper_left_xy=None,  # pixel.  (0,0) --> Upper left corner of screen.
) -> plt.Figure:
    if tile:
        fig = _tile_figure(name, tile_array=tile_array)
    else:
        fig = mpl_pyplot_figure(name, figsize=figsize)
        if upper_left_xy:
            x = upper_left_xy[0]
            y = upper_left_xy[1]
            fig.canvas.manager.window.move(x, y)
    if title and len(title) != 0:
        plt.title(title)
    # See "Bar Charts in Matplotlib," by Ben Klein.  https://benalexkeen.com/bar-charts-in-matplotlib/
    x_pos = [i for i, _ in enumerate(x_labels)]
    plt.bar(x_pos, y_values)
    plt.xticks(x_pos, x_labels)
    if show_figures:
        plt.show(block=False)
    return fig


def print_figure_summary() -> None:
    global fig_record_list
    for fig_record in fig_record_list:
        print()
        fig_record.print_comments()


def save_all_figures(output_path: str, format: str = None):
    """Saves all figures opened with setup_figure (since reset_figure_management) to the given directory.

    Args:
    -----
        - output_path (str): The directory to save figures to.
        - format (str): The file format for figures. None for RenderControlFigureRecord.save default. Defaults to None.

    Returns:
    --------
        - figs: list[str] The list of image files
        - txts: list[str] The list of image descriptor text files
        - failed: list[RenderControlFigureRecord] The list of figure records that failed to save
    """
    global fig_record_list  # TODO: convert from global to class member or save_all_figures parameter
    figs: list[str] = []
    txts: list[str] = []
    failed: list[RenderControlFigureRecord] = []

    try:
        for fig_record in fig_record_list:
            fig_file, txt_file = fig_record.save(output_path, format=format, close_after_save=False)
            figs.append(fig_file)
            txts.append(txt_file)
    except Exception as ex:
        err_msg = f"RuntimeError: figure_management.save_all_figures: failed to save figure {fig_record.figure_num} \"{fig_record.name}\""
        lt.error(err_msg)
        failed.append(fig_record)
        raise (ex)

    return figs, txts, failed


def formatted_fig_display(block: bool = False) -> None:
    plt.show(block=block)
