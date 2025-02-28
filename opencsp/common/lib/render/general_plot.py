"""

General Plotting Support

See also image_plot.py, pandas_plot.py, View3d.py, PlotAnnotation.py.



"""

import matplotlib.pyplot as plt

import opencsp.common.lib.render.figure_management as fm


def plot_xy_list(
    figure_control,
    xy_list,  # Data to plot.
    title,  # Plot title.
    style,  # A RenderControlPointSeq object.
    label,  # Legend label or None.
    x_axis_label,  # String or None.
    y_axis_label,  # String or None.
    x_axis_grid=False,  # Draw vertical grid lines.
    y_axis_grid=False,  # Draw horizontal grid lines.
    legend=True,
):  # Whether to draw the plot legend.
    """
    Plots a list of (x, y) points on a 2D graph.

    This function creates a 2D plot using the provided data points, styles, and labels.
    It sets up the figure, plots the data, and displays the plot.

    Parameters
    ----------
    figure_control : object
        Control object for managing the figure.
    xy_list : list[tuple[float, float]]
        A list of (x, y) tuples representing the data points to plot.
    title : str
        The title of the plot.
    style : RenderControlPointSeq
        An object defining the style of the plot (line style, color, etc.).
    label : str | None
        The label for the legend. If None, no label is shown.
    x_axis_label : str | None
        The label for the x-axis. If None, no label is shown.
    y_axis_label : str | None
        The label for the y-axis. If None, no label is shown.
    x_axis_grid : bool, optional
        If True, vertical grid lines are drawn on the x-axis. Defaults to False.
    y_axis_grid : bool, optional
        If True, horizontal grid lines are drawn on the y-axis. Defaults to False.
    legend : bool, optional
        If True, the plot legend is displayed. Defaults to True.

    Returns
    -------
    object
        A figure record object containing information about the created figure.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    figure_record = fm.setup_figure(figure_control, title=title)
    x_list = []
    y_list = []
    for xy in xy_list:
        x_list.append(xy[0])
        y_list.append(xy[1])
    plt.plot(
        x_list,
        y_list,
        linestyle=style.linestyle,
        color=style.color,
        linewidth=style.linewidth,
        marker=style.marker,
        markersize=style.markersize,
        label=label,
    )
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if x_axis_grid:
        plt.grid(axis="x")
    if y_axis_grid:
        plt.grid(axis="y")
    if legend:
        plt.legend()
    plt.show()
    return figure_record


def add_xy_list_to_plot(
    figure_record,  # Figure to add the curve to.
    xy_list,  # Data to plot.
    style,  # A RenderControlPointSeq object.
    label=None,
):  # Legend label or None.
    """
    Adds a list of (x, y) points to an existing plot.

    This function appends additional data points to an already created plot. It updates
    the figure with the new data and displays the updated plot.

    Parameters
    ----------
    figure_record : object
        The figure to which the new data will be added.
    xy_list : list[tuple[float, float]]
        A list of (x, y) tuples representing the data points to add to the plot.
    style : RenderControlPointSeq
        An object defining the style of the plot (line style, color, etc.).
    label : str | None, optional
        The label for the legend. If None, no label is shown. Defaults to None.

    Returns
    -------
    object
        The updated figure record object containing information about the modified figure.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    if (xy_list != None) and (len(xy_list) > 0):
        x_list = []
        y_list = []
        for xy in xy_list:
            x_list.append(xy[0])
            y_list.append(xy[1])
        plt.plot(
            x_list,
            y_list,
            linestyle=style.linestyle,
            color=style.color,
            linewidth=style.linewidth,
            marker=style.marker,
            markersize=style.markersize,
            label=label,
        )
        plt.show()
        return figure_record
