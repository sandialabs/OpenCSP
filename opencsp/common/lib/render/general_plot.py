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
