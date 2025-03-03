"""
Plotting Pandas Objects
"""

import matplotlib.pyplot as plt

import opencsp.common.lib.render.figure_management as fm


def dataframe_plot(
    figure_control,
    df,  # Dataframe with columns to plot.
    title,  # Plot title.
    x_column,  # String that is column heading to use for horizontal axis.
    y_column_label_styles,  # List of data curve specifications.
    # Form: [ [col_heading_1, legend_label_or_None_1, point_seq_render_control_1], ...]
    x_axis_label,  # String or None.
    y_axis_label,  # String or None.
    x_axis_grid=False,  # Draw vertical grid lines.
    y_axis_grid=False,  # Draw horizontal grid lines.
    legend=True,
):  # Whether to draw the plot legend.
    """
    Plots data from a Pandas DataFrame.

    This function creates a plot using data from the specified DataFrame, allowing for multiple
    curves to be plotted based on the provided column specifications. It sets up the figure,
    plots the data, and displays the plot.

    Parameters
    ----------
    figure_control : object
        Control object for managing the figure.
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    title : str
        The title of the plot.
    x_column : str
        The name of the column to use for the horizontal axis.
    y_column_label_styles : list[list]
        A list of specifications for the data curves to plot. Each specification should be a list
        containing the column heading, legend label (or None), and point sequence render control.
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

    Raises
    ------
    ValueError
        If the specified x_column or any y_column in y_column_label_styles does not exist in the DataFrame.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    figure_record = fm.setup_figure(figure_control, title=title)
    for y_column_label_style in y_column_label_styles:
        y_column = y_column_label_style[0]
        label = y_column_label_style[1]
        style = y_column_label_style[2]
        plt.plot(
            df[x_column],
            df[y_column],
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
