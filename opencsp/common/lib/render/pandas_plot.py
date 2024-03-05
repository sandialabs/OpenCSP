"""

Plotting Pandas Objects



"""

import matplotlib.pyplot as plt

import opencsp.common.lib.render.figure_management as fm


def dataframe_plot(figure_control,
                   df,                     # Dataframe with columns to plot.
                   title,                  # Plot title.
                   x_column,               # String that is column heading to use for horizontal axis.
                   y_column_label_styles,  # List of data curve specifications.  
                                           # Form: [ [col_heading_1, legend_label_or_None_1, point_seq_render_control_1], ...]
                   x_axis_label,           # String or None.
                   y_axis_label,           # String or None.
                   x_axis_grid=False,      # Draw vertical grid lines.
                   y_axis_grid=False,      # Draw horizontal grid lines.
                   legend=True):           # Whether to draw the plot legend.
    figure_record = fm.setup_figure(figure_control, title=title)
    for y_column_label_style in y_column_label_styles:
        y_column = y_column_label_style[0]
        label    = y_column_label_style[1]
        style    = y_column_label_style[2]
        plt.plot(df[x_column],
                 df[y_column],
                 linestyle=style.linestyle,
                 color=style.color,
                 linewidth=style.linewidth,
                 marker=style.marker,
                 markersize=style.markersize,
                 label=label)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if x_axis_grid:
        plt.grid(axis='x')
    if y_axis_grid:
        plt.grid(axis='y')
    if legend:
        plt.legend()
    plt.show()
    return figure_record
