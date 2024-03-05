"""


"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
import os
from PIL import Image

from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.axis_3d as ax3d
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface


class View3d:
    """
    Class representing a view of 3d data.

    The view may be 2d or 3d, including various projections.

    Parameters:

        view_spec: One of 3d, xy, xz, or yz. Built with Lib.Render.view_spec.view_spec_X().

        equal: Whether to ensure axes have equal size tick spacing.
    """

    def __init__(
        self,
        figure: Figure,
        axis: Axes,
        view_spec: dict,  # 3d, xy, xz, yz
        equal=None,  # Whether to ensure axes have equal size tick spacing.
        parent=None,
    ):
        # in-situ imports to avoid import cycles
        import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr

        parent: rcfr.RenderControlFigureRecord = parent

        super(View3d, self).__init__()

        # defaults if not set
        if parent != None:
            equal = equal if equal != None else parent.equal
        equal = equal if equal != None else True

        self.figure = figure
        self.axis = axis
        self.view_spec = view_spec
        self.equal = equal
        self.parent = parent
        self.x_limits = None
        self.y_limits = None
        self.z_limits = None

    # ACCESS

    def is_3d(self) -> bool:
        return self.view_spec['type'] == '3d'

    # RENDER

    def show(
        self,
        equal=None,  # Whether to force equal axis scale; consider turning off if axis limits set.
        x_limits=None,  # Optional x-axis limits in the form [x_min,x_max] or None.
        y_limits=None,  # Optional y-axis limits in the form [y_min,y_max] or None.
        z_limits=None,  # Optional z-axis limits in the form [z_min,z_max] or None.
        grid=None,  # Whether to include a grid on the plot axes.
        crop_to_image_frame=True,  # Set axis limits to image frame boundaries (camera views only).
        draw_image_frame=True,  # Draw the image frame boundaries (camera views only).
        image_frame_style=rcps.outline(color='r'),  # Image frame boundary style.
        image_frame_legend=False,  # Include image frame as a legend entry.
        legend=False,  # Draw the plot legend.
        block=False,
    ) -> None:
        """
        Shows a plot, ensuring that equal axis is set if applicable.
        """
        # in-situ imports to avoid import cycles
        import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr

        # defaults if not set
        # equal is simple
        # grid can be inherited from the parent's axis_control, or defaults to true
        equal = equal if equal != None else self.equal
        if self.parent != None:
            parent_grid = True
            if self.parent.axis_control != None:
                parent_grid = self.parent.axis_control.grid
            grid = grid if grid != None else parent_grid
        grid = grid if grid != None else True

        # If axis limits are not provided, clear any previous limits.
        if (x_limits == None) or (y_limits == None) or (z_limits == None):
            # This solution from stack overflow: https://stackoverflow.com/questions/18795172/using-matplotlib-and-ipython-how-to-reset-x-and-y-axis-limits-to-autoscale
            ax = plt.gca()  # get the current axes
            ax.relim()  # make sure all the data fits
            ax.autoscale()  # auto-scale
            self.x_limits = None
            self.y_limits = None
            self.z_limits = None
        # Axes aspect ratio.
        if equal:
            if self.view_spec['type'] == '3d':
                ax3d.set_3d_axes_equal(
                    self.axis
                )  # , set_zmin_zero=True, box_aspect=None)
            elif (
                (self.view_spec['type'] == 'xy')
                or (self.view_spec['type'] == 'xz')
                or (self.view_spec['type'] == 'yz')
                or (self.view_spec['type'] == 'image')
                or (self.view_spec['type'] == 'vplane')
                or (self.view_spec['type'] == 'camera')
            ):
                if (x_limits != None) or (y_limits != None):
                    lt.warn(
                        'WARNING: In View3d.show(), setting equal axes while also setting axis limits can prevent axis limits from taking effect.'
                    )
                self.axis.axis('equal')
            else:
                lt.error(
                    "ERROR: In View3d.show(), unrecognized view_spec['type'] = '"
                    + str(self.view_spec['type'])
                    + "' encountered."
                )
                assert False
        # Crop.
        if (self.view_spec['type'] == 'camera') and crop_to_image_frame:
            frame_box = self.view_spec['camera'].frame_box_pq
            pq_min = frame_box[0]
            pq_max = frame_box[1]
            p_min = pq_min[0]
            p_max = pq_max[0]
            q_min = pq_min[1]
            q_max = pq_max[1]
            self.axis.set_xlim([p_min, p_max])
            self.axis.set_ylim([q_min, q_max])
        # Image frame.
        if (self.view_spec['type'] == 'camera') and draw_image_frame:
            # Fetch the image frame corners, and repeat the first corner to produce a closed contour.
            corner_pq_list = self.view_spec['camera'].image_frame_corners()
            frame_pq_list = corner_pq_list.copy()
            frame_pq_list.append(corner_pq_list[0])
            if image_frame_legend:
                image_frame_label = 'Image Frame'
            else:
                image_frame_label = None
            self.axis.plot(
                [pq[0] for pq in frame_pq_list],
                [-pq[1] for pq in frame_pq_list],  # Negate y because image is flipped.
                label=image_frame_label,
                linestyle=image_frame_style.linestyle,
                linewidth=image_frame_style.linewidth,
                color=image_frame_style.color,
                marker=image_frame_style.marker,
                markersize=image_frame_style.markersize,
                markeredgecolor=image_frame_style.markeredgecolor,
                markeredgewidth=image_frame_style.markeredgewidth,
                markerfacecolor=image_frame_style.markerfacecolor,
            )
        # Limits.
        if x_limits != None:
            self.axis.set_xlim(x_limits)
            self.x_limits = x_limits
        if y_limits != None:
            self.axis.set_ylim(y_limits)
            self.y_limits = y_limits
        if z_limits != None:
            if self.view_spec['type'] == '3d':
                self.axis.set_zlim(z_limits)
                self.z_limits = z_limits
            elif (self.view_spec['type'] == 'xz') or (self.view_spec['type'] == 'yz'):
                self.axis.set_ylim(z_limits)
                self.z_limits = z_limits
        # Grid.
        if grid:
            plt.grid()
        # Legend.
        if legend:
            self.axis.legend()
        # Draw.
        plt.show(block=block)

    # WRITE

    def show_and_save_multi_axis_limits(
        self, output_dir, output_figure_body, limits_list, grid=True
    ):
        # Draw and save.
        if limits_list != None:
            for limits in limits_list:
                if limits == None:
                    self.show_and_save(output_dir, output_figure_body, grid=grid)
                else:
                    if self.is_3d():
                        self.show_and_save(
                            output_dir,
                            output_figure_body,
                            x_limits=limits[0],
                            y_limits=limits[1],
                            z_limits=limits[2],
                            grid=grid,
                        )
                    else:
                        self.show_and_save(
                            output_dir,
                            output_figure_body,
                            x_limits=limits[0],
                            y_limits=limits[1],
                            grid=grid,
                        )

    def show_and_save(
        self,
        output_dir,  # Where to write the figure.
        output_figure_body,  # Base filename.  Directory path, suffixes, and extension will be added.
        x_limits=None,  # Optional x-axis limits in the form [x_min,x_max] or None.
        y_limits=None,  # Optional y-axis limits in the form [y_min,y_max] or None.
        z_limits=None,  # Optional z-axis limits in the form [z_min,z_max] or None.
        grid=True,  # Whether to include a grid on the plot axes.
        crop_to_image_frame=True,  # Set axis limits to image frame boundaries (camera views only).
        draw_image_frame=True,  # Draw the image frame boundaries (camera views only).
        image_frame_style=rcps.outline(color='r'),  # Image frame boundary style.
        image_frame_legend=False,  # Include image frame as a legend entry.
        legend=True,  # Whether to draw the plot legend.
        format='png',  # Format to save the figure.  See matplotlib documentation for options.
        dpi=600,
    ):  # Dots per inch to save the plot.
        """
        Constructs and displays the figure, and then saves all figures to disk.
        This routine is useful when you construct a complex figure, and then want to save versions
        with different axis limits.

        The save_all_figures() routine avoids overwriting previously-existing figures, so the work
        of saving is not replicated, even though it considers saving figures that might already have
        been saved.
        """
        # Generate the plot, setting limits if specified.
        # (This also sets equal axes, especially for 3-d plots.)
        self.show(
            x_limits=x_limits,
            y_limits=y_limits,
            z_limits=z_limits,
            grid=grid,
            crop_to_image_frame=crop_to_image_frame,
            draw_image_frame=draw_image_frame,
            image_frame_style=image_frame_style,
            image_frame_legend=image_frame_legend,
            legend=legend,
        )
        # Save all figures, including this one.
        self.save(output_dir, output_figure_body, format=format, dpi=dpi)

    def save(self, output_dir, output_figure_body, format='png', dpi=300) -> str:
        # Ensure the output destination is available.
        if not (os.path.exists(output_dir)):
            os.makedirs(output_dir)
        # Add the projection choice.
        output_figure_body += '_' + self.view_spec['type']
        # Add axis limit suffix.
        output_figure_body += self.limit_suffix()
        # Join with output directory.
        output_figure_dir_body = os.path.join(output_dir, output_figure_body)
        # Save the figure.
        output_figure_dir_body_ext = output_figure_dir_body + '.' + format
        lt.info('In View3d.save(), saving figure: ' + output_figure_dir_body_ext)
        # plt.savefig(output_figure_dir_body_ext, format=format, dpi=dpi)
        self.figure.savefig(output_figure_dir_body_ext, format=format, dpi=dpi)
        # Return the outptu file path and directory.
        return output_figure_dir_body_ext

    def limit_suffix(self):
        limit_suffix_str = ''
        if self.x_limits:
            limit_suffix_str += (
                '_' + str(self.x_limits[0]) + 'x' + str(self.x_limits[1])
            )
        if self.y_limits:
            limit_suffix_str += (
                '_' + str(self.y_limits[0]) + 'y' + str(self.y_limits[1])
            )
        if self.z_limits:
            limit_suffix_str += (
                '_' + str(self.z_limits[0]) + 'z' + str(self.z_limits[1])
            )
        return limit_suffix_str

    # Image Plotting
    def imshow(self, *args, colorbar=False, **kwargs) -> None:
        """Draw an image on a 2D plot. Requires view_spec type to be 'image'.

        This method is best for drawing an image by itself. For drawing images on
        top of other plots (example on top of 3D data) use draw_image instead."""
        if self.view_spec['type'] == 'image':
            # load the image, as necessary
            load_as_necessary = (
                lambda img: img if not isinstance(img, str) else Image.open(img)
            )
            if 'X' in kwargs:
                img = kwargs['X']
                kwargs['X'] = load_as_necessary(img)
            elif len(args) > 0:
                img = args[0]
                args = list(args)
                args[0] = load_as_necessary(img)

            im = self.axis.imshow(*args, interpolation='none', **kwargs)
            if self.equal:
                # self.axis.set_box_aspect(1)
                pass
            if colorbar:
                plt.title('')
                plt.colorbar(im, shrink=0.9)

    def draw_image(self, path_or_array: str | np.ndarray):
        """Draw an image on top of an existing plot.

        This method is best for drawing images on top of other plots
        (example on top of 3D data). For drawing an image by itself
        use imshow instead."""
        if isinstance(path_or_array, str):
            img = mpimg.imread(path_or_array)
        else:
            img: np.ndarray = path_or_array
        imgw, imgh = img.shape[1], img.shape[0]
        xbnd, ybnd = self.axis.get_xbound(), self.axis.get_ybound()
        xdraw = xbnd

        # stretch the image to fit it's original proportions in the y dimension
        width = xbnd[1] - xbnd[0]
        height = imgh * (width / imgw)
        ymid = (ybnd[1] - ybnd[0]) / 2 + ybnd[0]
        ydraw = [ymid - height / 2, ymid + height / 2]

        self.axis.imshow(
            img, extent=[xdraw[0], xdraw[1], ydraw[0], ydraw[1]], zorder=-1
        )

    def pcolormesh(self, *args, colorbar=False, **kwargs) -> None:
        """Allows plotting like imshow, with the additional option of sizing the boxes at will.
        Look at matplotlib.axes.Axes.pcolormesh for more information.

        Parameters
        -----------
        x: iterable
            The coordinates of the x values of quadrilaterals of a pcolormesh
        y: iterable
            The coordinates of the y values of quadrilaterals of a pcolormesh
        C: 2d numpy array
            The vaues corresponding to the regtangle made by the x and y lists.
        """
        if self.view_spec['type'] in ['image']:
            im = self.axis.pcolormesh(*args, **kwargs)
            # self.axis.set_box_aspect(1)
            if colorbar:
                plt.title('')
                plt.colorbar(im, shrink=0.9)

    def contour(self, *args, colorbar=False, **kwargs) -> None:
        """Will plot the contour lines on top of an image.
        See matplotlib.axes.Axes.contour for more information.

        Parameters
        -----------
        X, Y
           They must both be 1-D such that len(X) == N is the number
           of columns in Z and len(Y) == M is the number of rows in Z.
        Z
           The height values over which the contour is drawn. Color-mapping is controlled by cmap, norm, vmin, and vmax.

        """
        if self.view_spec['type'] == 'image':
            im = self.axis.contour(*args, **kwargs)
            self.axis.set_box_aspect(1)
            if colorbar:
                plt.title('')
                plt.colorbar(im, shrink=0.9)

    def draw_hist2d(self, h, xedges, yedges, *args, colorbar=False, **kwargs):
        if self.view_spec['type'] == 'image':
            im = self.axis.imshow(h, **kwargs)
            plt.set_xticks(xedges)
            plt.set_yticks(yedges)
            self.axis.set_box_aspect(1)
            if colorbar:
                plt.title('')
                plt.colorbar(im, shrink=0.9)

    # XYZ <---> PQ CONVERSION

    def xyz2pqw(self, xyz):
        return vs.xyz2pqw(xyz, self.view_spec)

    def xyz2pq(self, xyz):
        return vs.xyz2pq(xyz, self.view_spec)

    def pqw2xyz(self, pqw):
        return vs.pqw2xyz(pqw, self.view_spec)

    def pq2xyz(self, pq):
        return vs.pq2xyz(pq, self.view_spec)

    # XYZ PLOTTING

    def draw_xyz_text(self, xyz, text, style=rctxt.default()):  # An xyz is [x,y,z]
        if len(xyz) != 3:
            lt.error(
                'ERROR: In draw_xyz_text(), len(xyz)=', len(xyz), ' is not equal to 3.'
            )
            assert False
        if self.view_spec['type'] == '3d':
            self.axis.text(
                xyz[0],
                xyz[1],
                xyz[2],
                text,
                horizontalalignment=style.horizontalalignment,
                verticalalignment=style.verticalalignment,
                fontsize=style.fontsize,
                fontstyle=style.fontstyle,
                fontweight=style.fontweight,
                zdir=style.zdir,
                color=style.color,
            )
        elif self.view_spec['type'] == 'xy':
            self.axis.text(
                xyz[0],
                xyz[1],
                text,
                horizontalalignment=style.horizontalalignment,
                verticalalignment=style.verticalalignment,
                fontsize=style.fontsize,
                fontstyle=style.fontstyle,
                fontweight=style.fontweight,
                color=style.color,
            )
        elif self.view_spec['type'] == 'xz':
            self.axis.text(
                xyz[0],
                xyz[2],
                text,
                horizontalalignment=style.horizontalalignment,
                verticalalignment=style.verticalalignment,
                fontsize=style.fontsize,
                fontstyle=style.fontstyle,
                fontweight=style.fontweight,
                color=style.color,
            )
        elif self.view_spec['type'] == 'yz':
            self.axis.text(
                xyz[1],
                xyz[2],
                text,
                horizontalalignment=style.horizontalalignment,
                verticalalignment=style.verticalalignment,
                fontsize=style.fontsize,
                fontstyle=style.fontstyle,
                fontweight=style.fontweight,
                color=style.color,
            )
        elif self.view_spec['type'] == 'vplane':
            pq = vs.xyz2pq(xyz, self.view_spec)
            self.axis.text(
                pq[0],
                pq[1],
                text,
                horizontalalignment=style.horizontalalignment,
                verticalalignment=style.verticalalignment,
                fontsize=style.fontsize,
                fontstyle=style.fontstyle,
                fontweight=style.fontweight,
                color=style.color,
            )
        elif self.view_spec['type'] == 'camera':
            pq = vs.xyz2pq(xyz, self.view_spec)
            if pq:
                self.axis.text(
                    pq[0],
                    pq[1],
                    text,
                    horizontalalignment=style.horizontalalignment,
                    verticalalignment=style.verticalalignment,
                    fontsize=style.fontsize,
                    fontstyle=style.fontstyle,
                    fontweight=style.fontweight,
                    color=style.color,
                    clip_box=self.axis.clipbox,
                    clip_on=True,
                )
        else:
            lt.error(
                "ERROR: In View3d.draw_xyz_text(), unrecognized view_spec['type'] = '"
                + str(self.view_spec['type'])
                + "' encountered."
            )
            assert False

    def draw_xyz(
        self,
        xyz,  # An xyz is [x,y,z]
        style: rcps.RenderControlPointSeq = None,
        label: str = None,
    ):
        """Plots a single point, I think (BGB)."""
        if style == None:
            style = rcps.default()
        if len(xyz) != 3:
            lt.error('ERROR: In draw_xyz(), len(xyz)=', len(xyz), ' is not equal to 3.')
            assert False
        if self.view_spec['type'] == '3d':
            self.axis.plot3D(
                [xyz[0]],
                [xyz[1]],
                [xyz[2]],
                label=label,
                color=style.color,
                marker=style.marker,
                markersize=style.markersize,
                markeredgecolor=style.markeredgecolor,
                markeredgewidth=style.markeredgewidth,
                markerfacecolor=style.markerfacecolor,
            )
        elif self.view_spec['type'] == 'xy':
            self.axis.plot(
                [xyz[0]],
                [xyz[1]],
                label=label,
                color=style.color,
                marker=style.marker,
                markersize=style.markersize,
                markeredgecolor=style.markeredgecolor,
                markeredgewidth=style.markeredgewidth,
                markerfacecolor=style.markerfacecolor,
            )
        elif self.view_spec['type'] == 'xz':
            self.axis.plot(
                [xyz[0]],
                [xyz[2]],
                label=label,
                color=style.color,
                marker=style.marker,
                markersize=style.markersize,
                markeredgecolor=style.markeredgecolor,
                markeredgewidth=style.markeredgewidth,
                markerfacecolor=style.markerfacecolor,
            )
        elif self.view_spec['type'] == 'yz':
            self.axis.plot(
                [xyz[1]],
                [xyz[2]],
                label=label,
                color=style.color,
                marker=style.marker,
                markersize=style.markersize,
                markeredgecolor=style.markeredgecolor,
                markeredgewidth=style.markeredgewidth,
                markerfacecolor=style.markerfacecolor,
            )
        elif self.view_spec['type'] == 'vplane':
            pq = vs.xyz2pq(xyz, self.view_spec)
            self.axis.plot(
                [pq[0]],
                [pq[1]],
                label=label,
                color=style.color,
                marker=style.marker,
                markersize=style.markersize,
                markeredgecolor=style.markeredgecolor,
                markeredgewidth=style.markeredgewidth,
                markerfacecolor=style.markerfacecolor,
            )
        elif self.view_spec['type'] == 'camera':
            pq = vs.xyz2pq(xyz, self.view_spec)
            if pq:
                self.axis.plot(
                    [pq[0]],
                    [pq[1]],
                    label=label,
                    color=style.color,
                    marker=style.marker,
                    markersize=style.markersize,
                    markeredgecolor=style.markeredgecolor,
                    markeredgewidth=style.markeredgewidth,
                    markerfacecolor=style.markerfacecolor,
                )
        else:
            lt.error(
                "ERROR: In View3d.draw_xyz(), unrecognized view_spec['type'] = '"
                + str(self.view_spec['type'])
                + "' encountered."
            )
            assert False

    def draw_single_Pxyz(
        self,
        p: Pxyz,
        style: rcps.RenderControlPointSeq = None,
        labels: list[str] = None,
    ):
        if labels == None:
            labels = [None] * len(p)
        if style == None:
            style = rcps.default(markersize=2)
        for x, y, z, label in zip(p.x, p.y, p.z, labels):
            self.draw_xyz((x, y, z), style, label)

    def draw_xyz_list(
        self, input_xyz_list: list[list], close=False, style=None, label=None
    ) -> None:
        """Draw lines or closed polygons.

        Parameters
        ----------
            input_xyz_list: List of xyz three vectors (eg [[0,0,0], [1,1,1]])
            close: Draw as a closed polygon (ignored if input_xyz_list < 3 points)"""

        if style == None:
            style = rcps.default()

        if len(input_xyz_list) > 0:
            # Construct the point list to draw, including closing the polygon if desired.
            if close and (len(input_xyz_list) > 2):
                xyz_list = input_xyz_list.copy()
                xyz_list.append(input_xyz_list[0])
            else:
                xyz_list = input_xyz_list
            # Draw the point list.
            if self.view_spec['type'] == '3d':
                self.axis.plot3D(
                    [xyz[0] for xyz in xyz_list],
                    [xyz[1] for xyz in xyz_list],
                    [xyz[2] for xyz in xyz_list],
                    label=label,
                    linestyle=style.linestyle,
                    linewidth=style.linewidth,
                    color=style.color,
                    marker=style.marker,
                    markersize=style.markersize,
                    markeredgecolor=style.markeredgecolor,
                    markeredgewidth=style.markeredgewidth,
                    markerfacecolor=style.markerfacecolor,
                )
            elif self.view_spec['type'] == 'xy':
                self.axis.plot(
                    [xyz[0] for xyz in xyz_list],
                    [xyz[1] for xyz in xyz_list],
                    label=label,
                    linestyle=style.linestyle,
                    linewidth=style.linewidth,
                    color=style.color,
                    marker=style.marker,
                    markersize=style.markersize,
                    markeredgecolor=style.markeredgecolor,
                    markeredgewidth=style.markeredgewidth,
                    markerfacecolor=style.markerfacecolor,
                )
            elif self.view_spec['type'] == 'xz':
                self.axis.plot(
                    [xyz[0] for xyz in xyz_list],
                    [xyz[2] for xyz in xyz_list],
                    label=label,
                    linestyle=style.linestyle,
                    linewidth=style.linewidth,
                    color=style.color,
                    marker=style.marker,
                    markersize=style.markersize,
                    markeredgecolor=style.markeredgecolor,
                    markeredgewidth=style.markeredgewidth,
                    markerfacecolor=style.markerfacecolor,
                )
            elif self.view_spec['type'] == 'yz':
                self.axis.plot(
                    [xyz[1] for xyz in xyz_list],
                    [xyz[2] for xyz in xyz_list],
                    label=label,
                    linestyle=style.linestyle,
                    linewidth=style.linewidth,
                    color=style.color,
                    marker=style.marker,
                    markersize=style.markersize,
                    markeredgecolor=style.markeredgecolor,
                    markeredgewidth=style.markeredgewidth,
                    markerfacecolor=style.markerfacecolor,
                )
            elif self.view_spec['type'] == 'vplane':
                pq_list = [vs.xyz2pq(xyz, self.view_spec) for xyz in xyz_list]
                self.axis.plot(
                    [pq[0] for pq in pq_list],
                    [pq[1] for pq in pq_list],
                    label=label,
                    linestyle=style.linestyle,
                    linewidth=style.linewidth,
                    color=style.color,
                    marker=style.marker,
                    markersize=style.markersize,
                    markeredgecolor=style.markeredgecolor,
                    markeredgewidth=style.markeredgewidth,
                    markerfacecolor=style.markerfacecolor,
                )
            elif self.view_spec['type'] == 'camera':
                pq_list = [vs.xyz2pq(xyz, self.view_spec) for xyz in xyz_list]
                # Discard all "None" entries, and split into separate contiguous lists.
                list_of_pq_lists = []
                pq_list_2 = []
                for idx in range(len(pq_list)):
                    if pq_list[idx] != None:
                        pq_list_2.append(pq_list[idx])
                    else:
                        if len(pq_list_2) > 0:
                            list_of_pq_lists.append(pq_list_2)
                        pq_list_2 = []
                if len(pq_list_2) > 0:
                    list_of_pq_lists.append(pq_list_2)
                # Plot the contiguous pq sequences.
                for pq_list_3 in list_of_pq_lists:
                    self.axis.plot(
                        [pq[0] for pq in pq_list_3],
                        [pq[1] for pq in pq_list_3],
                        label=label,
                        linestyle=style.linestyle,
                        linewidth=style.linewidth,
                        color=style.color,
                        marker=style.marker,
                        markersize=style.markersize,
                        markeredgecolor=style.markeredgecolor,
                        markeredgewidth=style.markeredgewidth,
                        markerfacecolor=style.markerfacecolor,
                    )
            else:
                lt.error(
                    "ERROR: In View3d.draw_xyz_list(), unrecognized view_spec['type'] = '"
                    + str(self.view_spec['type'])
                    + "' encountered."
                )
                assert False

    def draw_Vxyz(self, V: Vxyz, close=False, style=None, label=None) -> None:
        """Alternative to View3d.drawxyz_list that used the Vxyz class instead"""
        self.draw_xyz_list(V.data.T, close, style, label)

    # TODO tjlarki: only implemented for 3d views, should extend
    def draw_xyz_surface(
        self,
        x_mesh: ndarray,
        y_mesh: ndarray,
        z_mesh: ndarray,
        surface_style: RenderControlSurface = RenderControlSurface(),
    ):
        if self.view_spec['type'] == '3d':
            self.axis.plot_surface(
                x_mesh.flatten(),
                y_mesh.flatten(),
                z_mesh.flatten(),
                color=surface_style.color,
                alpha=surface_style.alpha,
            )

    def draw_xyz_trisurface(
        self,
        x: ndarray,
        y: ndarray,
        z: ndarray,
        surface_style: RenderControlSurface = None,
        **kwargs
    ):
        if surface_style == None:
            surface_style = RenderControlSurface()
        if self.view_spec['type'] == '3d':
            self.axis.plot_trisurf(
                x, y, z, color=surface_style.color, alpha=surface_style.alpha, **kwargs
            )

    # TODO tjlarki: currently unused
    # TODO tjlarki: might want to remove, this is a very slow function
    def quiver(
        self,
        X: ndarray,
        Y: ndarray,
        Z: ndarray,
        U: ndarray,
        V: ndarray,
        W: ndarray,
        length: float = 0,
    ) -> None:
        self.axis.quiver(X, Y, Z, U, V, W, length=0)

    # PQ PLOTTING

    def draw_pq_text(self, pq, text, style=rctxt.default()):  # A pq is [p,q]
        if (len(pq) != 2) and (len(pq) != 3):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In draw_pq_text(), len(pq)=',
                len(pq),
                ' is not equal to 2 or 3.',
            )
        if self.view_spec['type'] == '3d':
            lt.error_and_raise(
                RuntimeError,
                "ERROR: In View3d.draw_pq_text(), incompatible view_spec['type'] = '"
                + str(self.view_spec['type'])
                + "' encountered.",
            )
        elif (
            (self.view_spec['type'] == 'xy')
            or (self.view_spec['type'] == 'xz')
            or (self.view_spec['type'] == 'yz')
            or (self.view_spec['type'] == 'vplane')
            or (self.view_spec['type'] == 'camera')
        ):
            self.axis.text(
                pq[0],
                pq[1],
                text,
                horizontalalignment=style.horizontalalignment,
                verticalalignment=style.verticalalignment,
                fontsize=style.fontsize,
                fontstyle=style.fontstyle,
                fontweight=style.fontweight,
                color=style.color,
                clip_box=self.axis.clipbox,
                clip_on=True,
            )
        else:
            lt.error_and_raise(
                RuntimeError,
                "ERROR: In View3d.draw_pq_text(), unrecognized view_spec['type'] = '"
                + str(self.view_spec['type'])
                + "' encountered.",
            )

    def draw_pq(self, pq, style=rcps.default(), label=None):  # A pq is [p,q]
        if (len(pq) != 2) and (len(pq) != 3):
            lt.error(
                'ERROR: In draw_pq_text(), len(pq)=',
                len(pq),
                ' is not equal to 2 or 3.',
            )
            assert False
        if self.view_spec['type'] == '3d':
            lt.error(
                "ERROR: In View3d.draw_pq_list(), incompatible view_spec['type'] = '"
                + str(self.view_spec['type'])
                + "' encountered."
            )
            assert False
        elif (
            (self.view_spec['type'] == 'xy')
            or (self.view_spec['type'] == 'xz')
            or (self.view_spec['type'] == 'yz')
            or (self.view_spec['type'] == 'vplane')
            or (self.view_spec['type'] == 'camera')
        ):
            self.axis.plot(
                [pq[0]],
                [pq[1]],
                label=label,
                color=style.color,
                marker=style.marker,
                markersize=style.markersize,
                markeredgecolor=style.markeredgecolor,
                markeredgewidth=style.markeredgewidth,
                markerfacecolor=style.markerfacecolor,
            )
        else:
            lt.error(
                "ERROR: In View3d.draw_pq(), unrecognized view_spec['type'] = '"
                + str(self.view_spec['type'])
                + "' encountered."
            )
            assert False

    def draw_p_list(self, input_p_list, style=rcps.default(), label=None):
        pq_list = [(i, input_p_list[i]) for i in range(len(input_p_list))]
        self.draw_pq_list(pq_list, style=style, label=label)

    def draw_pq_list(
        self,
        input_pq_list,  # A list of pq pairs, where a pq is [p,q]
        close=False,  # Draw as a closed polygon.  Ignored if lss than three points.
        style=rcps.default(),
        label=None,
    ):
        if len(input_pq_list) > 0:
            # Construct the point list to draw, including closing the polygon if desired.
            if close and (len(input_pq_list) > 2):
                pq_list = input_pq_list.copy()
                pq_list.append(input_pq_list[0])
            else:
                pq_list = input_pq_list
            # Draw the point list.
            if self.view_spec['type'] == '3d':
                lt.error(
                    "ERROR: In View3d.draw_pq_list(), incompatible view_spec['type'] = '"
                    + str(self.view_spec['type'])
                    + "' encountered."
                )
                assert False
            elif (
                (self.view_spec['type'] == 'xy')
                or (self.view_spec['type'] == 'xz')
                or (self.view_spec['type'] == 'yz')
                or (self.view_spec['type'] == 'vplane')
                or (self.view_spec['type'] == 'camera')
            ):
                self.axis.plot(
                    [pq[0] for pq in pq_list],
                    [pq[1] for pq in pq_list],
                    label=label,
                    linestyle=style.linestyle,
                    linewidth=style.linewidth,
                    color=style.color,
                    marker=style.marker,
                    markersize=style.markersize,
                    markeredgecolor=style.markeredgecolor,
                    markeredgewidth=style.markeredgewidth,
                    markerfacecolor=style.markerfacecolor,
                )
            else:
                lt.error_and_raise(
                    RuntimeError,
                    "ERROR: In View3d.draw_pq_list(), unrecognized view_spec['type'] = '"
                    + str(self.view_spec['type'])
                    + "' encountered.",
                )

    # VECTOR FIELD PLOTTING

    def draw_xyzdxyz_list(
        self,
        input_xyzdxyz_list: list[
            list[list, list]
        ],  # An xyzdxyz is [[x,y,z], [dx,dy,dz]]
        close: bool = False,  # Draw as a closed polygon. Ignore if less than three points.
        style: rcps.RenderControlPointSeq = rcps.default(),
        label: str = None,
    ):
        if len(input_xyzdxyz_list) > 0:
            # No need to close the xyzdxyz list, since draw_xyz_list will do it.
            xyzdxyz_list = input_xyzdxyz_list
            # Draw the point list.
            xyz_list = [xyzdxyz[0] for xyzdxyz in xyzdxyz_list]
            self.draw_xyz_list(xyz_list, close=close, style=style, label=label)
            # Setup the vector drawing style.
            vector_style = rcps.outline(
                color=style.vector_color, linewidth=style.vector_linewidth
            )
            # Draw the vectors.
            for xyzdxyz in xyzdxyz_list:
                xyz0 = xyzdxyz[0]
                x0 = xyz0[0]
                y0 = xyz0[1]
                z0 = xyz0[2]
                dxyz = xyzdxyz[1]
                dx = dxyz[0]
                dy = dxyz[1]
                dz = dxyz[2]
                # Construct a ray.
                scale = style.vector_scale
                x1 = x0 + (scale * dx)
                y1 = y0 + (scale * dy)
                z1 = z0 + (scale * dz)
                xyz1 = [x1, y1, z1]
                ray = [xyz0, xyz1]
                self.draw_xyz_list(ray, close=False, style=vector_style, label=None)

    def draw_pqdpq_list(
        self,
        input_pqdpq_list,  # A pqdpq is [[p,q], [dp,dq]]
        close=False,  # Draw as a closed polygon. Ignore if less than three points.
        style=rcps.default(),
        label=None,
    ):
        if len(input_pqdpq_list) > 0:
            # No need to close the pqdpq list, since draw_pq_list will do it.
            pqdpq_list = input_pqdpq_list
            # Draw the point list.
            pq_list = [pqdpq[0] for pqdpq in pqdpq_list]
            self.draw_pq_list(pq_list, close=close, style=style, label=label)
            # Setup the vector drawing style.
            vector_style = rcps.outline(
                color=style.vector_color, linewidth=style.vector_linewidth
            )
            # Draw the vectors.
            for pqdpq in pqdpq_list:
                pq0 = pqdpq[0]
                p0 = pq0[0]
                q0 = pq0[1]
                dpq = pqdpq[1]
                px = dpq[0]
                qy = dpq[1]
                # Construct a ray.
                scale = style.vector_scale
                p1 = p0 + (scale * px)
                q1 = q0 + (scale * qy)
                pq1 = [p1, q1]
                ray = [pq0, pq1]
                self.draw_pq_list(ray, close=False, style=vector_style, label=None)
