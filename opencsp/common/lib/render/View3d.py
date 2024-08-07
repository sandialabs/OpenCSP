import copy
import os
import numbers
import time
from typing import Callable, Iterable

import matplotlib.backend_bases as backb
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from PIL import Image

from matplotlib.axes import Axes
import matplotlib.backend_bases as backb
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from PIL import Image
import scipy.ndimage

from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.axis_3d as ax3d
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.render.lib.AbstractPlotHandler as aph
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface


class View3d(aph.AbstractPlotHandler):
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

        # interactive graphing values
        self._callbacks: dict[str, int] = {}

        # other values
        self.view = figure
        self.axis = axis
        self.view_spec = view_spec
        self.equal = equal
        self.parent = parent
        self.x_limits = None
        self.y_limits = None
        self.z_limits = None

    @property
    def view(self) -> Figure:
        return self._figure

    @view.setter
    def view(self, val: Figure):
        self._figure = val
        self._register_plot(val)

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
                ax3d.set_3d_axes_equal(self.axis)  # , set_zmin_zero=True, box_aspect=None)
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

    # INTERACTION

    def register_event_handler(self, event_type: str, callback: Callable[[backb.Event], None]):
        # deregister the previous callback
        if event_type in self._callbacks:
            self.view.figure.canvas.mpl_disconnect(self._callbacks[event_type])
            del self._callbacks[event_type]

        # register the new callback
        self._callbacks[event_type] = self.view.figure.canvas.mpl_connect(event_type, callback)

    def on_key_press(self, event: backb.KeyEvent, draw_func: Callable):
        if event.key == 'f5':
            lt.info(time.time())
            self.clear()
            draw_func()

    # HELPER

    def _plot(self, x: list | list[list], y: list | list[list], *vargs, style: any = None, **kwargs):
        """
        Like matplotlib.pyplot.plot(), except that we've overloaded this to plot
        normally or to plot arrows.

        To plot with arrows, set style.marker to 'arrow'. Note that some of the
        style choices might be presented differently (or not at all) as compared
        to more standard scatter or line plots.

        Parameters
        ----------
        x : list | list[list]
            The x locations of the values to plot. For example [1, 2, 3] or [np.array([1, 2, 3])].
        y : list | list[list]
            The y locations of the values to plot. For example [1, 2, 3] or [np.array([1, 2, 3])].
        style : any, optional
            The marker property of the style controls if we draw normally or
            with arrows. This is the only property that is used from this
            parameter.
            This will typically be a RenderControlPointSeq instance. By default
            None.
        """
        if style is not None and isinstance(style, rcps.RenderControlPointSeq) and style.marker == "arrow":
            # some of the arguments between plot and arrow are different
            kwargs = copy.copy(kwargs)
            toremove = ["marker", "markeredgewidth"]
            for remove_kw in toremove:
                if remove_kw in kwargs:
                    del kwargs[remove_kw]
            substitutions = [
                ("markersize", "head_width"),
                ("markeredgecolor", "facecolor"),
                ("markerfacecolor", "facecolor"),
            ]
            for from_kw, to_kw in substitutions:
                if from_kw in kwargs:
                    if from_kw is not None:
                        kwargs[to_kw] = kwargs[from_kw]
                    del kwargs[from_kw]

            # encapsulate x and y in a list, if not already a list
            if not hasattr(x[0], 'len'):
                x = [x]
                y = [y]

            # draw the arrows!
            for list_idx in range(len(x)):
                for arrow_idx in range(len(x[list_idx]) - 1):
                    c1 = (x[list_idx][arrow_idx], y[list_idx][arrow_idx])
                    c2 = (x[list_idx][arrow_idx + 1], y[list_idx][arrow_idx + 1])
                    self.axis.arrow(
                        c1[0], c1[1], c2[0] - c1[0], c2[1] - c1[1], *vargs, length_includes_head=True, **kwargs
                    )
        else:
            self.axis.plot(x, y, *vargs, **kwargs)

    # WRITE

    def show_and_save_multi_axis_limits(self, output_dir, output_figure_body, limits_list, grid=True):
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
                            output_dir, output_figure_body, x_limits=limits[0], y_limits=limits[1], grid=grid
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
        self.view.set_size_inches(self.view.get_figwidth(), self.view.get_figheight(), forward=True)
        #    it could be that another backend requires the following instead:
        #    self.view.set_figwidth(self.view.get_figwidth() * dpi)
        #    self.view.set_figheight(self.view.get_figheight() * dpi)
        self.view.set_dpi(dpi)
        self.view.savefig(output_figure_dir_body_ext, format=format, dpi=dpi)
        # Return the outptu file path and directory.
        return output_figure_dir_body_ext

    def limit_suffix(self):
        limit_suffix_str = ''
        if self.x_limits:
            limit_suffix_str += '_' + str(self.x_limits[0]) + 'x' + str(self.x_limits[1])
        if self.y_limits:
            limit_suffix_str += '_' + str(self.y_limits[0]) + 'y' + str(self.y_limits[1])
        if self.z_limits:
            limit_suffix_str += '_' + str(self.z_limits[0]) + 'z' + str(self.z_limits[1])
        return limit_suffix_str

    # Image Plotting
    def imshow(self, *args, colorbar=False, **kwargs) -> None:
        """Draw an image on a 2D plot. Requires view_spec type to be 'image'.

        This method is best for drawing an image by itself. For drawing images on
        top of other plots (example on top of 3D data) use draw_image instead."""
        if self.view_spec['type'] == 'image':
            # load the image, as necessary
            load_as_necessary = lambda img: (img if not isinstance(img, str) else Image.open(img))
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

        self.axis.imshow(img, extent=[xdraw[0], xdraw[1], ydraw[0], ydraw[1]], zorder=-1)

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
            lt.error('ERROR: In draw_xyz_text(), len(xyz)=', len(xyz), ' is not equal to 3.')
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
        xyz: (
            tuple[float, float] | tuple[float, float, float] | tuple[list, list] | tuple[list, list, list] | np.ndarray
        ),
        style: rcps.RenderControlPointSeq = None,
        label: str = None,
    ):  # An xyz is [x,y,z]
        """
        Plots one or more points.

        This is similar to draw_xyz_list, except that it accepts the point
        locations in a different format. Example usage::

            # viewspec xy or pq
            draw_xyz((0, 1))
            draw_xyz((2, 3))
            # or
            draw_xyz(([0, 2], [1, 3]))
            # or
            draw_xyz(np.array([[0, 1], [2, 3]]))

            # viewspec xyz or pqr
            draw_xyz((0, 1, 2))
            draw_xyz((3, 4, 5))
            # or
            draw_xyz(([0, 3], [1, 4], [2, 5]))
            draw_xyz(np.array([[0, 1, 2], [3, 4, 5]]))

        Parameters
        ----------
        xyz : tuple[any, any] | tuple[any, any, any] | np.ndarray
            A set of x, y, and z points. This can be a set of numbers, a set of
            equal-length lists, or a numpy array with shape (2,N) or (3,N).
        style : rcps.RenderControlPointSeq, optional
            The style used to render the points, or None for the default style
            (blue, marker '.', line style '-'). By default None.
        label : str, optional
            The label used to identify this plot on the graph legend. None not
            to be included in the legend. By default None.
        """
        if isinstance(xyz, np.ndarray):
            lval = xyz
        elif len(xyz) == 2:
            if isinstance(xyz[0], numbers.Number):
                lval = [(xyz[0], xyz[1])]
            else:
                lval = [(xyz[0][i], xyz[1][i]) for i in range(len(xyz[0]))]
        else:
            if isinstance(xyz[0], numbers.Number):
                lval = [(xyz[0], xyz[1], xyz[2])]
            else:
                lval = [(xyz[0][i], xyz[1][i], xyz[2][i]) for i in range(len(xyz[0]))]

        self.draw_xyz_list(lval, style=style, label=label)

    def draw_single_Pxyz(self, p: Pxyz, style: rcps.RenderControlPointSeq = None, labels: list[str] = None):
        if labels == None:
            labels = [None] * len(p)
        if style == None:
            style = rcps.default(markersize=2)
        for x, y, z, label in zip(p.x, p.y, p.z, labels):
            self.draw_xyz((x, y, z), style, label)

    def draw_xyz_list(
        self,
        input_xyz_list: Iterable[tuple[float, float, float]],
        close=False,
        style: rcps.RenderControlPointSeq = None,
        label: str = None,
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

            # Draw the point list in 3d.
            if self.view_spec['type'] == '3d':
                x_list = [xyz[0] for xyz in xyz_list]
                y_list = [xyz[1] for xyz in xyz_list]
                z_list = [xyz[2] for xyz in xyz_list]
                self.axis.plot3D(
                    x_list,
                    y_list,
                    z_list,
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

            # Draw the point list in 2d.
            elif self.view_spec['type'] in ['xy', 'yz', 'xz', 'vplane']:
                if self.view_spec['type'] == 'xy':
                    coords1 = [xyz[0] for xyz in xyz_list]
                    coords2 = [xyz[1] for xyz in xyz_list]
                if self.view_spec['type'] == 'xz':
                    coords1 = [xyz[0] for xyz in xyz_list]
                    coords2 = [xyz[2] for xyz in xyz_list]
                if self.view_spec['type'] == 'yz':
                    coords1 = [xyz[1] for xyz in xyz_list]
                    coords2 = [xyz[2] for xyz in xyz_list]
                if self.view_spec['type'] == 'vplane':
                    pq_list = [vs.xyz2pq(xyz, self.view_spec) for xyz in xyz_list]
                    coords1 = [pq[0] for pq in pq_list]
                    coords2 = [pq[1] for pq in pq_list]

                self._plot(
                    coords1,
                    coords2,
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

            # Draw the point list in the camera's perspective.
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
                    self._plot(
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
                lt.error_and_raise(
                    RuntimeError,
                    "ERROR: In View3d.draw_xyz_list(), unrecognized view_spec['type'] = '"
                    + str(self.view_spec['type'])
                    + "' encountered.",
                )

    def draw_Vxyz(self, V: Vxyz, close=False, style=None, label=None) -> None:
        """Alternative to View3d.drawxyz_list that used the Vxyz class instead"""
        self.draw_xyz_list(list(V.data.T), close, style, label)

    # TODO TJL: only implemented for 3d views, should extend
    def draw_xyz_surface(
        self,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        z_mesh: np.ndarray,
        surface_style: RenderControlSurface = None,
        **kwargs,
    ):
        if surface_style is None:
            surface_style = RenderControlSurface()

        if self.view_spec['type'] == '3d':
            axis: Axes3D = self.axis

            # Draw the surface
            axis.plot_surface(
                x_mesh,
                y_mesh,
                z_mesh,
                color=surface_style.color,
                cmap=surface_style.color_map,
                edgecolor=surface_style.edgecolor,
                linewidth=surface_style.linewidth,
                alpha=surface_style.alpha,
                antialiased=surface_style.antialiased,
                **kwargs,
            )

            # Draw the contour plots
            if surface_style.contour:
                for ax, mesh in [('x', x_mesh), ('y', y_mesh), ('z', z_mesh)]:
                    if surface_style.contours[ax]:
                        mmin, mmax = np.min(mesh), np.max(mesh)
                        height = mmax - mmin

                        # placement is determined by graph's orientation
                        lower_offset = mmin - np.max([height / 3, 1])
                        upper_offset = mmax + np.max([height / 3, 1])
                        offset = lower_offset
                        elev, azim = axis.elev, axis.azim  # angle 0-360
                        if ax == 'z':
                            if elev < 0 or elev > 180:
                                offset = upper_offset
                        elif ax == 'x':
                            if azim > 90 and azim < 270:
                                offset = upper_offset
                        elif ax == 'y':
                            if azim < 0 or azim > 180:
                                offset = upper_offset

                        # axis-specific arguments
                        contourf_kwargs = {}
                        if ax != 'z':
                            contourf_kwargs = {'zdir': ax}

                        axis.contourf(
                            x_mesh,
                            y_mesh,
                            z_mesh,
                            offset=offset,
                            cmap=surface_style.contour_color_map,
                            alpha=surface_style.contour_alpha,
                            **contourf_kwargs,
                        )

            # Draw the title
            if surface_style.draw_title:
                if self.parent is not None:
                    axis.set_title(self.parent.title)

    def draw_xyz_surface_customshape(
        self,
        x_mesh: np.ndarray,
        y_mesh: np.ndarray,
        z_mesh: np.ndarray,
        surface_style: RenderControlSurface = None,
        **kwargs,
    ):
        draw_callback = lambda: self._draw_xyz_surface_customshape(x_mesh, y_mesh, z_mesh, surface_style, **kwargs)
        self.register_event_handler('key_release_event', lambda event: self.on_key_press(event, draw_callback))
        draw_callback()

    def draw_xyz_surface(self, surface: np.ndarray, surface_style: RenderControlSurface = None, **kwargs):
        """
        Draw a 3D plot for the given z_mesh surface.

        Example from https://matplotlib.org/stable/plot_types/3D/surface3d_simple.html#sphx-glr-plot-types-3d-surface3d-simple-py::

            # Make data
            X = np.arange(-5, 5)
            Y = np.arange(-5, 5)
            X_mesh, Y_mesh = np.meshgrid(X, Y)
            R = np.sqrt(X_mesh**2 + Y_mesh**2)
            Z = np.sin(R)

            print(X)
            # array([[-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4],
            #        [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4]])
            print(np.round(Z, 2))
            # array([[ 0.71,  0.12, -0.44, -0.78, -0.93, -0.96, -0.93, -0.78, -0.44,  0.12],
            #        [ 0.12, -0.59, -0.96, -0.97, -0.83, -0.76, -0.83, -0.97, -0.96, -0.59],
            #        [-0.44, -0.96, -0.89, -0.45, -0.02,  0.14, -0.02, -0.45, -0.89, -0.96],
            #        [-0.78, -0.97, -0.45,  0.31,  0.79,  0.91,  0.79,  0.31, -0.45, -0.97],
            #        [-0.93, -0.83, -0.02,  0.79,  0.99,  0.84,  0.99,  0.79, -0.02, -0.83],
            #        [-0.96, -0.76,  0.14,  0.91,  0.84,     0,  0.84,  0.91,  0.14, -0.76],
            #        [-0.93, -0.83, -0.02,  0.79,  0.99,  0.84,  0.99,  0.79, -0.02, -0.83],
            #        [-0.78, -0.97, -0.45,  0.31,  0.79,  0.91,  0.79,  0.31, -0.45, -0.97],
            #        [-0.44, -0.96, -0.89, -0.45, -0.02,  0.14, -0.02, -0.45, -0.89, -0.96],
            #        [ 0.12, -0.59, -0.96, -0.97, -0.83, -0.76, -0.83, -0.97, -0.96, -0.59]])

            # Plot the surface
            rc_fig = rcf.RenderControlFigure(tile=False)
            rc_axis = rca.RenderControlAxis(z_label='Light Intensity')
            rc_surf = rcs.RenderControlSurface()
            fig_record = fm.setup_figure_for_3d_data(
                rc_fig, rc_axis, equal=False, name='Light Intensity', code_tag=f"{__file__}")
            view = fig_record.view
            view.draw_xyz_surface(Z)
            plt.show()

        Parameters
        ----------
        surface : ndarray
            2D array of surface values.
        surface_style : RenderControlSurface, optional
            How to style the surface, by default  RenderControlSurface()
        """
        # validate the inputs
        conformed_surface = surface
        if conformed_surface.ndim > 2:
            conformed_surface = np.squeeze(conformed_surface)
        if conformed_surface.ndim != 2:
            lt.error_and_raise(
                ValueError,
                "Error in View3d.draw_xyz_surface(): "
                + f"given surface should have 2 dimensions, but shape is {conformed_surface.shape}",
            )

        # generate the x_mesh and y_mesh
        width = conformed_surface.shape[1]
        height = conformed_surface.shape[0]
        x_arr = np.arange(0, width)
        y_arr = np.arange(0, height)
        x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

        # draw!
        self.draw_xyz_surface_customshape(x_mesh, y_mesh, conformed_surface, surface_style, **kwargs)

    def draw_xyz_trisurface(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, surface_style: RenderControlSurface = None, **kwargs
    ):
        if surface_style == None:
            surface_style = RenderControlSurface()
        if self.view_spec['type'] == '3d':
            self.axis.plot_trisurf(x, y, z, color=surface_style.color, alpha=surface_style.alpha, **kwargs)

    # TODO TJL: currently unused
    # TODO TJL: might want to remove, this is a very slow function
    def quiver(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        W: np.ndarray,
        length: float = 0,
    ) -> None:
        self.axis.quiver(X, Y, Z, U, V, W, length=0)

    # PQ PLOTTING

    def draw_pq_text(self, pq, text, style=rctxt.default()):  # A pq is [p,q]
        if (len(pq) != 2) and (len(pq) != 3):
            lt.error_and_raise(RuntimeError, 'ERROR: In draw_pq_text(), len(pq)=', len(pq), ' is not equal to 2 or 3.')
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

    def draw_pq(self, pq: tuple[list, list], style=rcps.default(), label=None):  # A pq is [p,q]
        """
        Draws the given points to this view. Only draws the points.

        Parameters
        ----------
        pq : tuple[list,list]
            A pair of pq lists to be plotted on the x and y axis, respectively.
            Most typically these will be lists of floats. For
            example: ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        style : RenderControlPointSeq, optional
            The style used to render the points. By default rcps.default().
        label : str, optional
            The label for this plot for use in the legend, or None for no label. By default None.
        """

        if (len(pq) != 2) and (len(pq) != 3):
            lt.error('ERROR: In draw_pq_text(), len(pq)=', len(pq), ' is not equal to 2 or 3.')
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
            self._plot(
                [pq[0]],
                [pq[1]],
                style=style,
                label=label,
                linestyle=None,
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
        input_pq_list: Iterable[tuple[any, any]],
        close: bool = False,
        style: rcps.RenderControlPointSeq = None,
        label: str = None,
    ):
        """
        Draws the given list to this view.

        Parameters
        ----------
        input_pq_list : Iterable[tuple[any, any]]
            A list of pq pairs to be plotted on the x and y axis, respectively.
            Most typically this will be a list of [p,q] float pairs. For
            example: [[0,5], [1,3], [2,5], ...]
        close : bool
            Draw as a closed polygon. Ignored if less than three points. By default False.
        style : RenderControlPointSeq, optional
            The style used to render the points. By default rcps.default(),
            which will draw a line plot.
        label : str, optional
            The label for this plot for use in the legend, or None for no label. By default None.
        """
        if style is None:
            style = rcps.default()
        if isinstance(input_pq_list, Iterable):
            if not isinstance(input_pq_list, list):
                input_pq_list = list(input_pq_list)
        if len(input_pq_list) == 0:
            return

        # Construct the point list to draw, including closing the polygon if desired.
        if close and (len(input_pq_list) > 2):
            pq_list = input_pq_list.copy()
            pq_list.append(input_pq_list[0])
        else:
            pq_list = input_pq_list
        allowed_vs_types = ['xy', 'xz', 'yz', 'vplane', 'camera']
        if self.view_spec['type'] not in allowed_vs_types:
            lt.error_and_raise(
                RuntimeError,
                "ERROR: In View3d.draw_pq_list(), "
                + f"unrecognized view_spec['type'] = '{self.view_spec['type']}' encountered. "
                + f"Should be one of {allowed_vs_types}.",
            )

        # Draw the point list lines and markers
        self._plot(
            [pq[0] for pq in pq_list],
            [pq[1] for pq in pq_list],
            style=style,
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

    # VECTOR FIELD PLOTTING

    def draw_xyzdxyz_list(
        self,
        input_xyzdxyz_list: list[list[list, list]],  # An xyzdxyz is [[x,y,z], [dx,dy,dz]]
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
            vector_style = rcps.outline(color=style.vector_color, linewidth=style.vector_linewidth)
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
            vector_style = rcps.outline(color=style.vector_color, linewidth=style.vector_linewidth)
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
