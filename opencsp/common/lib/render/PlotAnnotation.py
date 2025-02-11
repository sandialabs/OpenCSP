"""
Annotations to add to plots and images.



"""

from cv2 import cv2 as cv
import matplotlib.pyplot as plt

import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.tool.math_tools as mt


class PlotAnnotation:
    """
    Class representing an annotation to add to a plot or image.

    """

    def __init__(
        self,
        type,  # Values are 'point_seq', 'text'
        pt_list,  # List of (x,y) or (x,y,z) points, approriate for the active plot.
        text,  # If type is text, this is the text.  Otherwise interpreted as a legend label.
        style,
    ):  # Appropriate for type, e.g, RenderControlText or RenderControlPointSeq.
        super(PlotAnnotation, self).__init__()

        self.type = type
        self.pt_list = pt_list
        self.text = text
        self.style = style

    def plot(self, crop_box=None):  # Crop box is [[x_min, y_min], [x_max, y_max]] or None.  Only applies to 2-d points.
        """
        Plot the annotation.  assumes that desired plot is active.
        """
        # Point sequence.
        if self.type == 'point_seq':
            if (self.pt_list != None) and (len(self.pt_list) > 0):
                pt0 = self.pt_list[0]  # Sample for length check.
                if len(pt0) == 2:
                    cropped_pt_list = self.crop_pt_list_to_box(self.pt_list, crop_box)
                    if (len(cropped_pt_list) < len(self.pt_list)) and (
                        (isinstance(self.style, type(rcps.default()))) and (self.style.linestyle != 'None')
                    ):
                        print(
                            'WARNING: In PlotAnnotation.plot(), current implementation of point cropping does not ensure proper drawing of lines connecting points.'
                        )
                    plt.plot(
                        [pt[0] for pt in cropped_pt_list],
                        [pt[1] for pt in cropped_pt_list],
                        label=self.text,
                        linestyle=self.style.linestyle,
                        linewidth=self.style.linewidth,
                        color=self.style.color,
                        marker=self.style.marker,
                        markersize=self.style.markersize,
                        markeredgecolor=self.style.markeredgecolor,
                        markeredgewidth=self.style.markeredgewidth,
                        markerfacecolor=self.style.markerfacecolor,
                    )
                elif len(pt0) == 3:
                    plt.plot(
                        [pt[0] for pt in self.pt_list],
                        [pt[1] for pt in self.pt_list],
                        [pt[2] for pt in self.pt_list],
                        label=self.text,
                        linestyle=self.style.linestyle,
                        linewidth=self.style.linewidth,
                        color=self.style.color,
                        marker=self.style.marker,
                        markersize=self.style.markersize,
                        markeredgecolor=self.style.markeredgecolor,
                        markeredgewidth=self.style.markeredgewidth,
                        markerfacecolor=self.style.markerfacecolor,
                    )
                else:
                    print(
                        'ERROR: In PlotAnnotation.plot(), when plotting a point sequence, unexpected point length len(pt0)="'
                        + str(len(pt0))
                        + '" encountered.'
                    )
                    assert False

        # Text.
        elif self.type == 'text':
            if (self.pt_list != None) and (len(self.pt_list) > 0):
                pt = self.pt_list[0]  # Any points beyond first are ignored.
                if len(pt) == 2:
                    if self.point_is_in_crop_box(pt, crop_box):
                        plt.text(
                            pt[0],
                            pt[1],
                            self.text,
                            horizontalalignment=self.style.horizontalalignment,
                            verticalalignment=self.style.verticalalignment,
                            fontsize=self.style.fontsize,
                            fontstyle=self.style.fontstyle,
                            fontweight=self.style.fontweight,
                            color=self.style.color,
                        )
                elif len(pt) == 3:
                    plt.text(
                        pt[0],
                        pt[1],
                        pt[2],
                        self.text,
                        horizontalalignment=self.style.horizontalalignment,
                        verticalalignment=self.style.verticalalignment,
                        fontsize=self.style.fontsize,
                        fontstyle=self.style.fontstyle,
                        fontweight=self.style.fontweight,
                        color=self.style.color,
                    )
                else:
                    print(
                        'ERROR: In PlotAnnotation.plot(), when plotting text, unexpected point length len(pt)="'
                        + str(len(pt))
                        + '" encountered.'
                    )
                    assert False

        # Error trap.
        else:
            print('ERROR: In PlotAnnotation.plot(), unexpected type="' + str(type) + '" encountered.')
            assert False

    def image_draw(
        self, image, crop_box=None  # Image to annotate. Modifies image as a side effect.
    ):  # OpenCV drawing routines automatically crop to image boundaries.
        # Only use this if you want to crop to a box that is a subset of the image.
        # crop_box is [[x_min, y_min], [x_max, y_max]] or None.
        """
        Add annotations to the image, modifying the image.

        Uses OpenCV drawing routines.  For documentation, see:
            https://docs.opencv.org/4.5.3/d3/d96/tutorial_basic_geometric_drawing.html
            https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
            https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#gaf076ef45de481ac96e0ab3dc2c29a777

        Here is some demonstration code:
            cv.circle(img, ( 700,1500), 100, (255,0,0), 10, 8)  # Blue
            cv.circle(img, (1000,1500), 100, (0,255,0), -1, 8)  # Green (filled)
            cv.circle(img, (1300,1500), 100, (0,0,255), 30, 8)  # Red
            cv.line(img,   ( 700,1500), ( 900,1700),  (255,255,0), 10, 8)  # Cyan
            cv.line(img,   (1000,1500), (1200, -50),  (255,0,255), 10, 8)  # Magenta (cropped)
            cv.line(img,   (1300,1500), (1500,7000), (0,255,255), 10, 8)  # Yellow (cropped)
            example_xy_list = [ [200,100], [800,150], [1800,400], [300,900] ]
            example_xy_array = np.array(example_xy_list, np.int32)
            example_xy_reshaped_array = example_xy_array.reshape((-1, 1, 2))
            cv.fillPoly(img, [example_xy_reshaped_array], (255,0,255), 8)  # Magenta (filled)
            cv.polylines(img, [example_xy_reshaped_array*2], True, (255,255,0), 8)  # Cyan (outline, closed)
            cv.polylines(img, [example_xy_reshaped_array*3], False, (255,0,0), 8)  # Blue (outline, not closed)
            print('In demo_routine(), saving annotated image file: ', img_dir_body_ext)
            cv.imwrite(img_dir_body_ext, img)
        """
        # Point sequence.
        if self.type == 'point_seq':
            if (self.pt_list != None) and (len(self.pt_list) > 0):
                pt0 = self.pt_list[0]  # Sample for length check.
                if (
                    False
                ):  # len(pt0) != 2:  # ?? SCAFFOLDING RCB -- THERE IS A BUG IN 180_heliostat3d.py, __init()__, WHERE THE VARIABLE list_of_frame_id_xy_lists_to_process contains elements [-1.0, -1, 0] FOR MISSING POINTS.  I DON'T HAVE TIME RIGHT NOW TO FIND THIS AND FIX IT, SO IGNORE TEMPORARILY.
                    print(
                        'ERROR: In PlotAnnotation.image_draw(), when drawing a point sequence, unexpected point length len(pt0)="'
                        + str(len(pt0))
                        + '" encountered.'
                    )
                    assert False
                else:
                    # Crop to box, if given.  (Recall that OpenCV automatically crops to image boundary.)
                    cropped_pt_list = self.crop_pt_list_to_box(self.pt_list, crop_box)
                    if (len(cropped_pt_list) < len(self.pt_list)) and (
                        (isinstance(self.style, type(rcps.default()))) and (self.style.linestyle != 'None')
                    ):
                        print(
                            'WARNING: In PlotAnnotation.image_draw(), current implementation of point cropping does not ensure proper drawing of lines connecting points.'
                        )
                    # Connecting lines.
                    if (len(cropped_pt_list) > 1) and (self.style.linestyle != 'None'):
                        if (self.style.linestyle != '-') and (self.style.linestyle != 'solid'):
                            print(
                                'WARNING: In PlotAnnotation.image_draw(), dashed or dotted lines are not implemented yet.  Drawing a solid line instead.'
                            )
                        color = self.opencv_color(self.style.color)
                        thickness = int(self.style.linewidth)
                        if thickness == 0:
                            thickness = 1
                        prev_int_pt = None
                        for int_or_float_pt in cropped_pt_list:
                            int_pt = [int(int_or_float_pt[0]), int(int_or_float_pt[1])]
                            if prev_int_pt is not None:
                                connect_line_type = self.opencv_line_type()
                                # Draw connecting line.
                                cv.line(image, prev_int_pt, int_pt, color, thickness, connect_line_type)
                            prev_int_pt = int_pt
                    # Markers.
                    if self.style.marker != 'None':
                        for int_or_float_pt in cropped_pt_list:
                            center = [int(int_or_float_pt[0]), int(int_or_float_pt[1])]
                            radius = int(
                                self.style.markersize / 2
                            )  # ?? SCAFFOLDING RCB -- LATER, REPLACE WITH "image_marker_size"
                            if radius == 0:
                                radius = 1
                            marker = self.style.marker
                            if marker == '.':
                                plot_color = self.style.markerfacecolor
                                thickness = -1  # Filled
                            elif marker == 'o':
                                plot_color = self.style.markeredgecolor
                                thickness = int(
                                    self.style.markeredgewidth
                                )  # ?? SCAFFOLDING RCB -- LATER, REPLACE WITH "image_marker_edge_width"
                            else:
                                print(
                                    'WARNING: In PlotAnnotation.image_draw(), marker type "'
                                    + str(marker)
                                    + '" is not implemented yet.  Drawing filled circle instead.'
                                )
                                plot_color = self.style.markerfacecolor
                                thickness = -1  # Filled
                            # OpenCV drawing parameters.
                            color = self.opencv_color(plot_color)
                            marker_line_type = self.opencv_line_type()
                            # Draw marker.
                            cv.circle(image, center, radius, color, thickness, marker_line_type)
                    # Label text.
                    if (self.text is not None) and (len(self.text) > 0):
                        print(
                            'WARNING: In PlotAnnotation.image_draw(), drawing legend labels for points is not implemented yet.'
                        )

        # Text.
        elif self.type == 'text':
            if (self.pt_list != None) and (len(self.pt_list) > 0):
                int_or_float_pt0 = self.pt_list[0]  # Any points beyond first are ignored.
                if len(int_or_float_pt0) != 2:
                    print(
                        'ERROR: In PlotAnnotation.image_draw(), when drawing a text string, unexpected point length len(int_or_float_pt0)="'
                        + str(len(int_or_float_pt0))
                        + '" encountered.'
                    )
                    assert False
                else:
                    int_pt0 = [int(int_or_float_pt0[0]), int(int_or_float_pt0[1])]
                    if self.point_is_in_crop_box(int_pt0, crop_box):
                        # OpenCV drawing parameters.
                        color = self.opencv_color(self.style.color)
                        font = self.opencv_font(self.style.fontstyle)
                        font_scale = self.opencv_font_scale(self.style.fontsize)
                        font_thickness = self.opencv_font_thickness(self.style.fontweight)
                        font_line_type = self.opencv_font_line_type()
                        origin = self.opencv_origin(
                            self.text,
                            int_pt0,
                            font,
                            font_scale,
                            font_thickness,
                            self.style.horizontalalignment,
                            self.style.verticalalignment,
                        )
                        # Draw text.
                        cv.putText(image, self.text, origin, font, font_scale, color, font_thickness, font_line_type)

        # Error trap.
        else:
            print('ERROR: In PlotAnnotation.image_draw(), unexpected type="' + str(type) + '" encountered.')
            assert False

    def crop_pt_list_to_box(self, input_point_list, crop_box):
        if crop_box is None:
            return input_point_list
        else:
            xy_min = crop_box[0]
            xy_max = crop_box[1]
            x_min = xy_min[0]
            y_min = xy_min[1]
            x_max = xy_max[0]
            y_max = xy_max[1]
            cropped_pt_list = []
            for pt in input_point_list:
                x = pt[0]
                y = pt[1]
                if (x_min <= x) and (x <= x_max) and (y_min <= y) and (y <= y_max):
                    cropped_pt_list.append(pt)
            return cropped_pt_list

    def point_is_in_crop_box(self, point, crop_box):
        if crop_box is None:
            return True
        else:
            xy_min = crop_box[0]
            xy_max = crop_box[1]
            x_min = xy_min[0]
            y_min = xy_min[1]
            x_max = xy_max[0]
            y_max = xy_max[1]
            x = point[0]
            y = point[1]
            if (x_min <= x) and (x <= x_max) and (y_min <= y) and (y <= y_max):
                return True
            else:
                return False

    def opencv_color(self, plot_color):
        """
        OpenCV colors are (B,G,R) tuples.
        """
        if (plot_color == 'k') or (plot_color == 'black'):
            return (0, 0, 0)
        elif (plot_color == 'w') or (plot_color == 'white'):
            return (255, 255, 255)
        elif (plot_color == 'r') or (plot_color == 'red'):
            return (0, 0, 255)
        elif (plot_color == 'g') or (plot_color == 'green'):
            return (0, 255, 0)
        elif (plot_color == 'b') or (plot_color == 'blue'):
            return (255, 0, 0)
        elif (plot_color == 'c') or (plot_color == 'cyan'):
            return (255, 255, 0)
        elif (plot_color == 'm') or (plot_color == 'magenta'):
            return (255, 0, 255)
        elif (plot_color == 'y') or (plot_color == 'yellow'):
            return (0, 255, 255)
        else:
            print(
                'WARNING: In PlotAnnotation.opencv_color(), plot_color "'
                + str(plot_color)
                + '" is not implemented yet.  Using medium grey instead.'
            )
            return (127, 127, 127)

    def opencv_line_type(self):
        """
        See https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#gaf076ef45de481ac96e0ab3dc2c29a777
        """
        # line_type = cv.LINE_4  # 4-connected line
        line_type = cv.LINE_8  # 8-connected line
        # line_type = cv.LINE_AA  # anti-aliased line
        return line_type

    def opencv_font_line_type(self):
        """
        See https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#gaf076ef45de481ac96e0ab3dc2c29a777
        """
        # line_type = cv.LINE_4  # 4-connected line
        # line_type = cv.LINE_8  # 8-connected line
        line_type = cv.LINE_AA  # anti-aliased line
        return line_type

    def opencv_font(self, plot_fontstyle):
        """
        See https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html
        """
        if plot_fontstyle == 'normal':
            font = cv.FONT_HERSHEY_SIMPLEX
        elif plot_fontstyle == 'italic':
            font = cv.FONT_HERSHEY_SIMPLEX
            font += cv.FONT_ITALIC
        elif plot_fontstyle == 'oblique':
            font = cv.FONT_HERSHEY_SIMPLEX
            font += cv.FONT_ITALIC
        else:
            print(
                'ERROR: In PlotAnnotation.opencv_font(), unexpected plot_fontstyle="'
                + str(plot_fontstyle)
                + '" encountered.'
            )
            assert False
        return font

    def opencv_font_scale(self, plot_fontsize):
        """
        OpenCV font thickness must be an integer.
        """
        standard_scale = 0.4
        if mt.is_number(plot_fontsize):
            scale = standard_scale * plot_fontsize
        elif plot_fontsize == 'xx-small':
            scale = standard_scale * 0.25
        elif plot_fontsize == 'x-small':
            scale = standard_scale * 0.50
        elif plot_fontsize == 'small':
            scale = standard_scale * 0.75
        elif plot_fontsize == 'medium':
            scale = standard_scale * 1.00
        elif plot_fontsize == 'large':
            scale = standard_scale * 1.5
        elif plot_fontsize == 'x-large':
            scale = standard_scale * 2.0
        elif plot_fontsize == 'xx-large':
            scale = standard_scale * 3.0
        else:
            print(
                'ERROR: In PlotAnnotation.opencv_font_scale(), unexpected plot_fontsize="'
                + str(plot_fontsize)
                + '" encountered.'
            )
            assert False
        return scale

    def opencv_font_thickness(self, plot_fontweight):
        """
        OpenCV font thickness must be an integer.
        """
        if plot_fontweight == 'light':
            thickness = 1
        elif plot_fontweight == 'normal':
            thickness = 3
        elif plot_fontweight == 'bold':
            thickness = 6
        else:
            print(
                'ERROR: In PlotAnnotation.opencv_font_thickness(), unexpected plot_fontweight="'
                + str(plot_fontweight)
                + '" encountered.'
            )
            assert False
        return thickness

    def opencv_origin(
        self, text, int_pt, font, font_scale, font_thickness, plot_horizontalalignment, plot_verticalalignment
    ):
        # Determine size of text.
        text_box, baseline = cv.getTextSize(text, font, font_scale, font_thickness)
        width = text_box[0]
        height = text_box[1]
        # Fetch x and y coordinates.
        input_x = int_pt[0]
        input_y = int_pt[1]
        # Compute origin x.
        if plot_horizontalalignment == 'left':
            origin_x = input_x
        elif plot_horizontalalignment == 'center':
            origin_x = input_x - int(width / 2)
        elif plot_horizontalalignment == 'right':
            origin_x = input_x - width
        else:
            print(
                'ERROR: In PlotAnnotation.opencv_origin(), unexpected plot_horizontalalignment="'
                + str(plot_horizontalalignment)
                + '" encountered.'
            )
            assert False
        # Compute origin y.
        if plot_verticalalignment == 'bottom':
            origin_y = input_y
        elif plot_verticalalignment == 'center':
            origin_y = input_y + int(height / 2)  # Recall that in images, y is flipped.
        elif plot_verticalalignment == 'top':
            origin_y = input_y + height  # Recall that in images, y is flipped.
        else:
            print(
                'ERROR: In PlotAnnotation.opencv_origin(), unexpected plot_verticalalignment="'
                + str(plot_verticalalignment)
                + '" encountered(1).'
            )
            assert False
        # Return.
        return [origin_x, origin_y]


# COMMON CASES


def outline_annotation(point_list, color='k', linewidth=1):
    """
    Outlines of physical objects.
    """
    return PlotAnnotation('point_seq', point_list, None, rcps.outline(color=color, linewidth=linewidth))


def data_curve_annotation(point_list, color='b', linewidth=1):
    """
    A data curve with data points identified.
    """
    return PlotAnnotation('point_seq', point_list, None, rcps.data_curve(color=color, linewidth=linewidth))


def marker_annotation(point_list, marker='o', color='b', markersize=3):
    """
    A data curve with data points identified.
    """
    return PlotAnnotation('point_seq', point_list, None, rcps.marker(marker='o', color='b', markersize=3))


def text_annotation(
    point, text_str, fontsize='medium', color='b', horizontalalignment='center', verticalalignment='center'
):
    """
    A text annotation.
    """
    return PlotAnnotation(
        'text',
        [point],
        text_str,
        rctxt.RenderControlText(
            fontsize=fontsize, color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
        ),
    )
