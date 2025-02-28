import math

import opencsp.common.lib.geometry.Vxyz as Vxyz
import opencsp.common.lib.geometry.Uxyz as Uxyz


def matlab_color_bar():
    """
    Generate a color bar based on manually measured slope magnitude plots from the original MATLAB SOFAST version.

    Returns
    -------
    list of tuple
        A list of RGB tuples representing the color bar, where each tuple contains three integers
        corresponding to the red, green, and blue color channels (0-255).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    color_bar = [
        (0, 0, 143),
        (0, 0, 159),
        (0, 0, 175),
        (0, 0, 191),
        (0, 0, 207),
        (0, 0, 223),
        (0, 0, 239),
        (0, 0, 255),
        (0, 16, 255),
        (0, 32, 255),
        (0, 48, 255),
        (0, 64, 255),
        (0, 80, 255),
        (0, 96, 255),
        (0, 112, 255),
        (0, 128, 255),
        (0, 143, 255),
        (0, 159, 255),
        (0, 175, 255),
        (0, 191, 255),
        (0, 207, 255),
        (0, 223, 255),
        (0, 239, 255),
        (0, 255, 255),
        (32, 255, 223),
        (48, 255, 207),
        (64, 255, 191),
        (75, 255, 180),
        (80, 255, 175),
        (96, 255, 159),
        (112, 255, 143),
        (128, 255, 128),
        (143, 255, 112),
        (159, 255, 96),
        (175, 255, 80),
        (191, 255, 64),
        (207, 255, 48),
        (223, 255, 32),
        (239, 255, 16),
        (255, 255, 0),
        (255, 239, 0),
        (255, 223, 0),
        (255, 207, 0),
        (255, 191, 0),
        (255, 175, 0),
        (255, 159, 0),
        (255, 143, 0),
        (255, 128, 0),
        (255, 112, 0),
        (255, 96, 0),
        (255, 80, 0),
        (255, 64, 0),
        (255, 48, 0),
        (255, 32, 0),
        (255, 16, 0),
        (255, 0, 0),
        (239, 0, 0),
        (223, 0, 0),
        (207, 0, 0),
        (191, 0, 0),
        (175, 0, 0),
        (159, 0, 0),
        (143, 0, 0),
        (128, 0, 0),
    ]  # ?? SCAFFOLDING RCB -- BUG: THIS LAST ENTRY IS AN ACTIVE CELL, NOT A BOUNDARY AT THE END OF THE CELL LIST.
    return color_bar


### # ?? SCAFFOLDING RCB -- USE THIS TO CLARIFY AND THEN FIXPROBLEMS WITH COLOR INTEROPLATION.
### def corner_tour_closed_color_bar():
###     """
###     This color bar traverses the corners of the [R,G,B] space, excluding black and white, returning to the start point.
###     """
###     color_bar = [ (0,0,255),
###                   (0,255,255),
###                   (0,255,0),
###                   (255,255,0),
###                   (255,0,0),
###                   (255,0,255),
###                   (0,0,255) ]  # End boundary of the color bar, not the color of the last cell in the color bar.
###     return color_bar


# ?? SCAFFOLDING RCB -- USE THIS TO CLARIFY AND THEN FIXPROBLEMS WITH COLOR INTEROPLATION.
def corner_tour_closed_color_bar():
    """
    Generate a color bar that traverses the corners of the RGB space, excluding black and white,
    and returns to the starting point.

    Returns
    -------
    list of tuple
        A list of RGB tuples representing the color bar, where each tuple contains three floats
        corresponding to the red, green, and blue color channels (0-255).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    #                  Red    Green     Blue
    color_bar = [
        (0.00, 0.00, 255.00),
        (0.00, 127.50, 255.00),
        (0.00, 255.00, 255.00),
        (0.00, 255.00, 127.50),
        (0.00, 255.00, 0.00),
        (127.50, 255.00, 0.00),
        (255.00, 255.00, 0.00),
        (255.00, 127.50, 0.00),
        (255.00, 0.00, 0.00),
        (255.00, 0.00, 127.50),
        (255.00, 0.00, 255.00),
        (127.50, 0.00, 255.00),
        (0.00, 0.00, 255.00),
    ]  # End boundary of the color bar, not the color of the last cell in the color bar.
    return color_bar


# COLOR WHEEL COLOR BARS
#
# Color wheel naming convention:
#    O  - "O"
# Think of first letter as a visual picture of the color wheel:
#    "O" closed
#    "C" open
#    "G" open, with a special color at one end.


def O_color_bar():
    """
    Generate a color bar that moves smoothly through a sequence of colors:
    Blue --> Cyan --> Green --> Yellow --> Red --> Magenta --> Blue.

    Returns
    -------
    list of tuple
        A list of RGB tuples representing the color bar, where each tuple contains three floats
        corresponding to the red, green, and blue color channels (0-255).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    #                  Red    Green     Blue
    color_bar = [
        (0.00, 0.00, 255.00),
        (0.00, 63.75, 255.00),
        (0.00, 127.50, 255.00),
        (0.00, 191.25, 255.00),
        (0.00, 255.00, 255.00),
        (0.00, 255.00, 191.25),
        (0.00, 255.00, 127.50),
        (0.00, 255.00, 63.75),
        (0.00, 255.00, 0.00),
        (63.75, 255.00, 0.00),
        (127.50, 255.00, 0.00),
        (191.25, 255.00, 0.00),
        (255.00, 255.00, 0.00),
        (255.00, 191.25, 0.00),
        (255.00, 127.50, 0.00),
        (255.00, 63.75, 0.00),
        (255.00, 0.00, 0.00),
        (255.00, 0.00, 63.75),
        (255.00, 0.00, 127.50),
        (255.00, 0.00, 191.25),
        (255.00, 0.00, 255.00),
        (191.25, 0.00, 255.00),
        (127.50, 0.00, 255.00),
        (63.75, 0.00, 255.00),
        (0.00, 0.00, 255.00),
    ]  # End boundary of the color bar, not the color of the last cell in the color bar.
    return color_bar


def nikon_D3300_monitor_equal_step_color_bar():
    """
    Generate a color bar measured using an LCD monitor as a color reference on August 24, 2023.

    Returns
    -------
    list of tuple
        A list of RGB tuples representing the color bar, where each tuple contains three floats
        corresponding to the red, green, and blue color channels (0-255).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    #                  Red    Green     Blue
    color_bar = [
        (0.00, 14.00, 254.00),
        (0.00, 107.00, 255.00),
        (0.00, 168.00, 255.00),
        (0.00, 198.00, 255.00),
        (0.00, 229.00, 255.00),
        (0.00, 255.00, 249.00),
        (0.00, 255.00, 219.00),
        (0.00, 255.00, 188.00),
        (0.00, 255.00, 158.00),
        (0.00, 255.00, 127.00),
        (0.00, 255.00, 66.00),
        (117.00, 255.00, 0.00),
        (147.00, 255.00, 0.00),
        (178.00, 255.00, 0.00),
        (209.00, 255.00, 0.00),
        (239.00, 255.00, 0.00),
        (255.00, 239.00, 0.00),
        (255.00, 209.00, 0.00),
        (255.00, 178.00, 0.00),
        (255.00, 147.00, 0.00),
        (255.00, 86.00, 0.00),
        (255.00, 0.00, 96.00),
        (255.00, 0.00, 158.00),
        (255.00, 0.00, 188.00),
        (255.00, 0.00, 219.00),
        (255.00, 0.00, 249.00),
        (229.00, 0.00, 255.00),
        (198.00, 0.00, 255.00),
        (168.00, 0.00, 255.00),
        (137.00, 0.00, 255.00),
        (107.00, 0.00, 255.00),
        (76.00, 0.00, 255.00),
        (0.00, 14.00, 254.00),
    ]  # End boundary of the color bar, not the color of the last cell in the color bar.
    return color_bar


# Color lookup
def color_given_value(val, val_min, val_max, color_below_min, color_bar, color_above_max, discrete_or_continuous):
    """
    Lookup the color corresponding to a given value based on a specified color bar.

    Parameters
    ----------
    val : float
        The value for which to find the corresponding color.
    val_min : float
        The minimum value of the range.
    val_max : float
        The maximum value of the range.
    color_below_min : tuple
        The color to return if the value is below the minimum (RGB tuple).
    color_bar : list of tuple
        The color bar to use for mapping values to colors (list of RGB tuples).
    color_above_max : tuple
        The color to return if the value is above the maximum (RGB tuple).
    discrete_or_continuous : str
        Specifies whether to return a 'discrete' or 'continuous' color.

    Returns
    -------
    tuple
        The RGB color corresponding to the input value.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Out-of-bounds cases.
    if val < val_min:
        return color_below_min
    if val > val_max:
        return color_above_max
    n_colors = len(color_bar)
    if n_colors == 1:
        return color_bar[0]
    # In bounds cases.
    # From here on we know (val_min <= val <= val_max), and color_bar contains multiple colors.
    n_steps = n_colors - 1  # Last block on color bar is not a step.
    val_step = (val_max - val_min) / n_steps
    step = (val - val_min) / val_step  # Since (val_min <= val <= val_max), we know (0 <= step <= 1).
    idx = int(step)
    if discrete_or_continuous == "discrete":
        return color_bar[idx]
    elif discrete_or_continuous == "continuous":
        if idx < 0:
            return color_below_min
        elif idx > (n_colors - 1):
            return color_above_max
        elif idx == (n_colors - 1):
            return color_bar[n_colors - 1]
        else:
            # Interpolate color with the color after.
            color_0 = color_bar[idx]
            color_1 = color_bar[idx + 1]
            frac = step - idx
            d_red = color_1[0] - color_0[0]
            d_green = color_1[1] - color_0[1]
            d_blue = color_1[2] - color_0[2]
            # Return.
            return [(color_0[0] + (frac * d_red)), (color_0[1] + (frac * d_green)), (color_0[2] + (frac * d_blue))]
    else:
        print(
            "ERROR: In color_given_value(), encountered unexpected discrete_or_continuous value:",
            discrete_or_continuous,
        )
        assert False


def angle_between_color_vectors(rgb_1, rgb_2):
    """
    Calculate the angle between two RGB color vectors.

    Parameters
    ----------
    rgb_1 : tuple
        The first RGB color vector (RGB tuple).
    rgb_2 : tuple
        The second RGB color vector (RGB tuple).

    Returns
    -------
    float
        The angle in radians between the two RGB color vectors.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    rgb_1_uvec = Uxyz.Uxyz(rgb_1)
    rgb_2_uvec = Uxyz.Uxyz(rgb_2)
    cross_1_2_vec = rgb_1_uvec.cross(rgb_2_uvec)
    magnitude_cross_1_2 = cross_1_2_vec.magnitude()
    angle_1_2 = math.asin(magnitude_cross_1_2)
    return angle_1_2


def color_bar_segment_spanned_angle(idx, color_bar):
    """
    Calculate the angle spanned by a segment of the color bar between two adjacent colors.

    Parameters
    ----------
    idx : int
        The index of the color segment in the color bar.
    color_bar : list of tuple
        The color bar to use for calculating the angle (list of RGB tuples).

    Returns
    -------
    float
        The angle in radians spanned by the color segment.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    if idx < 0:
        print("ERROR: In angle_between_color_vectors(), idx = ", str(idx), " is less than zero.")
        assert False  # ?? SCAFFOLDING RCB -- REPLACE WITH RAISING AN EXCEPTION?  (THROUGHOUT)
    if idx >= (len(color_bar) - 1):  # Below we fetch color_bar(idx+1), so compare against (len(color-bar)-1).
        print(
            "ERROR: In angle_between_color_vectors(), idx = ",
            str(idx),
            " is greater than or equal to color bar length:",
            len(color_bar),
        )
        assert False  # ?? SCAFFOLDING RCB -- REPLACE WITH RAISING AN EXCEPTION?  (THROUGHOUT)
    rgb_1 = color_bar[idx]
    rgb_2 = color_bar[idx + 1]
    angle_1_2 = angle_between_color_vectors(rgb_1, rgb_2)
    return angle_1_2


def construct_color_bar_spanned_angle_list(color_bar):
    """
    Construct a list of angles spanned by each segment of the color bar.

    Parameters
    ----------
    color_bar : list of tuple
        The color bar to use for calculating the angles (list of RGB tuples).

    Returns
    -------
    list of float
        A list of angles in radians spanned by each segment of the color bar.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    angle_list = []
    for idx in range(len(color_bar) - 1):  # One less than n_colors, because we look at pairs of (idx, idx+1).
        angle_1_2 = color_bar_segment_spanned_angle(idx, color_bar)
        angle_list.append(angle_1_2)
    # Return.
    return angle_list


def survey_color_bar(color_bar):
    """
    Survey the color bar to calculate the total angle spanned and the indices of the first and last non-zero angles.

    Parameters
    ----------
    color_bar : list of tuple
        The color bar to survey (list of RGB tuples).

    Returns
    -------
    tuple
        A tuple containing the total angle sum (float), the index of the first non-zero angle (int),
        and the index of the last non-zero angle (int).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    angle_sum = 0.0
    first_non_zero_angle_idx = -1
    last_non_zero_angle_idx = -1
    for idx in range(len(color_bar) - 1):  # One less than n_colors, because we look at pairs of (idx, idx+1).
        angle_1_2 = color_bar_segment_spanned_angle(idx, color_bar)
        if (first_non_zero_angle_idx < 0) and (angle_1_2 > 0):
            first_non_zero_angle_idx = idx
        if angle_1_2 > 0:
            last_non_zero_angle_idx = idx
        angle_sum += angle_1_2
    # Return.
    return angle_sum, first_non_zero_angle_idx, last_non_zero_angle_idx


def construct_rgb_cumulative_angle_pair_list(color_bar, first_color_idx, last_color_idx):
    """
    Construct a list of RGB colors with their corresponding cumulative angles.

    Parameters
    ----------
    color_bar : list of tuple
        The color bar to use for constructing the list (list of RGB tuples).
    first_color_idx : int
        The index of the first color to include in the list.
    last_color_idx : int
        The index of the last color to include in the list.

    Returns
    -------
    list of tuple
        A list of tuples where each tuple contains an RGB color and its corresponding cumulative angle.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    cumulative_angle = 0.0
    rgb_cumulative_angle_list = []
    # TODO RCB:  This logic is confusing and opaque.  Can it be simplified?
    if last_color_idx < (len(color_bar) - 1):  # Routine color_bar_segment_spanned_angle() will access color_bar[idx+1].
        last_idx = last_color_idx
    else:
        last_idx = len(color_bar) - 1
    for idx in range(first_color_idx, last_idx):
        rgb = color_bar[idx]
        angle_1_2 = color_bar_segment_spanned_angle(idx, color_bar)
        previous_cumulative_angle = cumulative_angle
        cumulative_angle += angle_1_2
        rgb_cumulative_angle_list.append((rgb, previous_cumulative_angle))
    # print("idx=", idx, " rgb=", rgb, " previous_cumulative_angle=", math.degrees(previous_cumulative_angle), " cumulative_angle=", math.degrees(cumulative_angle))
    if last_color_idx >= (
        len(color_bar) - 1
    ):  # Routine color_bar_segment_spanned_angle() will access color_bar[idx+1].
        rgb_cumulative_angle_list.append((color_bar[last_color_idx], cumulative_angle))
    return rgb_cumulative_angle_list


def lookup_color_and_angle_before(rgb_angle_list, desired_angle):
    """
    Lookup the RGB color and angle before a specified desired angle.

    Parameters
    ----------
    rgb_angle_list : list of tuple
        A list of tuples where each tuple contains an RGB color and its corresponding angle.
    desired_angle : float
        The desired angle for which to find the preceding color and angle.

    Returns
    -------
    tuple
        A tuple containing the RGB color and the angle before the desired angle.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    previous_rgb_angle = rgb_angle_list[0]
    previous_rgb = previous_rgb_angle[0]
    previous_angle = previous_rgb_angle[1]
    # First color case.
    if desired_angle == 0:
        return previous_rgb, 0.0
    for idx in range(len(rgb_angle_list)):
        this_rgb_angle = rgb_angle_list[idx]
        this_rgb = this_rgb_angle[0]
        this_angle = this_rgb_angle[1]
        if this_angle > desired_angle:
            break
        previous_rgb = this_rgb
        previous_angle = this_angle
    # Return.
    return previous_rgb, previous_angle


def lookup_color_and_angle_after(rgb_angle_list, desired_angle):
    """
    Lookup the RGB color and angle after a specified desired angle.

    Parameters
    ----------
    rgb_angle_list : list of tuple
        A list of tuples where each tuple contains an RGB color and its corresponding angle.
    desired_angle : float
        The desired angle for which to find the following color and angle.

    Returns
    -------
    tuple
        A tuple containing the RGB color and the angle after the desired angle.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    for idx in range(len(rgb_angle_list)):
        this_rgb_angle = rgb_angle_list[idx]
        this_rgb = this_rgb_angle[0]
        this_angle = this_rgb_angle[1]
        if this_angle > desired_angle:
            break
    # Return.
    return this_rgb, this_angle


def interpolate_color(desired_angle, rgb_before, angle_before, rgb_after, angle_after):
    """
    Interpolate the RGB color at a specified desired angle between two RGB colors.

    Parameters
    ----------
    desired_angle : float
        The angle at which to interpolate the color.
    rgb_before : tuple
        The RGB color before the desired angle (RGB tuple).
    angle_before : float
        The angle corresponding to the rgb_before color.
    rgb_after : tuple
        The RGB color after the desired angle (RGB tuple).
    angle_after : float
        The angle corresponding to the rgb_after color.

    Returns
    -------
    tuple
        The interpolated RGB color as a tuple of integers (rounded).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # TODO: This isn't exactly right.  A correct routine would construct a ray at the desired angle in the [R,G,B] space, and
    # project that ray onto the line between rgb_before and rgb_after. But we'll live with this approximation for now.
    # Boundary case.
    if angle_before == angle_after:
        interpolated_r = rgb_before[0]
        interpolated_g = rgb_before[1]
        interpolated_b = rgb_before[2]
    else:
        # Interpolation fraction.
        angle_frac = (desired_angle - angle_before) / (angle_after - angle_before)
        # Red.
        r_before = rgb_before[0]
        r_after = rgb_after[0]
        interpolated_r = r_before + (angle_frac * (r_after - r_before))
        # Green.
        g_before = rgb_before[1]
        g_after = rgb_after[1]
        interpolated_g = g_before + (angle_frac * (g_after - g_before))
        # Blue.
        b_before = rgb_before[2]
        b_after = rgb_after[2]
        interpolated_b = b_before + (angle_frac * (b_after - b_before))
    # Assemble.
    # We round values because [R,G,B] colors are defined by ints.  Another approximation.
    interpolated_rgb = (round(interpolated_r), round(interpolated_g), round(interpolated_b))
    return interpolated_rgb


def interpolate_color_given_angle(rgb_angle_list, desired_angle):
    """
    Interpolate the RGB color corresponding to a specified desired angle using a list of RGB colors and angles.

    Parameters
    ----------
    rgb_angle_list : list of tuple
        A list of tuples where each tuple contains an RGB color and its corresponding angle.
    desired_angle : float
        The desired angle for which to interpolate the color.

    Returns
    -------
    tuple
        The interpolated RGB color as a tuple of integers.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    rgb_before, angle_before = lookup_color_and_angle_before(rgb_angle_list, desired_angle)
    rgb_after, angle_after = lookup_color_and_angle_after(rgb_angle_list, desired_angle)
    interpolated_rgb = interpolate_color(desired_angle, rgb_before, angle_before, rgb_after, angle_after)
    # print("In interpolate_color_given_angle(), angle before/desired/after: {ab:.4f} / {da:.4f} / {aa:.4f}  rgb before/interpolated/after: {rb} / {ir} / {ra}".format(ab=angle_before, da=desired_angle, aa=angle_after, rb=rgb_before, ir=interpolated_rgb, ra=rgb_after))
    return interpolated_rgb


def normalize_color_bar_to_equal_angles(color_bar):
    """
    Normalize a color bar to have equal angles between colors.

    Parameters
    ----------
    color_bar : list of tuple
        The color bar to normalize, represented as a list of RGB tuples.

    Returns
    -------
    list of tuple
        A new color bar with equal angles between colors, represented as a list of RGB tuples.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Determine the number of color bar bins.
    n_colors = len(color_bar)

    # Survey color bar, computing total color spanning angle and identifying preamble/postamble
    # boundary sections with zero color spanning angle.
    # Note: Input color bar must not have interior segments with zero color spanning angle.
    angle_sum, first_non_zero_angle_idx, last_non_zero_angle_idx = survey_color_bar(color_bar)

    # Interpolation parameters.
    first_color_idx = first_non_zero_angle_idx
    last_color_idx = last_non_zero_angle_idx + 1
    n_steps = n_colors - 1
    angle_step = angle_sum / n_steps

    # Construct reference color/angle pair list.
    rgb_cumulative_angle_list = construct_rgb_cumulative_angle_pair_list(color_bar, first_color_idx, last_color_idx)

    # Generate equal-angle colors, along the same path in the [R,G,B] color space.
    rgb_0 = color_bar[first_color_idx]
    cumulative_angle = 0.0
    equal_angle_rgb_angle_list = []
    equal_angle_rgb_angle_list.append([rgb_0, cumulative_angle])
    while cumulative_angle < (angle_sum - 1e-6):  # tolerance to prevent additional step
        cumulative_angle += angle_step
        interpolated_rgb = interpolate_color_given_angle(rgb_cumulative_angle_list, cumulative_angle)
        equal_angle_rgb_angle_list.append([interpolated_rgb, cumulative_angle])

    # Construct a color bar, without angles.
    equal_angle_color_bar = [ca[0] for ca in equal_angle_rgb_angle_list]

    # Return.
    return equal_angle_color_bar


if __name__ == "__main__":
    print("Hello, World! (target_color_convert.py)")

    # # Fetch Matlab color bar.
    # color_bar = matlab_color_bar()
    # print("\nmatlab_color_bar() = ", color_bar)

    # Fetch closed corner tour color bar.
    color_bar = corner_tour_closed_color_bar()
    print("\n corner_tour_closed_color_bar() = ", color_bar)

    # Print angle list.
    angle_list = construct_color_bar_spanned_angle_list(color_bar)
    print("\nSpanned angle list=", angle_list)

    # Construct equal-angle normalized version.
    equal_angle_color_bar = normalize_color_bar_to_equal_angles(color_bar)
    print("\nequal_angle_color_bar = ", equal_angle_color_bar)

    # Print angle list.
    normalized_angle_list = construct_color_bar_spanned_angle_list(equal_angle_color_bar)
    print("\nNormalized spanned angle list=", normalized_angle_list)
