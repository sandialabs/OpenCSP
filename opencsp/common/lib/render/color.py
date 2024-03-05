"""

Color Management

# ?? SCAFFOLDING RCB -- MODIFY FILENAME TO START WITH UPPERCASE "C" AND UPDATE CALLERS



"""


class Color:
    """
    Class representing an color.

    This will begin as a simple [R,G,B] model, but is likely to grow into something more sophisticated.

    Potential directions for growth:
      - Support matplotlib color names:  'g', 'cyan', etc.
      - Support transparency
      - Support more sophisticated color models.

    Perhaps there is an existing Python class that we should use instead.

    """

    def __init__(
        self,
        red: float,  # In [0.0, 1.0]
        green: float,  # In [0.0, 1.0]
        blue: float,  # In [0.0, 1.0]
        name: str,  # E.g., 'black'
        short_name: str,
    ):  # E.g., 'k'
        self.red = red
        self.green = green
        self.blue = blue
        self.name = name
        self.short_name = short_name

    def rgb(self):
        """
        Returns color in [R,G,B] format, with range [0,1] for each.
        """
        return (self.red, self.green, self.blue)

    def rgb_255(self):
        """
        Returns color in [R,G,B] format, with range [0,255] for each.
        """
        return (self.red * 255, self.green * 255, self.blue * 255)


# GENERATORS


def black():
    return Color(0.0, 0.0, 0.0, 'black', 'k')


def dark_grey():
    return Color(0.25, 0.25, 0.25, 'dark grey', 'dg')


def grey():
    return Color(0.5, 0.5, 0.5, 'grey', 'gy')


def light_grey():
    return Color(0.75, 0.75, 0.75, 'light grey', 'lg')


def white():
    return Color(1.0, 1.0, 1.0, 'white', 'w')


def red():
    return Color(1.0, 0.0, 0.0, 'red', 'r')


def green():
    return Color(0.0, 1.0, 0.0, 'green', 'g')


def blue():
    return Color(0.0, 0.0, 1.0, 'blue', 'b')


def cyan():
    return Color(0.0, 1.0, 1.0, 'cyan', 'c')


def magenta():
    return Color(1.0, 0.0, 1.0, 'magenta', 'm')


def yellow():
    return Color(1.0, 1.0, 0.0, 'yellow', 'y')


class _PlotColors:
    def __init__(self):
        """Matplotlib default colors, from https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html"""
        self.colors = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ]
        self.blue = self.colors[0]
        self.orange = self.colors[1]
        self.green = self.colors[2]
        self.red = self.colors[3]
        self.purple = self.colors[4]
        self.brown = self.colors[5]
        self.pink = self.colors[6]
        self.gray = self.colors[7]
        self.yellow = self.colors[8]
        self.cyan = self.colors[9]

    def __getitem__(self, key):
        return self.colors[key]

    def __iter__(self):
        return self.colors.__iter__()


plot_colors = _PlotColors()
""" Matplotlib default colors, from https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html

Color order: blue, orange, green, red, purple, brown, pink, gray, yellow, cyan """

# COLOR WHEEL
#
# Generates a sequence of colors.


def color_wheel():
    return ['r', 'g', 'b', 'c', 'magenta', 'y', 'grey']


def color(color_idx, wheel=None):
    if wheel == None:
        wheel = color_wheel()
    return wheel[color_idx % len(wheel)]
