"""

Color Management


"""

from typing import Iterator, Iterable, Union

import numpy as np
import matplotlib.colors


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

    def __init__(self, red: float, green: float, blue: float, name: str, short_name: str):
        """
        Parameters
        ----------
        red : float
            The red component in the RGB color space. Range 0-1.
        green : float
            The green component in the RGB color space. Range 0-1.
        blue : float
            The blue component in the RGB color space. Range 0-1.
        name : str
            A descriptive name for the color. For example "black".
        short_name : str
            A short hand name for the color. For example "k".
        """
        self.red = np.clip(red, 0, 1)
        self.green = np.clip(green, 0, 1)
        self.blue = np.clip(blue, 0, 1)
        self.name = name
        self.short_name = short_name

    @classmethod
    def from_i255(cls, red: int, green: int, blue: int, name: str, short_name: str):
        """
        Creates a Color instance from 8-bit RGB values.

        Parameters
        ----------
        red : int
            The red component in the RGB color space (0-255).
        green : int
            The green component in the RGB color space (0-255).
        blue : int
            The blue component in the RGB color space (0-255).
        name : str
            A descriptive name for the color.
        short_name : str
            A shorthand name for the color.

        Returns
        -------
        Color
            A Color instance with the specified RGB values.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return cls(red / 255, green / 255, blue / 255, name, short_name)

    @classmethod
    def from_hex(cls, hexval: str, name: str, short_name: str) -> "Color":
        """
        Creates a Color instance from a hexadecimal color string.

        Parameters
        ----------
        hexval : str
            The hexadecimal color string (e.g., "#RRGGBB").
        name : str
            A descriptive name for the color.
        short_name : str
            A shorthand name for the color.

        Returns
        -------
        Color
            A Color instance with the RGB values extracted from the hexadecimal string.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if hexval.startswith("0x"):
            hexval = "#" + hexval[2:]
        elif hexval.startswith("x"):
            hexval = "#" + hexval[1:]
        elif hexval.startswith("#"):
            pass
        else:
            hexval = "#" + hexval

        rgb = matplotlib.colors.to_rgb(hexval)
        return Color(rgb[0], rgb[1], rgb[2], name, short_name)

    @classmethod
    def from_hsv(cls, hue: float, saturation: float, value: float, name: str, short_name: str):
        """
        Creates a Color instance from HSV values.

        Parameters
        ----------
        hue : float
            The hue component in the HSV color space (0-1).
        saturation : float
            The saturation component in the HSV color space (0-1).
        value : float
            The value (brightness) component in the HSV color space (0-1).
        name : str
            A descriptive name for the color.
        short_name : str
            A shorthand name for the color.

        Returns
        -------
        Color
            A Color instance with the RGB values converted from the HSV values.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        rgb = matplotlib.colors.hsv_to_rgb((hue, saturation, value))
        return cls(rgb[0], rgb[1], rgb[2], name, value)

    @classmethod
    def from_str(cls, sval='b') -> "Color":
        """
        Creates a Color instance from a string representation.

        Parameters
        ----------
        sval : str, optional
            The string representation of the color (e.g., 'b' for blue). Defaults to 'b'.

        Returns
        -------
        Color
            A Color instance corresponding to the specified string representation.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        longhand = sval
        if sval in _plot_color_shorthands:
            longhand = _plot_color_shorthands[sval]

        rgb = matplotlib.colors.to_rgb(sval)

        return cls(rgb[0], rgb[1], rgb[2], longhand, sval)

    @classmethod
    def convert(cls, val: Union["Color", str, tuple, None]) -> "Color":
        """
        Converts various representations to a Color instance.

        Parameters
        ----------
        val : Color | str | tuple | None
            The value to convert, which can be a Color instance, a string, or a tuple of RGB values.

        Returns
        -------
        Color
            A Color instance corresponding to the input value.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        if val is None:
            return None
        elif isinstance(val, Color):
            return val
        elif isinstance(val, str):
            return cls.from_str(val)
        else:
            rgb = val
            return cls(rgb[0], rgb[1], rgb[2], "tuple", "tuple")

    def rgb(self) -> tuple[float, float, float]:
        """
        Returns the RGB values of the color.

        Returns
        -------
        tuple[float, float, float]
            A tuple containing the RGB values of the color, each in the range [0, 1].
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return (self.red, self.green, self.blue)

    def rgba(self, alpha=1.0) -> tuple[float, float, float, float]:
        """
        Returns the RGBA values of the color.

        Parameters
        ----------
        alpha : float, optional
            The alpha (transparency) value of the color. Defaults to 1.0 (fully opaque).

        Returns
        -------
        tuple[float, float, float, float]
            A tuple containing the RGBA values of the color, each in the range [0, 1].
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return (self.red, self.green, self.blue, alpha)

    def rgb_255(self) -> tuple[int, int, int]:
        """
        Returns the RGB values of the color in the range [0, 255].

        Returns
        -------
        tuple[int, int, int]
            A tuple containing the RGB values of the color, each in the range [0, 255].
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return (int(self.red * 255), int(self.green * 255), int(self.blue * 255))

    def to_hex(self) -> str:
        """
        Converts the color to a hexadecimal string representation.

        Returns
        -------
        str
            The hexadecimal string representation of the color (e.g., "#RRGGBB").
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        return matplotlib.colors.to_hex(self.rgb()).upper()

    def to_hsv(self) -> tuple[float, float, float]:
        """
        Converts the RGB color to HSV representation.

        Returns
        -------
        tuple[float, float, float]
            A tuple containing the HSV values of the color, each in the range [0, 1].
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        ret = matplotlib.colors.rgb_to_hsv(self.rgb())
        return float(ret[0]), float(ret[1]), float(ret[2])

    def build_colormap(self, *next_colors: "Color") -> matplotlib.colors.Colormap:
        """
        Builds a colormap that transitions between this color and the specified next colors,
        given a value between 0 and 1.

        Parameters
        ----------
        next_colors : Color
            The color(s) to fade this instance with. Typically this will only be one color,
            such as for a red-to-blue fade. However, it could also be multiple colors, such
            as for a blue-to-purple-to-yellow fade (the matplotlib 'viridis' default).

        Returns
        -------
        matplotlib.colors.Colormap
            A colormap object that can be used for rendering.
        """
        # "ChatGPT 4o" assisted with generating this docstring.

        # TODO add a "N" parameter to increase the number of colors in the
        # colormap. See the "N" parameter of
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html#matplotlib-colors-listedcolormap.
        colors_sequence = [self] + list(next_colors)
        colors_rgb = [np.array(list(clr.rgb())) for clr in colors_sequence]
        ncolors = len(colors_sequence)

        mixed_colors = [np.zeros_like(colors_rgb[0]) for i in range(256)]
        fade_ranges = np.array_split(np.arange(256, dtype=np.uint8), ncolors - 1)
        for fade_idx, fade_range in enumerate(fade_ranges):
            fade_range = fade_range.tolist()
            fade_length = len(fade_range)
            fade_start = fade_range[0]

            # mix the colors for this fade
            color1 = colors_rgb[fade_idx]
            color2 = colors_rgb[fade_idx + 1]
            step_size = 1.0 / (fade_length - 1)
            mix1 = [color1 * i for i in np.arange(1.0, 0.0 - step_size * 0.5, -step_size)]
            mix2 = [color2 * i for i in np.arange(0.0, 1.0 + step_size * 0.5, step_size)]
            for i in fade_range:
                mixed_colors[i] = np.array(mix1[i - fade_start]) + np.array(mix2[i - fade_start])

        new_colors = np.clip(mixed_colors, 0.0, 1.0)
        name = "".join([clr.name.replace(" ", "") for clr in colors_sequence])
        newcmap = matplotlib.colors.ListedColormap(new_colors, name=name)

        # clamp to the extremis (for inputs < 0 or >= 1)
        newcmap.set_under(new_colors[0])
        newcmap.set_over(new_colors[-1])

        return newcmap


# GENERATORS


def black():
    """
    Returns a Color instance representing black.

    Returns
    -------
    Color
        A Color instance with RGB values (0.0, 0.0, 0.0).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(0.0, 0.0, 0.0, 'black', 'k')


def dark_grey():
    """
    Returns a Color instance representing dark grey.

    Returns
    -------
    Color
        A Color instance with RGB values (0.25, 0.25, 0.25).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(0.25, 0.25, 0.25, 'dark grey', 'dg')


def grey():
    """
    Returns a Color instance representing grey.

    Returns
    -------
    Color
        A Color instance with RGB values (0.5, 0.5, 0.5).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(0.5, 0.5, 0.5, 'grey', 'gy')


def light_grey():
    """
    Returns a Color instance representing light grey.

    Returns
    -------
    Color
        A Color instance with RGB values (0.75, 0.75, 0.75).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(0.75, 0.75, 0.75, 'light grey', 'lg')


def white():
    """
    Returns a Color instance representing white.

    Returns
    -------
    Color
        A Color instance with RGB values (1.0, 1.0, 1.0).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(1.0, 1.0, 1.0, 'white', 'w')


def red():
    """
    Returns a Color instance representing red.

    Returns
    -------
    Color
        A Color instance with RGB values (1.0, 0.0, 0.0).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(1.0, 0.0, 0.0, 'red', 'r')


def green():
    """
    Returns a Color instance representing green.

    Returns
    -------
    Color
        A Color instance with RGB values (0.0, 1.0, 0.0).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(0.0, 1.0, 0.0, 'green', 'g')


def blue():
    """
    Returns a Color instance representing blue.

    Returns
    -------
    Color
        A Color instance with RGB values (0.0, 0.0, 1.0).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(0.0, 0.0, 1.0, 'blue', 'b')


def cyan():
    """
    Returns a Color instance representing cyan.

    Returns
    -------
    Color
        A Color instance with RGB values (0.0, 1.0, 1.0).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(0.0, 1.0, 1.0, 'cyan', 'c')


def magenta():
    """
    Returns a Color instance representing magenta.

    Returns
    -------
    Color
        A Color instance with RGB values (1.0, 0.0, 1.0).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(1.0, 0.0, 1.0, 'magenta', 'm')


def yellow():
    """
    Returns a Color instance representing yellow.

    Returns
    -------
    Color
        A Color instance with RGB values (1.0, 1.0, 0.0).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return Color(1.0, 1.0, 0.0, 'yellow', 'y')


def color_map(*colors_sequence: "Color"):
    """
    Creates a colormap that transitions between a sequence of colors.

    This function takes a sequence of Color instances and builds a colormap that smoothly
    transitions between the specified colors. The first color in the sequence is used as the
    starting point, and the subsequent colors are used to define the transitions.

    Parameters
    ----------
    colors_sequence : Color
        A variable number of Color instances that define the colors to be included in the colormap.
        The first color is the starting color, and the subsequent colors define the transitions.

    Returns
    -------
    matplotlib.colors.Colormap
        A colormap object that can be used for rendering.

    Raises
    ------
    ValueError
        If no colors are provided in the sequence.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    colors_sequence = list(colors_sequence)
    return colors_sequence[0].build_colormap(*colors_sequence[1:])


class _PlotColors(Iterable[Color]):
    """Matplotlib default 'tab10' colors, from https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html"""

    def __init__(self):
        self._color_hexes = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        self.blue = Color.from_hex(self._color_hexes[0], "blue", "b")
        self.orange = Color.from_hex(self._color_hexes[1], "orange", "o")
        self.green = Color.from_hex(self._color_hexes[2], "green", "g")
        self.red = Color.from_hex(self._color_hexes[3], "red", "r")
        self.purple = Color.from_hex(self._color_hexes[4], "purple", "p")
        self.brown = Color.from_hex(self._color_hexes[5], "brown", "br")
        self.pink = Color.from_hex(self._color_hexes[6], "pink", "pi")
        self.gray = Color.from_hex(self._color_hexes[7], "gray", "gr")
        self.yellow = Color.from_hex(self._color_hexes[8], "yellow", "y")
        self.cyan = Color.from_hex(self._color_hexes[9], "cyan", "c")

        self._color_names = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "yellow", "cyan"]
        self._colors = [self[cn] for cn in self._color_names]

    def __getitem__(self, key: int | str) -> Color:
        if isinstance(key, int):
            key %= len(self._colors)
            return self._colors[key]
        elif isinstance(key, str):
            return getattr(self, key)
        else:
            raise KeyError

    def __iter__(self) -> Iterator[Color]:
        return iter(self._colors)


_plot_colors = _PlotColors()
_plot_color_names = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "yellow", "cyan"]
plot_colors = {name: _plot_colors[name] for name in _plot_color_names}
plot_colorsi = {i: _plot_colors[i] for i in range(len(_plot_color_names))}
""" Matplotlib default 'tab10' colors, from https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html

We enumerate these colors here as the simplest possible way of accessing these
colors, so that we can use or remix them as necessary.

Color order: blue, orange, green, red, purple, brown, pink, gray, yellow, cyan """


_plot_color_shorthands = {
    "b": "blue",
    "g": "green",
    "r": "red",
    "c": "cyan",
    "m": "magenta",
    "y": "yellow",
    "k": "black",
    "w": "white",
}
""" From https://matplotlib.org/stable/users/explain/colors/colors.html """
