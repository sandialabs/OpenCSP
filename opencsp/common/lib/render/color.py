"""

Color Management

# ?? SCAFFOLDING RCB -- MODIFY FILENAME TO START WITH UPPERCASE "C" AND UPDATE CALLERS



"""

from typing import Iterator, Iterable

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
        return cls(red / 255, green / 255, blue / 255, name, short_name)

    @classmethod
    def from_hex(cls, hexval: str, name: str, short_name: str) -> "Color":
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
        rgb = matplotlib.colors.hsv_to_rgb((hue, saturation, value))
        return cls(rgb[0], rgb[1], rgb[2], name, value)

    def rgb(self) -> tuple[float, float, float]:
        """
        Returns color in [R,G,B] format, with range [0,1] for each.
        """
        return (self.red, self.green, self.blue)

    def rgb_255(self) -> tuple[int, int, int]:
        """
        Returns color in [R,G,B] format, with range [0,255] for each.
        """
        return (int(self.red * 255), int(self.green * 255), int(self.blue * 255))

    def to_hex(self) -> str:
        return matplotlib.colors.to_hex(self.rgb()).upper()

    def to_hsv(self) -> tuple[float, float, float]:
        ret = matplotlib.colors.rgb_to_hsv(self.rgb())
        return float(ret[0]), float(ret[1]), float(ret[2])

    def build_colormap(self, *next_colors: "Color") -> matplotlib.colors.Colormap:
        """
        Build a colormap that will return a color between this instance and the
        next color(s), given a value between 0 and 1.

        TODO add a "N" parameter to increase the number of colors in the
        colormap. See the "N" parameter of
        https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html#matplotlib-colors-listedcolormap.

        Parameters
        ----------
        next_colors: list[Color]
            The color(s) to fade this instance with. Typically this will only be
            one color, such as for a red-to-blue fade. However, it could also be
            multiple colors, such as for a blue-to-purple-to-yellow fade (the
            matplotlib 'viridis' default).
        """
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


def color_map(*colors_sequence: "Color"):
    colors_sequence = list(colors_sequence)
    return colors_sequence[0].build_colormap(*colors_sequence[1:])


class _PlotColors(Iterable[Color]):
    """Matplotlib default 'tab10' colors, from https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html"""

    def __init__(self):
        self._color_hexes = [
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
