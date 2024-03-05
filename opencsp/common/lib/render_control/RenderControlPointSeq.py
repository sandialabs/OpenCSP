"""


"""

class RenderControlPointSeq():
    """
    Render control for sequences of points.
    
    Controls style of point markers and lines connecting points.
    
    Choices from:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

        Line Styles
        -----------
        '-' 	solid line style
        '--' 	dashed line style
        '-.' 	dash-dot line style
        ':' 	dotted line style        
        
        
        Colors
        ------
        'b' 	blue
        'g' 	green
        'r' 	red
        'c' 	cyan
        'm' 	magenta
        'y' 	yellow
        'k' 	black
        'w' 	white
        
        For more colors, see:
          https://matplotlib.org/stable/api/colors_api.html#module-matplotlib.colors
             
        Markers
        -------
        '.' 	point marker
        ',' 	pixel marker
        'o' 	circle marker
        'v' 	triangle_down marker
        '^' 	triangle_up marker
        '<' 	triangle_left marker
        '>' 	triangle_right marker
        '1' 	tri_down marker
        '2' 	tri_up marker
        '3' 	tri_left marker
        '4' 	tri_right marker
        '8' 	octagon marker
        's' 	square marker
        'p' 	pentagon marker
        'P' 	plus (filled) marker
        '*' 	star marker
        'h' 	hexagon1 marker
        'H' 	hexagon2 marker
        '+' 	plus marker
        'x' 	x marker
        'X' 	x (filled) marker
        'D' 	diamond marker
        'd' 	thin_diamond marker
        '|' 	vline marker
        '_' 	hline marker    
        'None'  no marker

    """

    def __init__(self,                 # See above for details:
                 linestyle='-',        # '-', '--', '-.', ':', '' or 'None'
                 linewidth=1,          # float
                 color='b',            #    bgrcmykw
                 marker='x',           #    .,ov^<>12348sp*hH+xXDd|_ or None
                 markersize=6,         # float
                 markeredgecolor=None, # Defaults to color above if not set.
                 markeredgewidth=None, # Defaults to linewidth if not set.
                 markerfacecolor=None, # Defaults to color above if not set.
                 vector_color='b',     # Used if points are in a vector field.
                 vector_linewidth=1,   # Used if points are in a vector field.
                 vector_scale=1.0,     # Facter to grow/srhink vector length, for points in a vector field.
                 ):

        super(RenderControlPointSeq, self).__init__()

        # Set defaults.
        if markeredgecolor==None:
            markeredgecolor=color
        if markeredgewidth==None:
            markeredgewidth=linewidth
        if markerfacecolor==None:
            markerfacecolor=color

        # Set fields.
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.color = color
        self.marker = marker
        self.markersize = markersize
        self.markeredgecolor = markeredgecolor
        self.markeredgewidth = markeredgewidth
        self.markerfacecolor = markerfacecolor
        self.vector_color = vector_color
        self.vector_linewidth = vector_linewidth
        self.vector_scale = vector_scale
    
    # MODIFICATION

    def set_color(self, color):
        self.color=color
        self.markeredgecolor=color
        self.markerfacecolor=color


# COMMON CASES

def default(marker='o', color='b', linewidth=1, markersize=8):
    """
    What to draw if no particular preference is expressed.
    """
    return RenderControlPointSeq(linestyle='-',
                                 linewidth=1,
                                 color=color,
                                 marker='.',
                                 markersize=markersize)


def outline(color='k', linewidth=1):
    """
    Outlines of physical objects.
    """
    return RenderControlPointSeq(linestyle='-',
                                 linewidth=linewidth,
                                 color=color,
                                 marker='None')


def data_curve(color='b', linewidth=1, marker='.', markersize=3) -> RenderControlPointSeq:
    """
    A data curve with data points identified.
    """
    return RenderControlPointSeq(linestyle='-',
                                 linewidth=linewidth,
                                 color=color,
                                 marker=marker,
                                 markersize=markersize)


def marker(marker='o', color='b', markersize=3) -> RenderControlPointSeq:
    """
    A data curve with data points identified.
    """
    return RenderControlPointSeq(linestyle='None',
                                 color=color,
                                 marker=marker,
                                 markersize=markersize)


def vector_field(marker='.', color='b', markersize=3, vector_linewidth=1, vector_scale=1.0) -> RenderControlPointSeq:
    """
    A field of vector needles.
    """
    return RenderControlPointSeq(linestyle='None',
                                 color=color,
                                 marker=marker,
                                 markersize=markersize,
                                 vector_color=color,
                                 vector_linewidth=vector_linewidth,
                                 vector_scale=vector_scale)

def thin(marker=',', linewidth = 0.3, color='y') -> RenderControlPointSeq:
    return RenderControlPointSeq(color=color, marker=marker, linewidth=linewidth)