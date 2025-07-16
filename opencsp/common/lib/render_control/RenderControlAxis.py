""" """


class RenderControlAxis:
    """
    Render control for plot axes.
    """

    def __init__(self, x_label="x", y_label="y", z_label="z", p_label="p", q_label="q", w_label="w", grid=True):
        super(RenderControlAxis, self).__init__()

        # Axis control.
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.p_label = p_label
        self.q_label = q_label
        self.w_label = w_label
        self.grid = grid


def meters(grid=True):
    """
    Labels indicating units of meters.
    """
    return RenderControlAxis(
        x_label="x (m)", y_label="y (m)", z_label="z (m)", p_label="p (m)", q_label="q (m)", w_label="w (m)", grid=grid
    )


def latlon(decimal_t_degminsecs_f=True, grid=True):
    """
    Labels indicating units of latitude and longitude.
    """
    unit = "deg" if decimal_t_degminsecs_f else "deg,min,sec"
    return RenderControlAxis(
        x_label=f"longitude ({unit})",
        y_label=f"latitude ({unit})",
        z_label=f"z ({unit})",
        p_label=f"p ({unit})",
        q_label=f"q ({unit})",
        w_label=f"w ({unit})",
        grid=grid,
    )


def image(grid=True):
    """
    Labels indicating image.
    """
    return RenderControlAxis(
        x_label="x N/A",
        y_label="y N/A",
        z_label="z N/A",
        p_label="x (pix)",
        q_label="y (pix)",
        w_label="w N/A",
        grid=grid,
    )
