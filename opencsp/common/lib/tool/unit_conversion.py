"""
Converting units, e.g, inches to meters.



"""

MM_PER_INCH = 25.4
M_PER_INCH = MM_PER_INCH / 1000.0


# Inch <--> Meters
def inch_to_meter(value_inch: float) -> float:
    """
    Convert a value in inches to meters.

    Parameters
    ----------
    value_inch : float
        The value in inches to be converted.

    Returns
    -------
    float
        The equivalent value in meters.

    Examples
    --------
    >>> inch_to_meter(1.0)
    0.0254
    >>> inch_to_meter(12.0)
    0.3048
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return value_inch * M_PER_INCH


def meter_to_inch(value_meter: float) -> float:
    """
    Convert a value in meters to inches.

    Parameters
    ----------
    value_meter : float
        The value in meters to be converted.

    Returns
    -------
    float
        The equivalent value in inches.

    Examples
    --------
    >>> meter_to_inch(0.0254)
    1.0
    >>> meter_to_inch(0.3048)
    12.0
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return value_meter / M_PER_INCH


# Dots per Inch  <-->  Dots per Meters
def dpi_to_dpm(value_dpi: float) -> float:
    """
    Convert a value in dots per inch (DPI) to dots per meter (DPM).

    Parameters
    ----------
    value_dpi : float
        The value in dots per inch to be converted.

    Returns
    -------
    float
        The equivalent value in dots per meter.

    Examples
    --------
    >>> dpi_to_dpm(300.0)
    11811.023622047244
    >>> dpi_to_dpm(72.0)
    2834.645669291338
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return value_dpi / M_PER_INCH


def dpm_to_dpi(value_dpi: float) -> float:
    """
    Convert a value in dots per meter (DPM) to dots per inch (DPI).

    Parameters
    ----------
    value_dpm : float
        The value in dots per meter to be converted.

    Returns
    -------
    float
        The equivalent value in dots per inch.

    Examples
    --------
    >>> dpm_to_dpi(11811.023622047244)
    300.0
    >>> dpm_to_dpi(2834.645669291338)
    72.0
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return value_dpi * M_PER_INCH
